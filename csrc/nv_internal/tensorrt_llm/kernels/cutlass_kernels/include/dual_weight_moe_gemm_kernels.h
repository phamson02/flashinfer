/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped_dual_weight.h"
#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel_dual_weight.h"
#include "moe_gemm_kernels.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80_dual_weight.h"

namespace tensorrt_llm::kernels::cutlass_kernels {

// Dual-weight grouped GEMM input structure
template <typename T, typename WeightType, typename ScaleBiasType, typename OutputType>
struct DualWeightGroupedGemmInput
    : public GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> {
  WeightType const* B_lower{nullptr};

  DualWeightGroupedGemmInput() = default;
};

// Dual-weight MOE GEMM kernel launcher for FP16 activations with FP8 weights on SM80/89
template <typename T, typename WeightType, typename GemmOutputType, typename arch,
          cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape, int Stages>
struct genericDualWeightMoeGemmKernelLauncher {
  static void call(DualWeightGroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
                   int sm_count_) {
    static_assert(std::is_same_v<T, half>, "Dual-weight MOE only supports FP16 activations");
    static_assert(std::is_same_v<WeightType, __nv_fp8_e4m3>,
                  "Dual-weight MOE only supports FP8 E4M3 weights");
    static_assert(arch::kMinComputeCapability == 80, "Dual-weight MOE only supports SM80");
    static_assert(QuantOp == cutlass::WeightOnlyQuantOp::UNDEFINED,
                  "Dual-weight MOE does not support quantization ops");

    using ElementType = typename TllmToCutlassTypeAdapter<T>::type;
    using CutlassGemmOutputType = typename TllmToCutlassTypeAdapter<GemmOutputType>::type;
    using CutlassWeightType = typename TllmToCutlassTypeAdapter<WeightType>::type;

    if (!inputs.use_fused_moe) {
      // We need separate config for each architecture since we will target different tensorcore
      // instructions. For float, we do not target TCs.
      using MixedGemmArchTraits =
          cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
      using ElementAccumulator = typename MixedGemmArchTraits::AccType;

      using EpilogueOp = typename tensorrt_llm::cutlass_extensions::Epilogue<
          CutlassGemmOutputType, MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator,
          EpilogueTag>::Op;
      
      typename EpilogueOp::Params epilogue_op(
          ElementAccumulator(1.f),
          inputs.biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));
      using TaggedOperator =
          typename cutlass::arch::TagOperator<typename MixedGemmArchTraits::Operator,
                                              QuantOp>::TaggedOperator;
      
      // Finally, set up the kernel.
      using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGroupedDualWeight<
          ElementType, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
          MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType,
          typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
          MixedGemmArchTraits::ElementsPerAccessB, CutlassGemmOutputType, cutlass::layout::RowMajor,
          ElementAccumulator, typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape,
          WarpShape, typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
          cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, TaggedOperator>::GemmKernel;

      using GemmKernel =
          cutlass::gemm::kernel::MoeFCGemmDualWeight<typename GemmKernel_::Mma,
                                                     typename GemmKernel_::Epilogue,
                                                     typename GemmKernel_::ThreadblockSwizzle, 
                                                     arch, // Ensure top level arch is used for dispatch
                                                     GemmKernel_::kGroupScheduleMode>;

      using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

      if (inputs.occupancy != nullptr) {
        *inputs.occupancy =
            tensorrt_llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
        return;
      }
      int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
      TLLM_CHECK_WITH_INFO(
          occupancy > 0,
          "GPU lacks the shared memory resources to run dual-weight GroupedGEMM kernel");
      int const threadblock_count = sm_count_ * occupancy;

      int const gemm_group_size = inputs.k;
      typename GemmGrouped::Arguments args(
          inputs.num_experts, threadblock_count, gemm_group_size, epilogue_op,
          reinterpret_cast<ElementType const*>(inputs.A),
          reinterpret_cast<CutlassWeightType const*>(inputs.B),        // Upper weights
          reinterpret_cast<CutlassWeightType const*>(inputs.B_lower),  // Lower weights
          reinterpret_cast<CutlassGemmOutputType const*>(inputs.scales),
          reinterpret_cast<CutlassGemmOutputType const*>(inputs.zeros),
          reinterpret_cast<CutlassGemmOutputType const*>(inputs.biases), inputs.bias_is_broadcast,
          reinterpret_cast<CutlassGemmOutputType*>(inputs.C), inputs.total_tokens_including_expert,
          inputs.n, inputs.k);
      
      GemmGrouped gemm;

      auto can_implement = gemm.can_implement(args);
      TLLM_CHECK_WITH_INFO(can_implement == cutlass::Status::kSuccess,
                           "Dual-weight MoE FC kernel will fail for params. Error: " +
                               std::string(cutlassGetStatusString(can_implement)));

      auto init_status = gemm.initialize(args);
      TLLM_CHECK_WITH_INFO(init_status == cutlass::Status::kSuccess,
                           "Failed to initialize dual-weight cutlass grouped gemm. Error: " +
                               std::string(cutlassGetStatusString(init_status)));
      
      auto run_status = gemm.run(inputs.stream);
      TLLM_CHECK_WITH_INFO(run_status == cutlass::Status::kSuccess,
                           "Failed to run dual-weight cutlass grouped gemm. Error: " +
                               std::string(cutlassGetStatusString(run_status)));
    } else if constexpr (sizeof(ElementType) == 2 && sizeof(CutlassWeightType) == 1 &&
                         std::is_same_v<CutlassWeightType, cutlass::float_e4m3_t> &&
                         (std::is_same_v<EpilogueTag, cutlass_extensions::EpilogueOpDefaultSilu> ||
                          std::is_same_v<
                              EpilogueTag,
                              cutlass_extensions::EpilogueOpDefaultFtGelu>))  // use fused moe gemm
                                                                              // kernel for FP16
                                                                              // activations with
                                                                              // FP8 E4M3 weights
    {
      sm80_dual_weight_fused_moe_gemm_kernelLauncher<ElementType, CutlassWeightType,
                                                     ThreadblockShape::kM, ThreadblockShape::kN,
                                                      ThreadblockShape::kK, Stages, EpilogueTag>(
          reinterpret_cast<ElementType const*>(inputs.A),
          reinterpret_cast<CutlassWeightType const*>(inputs.B),        // Upper weights
          reinterpret_cast<CutlassWeightType const*>(inputs.B_lower),  // Lower weights
          reinterpret_cast<ElementType const*>(inputs.biases), inputs.bias_is_broadcast,
          reinterpret_cast<ElementType*>(inputs.C), inputs.total_tokens_including_expert,
          inputs.num_rows, inputs.n, inputs.k, inputs.num_experts, sm_count_, inputs.stream,
          inputs.occupancy);
    }
  }
};

// Dispatch function for stages - dispatches to different stage counts
template <typename T, typename WeightType, typename GemmOutputType, typename arch,
          typename EpilogueTag, typename ThreadblockShape, typename WarpShape>
void dispatchDualWeightGemmStages(
    DualWeightGroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_,
    int stages) {
  switch (stages) {
    case 2:
      genericDualWeightMoeGemmKernelLauncher<T, WeightType, GemmOutputType, arch,
                                             cutlass::WeightOnlyQuantOp::UNDEFINED, EpilogueTag,
                                             ThreadblockShape, WarpShape, 2>::call(inputs,
                                                                                   sm_count_);
      break;
    case 3:
      genericDualWeightMoeGemmKernelLauncher<T, WeightType, GemmOutputType, arch,
                                             cutlass::WeightOnlyQuantOp::UNDEFINED, EpilogueTag,
                                             ThreadblockShape, WarpShape, 3>::call(inputs,
                                                                                   sm_count_);
      break;
    case 4:
      genericDualWeightMoeGemmKernelLauncher<T, WeightType, GemmOutputType, arch,
                                             cutlass::WeightOnlyQuantOp::UNDEFINED, EpilogueTag,
                                             ThreadblockShape, WarpShape, 4>::call(inputs,
                                                                                   sm_count_);
      break;
    default:
      TLLM_THROW("dispatchDualWeightGemmStages does not support stages %d",
                 inputs.gemm_config.stages);
      break;
  }
}

// Dispatch function for tile configs - matches FP16/FP16 configs from single weight
template <typename T, typename WeightType, typename GemmOutputType, typename arch,
          typename EpilogueTag>
void dispatchDualWeightMoeGemmToCutlass(
    DualWeightGroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    int sm_count_) {
  auto const& config = inputs.gemm_config;
  int const stages = config.stages == 0 ? 3 : config.stages;  // Default to 3 stages

  switch (config.tile_config_sm80) {
    case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      dispatchDualWeightGemmStages<T, WeightType, GemmOutputType, arch, EpilogueTag,
                                   cutlass::gemm::GemmShape<16, 128, 64>,
                                   cutlass::gemm::GemmShape<16, 32, 64>>(inputs, sm_count_, stages);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
      dispatchDualWeightGemmStages<T, WeightType, GemmOutputType, arch, EpilogueTag,
                                   cutlass::gemm::GemmShape<16, 256, 64>,
                                   cutlass::gemm::GemmShape<16, 64, 64>>(inputs, sm_count_, stages);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatchDualWeightGemmStages<T, WeightType, GemmOutputType, arch, EpilogueTag,
                                   cutlass::gemm::GemmShape<32, 128, 64>,
                                   cutlass::gemm::GemmShape<32, 32, 64>>(inputs, sm_count_, stages);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      dispatchDualWeightGemmStages<T, WeightType, GemmOutputType, arch, EpilogueTag,
                                   cutlass::gemm::GemmShape<64, 128, 64>,
                                   cutlass::gemm::GemmShape<32, 64, 64>>(inputs, sm_count_, stages);
      break;
    // case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
    //   dispatchDualWeightGemmStages<T, WeightType, GemmOutputType, arch, EpilogueTag,
    //                                cutlass::gemm::GemmShape<64, 128, 64>,
    //                                cutlass::gemm::GemmShape<64, 32, 64>>(inputs, sm_count_, stages);
    //   break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      dispatchDualWeightGemmStages<T, WeightType, GemmOutputType, arch, EpilogueTag,
                                   cutlass::gemm::GemmShape<128, 128, 64>,
                                   cutlass::gemm::GemmShape<64, 32, 64>>(inputs, sm_count_, stages);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      TLLM_THROW("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      TLLM_THROW("GEMM config should have already been set by heuristic.");
      break;
    default:
      TLLM_THROW("Config is invalid for same type tensorop GEMM.");
      break;
  }
}

template <typename T,                         /*The type used for activations*/
          typename WeightType,                /* The type for the MoE weights */
          typename OutputType,                /* The output type for the GEMM */
          typename ScaleBiasType = OutputType /* The type for the scales/bias */
          >
class DualWeightMoeGemmRunner {
 public:
  DualWeightMoeGemmRunner();

  // Type constraints for dual-weight MOE
  static_assert(std::is_same_v<T, half>, "Dual-weight MOE only supports FP16 activations");
  static_assert(std::is_same_v<WeightType, __nv_fp8_e4m3>,
                "Dual-weight MOE only supports FP8 E4M3 weights");

  void moeGemmBiasAct(DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs);

  void moeGemm(DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs);

  std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs() const;
  static std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs(int sm);
  static std::vector<cutlass_extensions::CutlassGemmConfig> getAmpereConfigs(int sm);

  [[nodiscard]] bool isFusedGatedActivation(cutlass_extensions::CutlassGemmConfig gemm_config,
                                            ActivationType activation_type, int gemm_n,
                                            int gemm_k) const;
  [[nodiscard]] bool supportsFusedGatedActivation(ActivationType activation_type, int gemm_n,
                                                  int gemm_k) const;

  size_t getMaxWorkspaceSize(int num_experts) const;

  [[nodiscard]] int getSM() const;

 private:
  template <typename EpilogueTag>
  void dispatchToArch(DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs);

  template <typename EpilogueTag>
  void runGemm(DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs);

 private:
  int sm_{};
  int multi_processor_count_{};
  mutable int num_experts_ = 0;
  mutable size_t gemm_workspace_size_ = 0;

  size_t calcMaxWorkspaceSize(int num_experts) const;
};

// ==================== Implementation ====================

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
int DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getSM() const {
  return this->sm_;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::supportsFusedGatedActivation(
    ActivationType activation_type, int gemm_n, int gemm_k) const {
  constexpr bool ENABLE_FUSED_GATED_ACTIVATION = false;
  return (activation_type == ActivationType::Swiglu || activation_type == ActivationType::Geglu) &&
        !std::is_same_v<T, float> &&
         (this->getSM() >= 80) && (gemm_k % 64 == 0) && (gemm_n % 64 == 0) &&
         ENABLE_FUSED_GATED_ACTIVATION;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::isFusedGatedActivation(
    cutlass_extensions::CutlassGemmConfig gemm_config, ActivationType activation_type, int gemm_n,
    int gemm_k) const {
  return supportsFusedGatedActivation(activation_type, gemm_n, gemm_k);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::DualWeightMoeGemmRunner() {
  int device{-1};
  tensorrt_llm::common::check_cuda_error(cudaGetDevice(&device));
  sm_ = tensorrt_llm::common::getSMVersion();
  tensorrt_llm::common::check_cuda_error(
      cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
template <typename EpilogueTag>
void DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::dispatchToArch(
    DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs) {
  static_assert(
      std::is_same_v<ScaleBiasType, OutputType>,
      "Separate Scale/Bias type is not supported. This is assumed to be the gemm output type");

  // Only SM80 is currently supported
  if (sm_ >= 80 && sm_ < 90) {
    dispatchDualWeightMoeGemmToCutlass<T, WeightType, OutputType, cutlass::arch::Sm80, EpilogueTag>(
        inputs, multi_processor_count_);
  } else {
    TLLM_CHECK_WITH_INFO(
        false, "Dual-weight MOE GEMM only supports SM80-89. Current SM: " + std::to_string(sm_));
  }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getConfigs() const {
  return getConfigs(sm_);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getConfigs(int sm) {
  if (sm >= 90) {
    TLLM_THROW("Dual-weight MOE GEMM is not supported on SM90+");
    return {};
  }
  return getAmpereConfigs(sm);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getAmpereConfigs(int sm) {
  using tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
  static constexpr auto weight_only_flag = CutlassGemmConfig::NONE;
  static constexpr auto simt_only_flag =
      std::is_same<T, float>::value ? CutlassGemmConfig::SIMT_ONLY : CutlassGemmConfig::NONE;
  static constexpr auto fp8_only_flag = CutlassGemmConfig::NONE;
  int const max_split_k = 1;
  int const grouped_gemm_flag = CutlassGemmConfig::GROUPED_GEMM;
  int const enable_hopper = CutlassGemmConfig::NONE;

  auto config_type_param = static_cast<CutlassGemmConfig::CandidateConfigTypeParam>(
      weight_only_flag | simt_only_flag | grouped_gemm_flag | enable_hopper | fp8_only_flag);

  if (!kernels::cutlass_kernels::isValidAmpereMOESpecialisation<T, WeightType>()) {
    return {};
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> ampere_configs =
      kernels::cutlass_kernels::get_candidate_configs(sm, max_split_k, config_type_param);
  return ampere_configs;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
size_t DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getMaxWorkspaceSize(
    int num_experts) const {
  if (num_experts_ != num_experts) {
    TLLM_LOG_TRACE("Calling getMaxWorkspaceSize() with a new expert count %d vs %d", num_experts,
                   num_experts_);
    num_experts_ = num_experts;
    gemm_workspace_size_ = calcMaxWorkspaceSize(num_experts);
  }
  return gemm_workspace_size_;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
size_t DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::calcMaxWorkspaceSize(
    int num_experts) const {
  // Dual-weight grouped GEMM doesn't require additional workspace beyond the input buffers
  // The workspace is primarily used for intermediate results managed by the caller
  return 0;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
template <typename EpilogueTag>
void DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::runGemm(
    DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs) {
  dispatchToArch<EpilogueTag>(inputs);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
void DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::moeGemmBiasAct(
    DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs) {
  switch (inputs.activation_type) {
    case ActivationType::Relu:
      runGemm<cutlass_extensions::EpilogueOpDefaultReLU>(inputs);
      break;
    case ActivationType::Gelu:
      runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(inputs);
      break;
    case ActivationType::Silu:
      runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(inputs);
      break;
    case ActivationType::Identity:
      runGemm<cutlass_extensions::EpilogueOpDefault>(inputs);
      break;
    case ActivationType::Swiglu:
      runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(inputs);
      break;
    case ActivationType::Geglu:
      runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(inputs);
      break;
    case ActivationType::Relu2:
      TLLM_THROW("Relu2 is not supported.");
      break;
    case ActivationType::InvalidType:
      TLLM_THROW("Activation type for fpA_intB must be valid.");
      break;
    default:
      TLLM_THROW("Invalid activation type.");
      break;
  }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
void DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::moeGemm(
    DualWeightGroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs) {
  runGemm<cutlass_extensions::EpilogueOpDefault>(inputs);
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels
