/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#if defined(USING_OSS_CUTLASS_MOE_GEMM)
#include "dual_weight_moe_kernels.h"
#include "dual_weight_moe_gemm_kernels.h"
#else
#include "dual_weight_moe_kernels.h"
#include "dual_weight_moe_gemm_kernels.h"
#endif

// Include the implementation file for template definitions
#include "cutlass_dual_weight_fused_moe_kernels.cuh"

#include <tvm/ffi/extra/module.h>

#include <map>
#include <mutex>
#include <vector>

#include "../../tvm_ffi_utils.h"
#include "cutlass_kernel_selector.h"
#include "tensorrt_llm/common/workspace.h"

namespace common = tensorrt_llm::common;
namespace kernels = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
namespace cutlass_extensions = tensorrt_llm::cutlass_extensions;
using ActivationParams = CUTLASS_MOE_GEMM_NAMESPACE::ActivationParams;
using ActivationType = CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
using profiler_backend = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::DualWeightGemmProfilerBackend;

using tvm::ffi::Array;
using tvm::ffi::DLDataTypeToString;
using tvm::ffi::Function;
using tvm::ffi::Optional;

class DualWeightFusedMoeRunner : public tvm::ffi::ModuleObj {
 public:
  DualWeightFusedMoeRunner(DLDataType activation_dtype, DLDataType weight_dtype,
                           DLDataType output_dtype) {
    // Dual-weight path is currently hard-wired to half / fp8-e4m3 / half.
    TVM_FFI_ICHECK(activation_dtype.code == dl_float16.code &&
                   activation_dtype.bits == dl_float16.bits)
        << "Dual-weight fused MoE only supports float16 activations, got "
        << DLDataTypeToString(activation_dtype);
    TVM_FFI_ICHECK(weight_dtype.code == dl_float8_e4m3fn.code &&
                   weight_dtype.bits == dl_float8_e4m3fn.bits)
        << "Dual-weight fused MoE only supports fp8-e4m3 weights, got "
        << DLDataTypeToString(weight_dtype);
    TVM_FFI_ICHECK(output_dtype.code == dl_float16.code &&
                   output_dtype.bits == dl_float16.bits)
        << "Dual-weight fused MoE only supports float16 outputs, got "
        << DLDataTypeToString(output_dtype);
    
    mActivationDtype = activation_dtype;
    mWeightDtype = weight_dtype;
    mOutputDtype = output_dtype;

    mKernelRunner =
        std::make_shared<kernels::DualWeightMoeFCRunner<half, __nv_fp8_e4m3, half, half>>();
    if (!mKernelRunner) {
      TVM_FFI_ICHECK(false)
          << "Could not construct fused moe op with the requested input combination Activation: "
          << DLDataTypeToString(mActivationDtype)
          << ", Weight: " << DLDataTypeToString(mWeightDtype)
          << ", Output: " << DLDataTypeToString(mOutputDtype);
    }

    mProfiler = std::make_shared<kernels::DualWeightGemmProfilerBackend>();
    mAllProfiles = mKernelRunner->getTactics();
    TVM_FFI_ICHECK(!mAllProfiles.empty()) << "No valid tactics available for dual-weight fused MoE";
  }

  void runMoeDualWeight(
      TensorView output, TensorView input, TensorView token_selected_experts,
      Optional<TensorView> token_final_scales, TensorView fc1_upper, TensorView fc1_lower,
      TensorView fc2_upper, TensorView fc2_lower, Optional<TensorView> swiglu_alpha,
      Optional<TensorView> swiglu_beta, Optional<TensorView> swiglu_limit, int64_t tp_size,
      int64_t tp_rank, int64_t ep_size, int64_t ep_rank, int64_t cluster_size, 
      int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode, Optional<Array<int64_t>> profile_ids,
      bool /*enable_pdl*/, ActivationType base_activation_type = ActivationType::Swiglu) {
    std::lock_guard<std::mutex> lock(mMutex);

    TVM_FFI_ICHECK(cluster_size == 1 && cluster_rank == 0)
        << "smart_router is supported in min_latency mode";

    TVM_FFI_ICHECK(!min_latency_mode)
        << "Min latency mode is not implemented for SM80 dual weight.";

    CHECK_INPUT_TYPE(input, dl_float16);
    CHECK_INPUT_TYPE(token_selected_experts, dl_int32);
    if (token_final_scales.has_value()) {
      CHECK_INPUT_TYPE(token_final_scales.value(), dl_float32);
    }
    CHECK_INPUT_TYPE(fc1_upper, dl_float8_e4m3fn);
    CHECK_INPUT_TYPE(fc1_lower, dl_float8_e4m3fn);
    CHECK_INPUT_TYPE(fc2_upper, dl_float8_e4m3fn);
    CHECK_INPUT_TYPE(fc2_lower, dl_float8_e4m3fn);

    CHECK_DIM(2, input);
    CHECK_DIM(2, token_selected_experts);

    CHECK_DIM(3, fc1_upper);
    CHECK_DIM(3, fc1_lower);
    CHECK_DIM(3, fc2_upper);
    CHECK_DIM(3, fc2_lower);

    TVM_FFI_ICHECK_EQ(fc1_upper.size(0), fc1_lower.size(0))
        << "fc1 upper and lower weights must have identical shapes.";
    TVM_FFI_ICHECK_EQ(fc1_upper.size(1), fc1_lower.size(1))
        << "fc1 upper and lower weights must have identical shapes.";
    TVM_FFI_ICHECK_EQ(fc1_upper.size(2), fc1_lower.size(2))
        << "fc1 upper and lower weights must have identical shapes.";
    TVM_FFI_ICHECK_EQ(fc2_upper.size(0), fc2_lower.size(0))
        << "fc2 upper and lower weights must have identical shapes.";
    TVM_FFI_ICHECK_EQ(fc2_upper.size(1), fc2_lower.size(1))
        << "fc2 upper and lower weights must have identical shapes.";
    TVM_FFI_ICHECK_EQ(fc2_upper.size(2), fc2_lower.size(2))
        << "fc2 upper and lower weights must have identical shapes.";

    TVM_FFI_ICHECK_EQ(input.size(0), token_selected_experts.size(0))
        << "input and token_selected_experts must have the same num tokens.";
    if (token_final_scales.has_value()) {
      TVM_FFI_ICHECK_EQ(input.size(0), token_final_scales.value().size(0))
          << "input and token_final_scales must have the same num tokens.";
      TVM_FFI_ICHECK_EQ(token_selected_experts.size(1), token_final_scales.value().size(1))
          << "token_selected_experts and token_final_scales must have the same experts-per-token.";
    }
    TVM_FFI_ICHECK_EQ(fc1_upper.size(0), fc2_upper.size(0))
        << "fc1 and fc2 weights must have the same number of experts.";
    if (isGatedActivation(base_activation_type)) {
      TVM_FFI_ICHECK_EQ(fc1_upper.size(1), fc2_upper.size(2) * 2)
          << "fc1 upper inter size must be 2 times fc2 upper inter size.";
    } else {
      TVM_FFI_ICHECK_EQ(fc1_upper.size(1), fc2_upper.size(2))
          << "fc1 upper inter size must be equal to fc2 upper inter size.";
    }

    int experts_per_token = token_selected_experts.size(1);
    int64_t num_rows = input.size(0);
    int64_t hidden_size_in = input.size(1);
    int64_t hidden_size = fc2_upper.size(1);
    int64_t inter_size = fc2_upper.size(2);
    int const num_experts_on_rank = fc2_upper.size(0);
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
    if (swiglu_alpha.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_alpha.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_alpha.value().size(0), num_experts_on_rank)
          << "swiglu_alpha must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    if (swiglu_beta.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_beta.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_beta.value().size(0), num_experts_on_rank)
          << "swiglu_beta must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    if (swiglu_limit.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_limit.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_limit.value().size(0), num_experts_on_rank)
          << "swiglu_limit must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    auto activation_params = ActivationParams(
        base_activation_type,
        reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().data_ptr()
                                                                : nullptr),
        reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().data_ptr()
                                                               : nullptr),
        reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().data_ptr()
                                                                : nullptr));

    setRunnerProfiles(profile_ids);

    auto stream = get_stream(input.device());

    WorkspaceInfo workspace_info = getWorkspaceInfo(
        num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
        base_activation_type, parallelism_config);

    auto token_final_scales_ptr = token_final_scales.has_value()
                                      ? reinterpret_cast<float const*>(
                                            token_final_scales.value().data_ptr())
                                      : nullptr;

    mKernelRunner->runMoe(
        input.data_ptr(),
        reinterpret_cast<int32_t const*>(token_selected_experts.data_ptr()),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value().data_ptr())
            : nullptr,
        fc1_upper.data_ptr(), fc1_lower.data_ptr(),
        nullptr, // fc1_expert_biases
        activation_params,
        fc2_upper.data_ptr(), fc2_lower.data_ptr(),
        nullptr, // fc2_expert_biases
        num_rows, hidden_size, inter_size, num_experts_total,
        static_cast<int>(experts_per_token),
        static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
        static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, enable_alltoall,
        stream);
  }

  int64_t getTacticNum() {
    std::lock_guard<std::mutex> lock(mMutex);
    return mAllProfiles.size();
  }

  void runGemmProfileDualWeight(TensorView input, TensorView fc1_upper,
                                TensorView fc1_lower, TensorView fc2_upper,
                                TensorView fc2_lower, int64_t top_k, int64_t tp_size,
                                int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                                int64_t cluster_size, int64_t cluster_rank,
                                bool enable_alltoall, int64_t gemm_idx, int64_t profile_id,
                                bool do_preparation, ActivationType activation_type) {
    std::lock_guard<std::mutex> lock(mMutex);

    TVM_FFI_ICHECK(!mAllProfiles.empty()) << "No tactics available for dual-weight fused MoE";

    int64_t num_rows = input.size(0);
    int64_t hidden_size = fc2_upper.size(1);
    int64_t inter_size = fc2_upper.size(2);
    int const num_experts = static_cast<int>(fc2_upper.size(0) * ep_size);

    // Get specific profile configs according to the profile_id.
    // Fallback tactic is set to be 0
    auto profile = profile_id == -1 ? mAllProfiles.front() : mAllProfiles[profile_id];

    auto stream = get_stream(input.device());

    auto const* upper_weights_ptr = (gemm_idx == 1) ? fc1_upper.data_ptr() : fc2_upper.data_ptr();
    auto const* lower_weights_ptr = (gemm_idx == 1) ? fc1_lower.data_ptr() : fc2_lower.data_ptr();

    // Preparation phase: initialize profiler and workspace
    if (do_preparation) {
      // Set profiled gemm idx
      mProfiler->mGemmToProfile = (gemm_idx == 1) 
          ? profiler_backend::GemmToProfile::GEMM_1
          : profiler_backend::GemmToProfile::GEMM_2;

      // Initialize profiler
      auto parallelism_config = kernels::MOEParallelismConfig(
          static_cast<int>(tp_size), static_cast<int>(tp_rank),
          static_cast<int>(ep_size), static_cast<int>(ep_rank),
          static_cast<int>(cluster_size), static_cast<int>(cluster_rank));

      // bool USE_BIAS = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
      bool USE_BIAS = false;
      bool USE_LORA = false;

      mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                      nvinfer1::DataType::kHALF, nvinfer1::DataType::kFP8,
                      nvinfer1::DataType::kHALF,
                      num_experts, static_cast<int>(top_k), hidden_size, inter_size,
                      activation_type, USE_BIAS, USE_LORA, parallelism_config, enable_alltoall);
        
      // Allocate workspace
      size_t profile_workspace_size = mProfiler->getWorkspaceSize(num_rows);
      int device_id;
      cudaGetDevice(&device_id);
      mProfileWorkspace = alloc_tensor({static_cast<int64_t>(profile_workspace_size)}, dl_int8,
                                       DLDevice{kDLCUDA, device_id});

      // Prepare profiler with workspace and weights
      mProfiler->prepare(num_rows, static_cast<char*>(mProfileWorkspace.data_ptr()),
                         upper_weights_ptr, lower_weights_ptr, stream);
    }

    // Profile specific tactic
    mProfiler->runProfiler(num_rows, profile, static_cast<char*>(mProfileWorkspace.data_ptr()),
                           upper_weights_ptr, lower_weights_ptr, stream);
  }

  const char* kind() const final { return "dual_weight_fused_moe_runner"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "get_tactic_num") {
      return Function::FromTyped([this]() { return getTacticNum(); });
    }
    if (name == "run_gemm_profile_dual_weight") {
      return Function::FromTyped([this](TensorView input, TensorView fc1_upper,
                                        TensorView fc1_lower, TensorView fc2_upper,
                                        TensorView fc2_lower, int64_t top_k, int64_t tp_size,
                                        int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                                        int64_t cluster_size, int64_t cluster_rank,
                                        bool enable_alltoall, int64_t gemm_idx, int64_t tactic,
                                        bool do_preparation, int64_t base_activation_type) {
        runGemmProfileDualWeight(input, fc1_upper, fc1_lower, fc2_upper, fc2_lower, top_k, tp_size,
                                 tp_rank, ep_size, ep_rank, cluster_size, cluster_rank,
                                 enable_alltoall, gemm_idx, tactic, do_preparation,
                                 static_cast<ActivationType>(base_activation_type));
      });
    }
    if (name == "run_moe_dual_weight") {
      return Function::FromTyped(
          [this](TensorView output, TensorView input, TensorView token_selected_experts,
                 Optional<TensorView> token_final_scales, TensorView fc1_upper,
                 TensorView fc1_lower, TensorView fc2_upper, TensorView fc2_lower, 
                 Optional<TensorView> swiglu_alpha, Optional<TensorView> swiglu_beta,
                 Optional<TensorView> swiglu_limit, int64_t tp_size,
                 int64_t tp_rank, int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                 int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode,
                 Optional<Array<int64_t>> profile_ids, bool enable_pdl, int64_t base_activation_type) {
            runMoeDualWeight(output, input, token_selected_experts, token_final_scales, fc1_upper,
                             fc1_lower, fc2_upper, fc2_lower, swiglu_alpha, swiglu_beta, swiglu_limit,
                             tp_size, tp_rank, ep_size, ep_rank, cluster_size, cluster_rank,
                             enable_alltoall, min_latency_mode,
                             profile_ids, enable_pdl, static_cast<ActivationType>(base_activation_type));
          });
    }

    return Function(nullptr);
  }

 private:
  struct WorkspaceInfo {
    Tensor workspace{};
    void* src_to_dest_map{};
  };

  std::mutex mMutex;
  std::shared_ptr<kernels::DualWeightMoeFCRunnerInterface> mKernelRunner;
  std::shared_ptr<kernels::DualWeightGemmProfilerBackend> mProfiler;
  DLDataType mActivationDtype;
  DLDataType mWeightDtype;
  DLDataType mOutputDtype;

  // Profiling workspace tensor (keeps memory alive)
  Tensor mProfileWorkspace;

  using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
  std::vector<Profile> mAllProfiles;

  void setRunnerProfiles(Optional<Array<int64_t>> profile_ids) {
    auto best_gemm1_profile = mAllProfiles.front();
    auto best_gemm2_profile = mAllProfiles.front();
    if (profile_ids.has_value()) {
      TVM_FFI_ICHECK_EQ(profile_ids.value().size(), 2) << "Expecting 2 profile ids";
      best_gemm1_profile = profile_ids.value()[0] == -1 ? best_gemm1_profile
                                                        : mAllProfiles.at(profile_ids.value()[0]);
      best_gemm2_profile = profile_ids.value()[1] == -1 ? best_gemm2_profile
                                                        : mAllProfiles.at(profile_ids.value()[1]);
    }
    mKernelRunner->setTactic(best_gemm1_profile, best_gemm2_profile);
  }

  WorkspaceInfo getWorkspaceInfo(
      int64_t num_rows, int64_t hidden_size, int64_t inter_size, int num_experts,
      int experts_per_token, ActivationType activation_type,
      kernels::MOEParallelismConfig parallelism_config) {
    size_t moe_workspace_size = mKernelRunner->getWorkspaceSize(
        num_rows, hidden_size, inter_size, num_experts, experts_per_token, activation_type,
        parallelism_config);
    size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);

    std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};

    size_t total_workspace_size =
        common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

    WorkspaceInfo info{};
    int device_id;
    cudaGetDevice(&device_id);
    info.workspace = alloc_tensor({static_cast<int64_t>(total_workspace_size)}, dl_int8,
                                  DLDevice{kDLCUDA, device_id});
    info.src_to_dest_map = common::nextWorkspacePtr(static_cast<int8_t*>(info.workspace.data_ptr()),
                                                    moe_workspace_size);
    return info;
  }

};

tvm::ffi::Module init(DLDataType activation_dtype, DLDataType weight_dtype,
                      DLDataType output_dtype) {
  auto ptr =
      tvm::ffi::make_object<DualWeightFusedMoeRunner>(activation_dtype, weight_dtype, output_dtype);
  return tvm::ffi::Module(ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);

