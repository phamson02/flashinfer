/*
 * Copyright (c) 2026, FlashInfer.
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
#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/dense_dual_weight_gemm.h"
#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

class CutlassDualWeightGemmRunnerInterface {
 public:
  virtual ~CutlassDualWeightGemmRunnerInterface() = default;

  virtual void gemm(void const* a, void const* b_upper, void const* b_lower,
                    void* c, int m, int n, int k, int64_t lda, int64_t ldb,
                    int64_t ldc, tkc::CutlassGemmConfig gemm_config,
                    char* workspace_ptr, size_t workspace_bytes,
                    cudaStream_t stream) = 0;

  virtual size_t getWorkspaceSize(int m, int n, int k) const = 0;

  virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;
};

template <typename WeightType = __nv_fp8_e4m3, bool kTruncateE5M2 = false, bool kFastE4M3 = false>
class CutlassDualWeightGemmRunner : public CutlassDualWeightGemmRunnerInterface {
 public:
  CutlassDualWeightGemmRunner();
  ~CutlassDualWeightGemmRunner() override = default;

  void gemm(void const* a, void const* b_upper, void const* b_lower, void* c,
            int m, int n, int k, int64_t lda, int64_t ldb, int64_t ldc,
            tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
            size_t workspace_bytes, cudaStream_t stream) override;

  size_t getWorkspaceSize(int m, int n, int k) const override;

  std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

 private:
  template <typename Arch, typename ThreadblockShape, typename WarpShape, int Stages,
            bool kKBlockInterleaved = false>
  static void launchGemm(void const* a, void const* b_upper, void const* b_lower,
                         void* c, int m, int n, int k, int64_t lda, int64_t ldb,
                         int64_t ldc, tkc::CutlassGemmConfig gemm_config,
                         char* workspace_ptr, size_t workspace_bytes,
                         cudaStream_t stream);

  template <typename Arch, typename ThreadblockShape, typename WarpShape,
            bool kKBlockInterleaved = false>
  static void dispatchGemmStages(void const* a, void const* b_upper,
                                 void const* b_lower, void* c, int m, int n,
                                 int k, int64_t lda, int64_t ldb, int64_t ldc,
                                 tkc::CutlassGemmConfig gemm_config,
                                 char* workspace_ptr, size_t workspace_bytes,
                                 cudaStream_t stream);

  template <typename Arch, bool kKBlockInterleaved>
  static void dispatchGemmToCutlass(void const* a, void const* b_upper,
                                    void const* b_lower, void* c, int m, int n,
                                    int k, int64_t lda, int64_t ldb, int64_t ldc,
                                    tkc::CutlassGemmConfig gemm_config,
                                    char* workspace_ptr, size_t workspace_bytes,
                                    cudaStream_t stream);

  // kTruncateE5M2 is a class-level template parameter — no per-method parameter needed.

  int sm_;
  int multi_processor_count_;

  static constexpr int kSplitKLimit = 7;
  static constexpr int kMinMTile = 16;
  static constexpr int kMinNTile = 64;
};

template <typename WeightType, bool kTruncateE5M2, bool kFastE4M3>
template <typename Arch, typename ThreadblockShape, typename WarpShape, int Stages,
          bool kKBlockInterleaved>
void CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2, kFastE4M3>::launchGemm(
    void const* a, void const* b_upper, void const* b_lower, void* c, int m,
    int n, int k, int64_t lda, int64_t ldb, int64_t ldc,
    tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
    size_t workspace_bytes, cudaStream_t stream) {
  using ActivationType = half;
  using OutputType = half;

  using CutlassActivationType = typename TllmToCutlassTypeAdapter<ActivationType>::type;
  using CutlassWeightType = typename TllmToCutlassTypeAdapter<WeightType>::type;
  using CutlassOutputType = typename TllmToCutlassTypeAdapter<OutputType>::type;
  using MixedGemmArchTraits =
      cutlass::gemm::kernel::MixedGemmArchTraits<CutlassActivationType,
                                                 CutlassWeightType, Arch>;
  using ElementAccumulator = typename MixedGemmArchTraits::AccType;
  constexpr int ElementsPerAccessC =
      128 / cutlass::sizeof_bits<CutlassOutputType>::value;

  using EpilogueOp = typename tkc::Epilogue<CutlassOutputType, ElementsPerAccessC,
                                            ElementAccumulator,
                                            tkc::EpilogueOpDefault>::Op;
  using TaggedOperator = typename cutlass::arch::TagOperator<
      typename MixedGemmArchTraits::Operator,
      cutlass::WeightOnlyQuantOp::UNDEFINED>::TaggedOperator;

  using Gemm = cutlass::gemm::device::DualWeightGemm<
      CutlassActivationType, cutlass::layout::RowMajor,
      CutlassWeightType, typename MixedGemmArchTraits::LayoutB,
      CutlassOutputType, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, Arch, ThreadblockShape, WarpShape,
      typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Stages,
      MixedGemmArchTraits::ElementsPerAccessA,
      MixedGemmArchTraits::ElementsPerAccessB, true, TaggedOperator,
      kKBlockInterleaved, kTruncateE5M2, kFastE4M3>;

  if (gemm_config.enableCudaKernel) {
    throw std::runtime_error(
        "CUDA-kernel tactics are not supported for dual_weight_mm.");
  }

  typename EpilogueOp::Params epilogue(ElementAccumulator(1.f),
                                       ElementAccumulator(0.f));
  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<CutlassActivationType const*>(a), lda},
      {reinterpret_cast<CutlassWeightType const*>(b_upper), ldb},
      {reinterpret_cast<CutlassWeightType const*>(b_lower), ldb},
      {reinterpret_cast<CutlassOutputType const*>(c), ldc},
      {reinterpret_cast<CutlassOutputType*>(c), ldc},
      epilogue, gemm_config.split_k_factor);

  Gemm gemm;
  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg =
        "dual_weight_mm CUTLASS kernel will fail for params. Error: " +
        std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[TensorRT LLM Error][dual_weight_mm] " + err_msg);
  }

  if (gemm.get_workspace_size(args) > workspace_bytes) {
    typename Gemm::Arguments fallback_args = args;
    fallback_args.split_k_slices = 1;
    auto init_status = gemm.initialize(fallback_args, workspace_ptr, stream);
    if (init_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to initialize dual_weight_mm fallback gemm. Error: " +
          std::string(cutlassGetStatusString(init_status));
      throw std::runtime_error("[TensorRT LLM Error][dual_weight_mm] " + err_msg);
    }
  } else {
    auto init_status = gemm.initialize(args, workspace_ptr, stream);
    if (init_status != cutlass::Status::kSuccess) {
      std::string err_msg =
          "Failed to initialize dual_weight_mm gemm. Error: " +
          std::string(cutlassGetStatusString(init_status));
      throw std::runtime_error("[TensorRT LLM Error][dual_weight_mm] " + err_msg);
    }
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run dual_weight_mm gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("[TensorRT LLM Error][dual_weight_mm] " + err_msg);
  }
}

template <typename WeightType, bool kTruncateE5M2, bool kFastE4M3>
template <typename Arch, typename ThreadblockShape, typename WarpShape,
          bool kKBlockInterleaved>
void CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2, kFastE4M3>::dispatchGemmStages(
    void const* a, void const* b_upper, void const* b_lower, void* c, int m,
    int n, int k, int64_t lda, int64_t ldb, int64_t ldc,
    tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
    size_t workspace_bytes, cudaStream_t stream) {
  switch (gemm_config.stages) {
    case 2:
      launchGemm<Arch, ThreadblockShape, WarpShape, 2, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case 3:
      launchGemm<Arch, ThreadblockShape, WarpShape, 3, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case 4:
      launchGemm<Arch, ThreadblockShape, WarpShape, 4, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    default:
      throw std::runtime_error(
          "dual_weight_mm does not support the requested pipeline stages.");
  }
}

// Tile-level dispatch; kKBlockInterleaved selects the mainloop reconstruction order.
template <typename WeightType, bool kTruncateE5M2, bool kFastE4M3>
template <typename Arch, bool kKBlockInterleaved>
void CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2, kFastE4M3>::dispatchGemmToCutlass(
    void const* a, void const* b_upper, void const* b_lower, void* c, int m,
    int n, int k, int64_t lda, int64_t ldb, int64_t ldc,
    tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
    size_t workspace_bytes, cudaStream_t stream) {
  switch (gemm_config.tile_config_sm80) {
    case tkc::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<16, 128, 64>,
                         cutlass::gemm::GemmShape<16, 32, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<16, 256, 64>,
                         cutlass::gemm::GemmShape<16, 64, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<32, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<128, 32, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    // Extended tile configs
    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<32, 64, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<128, 64, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
      dispatchGemmStages<Arch, cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<64, 64, 64>, kKBlockInterleaved>(
          a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
          workspace_ptr, workspace_bytes, stream);
      break;
    case tkc::CutlassTileConfig::ChooseWithHeuristic:
    case tkc::CutlassTileConfig::Undefined:
      throw std::runtime_error(
          "dual_weight_mm received an unresolved CUTLASS config.");
    default:
      throw std::runtime_error(
          "dual_weight_mm received an unsupported CUTLASS tile config.");
  }
}

template <typename WeightType, bool kTruncateE5M2, bool kFastE4M3>
inline CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2, kFastE4M3>::CutlassDualWeightGemmRunner() {
  int device{-1};
  tk::check_cuda_error(cudaGetDevice(&device));
  sm_ = tk::getSMVersion();
  tk::check_cuda_error(cudaDeviceGetAttribute(
      &multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename WeightType, bool kTruncateE5M2, bool kFastE4M3>
inline void CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2, kFastE4M3>::gemm(
    void const* a, void const* b_upper, void const* b_lower, void* c, int m,
    int n, int k, int64_t lda, int64_t ldb, int64_t ldc,
    tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
    size_t workspace_bytes, cudaStream_t stream) {
  if (sm_ != 80) {
    throw std::runtime_error(
        "dual_weight_mm is currently supported on SM80 only.");
  }

  if (gemm_config.kblock_interleaved) {
    dispatchGemmToCutlass<cutlass::arch::Sm80, true>(
        a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
        workspace_ptr, workspace_bytes, stream);
  } else {
    dispatchGemmToCutlass<cutlass::arch::Sm80, false>(
        a, b_upper, b_lower, c, m, n, k, lda, ldb, ldc, gemm_config,
        workspace_ptr, workspace_bytes, stream);
  }
}

template <typename WeightType, bool kTruncateE5M2, bool kFastE4M3>
inline size_t CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2, kFastE4M3>::getWorkspaceSize(int m, int n,
                                                            int k) const {
  int max_grid_m = cutlass::ceil_div(m, kMinMTile);
  int max_grid_n = cutlass::ceil_div(n, kMinNTile);
  return static_cast<size_t>(max_grid_m * max_grid_n * kSplitKLimit * 4);
}

template <typename WeightType, bool kTruncateE5M2, bool kFastE4M3>
inline std::vector<tkc::CutlassGemmConfig>
CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2, kFastE4M3>::getConfigs() const {
  if (sm_ != 80) {
    return {};
  }

  // Original 5 tiles (keep first so tactic IDs 0-104 are unchanged) + 7 extended tiles.
  static const tkc::CutlassTileConfig kTiles[] = {
      // Original WEIGHT_ONLY tiles (sm>=75)
      tkc::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
      tkc::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64,
      tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
      tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
      tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64,
      // Extended tiles
      tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64,
      tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
      tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
      tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64,
  };

  // Emit configs for both standard and k-block interleaved mainloop variants.
  // Standard configs come first (tactic indices 0 .. 9*21-1 = 188).
  // K-block interleaved configs follow (tactic indices 189 .. 378).
  std::vector<tkc::CutlassGemmConfig> configs;
  for (bool kblock_interleaved : {false, true}) {
    for (auto const& tile : kTiles) {
      for (int stages = 2; stages <= 4; ++stages) {
        tkc::CutlassGemmConfig cfg(tile, tkc::SplitKStyle::NO_SPLIT_K, 1, stages);
        cfg.kblock_interleaved = kblock_interleaved;
        configs.push_back(cfg);
        for (int split_k = 2; split_k <= kSplitKLimit; ++split_k) {
          tkc::CutlassGemmConfig sk_cfg(tile, tkc::SplitKStyle::SPLIT_K_SERIAL, split_k, stages);
          sk_cfg.kblock_interleaved = kblock_interleaved;
          configs.push_back(sk_cfg);
        }
      }
    }
  }
  return configs;
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm
