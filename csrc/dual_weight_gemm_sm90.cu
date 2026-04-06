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

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "tvm_ffi_utils.h"

using namespace cute;

//-----------------------------------------------------------------------
// SM90 dual-weight GEMM context for a specific tile shape.
// Computes D(N,M) = reconstruct(A1_upper, A2_lower)(N,K) × B(M,K)^T
// where A1/A2 are FP8 dual-weight halves and B is FP16 input.
//
// ScheduleSelector: maps (IsCooperative) → kernel schedule tag.
// Default selects E4M3 RTN schedules; E5M2 RTN uses a different selector.
//-----------------------------------------------------------------------
struct E4M3ScheduleSelector {
  using NonCooperative = cutlass::gemm::KernelTmaWarpSpecializedCustom;
  using Cooperative    = cutlass::gemm::KernelTmaWarpSpecializedCooperativeCustom;
};
struct E5M2ScheduleSelector {
  using NonCooperative = cutlass::gemm::KernelTmaWarpSpecializedDualWeightE5M2;
  using Cooperative    = cutlass::gemm::KernelTmaWarpSpecializedCooperativeDualWeightE5M2;
};

template<int TileM, int TileN, int TileK,
         int CgaM = 1, int CgaN = 1, int CgaK = 1,
         class ScheduleSelector = E4M3ScheduleSelector>
class Sm90DualWeightGemmCtx {
 public:
  static constexpr bool IsCooperative = (CgaM > 1 || CgaN > 1);
  using MainloopScheduleType = cute::conditional_t<IsCooperative,
      typename ScheduleSelector::Cooperative,
      typename ScheduleSelector::NonCooperative>;
  using EpilogueScheduleType = cute::conditional_t<IsCooperative,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::TmaWarpSpecialized>;
  using TileSchedulerType    = cutlass::gemm::PersistentScheduler;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          Shape<Int<TileM>, Int<TileN>, Int<TileK>>,
          Shape<_1, _1, _1>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          float, float,
          cutlass::half_t, cutlass::layout::ColumnMajor, 8,
          cutlass::half_t, cutlass::layout::ColumnMajor, 8,
          EpilogueScheduleType>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          cutlass::half_t, cutlass::layout::RowMajor,    8,
          cutlass::half_t, cutlass::layout::ColumnMajor, 8,
          float,
          Shape<Int<TileM>, Int<TileN>, Int<TileK>>,
          Shape<Int<CgaM>, Int<CgaN>, Int<CgaK>>,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel   = cutlass::gemm::kernel::GemmUniversal<
                         Shape<int,int,int,int>,
                         CollectiveMainloop,
                         CollectiveEpilogue,
                         TileSchedulerType>;
  using DeviceKernel = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ElementCompute = typename DeviceKernel::EpilogueOutputOp::ElementCompute;

  using StrideA = typename DeviceKernel::GemmKernel::StrideA;
  using StrideB = typename DeviceKernel::GemmKernel::StrideB;
  using StrideC = typename DeviceKernel::GemmKernel::StrideC;
  using StrideD = typename DeviceKernel::GemmKernel::StrideD;

  void maybe_reinit(int M, int N, int K) {
    if (initialized_ && M_ == M && N_ == N && K_ == K) return;
    M_ = M; N_ = N; K_ = K;
    size_t ws = DeviceKernel::get_workspace_size(typename DeviceKernel::Arguments{});
    if (ws > workspace_size_) {
      if (workspace_ptr_) cudaFree(workspace_ptr_);
      cudaMalloc(&workspace_ptr_, ws);
      workspace_size_ = ws;
    }
    int device_id; cudaGetDevice(&device_id);
    hw_.device_id = device_id;
    hw_.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    initialized_ = true;
  }

  void run(cudaStream_t stream,
           const cutlass::float_e4m3_t* A1,
           const cutlass::float_e4m3_t* A2,
           const cutlass::half_t* B,
           cutlass::half_t* D) {
    auto args = typename DeviceKernel::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {M_, N_, K_, 1},
      { A1, A2,
        cutlass::make_cute_packed_stride(StrideA{}, make_shape(M_, K_, 1)),
        B,
        cutlass::make_cute_packed_stride(StrideB{}, make_shape(N_, K_, 1)) },
      { {ElementCompute(1.f), ElementCompute(0.f)},
        nullptr,
        cutlass::make_cute_packed_stride(StrideC{}, make_shape(M_, N_, 1)),
        D,
        cutlass::make_cute_packed_stride(StrideD{}, make_shape(M_, N_, 1)) },
      hw_
    };
    auto st = gemm_.run(args, static_cast<uint8_t*>(workspace_ptr_), stream);
    TVM_FFI_ICHECK(st == cutlass::Status::kSuccess) << "SM90 dual-weight GEMM failed";
  }

  ~Sm90DualWeightGemmCtx() {
    if (workspace_ptr_) cudaFree(workspace_ptr_);
  }

 private:
  bool initialized_{false};
  int M_{0}, N_{0}, K_{0};
  void* workspace_ptr_{nullptr};
  size_t workspace_size_{0};
  DeviceKernel gemm_;
  cutlass::KernelHardwareInfo hw_;
};

//-----------------------------------------------------------------------
// Tactic dispatch: select tile shape by tactic index.
//-----------------------------------------------------------------------
namespace {

// SM90 tile shape + cluster shape registry — matches dual-weight fused MOE SM90 configs.
// Tile shapes: M∈{128,256} × N∈{16,32,64,128,256} × K=64
// Cluster shapes: (1,1,1), (2,1,1), (1,2,1), (2,2,1).
// Rules: (2,1,1) requires M>=128, (1,2,1) requires N>=128, (2,2,1) requires both.
static constexpr int kNumTactics = 18;

#define DECL_CTX(id, M_, N_, K_, C1_, C2_, C3_) \
  static thread_local Sm90DualWeightGemmCtx<M_, N_, K_, C1_, C2_, C3_> ctx_##id;

// Non-cooperative (cluster 1,1,1)
DECL_CTX(0,  128,  16, 64, 1,1,1)
DECL_CTX(1,  128,  32, 64, 1,1,1)
DECL_CTX(2,  128,  64, 64, 1,1,1)
DECL_CTX(3,  128, 128, 64, 1,1,1)
DECL_CTX(4,  128, 256, 64, 1,1,1)
DECL_CTX(5,  256, 128, 64, 1,1,1)
// Cooperative (cluster 2,1,1) — M>=128
DECL_CTX(6,  128,  16, 64, 2,1,1)
DECL_CTX(7,  128,  32, 64, 2,1,1)
DECL_CTX(8,  128,  64, 64, 2,1,1)
DECL_CTX(9,  128, 128, 64, 2,1,1)
DECL_CTX(10, 128, 256, 64, 2,1,1)
DECL_CTX(11, 256, 128, 64, 2,1,1)
// Cooperative (cluster 1,2,1) — N>=128
DECL_CTX(12, 128, 128, 64, 1,2,1)
DECL_CTX(13, 128, 256, 64, 1,2,1)
DECL_CTX(14, 256, 128, 64, 1,2,1)
// Cooperative (cluster 2,2,1) — M>=128, N>=128
DECL_CTX(15, 128, 128, 64, 2,2,1)
DECL_CTX(16, 128, 256, 64, 2,2,1)
DECL_CTX(17, 256, 128, 64, 2,2,1)

#undef DECL_CTX

// E5M2 RTN tactic registry — same tile/cluster shapes, different reconstruction.
static constexpr int kNumTacticsE5M2 = 18;

#define DECL_CTX_E5M2(id, M_, N_, K_, C1_, C2_, C3_) \
  static thread_local Sm90DualWeightGemmCtx<M_, N_, K_, C1_, C2_, C3_, E5M2ScheduleSelector> ctx_e5m2_##id;

DECL_CTX_E5M2(0,  128,  16, 64, 1,1,1)
DECL_CTX_E5M2(1,  128,  32, 64, 1,1,1)
DECL_CTX_E5M2(2,  128,  64, 64, 1,1,1)
DECL_CTX_E5M2(3,  128, 128, 64, 1,1,1)
DECL_CTX_E5M2(4,  128, 256, 64, 1,1,1)
DECL_CTX_E5M2(5,  256, 128, 64, 1,1,1)
DECL_CTX_E5M2(6,  128,  16, 64, 2,1,1)
DECL_CTX_E5M2(7,  128,  32, 64, 2,1,1)
DECL_CTX_E5M2(8,  128,  64, 64, 2,1,1)
DECL_CTX_E5M2(9,  128, 128, 64, 2,1,1)
DECL_CTX_E5M2(10, 128, 256, 64, 2,1,1)
DECL_CTX_E5M2(11, 256, 128, 64, 2,1,1)
DECL_CTX_E5M2(12, 128, 128, 64, 1,2,1)
DECL_CTX_E5M2(13, 128, 256, 64, 1,2,1)
DECL_CTX_E5M2(14, 256, 128, 64, 1,2,1)
DECL_CTX_E5M2(15, 128, 128, 64, 2,2,1)
DECL_CTX_E5M2(16, 128, 256, 64, 2,2,1)
DECL_CTX_E5M2(17, 256, 128, 64, 2,2,1)

#undef DECL_CTX_E5M2

void run_tactic(int tactic, int M, int N, int K, cudaStream_t stream,
                const cutlass::float_e4m3_t* A1, const cutlass::float_e4m3_t* A2,
                const cutlass::half_t* B, cutlass::half_t* D) {
#define CASE(id, ctx) case id: ctx.maybe_reinit(M, N, K); ctx.run(stream, A1, A2, B, D); break;
  switch (tactic) {
    CASE(0, ctx_0) CASE(1, ctx_1) CASE(2, ctx_2) CASE(3, ctx_3) CASE(4, ctx_4) CASE(5, ctx_5)
    CASE(6, ctx_6) CASE(7, ctx_7) CASE(8, ctx_8) CASE(9, ctx_9) CASE(10, ctx_10) CASE(11, ctx_11)
    CASE(12, ctx_12) CASE(13, ctx_13) CASE(14, ctx_14) CASE(15, ctx_15) CASE(16, ctx_16) CASE(17, ctx_17)
    default: TVM_FFI_ICHECK(false) << "Invalid tactic: " << tactic << " (max: " << kNumTactics - 1 << ")";
  }
#undef CASE
}

void run_tactic_e5m2(int tactic, int M, int N, int K, cudaStream_t stream,
                     const cutlass::float_e4m3_t* A1, const cutlass::float_e4m3_t* A2,
                     const cutlass::half_t* B, cutlass::half_t* D) {
#define CASE(id, ctx) case id: ctx.maybe_reinit(M, N, K); ctx.run(stream, A1, A2, B, D); break;
  switch (tactic) {
    CASE(0, ctx_e5m2_0) CASE(1, ctx_e5m2_1) CASE(2, ctx_e5m2_2) CASE(3, ctx_e5m2_3) CASE(4, ctx_e5m2_4) CASE(5, ctx_e5m2_5)
    CASE(6, ctx_e5m2_6) CASE(7, ctx_e5m2_7) CASE(8, ctx_e5m2_8) CASE(9, ctx_e5m2_9) CASE(10, ctx_e5m2_10) CASE(11, ctx_e5m2_11)
    CASE(12, ctx_e5m2_12) CASE(13, ctx_e5m2_13) CASE(14, ctx_e5m2_14) CASE(15, ctx_e5m2_15) CASE(16, ctx_e5m2_16) CASE(17, ctx_e5m2_17)
    default: TVM_FFI_ICHECK(false) << "Invalid E5M2 tactic: " << tactic << " (max: " << kNumTacticsE5M2 - 1 << ")";
  }
#undef CASE
}

//-----------------------------------------------------------------------
// Shared validation + dispatch logic
//-----------------------------------------------------------------------
void dual_weight_mm_sm90_impl(TensorView a, TensorView b_upper, TensorView b_lower,
                              TensorView out, TensorView workspace_buffer,
                              int64_t tactic) {
  CHECK_CUDA(a);
  CHECK_CUDA(b_upper);
  CHECK_CUDA(b_lower);
  CHECK_CUDA(out);
  CHECK_INPUT_TYPE(a, dl_float16);
  CHECK_INPUT_TYPE(out, dl_float16);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b_upper);
  CHECK_DIM(2, b_lower);
  CHECK_DIM(2, out);
  CHECK_DEVICE(a, b_upper);
  CHECK_DEVICE(a, b_lower);
  CHECK_DEVICE(a, out);

  TVM_FFI_ICHECK_EQ(b_upper.stride(0), 1) << "b_upper must be column-major";
  TVM_FFI_ICHECK_EQ(b_lower.stride(0), 1) << "b_lower must be column-major";
  TVM_FFI_ICHECK_EQ(a.stride(1), 1) << "a must be row-major";
  TVM_FFI_ICHECK_EQ(out.stride(1), 1) << "out must be row-major";

  TVM_FFI_ICHECK_EQ(b_upper.size(0), b_lower.size(0)) << "b_upper and b_lower shape mismatch (dim 0).";
  TVM_FFI_ICHECK_EQ(b_upper.size(1), b_lower.size(1)) << "b_upper and b_lower shape mismatch (dim 1).";

  // a is (M_input, K), b_upper/b_lower are (K, N_weight) col-major, out is (M_input, N_weight)
  int64_t M_input = a.size(0);
  int64_t K_dim = a.size(1);
  int64_t N_weight = b_upper.size(1);

  TVM_FFI_ICHECK_EQ(b_upper.size(0), K_dim) << "K dimension mismatch.";
  TVM_FFI_ICHECK_EQ(out.size(0), M_input) << "out M dimension mismatch.";
  TVM_FFI_ICHECK_EQ(out.size(1), N_weight) << "out N dimension mismatch.";

  if (tactic == -1) tactic = 0;
  TVM_FFI_ICHECK(tactic >= 0 && tactic < kNumTactics)
      << "tactic must be in [0, " << kNumTactics << "), got " << tactic;

  // CUTLASS GEMM: D(gemm_M, gemm_N) = A(gemm_M, K) × B(gemm_N, K)^T
  // where A = dual FP8 weight, B = FP16 input
  // gemm_M = N_weight, gemm_N = M_input
  int gemm_M = static_cast<int>(N_weight);
  int gemm_N = static_cast<int>(M_input);
  int gemm_K = static_cast<int>(K_dim);

  cudaStream_t stream = get_stream(a.device());
  run_tactic(static_cast<int>(tactic), gemm_M, gemm_N, gemm_K, stream,
             reinterpret_cast<const cutlass::float_e4m3_t*>(b_upper.data_ptr()),
             reinterpret_cast<const cutlass::float_e4m3_t*>(b_lower.data_ptr()),
             reinterpret_cast<const cutlass::half_t*>(a.data_ptr()),
             reinterpret_cast<cutlass::half_t*>(out.data_ptr()));
}

//-----------------------------------------------------------------------
// E5M2 RTN validation + dispatch logic
//-----------------------------------------------------------------------
void dual_weight_mm_sm90_e5m2_impl(TensorView a, TensorView b_upper, TensorView b_lower,
                                    TensorView out, TensorView workspace_buffer,
                                    int64_t tactic) {
  CHECK_CUDA(a);
  CHECK_CUDA(b_upper);
  CHECK_CUDA(b_lower);
  CHECK_CUDA(out);
  CHECK_INPUT_TYPE(a, dl_float16);
  CHECK_INPUT_TYPE(out, dl_float16);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b_upper);
  CHECK_DIM(2, b_lower);
  CHECK_DIM(2, out);
  CHECK_DEVICE(a, b_upper);
  CHECK_DEVICE(a, b_lower);
  CHECK_DEVICE(a, out);

  TVM_FFI_ICHECK_EQ(b_upper.stride(0), 1) << "b_upper must be column-major";
  TVM_FFI_ICHECK_EQ(b_lower.stride(0), 1) << "b_lower must be column-major";
  TVM_FFI_ICHECK_EQ(a.stride(1), 1) << "a must be row-major";
  TVM_FFI_ICHECK_EQ(out.stride(1), 1) << "out must be row-major";

  TVM_FFI_ICHECK_EQ(b_upper.size(0), b_lower.size(0)) << "b_upper and b_lower shape mismatch (dim 0).";
  TVM_FFI_ICHECK_EQ(b_upper.size(1), b_lower.size(1)) << "b_upper and b_lower shape mismatch (dim 1).";

  int64_t M_input = a.size(0);
  int64_t K_dim = a.size(1);
  int64_t N_weight = b_upper.size(1);

  TVM_FFI_ICHECK_EQ(b_upper.size(0), K_dim) << "K dimension mismatch.";
  TVM_FFI_ICHECK_EQ(out.size(0), M_input) << "out M dimension mismatch.";
  TVM_FFI_ICHECK_EQ(out.size(1), N_weight) << "out N dimension mismatch.";

  if (tactic == -1) tactic = 0;
  TVM_FFI_ICHECK(tactic >= 0 && tactic < kNumTacticsE5M2)
      << "tactic must be in [0, " << kNumTacticsE5M2 << "), got " << tactic;

  int gemm_M = static_cast<int>(N_weight);
  int gemm_N = static_cast<int>(M_input);
  int gemm_K = static_cast<int>(K_dim);

  cudaStream_t stream = get_stream(a.device());
  run_tactic_e5m2(static_cast<int>(tactic), gemm_M, gemm_N, gemm_K, stream,
                  reinterpret_cast<const cutlass::float_e4m3_t*>(b_upper.data_ptr()),
                  reinterpret_cast<const cutlass::float_e4m3_t*>(b_lower.data_ptr()),
                  reinterpret_cast<const cutlass::half_t*>(a.data_ptr()),
                  reinterpret_cast<cutlass::half_t*>(out.data_ptr()));
}

}  // namespace

//-----------------------------------------------------------------------
// TVM FFI exports
//-----------------------------------------------------------------------
namespace torch_ext {

void dual_weight_mm_sm90(TensorView a, TensorView b_upper, TensorView b_lower,
                         TensorView out, TensorView workspace_buffer,
                         int64_t tactic) {
  TVM_FFI_ICHECK(b_upper.dtype() == dl_float8_e4m3fn) << "b_upper must be float8_e4m3fn";
  TVM_FFI_ICHECK(b_lower.dtype() == dl_float8_e4m3fn) << "b_lower must be float8_e4m3fn";
  dual_weight_mm_sm90_impl(a, b_upper, b_lower, out, workspace_buffer, tactic);
}

int64_t dual_weight_mm_sm90_tactic_num() {
  return kNumTactics;
}

void dual_weight_mm_sm90_e5m2(TensorView a, TensorView b_upper, TensorView b_lower,
                               TensorView out, TensorView workspace_buffer,
                               int64_t tactic) {
  TVM_FFI_ICHECK(b_upper.dtype() == dl_float8_e5m2) << "b_upper must be float8_e5m2";
  TVM_FFI_ICHECK(b_lower.dtype() == dl_float8_e5m2) << "b_lower must be float8_e5m2";
  dual_weight_mm_sm90_e5m2_impl(a, b_upper, b_lower, out, workspace_buffer, tactic);
}

int64_t dual_weight_mm_sm90_e5m2_tactic_num() {
  return kNumTacticsE5M2;
}

void dual_weight_mm_sm90_e5m2_trunc(TensorView a, TensorView b_upper, TensorView b_lower,
                                     TensorView out, TensorView workspace_buffer,
                                     int64_t tactic) {
  TVM_FFI_ICHECK(b_upper.dtype() == dl_float8_e5m2) << "b_upper must be float8_e5m2";
  TVM_FFI_ICHECK(b_lower.dtype() == dl_float8_e5m2) << "b_lower must be float8_e5m2";
  // TODO: E5M2 trunc needs separate kernel instantiation with transform2_e5m2_trunc.
  // For now, falls through to E4M3 reconstruction — incorrect but placeholder.
  dual_weight_mm_sm90_impl(a, b_upper, b_lower, out, workspace_buffer, tactic);
}

int64_t dual_weight_mm_sm90_e5m2_trunc_tactic_num() {
  return kNumTactics;
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm90, torch_ext::dual_weight_mm_sm90);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm90_tactic_num,
                              torch_ext::dual_weight_mm_sm90_tactic_num);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm90_e5m2,
                              torch_ext::dual_weight_mm_sm90_e5m2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm90_e5m2_tactic_num,
                              torch_ext::dual_weight_mm_sm90_e5m2_tactic_num);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm90_e5m2_trunc,
                              torch_ext::dual_weight_mm_sm90_e5m2_trunc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm90_e5m2_trunc_tactic_num,
                              torch_ext::dual_weight_mm_sm90_e5m2_trunc_tactic_num);
