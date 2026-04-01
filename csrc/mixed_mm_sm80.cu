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

/*
 * Mixed-precision GEMM on SM80 (A100): FP8-weight × FP16-activation → FP16.
 *
 * B (weights) must be pre-shuffled to column-major [k, n] layout using
 * prepare_mixed_mm_weights() before calling these kernels.
 * A per-output-column FP16 scale is applied in Python after the GEMM.
 *
 * E4M3: CutlassDualWeightGemmRunner<e4m3, kTruncateE5M2=false, kFastE4M3=true>
 *       b_lower = b_upper (reconstruction ignores b_lower when kFastE4M3=true)
 * E5M2: CutlassDualWeightGemmRunner<e5m2, kTruncateE5M2=true, kFastE4M3=false>
 *       b_lower = zeros  (FP16 = FP8_byte << 8 when lower byte is 0)
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <vector>

#include "nv_internal/tensorrt_llm/kernels/cutlass_kernels/dual_weight_gemm/dual_weight_gemm.h"
#include "tvm_ffi_utils.h"

using tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
using tensorrt_llm::kernels::cutlass_kernels::CutlassDualWeightGemmRunner;

namespace torch_ext {

namespace {

// E4M3 runner: kFastE4M3=true uses FastNumericArrayConverter decoding.
using Fp8E4M3Runner = CutlassDualWeightGemmRunner<__nv_fp8_e4m3, /*kTruncateE5M2=*/false,
                                                  /*kFastE4M3=*/true>;
// E5M2 runner: kTruncateE5M2=true, b_lower=0 gives FP8_byte << 8 decoding.
using Fp8E5M2Runner = CutlassDualWeightGemmRunner<__nv_fp8_e5m2, /*kTruncateE5M2=*/true,
                                                  /*kFastE4M3=*/false>;

template <typename Runner>
CutlassGemmConfig getFp8WConfig(int64_t tactic) {
  static Runner runner;
  static std::vector<CutlassGemmConfig> configs = runner.getConfigs();
  TVM_FFI_ICHECK(tactic >= 0 && tactic < static_cast<int64_t>(configs.size()))
      << "tactic must be between 0 and " << configs.size() - 1;
  return configs[tactic];
}

// Common GEMM launch helper.
// b: [k, n] column-major (stride(0)=1, stride(1)=k), pre-shuffled.
// b_lower: pointer to lower weight bytes (may equal b_upper for kFastE4M3).
template <typename Runner>
void run_mixed_mm(Runner& runner, TensorView a, TensorView b,
                       void const* b_lower_ptr, TensorView out,
                       TensorView workspace_buffer, int64_t tactic) {
  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b.size(1);

  if (tactic == -1) tactic = 0;

  auto config = getFp8WConfig<Runner>(tactic);
  int64_t required_ws = static_cast<int64_t>(runner.getWorkspaceSize(m, n, k));
  int64_t provided_ws = workspace_buffer.numel() * get_element_size(workspace_buffer);

  int64_t lda = a.stride(0);     // k (a is [m,k] row-major)
  int64_t ldb = b.stride(1);     // k (b is [k,n] column-major, stride(1)=k)
  int64_t ldc = out.stride(0);   // n (out is [m,n] row-major)

  cudaStream_t stream = get_stream(a.device());

  auto run_kernel = [&](void* workspace) {
    runner.gemm(a.data_ptr(), b.data_ptr(), b_lower_ptr, out.data_ptr(),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                lda, ldb, ldc, config,
                static_cast<char*>(workspace), static_cast<size_t>(required_ws), stream);
  };

  if (provided_ws < required_ws) {
    Tensor new_ws = alloc_tensor({required_ws}, DLDataType{kDLInt, 8, 1}, a.device());
    run_kernel(new_ws.data_ptr());
  } else {
    run_kernel(workspace_buffer.data_ptr());
  }
}

void check_mixed_mm_inputs(TensorView a, TensorView b, TensorView out,
                                TensorView workspace_buffer,
                                DLDataType expected_weight_dtype) {
  CHECK_CUDA(a);
  CHECK_CUDA(b);
  CHECK_CUDA(out);
  CHECK_CUDA(workspace_buffer);
  CHECK_INPUT_TYPE(a, dl_float16);
  TVM_FFI_ICHECK_EQ(b.dtype(), expected_weight_dtype) << "b has unexpected dtype.";
  CHECK_INPUT_TYPE(out, dl_float16);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(2, out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, out);

  // a: [m, k] row-major
  TVM_FFI_ICHECK_EQ(a.stride(1), 1) << "a must be row-major.";
  // b: [k, n] column-major (pre-shuffled), stride(0)=1
  TVM_FFI_ICHECK_EQ(b.stride(0), 1) << "b must be column-major ([k, n]) with stride(0)=1.";
  // out: [m, n] row-major
  TVM_FFI_ICHECK_EQ(out.stride(1), 1) << "out must be row-major.";

  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b.size(1);

  TVM_FFI_ICHECK_EQ(b.size(0), k) << "a and b shapes cannot be multiplied (k mismatch).";
  TVM_FFI_ICHECK_EQ(out.size(0), m) << "out has incorrect M dimension.";
  TVM_FFI_ICHECK_EQ(out.size(1), n) << "out has incorrect N dimension.";
}

}  // namespace

// E4M3: b_lower = b_upper (kFastE4M3 reconstruction ignores b_lower)
void mixed_mm_e4m3_sm80(TensorView a, TensorView b, TensorView out,
                        TensorView workspace_buffer, int64_t tactic) {
  check_mixed_mm_inputs(a, b, out, workspace_buffer, dl_float8_e4m3fn);
  static Fp8E4M3Runner runner;
  run_mixed_mm(runner, a, b, b.data_ptr(), out, workspace_buffer, tactic);
}

int64_t mixed_mm_e4m3_tactic_num() {
  static Fp8E4M3Runner runner;
  return static_cast<int64_t>(runner.getConfigs().size());
}

// E5M2: allocate zero b_lower (kTruncateE5M2 needs lower byte = 0)
void mixed_mm_e5m2_sm80(TensorView a, TensorView b, TensorView out,
                        TensorView workspace_buffer, int64_t tactic) {
  check_mixed_mm_inputs(a, b, out, workspace_buffer, dl_float8_e5m2);

  int64_t k = b.size(0);
  int64_t n = b.size(1);
  Tensor b_lower_zero = alloc_tensor({k * n}, DLDataType{kDLUInt, 8, 1}, a.device());
  cudaMemsetAsync(b_lower_zero.data_ptr(), 0, static_cast<size_t>(k * n),
                  get_stream(a.device()));

  static Fp8E5M2Runner runner;
  run_mixed_mm(runner, a, b, b_lower_zero.data_ptr(), out, workspace_buffer, tactic);
}

int64_t mixed_mm_e5m2_tactic_num() {
  static Fp8E5M2Runner runner;
  return static_cast<int64_t>(runner.getConfigs().size());
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_mm_e4m3_sm80, torch_ext::mixed_mm_e4m3_sm80);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_mm_e4m3_tactic_num,
                               torch_ext::mixed_mm_e4m3_tactic_num);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_mm_e5m2_sm80, torch_ext::mixed_mm_e5m2_sm80);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_mm_e5m2_tactic_num,
                               torch_ext::mixed_mm_e5m2_tactic_num);
