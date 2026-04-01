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

#include <vector>

#include "nv_internal/tensorrt_llm/kernels/cutlass_kernels/dual_weight_gemm/dual_weight_gemm.h"
#include "tvm_ffi_utils.h"

using tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
using tensorrt_llm::kernels::cutlass_kernels::CutlassDualWeightGemmRunner;

namespace torch_ext {

namespace {

template <typename WeightType, bool kTruncateE5M2 = false>
CutlassGemmConfig getDualWeightGemmConfig(int64_t tactic) {
  static CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2> runner;
  static std::vector<CutlassGemmConfig> configs = runner.getConfigs();
  TVM_FFI_ICHECK(tactic >= 0 && tactic < static_cast<int64_t>(configs.size()))
      << "tactic must be between 0 and " << configs.size();
  return configs[tactic];
}

template <typename WeightType, bool kTruncateE5M2 = false>
void dual_weight_mm_impl(TensorView a, TensorView b_upper, TensorView b_lower,
                         TensorView out, TensorView workspace_buffer,
                         int64_t tactic, DLDataType expected_weight_dtype) {
  CHECK_CUDA(a);
  CHECK_CUDA(b_upper);
  CHECK_CUDA(b_lower);
  CHECK_CUDA(out);
  CHECK_CUDA(workspace_buffer);
  CHECK_INPUT_TYPE(a, dl_float16);
  TVM_FFI_ICHECK_EQ(b_upper.dtype(), expected_weight_dtype) << "b_upper has unexpected dtype.";
  TVM_FFI_ICHECK_EQ(b_lower.dtype(), expected_weight_dtype) << "b_lower has unexpected dtype.";
  CHECK_INPUT_TYPE(out, dl_float16);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b_upper);
  CHECK_DIM(2, b_lower);
  CHECK_DIM(2, out);
  CHECK_DEVICE(a, b_upper);
  CHECK_DEVICE(a, b_lower);
  CHECK_DEVICE(a, out);

  TVM_FFI_ICHECK_EQ(a.stride(1), 1)
      << "a must be row-major with contiguous last dimension.";
  TVM_FFI_ICHECK_EQ(out.stride(1), 1)
      << "out must be row-major with contiguous last dimension.";
  TVM_FFI_ICHECK_EQ(b_upper.stride(0), 1)
      << "b_upper must be column-major with contiguous leading dimension.";
  TVM_FFI_ICHECK_EQ(b_lower.stride(0), 1)
      << "b_lower must be column-major with contiguous leading dimension.";

  TVM_FFI_ICHECK_EQ(b_upper.size(0), b_lower.size(0))
      << "b_upper and b_lower must have identical shapes.";
  TVM_FFI_ICHECK_EQ(b_upper.size(1), b_lower.size(1))
      << "b_upper and b_lower must have identical shapes.";

  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b_upper.size(1);

  TVM_FFI_ICHECK_EQ(b_upper.size(0), k)
      << "a and b_upper shapes cannot be multiplied.";
  TVM_FFI_ICHECK_EQ(b_lower.size(0), k)
      << "a and b_lower shapes cannot be multiplied.";
  TVM_FFI_ICHECK_EQ(out.size(0), m) << "out has incorrect M dimension.";
  TVM_FFI_ICHECK_EQ(out.size(1), n) << "out has incorrect N dimension.";

  if (tactic == -1) {
    tactic = 0;
  }

  auto config = getDualWeightGemmConfig<WeightType, kTruncateE5M2>(tactic);
  static CutlassDualWeightGemmRunner<WeightType, kTruncateE5M2> runner;
  int64_t required_workspace_size = runner.getWorkspaceSize(m, n, k);
  int64_t provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);

  auto run_kernel = [&](void* workspace) {
    runner.gemm(a.data_ptr(), b_upper.data_ptr(), b_lower.data_ptr(), out.data_ptr(),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                a.stride(0), b_upper.stride(1), out.stride(0), config,
                static_cast<char*>(workspace), required_workspace_size,
                get_stream(a.device()));
  };

  if (provided_workspace_size < required_workspace_size) {
    Tensor new_workspace =
        alloc_tensor({required_workspace_size}, DLDataType{kDLInt, 8, 1}, a.device());
    run_kernel(new_workspace.data_ptr());
  } else {
    run_kernel(workspace_buffer.data_ptr());
  }
}

}  // namespace

void dual_weight_mm_sm80(TensorView a, TensorView b_upper, TensorView b_lower,
                         TensorView out, TensorView workspace_buffer,
                         int64_t tactic) {
  dual_weight_mm_impl<__nv_fp8_e4m3>(a, b_upper, b_lower, out, workspace_buffer,
                                     tactic, dl_float8_e4m3fn);
}

int64_t dual_weight_mm_tactic_num() {
  static CutlassDualWeightGemmRunner<> runner;
  return static_cast<int64_t>(runner.getConfigs().size());
}

void dual_weight_mm_sm80_e5m2(TensorView a, TensorView b_upper, TensorView b_lower,
                               TensorView out, TensorView workspace_buffer,
                               int64_t tactic) {
  dual_weight_mm_impl<__nv_fp8_e5m2>(a, b_upper, b_lower, out, workspace_buffer,
                                     tactic, dl_float8_e5m2);
}

int64_t dual_weight_mm_e5m2_tactic_num() {
  static CutlassDualWeightGemmRunner<__nv_fp8_e5m2> runner;
  return static_cast<int64_t>(runner.getConfigs().size());
}

void dual_weight_mm_sm80_e5m2_trunc(TensorView a, TensorView b_upper, TensorView b_lower,
                                     TensorView out, TensorView workspace_buffer,
                                     int64_t tactic) {
  dual_weight_mm_impl<__nv_fp8_e5m2, /*kTruncateE5M2=*/true>(
      a, b_upper, b_lower, out, workspace_buffer, tactic, dl_float8_e5m2);
}

int64_t dual_weight_mm_e5m2_trunc_tactic_num() {
  static CutlassDualWeightGemmRunner<__nv_fp8_e5m2, /*kTruncateE5M2=*/true> runner;
  return static_cast<int64_t>(runner.getConfigs().size());
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm80, torch_ext::dual_weight_mm_sm80);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_tactic_num,
                              torch_ext::dual_weight_mm_tactic_num);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm80_e5m2,
                              torch_ext::dual_weight_mm_sm80_e5m2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_e5m2_tactic_num,
                              torch_ext::dual_weight_mm_e5m2_tactic_num);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_sm80_e5m2_trunc,
                              torch_ext::dual_weight_mm_sm80_e5m2_trunc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dual_weight_mm_e5m2_trunc_tactic_num,
                              torch_ext::dual_weight_mm_e5m2_trunc_tactic_num);
