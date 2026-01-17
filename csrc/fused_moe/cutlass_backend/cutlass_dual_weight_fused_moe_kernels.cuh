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

#include <cuda.h>

#include "dual_weight_moe_kernels.h"
#include "moe_kernels.h"
#include "moe_util_kernels.h"
#include "tensorrt_llm/common/workspace.h"
#include "cutlass_fused_moe_kernels.cuh"

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels::cutlass_kernels {

template <typename T, typename WeightType, typename OutputType, typename InputType,
          typename BackBoneType, typename Enable>
DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::DualWeightMoeFCRunner() {}

template <typename T, typename WeightType, typename OutputType, typename InputType,
          typename BackBoneType, typename Enable>
std::map<std::string, std::pair<size_t, size_t>>
DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType,
                      Enable>::getWorkspaceDeviceBufferSizes(int64_t const num_rows,
                                                             int64_t const hidden_size,
                                                             int64_t const inter_size,
                                                             int const num_experts_per_node,
                                                             int const experts_per_token,
                                                             ActivationType activation_type) {
  size_t num_moe_inputs = experts_per_token * num_rows;
  size_t const permuted_elems = num_moe_inputs * hidden_size;
  size_t const interbuf_elems = num_moe_inputs * inter_size;
  size_t glu_inter_elems = 0;
  bool is_gated_activation = isGatedActivation(activation_type);
  if (is_gated_activation) {
    glu_inter_elems = interbuf_elems * 2;
  } else if (mayHaveDifferentGEMMOutputType()) {
    // In this case we are using activation quantization, and some intermediate buffers will be
    // unquantized We need to have separate memory for these as we can no longer alias the output
    // buffer for reuse
    glu_inter_elems = interbuf_elems;
  }

  size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);
  constexpr float dtype_size = sizeof(T);

  size_t const permuted_row_to_unpermuted_row_size = num_moe_inputs * sizeof(int);
  size_t const permuted_token_selected_experts_size = num_moe_inputs * sizeof(int);

  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_rows, num_experts_per_node);
  int64_t const num_blocks_per_seq = tensorrt_llm::common::ceilDiv(num_rows, num_tokens_per_block);
  size_t const blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size = num_experts_per_node * num_rows * sizeof(int);

  size_t const permuted_data_size = permuted_elems * dtype_size;
  size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
  size_t const permuted_token_final_scales_size =
      mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
  size_t const glu_inter_size =
      glu_inter_elems * gemm_output_dtype;  // May be an intermediate type for quantization
  size_t const fc1_result_size =
      interbuf_elems * dtype_size;  // Activation quantizes so back to dtype_size
  size_t const fc2_result_size = num_moe_inputs * hidden_size *
                                 gemm_output_dtype;  // May be an intermediate type for quantization

  size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);

  // We do some overlapping of the large workspace buffers. Although we could overlap some of the
  // other buffers, they are small enough (i.e no factor of hidden size) they will only be a couple
  // MiB at most, so we don't bother in the case of fused activation we overlap permuted_data and
  // fc2_result in the case of unfused activation we overlap permuted_data and fc1_result we need to
  // calculate the max possible size, so use the max of all three
  size_t overlapped_gemm1_gemm2_inputs_size = std::max(permuted_data_size, fc2_result_size);
  // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_inputs_size =
        std::max(overlapped_gemm1_gemm2_inputs_size, fc1_result_size);
  }

  // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
  size_t overlapped_gemm1_gemm2_outputs_size = fc1_result_size;
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_outputs_size =
        std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs_size);
  }

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                        \
  do {                                                                              \
    auto aligned_size =                                                             \
        tensorrt_llm::common::alignSize(size, tensorrt_llm::common::kCudaMemAlign); \
    out_map[#name] = std::pair{aligned_size, map_offset};                           \
    map_offset += aligned_size;                                                     \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(permuted_row_to_unpermuted_row);
  ADD(permuted_token_selected_experts);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(expert_first_token_offset);
  ADD(permuted_token_final_scales);
  ADD(overlapped_gemm1_gemm2_inputs);
  ADD(overlapped_gemm1_gemm2_outputs);
  ADD(gemm_workspace);

  return out_map;

#undef ADD_NAME
#undef ADD
}

template <typename T, typename WeightType, typename OutputType, typename InputType,
          typename BackBoneType, typename Enable>
size_t
DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::getWorkspaceSize(
    int64_t const num_rows, int64_t const hidden_size, int64_t const fc1_output_size,
    int const num_experts, int const experts_per_token, ActivationType activation_type,
    MOEParallelismConfig parallelism_config) {
  int const ep_size = parallelism_config.ep_size;
  TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0,
                       "Number of experts must be a multiple of ep size");
  auto sizes_map =
      getWorkspaceDeviceBufferSizes(num_rows, hidden_size, fc1_output_size, num_experts / ep_size,
                                    experts_per_token, activation_type);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(),
                 [](auto& v) { return v.second.first; });
  size_t size = tensorrt_llm::common::calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  TLLM_LOG_TRACE("Mixture Of Experts Plugin requires workspace of %2f MiB", size / 1024.f / 1024.f);
  return size;
}

template <typename T, typename WeightType, typename OutputType, typename InputType,
          typename BackBoneType, typename Enable>
void DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType,
                           Enable>::configureWsPtrs(char* ws_ptr, int64_t const num_rows,
                                                    int64_t const hidden_size,
                                                    int64_t const inter_size,
                                                    int const num_experts_per_node,
                                                    int const experts_per_token,
                                                    ActivationType activation_type,
                                                    MOEParallelismConfig parallelism_config) {
  auto workspaces = getWorkspaceDeviceBufferSizes(
      num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token, activation_type);

  auto getWsPtr = [&](auto type, std::string const& name) {
    return workspaces.at(name).first
               ? reinterpret_cast<decltype(type)*>(ws_ptr + workspaces.at(name).second)
               : nullptr;
  };
  permuted_row_to_unpermuted_row_ = getWsPtr(int{}, "permuted_row_to_unpermuted_row");
  permuted_token_selected_experts_ = getWsPtr(int{}, "permuted_token_selected_experts");
  blocked_expert_counts_ = getWsPtr(int{}, "blocked_expert_counts");
  blocked_expert_counts_cumsum_ = getWsPtr(int{}, "blocked_expert_counts_cumsum");
  blocked_row_to_unpermuted_row_ = getWsPtr(int{}, "blocked_row_to_unpermuted_row");
  expert_first_token_offset_ = getWsPtr(int64_t{}, "expert_first_token_offset");
  permuted_token_final_scales_ = getWsPtr(float{}, "permuted_token_final_scales");

  bool const is_gated_activation = isGatedActivation(activation_type);
  bool const gemm1_using_fused_moe = moe_gemm_runner_.isFusedGatedActivation(
      *gemm1_config_, activation_type, inter_size, hidden_size);
  // We always use fused path if we can
  bool const has_glu_inter_result = !gemm1_using_fused_moe && is_gated_activation;

  // Always same value, but overlapped with either fc1_result_ or fc2_result_
  permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  // Always same value, ignored if not needed
  glu_inter_result_ =
      has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs") : nullptr;

  // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which
  // is overlapped NOTE: It is important to get the overlapped pointers correct as the wrong order
  // will cause the buffer to be used as an input and output for the same gemm, which will cause
  // corruption
  fc1_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
  fc2_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          class Enable>
void DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::gemm1(
    DualWeightMoeGemmRunner<T, WeightType, T, ScaleBiasType>& gemm_runner, T const* const input,
    T* const output, void* const intermediate_result,
    int64_t const* const expert_first_token_offset,
    WeightType const* const fc1_upper_expert_weights,
    WeightType const* const fc1_lower_expert_weights, ScaleBiasType const* const fc1_expert_biases,
    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, ActivationParams fc1_activation_type, bool bias_is_broadcast,
    cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
    int* active_expert_global_ids) {
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);
  bool const use_ampere_activation_fusion = gemm_runner.isFusedGatedActivation(
      config, fc1_activation_type.activation_type, inter_size, hidden_size);
  size_t const fc1_out_size =
      ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  if (!is_gated_activation) {
    TLLM_CHECK(!use_ampere_activation_fusion);
    DualWeightGroupedGemmInput<T, WeightType, T, T> universal_input;
    universal_input.A = input;
    universal_input.total_tokens_including_expert = total_tokens_including_expert;
    universal_input.B = fc1_upper_expert_weights;
    universal_input.B_lower = fc1_lower_expert_weights;
    universal_input.scales = nullptr;
    universal_input.zeros = nullptr;
    universal_input.biases = fc1_expert_biases;
    universal_input.C = reinterpret_cast<T*>(output);
    universal_input.alpha_scales = nullptr;
    universal_input.occupancy = nullptr;
    universal_input.activation_type = fc1_activation_type.activation_type;
    universal_input.num_rows = expanded_num_rows;
    universal_input.n = int64_t(fc1_out_size);
    universal_input.k = hidden_size;
    universal_input.num_experts = num_experts_per_node;
    universal_input.bias_is_broadcast = bias_is_broadcast;
    universal_input.use_fused_moe = false;
    universal_input.stream = stream;
    universal_input.gemm_config = config;

    gemm_runner.moeGemmBiasAct(universal_input);

    sync_check_cuda_error(stream);
  } else {
    TLLM_CHECK(is_gated_activation);
    TLLM_CHECK_WITH_INFO(!use_ampere_activation_fusion || input != output,
                         "Input and output buffers are overlapping");

    // Run the GEMM with activation function overridden with `Identity`, we do the activation
    // separately
    DualWeightGroupedGemmInput<T, WeightType, T, T> universal_input;
    universal_input.A = input;
    universal_input.total_tokens_including_expert = total_tokens_including_expert;
    universal_input.B = fc1_upper_expert_weights;
    universal_input.B_lower = fc1_lower_expert_weights;
    universal_input.scales = nullptr;
    universal_input.zeros = nullptr;
    universal_input.biases = fc1_expert_biases;
    universal_input.C = static_cast<T*>(use_ampere_activation_fusion ? output : intermediate_result);
    universal_input.alpha_scales = nullptr;
    universal_input.occupancy = nullptr;
    universal_input.activation_type = use_ampere_activation_fusion ? fc1_activation_type.activation_type : ActivationType::Identity;
    universal_input.num_rows = expanded_num_rows;
    universal_input.n = int64_t(fc1_out_size);
    universal_input.k = hidden_size;
    universal_input.num_experts = num_experts_per_node;
    universal_input.bias_is_broadcast = bias_is_broadcast;
    universal_input.use_fused_moe = use_ampere_activation_fusion;
    universal_input.stream = stream;
    universal_input.gemm_config = config;

    gemm_runner.moeGemmBiasAct(universal_input);

    sync_check_cuda_error(stream);

    if (!use_ampere_activation_fusion) {
      using GatedActOutputType = T;
      doGatedActivation<GatedActOutputType, UnfusedGemmOutputType>(
          reinterpret_cast<GatedActOutputType*>(output),
          static_cast<UnfusedGemmOutputType const*>(intermediate_result), expert_first_token_offset,
          inter_size, expanded_num_rows, num_experts_per_node, fc1_activation_type, stream);

      sync_check_cuda_error(stream);
    }
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          class Enable>
void DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::gemm2(
    DualWeightMoeGemmRunner<T, WeightType, T, ScaleBiasType>& gemm_runner, T const* const input,
    void* const gemm_output, OutputType* const final_output,
    int64_t const* const expert_first_token_offset,
    WeightType const* const fc2_expert_upper_weights,
    WeightType const* const fc2_expert_lower_weights, ScaleBiasType const* const fc2_expert_biases,
    float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
    int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
    int const* const token_selected_experts, int64_t const* const num_valid_tokens_ptr,
    int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size,
    int64_t const inter_size, int const num_experts_per_node, int64_t const experts_per_token,
    cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
    cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
    int* active_expert_global_ids) {
  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  // FC2 GEMM: intermediate -> output, no activation
  DualWeightGroupedGemmInput<T, WeightType, OutputType, OutputType> universal_input;
  universal_input.A = input;
  universal_input.total_tokens_including_expert = total_tokens_including_expert;
  universal_input.B = fc2_expert_upper_weights;
  universal_input.B_lower = fc2_expert_lower_weights;
  universal_input.scales = nullptr;
  universal_input.zeros = nullptr;
  universal_input.biases = nullptr;  // FC2 biases are applied in finalize
  universal_input.C = static_cast<OutputType*>(gemm_output);
  universal_input.alpha_scales = nullptr;
  universal_input.occupancy = nullptr;
  universal_input.activation_type = ActivationType::Identity;
  universal_input.num_rows = expanded_num_rows;
  universal_input.n = hidden_size;
  universal_input.k = inter_size;
  universal_input.num_experts = num_experts_per_node;
  universal_input.bias_is_broadcast = false;
  universal_input.use_fused_moe = false;
  universal_input.stream = stream;
  universal_input.gemm_config = config;

  gemm_runner.moeGemmBiasAct(universal_input);

  sync_check_cuda_error(stream);

  // Finalize: unpermute and scale outputs
  finalizeMoeRoutingKernelLauncher<OutputType, T>(
      static_cast<T const*>(gemm_output), final_output, fc2_expert_biases,
      token_topk_unpermuted_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
      token_selected_experts, expert_first_token_offset, num_rows, hidden_size, hidden_size,
      experts_per_token, num_experts_per_node, parallelism_config, enable_alltoall,
      /*enable_pdl*/ false, stream);

  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          class Enable>
void DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::runMoe(
    void const* input_activations_void, int const* token_selected_experts,
    float const* token_final_scales, void const* fc1_upper_expert_weights_void,
    void const* fc1_lower_expert_weights_void, void const* fc1_expert_biases_void,
    ActivationParams fc1_activation_type, void const* fc2_upper_expert_weights_void,
    void const* fc2_lower_expert_weights_void, void const* fc2_expert_biases_void, int64_t const num_rows,
    int64_t const hidden_size, int64_t const inter_size, int const full_num_experts,
    int const experts_per_token, char* workspace_ptr, void* final_output_void,
    int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config,
    bool const enable_alltoall, cudaStream_t stream) {
  auto const* input_activations = static_cast<InputType const*>(input_activations_void);
  auto const* fc1_upper_expert_weights =
      static_cast<WeightType const*>(fc1_upper_expert_weights_void);
  auto const* fc1_lower_expert_weights =
      static_cast<WeightType const*>(fc1_lower_expert_weights_void);
  auto const* fc1_expert_biases = static_cast<ScaleBiasType const*>(fc1_expert_biases_void);
  auto const* fc2_upper_expert_weights =
      static_cast<WeightType const*>(fc2_upper_expert_weights_void);
  auto const* fc2_lower_expert_weights =
      static_cast<WeightType const*>(fc2_lower_expert_weights_void);
  auto const* fc2_expert_biases = static_cast<ScaleBiasType const*>(fc2_expert_biases_void);
  auto* final_output = static_cast<OutputType*>(final_output_void);
  float const* token_topk_unpermuted_scales = token_final_scales;

  TLLM_CHECK(input_activations);
  TLLM_CHECK(token_selected_experts);
  TLLM_CHECK(fc1_upper_expert_weights);
  TLLM_CHECK(fc1_lower_expert_weights);
  TLLM_CHECK(fc2_upper_expert_weights);
  TLLM_CHECK(fc2_lower_expert_weights);
  TLLM_CHECK(workspace_ptr);
  // TLLM_CHECK(token_topk_unpermuted_scales);
  TLLM_CHECK(unpermuted_row_to_permuted_row);
  TLLM_CHECK(full_num_experts % parallelism_config.ep_size == 0);
  TLLM_CHECK(full_num_experts % parallelism_config.cluster_size == 0);

  // Require at least 128 bits of alignment for MOE GEMM
  TLLM_CHECK_WITH_INFO(
      hidden_size % (128 / sizeof_bits<WeightType>::value) == 0,
      "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d",
      (int)hidden_size, (int)(128 / sizeof_bits<WeightType>::value));
  TLLM_CHECK_WITH_INFO(
      inter_size % (128 / sizeof_bits<WeightType>::value) == 0,
      "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d",
      (int)inter_size, (int)(128 / sizeof_bits<WeightType>::value));

  // These values must fit into an int for building the source maps
  TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
  TLLM_CHECK_WITH_INFO(num_rows * full_num_experts <= std::numeric_limits<int>::max(),
                       "Number of rows * num_experts is too large");
  TLLM_CHECK_WITH_INFO(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
                       "experts_per_token * num_experts is too large");

  TLLM_CHECK_WITH_INFO(gemm1_config_, "MOE GEMM1 Config is not set");
  TLLM_CHECK_WITH_INFO(gemm2_config_, "MOE GEMM2 Config is not set");

  int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;

  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node, 
                  experts_per_token, fc1_activation_type.activation_type, parallelism_config);

  int start_expert = parallelism_config.ep_rank * num_experts_per_node;
  int end_expert = start_expert + num_experts_per_node;

  bool const needs_num_valid = parallelism_config.ep_size > 1;
  int64_t const* num_valid_tokens_ptr =
      needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

  auto expanded_num_rows = num_rows * experts_per_token;

  bool fused_prologue_result = false;
  fused_prologue_result = fusedBuildExpertMapsSortFirstToken(
      token_selected_experts, permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row,
      expert_first_token_offset_, num_rows, num_experts_per_node, experts_per_token,
      start_expert, end_expert,  /*enable_pdl*/ false, stream);

  if (!fused_prologue_result) {
    TLLM_LOG_TRACE("Falling back to unfused prologue");
    threeStepBuildExpertMapsSortFirstToken(
        token_selected_experts, permuted_token_selected_experts_, permuted_row_to_unpermuted_row_,
        unpermuted_row_to_permuted_row, expert_first_token_offset_, blocked_expert_counts_,
        blocked_expert_counts_cumsum_, blocked_row_to_unpermuted_row_, num_rows,
        num_experts_per_node, experts_per_token, start_expert, /*enable_pdl*/ false, stream);
  }

  sync_check_cuda_error(stream);

  bool is_gated_activation = isGatedActivation(fc1_activation_type);
  T* gemm1_input_expand = reinterpret_cast<T*>(permuted_data_);
  QuantParams empty_quant_params;  // Default-initialized, no quantization for dual-weight
  expandInputRowsKernelLauncher(reinterpret_cast<T const*>(input_activations), gemm1_input_expand,
                                token_final_scales, permuted_token_final_scales_,
                                permuted_row_to_unpermuted_row_, num_rows, hidden_size,
                                experts_per_token, num_experts_per_node, empty_quant_params,
                                /*use_per_expert_act_scale*/ false, expert_first_token_offset_,
                                /*fc1_act_sf_flat*/ nullptr, /*input_sf*/ nullptr,
                                /*swizzled_input_sf*/ false,
                                /*prequant_scales*/ nullptr, /*enable_pdl*/ false, stream);
  auto const* gemm1_input = gemm1_input_expand;

  sync_check_cuda_error(stream);

  Self::gemm1(
      moe_gemm_runner_, gemm1_input, fc1_result_, glu_inter_result_,
      expert_first_token_offset_, fc1_upper_expert_weights, fc1_lower_expert_weights,
      fc1_expert_biases, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size, inter_size,
      num_experts_per_node, fc1_activation_type,
      /*bias_is_broadcast*/ true, stream, *gemm1_config_, /*num_active_experts_per*/ nullptr,
      /*active_expert_global_ids*/ nullptr);
  sync_check_cuda_error(stream);

  T* gemm2_input = reinterpret_cast<T*>(fc1_result_);
  Self::gemm2(
      moe_gemm_runner_, gemm2_input, fc2_result_, final_output, expert_first_token_offset_,
      fc2_upper_expert_weights, fc2_lower_expert_weights, fc2_expert_biases,
      token_topk_unpermuted_scales, permuted_token_final_scales_, unpermuted_row_to_permuted_row,
      permuted_row_to_unpermuted_row_, token_selected_experts, num_valid_tokens_ptr, num_rows,
      expanded_num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token, stream,
      parallelism_config, enable_alltoall, *gemm2_config_,
      /*num_active_experts_per*/ nullptr,
      /*active_expert_global_ids*/ nullptr);
  sync_check_cuda_error(stream);
}

std::map<std::string, std::pair<size_t, size_t>> DualWeightGemmProfilerBackend::getProfilerWorkspaces(
    int maxM) {
  size_t k = mK;
  size_t num_expanded_tokens = maxM * k;

  TLLM_CHECK(mDType != nvinfer1::DataType::kINT4);
  float dtype_bytes = static_cast<float>(getDTypeSize(mDType));
  float weight_bytes = static_cast<float>(getDTypeSize(mWType));
  size_t output_bytes = getDTypeSize(mOType);
  size_t gemm_output_bytes = output_bytes;

  size_t hidden_size = mExpertHiddenSize;
  size_t inter_size = mExpertInterSize;  // Already divided by TP
  size_t num_experts_per_node = mNumExpertsPerNode;

  size_t fc1_out_size = inter_size;
  if (isGatedActivation(mActivationType)) {
    fc1_out_size = inter_size * 2;
  }

  // TODO Needs updated when gather/finalize fusion is integrated
  size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
  size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

  size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
  size_t output_size2 = hidden_size * output_bytes;

  size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
  size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

  // This may allocate a pointer when not required. That's fine it will be ignored at the cost of
  // some memory
  size_t intermediate_size1 =
      fc1_out_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes
  size_t intermediate_size2 =
      hidden_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes

  size_t intermediate_size =
      mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

  size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
  size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
  size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

  // For dual-weight, we need separate upper and lower weight workspaces
  size_t upper_weights_size =
      mNeedWeights ? (mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2) : 0;
  size_t lower_weights_size =
      mNeedWeights ? (mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2) : 0;
  size_t bias_size = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

  size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);

  // Routing info
  size_t expert_first_token_offset_size =
      (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
  size_t map_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t unpermuted_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t permuted_size = num_expanded_tokens * sizeof(int);
  size_t token_topk_unpermuted_scales_size = num_expanded_tokens * sizeof(float);

  int64_t const num_tokens_per_block = computeNumTokensPerBlock(maxM, num_experts_per_node);
  int64_t const num_blocks_per_seq = tensorrt_llm::common::ceilDiv(maxM, num_tokens_per_block);
  size_t const blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size = num_experts_per_node * maxM * sizeof(int);

  // The follow buffers are used in min_latency_mode
  size_t num_active_experts_per_node_size = 0;
  size_t active_expert_global_ids_size = 0;

  bool is_swiglu_bias =
      mActivationType == ActivationType::SwigluBias && mGemmToProfile == GemmToProfile::GEMM_1;
  size_t swiglu_alpha_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;
  size_t swiglu_beta_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;
  size_t swiglu_limit_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                              \
  do {                                                    \
    auto aligned_size = alignSize(size, kCudaMemAlign);   \
    out_map[#name] = std::pair{aligned_size, map_offset}; \
    map_offset += aligned_size;                           \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(expert_first_token_offset);
  ADD_NAME(unpermuted_row_to_permuted_row, map_size);
  ADD_NAME(permuted_row_to_unpermuted_row, map_size);
  ADD_NAME(token_selected_experts, unpermuted_size);
  ADD_NAME(permuted_token_selected_experts, permuted_size);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(token_topk_unpermuted_scales);
  ADD(num_active_experts_per_node);
  ADD(active_expert_global_ids);
  ADD(input);
  ADD(output);
  ADD(intermediate);
  ADD(upper_weights);
  ADD(lower_weights);
  ADD(bias);
  ADD(gemm_workspace);
  ADD(swiglu_alpha);
  ADD(swiglu_beta);
  ADD(swiglu_limit);

#undef ADD_NAME
#undef ADD

  return out_map;
}

void DualWeightGemmProfilerBackend::prepareRouting(int num_tokens, char* workspace_ptr_char,
                                         cudaStream_t stream) {
  auto workspaces = getProfilerWorkspaces(num_tokens);
#define GET_WS_PTR_BASE(type, name)                                                   \
  auto* name##_base =                                                                 \
      (workspaces.at(#name).first                                                     \
           ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
           : nullptr)
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR_BASE(int64_t*, expert_first_token_offset);
  GET_WS_PTR_BASE(int*, unpermuted_row_to_permuted_row);
  GET_WS_PTR_BASE(int*, permuted_row_to_unpermuted_row);
  GET_WS_PTR_BASE(int*, token_selected_experts);
  GET_WS_PTR(int*, permuted_token_selected_experts);
  GET_WS_PTR(int*, blocked_expert_counts);
  GET_WS_PTR(int*, blocked_expert_counts_cumsum);
  GET_WS_PTR(int*, blocked_row_to_unpermuted_row);
  GET_WS_PTR(int*, num_active_experts_per_node);
  GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR_BASE
#undef GET_WS_PTR

  int64_t const num_expanded_tokens = num_tokens * mK;
  int const start_expert_id = mNumExpertsPerNode * mParallelismConfig.ep_rank;

  uint32_t num_threads = 256;
  dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
  prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(token_selected_experts_base,
                                                                  num_tokens, mK, mNumExperts);
  sync_check_cuda_error(stream);

  for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
    int64_t* expert_first_token_offset =
        expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
    int* unpermuted_row_to_permuted_row =
        unpermuted_row_to_permuted_row_base + i * num_expanded_tokens;
    int* permuted_row_to_unpermuted_row =
        permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;
    int* token_selected_experts = token_selected_experts_base + i * num_expanded_tokens;

    threeStepBuildExpertMapsSortFirstToken(
        token_selected_experts, permuted_token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, expert_first_token_offset, blocked_expert_counts,
        blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens,
        mNumExpertsPerNode, mK, start_expert_id, /*enable_pdl*/ false, stream);
    sync_check_cuda_error(stream);
  }
}

void DualWeightGemmProfilerBackend::prepare(int num_tokens, char* workspace_ptr_char,
                                  void const* upper_expert_weights,
                                  void const* lower_expert_weights,
                                  cudaStream_t stream) {
  mAllTacticsSaved = mInterface->getTactics();
  mSampleIndex = 0;

  auto workspace_size = getWorkspaceSize(num_tokens);
  populateRandomBuffer(workspace_ptr_char, workspace_size, stream);

  prepareRouting(num_tokens, workspace_ptr_char, stream);
}

size_t DualWeightGemmProfilerBackend::getWorkspaceSize(int maxM) {
  auto sizes_map = getProfilerWorkspaces(maxM);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(),
                 [](auto& v) { return v.second.first; });
  size_t size = calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  TLLM_LOG_TRACE("MOE profiler workspace size: %zu", size);
  return size;
}

void DualWeightGemmProfilerBackend::runProfiler(int original_num_tokens, Config const& tactic,
                                      char* workspace_ptr_char, void const* upper_expert_weights,
                                      void const* lower_expert_weights,
                                      cudaStream_t const& stream) {
  int64_t expanded_num_tokens = original_num_tokens * mK;
  int64_t num_experts_per_node = mNumExpertsPerNode;

  mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

  auto workspaces = getProfilerWorkspaces(original_num_tokens);

#define GET_WS_PTR_OFFSET(type, name, offset)                                                    \
  auto* name =                                                                                   \
      (workspaces.at(#name).first                                                                \
           ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) + (offset) \
           : nullptr)
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR_OFFSET(int64_t const*, expert_first_token_offset,
                    (mSampleIndex * (mNumExpertsPerNode + 1)));
  GET_WS_PTR_OFFSET(int const*, unpermuted_row_to_permuted_row,
                    (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, permuted_row_to_unpermuted_row,
                    (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, token_selected_experts, (mSampleIndex * expanded_num_tokens));

  GET_WS_PTR(float const*, token_topk_unpermuted_scales);
  auto const* token_topk_permuted_scales = token_topk_unpermuted_scales;

  GET_WS_PTR_OFFSET(int*, num_active_experts_per_node, mSampleIndex);
  GET_WS_PTR_OFFSET(int*, active_expert_global_ids, (mSampleIndex * mNumExpertsPerNode));
  GET_WS_PTR(void const*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void const*, upper_weights);
  GET_WS_PTR(void const*, lower_weights);
  TLLM_CHECK(mNeedWeights == (upper_expert_weights == nullptr || lower_expert_weights == nullptr));
  void const* upper_weights_sel = mNeedWeights ? upper_weights : upper_expert_weights;
  void const* lower_weights_sel = mNeedWeights ? lower_weights : lower_expert_weights;
  GET_WS_PTR(void const*, bias);

  GET_WS_PTR(void*, gemm_workspace);

  GET_WS_PTR(float*, swiglu_alpha);
  GET_WS_PTR(float*, swiglu_beta);
  GET_WS_PTR(float*, swiglu_limit);

#undef GET_WS_PTR_OFFSET
#undef GET_WS_PTR

  mInterface->is_profiler = true;
  if (mGemmToProfile == GemmToProfile::GEMM_1) {
    mInterface->gemm1(
        input,                                             //
        output,                                            //
        intermediate,                                      //
        expert_first_token_offset,                         //
        upper_weights_sel,                                 //
        lower_weights_sel,                                 //
        bias,                                              //
        expert_first_token_offset + num_experts_per_node,  //
        original_num_tokens,                                                         //
        expanded_num_tokens,                                                         //
        mExpertHiddenSize,                                                           //
        mExpertInterSize,                                                            //
        num_experts_per_node,                                                        //
        ActivationParams(mActivationType, swiglu_alpha, swiglu_beta, swiglu_limit),  //
        /*bias_is_broadcast*/ false,                                                 //
        stream,                                                                      //
        tactic,                                                                      //
        num_active_experts_per_node,                                                 //
        active_expert_global_ids);                                                   //
  } else {
    TLLM_CHECK(mGemmToProfile == GemmToProfile::GEMM_2);
    mInterface->gemm2(
        input, intermediate, output, expert_first_token_offset, upper_weights_sel, lower_weights_sel,
        bias, token_topk_unpermuted_scales, token_topk_permuted_scales,
        unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row, token_selected_experts,
        expert_first_token_offset + mNumExpertsPerNode, original_num_tokens, expanded_num_tokens,
        mExpertHiddenSize, mExpertInterSize, num_experts_per_node, mK,
        stream, mParallelismConfig, mEnableAlltoall, tactic,
        num_active_experts_per_node, active_expert_global_ids);
  }
  mInterface->is_profiler = false;

  sync_check_cuda_error(stream);
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels
