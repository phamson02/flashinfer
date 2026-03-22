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
#include <vector>

#include "cutlass_fused_moe_kernels.cuh"
#include "dual_weight_moe_kernels.h"
#include "moe_kernels.h"
#include "moe_util_kernels.h"
#include "tensorrt_llm/common/workspace.h"

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels::cutlass_kernels {

template <typename T, typename WeightType, typename OutputType, typename InputType,
          typename BackBoneType, typename Enable>
DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType,
                      Enable>::DualWeightMoeFCRunner() {}

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

  bool using_tma_ws = moe_gemm_runner_.supportsTmaWarpSpecialized();

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

  size_t const tma_ws_size = using_tma_ws ? DualWeightTmaWarpSpecializedGroupedGemmInput::workspaceSize(
                                                num_experts_per_node, getScalingType())
                                          : 0;

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
  ADD_NAME(tma_ws_gemm1_workspace, tma_ws_size);
  ADD_NAME(tma_ws_gemm2_workspace, tma_ws_size);
  ADD(gemm_workspace);

  return out_map;

#undef ADD_NAME
#undef ADD
}

template <typename T, typename WeightType, typename OutputType, typename InputType,
          typename BackBoneType, typename Enable>
size_t
DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::getWorkspaceSize(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts, int const experts_per_token, ActivationType activation_type,
    MOEParallelismConfig parallelism_config) {
  int const ep_size = parallelism_config.ep_size;
  TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0,
                       "Number of experts must be a multiple of ep size");
  auto sizes_map = getWorkspaceDeviceBufferSizes(
      num_rows, hidden_size, inter_size, num_experts / ep_size, experts_per_token, activation_type);
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

  // We check if the provided config uses fused finalize and disable it if it does not
  bool gemm2_using_finalize_fusion =
      gemm2_config_->epilogue_fusion_type ==
      cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
  permuted_token_final_scales_ =
      gemm2_using_finalize_fusion ? getWsPtr(float{}, "permuted_token_final_scales") : nullptr;

  bool const is_gated_activation = isGatedActivation(activation_type);
  bool const gemm1_using_fused_moe = moe_gemm_runner_.isFusedGatedActivation(
      *gemm1_config_, activation_type, inter_size, hidden_size);
  bool const gemm1_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
  bool const tma_ws_has_glu =
      gemm1_using_tma_ws && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
  // We always use fused path if we can
  bool const non_tma_ws_has_glu = !gemm1_using_fused_moe && is_gated_activation;
  bool const has_glu_inter_result = tma_ws_has_glu || non_tma_ws_has_glu;

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

  tma_ws_grouped_gemm1_input_ = {};
  tma_ws_grouped_gemm2_input_ = {};
  if (moe_gemm_runner_.supportsTmaWarpSpecialized()) {
    tma_ws_grouped_gemm1_input_.configureWorkspace(
        getWsPtr(int8_t{}, "tma_ws_gemm1_workspace"), num_experts_per_node,
        getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
        getScalingType());
    tma_ws_grouped_gemm2_input_.configureWorkspace(
        getWsPtr(int8_t{}, "tma_ws_gemm2_workspace"), num_experts_per_node,
        getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
        getScalingType());
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          class Enable>
void DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::gemm1(
    DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    T const* const input, T* const output, void* const intermediate_result,
    int64_t const* const expert_first_token_offset,
    DualWeightTmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
    WeightType const* const fc1_upper_expert_weights,
    WeightType const* const fc1_lower_expert_weights, ScaleBiasType const* const fc1_expert_biases,
    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, ActivationParams fc1_activation_type, bool bias_is_broadcast,
    cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
    int* active_expert_global_ids, bool enable_pdl) {
  bool const using_tma_ws_gemm1 = gemm_runner.isTmaWarpSpecialized(config);
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);
  bool const use_ampere_activation_fusion = gemm_runner.isFusedGatedActivation(
      config, fc1_activation_type.activation_type, inter_size, hidden_size);
  size_t const fc1_out_size =
      ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  if (using_tma_ws_gemm1) {
    TLLM_CHECK(config.is_tma_warp_specialized);
    TLLM_CHECK(!use_ampere_activation_fusion);

    bool has_different_gemm_output_type = false;
    bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
    TLLM_CHECK_WITH_INFO(has_intermediate || input != output,
                         "Input and output buffers are overlapping");
    
    auto* gemm_output = has_intermediate ? intermediate_result : static_cast<void*>(output);

    auto tma_ws_input = tma_ws_input_template;

    DualWeightGroupedGemmInput<T, WeightType, OutputType, OutputType> universal_input;
    universal_input.A = input;
    universal_input.total_tokens_including_expert = total_tokens_including_expert;
    universal_input.B = nullptr;
    universal_input.B_lower = nullptr;
    universal_input.scales = nullptr;
    universal_input.zeros = nullptr;
    universal_input.biases = nullptr;
    universal_input.C = nullptr;
    universal_input.alpha_scales = nullptr;
    universal_input.occupancy = nullptr;
    universal_input.activation_type = fc1_activation_type.activation_type;
    universal_input.num_rows = num_rows;
    universal_input.n = int64_t(fc1_out_size);
    universal_input.k = hidden_size;
    universal_input.num_experts = num_experts_per_node;
    universal_input.bias_is_broadcast = true;
    universal_input.use_fused_moe = false;
    universal_input.stream = stream;
    universal_input.gemm_config = config;

    gemm_runner.moeGemm(universal_input, tma_ws_input);

    sync_check_cuda_error(stream);

    // TODO: when bias_is_broadcast is false, fuse bias to gemm
    using GatedActOutputType = T;
    QuantParams empty_quant_params;
    doActivation<GatedActOutputType, UnfusedGemmOutputType>(
        reinterpret_cast<GatedActOutputType*>(output),
        static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr, fc1_expert_biases,
        bias_is_broadcast, expert_first_token_offset, num_experts_per_node, inter_size,
        expanded_num_rows, fc1_activation_type, empty_quant_params, /*use_per_expert_act_scale*/ false,
        /*fc2_act_sf_flat=*/ nullptr, enable_pdl, stream);

    sync_check_cuda_error(stream);
  } else if (!is_gated_activation) {
    TLLM_CHECK(!use_ampere_activation_fusion);
    TLLM_CHECK(!config.is_tma_warp_specialized);

    DualWeightGroupedGemmInput<T, WeightType, OutputType, OutputType> universal_input;
    universal_input.A = input;
    universal_input.total_tokens_including_expert = total_tokens_including_expert;
    universal_input.B = fc1_upper_expert_weights;
    universal_input.B_lower = fc1_lower_expert_weights;
    universal_input.scales = nullptr;
    universal_input.zeros = nullptr;
    universal_input.biases = fc1_expert_biases;
    universal_input.C = reinterpret_cast<OutputType*>(output);
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

    gemm_runner.moeGemmBiasAct(universal_input, DualWeightTmaWarpSpecializedGroupedGemmInput{});

    sync_check_cuda_error(stream);
  } else {
    TLLM_CHECK(!config.is_tma_warp_specialized);
    TLLM_CHECK(is_gated_activation);
    TLLM_CHECK_WITH_INFO(!use_ampere_activation_fusion || input != output,
                         "Input and output buffers are overlapping");

    // Run the GEMM with activation function overridden with `Identity`, we do the activation
    // separately
    DualWeightGroupedGemmInput<T, WeightType, OutputType, OutputType> universal_input;
    universal_input.A = input;
    universal_input.total_tokens_including_expert = total_tokens_including_expert;
    universal_input.B = fc1_upper_expert_weights;
    universal_input.B_lower = fc1_lower_expert_weights;
    universal_input.scales = nullptr;
    universal_input.zeros = nullptr;
    universal_input.biases = fc1_expert_biases;
    universal_input.C =
        static_cast<OutputType*>(use_ampere_activation_fusion ? output : intermediate_result);
    universal_input.alpha_scales = nullptr;
    universal_input.occupancy = nullptr;
    universal_input.activation_type = use_ampere_activation_fusion
                                          ? fc1_activation_type.activation_type
                                          : ActivationType::Identity;
    universal_input.num_rows = expanded_num_rows;
    universal_input.n = int64_t(fc1_out_size);
    universal_input.k = hidden_size;
    universal_input.num_experts = num_experts_per_node;
    universal_input.bias_is_broadcast = bias_is_broadcast;
    universal_input.use_fused_moe = use_ampere_activation_fusion;
    universal_input.stream = stream;
    universal_input.gemm_config = config;

    gemm_runner.moeGemmBiasAct(universal_input, DualWeightTmaWarpSpecializedGroupedGemmInput{});

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
    DualWeightMoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    T const* const input, void* const gemm_output, OutputType* const final_output,
    int64_t const* const expert_first_token_offset,
    DualWeightTmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
    WeightType const* const fc2_expert_upper_weights,
    WeightType const* const fc2_expert_lower_weights, ScaleBiasType const* const fc2_expert_biases,
    float const* const unpermuted_final_scales, float const* const permuted_final_scales,
    int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
    int const* const token_selected_experts, int64_t const* const num_valid_tokens_ptr,
    int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size,
    int64_t const unpadded_hidden_size, int64_t const inter_size, int const num_experts_per_node, int64_t const k,
    cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
    cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
    int* active_expert_global_ids, bool enable_pdl) {
  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  bool const using_tma_ws_gemm2 = gemm_runner.isTmaWarpSpecialized(config);

  DualWeightTmaWarpSpecializedGroupedGemmInput tma_ws_input{};
  if (using_tma_ws_gemm2) {
    tma_ws_input = tma_ws_input_template;
    if (tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
      // TODO For some reason this has to be done here, it should not overlap with anything else,
      // but doing it in setupTmaWarpSpecializedInputs gives a different result. Ideally, we want
      // this to run on a second stream and overlap with everything else
      //
      // This also means it is included in the timing for the profiler, which is probably more
      // representative until we can overlap it
      check_cuda_error(cudaMemsetAsync(
          final_output, 0x0, sizeof(OutputType) * num_rows * unpadded_hidden_size, stream));
    }
  }

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

  gemm_runner.moeGemmBiasAct(universal_input, tma_ws_input);
  sync_check_cuda_error(stream);

  bool using_fused_finalize =
      tma_ws_input.fusion == DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool has_different_output_type_tma_ws = !using_fused_finalize && using_tma_ws_gemm2;

  if (has_different_output_type_tma_ws) {
    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases, unpermuted_final_scales,
        unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row, token_selected_experts,
        expert_first_token_offset, num_rows, hidden_size, unpadded_hidden_size, k,
        num_experts_per_node, parallelism_config, enable_alltoall, enable_pdl, stream);
  } else if (!using_tma_ws_gemm2) {
    finalizeMoeRoutingKernelLauncher<OutputType, T>(
        static_cast<T const*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, expert_first_token_offset, num_rows, hidden_size,
        unpadded_hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
        enable_pdl, stream);
  }

  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          class Enable>
void DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::runMoe(
    void const* input_activations_void, int const* token_selected_experts,
    float const* token_final_scales, void const* fc1_upper_expert_weights_void,
    void const* fc1_lower_expert_weights_void, void const* fc1_expert_biases_void,
    ActivationParams fc1_activation_type, void const* fc2_upper_expert_weights_void,
    void const* fc2_lower_expert_weights_void, void const* fc2_expert_biases_void,
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const full_num_experts, int const experts_per_token, char* workspace_ptr,
    void* final_output_void, int* unpermuted_row_to_permuted_row,
    MOEParallelismConfig parallelism_config, bool const enable_alltoall, bool enable_pdl,
    cudaStream_t stream) {
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

  // For NoSmem epilogue schedule, we need to align the output of the GEMM to 256 bits, for gated
  // activation this is automatic if the usual alignment requirement is met
  if (gemm1_config_->epilogue_schedule == cutlass_extensions::EpilogueScheduleType::NO_SMEM &&
      !isGatedActivation(fc1_activation_type)) {
    TLLM_CHECK_WITH_INFO(
        inter_size % (256 / sizeof_bits<WeightType>::value) == 0,
        "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d",
        (int)inter_size, (int)(256 / sizeof_bits<WeightType>::value));
  }

  if (gemm2_config_->epilogue_schedule == cutlass_extensions::EpilogueScheduleType::NO_SMEM) {
    TLLM_CHECK_WITH_INFO(
        gemm2_config_->epilogue_fusion_type !=
            cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE,
        "Got NoSmem epilogue schedule, which is not supported for finalize fusion");
    TLLM_CHECK_WITH_INFO(
        hidden_size % (256 / sizeof_bits<WeightType>::value) == 0,
        "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d",
        (int)hidden_size, (int)(256 / sizeof_bits<WeightType>::value));
  }

  // Require at least 128 bits of alignment for MOE GEMM
  TLLM_CHECK_WITH_INFO(
      hidden_size % (128 / sizeof_bits<WeightType>::value) == 0,
      "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d",
      (int)hidden_size, (int)(128 / sizeof_bits<WeightType>::value));
  TLLM_CHECK_WITH_INFO(inter_size % (128 / sizeof_bits<WeightType>::value) == 0,
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
  int64_t const unpadded_hidden_size = hidden_size;

  int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;

  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node,
                  experts_per_token, fc1_activation_type.activation_type, parallelism_config);

  int start_expert = num_experts_per_node * parallelism_config.ep_rank;
  int end_expert = start_expert + num_experts_per_node;

  bool const needs_num_valid = parallelism_config.ep_size > 1;
  int64_t const* num_valid_tokens_ptr =
      needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

  auto expanded_num_rows = num_rows * experts_per_token;

  bool fused_prologue_result = fusedBuildExpertMapsSortFirstToken(
      token_selected_experts, permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row,
      expert_first_token_offset_, num_rows, num_experts_per_node, experts_per_token, start_expert,
      end_expert, enable_pdl, stream);

  if (!fused_prologue_result) {
    TLLM_LOG_TRACE("Falling back to unfused prologue");
    threeStepBuildExpertMapsSortFirstToken(
        token_selected_experts, permuted_token_selected_experts_, permuted_row_to_unpermuted_row_,
        unpermuted_row_to_permuted_row, expert_first_token_offset_, blocked_expert_counts_,
        blocked_expert_counts_cumsum_, blocked_row_to_unpermuted_row_, num_rows,
        num_experts_per_node, experts_per_token, start_expert, enable_pdl, stream);
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
                                /*prequant_scales*/ nullptr, enable_pdl, stream);
  auto const* gemm1_input = gemm1_input_expand;

  sync_check_cuda_error(stream);

  auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(
      num_rows, expanded_num_rows, fc1_activation_type, hidden_size, unpadded_hidden_size,
      inter_size, num_experts_per_node, input_activations_void, final_output,
      fc1_upper_expert_weights, fc1_lower_expert_weights, fc2_upper_expert_weights,
      fc2_lower_expert_weights, fc1_expert_biases, fc2_expert_biases, parallelism_config,
      enable_pdl, stream);

  Self::gemm1(moe_gemm_runner_, gemm1_input, fc1_result_, glu_inter_result_,
              expert_first_token_offset_, gemm1_tma_ws_input, fc1_upper_expert_weights,
              fc1_lower_expert_weights,
              fc1_expert_biases, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size,
              inter_size, num_experts_per_node, fc1_activation_type,
              /*bias_is_broadcast*/ true, stream, *gemm1_config_,
              /*num_active_experts_per*/ nullptr,
              /*active_expert_global_ids*/ nullptr, enable_pdl);
  sync_check_cuda_error(stream);

  T* gemm2_input = reinterpret_cast<T*>(fc1_result_);
  Self::gemm2(moe_gemm_runner_, gemm2_input, fc2_result_, final_output, expert_first_token_offset_,
              gemm2_tma_ws_input, fc2_upper_expert_weights, fc2_lower_expert_weights,
              fc2_expert_biases,
              token_topk_unpermuted_scales, permuted_token_final_scales_,
              unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
              token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
              hidden_size, unpadded_hidden_size, inter_size, num_experts_per_node, experts_per_token,
              stream, parallelism_config, enable_alltoall, *gemm2_config_,
              /*num_active_experts_per*/ nullptr,
              /*active_expert_global_ids*/ nullptr, enable_pdl);
  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__device__ void computeDualWeightTmaWarpSpecializedInputPointers(
    DualWeightTmaWarpSpecializedGroupedGemmInput& layout_info, int64_t gemm_m, int64_t gemm_n,
    int64_t gemm_k, int num_tokens_before_expert, int64_t expert, T const* in,
    WeightType const* upper_weights, WeightType const* lower_weights, ScaleBiasType const* bias,
    OutputType* output, float const* router_scales, int const* permuted_row_to_unpermuted_row,
    int64_t const out_idx) {
  // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens.
  layout_info.ptr_act[out_idx] = safe_inc_ptr(in, num_tokens_before_expert * gemm_k);
  // Each expert's upper/lower weight matrix is a constant size NxK.
  layout_info.ptr_weight[out_idx] = safe_inc_ptr(upper_weights, expert * (gemm_n * gemm_k));
  layout_info.ptr_weight_2[out_idx] = safe_inc_ptr(lower_weights, expert * (gemm_n * gemm_k));

  if (layout_info.fusion == DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
    layout_info.ptr_d[out_idx] = safe_inc_ptr(output, num_tokens_before_expert * gemm_n);
  }
  if (layout_info.fusion ==
      DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
    layout_info.fused_finalize_epilogue.ptr_source_token_index[expert] =
        permuted_row_to_unpermuted_row + num_tokens_before_expert;
    layout_info.fused_finalize_epilogue.ptr_router_scales[expert] =
        router_scales + num_tokens_before_expert;
    if (layout_info.fused_finalize_epilogue.ptr_bias != nullptr) {
      layout_info.fused_finalize_epilogue.ptr_bias[expert] = bias + gemm_n * expert;
    }
  }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeDualWeightStridesTmaWarpSpecializedKernel(
    int64_t const* expert_first_token_offset,
    DualWeightTmaWarpSpecializedGroupedGemmInput layout_info1,
    DualWeightTmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens,
    int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
    int64_t const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
    WeightType const* gemm1_upper_weights, WeightType const* gemm1_lower_weights,
    WeightType const* gemm2_upper_weights, WeightType const* gemm2_lower_weights,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* gemm1_output,
    OutputType* gemm2_output, float const* router_scales,
    int const* permuted_row_to_unpermuted_row) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Both gemms use the same token offset
  auto const num_tokens_before_expert = expert_first_token_offset[expert];
  auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
  auto const gemm_m = num_tokens_including_expert - num_tokens_before_expert;

  (void)num_tokens;
  (void)expanded_num_tokens;
  // M and N transposed since we are using the #tokens as the N dimension
  layout_info1.shape_info.problem_shapes[expert] =
      DualWeightTmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info1.swap_ab ? gemm1_n : gemm_m, layout_info1.swap_ab ? gemm_m : gemm1_n,
          gemm1_k);
  layout_info2.shape_info.problem_shapes[expert] =
      DualWeightTmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info2.swap_ab ? gemm2_n : gemm_m, layout_info2.swap_ab ? gemm_m : gemm2_n,
          gemm2_k);

  assert(gemm_m <= INT32_MAX);
  assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
  assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
  assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
  assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
  computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
  computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);

  computeDualWeightTmaWarpSpecializedInputPointers(
      layout_info1, gemm_m, gemm1_n, gemm1_k, num_tokens_before_expert, expert, gemm1_in,
      gemm1_upper_weights, gemm1_lower_weights, bias1, gemm1_output, nullptr, nullptr, expert);
  computeDualWeightTmaWarpSpecializedInputPointers(
      layout_info2, gemm_m, gemm2_n, gemm2_k, num_tokens_before_expert, expert, gemm2_in,
      gemm2_upper_weights, gemm2_lower_weights, bias2, gemm2_output, router_scales,
      permuted_row_to_unpermuted_row, expert);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          class Enable>
std::pair<DualWeightTmaWarpSpecializedGroupedGemmInput, DualWeightTmaWarpSpecializedGroupedGemmInput>
DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::
    computeStridesTmaWarpSpecialized(
        int64_t const* expert_first_token_offset, DualWeightTmaWarpSpecializedGroupedGemmInput layout_info1,
        DualWeightTmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens,
        int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
        int64_t gemm2_k, int const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
        WeightType const* gemm1_upper_weights, WeightType const* gemm1_lower_weights,
        WeightType const* gemm2_upper_weights, WeightType const* gemm2_lower_weights,
        ScaleBiasType const* bias1, ScaleBiasType const* bias2,
        UnfusedGemmOutputType* gemm1_output, UnfusedGemmOutputType* gemm2_output,
        float const* router_scales, int const* permuted_row_to_unpermuted_row, bool enable_pdl,
        cudaStream_t stream) {
  // Always nullptr
  layout_info1.ptr_c = nullptr;
  layout_info1.stride_c = nullptr;
  layout_info1.alpha_scale_ptr_array = nullptr;
  layout_info2.ptr_c = nullptr;
  layout_info2.stride_c = nullptr;
  layout_info2.alpha_scale_ptr_array = nullptr;

  layout_info1.fused_finalize_epilogue.ptr_bias = nullptr;
  if (!bias2) {
    layout_info2.fused_finalize_epilogue.ptr_bias = nullptr;
  }

  layout_info1.fpX_block_scaling_type = getScalingType();
  layout_info2.fpX_block_scaling_type = getScalingType();

  int const threads = std::min(1024, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  auto* kernel_instance =
      &computeDualWeightStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1, layout_info2,
                     num_tokens, expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k,
                     num_experts_per_node, gemm1_in, gemm2_in, gemm1_upper_weights,
                     gemm1_lower_weights, gemm2_upper_weights, gemm2_lower_weights, bias1, bias2,
                     gemm1_output, gemm2_output, router_scales, permuted_row_to_unpermuted_row);

  return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          class Enable>
std::pair<DualWeightTmaWarpSpecializedGroupedGemmInput, DualWeightTmaWarpSpecializedGroupedGemmInput>
DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::
    setupTmaWarpSpecializedInputs(int64_t num_rows, int64_t expanded_num_rows,
                                  ActivationParams fc1_activation_type, int64_t hidden_size,
                                  int64_t unpadded_hidden_size, int64_t inter_size,
                                  int64_t num_experts_per_node, void const* input_activations_void, void* final_output,
                                  WeightType const* fc1_upper_expert_weights,
                                  WeightType const* fc1_lower_expert_weights,
                                  WeightType const* fc2_upper_expert_weights,
                                  WeightType const* fc2_lower_expert_weights,
                                  ScaleBiasType const* fc1_expert_biases,
                                  ScaleBiasType const* fc2_expert_biases, MOEParallelismConfig parallelism_config,
                                  bool enable_pdl, cudaStream_t stream) {
  auto gemm1_tma_ws_input = tma_ws_grouped_gemm1_input_;
  auto gemm2_tma_ws_input = tma_ws_grouped_gemm2_input_;

  // Set enable_pdl for both GEMM inputs
  gemm1_tma_ws_input.enable_pdl = enable_pdl;
  gemm2_tma_ws_input.enable_pdl = enable_pdl;
  if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) &&
      !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)) {
    return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
  }

  bool is_gated_activation = isGatedActivation(fc1_activation_type);
  int64_t const fc1_out_size = is_gated_activation ? inter_size * 2 : inter_size;

  bool has_different_gemm_output_type = !std::is_same_v<T, UnfusedGemmOutputType>;
  bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
  auto* gemm1_output = has_intermediate ? glu_inter_result_ : static_cast<void*>(fc1_result_);

  auto gemm2_input = fc1_result_;

  auto gemm1_input = permuted_data_;
  gemm1_tma_ws_input.fusion = DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
  gemm2_tma_ws_input.fusion = DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

  bool const force_sm90_dual_swap_ab =
      moe_gemm_runner_.supportsTmaWarpSpecialized() && moe_gemm_runner_.getSM() == 90;
  gemm1_tma_ws_input.swap_ab = force_sm90_dual_swap_ab ? true : gemm1_config_->swap_ab;
  gemm2_tma_ws_input.swap_ab = force_sm90_dual_swap_ab ? true : gemm2_config_->swap_ab;

  bool apply_bias = parallelism_config.tp_rank == 0;
  auto* fc2_bias = apply_bias ? fc2_expert_biases : nullptr;
  bool gemm2_using_finalize_fusion =
      gemm2_config_->epilogue_fusion_type ==
      cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
  bool using_fused_finalize = mayHaveFinalizeFused() && gemm2_using_finalize_fusion;
  TLLM_CHECK_WITH_INFO(
      using_fused_finalize == gemm2_using_finalize_fusion,
      "GEMM2 tactic requests finalize fusion, but the runner is not configured to use it");
  if (using_fused_finalize) {
    bool use_reduction = expanded_num_rows > num_rows;
    gemm2_tma_ws_input.fusion =
        DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
    gemm2_tma_ws_input.setFinalizeFusionParams(final_output, unpadded_hidden_size, num_rows,
                                                use_reduction);
  }

  TLLM_CHECK_WITH_INFO(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
  return Self::computeStridesTmaWarpSpecialized(
      expert_first_token_offset_, gemm1_tma_ws_input, gemm2_tma_ws_input, num_rows,
      expanded_num_rows, fc1_out_size, hidden_size, hidden_size, inter_size, num_experts_per_node,
      reinterpret_cast<T const*>(gemm1_input), reinterpret_cast<T const*>(gemm2_input),
      fc1_upper_expert_weights, fc1_lower_expert_weights, fc2_upper_expert_weights,
      fc2_lower_expert_weights, fc1_expert_biases, fc2_bias,
      reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
      reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), permuted_token_final_scales_,
      permuted_row_to_unpermuted_row_, enable_pdl, stream);
}

std::map<std::string, std::pair<size_t, size_t>>
DualWeightGemmProfilerBackend::getProfilerWorkspaces(int maxM, bool is_tma_ws_input) {
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

  size_t tma_ws_input_workspace_size = 0;
  if (is_tma_ws_input) {
    tma_ws_input_workspace_size =
        DualWeightTmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node, mScalingType) *
        (NUM_ROUTING_SAMPLES * NUM_FUSION_TYPES * NUM_SWAP_AB_TYPES + 1);
  }

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
  ADD(tma_ws_input_workspace);
  ADD(gemm_workspace);
  ADD(swiglu_alpha);
  ADD(swiglu_beta);
  ADD(swiglu_limit);

#undef ADD_NAME
#undef ADD

  return out_map;
}

void DualWeightGemmProfilerBackend::prepareRouting(int num_tokens, char* workspace_ptr_char, bool enable_pdl,
                                                   cudaStream_t stream) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
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
        blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens, mNumExpertsPerNode,
        mK, start_expert_id, enable_pdl, stream);
    sync_check_cuda_error(stream);
  }
}

void DualWeightGemmProfilerBackend::prepareTmaWsInputs(
    int num_tokens, char* workspace_ptr_char, void const* upper_expert_weights,
    void const* lower_expert_weights, DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion fusion,
    bool swap_ab, bool enable_pdl, cudaStream_t stream) {
  if (mSM < 90) {
    return;
  }
  bool const effective_swap_ab = (mSM == 90) ? true : swap_ab;
  if (swap_ab != effective_swap_ab) {
    return;
  }

  bool const use_finalize_fusion =
      fusion == DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool const finalize_fusion_not_supported = !mInterface->use_fused_finalize_ ||
                                             mGemmToProfile != GemmToProfile::GEMM_2;
  if (use_finalize_fusion && finalize_fusion_not_supported) {
    return;
  }

  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);

#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR(int64_t*, expert_first_token_offset);
  int64_t* expert_first_token_offset_base = expert_first_token_offset;
  GET_WS_PTR(int*, permuted_row_to_unpermuted_row);
  int* permuted_row_to_unpermuted_row_base = permuted_row_to_unpermuted_row;
  GET_WS_PTR(void*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void*, upper_weights);
  GET_WS_PTR(void*, lower_weights);
  bool const upper_is_null = (upper_expert_weights == nullptr);
  bool const lower_is_null = (lower_expert_weights == nullptr);
  TLLM_CHECK_WITH_INFO(
      upper_is_null == lower_is_null,
      "Dual-weight profiler expects upper/lower weights to be both null or both non-null");
  TLLM_CHECK(mNeedWeights == upper_is_null);
  void const* upper_weights_sel = mNeedWeights ? upper_weights : upper_expert_weights;
  void const* lower_weights_sel = mNeedWeights ? lower_weights : lower_expert_weights;
  GET_WS_PTR(void*, bias);
  GET_WS_PTR(float*, token_topk_unpermuted_scales);
  GET_WS_PTR(int8_t*, tma_ws_input_workspace);
  GET_WS_PTR(void*, gemm_workspace);

#undef GET_WS_PTR

  size_t tma_ws_size =
      DualWeightTmaWarpSpecializedGroupedGemmInput::workspaceSize(mNumExpertsPerNode, mScalingType);

  DualWeightTmaWarpSpecializedGroupedGemmInput dummy_tma_ws_input;
  dummy_tma_ws_input.configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                        workspaces.at("gemm_workspace").first, mScalingType);
  dummy_tma_ws_input.enable_pdl = enable_pdl;  // Set enable_pdl for dummy input
  tma_ws_input_workspace += tma_ws_size;

  int workspace_index =
      static_cast<int>(use_finalize_fusion) * (NUM_SWAP_AB_TYPES * NUM_ROUTING_SAMPLES) +
      static_cast<int>(effective_swap_ab) * NUM_ROUTING_SAMPLES;
  tma_ws_input_workspace += workspace_index * tma_ws_size;

  size_t num_expanded_tokens = num_tokens * mK;
  for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
    // Note: Even though we have separate TMA WS inputs for finalize fusion on/off we reuse the same
    // pointers to save space.
    auto& cache_element = mTmaInputCache[use_finalize_fusion][effective_swap_ab][i];
    cache_element.configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                     workspaces.at("gemm_workspace").first, mScalingType);
    cache_element.enable_pdl = enable_pdl;  // Set enable_pdl for cache element
    tma_ws_input_workspace += tma_ws_size;

    int64_t* expert_first_token_offset =
        expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
    int* permuted_row_to_unpermuted_row =
        permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;

    auto& gemm1_tma_ws_input =
        mGemmToProfile == GemmToProfile::GEMM_1 ? cache_element : dummy_tma_ws_input;
    auto& gemm2_tma_ws_input =
        mGemmToProfile == GemmToProfile::GEMM_2 ? cache_element : dummy_tma_ws_input;
    if (mSM >= 90) {
      auto fc1_output_size =
          isGatedActivation(mActivationType) ? mExpertInterSize * 2 : mExpertInterSize;

      /* GEMM1 */
      gemm1_tma_ws_input.fusion = DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
      gemm2_tma_ws_input.fusion = DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

      gemm1_tma_ws_input.swap_ab = effective_swap_ab;
      gemm2_tma_ws_input.swap_ab = effective_swap_ab;

      if (use_finalize_fusion) {
        gemm2_tma_ws_input.fusion = DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
        gemm2_tma_ws_input.setFinalizeFusionParams(output, mExpertUnpaddedHiddenSize, num_tokens,
                                                   mK > 1);
      }

      std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) =
          mInterface->computeStridesTmaWarpSpecializedDispatch(
              expert_first_token_offset, gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens,
              num_tokens * mK, fc1_output_size, mExpertHiddenSize, mExpertHiddenSize,
              mExpertInterSize, mNumExpertsPerNode, input, input, upper_weights_sel, lower_weights_sel,
              upper_weights_sel, lower_weights_sel, bias, bias, intermediate, intermediate,
              token_topk_unpermuted_scales, permuted_row_to_unpermuted_row, enable_pdl, stream);

      sync_check_cuda_error(stream);
    }
  }
}

void DualWeightGemmProfilerBackend::prepare(int num_tokens, char* workspace_ptr_char,
                                            void const* upper_expert_weights,
                                            void const* lower_expert_weights, bool enable_pdl,
                                            cudaStream_t stream) {
  mSampleIndex = 0;

  auto workspace_size = getWorkspaceSize(num_tokens);
  populateRandomBuffer(workspace_ptr_char, workspace_size, stream);

  prepareRouting(num_tokens, workspace_ptr_char, enable_pdl, stream);
  for (auto fusion : {DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE,
                      DualWeightTmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE}) {
    for (auto swap_ab : {false, true}) {
      prepareTmaWsInputs(num_tokens, workspace_ptr_char, upper_expert_weights, lower_expert_weights,
                         fusion, swap_ab, enable_pdl, stream);
    }
  }
}

size_t DualWeightGemmProfilerBackend::getWorkspaceSize(int maxM) {
  auto sizes_map = getProfilerWorkspaces(maxM, mSM >= 90);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(),
                 [](auto& v) { return v.second.first; });
  size_t size = calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  TLLM_LOG_TRACE("MOE profiler workspace size: %zu", size);
  return size;
}

void DualWeightGemmProfilerBackend::runProfiler(int original_num_tokens, Config const& tactic,
                                                char* workspace_ptr_char,
                                                void const* upper_expert_weights,
                                                void const* lower_expert_weights,
                                                bool enable_pdl,
                                                cudaStream_t const& stream) {
  int64_t expanded_num_tokens = original_num_tokens * mK;
  int64_t num_experts_per_node = mNumExpertsPerNode;

  mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

  auto workspaces = getProfilerWorkspaces(original_num_tokens, tactic.is_tma_warp_specialized);

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
  bool const upper_is_null = (upper_expert_weights == nullptr);
  bool const lower_is_null = (lower_expert_weights == nullptr);
  TLLM_CHECK_WITH_INFO(
      upper_is_null == lower_is_null,
      "Dual-weight profiler expects upper/lower weights to be both null or both non-null");
  TLLM_CHECK(mNeedWeights == upper_is_null);
  void const* upper_weights_sel = mNeedWeights ? upper_weights : upper_expert_weights;
  void const* lower_weights_sel = mNeedWeights ? lower_weights : lower_expert_weights;
  GET_WS_PTR(void const*, bias);
  GET_WS_PTR(void*, gemm_workspace);

  GET_WS_PTR(float*, swiglu_alpha);
  GET_WS_PTR(float*, swiglu_beta);
  GET_WS_PTR(float*, swiglu_limit);

#undef GET_WS_PTR_OFFSET
#undef GET_WS_PTR

  DualWeightTmaWarpSpecializedGroupedGemmInput tma_ws_input_template;
  auto tactic_to_run = tactic;
  bool const effective_swap_ab =
      (tactic.is_tma_warp_specialized && mSM == 90) ? true : tactic.swap_ab;
  tactic_to_run.swap_ab = effective_swap_ab;
  if (tactic.is_tma_warp_specialized) {
    // Use non-finalize cache when finalize fusion is not supported for the current GEMM
    bool finalize_supported_this_gemm = (mGemmToProfile == GemmToProfile::GEMM_2) &&
                                        mInterface->use_fused_finalize_;
    bool request_finalize = tactic.epilogue_fusion_type ==
                            cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
    bool use_finalize_index = request_finalize && finalize_supported_this_gemm;

    tma_ws_input_template = mTmaInputCache[use_finalize_index][effective_swap_ab][mSampleIndex];
    TLLM_CHECK_WITH_INFO(tma_ws_input_template.isValid(),
                         "TMA WS input template is not initialized");
  }

  mInterface->is_profiler = true;
  if (mGemmToProfile == GemmToProfile::GEMM_1) {
    mInterface->gemm1(
        input,                                                                       //
        output,                                                                      //
        intermediate,                                                                //
        expert_first_token_offset,                                                   //
        tma_ws_input_template,                                                       //
        upper_weights_sel,                                                           //
        lower_weights_sel,                                                           //
        bias,                                                                        //
        expert_first_token_offset + num_experts_per_node,                            //
        original_num_tokens,                                                         //
        expanded_num_tokens,                                                         //
        mExpertHiddenSize,                                                           //
        mExpertInterSize,                                                            //
        num_experts_per_node,                                                        //
        ActivationParams(mActivationType, swiglu_alpha, swiglu_beta, swiglu_limit),  //
        /*bias_is_broadcast*/ true,                                                  //
        stream,                                                                      //
        tactic_to_run,                                                               //
        num_active_experts_per_node,                                                 //
        active_expert_global_ids,                                                    //
        enable_pdl);                                                                 //
  } else {
    TLLM_CHECK(mGemmToProfile == GemmToProfile::GEMM_2);
    mInterface->gemm2(input,                                           //
                      intermediate,                                    //
                      output,                                          //
                      expert_first_token_offset,                       //
                      tma_ws_input_template,                           //
                      upper_weights_sel,                               //
                      lower_weights_sel,                               //
                      bias,                                            //
                      token_topk_unpermuted_scales,                    //
                      token_topk_permuted_scales,                      //
                      unpermuted_row_to_permuted_row,                  //
                      permuted_row_to_unpermuted_row,                  //
                      token_selected_experts,                          //
                      expert_first_token_offset + mNumExpertsPerNode,  //
                      original_num_tokens,                             //
                      expanded_num_tokens,                             //
                      mExpertHiddenSize,                               //
                      mExpertUnpaddedHiddenSize,                       //
                      mExpertInterSize,                                //
                      num_experts_per_node,                            //
                      mK,                                              //
                      stream,                                          //
                      mParallelismConfig,                              //
                      mEnableAlltoall,                                 //
                      tactic_to_run,                                   //
                      num_active_experts_per_node,                     //
                      active_expert_global_ids,                        //
                      enable_pdl);                                     //
  }
  mInterface->is_profiler = false;

  sync_check_cuda_error(stream);
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels
