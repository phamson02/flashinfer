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
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "common.h"
#include "dual_weight_moe_gemm_kernels.h" 
#include "moe_kernels.h"
#include "tensorrt_llm/common/NvInferRuntime.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels::cutlass_kernels {

class DualWeightMoeFCRunnerInterface {
 public:
  virtual ~DualWeightMoeFCRunnerInterface() = default;
  virtual size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size,
                                  int64_t const inter_size, int const num_experts,
                                  int const experts_per_token, ActivationType activation_type,
                                  MOEParallelismConfig parallelism_config) = 0;
  virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                         std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) = 0;
  virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

  virtual void runMoe(void const* input_activations, int const* token_selected_experts,
                      float const* token_final_scales, void const* fc1_upper_expert_weights,
                      void const* fc1_lower_expert_weights, void const* fc1_expert_biases,
                      ActivationParams fc1_activation_type, void const* fc2_upper_expert_weights,
                      void const* fc2_lower_expert_weights, void const* fc2_expert_biases,
                      int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                      int const num_experts, int const experts_per_token, char* workspace_ptr,
                      void* final_output, int* unpermuted_row_to_permuted_row,
                      MOEParallelismConfig parallelism_config, bool const enable_alltoall,
                      cudaStream_t stream) = 0;

  // Aliases for profiling the gemms
  virtual void gemm1(void const* const input, void* const output, void* const intermediate_result,
                     int64_t const* const expert_first_token_offset,
                     void const* const fc1_expert_upper_weights, void const* const fc1_expert_lower_weights,
                     void const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
                     int64_t const num_rows, int64_t const expanded_num_rows,
                     int64_t const hidden_size, int64_t const inter_size,
                     int const num_experts_per_node, ActivationParams fc1_activation_type,
                     bool bias_is_broadcast, cudaStream_t stream,
                     cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
                     int* active_expert_global_ids) = 0;

  virtual void gemm2(
      void const* const input, void* const gemm_output, void* const final_output,
      int64_t const* const expert_first_token_offset, void const* const fc2_expert_upper_weights,
      void const* const fc2_expert_lower_weights, void const* const fc2_expert_biases,
      float const* const token_topk_unpermuted_scales,
      float const* const token_topk_permuted_scales,
      int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
      int const* const token_selected_experts, int64_t const* const num_valid_tokens_ptr,
      int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size,
      int64_t const inter_size, int const num_experts_per_node, int64_t const experts_per_token,
      cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
      cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
      int* active_expert_global_ids) = 0;

  virtual size_t getGemmWorkspaceSize(int num_experts_per_node) const = 0;

  bool is_profiler = false;
};

// Assumes inputs activations are row major. Weights need to be preprocessed by
// th_op/weight_quantize.cc . Nested in a class to avoid multiple calls to cudaGetDeviceProperties
// as this call can be expensive. Avoid making several duplicates of this class.
template <typename T,                         /*The type used for activations*/
          typename WeightType,                /* The type for the MoE weights */
          typename OutputType = T,            /* The type for the MoE final output */
          typename InputType = T,             /* The type for the MoE input */
          typename BackBoneType = OutputType, /* The unquantized backbone data type of the model */
          typename Enable = void>
class DualWeightMoeFCRunner : public DualWeightMoeFCRunnerInterface {
  using ScaleBiasType = BackBoneType;
  using Self = DualWeightMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType>;

  static_assert(std::is_same_v<T, half>, "Dual weight runner only supports fp16 activations");
  static_assert(std::is_same_v<WeightType, __nv_fp8_e4m3>,
                "Dual weight runner only supports fp8_e4m3 weights");

  // This should leave the variable unchanged in any currently supported configuration
  using UnfusedGemmOutputType = BackBoneType;

  // We introduce this as a separate parameter, so that if we ever remove the above condition we can
  // decouple BackBoneType and OutputType easily. For now these are required to be equivalent
  static_assert(std::is_same_v<OutputType, BackBoneType>,
                "Scale and bias types must match OutputType");

 public:
  DualWeightMoeFCRunner();

  ~DualWeightMoeFCRunner() override = default;

  static_assert(std::is_same_v<T, WeightType> || !std::is_same_v<T, float>,
                "Does not support float with quantized weights");

  size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size,
                          int64_t const fc1_output_size, int const num_experts,
                          int const experts_per_token, ActivationType activation_type,
                          MOEParallelismConfig parallelism_config) override;

  void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                 std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) override {
    gemm1_config_ = std::move(gemm1_config);
    gemm2_config_ = std::move(gemm2_config);
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override {
    return moe_gemm_runner_.getConfigs();
  }

  static std::vector<cutlass_extensions::CutlassGemmConfig> getTactics(int sm) {
    using RunnerType = decltype(moe_gemm_runner_);
    return RunnerType::getConfigs(sm);
  }

  void runMoe(void const* input_activations, int const* token_selected_experts,
              float const* token_final_scales, void const* fc1_upper_expert_weights,
              void const* fc1_lower_expert_weights, void const* fc1_expert_biases,
              ActivationParams fc1_activation_type, void const* fc2_upper_expert_weights,
              void const* fc2_lower_expert_weights, void const* fc2_expert_biases,
              int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
              int const num_experts, int const experts_per_token, char* workspace_ptr,
              void* final_output, int* unpermuted_row_to_permuted_row,
              MOEParallelismConfig parallelism_config, bool const enable_alltoall,
              cudaStream_t stream) override;

  // We make these GEMM1 & GEMM2 static because they need to be stateless for the profiler to work
  static void gemm1(DualWeightMoeGemmRunner<T, WeightType, T, ScaleBiasType>& gemm_runner,
                    T const* const input, T* const output, void* const intermediate_result,
                    int64_t const* const expert_first_token_offset,
                    WeightType const* const fc1_upper_expert_weights,
                    WeightType const* const fc1_lower_expert_weights,
                    ScaleBiasType const* const fc1_expert_biases,
                    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
                    int64_t const expanded_num_rows, int64_t const hidden_size,
                    int64_t const inter_size, int const num_experts_per_node,
                    ActivationParams fc1_activation_type, bool bias_is_broadcast,
                    cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config,
                    int* num_active_experts_per, int* active_expert_global_ids);

  static void gemm2(
      DualWeightMoeGemmRunner<T, WeightType, T, ScaleBiasType>& gemm_runner, T const* const input,
      void* const gemm_output, OutputType* const final_output,
      int64_t const* const expert_first_token_offset,
      WeightType const* const fc2_expert_upper_weights,
      WeightType const* const fc2_expert_lower_weights,
      ScaleBiasType const* const fc2_expert_biases, float const* const token_topk_unpermuted_scales,
      float const* const token_topk_permuted_scales,
      int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
      int const* const token_selected_experts, int64_t const* const num_valid_tokens_ptr,
      int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size,
      int64_t const inter_size, int const num_experts_per_node, int64_t const experts_per_token,
      cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
      cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
      int* active_expert_global_ids);

  // Overrides to allow us to forward on to the internal functions with the pointers using the
  // correct type
  void gemm1(void const* const input, void* const output, void* const intermediate_result,
             int64_t const* const expert_first_token_offset,
             void const* const fc1_expert_upper_weights, void const* const fc1_expert_lower_weights,
             void const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
             int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size,
             int64_t const inter_size, int const num_experts_per_node,
             ActivationParams fc1_activation_type, bool bias_is_broadcast, cudaStream_t stream,
             cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
             int* active_expert_global_ids) override {
    return Self::gemm1(moe_gemm_runner_, static_cast<T const*>(input), static_cast<T*>(output),
                       intermediate_result, expert_first_token_offset,
                       static_cast<WeightType const*>(fc1_expert_upper_weights),
                       static_cast<WeightType const*>(fc1_expert_lower_weights),
                       static_cast<ScaleBiasType const*>(fc1_expert_biases), num_valid_tokens_ptr,
                       num_rows, expanded_num_rows, hidden_size, inter_size, num_experts_per_node,
                       fc1_activation_type, bias_is_broadcast, stream, config,
                       num_active_experts_per, active_expert_global_ids);
  }

  void gemm2(void const* const input, void* const gemm_output, void* const final_output,
             int64_t const* const expert_first_token_offset,
             void const* const fc2_expert_upper_weights, void const* const fc2_expert_lower_weights,
             void const* const fc2_expert_biases, float const* const token_topk_unpermuted_scales,
             float const* const token_topk_permuted_scales,
             int const* const unpermuted_row_to_permuted_row,
             int const* permuted_row_to_unpermuted_row, int const* const token_selected_experts,
             int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
             int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
             int const num_experts_per_node, int64_t const experts_per_token, cudaStream_t stream,
             MOEParallelismConfig parallelism_config, bool const enable_alltoall,
             cutlass_extensions::CutlassGemmConfig config, int* num_active_experts_per,
             int* active_expert_global_ids) override {
    return Self::gemm2(
        moe_gemm_runner_, static_cast<T const*>(input), gemm_output,
        static_cast<OutputType*>(final_output), expert_first_token_offset,
        static_cast<WeightType const*>(fc2_expert_upper_weights),
        static_cast<WeightType const*>(fc2_expert_lower_weights),
        static_cast<ScaleBiasType const*>(fc2_expert_biases), token_topk_unpermuted_scales,
        token_topk_permuted_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size,
        inter_size, num_experts_per_node, experts_per_token, stream, parallelism_config,
        enable_alltoall, config, num_active_experts_per, active_expert_global_ids);
  }

  size_t getGemmWorkspaceSize(int num_experts_per_node) const override {
    return moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);
  }

 private:
  std::map<std::string, std::pair<size_t, size_t>> getWorkspaceDeviceBufferSizes(
      int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
      int const num_experts_per_node, int const experts_per_token, ActivationType activation_type);
  void configureWsPtrs(char* ws_ptr, int64_t const num_rows, int64_t const hidden_size,
                       int64_t const inter_size, int const num_experts_per_node,
                       int const experts_per_token, ActivationType activation_type,
                       MOEParallelismConfig parallelism_config);

 private:
  bool mayHaveDifferentGEMMOutputType() const { return false; }

  bool mayHaveFinalizeFused() const { return false; }

  std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config_;
  std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config_;

  DualWeightMoeGemmRunner<T, WeightType, T, ScaleBiasType> moe_gemm_runner_;

  // Pointers
  int* permuted_row_to_unpermuted_row_{};
  int* permuted_token_selected_experts_{};
  int* blocked_expert_counts_{};
  int* blocked_expert_counts_cumsum_{};
  int* blocked_row_to_unpermuted_row_{};
  T* permuted_data_{};
  float* permuted_token_final_scales_{};

  int64_t* expert_first_token_offset_{};

  void* glu_inter_result_{};
  void* fc2_result_{};
  T* fc1_result_{};
};

struct DualWeightGemmProfilerBackend {
  public:
    using Config = cutlass_extensions::CutlassGemmConfig;
    enum class GemmToProfile { Undefined = 0, GEMM_1, GEMM_2 };


  void init(DualWeightMoeFCRunnerInterface& runner, GemmToProfile gemm_to_profile,
            nvinfer1::DataType dtype, nvinfer1::DataType wtype, nvinfer1::DataType otype,
            int num_experts, int k, int64_t hidden_size, int64_t inter_size, 
            ActivationType activation_type, bool bias,
            bool need_weights, MOEParallelismConfig parallelism_config,
            bool const enable_alltoall) {
    mInterface = &runner;
    mGemmToProfile = gemm_to_profile;
    mDType = dtype;
    mWType = wtype;
    mOType = otype;
    mNumExperts = num_experts;
    mNumExpertsPerNode = num_experts / parallelism_config.ep_size;
    mK = k;
    mExpertHiddenSize = hidden_size;
    mExpertInterSize = inter_size;  // Already divided by tp_size
    mActivationType = activation_type;
    mBias = bias;
    mNeedWeights = need_weights;
    mParallelismConfig = parallelism_config;
    mEnableAlltoall = enable_alltoall;
    mSM = common::getSMVersion();
  }

  void prepare(int num_tokens, char* workspace, void const* upper_expert_weights,
               void const* lower_expert_weights,
               cudaStream_t stream);

  std::map<std::string, std::pair<size_t, size_t>> getProfilerWorkspaces(int maxM);
  size_t getWorkspaceSize(int maxM);

  void runProfiler(int num_tokens, Config const& tactic, char* workspace_ptr_char,
                   void const* upper_expert_weights, void const* lower_expert_weights,
                   cudaStream_t const& stream);

  DualWeightMoeFCRunnerInterface* mInterface;

  GemmToProfile mGemmToProfile = GemmToProfile::Undefined;
  std::vector<Config> mAllTacticsSaved;
  int mSM{};
  int64_t mNumExperts{};
  int64_t mNumExpertsPerNode{};
  int64_t mK{};
  int64_t mExpertHiddenSize{};
  int64_t mExpertInterSize{};
  int64_t mGroupSize{};
  ActivationType mActivationType{};
  MOEParallelismConfig mParallelismConfig{};
  bool mEnableAlltoall = false;

  int mSampleIndex = 0;

  nvinfer1::DataType mDType{};
  nvinfer1::DataType mWType{};
  nvinfer1::DataType mOType{};

  // This will be a unique value for every iteration of warmup and actual bench
  constexpr static int64_t NUM_ROUTING_SAMPLES = 16;

  bool mBias{};
  bool mNeedWeights{};

 private:
  void prepareRouting(int num_tokens, char* workspace, cudaStream_t stream);
};

// Populates a buffer with random values for use with MOE benchmarking
void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::cutlass_kernels
