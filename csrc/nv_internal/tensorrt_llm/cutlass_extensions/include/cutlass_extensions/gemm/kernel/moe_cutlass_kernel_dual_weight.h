/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*! \file
    \brief
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/trace.h"
#include "cutlass_extensions/gemm/kernel/gemm_moe_problem_visitor.h"
#include "cutlass_extensions/tile_interleaved_layout.h"
#include "cutlass_extensions/weight_only_quant_op.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_tma_warp_specialized_traits.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,                 ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          typename KernelArch,  ///! The Architecture this kernel is compiled for. Used since SIMT
                                /// kernels lose top-level
                                /// arch.
          GroupScheduleMode GroupScheduleMode_  ///! Type of scheduling to perform
          >
struct MoeFCGemmDualWeight {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = false;

  // Optional transpose
  using MapArguments = kernel::detail::MapArguments<
      typename Mma::IteratorA::Element, typename Mma::IteratorA::Layout, Mma::kTransformA,
      Mma::IteratorA::AccessType::kElements, typename Mma::IteratorB::Element,
      typename Mma::IteratorB::Layout, Mma::kTransformB, Mma::IteratorB::AccessType::kElements,
      typename Mma::LayoutC, kTransposed>;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  static_assert(!kTransposed, "Transpose problem not supported");
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;
  using ElementScale = ElementC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmMoeProblemVisitor<ThreadblockShape, kGroupScheduleMode, kThreadCount,
                                               kThreadCount, kTransposed>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    int problem_count;
    int threadblock_count;
    int group_size;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B_upper;
    ElementB* ptr_B_lower;
    ElementScale* weight_scales;
    ElementScale* weight_zeros;
    ElementC* ptr_C;
    ElementC* ptr_D;
    bool C_is_broadcast;

    int64_t const* total_tokens_including_expert;
    int64_t gemm_n;
    int64_t gemm_k;

    // Only used by device-level operator
    GemmCoord* host_problem_sizes;

    // For gather+scatter operations, default nullptr
    int const* gather_A_indices{};
    int const* gather_B_indices{};
    int const* scatter_D_indices{};

    // Included so we can use Gemm Universal
    int batch_stride_D = 0;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments()
        : problem_count(0),
          threadblock_count(0),
          ptr_A(nullptr),
          ptr_B_upper(nullptr),
          ptr_B_lower(nullptr),
          weight_scales(nullptr),
          weight_zeros(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          total_tokens_including_expert(nullptr),
          gemm_n(0),
          gemm_k(0),
          host_problem_sizes(nullptr),
          C_is_broadcast{true},
          gather_A_indices(nullptr),
          gather_B_indices(nullptr),
          scatter_D_indices(nullptr),
          batch_stride_D(0) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(int problem_count, int threadblock_count, int group_size,
              typename EpilogueOutputOp::Params output_op, ElementA const* ptr_A,
              ElementB const* ptr_B_upper, ElementB const* ptr_B_lower,
              ElementScale const* weight_scales, ElementScale const* weight_zeros,
              ElementC const* ptr_C, bool C_is_broadcast, ElementC* ptr_D,
              int64_t const* total_tokens_including_expert, int64_t gemm_n, int64_t gemm_k,
              GemmCoord* host_problem_sizes = nullptr)
        : problem_count(problem_count),
          threadblock_count(threadblock_count),
          group_size(group_size),
          output_op(output_op),
          ptr_A(const_cast<ElementA*>(ptr_A)),
          ptr_B_upper(const_cast<ElementB*>(ptr_B_upper)),
          ptr_B_lower(const_cast<ElementB*>(ptr_B_lower)),
          weight_scales(const_cast<ElementScale*>(weight_scales)),
          weight_zeros(const_cast<ElementScale*>(weight_zeros)),
          ptr_C(const_cast<ElementC*>(ptr_C)),
          C_is_broadcast{C_is_broadcast},
          ptr_D(ptr_D),
          total_tokens_including_expert(total_tokens_including_expert),
          gemm_n(gemm_n),
          gemm_k(gemm_k),
          host_problem_sizes(nullptr) {
      this->gather_A_indices = nullptr;
      this->gather_B_indices = nullptr;
      this->scatter_D_indices = nullptr;
      this->batch_stride_D = 0;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;
    int group_size;
    bool C_is_broadcast;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B_upper;
    ElementB* ptr_B_lower;
    ElementScale* weight_scales;
    ElementScale* weight_zeros;
    ElementC* ptr_C;
    ElementC* ptr_D;

    // For gather+scatter operations, default nullptr.
    int const* gather_A_indices;
    int const* gather_B_indices;
    int const* scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : ptr_A(nullptr),
          ptr_B_upper(nullptr),
          ptr_B_lower(nullptr),
          weight_scales(nullptr),
          weight_zeros(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          C_is_broadcast(true),
          gather_A_indices(nullptr),
          gather_B_indices(nullptr),
          scatter_D_indices(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
        : problem_visitor(args.total_tokens_including_expert, args.gemm_n, args.gemm_k,
                          args.problem_count, workspace, tile_count),
          threadblock_count(args.threadblock_count),
          group_size(args.group_size),
          output_op(args.output_op),
          ptr_A(args.ptr_A),
          ptr_B_upper(args.ptr_B_upper),
          ptr_B_lower(args.ptr_B_lower),
          weight_scales(args.weight_scales),
          weight_zeros(args.weight_zeros),
          ptr_C(args.ptr_C),
          ptr_D(args.ptr_D),
          C_is_broadcast(args.C_is_broadcast),
          gather_A_indices(args.gather_A_indices),
          gather_B_indices(args.gather_B_indices),
          scatter_D_indices(args.scatter_D_indices) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const& args, void* workspace = nullptr, int tile_count = 0) {
      problem_visitor =
          typename ProblemVisitor::Params(args.total_tokens_including_expert, args.gemm_n,
                                          args.gemm_k, args.problem_count, workspace, tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B_upper = args.ptr_B_upper;
      ptr_B_lower = args.ptr_B_lower;
      weight_scales = args.weight_scales;
      weight_zeros = args.weight_zeros;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      C_is_broadcast = args.C_is_broadcast;
      gather_A_indices = args.gather_A_indices;
      gather_B_indices = args.gather_B_indices;
      scatter_D_indices = args.scatter_D_indices;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename ProblemVisitor::SharedStorage problem_visitor;
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  MoeFCGemmDualWeight() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const& args) {
    // Validate required pointers for dual weight
    if (args.ptr_A == nullptr) {
      CUTLASS_TRACE_HOST("MoeFCGemmDualWeight::can_implement() - ptr_A is nullptr");
      return Status::kInvalid;
    }
    if (args.ptr_B_upper == nullptr) {
      CUTLASS_TRACE_HOST("MoeFCGemmDualWeight::can_implement() - ptr_B_upper is nullptr");
      return Status::kInvalid;
    }
    if (args.ptr_B_lower == nullptr) {
      CUTLASS_TRACE_HOST("MoeFCGemmDualWeight::can_implement() - ptr_B_lower is nullptr");
      return Status::kInvalid;
    }
    if (args.ptr_D == nullptr) {
      CUTLASS_TRACE_HOST("MoeFCGemmDualWeight::can_implement() - ptr_D is nullptr");
      return Status::kInvalid;
    }
    if (args.total_tokens_including_expert == nullptr) {
      CUTLASS_TRACE_HOST(
          "MoeFCGemmDualWeight::can_implement() - total_tokens_including_expert is nullptr");
      return Status::kInvalid;
    }

    // Validate problem dimensions
    if (args.problem_count < 1) {
      CUTLASS_TRACE_HOST("MoeFCGemmDualWeight::can_implement() - problem_count must be >= 1");
      return Status::kInvalid;
    }
    if (args.gemm_n < 1) {
      CUTLASS_TRACE_HOST("MoeFCGemmDualWeight::can_implement() - gemm_n must be >= 1");
      return Status::kInvalid;
    }
    if (args.gemm_k < 1) {
      CUTLASS_TRACE_HOST("MoeFCGemmDualWeight::can_implement() - gemm_k must be >= 1");
      return Status::kInvalid;
    }

    // Validate minimum alignment for gemm_n
    // For dual weight FP8, B iterator access type determines minimum alignment
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    if (args.gemm_n < kAlignmentB) {
      CUTLASS_TRACE_HOST(
          "MoeFCGemmDualWeight::can_implement() - gemm_n is smaller than the B matrix alignment "
          "requirement");
      return Status::kInvalid;
    }

    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(Arguments const& args,
                                         cutlass::gemm::GemmCoord const& grid_tiled_shape) {
    return 0;
  }

  CUTLASS_DEVICE
  void run_kernel_(Params const& params, SharedStorage& shared_storage) {
    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    static constexpr int kInterleave = Mma::IteratorB::Shape::kRow / Mma::Shape::kK;
    static_assert(platform::is_same<LayoutB, layout::RowMajor>::value && kInterleave == 1 ||
                      platform::is_same<LayoutB, layout::ColumnMajor>::value && kInterleave >= 1,
                  "B must be row major/col major OR col major interleaved.");

    //
    // Problem visitor.
    //
    ProblemVisitor problem_visitor(params.problem_visitor, shared_storage.problem_visitor,
                                   blockIdx.x);

    const int64_t gemm_k = params.problem_visitor.gemm_k;
    const int64_t gemm_n = params.problem_visitor.gemm_n;
    int64_t bytes_per_expert_matrix = (gemm_k * gemm_n / 8) * cutlass::sizeof_bits<ElementB>::value;

    // Outer 'persistent' loop to iterate over tiles
    int loop = 0;
    while (problem_visitor.next_tile()) {
      loop++;

      GemmCoord problem_size = problem_visitor.problem_size();
      int32_t problem_idx = problem_visitor.problem_index();
      int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(int(cta_idx / grid_shape.n()) * Mma::Shape::kM,
                                                  int(cta_idx % grid_shape.n()) * Mma::Shape::kN,
                                                  0);

      // Load element pointers. Exchange pointers and strides if working on the transpose
      const int64_t rows_to_jump =
          problem_idx == 0 ? 0 : params.problem_visitor.last_row_for_problem[problem_idx - 1];
      ElementA* ptr_A = reinterpret_cast<ElementA*>(params.ptr_A) + rows_to_jump * gemm_k;
      typename LayoutA::LongIndex ldm_A = gemm_k;

      // Calculate B pointers for this expert
      char* byte_ptr_B_upper = ((char*)params.ptr_B_upper) + problem_idx * bytes_per_expert_matrix;
      ElementB* ptr_B_upper = reinterpret_cast<ElementB*>(byte_ptr_B_upper);

      char* byte_ptr_B_lower = ((char*)params.ptr_B_lower) + problem_idx * bytes_per_expert_matrix;
      ElementB* ptr_B_lower = reinterpret_cast<ElementB*>(byte_ptr_B_lower);

      typename LayoutB::LongIndex ldm_B =
          platform::is_same<layout::RowMajor, LayoutB>::value ? gemm_n : gemm_k * kInterleave;

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
          threadblock_offset.m(),
          0,
      };

      cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n() / kInterleave};

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
          LayoutA(ldm_A), ptr_A, {problem_size.m(), problem_size.k()}, thread_idx, tb_offset_A);

      typename Mma::IteratorB iterator_B_upper(
          LayoutB(ldm_B), ptr_B_upper,
          {problem_size.k() * kInterleave, problem_size.n() / kInterleave}, thread_idx,
          tb_offset_B);

      typename Mma::IteratorB iterator_B_lower(
          LayoutB(ldm_B), ptr_B_lower,
          {problem_size.k() * kInterleave, problem_size.n() / kInterleave}, thread_idx,
          tb_offset_B);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      // We use the custom MmaMultistageDualWeight
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();

      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B_upper, iterator_B_lower,
          accumulators);

      //
      // Epilogue
      //

      ElementC* ptr_C = (params.ptr_C == nullptr)
                            ? nullptr
                            : reinterpret_cast<ElementC*>(params.ptr_C) +
                                  (params.C_is_broadcast ? problem_idx : rows_to_jump) * gemm_n;
      ElementC* ptr_D = reinterpret_cast<ElementC*>(params.ptr_D) + rows_to_jump * gemm_n;

      // lora need to set as layout_C(gemm_n)
      LayoutC layout_C = params.C_is_broadcast ? LayoutC(0) : LayoutC(gemm_n);
      LayoutC layout_D(gemm_n);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(params_C, ptr_C, problem_size.mn(),
                                                       thread_idx, threadblock_offset.mn(),
                                                       params.scatter_D_indices);

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(params_D, ptr_D, problem_size.mn(),
                                                       thread_idx, threadblock_offset.mn(),
                                                       params.scatter_D_indices);

      Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      if constexpr (platform::is_same<
                        EpilogueOutputOp,
                        cutlass::epilogue::thread::LinearCombination<
                            typename EpilogueOutputOp::ElementOutput, EpilogueOutputOp::kCount,
                            typename EpilogueOutputOp::ElementAccumulator,
                            typename EpilogueOutputOp::ElementCompute, EpilogueOutputOp::kScale,
                            EpilogueOutputOp::kRound>>::value) {
        EpilogueOutputOp output_op(params.output_op, problem_idx);
        epilogue(output_op, iterator_D, accumulators, iterator_C);
      } else {
        EpilogueOutputOp output_op(params.output_op);
        epilogue(output_op, iterator_D, accumulators, iterator_C);
      }

      // Next tile
      problem_visitor.advance(gridDim.x);
    }
  }

  template <typename CompilationArch>
  CUTLASS_DEVICE void run_kernel(Params const& params, SharedStorage& shared_storage) {
    if constexpr (platform::is_same<KernelArch, CompilationArch>::value) {
      run_kernel_(params, shared_storage);
    } else {
      CUTLASS_NOT_IMPLEMENTED();
    }
  }

  /*
    To improve compilation speed, we do not compile the device operator if the CUDA_ARCH does not
    correspond to the ArchTag of the cutlass kernel operator.
  */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 890)
    run_kernel<arch::Sm80>(params, shared_storage);
#elif (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 900)
    constexpr bool isFp8 = platform::is_same<ElementA, cutlass::float_e4m3_t>::value ||
                           platform::is_same<ElementA, cutlass::float_e5m2_t>::value;
    if constexpr (isFp8) {
      run_kernel<arch::Sm89>(params, shared_storage);
    } else {  // reuse sm80 kernel for other types, align with dispatchToArch
      run_kernel<arch::Sm80>(params, shared_storage);
    }
#elif (__CUDA_ARCH__ >= 900)
    constexpr bool isFp8 = platform::is_same<ElementA, cutlass::float_e4m3_t>::value ||
                           platform::is_same<ElementA, cutlass::float_e5m2_t>::value;
    if constexpr (isFp8) {
      run_kernel<arch::Sm89>(params, shared_storage);
    } else {  // reuse sm80 kernel for other types, align with dispatchToArch
      run_kernel<arch::Sm80>(params, shared_storage);
    }
#else
    static_assert(false,
                  "Invalid architecture being compiled. Only Ampere+ supported in weight-only "
                  "quantization kernels.");
#endif
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
