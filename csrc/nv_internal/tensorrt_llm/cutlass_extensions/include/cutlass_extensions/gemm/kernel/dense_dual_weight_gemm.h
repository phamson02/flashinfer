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

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped_dual_weight.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/permute.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool SplitKSerial>
struct GemmDualWeight {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B_upper;
    typename Mma::IteratorB::TensorRef ref_B_upper;
    typename Mma::IteratorB::Params params_B_lower;
    typename Mma::IteratorB::TensorRef ref_B_lower;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int* semaphore;
    int gemm_k_size;
    int const* gather_A_indices;
    int const* gather_B_indices;
    int const* scatter_D_indices;

    CUTLASS_HOST_DEVICE
    Params()
        : swizzle_log_tile(0),
          semaphore(nullptr),
          gemm_k_size(0),
          gather_A_indices(nullptr),
          gather_B_indices(nullptr),
          scatter_D_indices(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(cutlass::gemm::GemmCoord const& problem_size_,
           cutlass::gemm::GemmCoord const& grid_tiled_shape_,
           typename Mma::IteratorA::TensorRef ref_A_,
           typename Mma::IteratorB::TensorRef ref_B_upper_,
           typename Mma::IteratorB::TensorRef ref_B_lower_,
           typename Epilogue::OutputTileIterator::TensorRef ref_C_,
           typename Epilogue::OutputTileIterator::TensorRef ref_D_,
           typename OutputOp::Params output_op_ = typename OutputOp::Params(),
           int* workspace = nullptr, int const* gather_A_indices_ = nullptr,
           int const* gather_B_indices_ = nullptr,
           int const* scatter_D_indices_ = nullptr)
        : problem_size(problem_size_),
          grid_tiled_shape(grid_tiled_shape_),
          swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape_)),
          params_A(ref_A_.layout()),
          ref_A(ref_A_),
          params_B_upper(ref_B_upper_.layout()),
          ref_B_upper(ref_B_upper_),
          params_B_lower(ref_B_lower_.layout()),
          ref_B_lower(ref_B_lower_),
          params_C(ref_C_.layout()),
          ref_C(ref_C_),
          params_D(ref_D_.layout()),
          ref_D(ref_D_),
          output_op(output_op_),
          semaphore(workspace),
          gather_A_indices(gather_A_indices_),
          gather_B_indices(gather_B_indices_),
          scatter_D_indices(scatter_D_indices_) {
      int total_gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations =
          (total_gemm_k_iterations + grid_tiled_shape.k() - 1) /
          grid_tiled_shape.k();
      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
    }
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_HOST_DEVICE
  GemmDualWeight() {}

  CUTLASS_HOST_DEVICE
  static Status can_implement(
      cutlass::gemm::GemmCoord const& problem_size,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B_upper,
      typename Mma::IteratorB::TensorRef ref_B_lower,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D) {
    static int const kAlignmentA =
        (platform::is_same<typename Mma::IteratorA::Layout,
                           layout::ColumnMajorInterleaved<32>>::value)
            ? 32
            : (platform::is_same<typename Mma::IteratorA::Layout,
                                 layout::ColumnMajorInterleaved<64>>::value)
                  ? 64
                  : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =
        (platform::is_same<typename Mma::IteratorB::Layout,
                           layout::RowMajorInterleaved<32>>::value)
            ? 32
            : (platform::is_same<typename Mma::IteratorB::Layout,
                                 layout::RowMajorInterleaved<64>>::value)
                  ? 64
                  : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC =
        (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                           layout::ColumnMajorInterleaved<32>>::value)
            ? 32
            : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                 layout::ColumnMajorInterleaved<64>>::value)
                  ? 64
                  : Epilogue::OutputTileIterator::kElementsPerAccess;

    if (problem_size.m() < 1 || problem_size.n() < 1 || problem_size.k() < 1) {
      return Status::kErrorInvalidProblem;
    }

    if (problem_size.n() < kAlignmentB) {
      return Status::kErrorInvalidProblem;
    }

    if (!TensorRef_aligned(ref_A, kAlignmentA) ||
        !TensorRef_aligned(ref_B_upper, kAlignmentB) ||
        !TensorRef_aligned(ref_B_lower, kAlignmentB) ||
        !TensorRef_aligned(ref_C, kAlignmentC) ||
        !TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
        threadblock_tile_offset.k() * params.gemm_k_size,
        threadblock_tile_offset.n() * Mma::Shape::kN,
    };

    int problem_size_k =
        min(params.problem_size.k(),
            (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
    int gemm_k_iterations =
        (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) /
        Mma::Shape::kK;

    int thread_idx = threadIdx.x;

    typename Mma::IteratorA iterator_A(
        params.params_A, params.ref_A.data(),
        {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_A,
        params.gather_A_indices);

    typename Mma::IteratorB iterator_B_upper(
        params.params_B_upper, params.ref_B_upper.data(),
        {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_B,
        params.gather_B_indices);

    typename Mma::IteratorB iterator_B_lower(
        params.params_B_lower, params.ref_B_lower.data(),
        {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_B,
        params.gather_B_indices);

    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B_upper,
          iterator_B_lower, accumulators);
    }

    OutputOp output_op(params.output_op);

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                   threadblock_tile_offset.n() * Mma::Shape::kN);

    int block_idx = threadblock_tile_offset.m() +
                    threadblock_tile_offset.n() * params.grid_tiled_shape.m();
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      semaphore.fetch();
      output_op.set_k_partition(threadblock_tile_offset.k(),
                                params.grid_tiled_shape.k());
    }

    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C, params.ref_C.data(), params.problem_size.mn(),
        thread_idx, threadblock_offset, params.scatter_D_indices);

    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D, params.ref_D.data(), params.problem_size.mn(),
        thread_idx, threadblock_offset, params.scatter_D_indices);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }
      semaphore.wait(threadblock_tile_offset.k());
    }

    epilogue(output_op, iterator_D, accumulators, iterator_C);

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      int lock = 0;
      if (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1) {
        lock = threadblock_tile_offset.k() + 1;
      }
      semaphore.release(lock);
    }
  }
};

}  // namespace kernel

namespace device {

template <typename ElementA_, typename LayoutA_, typename ElementB_,
          typename LayoutB_, typename ElementC_, typename LayoutC_,
          typename ElementAccumulator_ = ElementC_,
          typename OperatorClass_ = arch::OpClassSimt,
          typename ArchTag_ = arch::Sm70,
          typename ThreadblockShape_ =
              typename DefaultGemmConfiguration<
                  OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                  ElementAccumulator_>::ThreadblockShape,
          typename WarpShape_ =
              typename DefaultGemmConfiguration<
                  OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                  ElementAccumulator_>::WarpShape,
          typename InstructionShape_ =
              typename DefaultGemmConfiguration<
                  OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                  ElementAccumulator_>::InstructionShape,
          typename EpilogueOutputOp_ =
              typename DefaultGemmConfiguration<
                  OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                  ElementAccumulator_>::EpilogueOutputOp,
          typename ThreadblockSwizzle_ =
              typename threadblock::GemmIdentityThreadblockSwizzle<>,
          int Stages = DefaultGemmConfiguration<
              OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
              ElementAccumulator_>::kStages,
          int AlignmentA =
              DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_,
                                       ElementB_, ElementC_,
                                       ElementAccumulator_>::kAlignmentA,
          int AlignmentB =
              DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_,
                                       ElementB_, ElementC_,
                                       ElementAccumulator_>::kAlignmentB,
          bool SplitKSerial = false,
          typename Operator_ = typename DefaultGemmConfiguration<
              OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
              ElementAccumulator_>::Operator,
          /// When true, use k-block interleaved FP8->FP16 reconstruction.
          bool kKBlockInterleaved = false,
          /// When true and ElementB is float_e5m2_t, use truncation reconstruction.
          bool kTruncateE5M2 = false,
          /// When true and ElementB is float_e4m3_t, use FastNumericArrayConverter reconstruction.
          bool kFastE4M3 = false>
class DualWeightGemm {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;

  using DefaultKernel =
      typename kernel::DefaultGemmDualWeight<
          ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
          ElementC, LayoutC, ElementAccumulator, ThreadblockShape, WarpShape,
          InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, kStages,
          kSplitKSerial, Operator, SharedMemoryClearOption::kNone, false, false,
          false, layout::NoPermute, layout::NoPermute, layout::NoPermute,
          kKBlockInterleaved, kTruncateE5M2, kFastE4M3>::GemmKernel;

  using GemmKernel = kernel::GemmDualWeight<typename DefaultKernel::Mma,
                                            typename DefaultKernel::Epilogue,
                                            ThreadblockSwizzle, kSplitKSerial>;

  struct Arguments {
    GemmCoord problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B_upper;
    TensorRefB ref_B_lower;
    TensorRefC ref_C;
    TensorRefD ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;

    CUTLASS_HOST_DEVICE
    Arguments() : problem_size(0, 0, 0), split_k_slices(1) {}

    CUTLASS_HOST_DEVICE
    Arguments(GemmCoord problem_size_, TensorRefA ref_A_, TensorRefB ref_B_upper_,
              TensorRefB ref_B_lower_, TensorRefC ref_C_, TensorRefD ref_D_,
              typename EpilogueOutputOp::Params epilogue_ =
                  typename EpilogueOutputOp::Params(),
              int split_k_slices_ = 1)
        : problem_size(problem_size_),
          ref_A(ref_A_),
          ref_B_upper(ref_B_upper_),
          ref_B_lower(ref_B_lower_),
          ref_C(ref_C_),
          ref_D(ref_D_),
          epilogue(epilogue_),
          split_k_slices(split_k_slices_) {}
  };

 private:
  typename GemmKernel::Params params_;

 public:
  DualWeightGemm() {}

  static Status can_implement(Arguments const& args) {
    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }
    return GemmKernel::can_implement(args.problem_size, args.ref_A.non_const_ref(),
                                     args.ref_B_upper.non_const_ref(),
                                     args.ref_B_lower.non_const_ref(),
                                     args.ref_C.non_const_ref(), args.ref_D);
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t bytes = 0;
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord tiled_shape =
        threadblock_swizzle.get_tiled_shape(
            args.problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            args.split_k_slices);

    if (kSplitKSerial && args.split_k_slices > 1) {
      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }
    return bytes;
  }

  Status initialize(Arguments const& args, void* workspace = nullptr,
                    cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord grid_shape =
        threadblock_swizzle.get_tiled_shape(
            args.problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            args.split_k_slices);

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        size_t bytes = get_workspace_size(args);
        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);
        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    } else if (args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    params_ = typename GemmKernel::Params(
        args.problem_size, grid_shape, args.ref_A.non_const_ref(),
        args.ref_B_upper.non_const_ref(), args.ref_B_lower.non_const_ref(),
        args.ref_C.non_const_ref(), args.ref_D, args.epilogue,
        static_cast<int*>(workspace));
    return Status::kSuccess;
  }

  Status update(Arguments const& args, void* workspace = nullptr) {
    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B_upper.reset(args.ref_B_upper.non_const_ref().data());
    params_.ref_B_lower.reset(args.ref_B_lower.non_const_ref().data());
    params_.ref_C.reset(args.ref_C.non_const_ref().data());
    params_.ref_D.reset(args.ref_D.data());
    params_.output_op = args.epilogue;
    params_.semaphore = static_cast<int*>(workspace);
    return Status::kSuccess;
  }

  Status run(cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(
          Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::arch::synclog_setup();
    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
    cudaError_t result = cudaGetLastError();
    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  Status operator()(Arguments const& args, void* workspace = nullptr,
                    cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (status == Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }
};

}  // namespace device
}  // namespace gemm
}  // namespace cutlass
