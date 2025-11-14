#pragma once
#include "cutlass_extensions/gemm/kernel/dual_weight_fused_moe_kernel_traits.cuh"

namespace dual_weight_fused_moe {

// Shuffle fragment data within warp to match the mma.sync operand layout.
// This is required when loading narrow types (8-bit) that will be used with
// wider MMA instructions (16-bit). The ldmatrix loads data in the narrow type's
// thread-value layout, but mma.sync expects the wide type's layout.
// This shuffle redistributes data between threads to match the expected layout.
//
// Based on CUTLASS FragmentShuffler for Operand B with 2:1 ratio (16b MMA, 8b load).
//
// Shuffle two fragments together (upper and lower) for paired reconstruction.
// This variant shuffles both fragments in lockstep, which is needed when
// upper and lower halves need to maintain their correspondence after shuffle.
template <class TensorType1, class TensorType2>
CUTE_DEVICE void shuffle_fragment_B_paired(TensorType1&& tensor_upper, TensorType2&& tensor_lower) {
  using namespace cute;

  // Get lane ID within warp
  int lane_id = threadIdx.x % 32;

  // Compute shuffle deltas based on lane position
  int delta_up = (lane_id & 1) + ((lane_id & 2) >> 1);
  int delta_down = 2 - delta_up;
  int odd_even_lane_id = lane_id & 1;

  // Byte selectors for reordering
  uint32_t constexpr kSelectBytesEvenThread = 0x5410;
  uint32_t constexpr kSelectBytesOddThread = 0x7632;
  uint32_t byte_selector =
      odd_even_lane_id * kSelectBytesOddThread + (1 - odd_even_lane_id) * kSelectBytesEvenThread;

  // Recast tensors to uint32_t
  auto t1 = recast<uint32_t>(tensor_upper);
  auto t2 = recast<uint32_t>(tensor_lower);
  int const num_words = size(t1);

  CUTE_UNROLL
  for (int i = 0; i < num_words; i++) {
    uint32_t src1 = t1(i);
    uint32_t src2 = t2(i);

    // Shuffle both fragments in lockstep
    uint32_t tmp1_up = __shfl_up_sync(0xFFFFFFFF, src1, delta_up);
    uint32_t tmp1_down = __shfl_down_sync(0xFFFFFFFF, src1, delta_down);
    uint32_t tmp2_up = __shfl_up_sync(0xFFFFFFFF, src2, delta_up);
    uint32_t tmp2_down = __shfl_down_sync(0xFFFFFFFF, src2, delta_down);

    // Reorder data within 32-bit words
    t1(i) = __byte_perm(tmp1_up, tmp1_down, byte_selector);
    t2(i) = __byte_perm(tmp2_up, tmp2_down, byte_selector);
  }
}

// Reconstruct fp16 weights from dual fp8 buffers (upper/lower packed).
// Interleaves writes: even i -> tensor_out_0, odd i -> tensor_out_1
// This matches the expected MMA fragment layout where consecutive packed elements
// should alternate between the two output slices (k=0 and k=1).
template <typename T1, typename T2, typename T3, typename T4>
CUTE_DEVICE constexpr void reconstruct(T1 const& tensor_upper, T2 const& tensor_lower,
                                       T3 tensor_out_0,  // Logical 2*k
                                       T4 tensor_out_1)  // Logical 2*k + 1
{
  using namespace cute;

  int B = 4;
  int SZ = size(recast<float_e4m3_t>(tensor_upper));

  Tensor t1 = recast<uint32_t>(tensor_upper);
  Tensor t2 = recast<uint32_t>(tensor_lower);
  Tensor t3_0 = recast<uint32_t>(tensor_out_0);  // For even i
  Tensor t3_1 = recast<uint32_t>(tensor_out_1);  // For odd i

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    uint32_t s = a & 0x80808080;
    uint32_t sub = (b & 0x80808080) >> 7;
    t1(i) = (((a - sub) >> 1) & 0x3f3f3f3f) | s;
    uint32_t c = __byte_perm(t1(i), t2(i), 0x1504);
    uint32_t d = __byte_perm(t1(i), t2(i), 0x3726);

    // Interleave: even i goes to tensor_out_0, odd i goes to tensor_out_1
    int pos = (i / 2) * 2;  // Position within target tensor (0, 2, 4, 6)
    if (i % 2 == 0) {
      t3_0(pos) = c;
      t3_0(pos + 1) = d;
    } else {
      t3_1(pos) = c;
      t3_1(pos + 1) = d;
    }

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   auto c_vals = __half22float2(*reinterpret_cast<__half2 const*>(&c));
    //   auto d_vals = __half22float2(*reinterpret_cast<__half2 const*>(&d));
    //   printf("t3[%d]: c=(%f, %f) d=(%f, %f)\n", i, c_vals.x, c_vals.y, d_vals.x, d_vals.y);
    // }
  }
  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("\n");
  // }
}

template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int TileM_,
          int TileN_, int TileK_, int Stages_, Activation_Type activation_type_,
          typename Enable = void>
struct DualWeight_Fused_Moe_Kernel_routine_sm80;

template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int TileM_,
          int TileN_, int TileK_, int Stages_, Activation_Type activation_type_>
struct DualWeight_Fused_Moe_Kernel_routine_sm80<
    ElementInput_, ElementWeight_, ElementOutput_, TileM_, TileN_, TileK_, Stages_,
    activation_type_, std::enable_if_t<isGateActivation(activation_type_)>> {
  using KT =
      DualWeight_Fused_Moe_Kernel_traits_sm80<ElementInput_, ElementWeight_, ElementOutput_, TileM_,
                                              TileN_, TileK_, Stages_, activation_type_>;
  using Params = Routine_Params<ElementInput_, ElementWeight_, ElementOutput_>;

  CUTE_DEVICE auto gmem_tensor_init(int const problem_index, int const gemm_m,
                                    Params const& params) {
    using X = cute::Underscore;

    int const M = gemm_m;
    int const N1 = params.gemm_n;
    int const K1 = params.gemm_k;
    bool const bias_is_broadcast = params.bias_is_broadcast;

    size_t const problem_jump = problem_index;
    size_t const row_jump =
        ((problem_index == 0) ? 0 : params.total_tokens_including_expert[problem_index - 1]);
    typename KT::ElementInput const* ptr_input_ = params.ptr_input + row_jump * K1;
    typename KT::ElementWeight const* ptr_fc1_gate_upper_ =
        params.ptr_fc1_upper + (2 * problem_jump + 1) * N1 * K1;
    typename KT::ElementWeight const* ptr_fc1_upper_ =
        params.ptr_fc1_upper + 2 * problem_jump * N1 * K1;
    typename KT::ElementWeight const* ptr_fc1_gate_lower_ =
        params.ptr_fc1_lower + (2 * problem_jump + 1) * N1 * K1;
    typename KT::ElementWeight const* ptr_fc1_lower_ =
        params.ptr_fc1_lower + 2 * problem_jump * N1 * K1;
    typename KT::ElementInput const* ptr_bias_ =
        (params.ptr_bias == nullptr) ? nullptr
                                     : (bias_is_broadcast ? params.ptr_bias + 2 * problem_jump * N1
                                                          : params.ptr_bias + 2 * row_jump * N1);
    typename KT::ElementInput const* ptr_bias_gate_ =
        (params.ptr_bias == nullptr)
            ? nullptr
            : (bias_is_broadcast ? params.ptr_bias + (2 * problem_jump + 1) * N1
                                 : params.ptr_bias + (2 * row_jump + 1) * N1);
    typename KT::ElementOutput* ptr_output_ = params.ptr_output + row_jump * N1;

    cute::Tensor mInput_mk = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_input_)),
        cute::make_shape(M, K1), cute::make_stride(K1, cute::_1{}));

    cute::Tensor mfc1_gate_upper_nk = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<uint16_t const*>(ptr_fc1_gate_upper_)),
        cute::make_shape(N1, K1 / 2), cute::make_stride(K1 / 2, cute::_1{}));

    cute::Tensor mfc1_upper_nk =
        cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint16_t const*>(ptr_fc1_upper_)),
                          cute::make_shape(N1, K1 / 2), cute::make_stride(K1 / 2, cute::_1{}));

    cute::Tensor mfc1_gate_lower_nk = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<uint16_t const*>(ptr_fc1_gate_lower_)),
        cute::make_shape(N1, K1 / 2), cute::make_stride(K1 / 2, cute::_1{}));

    cute::Tensor mfc1_lower_nk =
        cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint16_t const*>(ptr_fc1_lower_)),
                          cute::make_shape(N1, K1 / 2), cute::make_stride(K1 / 2, cute::_1{}));

    cute::Tensor mBias_mn = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_bias_)),
        cute::make_shape(M, N1),
        cute::make_stride(bias_is_broadcast ? cute::Int<0>{} : N1 * 2,
                          cute::_1{}));  // trick: bias shape is [1, N], but we use [M, N].
    cute::Tensor mBias_gate_mn = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_bias_gate_)),
        cute::make_shape(M, N1),
        cute::make_stride(bias_is_broadcast ? cute::Int<0>{} : N1 * 2,
                          cute::_1{}));  // trick: bias shape is [1, N], but we use [M, N].

    cute::Tensor mOutput_mn =
        cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementInput*>(ptr_output_)),
                          cute::make_shape(M, N1), cute::make_stride(N1, cute::_1{}));

    // Tiler (BLK_M, BLK_K, BLK_K) = (16, 128, 64)
    cute::Tensor gInput_mk = cute::local_tile(
        mInput_mk, typename KT::TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, X, cute::_1>{});  // (BLK_M, BLK_K, m, k) // (16, 128, 1, 2)

    // Packed Tiler (BLK_M, BLK_N, BLK_K/2) = (16, 128, 32)
    auto packed_tiler = cute::make_shape(cute::size<0>(typename KT::TileShape{}),
                                         cute::size<1>(typename KT::TileShape{}),
                                         cute::size<2>(typename KT::TileShape{}) / cute::Int<2>{});
    cute::Tensor gfc1_gate_upper_nk = cute::local_tile(
        mfc1_gate_upper_nk, packed_tiler, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});  // (BLK_N, BLK_K/2, n, k) // (128, 32, 1, 2)
    cute::Tensor gfc1_upper_nk = cute::local_tile(
        mfc1_upper_nk, packed_tiler, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});  // (BLK_N, BLK_K/2, n, k) // (128, 32, 1, 2)
    cute::Tensor gfc1_gate_lower_nk = cute::local_tile(
        mfc1_gate_lower_nk, packed_tiler, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});  // (BLK_N, BLK_K/2, n, k) // (128, 32, 1, 2)
    cute::Tensor gfc1_lower_nk = cute::local_tile(
        mfc1_lower_nk, packed_tiler, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});  // (BLK_N, BLK_K/2, n, k) // (128, 32, 1, 2)

    cute::Tensor gBias_mn = cute::local_tile(
        mBias_mn, typename KT::TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, cute::_1, X>{});  // (BLK_M, BLK_N, m, n) // (16, 128, 1, 1)

    cute::Tensor gBias_gate_mn = cute::local_tile(
        mBias_gate_mn, typename KT::TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, cute::_1, X>{});  // (BLK_M, BLK_N, m, n) // (16, 128, 1, 1)

    cute::Tensor gOutput_mn = cute::local_tile(
        mOutput_mn, typename KT::TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, cute::_1, X>{});  // (BLK_M, BLK_N, m, n) // (16, 128, 1, 1)

    return cute::make_tuple(gInput_mk, gfc1_gate_upper_nk, gfc1_upper_nk, gfc1_gate_lower_nk,
                            gfc1_lower_nk, gBias_mn, gBias_gate_mn, gOutput_mn);
  }

  CUTE_DEVICE void run_routine(Params const& params, int const problem_index, int const block_m_idx,
                               int const block_n_idx, int const gemm_m) {
    extern __shared__ char smem_[];
    typename KT::SharedStorage& shared_storage =
        *reinterpret_cast<typename KT::SharedStorage*>(smem_);
    int const thread_idx = threadIdx.x;
    bool const bias_is_broadcast = params.bias_is_broadcast;
    // gmem tensor partition ..
    auto [gInput_mk, gfc1_gate_upper_nk, gfc1_upper_nk, gfc1_gate_lower_nk, gfc1_lower_nk, gBias_mn,
          gBias_gate_mn, gOutput_mn] = gmem_tensor_init(problem_index, gemm_m, params);
    int const residue_m = gemm_m - block_m_idx * cute::size<0>(gInput_mk);
    auto const n_tile_count = cute::size<2>(gfc1_gate_upper_nk);  // k_tiles = 1

    // smem tensor ..
    cute::Tensor sInput = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_input.data()),
                                            typename KT::SmemLayoutA{});  // (BLK_M, BLK_K, Stage)
    cute::Tensor sfc1_weight_upper =
        cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_weight_upper.data()),
                          typename KT::SmemLayoutB{});  // (BLK_N, BLK_K, Stage)
    cute::Tensor sfc1_gate_weight_upper =
        cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_gate_weight_upper.data()),
                          typename KT::SmemLayoutB{});  // (BLK_N, BLK_K, Stage)
    cute::Tensor sfc1_weight_lower =
        cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_weight_lower.data()),
                          typename KT::SmemLayoutB{});  // (BLK_N, BLK_K, Stage)
    cute::Tensor sfc1_gate_weight_lower =
        cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_gate_weight_lower.data()),
                          typename KT::SmemLayoutB{});  // (BLK_N, BLK_K, Stage)
    cute::Tensor sO = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_o.data()),
                                        typename KT::SmemLayoutO{});  // (BLK_M, BLK_N)

    // (1) first step, get the fc1_res and fc1_gate

    // (1.1) get partition for gmem -> smem
    cute::Tensor gInput = gInput_mk(cute::_, cute::_, block_m_idx,
                                    cute::_);  // (BLK_M, BLK_K, k) // (16,64,2):(128,1,64)
    cute::Tensor gfc1_upper = gfc1_upper_nk(cute::_, cute::_, block_n_idx,
                                            cute::_);  // (BLK_N, BLK_K, k) // (128,32,2):(64,1,32)
    cute::Tensor gfc1g_upper =
        gfc1_gate_upper_nk(cute::_, cute::_, block_n_idx, cute::_);  // (BLK_N, BLK_K, k)
    cute::Tensor gfc1_lower =
        gfc1_lower_nk(cute::_, cute::_, block_n_idx, cute::_);  // (BLK_N, BLK_K, k)
    cute::Tensor gfc1g_lower =
        gfc1_gate_lower_nk(cute::_, cute::_, block_n_idx, cute::_);  // (BLK_N, BLK_K, k)

    typename KT::GmemTiledCopyA gmem_tiled_copy_A;
    typename KT::GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    cute::Tensor tInputgInput =
        gmem_thr_copy_A.partition_S(gInput);  // (ACPY,ACPY_M,ACPY_K,k) // ((8, 1), 1, 1, 2)
    cute::Tensor tInputsInput = gmem_thr_copy_A.partition_D(
        sInput);  // (ACPY,ACPY_M,ACPY_K,Stage) // ((8, 1), 1, 1, (1, 3))
    cute::Tensor tfc1gfc1_upper =
        gmem_thr_copy_B.partition_S(gfc1_upper);  // (BCPY,BCPY_N,BCPY_K,k) // ((8, 1), 8, 1, 2)
    cute::Tensor tfc1sfc1_upper = gmem_thr_copy_B.partition_D(
        sfc1_weight_upper);  // (BCPY,BCPY_N,BCPY_K,Stage) // ((8, 1), 8, 1, (1, 3))
    cute::Tensor tfc1gfc1_lower =
        gmem_thr_copy_B.partition_S(gfc1_lower);  // (BCPY,BCPY_N,BCPY_K,k)
    cute::Tensor tfc1sfc1_lower =
        gmem_thr_copy_B.partition_D(sfc1_weight_lower);  // (BCPY,BCPY_N,BCPY_K,Stage)
    cute::Tensor tfc1ggfc1g_upper =
        gmem_thr_copy_B.partition_S(gfc1g_upper);  // (BCPY,BCPY_N,BCPY_K,k)
    cute::Tensor tfc1gsfc1g_upper =
        gmem_thr_copy_B.partition_D(sfc1_gate_weight_upper);  // (BCPY,BCPY_N,BCPY_K,Stage)
    cute::Tensor tfc1ggfc1g_lower =
        gmem_thr_copy_B.partition_S(gfc1g_lower);  // (BCPY,BCPY_N,BCPY_K,k)
    cute::Tensor tfc1gsfc1g_lower =
        gmem_thr_copy_B.partition_D(sfc1_gate_weight_lower);  // (BCPY,BCPY_N,BCPY_K,Stage)

    // Allocate predicate tensors for input and fc weight (actually we only need input predicate
    // tensor)
    cute::Tensor tInputpInput = cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(tInputsInput), cute::size<2>(tInputsInput)),
        cute::Stride<cute::_1, cute::_0>{});
    // Construct identity layout for sInput
    cute::Tensor cInput = make_identity_tensor(make_shape(
        cute::size<0>(sInput), cute::size<1>(sInput)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    cute::Tensor tInputcInput =
        gmem_thr_copy_A.partition_S(cInput);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < cute::size<0>(tInputpInput); ++m) {
      tInputpInput(m, 0) =
          cute::get<0>(tInputcInput(0, m, 0)) < residue_m;  // blk_m coord < residue_m
    }

    // (1.2) prefetch gmem -> smem
    cute::clear(tInputsInput);  // we don't need to clear tfc1sfc1..
    auto k_tile_iter = cute::make_coord_iterator(cute::size<2>(gInput));  // emm, iter start from 0
    int k_tile_count = cute::size<2>(gInput);
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < KT::Stages - 1; ++k_pipe) {
      if (k_tile_count <= 0) {
        cute::clear(tInputpInput);
      }
      cute::copy_if(gmem_tiled_copy_A, tInputpInput,
                    tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                    tInputsInput(cute::_, cute::_, cute::_, k_pipe));
      cute::copy(gmem_tiled_copy_B, tfc1gfc1_upper(cute::_, cute::_, cute::_, *k_tile_iter),
                 tfc1sfc1_upper(cute::_, cute::_, cute::_, k_pipe));
      cute::copy(gmem_tiled_copy_B, tfc1ggfc1g_upper(cute::_, cute::_, cute::_, *k_tile_iter),
                 tfc1gsfc1g_upper(cute::_, cute::_, cute::_, k_pipe));
      cute::copy(gmem_tiled_copy_B, tfc1gfc1_lower(cute::_, cute::_, cute::_, *k_tile_iter),
                 tfc1sfc1_lower(cute::_, cute::_, cute::_, k_pipe));
      cute::copy(gmem_tiled_copy_B, tfc1ggfc1g_lower(cute::_, cute::_, cute::_, *k_tile_iter),
                 tfc1gsfc1g_lower(cute::_, cute::_, cute::_, k_pipe));
      cute::cp_async_fence();
      k_tile_count--;
      if (k_tile_count > 0) {
        ++k_tile_iter;
      }
    }

    // (1.3) get partition for rf
    typename KT::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    cute::Tensor tOrInput =
        thr_mma.partition_fragment_A(sInput(cute::_, cute::_, 0));  // (MMA,MMA_M,MMA_K)
    cute::Tensor tOrfc1_upper =
        thr_mma.partition_fragment_B(sfc1_weight_upper(cute::_, cute::_, 0));  // (MMA,MMA_N,MMA_K)
    cute::Tensor tOrfc1g_upper = thr_mma.partition_fragment_B(
        sfc1_gate_weight_upper(cute::_, cute::_, 0));  // (MMA,MMA_N,MMA_K)
    cute::Tensor tOrfc1_lower =
        thr_mma.partition_fragment_B(sfc1_weight_lower(cute::_, cute::_, 0));  // (MMA,MMA_N,MMA_K)
    cute::Tensor tOrfc1g_lower = thr_mma.partition_fragment_B(
        sfc1_gate_weight_lower(cute::_, cute::_, 0));  // (MMA,MMA_N,MMA_K)

    cute::Tensor accum = cute::partition_fragment_C(
        tiled_mma, cute::take<0, 2>(typename KT::TileShape{}));  // (MMA,MMA_M,MMA_N)
    cute::Tensor accum_gate = cute::partition_fragment_C(
        tiled_mma, cute::take<0, 2>(typename KT::TileShape{}));  // (MMA,MMA_M,MMA_N)
    cute::clear(accum);
    cute::clear(accum_gate);

    // Define full layout for B (unpacked) to get correct fragment size for tmp1/tmp2
    // IMPORTANT: Use the fp16 layout atom (not fp8) since tmp1/tmp2 hold unpacked fp16 values.
    // SmemLayoutAtomB is designed for packed fp8 (uint16_t with 2 fp8), so it produces wrong
    // strides.
    using SmemLayoutAtomB_FP16 =
        typename DefaultGemm_TensorOpSm80_OperandB<cutlass::half_t, cutlass::layout::ColumnMajor,
                                                   KT::kAlignment, KT::kBlcokKSmem>::SmemLayoutAtom;
    using LayoutB_Full = decltype(cute::tile_to_shape(
        SmemLayoutAtomB_FP16{},
        cute::make_shape(cute::shape<1>(typename KT::TileShape{}),
                         cute::shape<2>(typename KT::TileShape{}), cute::Int<KT::Stages>{})));
    // Create a fake tensor to get the full fragment layout. Use half_t pointer to match MMA
    // expectation.
    cute::Tensor sfc1_fake =
        cute::make_tensor(cute::make_smem_ptr((cutlass::half_t*)nullptr), LayoutB_Full{});
    cute::Tensor tOrfc1_full = thr_mma.partition_fragment_B(sfc1_fake(cute::_, cute::_, 0));
    cute::Tensor tmp1 = cute::make_fragment_like<cutlass::half_t>(tOrfc1_full.layout());
    cute::Tensor tmp2 = cute::make_fragment_like<cutlass::half_t>(tOrfc1_full.layout());

    // checkout the shape
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrInput) == cute::size<1>(accum));            // MMA_M
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrInput) == cute::size<1>(accum_gate));       // MMA_M
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1_upper) == cute::size<2>(accum));        // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1_upper) == cute::size<2>(accum_gate));   // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1g_upper) == cute::size<2>(accum));       // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1g_upper) == cute::size<2>(accum_gate));  // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOrInput) ==
                         (cute::size<2>(tOrfc1_upper) * cute::Int<2>{}));  // MMA_K
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOrInput) ==
                         (cute::size<2>(tOrfc1g_upper) * cute::Int<2>{}));  // MMA_K
    CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_A) == cute::size(tiled_mma));
    CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_B) == cute::size(tiled_mma));

    // (1.4)retiling the smem and rf for copy..
    auto smem_tiled_copy_A = cute::make_tiled_copy_A(typename KT::SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    cute::Tensor tOsInput = smem_thr_copy_A.partition_S(sInput);  // (CPY,CPY_M,CPY_K,Stage)
    cute::Tensor tOrInput_copy_view = smem_thr_copy_A.retile_D(tOrInput);  // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsInput) == cute::size<1>(tOrInput_copy_view));  // CPY_M
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsInput) == cute::size<2>(tOrInput_copy_view));  // CPY_K

    auto smem_tiled_copy_B = cute::make_tiled_copy_B(typename KT::SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    cute::Tensor tOsfc1_upper_copy_view =
        smem_thr_copy_B.partition_S(sfc1_weight_upper);  // (CPY,CPY_N,CPY_K,Stage)
    cute::Tensor tOrfc1_upper_copy_view =
        smem_thr_copy_B.retile_D(tOrfc1_upper);  // (CPY,CPY_N,CPY_K)
    cute::Tensor tOsfc1g_upper_copy_view =
        smem_thr_copy_B.partition_S(sfc1_gate_weight_upper);  // (CPY,CPY_N,CPY_K,Stage)
    cute::Tensor tOrfc1g_upper_copy_view =
        smem_thr_copy_B.retile_D(tOrfc1g_upper);  // (CPY,CPY_N,CPY_K)
    cute::Tensor tOsfc1_lower_copy_view =
        smem_thr_copy_B.partition_S(sfc1_weight_lower);  // (CPY,CPY_N,CPY_K,Stage)
    cute::Tensor tOrfc1_lower_copy_view =
        smem_thr_copy_B.retile_D(tOrfc1_lower);  // (CPY,CPY_N,CPY_K)
    cute::Tensor tOsfc1g_lower_copy_view =
        smem_thr_copy_B.partition_S(sfc1_gate_weight_lower);  // (CPY,CPY_N,CPY_K,Stage)
    cute::Tensor tOrfc1g_lower_copy_view =
        smem_thr_copy_B.retile_D(tOrfc1g_lower);  // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1_upper_copy_view) ==
                         cute::size<1>(tOrfc1_upper_copy_view));  // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1_upper_copy_view) ==
                         cute::size<2>(tOrfc1_upper_copy_view));  // CPY_K
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1g_upper_copy_view) ==
                         cute::size<1>(tOrfc1g_upper_copy_view));  // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1g_upper_copy_view) ==
                         cute::size<2>(tOrfc1g_upper_copy_view));  // CPY_K
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1_lower_copy_view) ==
                         cute::size<1>(tOrfc1_lower_copy_view));  // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1_lower_copy_view) ==
                         cute::size<2>(tOrfc1_lower_copy_view));  // CPY_K
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1g_lower_copy_view) ==
                         cute::size<1>(tOrfc1g_lower_copy_view));  // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1g_lower_copy_view) ==
                         cute::size<2>(tOrfc1g_lower_copy_view));  // CPY_K

    // (1.5) mainloop
    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = KT::Stages - 1;

    cute::Tensor tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
    cute::Tensor tOsfc1_upper_p = tOsfc1_upper_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
    cute::Tensor tOsfc1g_upper_p =
        tOsfc1g_upper_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
    cute::Tensor tOsfc1_lower_p = tOsfc1_lower_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
    cute::Tensor tOsfc1g_lower_p =
        tOsfc1g_lower_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);

    constexpr int K_BLOCK_MAX = cute::size<2>(tOrInput);
    constexpr int K_BLOCK_PACKED_MAX = K_BLOCK_MAX / cute::_2{};
    // prefetch register pipeline
    if constexpr (K_BLOCK_PACKED_MAX > 1) {
      cute::cp_async_wait<KT::Stages - 2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrInput_copy_view(cute::_, cute::_, cute::Int<0>{}));
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, cute::Int<1>{}),
                 tOrInput_copy_view(cute::_, cute::_, cute::Int<1>{}));

      cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrfc1_upper_copy_view(cute::_, cute::_, cute::Int<0>{}));
      cute::copy(smem_tiled_copy_B, tOsfc1g_upper_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrfc1g_upper_copy_view(cute::_, cute::_, cute::Int<0>{}));
      cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrfc1_lower_copy_view(cute::_, cute::_, cute::Int<0>{}));
      cute::copy(smem_tiled_copy_B, tOsfc1g_lower_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrfc1g_lower_copy_view(cute::_, cute::_, cute::Int<0>{}));

      // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
      shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, cute::Int<0>{}),
                                tOrfc1_lower(cute::_, cute::_, cute::Int<0>{}));
      shuffle_fragment_B_paired(tOrfc1g_upper(cute::_, cute::_, cute::Int<0>{}),
                                tOrfc1g_lower(cute::_, cute::_, cute::Int<0>{}));

      reconstruct(tOrfc1_upper(cute::_, cute::_, cute::Int<0>{}),
                  tOrfc1_lower(cute::_, cute::_, cute::Int<0>{}),
                  tmp1(cute::_, cute::_, cute::Int<0>{}), tmp1(cute::_, cute::_, cute::Int<1>{}));
      reconstruct(tOrfc1g_upper(cute::_, cute::_, cute::Int<0>{}),
                  tOrfc1g_lower(cute::_, cute::_, cute::Int<0>{}),
                  tmp2(cute::_, cute::_, cute::Int<0>{}), tmp2(cute::_, cute::_, cute::Int<1>{}));
    }
    // k loop for mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_PACKED_MAX>{}, [&](auto k_block) {
        if (k_block == K_BLOCK_PACKED_MAX - 1) {
          tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_upper_p = tOsfc1_upper_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1g_upper_p = tOsfc1g_upper_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_lower_p = tOsfc1_lower_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1g_lower_p = tOsfc1g_lower_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          cute::cp_async_wait<KT::Stages - 2>();
          __syncthreads();
        }
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_PACKED_MAX;
        auto k_logical_next_0 = k_block_next * cute::_2{};
        auto k_logical_next_1 = k_block_next * cute::_2{} + cute::_1{};
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_0),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_0));
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_1),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_1));
        cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, k_block_next),
                   tOrfc1_upper_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1g_upper_p(cute::_, cute::_, k_block_next),
                   tOrfc1g_upper_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, k_block_next),
                   tOrfc1_lower_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1g_lower_p(cute::_, cute::_, k_block_next),
                   tOrfc1g_lower_copy_view(cute::_, cute::_, k_block_next));
        // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
        shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, k_block_next),
                                  tOrfc1_lower(cute::_, cute::_, k_block_next));
        shuffle_fragment_B_paired(tOrfc1g_upper(cute::_, cute::_, k_block_next),
                                  tOrfc1g_lower(cute::_, cute::_, k_block_next));
        reconstruct(tOrfc1_upper(cute::_, cute::_, k_block_next),
                    tOrfc1_lower(cute::_, cute::_, k_block_next),
                    tmp1(cute::_, cute::_, k_logical_next_0),
                    tmp1(cute::_, cute::_, k_logical_next_1));
        reconstruct(tOrfc1g_upper(cute::_, cute::_, k_block_next),
                    tOrfc1g_lower(cute::_, cute::_, k_block_next),
                    tmp2(cute::_, cute::_, k_logical_next_0),
                    tmp2(cute::_, cute::_, k_logical_next_1));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0) {
          cute::copy_if(gmem_tiled_copy_A, tInputpInput,
                        tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                        tInputsInput(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy(gmem_tiled_copy_B, tfc1gfc1_upper(cute::_, cute::_, cute::_, *k_tile_iter),
                     tfc1sfc1_upper(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy(gmem_tiled_copy_B, tfc1ggfc1g_upper(cute::_, cute::_, cute::_, *k_tile_iter),
                     tfc1gsfc1g_upper(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy(gmem_tiled_copy_B, tfc1gfc1_lower(cute::_, cute::_, cute::_, *k_tile_iter),
                     tfc1sfc1_lower(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy(gmem_tiled_copy_B, tfc1ggfc1g_lower(cute::_, cute::_, cute::_, *k_tile_iter),
                     tfc1gsfc1g_lower(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::cp_async_fence();
          if (k_tile_count - 1 > 0) {
            ++k_tile_iter;
          }

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
        }
        // Thread-level register gemm for k_block
        auto k_logical_0 = k_block * cute::_2{};
        auto k_logical_1 = k_block * cute::_2{} + cute::_1{};
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_0),
                   tmp1(cute::_, cute::_, k_logical_0), accum);
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_1),
                   tmp1(cute::_, cute::_, k_logical_1), accum);
        cute::gemm(tiled_mma, accum_gate, tOrInput(cute::_, cute::_, k_logical_0),
                   tmp2(cute::_, cute::_, k_logical_0), accum_gate);
        cute::gemm(tiled_mma, accum_gate, tOrInput(cute::_, cute::_, k_logical_1),
                   tmp2(cute::_, cute::_, k_logical_1), accum_gate);
      });
    }

    // load tail
    cute::for_each(cute::make_int_sequence<KT::Stages - 2>{}, [&](auto WaitIndex) {
      k_tile_count--;
      using WaitIndex_t = decltype(WaitIndex);
      cute::for_each(cute::make_int_sequence<K_BLOCK_PACKED_MAX>{}, [&](auto k_block) {
        if (k_block == K_BLOCK_PACKED_MAX - 1) {
          tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_upper_p = tOsfc1_upper_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1g_upper_p = tOsfc1g_upper_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_lower_p = tOsfc1_lower_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1g_lower_p = tOsfc1g_lower_copy_view(cute::_, cute::_, cute::_, smem_pipe_read);
          cute::cp_async_wait<KT::Stages - 3 - WaitIndex_t::value>();
          __syncthreads();
        }
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_PACKED_MAX;
        auto k_logical_next_0 = k_block_next * cute::_2{};
        auto k_logical_next_1 = k_block_next * cute::_2{} + cute::_1{};
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_0),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_0));
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_1),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_1));
        cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, k_block_next),
                   tOrfc1_upper_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1g_upper_p(cute::_, cute::_, k_block_next),
                   tOrfc1g_upper_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, k_block_next),
                   tOrfc1_lower_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1g_lower_p(cute::_, cute::_, k_block_next),
                   tOrfc1g_lower_copy_view(cute::_, cute::_, k_block_next));
        // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
        shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, k_block_next),
                                  tOrfc1_lower(cute::_, cute::_, k_block_next));
        shuffle_fragment_B_paired(tOrfc1g_upper(cute::_, cute::_, k_block_next),
                                  tOrfc1g_lower(cute::_, cute::_, k_block_next));
        reconstruct(tOrfc1_upper(cute::_, cute::_, k_block_next),
                    tOrfc1_lower(cute::_, cute::_, k_block_next),
                    tmp1(cute::_, cute::_, k_logical_next_0),
                    tmp1(cute::_, cute::_, k_logical_next_1));
        reconstruct(tOrfc1g_upper(cute::_, cute::_, k_block_next),
                    tOrfc1g_lower(cute::_, cute::_, k_block_next),
                    tmp2(cute::_, cute::_, k_logical_next_0),
                    tmp2(cute::_, cute::_, k_logical_next_1));
        if (k_block == 0) {
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
        }
        // Thread-level register gemm for k_block
        auto k_logical_0 = k_block * cute::_2{};
        auto k_logical_1 = k_block * cute::_2{} + cute::_1{};
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_0),
                   tmp1(cute::_, cute::_, k_logical_0), accum);
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_1),
                   tmp1(cute::_, cute::_, k_logical_1), accum);
        cute::gemm(tiled_mma, accum_gate, tOrInput(cute::_, cute::_, k_logical_0),
                   tmp2(cute::_, cute::_, k_logical_0), accum_gate);
        cute::gemm(tiled_mma, accum_gate, tOrInput(cute::_, cute::_, k_logical_1),
                   tmp2(cute::_, cute::_, k_logical_1), accum_gate);
      });
    });
    // mma tail
    cute::for_each(cute::make_int_sequence<K_BLOCK_PACKED_MAX>{}, [&](auto k_block) {
      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_PACKED_MAX;
      auto k_logical_next_0 = k_block_next * cute::_2{};
      auto k_logical_next_1 = k_block_next * cute::_2{} + cute::_1{};
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_0),
                 tOrInput_copy_view(cute::_, cute::_, k_logical_next_0));
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_1),
                 tOrInput_copy_view(cute::_, cute::_, k_logical_next_1));
      cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, k_block_next),
                 tOrfc1_upper_copy_view(cute::_, cute::_, k_block_next));
      cute::copy(smem_tiled_copy_B, tOsfc1g_upper_p(cute::_, cute::_, k_block_next),
                 tOrfc1g_upper_copy_view(cute::_, cute::_, k_block_next));
      cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, k_block_next),
                 tOrfc1_lower_copy_view(cute::_, cute::_, k_block_next));
      cute::copy(smem_tiled_copy_B, tOsfc1g_lower_p(cute::_, cute::_, k_block_next),
                 tOrfc1g_lower_copy_view(cute::_, cute::_, k_block_next));
      // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
      shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, k_block_next),
                                tOrfc1_lower(cute::_, cute::_, k_block_next));
      shuffle_fragment_B_paired(tOrfc1g_upper(cute::_, cute::_, k_block_next),
                                tOrfc1g_lower(cute::_, cute::_, k_block_next));
      reconstruct(tOrfc1_upper(cute::_, cute::_, k_block_next),
                  tOrfc1_lower(cute::_, cute::_, k_block_next),
                  tmp1(cute::_, cute::_, k_logical_next_0),
                  tmp1(cute::_, cute::_, k_logical_next_1));
      reconstruct(tOrfc1g_upper(cute::_, cute::_, k_block_next),
                  tOrfc1g_lower(cute::_, cute::_, k_block_next),
                  tmp2(cute::_, cute::_, k_logical_next_0),
                  tmp2(cute::_, cute::_, k_logical_next_1));
      // Thread-level register gemm for k_block
      auto k_logical_0 = k_block * cute::_2{};
      auto k_logical_1 = k_block * cute::_2{} + cute::_1{};
      cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_0),
                 tmp1(cute::_, cute::_, k_logical_0), accum);
      cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_1),
                 tmp1(cute::_, cute::_, k_logical_1), accum);
      cute::gemm(tiled_mma, accum_gate, tOrInput(cute::_, cute::_, k_logical_0),
                 tmp2(cute::_, cute::_, k_logical_0), accum_gate);
      cute::gemm(tiled_mma, accum_gate, tOrInput(cute::_, cute::_, k_logical_1),
                 tmp2(cute::_, cute::_, k_logical_1), accum_gate);
    });

    // (2) add bias if it has..
    if (params.ptr_bias != nullptr) {
      cute::Tensor gBias =
          gBias_mn(cute::_, cute::_, bias_is_broadcast ? 0 : block_m_idx, block_n_idx);
      cute::Tensor gBias_gate =
          gBias_gate_mn(cute::_, cute::_, bias_is_broadcast ? 0 : block_m_idx, block_n_idx);
      cute::Tensor tOgBias = thr_mma.partition_C(gBias);
      cute::Tensor tOgBiasg = thr_mma.partition_C(gBias_gate);
      for (int i = 0; i < cute::size(accum); i++) {
        accum(i) += tOgBias(i);
        accum_gate(i) += tOgBiasg(i);
      }
    }

    // (3) calculate swiglu
    using ActivationFn = typename KT::ActivationFn;
    ActivationFn fn{};
    CUTLASS_PRAGMA_UNROLL
    for (int temp_iter = 0; temp_iter < cute::size(accum); temp_iter++) {
      accum(temp_iter) = fn(accum_gate(temp_iter)) * accum(temp_iter);
    }

    // (4) push all the result to smem
    // (4.1) convert result from ElementAccum to ElementInput
    cute::Tensor temp_accum = util_convert_type<KT::ElementOutput>(accum);
    // (4.2) retile rf and smem for copy back..
    auto smem_tiled_copy_O = cute::make_tiled_copy_C(typename KT::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
    cute::Tensor taccumrO = smem_thr_copy_O.retile_S(temp_accum);
    cute::Tensor taccumsO = smem_thr_copy_O.partition_D(sO);

    // (4.3) copy rf result to smem (TODO: maybe use forloop for better performance..)
    cute::copy(smem_tiled_copy_O, taccumrO, taccumsO);
    __syncthreads();

    // (4.4) sO -> rO -> gO
    typename KT::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    cute::Tensor gO = gOutput_mn(cute::_, cute::_, block_m_idx, block_n_idx);
    auto tOsO = gmem_thr_copy_O.partition_S(sO);
    auto tOgO = gmem_thr_copy_O.partition_D(gO);
    cute::Tensor cOutput = cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(typename KT::TileShape{}), cute::size<1>(typename KT::TileShape{})));
    cute::Tensor tOcO = gmem_thr_copy_O.partition_D(cOutput);
    cute::Tensor tOrO = cute::make_tensor<KT::ElementOutput>(cute::shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < cute::size<1>(tOgO); ++m) {
      if (cute::get<0>(tOcO(0, m, 0)) < residue_m) {
        cute::copy(gmem_tiled_copy_O, tOrO(cute::_, m, cute::_), tOgO(cute::_, m, cute::_));
      }
    }
  }
};

template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int TileM_,
          int TileN_, int TileK_, int Stages_, Activation_Type activation_type_>
struct DualWeight_Fused_Moe_Kernel_routine_sm80<
    ElementInput_, ElementWeight_, ElementOutput_, TileM_, TileN_, TileK_, Stages_,
    activation_type_, std::enable_if_t<!isGateActivation(activation_type_)>> {
  using KT =
      DualWeight_Fused_Moe_Kernel_traits_sm80<ElementInput_, ElementWeight_, ElementOutput_, TileM_,
                                              TileN_, TileK_, Stages_, activation_type_>;
  using Params = Routine_Params<ElementInput_, ElementWeight_, ElementOutput_>;

  CUTE_DEVICE auto gmem_tensor_init(int const problem_index, int const gemm_m,
                                    Params const& params) {
    using X = cute::Underscore;

    int const M = gemm_m;
    int const N1 = params.gemm_n;
    int const K1 = params.gemm_k;
    bool const bias_is_broadcast = params.bias_is_broadcast;

    size_t const problem_jump = problem_index;
    size_t const row_jump =
        ((problem_index == 0) ? 0 : params.total_tokens_including_expert[problem_index - 1]);
    typename KT::ElementInput const* ptr_input_ = params.ptr_input + row_jump * K1;
    typename KT::ElementWeight const* ptr_fc1_upper_ =
        params.ptr_fc1_upper + problem_jump * N1 * K1;
    typename KT::ElementWeight const* ptr_fc1_lower_ =
        params.ptr_fc1_lower + problem_jump * N1 * K1;
    typename KT::ElementInput const* ptr_bias_ =
        (params.ptr_bias == nullptr) ? nullptr
                                     : (bias_is_broadcast ? params.ptr_bias + problem_jump * N1
                                                          : params.ptr_bias + row_jump * N1);
    typename KT::ElementOutput* ptr_output_ = params.ptr_output + row_jump * N1;

    cute::Tensor mInput_mk = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_input_)),
        cute::make_shape(M, K1), cute::make_stride(K1, cute::_1{}));

    cute::Tensor mfc1_upper_nk = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<typename KT::ElementWeight const*>(ptr_fc1_upper_)),
        cute::make_shape(N1, K1 / 2), cute::make_stride(K1 / 2, cute::_1{}));

    cute::Tensor mfc1_lower_nk = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<typename KT::ElementWeight const*>(ptr_fc1_lower_)),
        cute::make_shape(N1, K1 / 2), cute::make_stride(K1 / 2, cute::_1{}));

    cute::Tensor mBias_mn = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_bias_)),
        cute::make_shape(M, N1),
        cute::make_stride(bias_is_broadcast ? cute::Int<0>{} : N1,
                          cute::_1{}));  // trick: bias shape is [1, N], but we use [M, N].

    cute::Tensor mOutput_mn =
        cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementInput*>(ptr_output_)),
                          cute::make_shape(M, N1), cute::make_stride(N1, cute::_1{}));

    cute::Tensor gInput_mk = cute::local_tile(
        mInput_mk, typename KT::TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, X, cute::_1>{});  // (BLK_M, BLK_K, m, k)

    // Packed Tiler (BLK_M, BLK_N, BLK_K/2)
    auto packed_tiler = cute::make_shape(cute::size<0>(typename KT::TileShape{}),
                                         cute::size<1>(typename KT::TileShape{}),
                                         cute::size<2>(typename KT::TileShape{}) / cute::Int<2>{});

    cute::Tensor gfc1_upper_nk =
        cute::local_tile(mfc1_upper_nk, packed_tiler, cute::make_coord(cute::_, cute::_, cute::_),
                         cute::Step<X, cute::_1, cute::_1>{});  // (BLK_N, BLK_K, n, k)
    cute::Tensor gfc1_lower_nk =
        cute::local_tile(mfc1_lower_nk, packed_tiler, cute::make_coord(cute::_, cute::_, cute::_),
                         cute::Step<X, cute::_1, cute::_1>{});  // (BLK_N, BLK_K, n, k)

    cute::Tensor gBias_mn = cute::local_tile(
        mBias_mn, typename KT::TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, cute::_1, X>{});  // (BLK_M, BLK_N, m, n)
    cute::Tensor gOutput_mn = cute::local_tile(
        mOutput_mn, typename KT::TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, cute::_1, X>{});  // (BLK_M, BLK_N, m, n)

    return cute::make_tuple(gInput_mk, gfc1_upper_nk, gfc1_lower_nk, gBias_mn, gOutput_mn);
  }

  CUTE_DEVICE void run_routine(Params const& params, int const problem_index, int const block_m_idx,
                               int const block_n_idx, int const gemm_m) {
    extern __shared__ char smem_[];
    typename KT::SharedStorage& shared_storage =
        *reinterpret_cast<typename KT::SharedStorage*>(smem_);
    int const thread_idx = threadIdx.x;
    bool const bias_is_broadcast = params.bias_is_broadcast;
    // gmem tensor partition ..
    auto [gInput_mk, gfc1_upper_nk, gfc1_lower_nk, gBias_mn, gOutput_mn] =
        gmem_tensor_init(problem_index, gemm_m, params);
    int const residue_m = gemm_m - block_m_idx * cute::size<0>(gInput_mk);
    auto const n_tile_count = cute::size<2>(gfc1_upper_nk);

    // smem tensor ..
    cute::Tensor sInput = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_input.data()),
                                            typename KT::SmemLayoutA{});  // (BLK_M, BLK_K, Stage)
    cute::Tensor sfc1_weight_upper =
        cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_weight_upper.data()),
                          typename KT::SmemLayoutB{});  // (BLK_N, BLK_K, Stage)
    cute::Tensor sfc1_weight_lower =
        cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_weight_lower.data()),
                          typename KT::SmemLayoutB{});  // (BLK_N, BLK_K, Stage)
    cute::Tensor sO = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_o.data()),
                                        typename KT::SmemLayoutO{});  // (BLK_M, BLK_N)

    // (1) first step, get the fc1_res and fc1_gate

    // (1.1) get partition for gmem -> smem
    cute::Tensor gInput = gInput_mk(cute::_, cute::_, block_m_idx, cute::_);  // (BLK_M, BLK_K, k)
    cute::Tensor gfc1_upper =
        gfc1_upper_nk(cute::_, cute::_, block_n_idx, cute::_);  // (BLK_N, BLK_K, k)
    cute::Tensor gfc1_lower =
        gfc1_lower_nk(cute::_, cute::_, block_n_idx, cute::_);  // (BLK_N, BLK_K, k)

    typename KT::GmemTiledCopyA gmem_tiled_copy_A;
    typename KT::GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    cute::Tensor tInputgInput = gmem_thr_copy_A.partition_S(gInput);  // (ACPY,ACPY_M,ACPY_K,k)
    cute::Tensor tInputsInput = gmem_thr_copy_A.partition_D(sInput);  // (ACPY,ACPY_M,ACPY_K,Stage)
    cute::Tensor tfc1gfc1_upper =
        gmem_thr_copy_B.partition_S(gfc1_upper);  // (BCPY,BCPY_N,BCPY_K,k)
    cute::Tensor tfc1sfc1_upper =
        gmem_thr_copy_B.partition_D(sfc1_weight_upper);  // (BCPY,BCPY_N,BCPY_K,Stage)
    cute::Tensor tfc1gfc1_lower =
        gmem_thr_copy_B.partition_S(gfc1_lower);  // (BCPY,BCPY_N,BCPY_K,k)
    cute::Tensor tfc1sfc1_lower =
        gmem_thr_copy_B.partition_D(sfc1_weight_lower);  // (BCPY,BCPY_N,BCPY_K,Stage)

    // Allocate predicate tensors for input and fc weight (actually we only need input predicate
    // tensor)
    cute::Tensor tInputpInput = cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(tInputsInput), cute::size<2>(tInputsInput)),
        cute::Stride<cute::_1, cute::_0>{});
    // Construct identity layout for sInput
    cute::Tensor cInput = make_identity_tensor(make_shape(
        cute::size<0>(sInput), cute::size<1>(sInput)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    cute::Tensor tInputcInput =
        gmem_thr_copy_A.partition_S(cInput);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < cute::size<0>(tInputpInput); ++m) {
      tInputpInput(m, 0) =
          cute::get<0>(tInputcInput(0, m, 0)) < residue_m;  // blk_m coord < residue_m
    }

    // (1.2) prefetch gmem -> smem
    cute::clear(tInputsInput);  // we don't need to clear tfc1sfc1..
    auto k_tile_iter = cute::make_coord_iterator(cute::size<2>(gInput));  // emm, iter start from 0
    int k_tile_count = cute::size<2>(gInput);
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < KT::Stages - 1; ++k_pipe) {
      if (k_tile_count <= 0) {
        cute::clear(tInputpInput);
      }
      cute::copy_if(gmem_tiled_copy_A, tInputpInput,
                    tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                    tInputsInput(cute::_, cute::_, cute::_, k_pipe));
      cute::copy(gmem_tiled_copy_B, tfc1gfc1_upper(cute::_, cute::_, cute::_, *k_tile_iter),
                 tfc1sfc1_upper(cute::_, cute::_, cute::_, k_pipe));
      cute::copy(gmem_tiled_copy_B, tfc1gfc1_lower(cute::_, cute::_, cute::_, *k_tile_iter),
                 tfc1sfc1_lower(cute::_, cute::_, cute::_, k_pipe));
      cute::cp_async_fence();
      k_tile_count--;
      if (k_tile_count > 0) {
        ++k_tile_iter;
      }
    }

    // (1.3) get partition for rf
    typename KT::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    cute::Tensor tOrInput =
        thr_mma.partition_fragment_A(sInput(cute::_, cute::_, 0));  // (MMA,MMA_M,MMA_K)
    cute::Tensor tOrfc1_upper =
        thr_mma.partition_fragment_B(sfc1_weight_upper(cute::_, cute::_, 0));  // (MMA,MMA_N,MMA_K)
    cute::Tensor tOrfc1_lower =
        thr_mma.partition_fragment_B(sfc1_weight_lower(cute::_, cute::_, 0));  // (MMA,MMA_N,MMA_K)

    cute::Tensor accum = cute::partition_fragment_C(
        tiled_mma, cute::take<0, 2>(typename KT::TileShape{}));  // (MMA,MMA_M,MMA_N)
    cute::clear(accum);

    using SmemLayoutAtomB_FP16 =
        typename DefaultGemm_TensorOpSm80_OperandB<cutlass::half_t, cutlass::layout::ColumnMajor,
                                                   KT::kAlignment, KT::kBlcokKSmem>::SmemLayoutAtom;
    using LayoutB_Full = decltype(cute::tile_to_shape(
        SmemLayoutAtomB_FP16{},
        cute::make_shape(cute::shape<1>(typename KT::TileShape{}),
                         cute::shape<2>(typename KT::TileShape{}), cute::Int<KT::Stages>{})));
    // Create a fake tensor to get the full fragment layout. Use half_t pointer to match MMA
    // expectation.
    cute::Tensor sfc1_fake =
        cute::make_tensor(cute::make_smem_ptr((cutlass::half_t*)nullptr), LayoutB_Full{});
    cute::Tensor tOrfc1_full = thr_mma.partition_fragment_B(sfc1_fake(cute::_, cute::_, 0));
    cute::Tensor tmp1 = cute::make_fragment_like<cutlass::half_t>(tOrfc1_full.layout());

    // checkout the shape
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrInput) == cute::size<1>(accum));      // MMA_M
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1_upper) == cute::size<2>(accum));  // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1_lower) == cute::size<2>(accum));  // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOrInput) ==
                         (cute::size<2>(tOrfc1_upper) * cute::Int<2>{}));  // MMA_K
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOrInput) ==
                         (cute::size<2>(tOrfc1_lower) * cute::Int<2>{}));  // MMA_K
    CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_A) == cute::size(tiled_mma));
    CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_B) == cute::size(tiled_mma));

    // (1.4)retiling the smem and rf for copy..
    auto smem_tiled_copy_A = cute::make_tiled_copy_A(typename KT::SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    cute::Tensor tOsInput = smem_thr_copy_A.partition_S(sInput);  // (CPY,CPY_M,CPY_K,Stage)
    cute::Tensor tOrInput_copy_view = smem_thr_copy_A.retile_D(tOrInput);  // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsInput) == cute::size<1>(tOrInput_copy_view));  // CPY_M
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsInput) == cute::size<2>(tOrInput_copy_view));  // CPY_K

    auto smem_tiled_copy_B = cute::make_tiled_copy_B(typename KT::SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    cute::Tensor tOsfc1_upper =
        smem_thr_copy_B.partition_S(sfc1_weight_upper);  // (CPY,CPY_N,CPY_K,Stage)
    cute::Tensor tOrfc1_upper_copy_view =
        smem_thr_copy_B.retile_D(tOrfc1_upper);  // (CPY,CPY_N,CPY_K)
    cute::Tensor tOsfc1_lower =
        smem_thr_copy_B.partition_S(sfc1_weight_lower);  // (CPY,CPY_N,CPY_K,Stage)
    cute::Tensor tOrfc1_lower_copy_view =
        smem_thr_copy_B.retile_D(tOrfc1_lower);  // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1_upper) ==
                         cute::size<1>(tOrfc1_upper_copy_view));  // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1_upper) ==
                         cute::size<2>(tOrfc1_upper_copy_view));  // CPY_K
    CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1_lower) ==
                         cute::size<1>(tOrfc1_lower_copy_view));  // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1_lower) ==
                         cute::size<2>(tOrfc1_lower_copy_view));  // CPY_K

    // (1.5) mainloop
    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = KT::Stages - 1;

    cute::Tensor tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
    cute::Tensor tOsfc1_upper_p = tOsfc1_upper(cute::_, cute::_, cute::_, smem_pipe_read);
    cute::Tensor tOsfc1_lower_p = tOsfc1_lower(cute::_, cute::_, cute::_, smem_pipe_read);

    constexpr int K_BLOCK_MAX = cute::size<2>(tOrInput);
    constexpr int K_BLOCK_PACKED_MAX = K_BLOCK_MAX / cute::Int<2>{};
    // prefetch register pipeline
    if constexpr (K_BLOCK_PACKED_MAX > 1) {
      cute::cp_async_wait<KT::Stages - 2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrInput_copy_view(cute::_, cute::_, cute::Int<0>{}));
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, cute::Int<1>{}),
                 tOrInput_copy_view(cute::_, cute::_, cute::Int<1>{}));
      cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrfc1_upper_copy_view(cute::_, cute::_, cute::Int<0>{}));
      cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, cute::Int<0>{}),
                 tOrfc1_lower_copy_view(cute::_, cute::_, cute::Int<0>{}));

      // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
      shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, cute::Int<0>{}),
                                tOrfc1_lower(cute::_, cute::_, cute::Int<0>{}));

      // Reconstruct FP16 weights from dual FP8 buffers
      reconstruct(tOrfc1_upper(cute::_, cute::_, cute::Int<0>{}),
                  tOrfc1_lower(cute::_, cute::_, cute::Int<0>{}),
                  tmp1(cute::_, cute::_, cute::Int<0>{}), tmp1(cute::_, cute::_, cute::Int<1>{}));
    }
    // k loop for mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_PACKED_MAX>{}, [&](auto k_block) {
        if (k_block == K_BLOCK_PACKED_MAX - 1) {
          tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_upper_p = tOsfc1_upper(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_lower_p = tOsfc1_lower(cute::_, cute::_, cute::_, smem_pipe_read);
          cute::cp_async_wait<KT::Stages - 2>();
          __syncthreads();
        }
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_PACKED_MAX;
        auto k_logical_next_0 = k_block_next * cute::_2{};
        auto k_logical_next_1 = k_block_next * cute::_2{} + cute::_1{};
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_0),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_0));
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_1),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_1));
        cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, k_block_next),
                   tOrfc1_upper_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, k_block_next),
                   tOrfc1_lower_copy_view(cute::_, cute::_, k_block_next));
        // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
        shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, k_block_next),
                                  tOrfc1_lower(cute::_, cute::_, k_block_next));
        reconstruct(tOrfc1_upper(cute::_, cute::_, k_block_next),
                    tOrfc1_lower(cute::_, cute::_, k_block_next),
                    tmp1(cute::_, cute::_, k_logical_next_0),
                    tmp1(cute::_, cute::_, k_logical_next_1));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0) {
          cute::copy_if(gmem_tiled_copy_A, tInputpInput,
                        tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                        tInputsInput(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy(gmem_tiled_copy_B, tfc1gfc1_upper(cute::_, cute::_, cute::_, *k_tile_iter),
                     tfc1sfc1_upper(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy(gmem_tiled_copy_B, tfc1gfc1_lower(cute::_, cute::_, cute::_, *k_tile_iter),
                     tfc1sfc1_lower(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::cp_async_fence();
          if (k_tile_count - 1 > 0) {
            ++k_tile_iter;
          }

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
        }
        // Thread-level register gemm for k_block
        auto k_logical_0 = k_block * cute::_2{};
        auto k_logical_1 = k_block * cute::_2{} + cute::_1{};
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_0),
                   tmp1(cute::_, cute::_, k_logical_0), accum);
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_1),
                   tmp1(cute::_, cute::_, k_logical_1), accum);
      });
    }

    // load tail
    cute::for_each(cute::make_int_sequence<KT::Stages - 2>{}, [&](auto WaitIndex) {
      k_tile_count--;
      using WaitIndex_t = decltype(WaitIndex);
      cute::for_each(cute::make_int_sequence<K_BLOCK_PACKED_MAX>{}, [&](auto k_block) {
        if (k_block == K_BLOCK_PACKED_MAX - 1) {
          tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_upper_p = tOsfc1_upper(cute::_, cute::_, cute::_, smem_pipe_read);
          tOsfc1_lower_p = tOsfc1_lower(cute::_, cute::_, cute::_, smem_pipe_read);
          cute::cp_async_wait<KT::Stages - 3 - WaitIndex_t::value>();
          __syncthreads();
        }
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_PACKED_MAX;
        auto k_logical_next_0 = k_block_next * cute::_2{};
        auto k_logical_next_1 = k_block_next * cute::_2{} + cute::_1{};
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_0),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_0));
        cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_1),
                   tOrInput_copy_view(cute::_, cute::_, k_logical_next_1));
        cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, k_block_next),
                   tOrfc1_upper_copy_view(cute::_, cute::_, k_block_next));
        cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, k_block_next),
                   tOrfc1_lower_copy_view(cute::_, cute::_, k_block_next));
        // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
        shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, k_block_next),
                                  tOrfc1_lower(cute::_, cute::_, k_block_next));
        reconstruct(tOrfc1_upper(cute::_, cute::_, k_block_next),
                    tOrfc1_lower(cute::_, cute::_, k_block_next),
                    tmp1(cute::_, cute::_, k_logical_next_0),
                    tmp1(cute::_, cute::_, k_logical_next_1));
        if (k_block == 0) {
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
        }
        // Thread-level register gemm for k_block
        auto k_logical_0 = k_block * cute::_2{};
        auto k_logical_1 = k_block * cute::_2{} + cute::_1{};
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_0),
                   tmp1(cute::_, cute::_, k_logical_0), accum);
        cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_1),
                   tmp1(cute::_, cute::_, k_logical_1), accum);
      });
    });
    // mma tail
    cute::for_each(cute::make_int_sequence<K_BLOCK_PACKED_MAX>{}, [&](auto k_block) {
      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_PACKED_MAX;
      auto k_logical_next_0 = k_block_next * cute::_2{};
      auto k_logical_next_1 = k_block_next * cute::_2{} + cute::_1{};
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_0),
                 tOrInput_copy_view(cute::_, cute::_, k_logical_next_0));
      cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_logical_next_1),
                 tOrInput_copy_view(cute::_, cute::_, k_logical_next_1));
      cute::copy(smem_tiled_copy_B, tOsfc1_upper_p(cute::_, cute::_, k_block_next),
                 tOrfc1_upper_copy_view(cute::_, cute::_, k_block_next));
      cute::copy(smem_tiled_copy_B, tOsfc1_lower_p(cute::_, cute::_, k_block_next),
                 tOrfc1_lower_copy_view(cute::_, cute::_, k_block_next));
      // Shuffle fragments to match FP16 MMA thread-value layout before reconstruction
      shuffle_fragment_B_paired(tOrfc1_upper(cute::_, cute::_, k_block_next),
                                tOrfc1_lower(cute::_, cute::_, k_block_next));
      reconstruct(tOrfc1_upper(cute::_, cute::_, k_block_next),
                  tOrfc1_lower(cute::_, cute::_, k_block_next),
                  tmp1(cute::_, cute::_, k_logical_next_0),
                  tmp1(cute::_, cute::_, k_logical_next_1));
      // Thread-level register gemm for k_block
      auto k_logical_0 = k_block * cute::_2{};
      auto k_logical_1 = k_block * cute::_2{} + cute::_1{};
      cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_0),
                 tmp1(cute::_, cute::_, k_logical_0), accum);
      cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_logical_1),
                 tmp1(cute::_, cute::_, k_logical_1), accum);
    });

    // (2) add bias if it has..
    if (params.ptr_bias != nullptr) {
      cute::Tensor gBias =
          gBias_mn(cute::_, cute::_, bias_is_broadcast ? 0 : block_m_idx, block_n_idx);
      cute::Tensor tOgBias = thr_mma.partition_C(gBias);
      for (int i = 0; i < cute::size(accum); i++) {
        accum(i) += tOgBias(i);
      }
    }

    // (3) apply activation function
    using ActivationFn = typename KT::ActivationFn;
    ActivationFn fn{};
    CUTLASS_PRAGMA_UNROLL
    for (int temp_iter = 0; temp_iter < cute::size(accum); temp_iter++) {
      accum(temp_iter) = fn(accum(temp_iter));
    }

    // (4) push all the result to smem
    // (4.1) convert result from ElementAccum to ElementOutput
    cute::Tensor temp_accum = util_convert_type<KT::ElementOutput>(accum);
    // (4.2) retile rf and smem for copy back..
    auto smem_tiled_copy_O = cute::make_tiled_copy_C(typename KT::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
    cute::Tensor taccumrO = smem_thr_copy_O.retile_S(temp_accum);
    cute::Tensor taccumsO = smem_thr_copy_O.partition_D(sO);

    // (4.3) copy rf result to smem
    cute::copy(smem_tiled_copy_O, taccumrO, taccumsO);
    __syncthreads();

    // (4.4) sO -> rO -> gO
    typename KT::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    cute::Tensor gO = gOutput_mn(cute::_, cute::_, block_m_idx, block_n_idx);
    auto tOsO = gmem_thr_copy_O.partition_S(sO);
    auto tOgO = gmem_thr_copy_O.partition_D(gO);
    cute::Tensor cOutput = cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(typename KT::TileShape{}), cute::size<1>(typename KT::TileShape{})));
    cute::Tensor tOcO = gmem_thr_copy_O.partition_D(cOutput);
    cute::Tensor tOrO = cute::make_tensor<KT::ElementOutput>(cute::shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < cute::size<1>(tOgO); ++m) {
      if (cute::get<0>(tOcO(0, m, 0)) < residue_m) {
        cute::copy(gmem_tiled_copy_O, tOrO(cute::_, m, cute::_), tOgO(cute::_, m, cute::_));
      }
    }
  }
};

}  // namespace dual_weight_fused_moe
