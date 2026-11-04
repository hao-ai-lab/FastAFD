#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/util_type.cuh>

#if defined(CUDA_VERSION)
#define MINISGL_CUDA_VERSION CUDA_VERSION
#elif defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
#define MINISGL_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10)
#else
#define MINISGL_CUDA_VERSION 0
#endif

#if MINISGL_CUDA_VERSION >= 12090
#include <cuda/functional>
#endif

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

inline constexpr int kWarpSize = 32;

#if MINISGL_CUDA_VERSION >= 12090
using MaxReduceOp = cuda::maximum<>;
#else
using MaxReduceOp = cub::Max;
#endif

using cub_kvp = cub::KeyValuePair<int, float>;

template <typename T, int N, int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
  T data[N];
};

template <typename T>
struct TypeTag {
  using type = T;
};

template <typename T>
__device__ __forceinline__ float input_to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float input_to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ auto shfl_xor_width(
    unsigned mask, T value, int lane_mask, int width) -> T {
  return __shfl_xor_sync(mask, value, lane_mask, width);
}

template <int TPB>
__launch_bounds__(TPB) __global__ void moe_softmax_kernel(
    const float* input, float* output, int num_cols) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  __shared__ float normalizing_factor;
  __shared__ float row_max;

  const int row_offset = blockIdx.x * num_cols;
  float thread_data = -FLT_MAX;

  for (int col = threadIdx.x; col < num_cols; col += TPB) {
    const int idx = row_offset + col;
    const float val = input[idx];
    output[idx] = val;
    thread_data = max(val, thread_data);
  }

  const float max_elem = BlockReduce(tmp_storage).Reduce(thread_data, MaxReduceOp());
  if (threadIdx.x == 0) {
    row_max = max_elem;
  }
  __syncthreads();

  thread_data = 0.0f;
  for (int col = threadIdx.x; col < num_cols; col += TPB) {
    const int idx = row_offset + col;
    thread_data += expf(output[idx] - row_max);
  }

  const float denom = BlockReduce(tmp_storage).Sum(thread_data);
  if (threadIdx.x == 0) {
    normalizing_factor = 1.0f / denom;
  }
  __syncthreads();

  for (int col = threadIdx.x; col < num_cols; col += TPB) {
    const int idx = row_offset + col;
    output[idx] = expf(output[idx] - row_max) * normalizing_factor;
  }
}

namespace moe {

struct TopKPair {
  static constexpr int PAIR = 2;
  static constexpr int MAX_INDEX = 0;
  cub_kvp max;
  cub_kvp second_max;
};

struct TopKPairArgMax {
  __device__ __forceinline__ auto operator()(
      const TopKPair& lhs, const TopKPair& rhs) const -> TopKPair {
    cub_kvp global_max;
    cub_kvp global_second_max;

    if (lhs.max.value > rhs.max.value) {
      global_max = lhs.max;
    } else {
      global_max = rhs.max;
    }

    if (global_max.key == lhs.max.key) {
      global_second_max =
          (lhs.second_max.value > rhs.max.value) ? lhs.second_max : rhs.max;
    } else {
      global_second_max =
          (rhs.second_max.value > lhs.max.value) ? rhs.second_max : lhs.max;
    }
    return TopKPair{global_max, global_second_max};
  }
};

}  // namespace moe

template <int TPB>
__launch_bounds__(TPB) __global__ void moe_topk_fast_kernel(
    float* inputs_after_softmax, float* output, int* indices, int num_experts,
    int topk, bool renormalize) {
  using BlockReduce = cub::BlockReduce<moe::TopKPair, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  moe::TopKPair thread_pair;

  const int row = blockIdx.x;
  const int row_offset = row * num_experts;
  float row_sum = 0.0f;

  for (int k_idx = 0; k_idx < (topk + moe::TopKPair::PAIR - 1) / moe::TopKPair::PAIR;
       ++k_idx) {
    thread_pair.max = cub_kvp{0, -1.0f};
    thread_pair.second_max = cub_kvp{0, -1.0f};

    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const auto kvp = cub_kvp{expert, inputs_after_softmax[row_offset + expert]};
      if (kvp.value > thread_pair.max.value) {
        thread_pair.second_max = thread_pair.max;
        thread_pair.max = kvp;
      } else if (kvp.value > thread_pair.second_max.value) {
        thread_pair.second_max = kvp;
      }
    }

    const auto result_pair =
        BlockReduce(tmp_storage).Reduce(thread_pair, moe::TopKPairArgMax{});
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < moe::TopKPair::PAIR; ++i) {
        if (k_idx * 2 + i >= topk) {
          break;
        }
        const auto result =
            (i == moe::TopKPair::MAX_INDEX) ? result_pair.max : result_pair.second_max;
        const int expert = result.key;
        inputs_after_softmax[row_offset + expert] = -1.0f;
        const int out_idx = topk * row + k_idx * 2 + i;
        output[out_idx] = result.value;
        indices[out_idx] = expert;
        row_sum += result.value;
      }
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    const float inv = 1.0f / row_sum;
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      output[topk * row + k_idx] *= inv;
    }
  }
}

template <int TPB>
__launch_bounds__(TPB) __global__ void moe_topk_kernel(
    float* inputs_after_softmax, float* output, int* indices, int num_experts,
    int topk, bool renormalize) {
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;
  const int row = blockIdx.x;
  const int row_offset = row * num_experts;
  float row_sum = 0.0f;

#pragma unroll 8
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    thread_kvp = cub_kvp{0, -1.0f};

    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const auto kvp = cub_kvp{expert, inputs_after_softmax[row_offset + expert]};
      thread_kvp = arg_max(kvp, thread_kvp);
    }

    const auto result_kvp = BlockReduce(tmp_storage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int expert = result_kvp.key;
      const int out_idx = topk * row + k_idx;
      output[out_idx] = result_kvp.value;
      indices[out_idx] = expert;
      row_sum += result_kvp.value;
      inputs_after_softmax[row_offset + expert] = -1.0f;
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    const float inv = 1.0f / row_sum;
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      output[topk * row + k_idx] *= inv;
    }
  }
}

namespace detail {

template <int EXPERTS, int BYTES_PER_LDG, int INPUT_BYTES = sizeof(float)>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / INPUT_BYTES;
  static constexpr int VECS_PER_THREAD =
      std::max(1, EXPERTS / (ELTS_PER_LDG * kWarpSize));
  static constexpr int VPT = VECS_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = kWarpSize / THREADS_PER_ROW;
};

}  // namespace detail

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA * kWarpSize) __global__ void topk_gating_softmax_kernel(
    const float* input, float* output, int* indices, int num_rows, int topk,
    bool renormalize) {
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be <= 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / static_cast<int>(sizeof(float));
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  static constexpr int ELTS_PER_WARP = kWarpSize * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row >= num_rows) {
    return;
  }

  const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  using AccessType = AlignedArray<float, ELTS_PER_LDG>;

  float row_chunk[VPT];
  auto* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
  const auto* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(
        thread_max,
        shfl_xor_width(0xffffffff, thread_max, mask, THREADS_PER_ROW));
  }

  float row_sum = 0.0f;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += shfl_xor_width(0xffffffff, row_sum, mask, THREADS_PER_ROW);
  }

  const float reciprocal_row_sum = 1.0f / row_sum;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] *= reciprocal_row_sum;
  }

  int start_col = first_elt_read_by_thread;
  float row_sum_for_renormalize = 0.0f;

  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        const float val = row_chunk[ldg * ELTS_PER_LDG + ii];
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      const float other_max =
          shfl_xor_width(0xffffffff, max_val, mask, THREADS_PER_ROW);
      const int other_expert =
          shfl_xor_width(0xffffffff, expert, mask, THREADS_PER_ROW);
      if (other_max > max_val ||
          (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const int idx = topk * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = expert;
      row_sum_for_renormalize += max_val;
    }

    if (k_idx + 1 < topk) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
            -10000.0f;
      }
    }
  }

  if (renormalize && thread_group_idx == 0) {
    const float inv = 1.0f / row_sum_for_renormalize;
#pragma unroll
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      output[topk * thread_row + k_idx] *= inv;
    }
  }
}

template <typename InputT, typename IndexT, typename MapT, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA * kWarpSize) __global__ void topk_gating_softmax_group_local_kernel(
    const InputT* __restrict__ input, float* __restrict__ output,
    IndexT* __restrict__ indices, const MapT* __restrict__ expert_map,
    const int64_t* __restrict__ valid_token_count, int valid_count_size,
    int num_rows, int topk,
    bool renormalize) {
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be <= 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / static_cast<int>(sizeof(InputT));
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  static constexpr int ELTS_PER_WARP = kWarpSize * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row >= num_rows) {
    return;
  }

  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  bool row_is_valid = true;
  if (valid_count_size != 0) {
    const int valid_count = min(static_cast<int>(valid_token_count[0]), num_rows);
    row_is_valid = thread_row < valid_count;
  }
  if (!row_is_valid) {
    if (thread_group_idx == 0) {
#pragma unroll
      for (int k_idx = 0; k_idx < topk; ++k_idx) {
        const int idx = topk * thread_row + k_idx;
        output[idx] = 0.0f;
        indices[idx] = static_cast<IndexT>(-1);
      }
    }
    return;
  }

  const InputT* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const InputT* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  using AccessType = AlignedArray<InputT, ELTS_PER_LDG>;

  InputT input_chunk[VPT];
  float row_chunk[VPT];
  auto* input_chunk_vec_ptr = reinterpret_cast<AccessType*>(&input_chunk);
  const auto* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    input_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = input_to_float(input_chunk[ii]);
  }

  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(
        thread_max,
        shfl_xor_width(0xffffffff, thread_max, mask, THREADS_PER_ROW));
  }

  float row_sum = 0.0f;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = __expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += shfl_xor_width(0xffffffff, row_sum, mask, THREADS_PER_ROW);
  }

  const float reciprocal_row_sum = __frcp_rn(row_sum);
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] *= reciprocal_row_sum;
  }

  int start_col = first_elt_read_by_thread;
  float row_sum_for_renormalize = 0.0f;

  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        const float val = row_chunk[ldg * ELTS_PER_LDG + ii];
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      const float other_max =
          shfl_xor_width(0xffffffff, max_val, mask, THREADS_PER_ROW);
      const int other_expert =
          shfl_xor_width(0xffffffff, expert, mask, THREADS_PER_ROW);
      if (other_max > max_val ||
          (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const int idx = topk * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = static_cast<IndexT>(__ldg(expert_map + expert));
      row_sum_for_renormalize += max_val;
    }

    if (k_idx + 1 < topk) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
            -10000.0f;
      }
    }
  }

  if (renormalize && thread_group_idx == 0) {
    const float inv = __frcp_rn(row_sum_for_renormalize);
#pragma unroll
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      output[topk * thread_row + k_idx] *= inv;
    }
  }
}

template <int EXPERTS, int WARPS_PER_TB>
void topk_gating_softmax_launcher_helper(
    const float* gating_output, float* topk_weights, int* topk_indices,
    int num_tokens, int topk, bool renormalize, cudaStream_t stream) {
  static constexpr int kMaxBytesPerLdg = 16;
  static constexpr int kBytesPerLdg =
      std::min(kMaxBytesPerLdg, static_cast<int>(sizeof(float)) * EXPERTS);
  using Constants = detail::TopkConstants<EXPERTS, kBytesPerLdg>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_tokens + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(kWarpSize, WARPS_PER_TB);
  topk_gating_softmax_kernel<VPT, EXPERTS, WARPS_PER_TB, kBytesPerLdg>
      <<<num_blocks, block_dim, 0, stream>>>(
          gating_output, topk_weights, topk_indices, num_tokens, topk,
          renormalize);
}

#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB)                               \
  topk_gating_softmax_launcher_helper<NUM_EXPERTS, WARPS_PER_TB>(               \
      gating_output, topk_weights, topk_indices, num_tokens, topk, renormalize, \
      stream)

template <typename InputT, typename IndexT, typename MapT, int EXPERTS, int WARPS_PER_TB>
void topk_gating_softmax_group_local_launcher_helper(
    const InputT* gating_output, float* topk_weights, IndexT* topk_indices,
    const MapT* expert_map, const int64_t* valid_token_count, int valid_count_size,
    int num_tokens, int topk, bool renormalize, cudaStream_t stream) {
  static constexpr int kMaxBytesPerLdg = 16;
  static constexpr int kBytesPerLdg =
      std::min(kMaxBytesPerLdg, static_cast<int>(sizeof(InputT)) * EXPERTS);
  using Constants =
      detail::TopkConstants<EXPERTS, kBytesPerLdg, static_cast<int>(sizeof(InputT))>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_tokens + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(kWarpSize, WARPS_PER_TB);
  topk_gating_softmax_group_local_kernel<
      InputT,
      IndexT,
      MapT,
      VPT,
      EXPERTS,
      WARPS_PER_TB,
      kBytesPerLdg><<<num_blocks, block_dim, 0, stream>>>(
      gating_output,
      topk_weights,
      topk_indices,
      expert_map,
      valid_token_count,
      valid_count_size,
      num_tokens,
      topk,
      renormalize);
}

#define LAUNCH_GROUP_LOCAL_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB)                   \
  topk_gating_softmax_group_local_launcher_helper<InputT, IndexT, MapT,         \
                                                  NUM_EXPERTS, WARPS_PER_TB>(   \
      gating_output, topk_weights, topk_indices, expert_map, valid_token_count, \
      valid_count_size, num_tokens, topk, renormalize, stream)

// ============================================================================
// V2: topk gating with selectable scoring (softmax | sigmoid) + optional bias
// correction. Bias is added to the score *only* for the argmax comparison; the
// returned weights are the unbiased scores (matches vLLM / TRT-LLM / DeepSeek
// group-routing semantics).
//   - SOFTMAX: row-wise softmax then argmax (existing path)
//   - SIGMOID: element-wise σ(x) = 1/(1+e^-x), then argmax over σ(x) (+ bias)
// Warp-per-row fused path, power-of-2 NUM_EXPERTS only (≤ 256). Caller falls
// back to block-reduce path for non-power-of-2 via existing helpers.
// ============================================================================
enum class ScoringFunc : int { SOFTMAX = 0, SIGMOID = 1, IDENTITY = 2 };

template <typename InputT, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,
          ScoringFunc SF, bool WITH_BIAS>
__launch_bounds__(WARPS_PER_CTA * kWarpSize) __global__ void
topk_gating_kernel_v2(
    const InputT* __restrict__ input,
    const float* __restrict__ bias,  // (NUM_EXPERTS,) or nullptr when !WITH_BIAS
    float* __restrict__ output,
    int* __restrict__ indices,
    float* __restrict__ staged_output,
    int num_rows,
    int topk,
    bool renormalize) {
  ::host::minisgl_pdl_entry();
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16);

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / static_cast<int>(sizeof(InputT));
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  static constexpr int ELTS_PER_WARP = kWarpSize * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row >= num_rows) return;

  const InputT* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const InputT* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  using InputAccessType = AlignedArray<InputT, ELTS_PER_LDG>;

  InputT input_chunk[VPT];
  float row_chunk[VPT];
  auto* input_chunk_vec_ptr = reinterpret_cast<InputAccessType*>(&input_chunk);
  const auto* vec_thread_read_ptr =
      reinterpret_cast<const InputAccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    input_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = input_to_float(input_chunk[ii]);
  }

  // ---- Scoring ----
  if constexpr (SF == ScoringFunc::SOFTMAX) {
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) thread_max = max(thread_max, row_chunk[ii]);
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      thread_max = max(thread_max,
                       shfl_xor_width(0xffffffff, thread_max, mask, THREADS_PER_ROW));
    }
    float row_sum = 0.0f;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = __expf(row_chunk[ii] - thread_max);
      row_sum += row_chunk[ii];
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += shfl_xor_width(0xffffffff, row_sum, mask, THREADS_PER_ROW);
    }
    const float inv_sum = __frcp_rn(row_sum);
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) row_chunk[ii] *= inv_sum;
  } else if constexpr (SF == ScoringFunc::SIGMOID) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = __frcp_rn(1.0f + __expf(-row_chunk[ii]));
    }
  } else {  // IDENTITY: input already contains the unbiased routing score.
  }

  // ---- Load bias slice if applicable ----
  float bias_chunk[VPT];
  if constexpr (WITH_BIAS) {
#pragma unroll
    for (int ldg = 0, col = first_elt_read_by_thread; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        bias_chunk[ldg * ELTS_PER_LDG + ii] = bias[col + ii];
      }
    }
  }

  // ---- Top-K argmax over (score + bias if present); output unbiased score ----
  int start_col = first_elt_read_by_thread;
  float row_sum_for_renormalize = 0.0f;

  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    float max_score = WITH_BIAS ? row_chunk[0] + bias_chunk[0] : row_chunk[0];
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        const int idx_in_chunk = ldg * ELTS_PER_LDG + ii;
        const float val = row_chunk[idx_in_chunk];
        const float score = WITH_BIAS ? val + bias_chunk[idx_in_chunk] : val;
        if (score > max_score) {
          max_score = score;
          max_val = val;
          expert = col + ii;
        }
      }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      const float other_score =
          shfl_xor_width(0xffffffff, max_score, mask, THREADS_PER_ROW);
      const float other_val =
          shfl_xor_width(0xffffffff, max_val, mask, THREADS_PER_ROW);
      const int other_expert =
          shfl_xor_width(0xffffffff, expert, mask, THREADS_PER_ROW);
      if (other_score > max_score ||
          (other_score == max_score && other_expert < expert)) {
        max_score = other_score;
        max_val = other_val;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const int idx = topk * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = expert;
      row_sum_for_renormalize += max_val;
    }

    if (k_idx + 1 < topk) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        const int chunk_idx = ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert;
        // Clear both bias and score so this expert can't win again.
        row_chunk[chunk_idx] = -10000.0f;
        if constexpr (WITH_BIAS) bias_chunk[chunk_idx] = -10000.0f;
      }
    }
  }

  if (renormalize && thread_group_idx == 0) {
    const float inv = 1.0f / row_sum_for_renormalize;
#pragma unroll 8
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      output[topk * thread_row + k_idx] *= inv;
    }
  }
  if (staged_output != nullptr && thread_group_idx == 0) {
#pragma unroll 8
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      const int idx = topk * thread_row + k_idx;
      staged_output[idx] = output[idx];
    }
  }
}

template <typename InputT, int VPT, int NUM_EXPERTS, int PADDED_EXPERTS,
          int WARPS_PER_CTA, ScoringFunc SF, bool WITH_BIAS>
__launch_bounds__(WARPS_PER_CTA * kWarpSize) __global__ void
topk_gating_kernel_v2_padded(
    const InputT* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int* __restrict__ indices,
    float* __restrict__ staged_output,
    int num_rows,
    int topk,
    bool renormalize) {
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(PADDED_EXPERTS == (PADDED_EXPERTS & -PADDED_EXPERTS),
                "PADDED_EXPERTS must be power of 2");
  static_assert(PADDED_EXPERTS % VPT == 0, "invalid padded topk shape");

  static constexpr int THREADS_PER_ROW = PADDED_EXPERTS / VPT;
  static_assert(THREADS_PER_ROW == kWarpSize,
                "padded v2 currently maps one full warp to one row");
  static constexpr int ROWS_PER_WARP = 1;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  const int thread_row = blockIdx.x * ROWS_PER_CTA + threadIdx.y;
  if (thread_row >= num_rows) return;

  const int lane = threadIdx.x;
  const int start_col = lane * VPT;
  const InputT* row_ptr = input + thread_row * NUM_EXPERTS;

  float row_chunk[VPT];
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    const int col = start_col + ii;
    row_chunk[ii] = col < NUM_EXPERTS ? input_to_float(row_ptr[col]) : -FLT_MAX;
  }

  if constexpr (SF == ScoringFunc::SOFTMAX) {
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) thread_max = max(thread_max, row_chunk[ii]);
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
      thread_max = max(thread_max, shfl_xor_width(0xffffffff, thread_max, mask, kWarpSize));
    }
    float row_sum = 0.0f;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      const int col = start_col + ii;
      row_chunk[ii] = col < NUM_EXPERTS ? __expf(row_chunk[ii] - thread_max) : 0.0f;
      row_sum += row_chunk[ii];
    }
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
      row_sum += shfl_xor_width(0xffffffff, row_sum, mask, kWarpSize);
    }
    const float inv_sum = __frcp_rn(row_sum);
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) row_chunk[ii] *= inv_sum;
  } else if constexpr (SF == ScoringFunc::SIGMOID) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      const int col = start_col + ii;
      row_chunk[ii] = col < NUM_EXPERTS
          ? __frcp_rn(1.0f + __expf(-row_chunk[ii]))
          : 0.0f;
    }
  } else {  // IDENTITY: input already contains the unbiased routing score.
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      const int col = start_col + ii;
      row_chunk[ii] = col < NUM_EXPERTS ? row_chunk[ii] : 0.0f;
    }
  }

  float bias_chunk[VPT];
  if constexpr (WITH_BIAS) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      const int col = start_col + ii;
      bias_chunk[ii] = col < NUM_EXPERTS ? bias[col] : -FLT_MAX;
    }
  }

  float row_sum_for_renormalize = 0.0f;
#pragma unroll 8
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    float max_score = WITH_BIAS ? row_chunk[0] + bias_chunk[0] : row_chunk[0];
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      const int col = start_col + ii;
      const float val = row_chunk[ii];
      const float score = WITH_BIAS ? val + bias_chunk[ii] : val;
      if (score > max_score ||
          (score == max_score && col < expert)) {
        max_score = score;
        max_val = val;
        expert = col;
      }
    }

#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
      const float other_score = shfl_xor_width(0xffffffff, max_score, mask, kWarpSize);
      const float other_val = shfl_xor_width(0xffffffff, max_val, mask, kWarpSize);
      const int other_expert = shfl_xor_width(0xffffffff, expert, mask, kWarpSize);
      if (other_score > max_score ||
          (other_score == max_score && other_expert < expert)) {
        max_score = other_score;
        max_val = other_val;
        expert = other_expert;
      }
    }

    if (lane == 0) {
      const int idx = topk * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = expert;
      row_sum_for_renormalize += max_val;
    }

    if (k_idx + 1 < topk) {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        if (start_col + ii == expert) {
          row_chunk[ii] = -10000.0f;
          if constexpr (WITH_BIAS) bias_chunk[ii] = -10000.0f;
        }
      }
    }
  }

  if (renormalize && lane == 0) {
    const float inv = 1.0f / row_sum_for_renormalize;
#pragma unroll 8
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      output[topk * thread_row + k_idx] *= inv;
    }
  }
  if (staged_output != nullptr && lane == 0) {
#pragma unroll 8
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      const int idx = topk * thread_row + k_idx;
      staged_output[idx] = output[idx];
    }
  }
}

template <typename InputT, int EXPERTS, int WARPS_PER_TB, ScoringFunc SF, bool WITH_BIAS>
void topk_gating_v2_launcher_helper(
    const InputT* gating_output, const float* bias, float* topk_weights,
    int* topk_indices, float* staged_weights,
    int num_tokens, int topk, bool renormalize,
    cudaStream_t stream) {
  static constexpr int kMaxBytesPerLdg = 16;
  static constexpr int kBytesPerLdg =
      std::min(kMaxBytesPerLdg, static_cast<int>(sizeof(InputT)) * EXPERTS);
  using Constants =
      detail::TopkConstants<EXPERTS, kBytesPerLdg, static_cast<int>(sizeof(InputT))>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_tokens + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
  dim3 block_dim(kWarpSize, WARPS_PER_TB);
  ::host::LaunchKernel(dim3(num_blocks), block_dim, stream).with_attr(::host::minisgl_pdl_enabled())(
      topk_gating_kernel_v2<InputT, VPT, EXPERTS, WARPS_PER_TB, kBytesPerLdg, SF, WITH_BIAS>,
          gating_output, bias, topk_weights, topk_indices, staged_weights,
          num_tokens, topk,
          renormalize);
}

template <typename InputT, int EXPERTS, int PADDED_EXPERTS, int WARPS_PER_TB,
          ScoringFunc SF, bool WITH_BIAS>
void topk_gating_v2_padded_launcher_helper(
    const InputT* gating_output, const float* bias, float* topk_weights,
    int* topk_indices, float* staged_weights,
    int num_tokens, int topk, bool renormalize,
    cudaStream_t stream) {
  static constexpr int VPT = 8;
  const int num_blocks = (num_tokens + WARPS_PER_TB - 1) / WARPS_PER_TB;
  dim3 block_dim(kWarpSize, WARPS_PER_TB);
  topk_gating_kernel_v2_padded<InputT, VPT, EXPERTS, PADDED_EXPERTS,
                               WARPS_PER_TB, SF, WITH_BIAS>
      <<<num_blocks, block_dim, 0, stream>>>(
          gating_output, bias, topk_weights, topk_indices, staged_weights,
          num_tokens, topk,
          renormalize);
}

#define LAUNCH_V2_INNER(EXPERTS, WARPS_PER_TB, SF, WITH_BIAS)            \
  topk_gating_v2_launcher_helper<InputT, EXPERTS, WARPS_PER_TB, SF, WITH_BIAS>( \
      gating_output, bias, topk_weights, topk_indices, staged_weights, num_tokens, topk, \
      renormalize, stream)

#define LAUNCH_V2_PADDED_INNER(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, SF, WITH_BIAS) \
  topk_gating_v2_padded_launcher_helper<InputT, EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, SF, WITH_BIAS>( \
      gating_output, bias, topk_weights, topk_indices, staged_weights, num_tokens, topk, \
      renormalize, stream)

#define LAUNCH_V2(EXPERTS, WARPS_PER_TB)                                       \
  do {                                                                         \
    if (scoring == ScoringFunc::SOFTMAX) {                                     \
      if (has_bias) {                                                          \
        LAUNCH_V2_INNER(EXPERTS, WARPS_PER_TB, ScoringFunc::SOFTMAX, true);    \
      } else {                                                                 \
        LAUNCH_V2_INNER(EXPERTS, WARPS_PER_TB, ScoringFunc::SOFTMAX, false);   \
      }                                                                        \
    } else if (scoring == ScoringFunc::SIGMOID) {                            \
      if (has_bias) {                                                          \
        LAUNCH_V2_INNER(EXPERTS, WARPS_PER_TB, ScoringFunc::SIGMOID, true);    \
      } else {                                                                 \
        LAUNCH_V2_INNER(EXPERTS, WARPS_PER_TB, ScoringFunc::SIGMOID, false);   \
      }                                                                        \
    } else {                                                                   \
      if (has_bias) {                                                          \
        LAUNCH_V2_INNER(EXPERTS, WARPS_PER_TB, ScoringFunc::IDENTITY, true);   \
      } else {                                                                 \
        LAUNCH_V2_INNER(EXPERTS, WARPS_PER_TB, ScoringFunc::IDENTITY, false);  \
      }                                                                        \
    }                                                                          \
  } while (0)

#define LAUNCH_V2_PADDED(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB)                \
  do {                                                                         \
    if (scoring == ScoringFunc::SOFTMAX) {                                     \
      if (has_bias) {                                                          \
        LAUNCH_V2_PADDED_INNER(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, ScoringFunc::SOFTMAX, true); \
      } else {                                                                 \
        LAUNCH_V2_PADDED_INNER(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, ScoringFunc::SOFTMAX, false); \
      }                                                                        \
    } else if (scoring == ScoringFunc::SIGMOID) {                              \
      if (has_bias) {                                                          \
        LAUNCH_V2_PADDED_INNER(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, ScoringFunc::SIGMOID, true); \
      } else {                                                                 \
        LAUNCH_V2_PADDED_INNER(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, ScoringFunc::SIGMOID, false); \
      }                                                                        \
    } else {                                                                   \
      if (has_bias) {                                                          \
        LAUNCH_V2_PADDED_INNER(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, ScoringFunc::IDENTITY, true); \
      } else {                                                                 \
        LAUNCH_V2_PADDED_INNER(EXPERTS, PADDED_EXPERTS, WARPS_PER_TB, ScoringFunc::IDENTITY, false); \
      }                                                                        \
    }                                                                          \
  } while (0)

template <typename InputT>
void topk_gating_kernel_v2_launcher(
    const InputT* gating_output, const float* bias, float* topk_weights,
    int* topk_indices, float* staged_weights,
    int num_tokens, int num_experts, int topk,
    bool renormalize, ScoringFunc scoring, cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  const bool has_bias = bias != nullptr;
  switch (num_experts) {
    case 1:   LAUNCH_V2(1,   WARPS_PER_TB); break;
    case 2:   LAUNCH_V2(2,   WARPS_PER_TB); break;
    case 4:   LAUNCH_V2(4,   WARPS_PER_TB); break;
    case 8:   LAUNCH_V2(8,   WARPS_PER_TB); break;
    case 16:  LAUNCH_V2(16,  WARPS_PER_TB); break;
    case 32:  LAUNCH_V2(32,  WARPS_PER_TB); break;
    case 64:  LAUNCH_V2(64,  WARPS_PER_TB); break;
    case 128: LAUNCH_V2(128, WARPS_PER_TB); break;
    case 160: LAUNCH_V2_PADDED(160, 256, WARPS_PER_TB); break;
    case 256: LAUNCH_V2(256, WARPS_PER_TB); break;
    default:
      host::RuntimeCheck(
          false, "topk_gating_v2 supports power-of-two num_experts ≤ 256 and 160; got ",
          num_experts);
  }
}

#undef LAUNCH_V2_PADDED
#undef LAUNCH_V2_PADDED_INNER
#undef LAUNCH_V2
#undef LAUNCH_V2_INNER

void topk_gating_softmax_kernel_launcher(
    const float* gating_output, float* topk_weights, int* topk_indices,
    float* softmax_workspace, int num_tokens, int num_experts, int topk,
    bool renormalize, cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  switch (num_experts) {
    case 1:
      LAUNCH_SOFTMAX(1, WARPS_PER_TB);
      break;
    case 2:
      LAUNCH_SOFTMAX(2, WARPS_PER_TB);
      break;
    case 4:
      LAUNCH_SOFTMAX(4, WARPS_PER_TB);
      break;
    case 8:
      LAUNCH_SOFTMAX(8, WARPS_PER_TB);
      break;
    case 16:
      LAUNCH_SOFTMAX(16, WARPS_PER_TB);
      break;
    case 32:
      LAUNCH_SOFTMAX(32, WARPS_PER_TB);
      break;
    case 64:
      LAUNCH_SOFTMAX(64, WARPS_PER_TB);
      break;
    case 128:
      LAUNCH_SOFTMAX(128, WARPS_PER_TB);
      break;
    case 256:
      LAUNCH_SOFTMAX(256, WARPS_PER_TB);
      break;
    default: {
      constexpr int TPB = 256;
      moe_softmax_kernel<TPB><<<num_tokens, TPB, 0, stream>>>(
          gating_output, softmax_workspace, num_experts);
      if (topk == 1) {
        moe_topk_kernel<TPB><<<num_tokens, TPB, 0, stream>>>(
            softmax_workspace, topk_weights, topk_indices, num_experts, topk,
            renormalize);
      } else {
        moe_topk_fast_kernel<TPB><<<num_tokens, TPB, 0, stream>>>(
            softmax_workspace, topk_weights, topk_indices, num_experts, topk,
            renormalize);
      }
    }
  }
}

template <typename InputT, typename IndexT, typename MapT>
void topk_gating_softmax_group_local_kernel_launcher(
    const InputT* gating_output, float* topk_weights, IndexT* topk_indices,
    const MapT* expert_map, const int64_t* valid_token_count, int valid_count_size,
    int num_tokens, int num_experts, int topk, bool renormalize, cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  switch (num_experts) {
    case 1:
      LAUNCH_GROUP_LOCAL_SOFTMAX(1, WARPS_PER_TB);
      break;
    case 2:
      LAUNCH_GROUP_LOCAL_SOFTMAX(2, WARPS_PER_TB);
      break;
    case 4:
      LAUNCH_GROUP_LOCAL_SOFTMAX(4, WARPS_PER_TB);
      break;
    case 8:
      LAUNCH_GROUP_LOCAL_SOFTMAX(8, WARPS_PER_TB);
      break;
    case 16:
      LAUNCH_GROUP_LOCAL_SOFTMAX(16, WARPS_PER_TB);
      break;
    case 32:
      LAUNCH_GROUP_LOCAL_SOFTMAX(32, WARPS_PER_TB);
      break;
    case 64:
      LAUNCH_GROUP_LOCAL_SOFTMAX(64, WARPS_PER_TB);
      break;
    case 128:
      LAUNCH_GROUP_LOCAL_SOFTMAX(128, WARPS_PER_TB);
      break;
    case 256:
      LAUNCH_GROUP_LOCAL_SOFTMAX(256, WARPS_PER_TB);
      break;
    case 512:
      LAUNCH_GROUP_LOCAL_SOFTMAX(512, WARPS_PER_TB);
      break;
    default:
      host::RuntimeCheck(
          false,
          "group-local topk supports only power-of-two num_experts <= 512; got ",
          num_experts);
  }
}

struct MoeTopkSoftmaxKernel {
  static void run(
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_indices,
      const tvm::ffi::TensorView gating_output,
      const tvm::ffi::TensorView softmax_workspace, bool renormalize) {
    using namespace host;

    auto num_tokens = SymbolicSize{"M"};
    auto topk = SymbolicSize{"K"};
    auto num_experts = SymbolicSize{"E"};
    auto device_ref = SymbolicDevice{};

    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(topk_weights);
    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(topk_indices);
    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .verify(gating_output);
    TensorMatcher({-1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(softmax_workspace);

    RuntimeCheck(topk.unwrap() > 0, "topk must be positive");
    RuntimeCheck(
        topk.unwrap() <= num_experts.unwrap(),
        "topk must be <= num_experts, got topk=", topk.unwrap(),
        " num_experts=", num_experts.unwrap());

    const auto num_rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto topk_value = static_cast<int>(topk.unwrap());
    if (num_rows == 0) {
      return;
    }
    const auto actual_workspace_size = softmax_workspace.size(0);

    const bool is_pow_2 =
        (experts != 0) && ((experts & (experts - 1)) == 0);
    const bool needs_workspace = !is_pow_2 || experts > 256;
    const auto required_workspace =
        needs_workspace ? static_cast<int64_t>(num_rows) * experts : 0;
    RuntimeCheck(
        actual_workspace_size >= required_workspace,
        "softmax_workspace too small: expected at least ", required_workspace,
        " but got ", actual_workspace_size);

    topk_gating_softmax_kernel_launcher(
        static_cast<const float*>(gating_output.data_ptr()),
        static_cast<float*>(topk_weights.data_ptr()),
        static_cast<int*>(topk_indices.data_ptr()),
        static_cast<float*>(softmax_workspace.data_ptr()),
        num_rows,
        experts,
        topk_value,
        renormalize,
        LaunchKernel::resolve_device(device_ref.unwrap()));
  }
};

// v2: configurable scoring (softmax | sigmoid) + optional bias correction.
// Single entry point — `scoring_func` ∈ {0=softmax, 1=sigmoid}; bias may be
// an empty tensor (size 0) to skip the bias path. Fast path supports
// power-of-two num_experts ≤ 256 plus GLM's 160-expert padded variant.
struct MoeTopkGatingV2Kernel {
  static void run(
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_indices,
      const tvm::ffi::TensorView gating_output,
      const tvm::ffi::TensorView bias,
      int64_t scoring_func,
      bool renormalize,
      tvm::ffi::Optional<tvm::ffi::TensorView> staged_weights) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto topk = SymbolicSize{"K"};
    auto num_experts = SymbolicSize{"E"};
    auto device_ref = SymbolicDevice{};

    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(topk_weights);
    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(topk_indices);
    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .verify(gating_output);
    TensorMatcher({-1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(bias);

    const auto num_rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto topk_value = static_cast<int>(topk.unwrap());
    if (num_rows == 0) return;
    const auto gating_dtype = gating_output.dtype();
    const bool use_fp32_input =
        gating_dtype.code == DLDataTypeCode::kDLFloat && gating_dtype.bits == 32;
    const bool use_bf16_input =
        gating_dtype.code == DLDataTypeCode::kDLBfloat && gating_dtype.bits == 16;
    RuntimeCheck(
        use_fp32_input || use_bf16_input,
        "topk_gating_v2 gating_output must be float32 or bfloat16, got ",
        gating_dtype);
    const bool has_bias = bias.size(0) > 0;
    if (has_bias) {
      RuntimeCheck(bias.size(0) == experts,
                   "bias size must equal num_experts, got ", bias.size(0));
    }
    RuntimeCheck(scoring_func == 0 || scoring_func == 1 || scoring_func == 2,
                 "scoring_func must be 0 (softmax), 1 (sigmoid), or 2 (identity)");
    RuntimeCheck(topk_value > 0 && topk_value <= experts,
                 "topk must be in [1, num_experts]");
    const bool is_pow_2 = (experts != 0) && ((experts & (experts - 1)) == 0);
    RuntimeCheck((is_pow_2 && experts <= 256) || experts == 160,
                 "topk_gating_v2 supports power-of-two num_experts ≤ 256 and 160");

    auto* bias_ptr = has_bias ? static_cast<const float*>(bias.data_ptr()) : nullptr;
    auto* weights_ptr = static_cast<float*>(topk_weights.data_ptr());
    auto* indices_ptr = static_cast<int*>(topk_indices.data_ptr());
    float* staged_ptr = nullptr;
    if (staged_weights.has_value()) {
      TensorMatcher({num_tokens, topk})
          .with_device<kDLCUDA>(device_ref)
          .with_dtype<float>()
          .verify(staged_weights.value());
      staged_ptr = static_cast<float*>(staged_weights.value().data_ptr());
    }
    auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    auto scoring = scoring_func == 0
        ? ScoringFunc::SOFTMAX
        : (scoring_func == 1 ? ScoringFunc::SIGMOID : ScoringFunc::IDENTITY);
    if (use_fp32_input) {
      topk_gating_kernel_v2_launcher(
          static_cast<const float*>(gating_output.data_ptr()),
          bias_ptr,
          weights_ptr,
          indices_ptr,
          staged_ptr,
          num_rows, experts, topk_value, renormalize, scoring, stream);
    } else {
      topk_gating_kernel_v2_launcher(
          static_cast<const __nv_bfloat16*>(gating_output.data_ptr()),
          bias_ptr,
          weights_ptr,
          indices_ptr,
          staged_ptr,
          num_rows, experts, topk_value, renormalize, scoring, stream);
    }
  }
};

struct MoeTopkSoftmaxGroupLocalKernel {
  static void run(
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_indices,
      const tvm::ffi::TensorView gating_output,
      const tvm::ffi::TensorView expert_map,
      const tvm::ffi::TensorView valid_token_count,
      bool renormalize) {
    using namespace host;

    auto num_tokens = SymbolicSize{"M"};
    auto topk = SymbolicSize{"K"};
    auto num_experts = SymbolicSize{"E"};
    auto topk_indices_dtype = SymbolicDType{};
    auto expert_map_dtype = SymbolicDType{};
    auto device_ref = SymbolicDevice{};

    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(topk_weights);
    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t, int64_t>(topk_indices_dtype)
        .verify(topk_indices);
    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .verify(gating_output);
    TensorMatcher({num_experts})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t, int64_t>(expert_map_dtype)
        .verify(expert_map);
    TensorMatcher({-1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int64_t>()
        .verify(valid_token_count);

    RuntimeCheck(topk.unwrap() > 0, "topk must be positive");
    RuntimeCheck(
        topk.unwrap() <= num_experts.unwrap(),
        "topk must be <= num_experts, got topk=", topk.unwrap(),
        " num_experts=", num_experts.unwrap());
    RuntimeCheck(
        valid_token_count.size(0) == 0 || valid_token_count.size(0) == 1,
        "valid_token_count must be empty or shape [1], got ",
        valid_token_count.size(0));

    const auto num_rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto topk_value = static_cast<int>(topk.unwrap());
    const auto gating_dtype = gating_output.dtype();
    const bool use_fp32_input =
        gating_dtype.code == DLDataTypeCode::kDLFloat && gating_dtype.bits == 32;
    const bool use_bf16_input =
        gating_dtype.code == DLDataTypeCode::kDLBfloat && gating_dtype.bits == 16;
    RuntimeCheck(
        use_fp32_input || use_bf16_input,
        "group-local topk gating_output must be float32 or bfloat16, got ",
        gating_dtype);
    if (num_rows == 0) {
      return;
    }

    const bool use_int32_indices = topk_indices_dtype.unwrap().bits == 32;
    const bool use_int32_map = expert_map_dtype.unwrap().bits == 32;
    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    const auto count_ptr =
        valid_token_count.size(0) == 0
            ? nullptr
            : static_cast<const int64_t*>(valid_token_count.data_ptr());
    const auto count_size = static_cast<int>(valid_token_count.size(0));

    auto launch = [&](auto input_tag) {
      using InputT = typename decltype(input_tag)::type;
      const auto* input_ptr = static_cast<const InputT*>(gating_output.data_ptr());
      if (use_int32_indices && use_int32_map) {
        topk_gating_softmax_group_local_kernel_launcher<InputT, int32_t, int32_t>(
            input_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int32_t*>(topk_indices.data_ptr()),
            static_cast<const int32_t*>(expert_map.data_ptr()),
            count_ptr,
            count_size,
            num_rows,
            experts,
            topk_value,
            renormalize,
            stream);
      } else if (use_int32_indices && !use_int32_map) {
        topk_gating_softmax_group_local_kernel_launcher<InputT, int32_t, int64_t>(
            input_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int32_t*>(topk_indices.data_ptr()),
            static_cast<const int64_t*>(expert_map.data_ptr()),
            count_ptr,
            count_size,
            num_rows,
            experts,
            topk_value,
            renormalize,
            stream);
      } else if (!use_int32_indices && use_int32_map) {
        topk_gating_softmax_group_local_kernel_launcher<InputT, int64_t, int32_t>(
            input_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int64_t*>(topk_indices.data_ptr()),
            static_cast<const int32_t*>(expert_map.data_ptr()),
            count_ptr,
            count_size,
            num_rows,
            experts,
            topk_value,
            renormalize,
            stream);
      } else {
        topk_gating_softmax_group_local_kernel_launcher<InputT, int64_t, int64_t>(
            input_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int64_t*>(topk_indices.data_ptr()),
            static_cast<const int64_t*>(expert_map.data_ptr()),
            count_ptr,
            count_size,
            num_rows,
            experts,
            topk_value,
            renormalize,
            stream);
      }
    };

    if (use_fp32_input) {
      launch(TypeTag<float>{});
    } else {
      launch(TypeTag<__nv_bfloat16>{});
    }
  }
};

}  // namespace
