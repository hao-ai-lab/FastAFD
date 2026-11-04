/*
 * Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
/*
 * VENDORED (MINISGL_QWEN35_VLLM_KERNELS=1) from vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6
 * csrc/libtorch_stable/moe/topk_softmax_kernels.cu (sha256_12 5e0eae8dbd76, git blob_12 098644a1b901):
 *   lines 41-42   MAX / MIN macros                                      (byte for byte)
 *   lines 44-74   namespace vllm::moe, AlignedArray, toFloat, ScoringFunc (byte for byte)
 *   lines 278-625 topkGating, detail::TopkConstants, topkGatingLauncherHelper (byte for byte)
 * plus csrc/cuda_compat.h:37 (WARP_SIZE 32) and :49-50 (VLLM_SHFL_XOR_SYNC_WIDTH, CUDA branch).
 * vLLM builds this file without --use_fast_math (CMakeLists.txt: _moe_C_stable_libtorch COMPILE_FLAGS
 * VLLM_GPU_FLAGS; fast math is set only on the KDA / GDN / attn-res sources), as does load_jit here.
 * The host side below replaces vLLM's torch::stable dispatch (topk_softmax, lines 822-860) for the one
 * case Qwen3.5-397B runs: bf16 gating [M, 512], int32 ids, softmax, no bias, vLLM's is_padding mask (all false for
 * FastAFD's exact batches), scaling 1.0, which vLLM sends to LAUNCH_TOPK(512, WARPS_PER_TB=4,
 * BYTES_PER_LDG_POWER_OF_2=16) (lines 671-708). vLLM also allocates an unused M x 512 fp32 workspace (512 > 256,
 * lines 832-836); no kernel touches it, so it is not allocated here.
 */
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    }
}

// Scoring function enums
enum ScoringFunc {
  SCORING_SOFTMAX = 0, // apply softmax
  SCORING_SIGMOID = 1  // apply sigmoid
};

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename IndType,
          typename InputType = float, ScoringFunc SF>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGating(const InputType* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
        const float* bias, const double routed_scaling_factor, const bool* is_padding)
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> ||
                      std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
        static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
            "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Finally, we pull in the data from global mem
    float row_chunk[VPT];

    // NOTE(zhuhaoran): dispatch different input types loading, BF16/FP16 convert to float
    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2)
                    );
                }
            }
        } else { // ELTS_PER_LDG == 1
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __bfloat162float(*scalar_ptr);
            }
        }
    } else if constexpr (std::is_same_v<InputType, __half>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__half, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __half22float2(
                        *reinterpret_cast<const __half2*>(vec.data + jj * 2)
                    );
                }
            }
        } else { // ELTS_PER_LDG == 1
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __half2float(*scalar_ptr);
            }
        }
    }

    if constexpr (SF == SCORING_SOFTMAX) {
      // First, we perform a max reduce within the thread.
      float thread_max = row_chunk[0];
#pragma unroll
      for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
      }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
      {
        thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
      }

      // From this point, thread max in all the threads have the max within the row.
      // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
      float row_sum = 0;
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
      }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
      {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
      }

      // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
      // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
      // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
      // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
      // argmax after computing the softmax.
      const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
      }
    } else if constexpr (SF == SCORING_SIGMOID) {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
      }
    }

    // Fix: clamp NaN/Inf values to 0 to prevent duplicate expert IDs.
    // NaN gating (from degenerate hidden states in CUDA graph padding) causes
    // softmax to produce all-NaN, which makes the argmax loop always pick
    // expert 0 for every top-k slot, producing duplicate expert IDs that
    // crash FlashInfer's three-step MoE sort.
    // With 0s, the argmax uses index tie-breaking to pick [0,1,2,...,k-1].
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      if (isnan(row_chunk[ii]) || isinf(row_chunk[ii])) {
        row_chunk[ii] = 0.f;
      }
    }

    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    // If bias is not null, use biased value for selection
    float row_chunk_for_choice[VPT];
    // Apply correction bias
    if (bias != nullptr) {
#pragma unroll
      for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          const int expert = first_elt_read_by_thread + ldg * COLS_PER_GROUP_LDG + ii;
          float bias_val = expert < NUM_EXPERTS ? bias[expert] : 0.0f;
          row_chunk_for_choice[ldg * ELTS_PER_LDG + ii] = row_chunk[ldg * ELTS_PER_LDG + ii] + bias_val;
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk_for_choice[ii] = row_chunk[ii];
      }
    }

    // Now, row_chunk contains the softmax / sigmoid of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // First, each thread does the local argmax
        float max_val_for_choice = row_chunk_for_choice[0];
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val_for_choice = row_chunk_for_choice[ldg * ELTS_PER_LDG + ii];
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val_for_choice > max_val_for_choice)
                {
                    max_val_for_choice = val_for_choice;
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max_for_choice = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_for_choice, mask, THREADS_PER_ROW);
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max_for_choice > max_val_for_choice || (other_max_for_choice == max_val_for_choice && other_expert < expert))
            {
                max_val_for_choice = other_max_for_choice;
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // Add a guard to ignore experts not included by this node
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const bool is_pad_row = is_padding != nullptr && is_padding[thread_row];

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = is_pad_row ? static_cast<IndType>(-1)
                                       : (should_process_row ? (expert - start_expert) : NUM_EXPERTS);
            source_rows[idx] = k_idx * num_rows + thread_row;
            if (renormalize) {
                selected_sum += max_val;
            }
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk_for_choice[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    // Apply renormalization and routed scaling factor to final weights.
    if (thread_group_idx == 0) {
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = selected_sum > 0.f ? selected_sum : 1.f;
        scale /= denom;
      }
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * thread_row + k_idx;
        output[idx] = output[idx] * scale;
      }
    }
}

namespace detail
{
// Constructs some constants needed to partition the work across threads at compile time.
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename InputType>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, int MAX_BYTES_PER_LDG, typename IndType, typename InputType, ScoringFunc SF>
void topkGatingLauncherHelper(const InputType* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
    const float* bias, const double routed_scaling_factor, cudaStream_t stream, const bool* is_padding)
{
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
    topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, IndType, InputType, SF><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert, renormalize, bias, routed_scaling_factor, is_padding);
}

} // namespace moe
} // namespace vllm

struct VllmTopkSoftmaxKernel {
  // vLLM topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize,
  //                   bias=None, is_padding) for bf16 gating with 512 experts and int32 indices.
  // is_padding: vLLM's runner passes forward_context.is_padding[:M] (VLLM_MOE_SKIP_PADDING=1 default, envs.py:1627;
  // fused_topk_router.py:19-23, 26-40; v1/worker/gpu/input_batch.py:27, 177): a persistent bool [>= M] buffer, read by
  // topkGating per written index (kernel line 551). Passed as uint8 bytes (same 1-byte layout as torch.bool).
  static void run(const tvm::ffi::TensorView topk_weights,          // [M, K] fp32 out
                  const tvm::ffi::TensorView topk_indices,          // [M, K] int32 out
                  const tvm::ffi::TensorView token_expert_indices,  // [M, K] int32 out
                  const tvm::ffi::TensorView gating_output,         // [M, 512] bf16
                  bool renormalize,
                  const tvm::ffi::TensorView is_padding) {          // [>= M] uint8 view of torch.bool
    using namespace host;
    auto m_ = SymbolicSize{"M"};
    auto k_ = SymbolicSize{"K"};
    auto e_ = SymbolicSize{"E"};
    auto p_ = SymbolicSize{"P"};
    auto device_ = SymbolicDevice{};
    auto bf16_ = SymbolicDType{};
    TensorMatcher({p_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device_).verify(is_padding);
    TensorMatcher({m_, e_}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(gating_output);
    {
      const auto dt = bf16_.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1, "gating_output must be bf16");
    }
    TensorMatcher({m_, k_}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(topk_weights);
    TensorMatcher({m_, k_}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk_indices);
    TensorMatcher({m_, k_}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(token_expert_indices);
    const auto num_tokens = static_cast<int>(m_.unwrap());
    const auto topk = static_cast<int>(k_.unwrap());
    const auto num_experts = static_cast<int>(e_.unwrap());
    RuntimeCheck(num_experts == 512, "vllm_topk_softmax is instantiated for 512 experts, got ", num_experts);
    RuntimeCheck(static_cast<int64_t>(p_.unwrap()) >= static_cast<int64_t>(num_tokens), "is_padding shorter than M");
    RuntimeCheck(is_padding.stride(0) == 1, "is_padding must be contiguous");
    if (num_tokens == 0) {
      return;
    }
    const cudaStream_t stream = LaunchKernel::resolve_device(device_.unwrap());
    static constexpr int WARPS_PER_TB = 4;              // topkGatingKernelLauncher, line 671
    static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;  // line 672
    vllm::moe::topkGatingLauncherHelper<512, WARPS_PER_TB, WARP_SIZE, BYTES_PER_LDG_POWER_OF_2, int,
                                        __nv_bfloat16, vllm::moe::SCORING_SOFTMAX>(
        static_cast<const __nv_bfloat16*>(gating_output.data_ptr()), nullptr,
        static_cast<float*>(topk_weights.data_ptr()), static_cast<int*>(topk_indices.data_ptr()),
        static_cast<int*>(token_expert_indices.data_ptr()), num_tokens, topk, 0, num_experts, renormalize,
        nullptr, 1.0, stream, reinterpret_cast<const bool*>(is_padding.data_ptr()));
  }
};
