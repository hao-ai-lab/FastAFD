#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

inline constexpr int kWarpSize = 32;
inline constexpr int kThreads = 256;
inline constexpr int kWarps = kThreads / kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < kWarps ? warp_sums[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half(value);
}

template <typename DType>
__launch_bounds__(kThreads) __global__ void minimax_qk_cross_head_norm_kernel(
    DType* __restrict__ qkv,          // [T, q_dim + 2 * kv_dim]
    const DType* __restrict__ q_w,    // [q_dim]
    const DType* __restrict__ k_w,    // [kv_dim]
    int tokens,
    int qkv_stride,
    int q_dim,
    int kv_dim,
    float eps) {
  __shared__ float smem[kWarps];
  const int token = blockIdx.x;
  const bool is_k = blockIdx.y != 0;
  const int dim = is_k ? kv_dim : q_dim;
  const int offset = is_k ? q_dim : 0;
  DType* row = qkv + static_cast<size_t>(token) * qkv_stride + offset;
  const DType* weight = is_k ? k_w : q_w;

  float sum_sq = 0.0f;
  int i = threadIdx.x;
  for (; i + blockDim.x * 3 < dim; i += blockDim.x * 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float v = to_float<DType>(row[i + u * blockDim.x]);
      sum_sq += v * v;
    }
  }
  for (; i < dim; i += blockDim.x) {
    const float v = to_float<DType>(row[i]);
    sum_sq += v * v;
  }
  const float row_sum_sq = block_sum(sum_sq, smem);
  const float inv = rsqrtf(row_sum_sq / static_cast<float>(dim) + eps);

  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = to_float<DType>(row[i]);
    const float w = to_float<DType>(weight[i]);
    row[i] = from_float<DType>(v * inv * w);
  }
}

template <typename DType>
__launch_bounds__(kThreads) __global__ void minimax_qk_cross_head_norm_rope_store_kernel(
    DType* __restrict__ qkv,            // [T, q_dim + 2 * kv_dim]
    const DType* __restrict__ q_w,      // [q_dim]
    const DType* __restrict__ k_w,      // [kv_dim]
    const void* __restrict__ positions, // [T] int32/int64
    const float* __restrict__ cos_sin_cache,
    DType* __restrict__ k_cache,        // [capacity, kv_dim]
    DType* __restrict__ v_cache,        // [capacity, kv_dim]
    const void* __restrict__ out_loc,   // [T] int32/int64
    int tokens,
    int qkv_stride,
    int q_dim,
    int kv_dim,
    int head_dim,
    int rotary_dim,
    int cache_stride,
    long kv_cache_stride,
    bool positions_i64,
    bool loc_i64,
    float eps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __shared__ float smem[kWarps];
  const int token = blockIdx.x;
  const bool is_k = blockIdx.y != 0;
  const int dim = is_k ? kv_dim : q_dim;
  const int offset = is_k ? q_dim : 0;
  DType* row = qkv + static_cast<size_t>(token) * qkv_stride + offset;
  const DType* weight = is_k ? k_w : q_w;

  float sum_sq = 0.0f;
  int i = threadIdx.x;
  for (; i + blockDim.x * 3 < dim; i += blockDim.x * 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float v = to_float<DType>(row[i + u * blockDim.x]);
      sum_sq += v * v;
    }
  }
  for (; i < dim; i += blockDim.x) {
    const float v = to_float<DType>(row[i]);
    sum_sq += v * v;
  }
  const float row_sum_sq = block_sum(sum_sq, smem);
  const float inv = rsqrtf(row_sum_sq / static_cast<float>(dim) + eps);
  const int heads = dim / head_dim;
  const int rotary_half = rotary_dim / 2;
  const long long pos = positions_i64
      ? static_cast<const int64_t*>(positions)[token]
      : static_cast<const int32_t*>(positions)[token];
  const float* rope = cos_sin_cache + pos * static_cast<long long>(cache_stride);
  const float* cos_ptr = rope;
  const float* sin_ptr = rope + rotary_half;
  DType* k_dst = nullptr;
  DType* v_dst = nullptr;
  DType* v_src = nullptr;
  if (is_k) {
    const long long cache_pos = loc_i64
        ? static_cast<const int64_t*>(out_loc)[token]
        : static_cast<const int32_t*>(out_loc)[token];
    k_dst = k_cache + cache_pos * kv_cache_stride;
    v_dst = v_cache + cache_pos * kv_cache_stride;
    v_src = qkv + static_cast<size_t>(token) * qkv_stride + q_dim + kv_dim;
  }

  // Process each rotary pair in one thread so the original low/high values are
  // read before either element is overwritten.
  const int rotary_pairs = heads * rotary_half;
  for (int p = threadIdx.x; p < rotary_pairs; p += blockDim.x) {
    const int head = p / rotary_half;
    const int j = p - head * rotary_half;
    const int lo = head * head_dim + j;
    const int hi = lo + rotary_half;
    const float x_lo = to_float<DType>(from_float<DType>(
        to_float<DType>(row[lo]) * inv * to_float<DType>(weight[lo])));
    const float x_hi = to_float<DType>(from_float<DType>(
        to_float<DType>(row[hi]) * inv * to_float<DType>(weight[hi])));
    const float c = cos_ptr[j];
    const float s = sin_ptr[j];
    const DType out_lo = from_float<DType>(x_lo * c - x_hi * s);
    const DType out_hi = from_float<DType>(x_hi * c + x_lo * s);
    row[lo] = out_lo;
    row[hi] = out_hi;
    if (is_k) {
      k_dst[lo] = out_lo;
      k_dst[hi] = out_hi;
    }
  }

  const int tail = head_dim - rotary_dim;
  if (tail > 0) {
    const int tail_elems = heads * tail;
    for (int p = threadIdx.x; p < tail_elems; p += blockDim.x) {
      const int head = p / tail;
      const int j = p - head * tail;
      const int idx = head * head_dim + rotary_dim + j;
      const DType out = from_float<DType>(
          to_float<DType>(row[idx]) * inv * to_float<DType>(weight[idx]));
      row[idx] = out;
      if (is_k) {
        k_dst[idx] = out;
      }
    }
  }

  if (!is_k) {
    return;
  }

  if ((kv_dim & 7) == 0) {
    const int vecs = kv_dim / 8;
    const auto v_src_vec = reinterpret_cast<const uint4*>(v_src);
    auto v_dst_vec = reinterpret_cast<uint4*>(v_dst);
    for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
      v_dst_vec[v] = v_src_vec[v];
    }
  } else {
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
      v_dst[i] = v_src[i];
    }
  }
}

// The same norm + RoPE + store as minimax_qk_cross_head_norm_rope_store_kernel, bit for
// bit (the sum of squares in the same per-thread order and block reduction, the
// normalized values rounded to DType before the rotation, the same expressions),
// built for decode batches where that kernel is all latency: the row comes in once
// as 16-byte vectors into shared memory, and the norm, the rotation (its partner half
// read from shared memory) and the stores go 8 elements at a time instead of one
// dependent scalar access after another.
// Decode rows up to kThreads * kMaxPer wide: the same arithmetic as
// minimax_qk_cross_head_norm_rope_store_kernel with every load of a
// phase in flight at once. Thread t still sums the squares of elements t + 256 u in order of
// u (zeros past the row add nothing), so the norm and every output are bit-identical. The
// normalized row goes to shared memory; RoPE and the stores then move 8 elements at a time.
// Needs dim <= 256 * kMaxPer and 8-element alignment (checked by the launcher).
template <typename DType, int kMaxPer>
__launch_bounds__(kThreads) __global__ void minimax_qk_norm_rope_store_regs_kernel(
    DType* __restrict__ qkv,            // [T, q_dim + 2 * kv_dim]
    const DType* __restrict__ q_w,      // [q_dim]
    const DType* __restrict__ k_w,      // [kv_dim]
    const void* __restrict__ positions, // [T] int32/int64
    const float* __restrict__ cos_sin_cache,
    DType* __restrict__ k_cache,        // [capacity, kv_dim]
    DType* __restrict__ v_cache,        // [capacity, kv_dim]
    const void* __restrict__ out_loc,   // [T] int32/int64
    int tokens,
    int qkv_stride,
    int q_dim,
    int kv_dim,
    int head_dim,
    int rotary_dim,
    int cache_stride,
    long kv_cache_stride,
    bool positions_i64,
    bool loc_i64,
    float eps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  extern __shared__ __align__(16) unsigned char smem_raw[];
  DType* normed = reinterpret_cast<DType*>(smem_raw);  // [dim]
  __shared__ float warp_sums[kWarps];
  const int token = blockIdx.x;
  const bool is_k = blockIdx.y != 0;
  const int dim = is_k ? kv_dim : q_dim;
  const int offset = is_k ? q_dim : 0;
  DType* row = qkv + static_cast<size_t>(token) * qkv_stride + offset;
  const DType* weight = is_k ? k_w : q_w;
  const int tid = threadIdx.x;

  float x[kMaxPer];
  float w[kMaxPer];
#pragma unroll
  for (int u = 0; u < kMaxPer; ++u) {
    const int i = tid + u * kThreads;
    x[u] = i < dim ? to_float<DType>(row[i]) : 0.0f;
    w[u] = i < dim ? to_float<DType>(weight[i]) : 0.0f;
  }
  float sum_sq = 0.0f;
#pragma unroll
  for (int u = 0; u < kMaxPer; ++u) {
    sum_sq += x[u] * x[u];
  }
  const float row_sum_sq = block_sum(sum_sq, warp_sums);
  const float inv = rsqrtf(row_sum_sq / static_cast<float>(dim) + eps);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // late, as in minimax_qk_norm_rope_store_vec_kernel: only the normalize + store tail is left
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
#pragma unroll
  for (int u = 0; u < kMaxPer; ++u) {
    const int i = tid + u * kThreads;
    if (i < dim) {
      normed[i] = from_float<DType>(x[u] * inv * w[u]);
    }
  }
  __syncthreads();

  const int heads = dim / head_dim;
  const int rotary_half = rotary_dim / 2;
  const long long pos = positions_i64
      ? static_cast<const int64_t*>(positions)[token]
      : static_cast<const int32_t*>(positions)[token];
  const float* cos_ptr = cos_sin_cache + pos * static_cast<long long>(cache_stride);
  const float* sin_ptr = cos_ptr + rotary_half;
  DType* k_dst = nullptr;
  if (is_k) {
    const long long cache_pos = loc_i64
        ? static_cast<const int64_t*>(out_loc)[token]
        : static_cast<const int32_t*>(out_loc)[token];
    k_dst = k_cache + cache_pos * kv_cache_stride;
    // V is copied as is.
    const auto v_src = reinterpret_cast<const uint4*>(row + kv_dim);
    auto v_dst = reinterpret_cast<uint4*>(v_cache + cache_pos * kv_cache_stride);
    for (int v = tid; v < kv_dim / 8; v += kThreads) {
      v_dst[v] = v_src[v];
    }
  }

  // Per head: rotary_half / 8 units of 8 (low, high) pairs, then (head_dim - rotary_dim) / 8
  // units of 8 unrotated elements.
  const int rot_units = rotary_half / 8;
  const int units = rot_units + (head_dim - rotary_dim) / 8;
  for (int p = tid; p < heads * units; p += kThreads) {
    const int head = p / units;
    const int unit = p - head * units;
    if (unit < rot_units) {
      const int j = unit * 8;
      const int lo = head * head_dim + j;
      const int hi = lo + rotary_half;
      const uint4 raw_lo = *reinterpret_cast<const uint4*>(normed + lo);
      const uint4 raw_hi = *reinterpret_cast<const uint4*>(normed + hi);
      const auto y_lo = reinterpret_cast<const DType*>(&raw_lo);
      const auto y_hi = reinterpret_cast<const DType*>(&raw_hi);
      float c[8], s[8];
      *reinterpret_cast<float4*>(c) = *reinterpret_cast<const float4*>(cos_ptr + j);
      *reinterpret_cast<float4*>(c + 4) = *reinterpret_cast<const float4*>(cos_ptr + j + 4);
      *reinterpret_cast<float4*>(s) = *reinterpret_cast<const float4*>(sin_ptr + j);
      *reinterpret_cast<float4*>(s + 4) = *reinterpret_cast<const float4*>(sin_ptr + j + 4);
      uint4 out_lo, out_hi;
      auto o_lo = reinterpret_cast<DType*>(&out_lo);
      auto o_hi = reinterpret_cast<DType*>(&out_hi);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float x_lo = to_float<DType>(y_lo[q]);
        const float x_hi = to_float<DType>(y_hi[q]);
        o_lo[q] = from_float<DType>(x_lo * c[q] - x_hi * s[q]);
        o_hi[q] = from_float<DType>(x_hi * c[q] + x_lo * s[q]);
      }
      *reinterpret_cast<uint4*>(row + lo) = out_lo;
      *reinterpret_cast<uint4*>(row + hi) = out_hi;
      if (is_k) {
        *reinterpret_cast<uint4*>(k_dst + lo) = out_lo;
        *reinterpret_cast<uint4*>(k_dst + hi) = out_hi;
      }
    } else {
      const int idx = head * head_dim + rotary_dim + (unit - rot_units) * 8;
      const uint4 out = *reinterpret_cast<const uint4*>(normed + idx);
      *reinterpret_cast<uint4*>(row + idx) = out;
      if (is_k) {
        *reinterpret_cast<uint4*>(k_dst + idx) = out;
      }
    }
  }
}

inline constexpr int kVecMaxDim = 8192;

template <typename DType>
__launch_bounds__(kThreads) __global__ void minimax_qk_norm_rope_store_vec_kernel(
    DType* __restrict__ qkv,
    const DType* __restrict__ q_w,
    const DType* __restrict__ k_w,
    const void* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    DType* __restrict__ k_cache,
    DType* __restrict__ v_cache,
    const void* __restrict__ out_loc,
    int tokens,
    int qkv_stride,
    int q_dim,
    int kv_dim,
    int head_dim,
    int rotary_dim,
    int cache_stride,
    long kv_cache_stride,
    bool positions_i64,
    bool loc_i64,
    float eps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __shared__ float smem[kWarps];
  __shared__ uint4 srow_vec[kVecMaxDim / 8];
  const int token = blockIdx.x;
  const bool is_k = blockIdx.y != 0;
  const int dim = is_k ? kv_dim : q_dim;
  const int offset = is_k ? q_dim : 0;
  DType* row = qkv + static_cast<size_t>(token) * qkv_stride + offset;
  const DType* weight = is_k ? k_w : q_w;
  const int vecs = dim / 8;

  const auto row_vec = reinterpret_cast<const uint4*>(row);
  for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
    srow_vec[v] = row_vec[v];
  }
  DType* v_dst = nullptr;
  DType* k_dst = nullptr;
  if (is_k) {
    const long long cache_pos = loc_i64
        ? static_cast<const int64_t*>(out_loc)[token]
        : static_cast<const int32_t*>(out_loc)[token];
    k_dst = k_cache + cache_pos * kv_cache_stride;
    v_dst = v_cache + cache_pos * kv_cache_stride;
    const auto v_src_vec = reinterpret_cast<const uint4*>(
        qkv + static_cast<size_t>(token) * qkv_stride + q_dim + kv_dim);
    auto v_dst_vec = reinterpret_cast<uint4*>(v_dst);
    for (int v = threadIdx.x; v < kv_dim / 8; v += blockDim.x) {
      v_dst_vec[v] = v_src_vec[v];
    }
  }
  // this thread's weight, partner weight and cos/sin vectors, issued with the row's
  // loads above so the whole block pays one memory round trip
  const int rotary_half = rotary_dim / 2;
  const long long pos = positions_i64
      ? static_cast<const int64_t*>(positions)[token]
      : static_cast<const int32_t*>(positions)[token];
  const float* rope = cos_sin_cache + pos * static_cast<long long>(cache_stride);
  const float* cos_ptr = rope;
  const float* sin_ptr = rope + rotary_half;
  const int chunks_per_head = head_dim / 8;
  const int rot_chunks = rotary_dim / 8;
  const int half_chunks = rot_chunks / 2;
  constexpr int kMaxChunks = kVecMaxDim / 8 / kThreads;
  uint4 w_pre[kMaxChunks], pw_pre[kMaxChunks];
  float4 cs_pre[kMaxChunks][4];
#pragma unroll
  for (int k = 0; k < kMaxChunks; ++k) {
    const int c = threadIdx.x + k * kThreads;
    if (c < vecs) {
      const int head = c / chunks_per_head;
      const int ch = c - head * chunks_per_head;
      w_pre[k] = *reinterpret_cast<const uint4*>(weight + head * head_dim + ch * 8);
      if (ch < rot_chunks) {
        const bool is_lo = ch < half_chunks;
        const int pch = is_lo ? ch + half_chunks : ch - half_chunks;
        pw_pre[k] = *reinterpret_cast<const uint4*>(weight + head * head_dim + pch * 8);
        const int j0 = (is_lo ? ch : pch) * 8;
        cs_pre[k][0] = *reinterpret_cast<const float4*>(cos_ptr + j0);
        cs_pre[k][1] = *reinterpret_cast<const float4*>(cos_ptr + j0 + 4);
        cs_pre[k][2] = *reinterpret_cast<const float4*>(sin_ptr + j0);
        cs_pre[k][3] = *reinterpret_cast<const float4*>(sin_ptr + j0 + 4);
      }
    }
  }
  __syncthreads();
  const DType* srow = reinterpret_cast<const DType*>(srow_vec);

  float sum_sq = 0.0f;
  int i = threadIdx.x;
  for (; i + blockDim.x * 3 < dim; i += blockDim.x * 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float v = to_float<DType>(srow[i + u * blockDim.x]);
      sum_sq += v * v;
    }
  }
  for (; i < dim; i += blockDim.x) {
    const float v = to_float<DType>(srow[i]);
    sum_sq += v * v;
  }
  const float row_sum_sq = block_sum(sum_sq, smem);
  const float inv = rsqrtf(row_sum_sq / static_cast<float>(dim) + eps);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // late: the fmha behind has many CTAs; let it launch only for the normalize + store tail
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
#pragma unroll
  for (int k = 0; k < kMaxChunks; ++k) {
    const int c = threadIdx.x + k * kThreads;
    if (c >= vecs) {
      break;
    }
    const int head = c / chunks_per_head;
    const int ch = c - head * chunks_per_head;
    const int base = head * head_dim + ch * 8;
    const DType* w = reinterpret_cast<const DType*>(&w_pre[k]);
    float x[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      x[e] = to_float<DType>(from_float<DType>(
          to_float<DType>(srow[base + e]) * inv * to_float<DType>(w[e])));
    }
    uint4 out_raw;
    DType* out = reinterpret_cast<DType*>(&out_raw);
    if (ch < rot_chunks) {
      const bool is_lo = ch < half_chunks;
      const int pch = is_lo ? ch + half_chunks : ch - half_chunks;
      const int pbase = head * head_dim + pch * 8;
      const DType* pw = reinterpret_cast<const DType*>(&pw_pre[k]);
      const float* cs = reinterpret_cast<const float*>(cs_pre[k]);  // cos[8], sin[8]
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float px = to_float<DType>(from_float<DType>(
            to_float<DType>(srow[pbase + e]) * inv * to_float<DType>(pw[e])));
        const float c_ = cs[e];
        const float s_ = cs[8 + e];
        if (is_lo) {
          const float x_lo = x[e], x_hi = px;
          // the scalar kernel's contraction of x_lo * c - x_hi * s, spelled out
          out[e] = from_float<DType>(__fmaf_rn(x_lo, c_, -__fmul_rn(x_hi, s_)));
        } else {
          const float x_hi = x[e], x_lo = px;
          // the scalar kernel's contraction of x_hi * c + x_lo * s, spelled out
          out[e] = from_float<DType>(__fmaf_rn(x_hi, c_, __fmul_rn(x_lo, s_)));
        }
      }
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        out[e] = from_float<DType>(to_float<DType>(srow[base + e]) * inv * to_float<DType>(w[e]));
      }
    }
    *reinterpret_cast<uint4*>(row + base) = out_raw;
    if (is_k) {
      *reinterpret_cast<uint4*>(k_dst + base) = out_raw;
    }
  }
}

template <typename T>
struct TypeTag {
  using type = T;
};

}  // namespace

struct MiniMaxQKNormKernel {
  static void run(
      const tvm::ffi::TensorView qkv,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      int q_dim,
      int kv_dim,
      double eps) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto tokens_ = SymbolicSize{"T"};
    auto qkv_dim_ = SymbolicSize{"D"};

    TensorMatcher({tokens_, qkv_dim_})
        .with_strides({-1, 1})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(qkv);
    TensorMatcher({q_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(q_weight);
    TensorMatcher({kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_weight);
    RuntimeCheck(static_cast<int>(qkv_dim_.unwrap()) >= q_dim + 2 * kv_dim,
                 "qkv last dim is too small for MiniMax q/k norm");

    const auto tokens = static_cast<int>(tokens_.unwrap());
    if (tokens == 0) return;
    const auto device = device_.unwrap();
    const int qkv_stride = static_cast<int>(qkv.stride(0));

    auto launch = [&](auto type_tag) {
      using DType = typename decltype(type_tag)::type;
      LaunchKernel(dim3(tokens, 2, 1), kThreads, device)(
          minimax_qk_cross_head_norm_kernel<DType>,
          static_cast<DType*>(qkv.data_ptr()),
          static_cast<const DType*>(q_weight.data_ptr()),
          static_cast<const DType*>(k_weight.data_ptr()),
          tokens,
          qkv_stride,
          q_dim,
          kv_dim,
          static_cast<float>(eps));
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      launch(TypeTag<__nv_bfloat16>{});
    } else {
      launch(TypeTag<__half>{});
    }
  }

  static void run_rope_store(
      const tvm::ffi::TensorView qkv,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView out_loc,
      int q_dim,
      int kv_dim,
      int head_dim,
      double eps) {
    run_rope_store_impl(qkv, q_weight, k_weight, positions, cos_sin_cache, k_cache, v_cache,
                        out_loc, q_dim, kv_dim, head_dim, eps, true);
  }

  // the scalar kernel only (reference for tests and benchmarks)
  static void run_rope_store_scalar(
      const tvm::ffi::TensorView qkv,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView out_loc,
      int q_dim,
      int kv_dim,
      int head_dim,
      double eps) {
    run_rope_store_impl(qkv, q_weight, k_weight, positions, cos_sin_cache, k_cache, v_cache,
                        out_loc, q_dim, kv_dim, head_dim, eps, false);
  }

 private:
  static void run_rope_store_impl(
      const tvm::ffi::TensorView qkv,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView out_loc,
      int q_dim,
      int kv_dim,
      int head_dim,
      double eps,
      bool use_vec_) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto pos_dtype_ = SymbolicDType{};
    auto loc_dtype_ = SymbolicDType{};
    auto tokens_ = SymbolicSize{"T"};
    auto qkv_dim_ = SymbolicSize{"D"};
    auto rotary_dim_ = SymbolicSize{"R"};

    TensorMatcher({tokens_, qkv_dim_})
        .with_strides({-1, 1})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(qkv);
    TensorMatcher({q_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(q_weight);
    TensorMatcher({kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_weight);
    TensorMatcher({tokens_})
        .with_dtype<int64_t, int32_t>(pos_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(positions);
    TensorMatcher({-1, rotary_dim_})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device_)
        .verify(cos_sin_cache);
    TensorMatcher({-1, kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_cache);
    TensorMatcher({-1, kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(v_cache);
    TensorMatcher({tokens_})
        .with_dtype<int64_t, int32_t>(loc_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(out_loc);

    const auto tokens = static_cast<int>(tokens_.unwrap());
    if (tokens == 0) return;
    const int qkv_dim = static_cast<int>(qkv_dim_.unwrap());
    const int rotary_dim = static_cast<int>(rotary_dim_.unwrap());
    RuntimeCheck(qkv_dim >= q_dim + 2 * kv_dim,
                 "qkv last dim is too small for MiniMax q/k norm+rope+store");
    RuntimeCheck(head_dim > 0 && q_dim % head_dim == 0 && kv_dim % head_dim == 0,
                 "MiniMax q/k norm+rope+store requires head-aligned q/kv dims");
    RuntimeCheck(rotary_dim > 0 && rotary_dim <= head_dim && (rotary_dim % 2) == 0,
                 "MiniMax q/k norm+rope+store requires an even rotary_dim <= head_dim");

    const auto device = device_.unwrap();
    const int qkv_stride = static_cast<int>(qkv.stride(0));
    const int cache_stride = static_cast<int>(cos_sin_cache.stride(0));
    const long kv_cache_stride = static_cast<long>(k_cache.stride(0));

    // the vectorized kernel when everything is 16-byte addressable (else the scalar one)
    const int elem = data_dtype_.unwrap().bits / 8;
    auto aligned16 = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    const bool vec_ok = use_vec_ && q_dim % 8 == 0 && kv_dim % 8 == 0 && head_dim % 8 == 0 &&
                        rotary_dim % 16 == 0 && q_dim <= kVecMaxDim && kv_dim <= kVecMaxDim &&
                        qkv_stride % 8 == 0 && kv_cache_stride % 8 == 0 &&
                        aligned16(qkv.data_ptr()) && aligned16(q_weight.data_ptr()) &&
                        aligned16(k_weight.data_ptr()) && aligned16(k_cache.data_ptr()) &&
                        aligned16(v_cache.data_ptr()) && aligned16(cos_sin_cache.data_ptr()) &&
                        cache_stride % 4 == 0 && elem == 2;
    // Rows up to kThreads * kRegsMaxPer wide (MiniMax-M2.5's 6144-wide Q: 48 heads x 128) keep
    // the row in registers with every load of a phase in flight; it is faster than the
    // shared-memory staging above at every decode batch (bit-identical to both).
    constexpr int kRegsMaxPer = 24;
    const bool regs_ok = vec_ok && q_dim <= kThreads * kRegsMaxPer && kv_dim <= kThreads * kRegsMaxPer &&
                         (rotary_dim / 2) % 8 == 0 && (head_dim - rotary_dim) % 8 == 0;
    auto launch = [&](auto type_tag) {
      using DType = typename decltype(type_tag)::type;
      if (regs_ok) {
        const int smem = (q_dim > kv_dim ? q_dim : kv_dim) * static_cast<int>(sizeof(DType));
        LaunchKernel(dim3(tokens, 2, 1), kThreads, device, smem).with_attr(minisgl_pdl_enabled())(
            minimax_qk_norm_rope_store_regs_kernel<DType, kRegsMaxPer>,
            static_cast<DType*>(qkv.data_ptr()),
            static_cast<const DType*>(q_weight.data_ptr()),
            static_cast<const DType*>(k_weight.data_ptr()),
            positions.data_ptr(),
            static_cast<const float*>(cos_sin_cache.data_ptr()),
            static_cast<DType*>(k_cache.data_ptr()),
            static_cast<DType*>(v_cache.data_ptr()),
            out_loc.data_ptr(),
            tokens,
            qkv_stride,
            q_dim,
            kv_dim,
            head_dim,
            rotary_dim,
            cache_stride,
            kv_cache_stride,
            pos_dtype_.unwrap().bits == 64,
            loc_dtype_.unwrap().bits == 64,
            static_cast<float>(eps));
        return;
      }
      LaunchKernel(dim3(tokens, 2, 1), kThreads, device).with_attr(minisgl_pdl_enabled())(
          vec_ok ? minimax_qk_norm_rope_store_vec_kernel<DType>
                 : minimax_qk_cross_head_norm_rope_store_kernel<DType>,
          static_cast<DType*>(qkv.data_ptr()),
          static_cast<const DType*>(q_weight.data_ptr()),
          static_cast<const DType*>(k_weight.data_ptr()),
          positions.data_ptr(),
          static_cast<const float*>(cos_sin_cache.data_ptr()),
          static_cast<DType*>(k_cache.data_ptr()),
          static_cast<DType*>(v_cache.data_ptr()),
          out_loc.data_ptr(),
          tokens,
          qkv_stride,
          q_dim,
          kv_dim,
          head_dim,
          rotary_dim,
          cache_stride,
          kv_cache_stride,
          pos_dtype_.unwrap().bits == 64,
          loc_dtype_.unwrap().bits == 64,
          static_cast<float>(eps));
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      launch(TypeTag<__nv_bfloat16>{});
    } else {
      launch(TypeTag<__half>{});
    }
  }
};
