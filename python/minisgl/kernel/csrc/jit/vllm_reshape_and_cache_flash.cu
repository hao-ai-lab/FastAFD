/*
 * VENDORED (MINISGL_QWEN35_VLLM_FULLATTN) from vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6:
 * the KV-cache store vLLM's FlashInfer backend runs for Qwen3.5-397B's
 * full-attention layers (flashinfer.py:2673 torch.ops._C_cache_ops.reshape_and_cache_flash, bf16 KV, "auto"):
 *   csrc/libtorch_stable/quantization/vectorization.cuh       lines 12-16   vec_n_t                     (sha256_12 6b28cf0cfe9e, blob_12 9d5eea00e468)
 *   csrc/libtorch_stable/quantization/vectorization_utils.cuh lines 6-115   DefaultVecOp, vectorize_with_alignment (sha256_12 991d501211c9, blob_12 0cc89bf289d4)
 *   csrc/attention/dtype_fp8.cuh                              lines 15-19   enum Fp8KVCacheDataType     (sha256_12 dde54da39de7, blob_12 3d56859d8fc5)
 *   csrc/libtorch_stable/cache_kernels.cu                     lines 251-263 CopyWithScaleOp, 325-411 reshape_and_cache_flash_kernel (sha256_12 5ea4d65b32a1, blob_12 9130adf87d69)
 * all byte for byte. Not vendored: fp8::scaled_convert (only the FP8-KV branch of CopyWithScaleOp names it; vLLM's bf16
 * "auto" instantiation <__nv_bfloat16, __nv_bfloat16, kAuto> (quant_utils.cuh:310-311) never instantiates it), declared
 * below so the template parses. The host side re-creates reshape_and_cache_flash (cache_kernels.cu:820-890) for that one
 * case: same strides from the cache tensors, grid(num_tokens = slot_mapping.size(0)), block(min(num_heads*head_size, 512)).
 * KV layout is the caller's: vLLM's FlashInfer cache is (B, H, N, 2*hs) -> HND branch; FastAFD's MHAKVCache is NHD
 * (separate K / V pools) -> NHD branch of the same kernel (a named layout difference, kvcache/mha_pool.py).
 */
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace vllm {

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  __device__ __forceinline__ void operator()(
      vec_n_t<OutT, VEC_SIZE>& dst, const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp,
          typename ScaOp>
__device__ inline void vectorize_with_alignment(
    const InT* in, OutT* out, int len, int tid, int stride,
    VecOp&& vec_op,       // vec_n_t<InT,16> -> vec_n_t<OutT,16>
    ScaOp&& scalar_op) {  // InT -> OutT
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);       // eg: 16 B
  constexpr int OUT_WIDTH = VEC_SIZE * sizeof(OutT);  // eg: 16 B
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);
  uintptr_t out_addr = reinterpret_cast<uintptr_t>(out);

  // fast path when input and output are both fully aligned. The vector
  // load/store below go through vec_n_t<T, VEC_SIZE>, declared
  // __align__(VEC_SIZE * sizeof(T)), so each side must be aligned to its
  // own vector width. out is NOT generally co-aligned with in: e.g.
  // reshape_and_cache_flash writes KV-cache rows whose byte offset is a
  // multiple of head_size, which for head sizes that are not a multiple
  // of VEC_SIZE puts some rows off the vector-width boundary.
  bool can_vec = ((addr & (WIDTH - 1)) == 0) &&
                 ((out_addr & (OUT_WIDTH - 1)) == 0) &&
                 ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int i = tid; i < num_vec; i += stride) {
      vout_t tmp;
      // Make a local copy of the entire pack
      vin_t src = v_in[i];  // <- encourages a single vector ld
      vec_op(tmp, src);
      v_out[i] = tmp;  // <- encourages a single vector st
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);       // addr % 64
  int alignment_bytes = WIDTH - misalignment_offset;  // 64 - (addr % 64)
  int prefix_elems = alignment_bytes & (WIDTH - 1);   // handle 64
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);  // 0 ≤ prefix < 16

  // the prefix below aligns in; if that does not also align out (their
  // addresses differ modulo the vector width), vectorizing is impossible
  // and the whole copy must stay scalar.
  if (((out_addr + prefix_elems * sizeof(OutT)) & (OUT_WIDTH - 1)) != 0) {
    for (int i = tid; i < len; i += stride) {
      scalar_op(out[i], in[i]);
    }
    return;
  }

  // 1. prefill the when it is unsafe to vectorize
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);

  // 2. vectorize the main part
  for (int i = tid; i < num_vec; i += stride) {
    vout_t tmp;
    // Make a local copy of the entire pack
    vin_t src = v_in[i];  // <- encourages a single vector ld
    vec_op(tmp, src);
    v_out[i] = tmp;  // <- encourages a single vector st
  }

  // 3. handle the tail
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(out[i], in[i]);
  }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
__device__ __forceinline__ void vectorize_with_alignment(const InT* in,
                                                         OutT* out, int len,
                                                         int tid, int stride,
                                                         ScaOp&& scalar_op) {
  using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
  vectorize_with_alignment<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op},
                                     std::forward<ScaOp>(scalar_op));
}

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

namespace fp8 {
// declaration only (see the header): referenced by CopyWithScaleOp's FP8 branch, never instantiated for kAuto
template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__device__ Tout scaled_convert(const Tin& x, const float scale);
}  // namespace fp8

// Used to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // NHD or HND, shape see comments below
    cache_t* __restrict__ value_cache,   // same above
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale,
    const int kv_scale_stride) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n_elems = num_heads * head_size;

  // pointers to the beginning of the source row for this token.
  const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

  // find the start position inside the kv-cache for this token.
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  // this is true for the NHD layout where `head_stride == head_size`
  const bool is_contiguous_heads = (head_stride == head_size);

  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;

  if (is_contiguous_heads && kv_scale_stride == 0) {
    // NHD layout and k/v_scales are [1] (i.e. single scale for all heads)
    // kv cache: [num_blocks, block_size, num_heads, head_size]
    float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
    float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;

    CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x,
                                       blockDim.x, k_op);
    vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems,
                                       threadIdx.x, blockDim.x, v_op);
  } else {
    // HND layout OR k/v_scales are [num_heads] (i.e. per-attn-head)
    // HND layout: heads are strided, but each head_size segment is contiguous
    // kv cache: [num_blocks, num_heads, block_size, head_size]
    const int lane = threadIdx.x & 31;     // 0..31 within warp
    const int warp_id = threadIdx.x >> 5;  // warp index within block
    const int warps_per_block = blockDim.x >> 5;

    for (int head = warp_id; head < num_heads; head += warps_per_block) {
      const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
      const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

      cache_t* __restrict__ k_dst_h =
          key_dst + static_cast<int64_t>(head) * head_stride;
      cache_t* __restrict__ v_dst_h =
          value_dst + static_cast<int64_t>(head) * head_stride;

      float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto)
                              ? 0.f
                              : k_scale[head * kv_scale_stride];
      float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto)
                              ? 0.f
                              : v_scale[head * kv_scale_stride];

      CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
      CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

      // within each head, let the 32 threads of the warp perform the vector
      // copy
      vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32,
                                         k_op);

      vectorize_with_alignment<VEC_SIZE>(v_src_h, v_dst_h, head_size, lane, 32,
                                         v_op);
    }
  }
}

}  // namespace vllm

struct VllmReshapeAndCacheFlashKernel {
  // reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", k_scale, v_scale)
  // for bf16 key / value / cache (cache_kernels.cu:820-890, DISPATCH_BY_KV_CACHE_DTYPE kAuto + BFloat16).
  static void run(const tvm::ffi::TensorView key,           // [T, H, D] bf16 (row stride key.stride(0))
                  const tvm::ffi::TensorView value,         // [T, H, D] bf16
                  const tvm::ffi::TensorView key_cache,     // [num_blocks, block_size, H, D] bf16 (any strides)
                  const tvm::ffi::TensorView value_cache,   // same
                  const tvm::ffi::TensorView slot_mapping,  // [num_tokens] int64
                  const tvm::ffi::TensorView k_scale,       // [1] fp32 (unused by "auto")
                  const tvm::ffi::TensorView v_scale) {
    using namespace host;
    auto t_ = SymbolicSize{"T"};
    auto h_ = SymbolicSize{"H"};
    auto d_ = SymbolicSize{"D"};
    auto nb_ = SymbolicSize{"NB"};
    auto bs_ = SymbolicSize{"BS"};
    auto ns_ = SymbolicSize{"NS"};
    auto one_ = SymbolicSize{"ONE"};
    auto device_ = SymbolicDevice{};
    auto bf16_ = SymbolicDType{};
    TensorMatcher({t_, h_, d_}).with_strides({-1, -1, 1}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(key);
    {
      const auto dt = bf16_.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1, "key must be bf16");
    }
    TensorMatcher({t_, h_, d_}).with_strides({-1, -1, 1}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(value);
    TensorMatcher({nb_, bs_, h_, d_}).with_strides({-1, -1, -1, 1}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(key_cache);
    TensorMatcher({nb_, bs_, h_, d_}).with_strides({-1, -1, -1, 1}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(value_cache);
    TensorMatcher({ns_}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(slot_mapping);
    TensorMatcher({one_}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(k_scale);
    TensorMatcher({one_}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(v_scale);
    // per head: the kernel copies num_heads * head_size contiguous elements per token (NHD branch) or head_size per head
    RuntimeCheck(key.stride(1) == static_cast<int64_t>(d_.unwrap()) && value.stride(1) == static_cast<int64_t>(d_.unwrap()),
                 "key / value heads must be contiguous within a token row");
    RuntimeCheck(slot_mapping.stride(0) == 1, "slot_mapping must be contiguous");
    RuntimeCheck(key_cache.stride(0) == value_cache.stride(0), "key_cache / value_cache block strides differ");
    const int num_tokens = static_cast<int>(ns_.unwrap());
    const int num_heads = static_cast<int>(h_.unwrap());
    const int head_size = static_cast<int>(d_.unwrap());
    const int block_size = static_cast<int>(bs_.unwrap());
    const int64_t key_stride = key.stride(0);
    const int64_t value_stride = value.stride(0);
    const int64_t block_stride = key_cache.stride(0);
    const int64_t page_stride = key_cache.stride(1);
    const int64_t head_stride = key_cache.stride(2);
    const int kv_scale_stride = 0;  // k_scale.numel() == 1
    if (num_tokens == 0) {
      return;
    }
    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    const cudaStream_t stream = LaunchKernel::resolve_device(device_.unwrap());
    vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto>
        <<<grid, block, 0, stream>>>(
            static_cast<__nv_bfloat16*>(key.data_ptr()), static_cast<__nv_bfloat16*>(value.data_ptr()),
            static_cast<__nv_bfloat16*>(key_cache.data_ptr()), static_cast<__nv_bfloat16*>(value_cache.data_ptr()),
            static_cast<const int64_t*>(slot_mapping.data_ptr()), block_stride, page_stride, head_stride, key_stride,
            value_stride, num_heads, head_size, block_size, static_cast<const float*>(k_scale.data_ptr()),
            static_cast<const float*>(v_scale.data_ptr()), kv_scale_stride);
    CUDA_CHECK(cudaGetLastError());
  }
};
