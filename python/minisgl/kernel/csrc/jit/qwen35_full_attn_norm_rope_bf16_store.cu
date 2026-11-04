// Qwen3.5 AFD decode full attention with a BF16 KV pool, one kernel for everything between
// the QKV projection and the paged attention kernel:
//   q/k RMSNorm -> neox RoPE (partial rotary) -> paged BF16 KV store.
// It reads q, k and v straight from the projection output (rows of
// [32 x (query 256 | gate 256) | k 2x256 | v 2x256] BF16), writes the normalized and
// rotated query back in place (the attention kernel reads it through its strides) and
// writes the rotated key and the unchanged value into their page slots.
//
// Byte-for-byte replica of the chain it replaces on a BF16 pool:
//   1. FlashInfer 0.6.6 QKRMSNormKernel<8, bf16> (built with -use_fast_math): one warp per
//      (token, head), 8 contiguous elements per lane, `sum_sq += x * x`, butterfly 16..1,
//      rsqrt.approx.ftz(sum_sq / float(d) + eps), `x * rms_rcp * (weight_bias + w)`, RNE to
//      BF16. weight_bias is a runtime 0 as in FlashInfer (RMSNormOffset folds the +1 into w).
//   2. FlashInfer BatchQKApplyRotaryPosIdsCosSinCache*Kernel<interleave=false, 256, 8, 32>
//      (-use_fast_math): reads the stored BF16 normed values; lanes owning elements below
//      rotary_dim/2 use `x * cos + (-partner) * sin`, the next rotary_dim/2 use
//      `x * cos + partner * sin`; cos/sin from the fp32 cos_sin_cache; RNE to BF16.
//   3. minisgl store_cache: byte copy of each token's [kv_heads * 256] BF16 key (as rotated
//      in step 2) and value (as projected) row to slot out_loc.
// The source of those FlashInfer kernels is identical in 0.6.6 and 0.6.11.post3.

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kVec = 8;                          // FlashInfer: gcd(16 / sizeof(bf16), 256)
constexpr uint32_t kHeadDim = kWarpSize * kVec;       // 256
constexpr uint32_t kQHeads = 32;
constexpr uint32_t kKVHeads = 2;
constexpr uint32_t kQGate = 2 * kQHeads * kHeadDim;   // 16384
constexpr uint32_t kKVRow = kKVHeads * kHeadDim;      // 512 BF16 elements per token and cache
constexpr uint32_t kRowWidth = kQGate + 2 * kKVRow;   // 17408
constexpr uint32_t kJobsPerToken = kQHeads + 2 * kKVHeads;
constexpr uint32_t kWarpsPerBlock = 4;

// FlashInfer math::shfl_xor_sync / math::rsqrt, verbatim.
__device__ __forceinline__ float shfl_xor_bfly(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(y) : "f"(x), "r"(lane_mask));
  return y;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__launch_bounds__(kWarpSize * kWarpsPerBlock) __global__ void qwen35_full_attn_norm_rope_bf16_store_kernel(
    __nv_bfloat16* __restrict__ qkv,              // [rows, >= 17408], row stride `row_stride`
    const int64_t row_stride,
    const int32_t* __restrict__ positions,        // [rows]
    const __nv_bfloat16* __restrict__ q_weight,   // [256]
    const __nv_bfloat16* __restrict__ k_weight,   // [256]
    const float* __restrict__ cos_sin_cache,      // [max_pos, rotary_dim]
    __nv_bfloat16* __restrict__ k_cache,          // [slots, 512]
    __nv_bfloat16* __restrict__ v_cache,          // [slots, 512]
    const int32_t* __restrict__ out_loc,          // [rows]
    const uint32_t rows, const uint32_t d, const uint32_t rotary_dim,
    const float weight_bias, const float eps) {
  const uint32_t lane = threadIdx.x;
  const uint32_t job = blockIdx.x * kWarpsPerBlock + threadIdx.y;
  if (job >= rows * kJobsPerToken) {
    return;  // whole warps exit together; the shuffles below stay warp-uniform
  }
  const uint32_t token = job / kJobsPerToken;
  const uint32_t head = job % kJobsPerToken;
  __nv_bfloat16* row = qkv + static_cast<int64_t>(token) * row_stride;
  const int64_t slot = static_cast<int64_t>(out_loc[token]);

  if (head >= kQHeads + kKVHeads) {  // value head: store only
    const uint32_t kv_head = head - kQHeads - kKVHeads;
    *reinterpret_cast<uint4*>(v_cache + slot * kKVRow + kv_head * kHeadDim + lane * kVec) =
        *reinterpret_cast<const uint4*>(row + kQGate + kKVRow + kv_head * kHeadDim + lane * kVec);
    return;
  }

  const bool is_query = head < kQHeads;
  __nv_bfloat16* src = is_query ? row + head * (2 * kHeadDim)
                                : row + kQGate + (head - kQHeads) * kHeadDim;
  const __nv_bfloat16* weight = is_query ? q_weight : k_weight;

  // ---- FlashInfer QKRMSNormKernel, rounds = 1 for d = 256 ---------------------------
  const uint4 raw = *reinterpret_cast<const uint4*>(src + lane * kVec);
  const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float sum_sq = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kVec; j++) {
    sum_sq += float(in[j]) * float(in[j]);
  }
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_bfly(sum_sq, offset);
  }
  const float rms_rcp = rsqrt_approx_ftz(sum_sq / float(d) + eps);
  const uint4 weight_raw = *reinterpret_cast<const uint4*>(weight + lane * kVec);
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&weight_raw);
  float vec[kVec];
#pragma unroll
  for (uint32_t j = 0; j < kVec; j++) {
    // Stored as BF16 by the norm, re-read as float by the RoPE kernel.
    vec[j] = __bfloat162float(__float2bfloat16(float(in[j]) * rms_rcp * (weight_bias + float(w[j]))));
  }

  // ---- FlashInfer vec_apply_llama_rope_cos_sin (neox) -------------------------------
  // The partner of lane l is l ^ (rotary_dim / 2 / kVec): 4 for rotary_dim 64.
  const uint32_t half_lanes = rotary_dim / 2 / kVec;
  float permuted_vec[kVec];
#pragma unroll
  for (uint32_t j = 0; j < kVec; j++) {
    permuted_vec[j] = __shfl_xor_sync(0xffffffffu, vec[j], half_lanes);
  }
  if (lane * kVec < rotary_dim) {
    const int32_t pos = positions[token];
    const uint32_t vec_idx = (lane * kVec) % (rotary_dim / 2);
    const float* cs = cos_sin_cache + static_cast<int64_t>(pos) * rotary_dim;
    float cos[kVec], sin[kVec];
#pragma unroll
    for (uint32_t j = 0; j < kVec; j++) {
      cos[j] = cs[vec_idx + j];
      sin[j] = cs[rotary_dim / 2 + vec_idx + j];
    }
#pragma unroll
    for (uint32_t i = 0; i < kVec; ++i) {
      vec[i] = vec[i] * cos[i] + ((lane * kVec < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }

  // The rotated BF16 value: the query goes back in place, the key to its page slot
  // (what the RoPE kernel writes into the key tensor and store_cache then copies).
  uint4 out;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (uint32_t j = 0; j < kVec; j++) {
    o[j] = __float2bfloat16(vec[j]);
  }
  __nv_bfloat16* dst = is_query ? src : k_cache + slot * kKVRow + (head - kQHeads) * kHeadDim;
  *reinterpret_cast<uint4*>(dst + lane * kVec) = out;
}

}  // namespace

struct Qwen35FullAttnNormRopeBf16StoreKernel {
  static void run(
      const tvm::ffi::TensorView qkv,        // [rows, 17408] bf16, last dim contiguous, in/out (query part)
      const tvm::ffi::TensorView positions,  // [rows] int32
      const tvm::ffi::TensorView q_weight,   // [256] bf16
      const tvm::ffi::TensorView k_weight,   // [256] bf16
      const tvm::ffi::TensorView cos_sin,    // [max_pos, rotary_dim] fp32 contiguous
      const tvm::ffi::TensorView k_cache,    // [slots, 512] bf16 contiguous
      const tvm::ffi::TensorView v_cache,    // [slots, 512] bf16 contiguous
      const tvm::ffi::TensorView out_loc,    // [rows] int32
      double eps) {
    using namespace host;
    auto rows_ = SymbolicSize{"rows"};
    auto width_ = SymbolicSize{"width"};
    auto row_stride_ = SymbolicSize{"row_stride"};
    auto slots_ = SymbolicSize{"slots"};
    auto positions_max_ = SymbolicSize{"max_pos"};
    auto rotary_ = SymbolicSize{"rotary_dim"};
    auto device_ = SymbolicDevice{};
    auto bf16_ = SymbolicDType{};
    TensorMatcher({rows_, width_}).with_strides({row_stride_, 1}).with_dtype(bf16_)
        .with_device<kDLCUDA>(device_).verify(qkv);
    TensorMatcher({static_cast<int64_t>(kHeadDim)}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(q_weight).verify(k_weight);
    TensorMatcher({slots_, static_cast<int64_t>(kKVRow)}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(k_cache).verify(v_cache);
    {
      const auto dt = bf16_.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1,
                   "qkv, q/k norm weights and the KV caches must be bf16");
    }
    TensorMatcher({rows_}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(positions).verify(out_loc);
    TensorMatcher({positions_max_, rotary_}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(cos_sin);

    const auto rows = rows_.unwrap();
    const auto rotary_dim = rotary_.unwrap();
    RuntimeCheck(width_.unwrap() == kRowWidth, "qkv must be the [rows, 17408] Qwen3.5 projection output, got width ",
                 width_.unwrap());
    // The RoPE partner of lane l is l ^ (rotary_dim / 16): needs a power of two in [16, 256].
    RuntimeCheck((rotary_dim & (rotary_dim - 1)) == 0 && rotary_dim >= 2 * kVec && rotary_dim <= kHeadDim,
                 "unsupported rotary_dim ", rotary_dim);
    RuntimeCheck(row_stride_.unwrap() % kVec == 0, "qkv row stride must keep 16-byte alignment");
    for (const auto& view : {qkv, q_weight, k_weight, cos_sin, k_cache, v_cache}) {
      RuntimeCheck(reinterpret_cast<uintptr_t>(view.data_ptr()) % 16 == 0, "16-byte aligned tensors required");
    }
    if (rows == 0) {
      return;
    }
    const auto jobs = static_cast<uint64_t>(rows) * kJobsPerToken;
    const auto blocks = static_cast<unsigned>((jobs + kWarpsPerBlock - 1) / kWarpsPerBlock);
    LaunchKernel(dim3(blocks), dim3(kWarpSize, kWarpsPerBlock), device_.unwrap())(
        qwen35_full_attn_norm_rope_bf16_store_kernel,
        static_cast<__nv_bfloat16*>(qkv.data_ptr()), static_cast<int64_t>(row_stride_.unwrap()),
        static_cast<const int32_t*>(positions.data_ptr()),
        static_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
        static_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
        static_cast<const float*>(cos_sin.data_ptr()),
        static_cast<__nv_bfloat16*>(k_cache.data_ptr()), static_cast<__nv_bfloat16*>(v_cache.data_ptr()),
        static_cast<const int32_t*>(out_loc.data_ptr()),
        static_cast<uint32_t>(rows), kHeadDim, static_cast<uint32_t>(rotary_dim),
        0.f, static_cast<float>(eps));
  }
};
