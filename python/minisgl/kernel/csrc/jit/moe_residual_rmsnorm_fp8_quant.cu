// Qwen3.5 AFD decode input path in one kernel: the previous layer's
// `routed + shared` MoE add, the residual add, the input RMSNorm and the
// per-128-group packed-UE8M0 FP8 quantization of the normed activations.
//
// Derived from add_rmsnorm_fp8_quant.cu in this directory. That kernel has no
// caller and is not byte-identical to the three-kernel chain it would replace
// (different RMS reduction order, no second addend, no BF16 normed output, amax
// seeded with the norm eps). This one reproduces the chain byte for byte:
//   1. `routed + shared`: PyTorch BF16 add = IEEE fp32 add (no FTZ), RNE to BF16.
//   2. residual add, sum of squares, rsqrt and normalize: FlashInfer 0.6.6
//      FusedAddRMSNormKernel for H=4096 (block 32x16, 8 BF16 per thread, the
//      same butterfly reductions). FlashInfer builds its norm module with
//      --use_fast_math; so is this file (fp8_quant.py passes the flag).
//   3. quantization: per_token_fp8_packed.cu (IEEE division by 448, 1e-10
//      floor, UE8M0 ceil, SATFINITE E4M3, zeroed padding scale rows). That file
//      is built without fast math, so the few operations whose rounding could
//      differ use explicit IEEE PTX.
// The BF16 normed output feeds the BF16 b/a projection; FP8 values and scales
// feed the DeepGEMM QKV/z (or full-attention QKV) projections.

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarps = 16;  // FlashInfer: ceil(min(1024, H / 8) / 32) for H=4096
constexpr uint32_t kVec = 8;     // FlashInfer: gcd(16 / sizeof(bf16), H)
constexpr uint32_t kHidden = kWarpSize * kWarps * kVec;  // 4096
constexpr uint32_t kGroup = 128;
constexpr uint32_t kGroups = kHidden / kGroup;         // 32
constexpr uint32_t kPackedGroups = kGroups / 4;        // 8
constexpr uint32_t kGroupLanes = kGroup / kVec;        // 16 threads per 128-group

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

// IEEE operations without FTZ, independent of --use_fast_math.
__device__ __forceinline__ float add_rn(float a, float b) {
  float y;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ float div_rn(float a, float b) {
  float y;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
  float y;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
  float y;
  asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ float abs_noftz(float a) {
  float y;
  asm("abs.f32 %0, %1;" : "=f"(y) : "f"(a));
  return y;
}

__launch_bounds__(kWarpSize * kWarps) __global__ void moe_residual_rmsnorm_fp8_quant_kernel(
    __nv_bfloat16* __restrict__ residual,       // [m, H] in/out
    const __nv_bfloat16* __restrict__ routed,   // [m, H]
    const __nv_bfloat16* __restrict__ shared,   // [m, H]
    const __nv_bfloat16* __restrict__ weight,   // [H]
    __nv_bfloat16* __restrict__ normed,         // [m, H]
    uint8_t* __restrict__ out_q,                // [m, H] e4m3 bytes
    uint8_t* __restrict__ out_s,                // packed UE8M0 bytes, int32 stride (1, tma_aligned_mn)
    const uint32_t d, const uint32_t rows, const uint32_t tma_aligned_mn,
    const int64_t num_scale_elems, const float weight_bias, const float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * kWarpSize;
  const int64_t offset = static_cast<int64_t>(bx) * d + thread_id * kVec;
  __shared__ float smem[kWarps];

  const uint4 routed_v = *reinterpret_cast<const uint4*>(routed + offset);
  const uint4 shared_v = *reinterpret_cast<const uint4*>(shared + offset);
  uint4 residual_v = *reinterpret_cast<const uint4*>(residual + offset);
  const __nv_bfloat16* r = reinterpret_cast<const __nv_bfloat16*>(&routed_v);
  const __nv_bfloat16* s = reinterpret_cast<const __nv_bfloat16*>(&shared_v);
  __nv_bfloat16* res = reinterpret_cast<__nv_bfloat16*>(&residual_v);

  // ---- FlashInfer phase 1 (input = bf16(routed + shared)) -----------------
  float x_vec[kVec];
  float sum_sq = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kVec; j++) {
    const __nv_bfloat16 input = __float2bfloat16(add_rn(__bfloat162float(r[j]), __bfloat162float(s[j])));
    float x = __bfloat162float(input);
    x += __bfloat162float(res[j]);
    sum_sq += x * x;
    res[j] = __float2bfloat16(x);
    x_vec[j] = x;
  }
  *reinterpret_cast<uint4*>(residual + offset) = residual_v;

#pragma unroll
  for (uint32_t off = kWarpSize / 2; off > 0; off /= 2) {
    sum_sq += shfl_xor_bfly(sum_sq, off);
  }
  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t off = kWarpSize / 2; off > 0; off /= 2) {
      sum_sq += shfl_xor_bfly(sum_sq, off);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();
  const float rms_rcp = rsqrt_approx_ftz(smem[0] / float(d) + eps);

  // ---- FlashInfer phase 2: normalize, round to BF16 ------------------------
  const uint4 weight_v = *reinterpret_cast<const uint4*>(weight + thread_id * kVec);
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&weight_v);
  uint4 normed_v;
  __nv_bfloat16* n = reinterpret_cast<__nv_bfloat16*>(&normed_v);
  float amax = 1e-10f;  // per_token_fp8_packed seeds local_absmax with eps = 1e-10
#pragma unroll
  for (uint32_t j = 0; j < kVec; j++) {
    n[j] = __float2bfloat16(x_vec[j] * rms_rcp * (weight_bias + __bfloat162float(w[j])));
    amax = max_noftz(amax, abs_noftz(__bfloat162float(n[j])));
  }
  *reinterpret_cast<uint4*>(normed + offset) = normed_v;

  // ---- per_token_fp8_packed: 16 lanes own one 128-group ------------------
#pragma unroll
  for (uint32_t off = kGroupLanes / 2; off > 0; off >>= 1) {
    amax = max_noftz(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  float y_s = div_rn(amax, 448.0f);
  y_s = max_noftz(y_s, 1e-10f);
  const uint32_t bits = __float_as_uint(y_s);
  const uint8_t exp_byte = static_cast<uint8_t>(((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u));
  const float inv_y = div_rn(1.0f, __uint_as_float(static_cast<uint32_t>(exp_byte) << 23));

  __nv_fp8x2_storage_t out2[kVec / 2];
#pragma unroll
  for (uint32_t j = 0; j < kVec; j += 2) {
    out2[j / 2] = __nv_cvt_float2_to_fp8x2(
        make_float2(mul_rn(__bfloat162float(n[j]), inv_y), mul_rn(__bfloat162float(n[j + 1]), inv_y)),
        __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<uint2*>(out_q + offset) = *reinterpret_cast<const uint2*>(out2);

  const uint32_t group = thread_id / kGroupLanes;
  if (thread_id % kGroupLanes == 0) {
    const int64_t pack = group / 4, pos = group % 4;
    out_s[(pack * tma_aligned_mn + bx) * 4 + pos] = exp_byte;
    // The standalone cast also writes zero scales for TMA padding rows that
    // fall inside the scale allocation; the last block does it here.
    if (bx + 1 == rows) {
      for (uint32_t pad = rows; pad < tma_aligned_mn; ++pad) {
        const int64_t index = pack * tma_aligned_mn + pad;
        if (index < num_scale_elems) {
          out_s[index * 4 + pos] = 0;
        }
      }
    }
  }
}

}  // namespace

struct MoeResidualRMSNormFp8QuantKernel {
  static void run(
      const tvm::ffi::TensorView residual,  // [m, H] bf16, in/out
      const tvm::ffi::TensorView routed,    // [m, H] bf16
      const tvm::ffi::TensorView shared,    // [m, H] bf16
      const tvm::ffi::TensorView weight,    // [H] bf16
      const tvm::ffi::TensorView normed,    // [m, H] bf16 out
      const tvm::ffi::TensorView out_q,     // [m, H] uint8 out
      const tvm::ffi::TensorView out_s,     // [m, H/512] int32 out, strides (1, tma_aligned_mn)
      double eps) {
    using namespace host;
    auto m_ = SymbolicSize{"M"};
    auto h_ = SymbolicSize{"H"};
    auto pg_ = SymbolicSize{"PG"};
    auto device_ = SymbolicDevice{};
    auto bf16_ = SymbolicDType{};
    for (const auto& view : {residual, routed, shared, normed}) {
      TensorMatcher({m_, h_}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(view);
    }
    TensorMatcher({h_}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(weight);
    {
      const auto dt = bf16_.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1,
                   "residual/routed/shared/normed/weight must be bf16");
    }
    TensorMatcher({m_, h_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device_).verify(out_q);
    TensorMatcher({m_, pg_}).with_strides({-1, -1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(out_s);

    const auto m = m_.unwrap();
    const auto h = h_.unwrap();
    const auto tma_aligned_mn = ((m + 3) / 4) * 4;
    RuntimeCheck(h == kHidden, "moe_residual_rmsnorm_fp8_quant is specialized for H=4096, got ", h);
    RuntimeCheck(pg_.unwrap() == kPackedGroups, "packed scale groups must be ", kPackedGroups);
    RuntimeCheck(out_s.stride(0) == 1 && out_s.stride(1) == tma_aligned_mn,
                 "scale strides must be (1, tma_aligned_mn=", tma_aligned_mn, ")");
    if (m == 0) {
      return;
    }
    for (const auto& view : {residual, routed, shared, normed, weight, out_q}) {
      RuntimeCheck(reinterpret_cast<uintptr_t>(view.data_ptr()) % 16 == 0, "16-byte aligned tensors required");
    }
    const int64_t num_scale_elems = m + (kPackedGroups - 1) * tma_aligned_mn;
    LaunchKernel(dim3(static_cast<unsigned>(m)), dim3(kWarpSize, kWarps), device_.unwrap())(
        moe_residual_rmsnorm_fp8_quant_kernel,
        static_cast<__nv_bfloat16*>(residual.data_ptr()),
        static_cast<const __nv_bfloat16*>(routed.data_ptr()),
        static_cast<const __nv_bfloat16*>(shared.data_ptr()),
        static_cast<const __nv_bfloat16*>(weight.data_ptr()),
        static_cast<__nv_bfloat16*>(normed.data_ptr()),
        static_cast<uint8_t*>(out_q.data_ptr()),
        reinterpret_cast<uint8_t*>(out_s.data_ptr()),
        static_cast<uint32_t>(h), static_cast<uint32_t>(m), static_cast<uint32_t>(tma_aligned_mn),
        num_scale_elems, 0.f, static_cast<float>(eps));
  }
};
