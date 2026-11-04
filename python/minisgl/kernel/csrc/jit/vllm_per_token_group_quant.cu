// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * VENDORED (MINISGL_QWEN35_VLLM_KERNELS=1) from vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6
 * csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu
 * (sha256_12 380e1f066823, git blob_12 a0aee0cd9287):
 *   lines 530-695 per_token_group_quant_8bit_packed_register_kernel (byte for byte)
 *   the host side below re-creates per_token_group_quant_8bit_packed (lines 699-858) for bf16 -> fp8 e4m3fn,
 *   group 128: same checks, same grid / block (GetGroupsPerBlockX, lines 339-347), same PDL launch attribute.
 * vLLM instantiates the kernel with T = c10::BFloat16 and DST_DTYPE = c10::Float8_e4m3fn (VLLM_STABLE_DISPATCH_*,
 * libtorch_stable/dispatch_utils.h:69-81, 89-96) from the torch 2.13 headers of the vLLM env; FastAFD builds against
 * torch 2.10, so the two conversions the kernel uses are re-created from torch 2.13 (vLLM env
 * torch/include/torch/headeronly/util/):
 *   BFloat16::operator float()          BFloat16.h:142-150 (CUDA branch: __bfloat162float), sha256_12 cd57705b38b0
 *   Float8_e4m3fn(float) constructor    Float8_e4m3fn.h:171-243 + :255-256 (software RNE + saturation), sha256_12 7570fe37c23b
 *   fp32_from_bits / fp32_to_bits       floating_point_utils.h (CUDA branch: __uint_as_float / __float_as_uint), sha256_12 a37e889c098b
 * vLLM builds this file without --use_fast_math (CMakeLists.txt _C_stable_libtorch sources), as load_jit does here.
 */
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace vllm_torch213 {

__host__ __device__ inline float fp32_from_bits(uint32_t w) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#else
  float f;
  std::memcpy(&f, &w, sizeof(f));
  return f;
#endif
}

__host__ __device__ inline uint32_t fp32_to_bits(float f) {
#if defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#else
  uint32_t w;
  std::memcpy(&w, &f, sizeof(w));
  return w;
#endif
}

#define C10_HOST_DEVICE __host__ __device__
inline C10_HOST_DEVICE uint8_t fp8e4m3fn_from_fp32_value(float f) {
  /*
   * Binary representation of 480.0f, which is the first value
   * not representable in fp8e4m3fn range:
   * 0 1111 111 - fp8e4m3fn
   * 0 10000111 11100000000000000000000 - fp32
   */
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

  /*
   * A mask for converting fp32 numbers lower than fp8e4m3fn normal range
   * into denorm representation
   * magic number: ((127 - 7) + (23 - 3) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

  uint32_t f_bits = fp32_to_bits(f);

  uint8_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    if (f_bits > UINT32_C(0x7F800000)) {
      // NaN input → NaN output
      result = 0x7f;
    } else {
      // Finite overflow or inf → saturate to max finite value
      result = 0x7e;
    }
  } else {
    if (f_bits < (UINT32_C(121) << 23)) {
      // Input number is smaller than 2^(-6), which is the smallest
      // fp8e4m3fn normal number
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      // resulting mantissa is odd
      uint8_t mant_odd = (f_bits >> 20) & 1;

      // update exponent, rounding bias part 1
      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;

      // rounding bias part 2
      f_bits += mant_odd;

      // take the bits!
      result = static_cast<uint8_t>(f_bits >> 20);

      // Rounding may carry into the NaN bit pattern (0x7f); saturate to max
      if (result == 0x7f) {
        result = 0x7e;
      }
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}
#undef C10_HOST_DEVICE

struct alignas(2) BFloat16 {
  uint16_t x;
  __host__ __device__ inline operator float() const {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
  }
};

struct alignas(1) Float8_e4m3fn {
  uint8_t x;
  __host__ __device__ inline Float8_e4m3fn(float value) : x(fp8e4m3fn_from_fp32_value(value)) {}
};

}  // namespace vllm_torch213

// Register-resident fast path for group_size==128.
//
// Each thread holds 16 source elements (32 B = uint4 x 2) in registers across
// the absmax reduce -> scale compute -> quantize pipeline. No shared memory.
// UE8M0 scale extracted via bit math (bit-exact with exp2f(ceilf(log2f))).
//
// Loads two contiguous uint4s (16 B + 16 B = 32 B) per thread; on Blackwell
// nvcc fuses these into a single 256-bit LDG.E.256.
//
// Each thread quantizes 16 bf16/fp16 values, using 2 or 8 threads per group.
template <typename T, typename DST_DTYPE, int GROUP_SIZE, int kGroupsPerBlockX,
          int kRowsPerBlock>
__global__ void per_token_group_quant_8bit_packed_register_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    unsigned int* __restrict__ output_s_packed, const int padded_groups_per_row,
    const int groups_per_row, const int mn, const int output_q_mn_extent,
    const int tma_aligned_mn, const int64_t num_scale_elems, const float eps,
    const float min_8bit, const float max_8bit) {
  static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 128);
  constexpr int VEC_SIZE = 32 / sizeof(T);  // 16 for bf16/fp16
  constexpr int THREADS_PER_GROUP = GROUP_SIZE / VEC_SIZE;
  static_assert(GROUP_SIZE == THREADS_PER_GROUP * VEC_SIZE,
                "GROUP_SIZE must equal THREADS_PER_GROUP * VEC_SIZE");
  static_assert(32 % THREADS_PER_GROUP == 0,
                "THREADS_PER_GROUP must divide warp size for the shuffle "
                "mask to be valid");
  static_assert(
      kGroupsPerBlockX > 0 && (kGroupsPerBlockX & (kGroupsPerBlockX - 1)) == 0,
      "kGroupsPerBlockX must be a positive power of 2");
  static_assert(kRowsPerBlock > 0, "kRowsPerBlock must be positive");

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;

  const int sf_k_local = local_group_id % kGroupsPerBlockX;
  const int row_local = local_group_id / kGroupsPerBlockX;
  // Rows on grid.x: mn scales with tokens and can exceed the 65535 grid.y cap.
  const int sf_k_idx = blockIdx.y * kGroupsPerBlockX + sf_k_local;
  const int mn_idx = blockIdx.x * kRowsPerBlock + row_local;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  if (mn_idx >= tma_aligned_mn) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  const bool is_valid_group = (mn_idx < mn) && (sf_k_idx < groups_per_row);

  // Load 16 input elements (32 B) into registers as two adjacent uint4
  // loads. nvcc keeps these as 2x LDG.E.128 on sm_100; the per-thread cost
  // is dominated by HBM bandwidth at large MN, so a fused 256-bit load via
  // inline PTX gave no measurable speedup.
  // alignas(16) is required so the uint4* reinterpret_cast below is
  // well-defined for T == bf16/fp16 (default alignof is 2).
  alignas(16) T regs[VEC_SIZE];
  float local_absmax = eps;
  if (is_valid_group) {
    const T* group_input =
        input + static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
        sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
    uint4* dst = reinterpret_cast<uint4*>(&regs[0]);
    const uint4* src = reinterpret_cast<const uint4*>(group_input);
    dst[0] = src[0];
    dst[1] = src[1];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = fabsf(static_cast<float>(regs[i]));
      local_absmax = fmaxf(local_absmax, v);
    }
  }

#ifdef USE_ROCM
  const int lane_in_wave = threadIdx.x % warpSize;
  const unsigned long long mask = ((1ull << THREADS_PER_GROUP) - 1)
                                  << (lane_in_wave & ~(THREADS_PER_GROUP - 1));
#else
  const unsigned mask = ((1u << THREADS_PER_GROUP) - 1)
                        << ((threadIdx.x % 32) & ~(THREADS_PER_GROUP - 1));
#endif
#pragma unroll
  for (int offset = THREADS_PER_GROUP / 2; offset > 0; offset /= 2) {
    local_absmax =
        fmaxf(local_absmax,
              __shfl_xor_sync(mask, local_absmax, offset, THREADS_PER_GROUP));
  }

  float y_s = local_absmax / max_8bit;
  y_s = fmaxf(y_s, 1e-10f);
  uint32_t bits = __float_as_uint(y_s);
  uint32_t exp_bits = (bits >> 23) & 0xffu;
  uint32_t mant_bits = bits & 0x7fffffu;
  uint8_t exp_byte =
      static_cast<uint8_t>(exp_bits + (mant_bits != 0u ? 1u : 0u));

  // Lane 0 writes the packed scale byte.
  if (lane_id == 0) {
    const int sf_k_pack_idx = sf_k_idx / 4;
    const int pos = sf_k_idx % 4;
    const int out_idx = sf_k_pack_idx * tma_aligned_mn + mn_idx;
    if (is_valid_group) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = exp_byte;
    } else if (out_idx < num_scale_elems) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = 0;
    }
  }

  // For padded mn rows that fall within output_q's allocated extent, write
  // a uint4 of zeros to keep the buffer clean for downstream TMA loads.
  // Skip writes for sf_k padding (those positions don't exist in output_q).
  if (!is_valid_group) {
    if (sf_k_idx < groups_per_row && mn_idx >= mn &&
        mn_idx < output_q_mn_extent) {
      DST_DTYPE* group_output =
          static_cast<DST_DTYPE*>(output_q) +
          static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
          sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
      *reinterpret_cast<uint4*>(group_output) = make_uint4(0, 0, 0, 0);
    }
    return;
  }

  // Reconstruct y_s as a power-of-2 float and use its reciprocal.
  float y_s_q = __uint_as_float(static_cast<uint32_t>(exp_byte) << 23);
  float inv_y = 1.0f / y_s_q;

  // Quantize and pack into 16 fp8/int8 bytes (= uint4). VEC_SIZE==16 so we
  // fill four 32-bit words, four bytes each.
  uint32_t packed_lo = 0;
  uint32_t packed_lo_hi = 0;
  uint32_t packed_hi_lo = 0;
  uint32_t packed_hi = 0;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float q =
        fminf(fmaxf(static_cast<float>(regs[i]) * inv_y, min_8bit), max_8bit);
    DST_DTYPE qb = DST_DTYPE(q);
    uint8_t byte = *reinterpret_cast<uint8_t*>(&qb);
    const int shift = (i & 3) * 8;
    if (i < 4) {
      packed_lo |= static_cast<uint32_t>(byte) << shift;
    } else if (i < 8) {
      packed_lo_hi |= static_cast<uint32_t>(byte) << shift;
    } else if (i < 12) {
      packed_hi_lo |= static_cast<uint32_t>(byte) << shift;
    } else {
      packed_hi |= static_cast<uint32_t>(byte) << shift;
    }
  }

  uint4 packed_out =
      make_uint4(packed_lo, packed_lo_hi, packed_hi_lo, packed_hi);
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) +
      static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
      sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
  *reinterpret_cast<uint4*>(group_output) = packed_out;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

struct VllmPerTokenGroupQuantPackedKernel {
  // vLLM torch.ops._C.per_token_group_fp8_quant_packed(input, output_q, output_s_packed, group_size, eps,
  // fp8_min, fp8_max) (fp8_utils.py:744-752) for bf16 input, fp8 e4m3fn output, group_size 128.
  static void run(const tvm::ffi::TensorView input,            // [mn, k] bf16, contiguous
                  const tvm::ffi::TensorView output_q,         // [mn(+pad), k] uint8 view of fp8 e4m3fn
                  const tvm::ffi::TensorView output_s_packed,  // [mn, ceil(k/128/4)] int32, strides (1, tma_aligned_mn)
                  int64_t group_size, double eps, double min_8bit, double max_8bit) {
    using namespace host;
    auto mn_ = SymbolicSize{"MN"};
    auto k_ = SymbolicSize{"K"};
    auto mq_ = SymbolicSize{"MQ"};
    auto pk_ = SymbolicSize{"PK"};
    auto device_ = SymbolicDevice{};
    auto bf16_ = SymbolicDType{};
    TensorMatcher({mn_, k_}).with_dtype(bf16_).with_device<kDLCUDA>(device_).verify(input);
    {
      const auto dt = bf16_.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1, "input must be bf16");
    }
    TensorMatcher({mq_, k_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device_).verify(output_q);
    TensorMatcher({mn_, pk_}).with_strides({-1, -1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(output_s_packed);
    // per_token_group_quant.cu:704-756 (same checks)
    RuntimeCheck(group_size == 128, "FastAFD instantiates group_size 128 only, got ", group_size);
    const int64_t k = k_.unwrap();
    RuntimeCheck(k % group_size == 0, "input last dim k=", k, " is not divisible by group_size=", group_size);
    const int64_t mn = mn_.unwrap();
    const int64_t groups_per_row = k / group_size;
    const int64_t k_num_packed_sfk = (groups_per_row + 3) / 4;
    const int64_t tma_aligned_mn = ((mn + 3) / 4) * 4;
    const int64_t output_q_mn_actual = mq_.unwrap();
    RuntimeCheck(output_q_mn_actual >= mn, "output_q must have at least mn rows");
    const int64_t output_q_mn_extent = output_q_mn_actual < tma_aligned_mn ? output_q_mn_actual : tma_aligned_mn;
    RuntimeCheck(pk_.unwrap() == k_num_packed_sfk, "output_s_packed shape must be [mn, ", k_num_packed_sfk, "]");
    RuntimeCheck(output_s_packed.stride(0) == 1 && output_s_packed.stride(1) == tma_aligned_mn,
                 "output_s_packed strides must be [1, ", tma_aligned_mn, "]");
    if (mn == 0) {
      return;
    }
    const int threads_per_group = static_cast<int>(group_size / 16);
    const int64_t padded_groups_per_row = k_num_packed_sfk * 4;
    const int64_t num_scale_elems = mn + (k_num_packed_sfk - 1) * tma_aligned_mn;
    // GetGroupsPerBlockX (lines 339-347)
    const int kx = padded_groups_per_row % 16 == 0 ? 16 : (padded_groups_per_row % 8 == 0 ? 8 : 4);
    const int ry = 16 / kx;
    const int64_t row_blocks = (tma_aligned_mn + ry - 1) / ry;
    const int64_t sf_k_blocks = padded_groups_per_row / kx;
    const int num_threads = (kx * ry) * threads_per_group;
    RuntimeCheck(row_blocks <= static_cast<int64_t>(INT32_MAX) && sf_k_blocks <= 65535, "grid too large");
    const cudaStream_t stream = LaunchKernel::resolve_device(device_.unwrap());
    using T = vllm_torch213::BFloat16;
    using DST = vllm_torch213::Float8_e4m3fn;
    // LAUNCH_REG_KERNEL_INST (lines 776-799): cudaLaunchKernelEx with programmatic stream serialization
    auto launch = [&](auto kernel) {
      cudaLaunchConfig_t config = {};
      config.gridDim = dim3(static_cast<unsigned int>(row_blocks), static_cast<unsigned int>(sf_k_blocks));
      config.blockDim = dim3(num_threads);
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      cudaLaunchAttribute attrs[1];
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = 1;
      config.numAttrs = 1;
      config.attrs = attrs;
      CUDA_CHECK(cudaLaunchKernelEx(
          &config, kernel, static_cast<const T*>(input.data_ptr()), output_q.data_ptr(),
          reinterpret_cast<unsigned int*>(output_s_packed.data_ptr()), static_cast<int>(padded_groups_per_row),
          static_cast<int>(groups_per_row), static_cast<int>(mn), static_cast<int>(output_q_mn_extent),
          static_cast<int>(tma_aligned_mn), num_scale_elems, static_cast<float>(eps),
          static_cast<float>(min_8bit), static_cast<float>(max_8bit)));
    };
    if (kx == 16) {
      launch(per_token_group_quant_8bit_packed_register_kernel<T, DST, 128, 16, 1>);
    } else if (kx == 8) {
      launch(per_token_group_quant_8bit_packed_register_kernel<T, DST, 128, 8, 2>);
    } else {
      launch(per_token_group_quant_8bit_packed_register_kernel<T, DST, 128, 4, 4>);
    }
  }
};
