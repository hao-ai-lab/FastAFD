#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace {

inline constexpr int kWarpSize = 32;
inline constexpr int kMaxExperts = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

template <int kWarpsPerBlock>
__launch_bounds__(kWarpSize * kWarpsPerBlock) __global__
void minimax_route_fp32_logits_tile_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [M, H]
    const float* __restrict__ gate_w,          // [E, H]
    float* __restrict__ logits,                // [M, E]
    int num_tokens,
    int hidden_size,
    int num_experts) {
  const int token = blockIdx.x;
  const int expert_tile = blockIdx.y * kWarpsPerBlock;
  if (token >= num_tokens) return;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int expert = expert_tile + warp;
  if (expert >= num_experts) return;

  const __nv_bfloat16* h_row = hidden + static_cast<size_t>(token) * hidden_size;
  const float* w_row = gate_w + static_cast<size_t>(expert) * hidden_size;
  float acc = 0.0f;
  for (int k = lane; k < hidden_size; k += kWarpSize) {
    acc = fmaf(__bfloat162float(h_row[k]), w_row[k], acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    logits[static_cast<size_t>(token) * num_experts + expert] = acc;
  }
}

template <int kWarpsPerBlock, bool kApplySigmoid = false>
__launch_bounds__(kWarpSize * kWarpsPerBlock) __global__
void minimax_route_fp32_logits_tile_shared_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [M, H]
    const float* __restrict__ gate_w,          // [E, H]
    float* __restrict__ logits,                // [M, E]
    uint8_t* __restrict__ x_q,                 // optional [M, H] e4m3
    uint8_t* __restrict__ x_sf,                // optional raw [M, H/32] ue8m0
    int num_tokens,
    int hidden_size,
    int num_experts) {
  ::host::minisgl_pdl_entry();
  extern __shared__ __nv_bfloat16 h_s[];
  const int token = blockIdx.x;
  const int expert_tile = blockIdx.y * kWarpsPerBlock;
  if (token >= num_tokens) return;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int expert = expert_tile + warp;

  const __nv_bfloat16* h_row = hidden + static_cast<size_t>(token) * hidden_size;
  if ((hidden_size & 7) == 0) {
    const int vecs = hidden_size / 8;
    const auto h_vec = reinterpret_cast<const uint4*>(h_row);
    auto s_vec = reinterpret_cast<uint4*>(h_s);
    for (int v = tid; v < vecs; v += blockDim.x) {
      s_vec[v] = __ldg(h_vec + v);
    }
  } else {
    for (int k = tid; k < hidden_size; k += blockDim.x) {
      h_s[k] = h_row[k];
    }
  }
  __syncthreads();

  if (x_q != nullptr && expert_tile == 0) {
    const int num_groups = hidden_size / 32;
    for (int g = tid; g < num_groups; g += blockDim.x) {
      float vals[32];
      const auto src_vec = reinterpret_cast<const uint4*>(h_s + g * 32);
      uint4 raw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        raw[j] = src_vec[j];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const auto h = reinterpret_cast<const __nv_bfloat16*>(raw + j);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          vals[j * 8 + k] = __bfloat162float(h[k]);
        }
      }
      float amax = 1e-4f;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        amax = fmaxf(amax, fabsf(vals[j]));
      }
      const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
      const uint32_t e =
          ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
      x_sf[static_cast<size_t>(token) * num_groups + g] =
          static_cast<uint8_t>(e);
      const float inv_scale = __uint_as_float((254u - e) << 23);
      uint32_t packed[8];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        reinterpret_cast<__nv_fp8x2_storage_t*>(packed)[j] =
            __nv_cvt_float2_to_fp8x2(
                make_float2(vals[2 * j] * inv_scale,
                            vals[2 * j + 1] * inv_scale),
                __NV_SATFINITE, __NV_E4M3);
      }
      auto dst = reinterpret_cast<uint4*>(
          x_q + static_cast<size_t>(token) * hidden_size + g * 32);
      dst[0] = *reinterpret_cast<uint4*>(packed);
      dst[1] = *reinterpret_cast<uint4*>(packed + 4);
    }
  }

  if (expert >= num_experts) return;

  const float* w_row = gate_w + static_cast<size_t>(expert) * hidden_size;
  float acc = 0.0f;
  int k = lane;
  for (; k + kWarpSize * 7 < hidden_size; k += kWarpSize * 8) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = k + u * kWarpSize;
      acc = fmaf(__bfloat162float(h_s[idx]), w_row[idx], acc);
    }
  }
  for (; k < hidden_size; k += kWarpSize) {
    acc = fmaf(__bfloat162float(h_s[k]), w_row[k], acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    if constexpr (kApplySigmoid) {
      acc = __frcp_rn(1.0f + __expf(-acc));
    }
    logits[static_cast<size_t>(token) * num_experts + expert] = acc;
  }
}

// Logits for kTokens tokens x kWarpsPerBlock experts per block: one warp per expert, the
// tokens' hidden rows in shared memory. Each gate-weight element is read once for all
// kTokens tokens (one token per block re-read the 3 MB FP32 gate from L2 for every token).
// Every token accumulates exactly as with kTokens = 1 (lane-strided fmaf, then the warp
// butterfly), so the logits, hence the routing, do not depend on kTokens.
//
// kLaneMajor: gate_w is stored lane-major ([E, H / 128, 32, 4], see minimax_route_lane_major),
// so the four elements lane l accumulates next (l + 32 * (4m + i), i = 0..3) are adjacent and
// one float4 load fetches them; the hidden rows are staged the same way in shared memory.
// Each lane still accumulates its elements in the original order: identical logits with a
// quarter of the load instructions.
template <int kWarpsPerBlock, int kTokens, int kExpertsPerWarp, bool kApplySigmoid = false,
          bool kLaneMajor = false, int kUnroll = 4>
__launch_bounds__(kWarpSize * kWarpsPerBlock) __global__
void minimax_route_fp32_logits_tiled_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [M, H]
    const float* __restrict__ gate_w,          // [E, H]
    float* __restrict__ logits,                // [M, E]
    uint8_t* __restrict__ x_q,                 // optional [M, H] e4m3
    uint8_t* __restrict__ x_sf,                // optional raw [M, H/32] ue8m0
    int num_tokens,
    int hidden_size,
    int num_experts) {
  ::host::minisgl_pdl_entry();
  extern __shared__ __nv_bfloat16 h_s[];  // [kTokens, H]
  const int token0 = blockIdx.x * kTokens;
  const int ntok = min(kTokens, num_tokens - token0);
  const int expert_tile = blockIdx.y * kWarpsPerBlock * kExpertsPerWarp;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int expert0 = expert_tile + warp * kExpertsPerWarp;

  if constexpr (kLaneMajor) {
    // Element k = 128 m + 32 i + l goes to 128 m + 4 l + i (hidden_size % 128 == 0).
    const int vecs = hidden_size / 8;
    for (int t = 0; t < ntok; ++t) {
      const auto h_vec = reinterpret_cast<const uint4*>(
          hidden + static_cast<size_t>(token0 + t) * hidden_size);
      __nv_bfloat16* s_row = h_s + t * hidden_size;
      for (int v = tid; v < vecs; v += blockDim.x) {
        const uint4 raw = __ldg(h_vec + v);
        const auto h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int k = v * 8 + q;
          const int r = k & 127;
          s_row[(k - r) + (r & 31) * 4 + (r >> 5)] = h[q];
        }
      }
    }
  } else if ((hidden_size & 7) == 0) {
    const int vecs = hidden_size / 8;
    for (int t = 0; t < ntok; ++t) {
      const auto h_vec = reinterpret_cast<const uint4*>(
          hidden + static_cast<size_t>(token0 + t) * hidden_size);
      auto s_vec = reinterpret_cast<uint4*>(h_s + t * hidden_size);
      for (int v = tid; v < vecs; v += blockDim.x) {
        s_vec[v] = __ldg(h_vec + v);
      }
    }
  } else {
    for (int t = 0; t < ntok; ++t) {
      const __nv_bfloat16* h_row = hidden + static_cast<size_t>(token0 + t) * hidden_size;
      for (int k = tid; k < hidden_size; k += blockDim.x) {
        h_s[t * hidden_size + k] = h_row[k];
      }
    }
  }
  __syncthreads();

  if (x_q != nullptr && expert_tile == 0) {
    const int num_groups = hidden_size / 32;
    for (int i = tid; i < ntok * num_groups; i += blockDim.x) {
      const int t = i / num_groups;
      const int g = i - t * num_groups;
      const int token = token0 + t;
      float vals[32];
      // Natural order: from shared memory, or from global memory when staged lane-major.
      const auto src_vec = kLaneMajor
          ? reinterpret_cast<const uint4*>(hidden + static_cast<size_t>(token) * hidden_size + g * 32)
          : reinterpret_cast<const uint4*>(h_s + t * hidden_size + g * 32);
      uint4 raw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        raw[j] = src_vec[j];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const auto h = reinterpret_cast<const __nv_bfloat16*>(raw + j);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          vals[j * 8 + k] = __bfloat162float(h[k]);
        }
      }
      float amax = 1e-4f;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        amax = fmaxf(amax, fabsf(vals[j]));
      }
      const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
      const uint32_t e =
          ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
      x_sf[static_cast<size_t>(token) * num_groups + g] =
          static_cast<uint8_t>(e);
      const float inv_scale = __uint_as_float((254u - e) << 23);
      uint32_t packed[8];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        reinterpret_cast<__nv_fp8x2_storage_t*>(packed)[j] =
            __nv_cvt_float2_to_fp8x2(
                make_float2(vals[2 * j] * inv_scale,
                            vals[2 * j + 1] * inv_scale),
                __NV_SATFINITE, __NV_E4M3);
      }
      auto dst = reinterpret_cast<uint4*>(
          x_q + static_cast<size_t>(token) * hidden_size + g * 32);
      dst[0] = *reinterpret_cast<uint4*>(packed);
      dst[1] = *reinterpret_cast<uint4*>(packed + 4);
    }
  }

  if (expert0 >= num_experts) return;

  // kExpertsPerWarp experts per warp: each hidden element read from shared memory serves all of
  // them, each weight element read from global memory serves all kTokens tokens. Experts past
  // num_experts reuse the last row and are not stored.
  const float* w_rows[kExpertsPerWarp];
#pragma unroll
  for (int e = 0; e < kExpertsPerWarp; ++e) {
    w_rows[e] = gate_w + static_cast<size_t>(min(expert0 + e, num_experts - 1)) * hidden_size;
  }
  float acc[kExpertsPerWarp][kTokens];
#pragma unroll
  for (int e = 0; e < kExpertsPerWarp; ++e) {
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      acc[e][t] = 0.0f;
    }
  }
  auto step = [&](int idx) {
    float w[kExpertsPerWarp];
#pragma unroll
    for (int e = 0; e < kExpertsPerWarp; ++e) {
      w[e] = w_rows[e][idx];
    }
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      const float h = __bfloat162float(h_s[t * hidden_size + idx]);
#pragma unroll
      for (int e = 0; e < kExpertsPerWarp; ++e) {
        acc[e][t] = fmaf(h, w[e], acc[e][t]);
      }
    }
  };
  if constexpr (kLaneMajor) {
    // Chunk m holds elements 128 m + 32 i + lane at 4 lane + i: four FMAs per weight float4.
    auto fma_chunk = [&](int m, const float4 (&w)[kExpertsPerWarp]) {
#pragma unroll
      for (int t = 0; t < kTokens; ++t) {
        const uint2 raw = *reinterpret_cast<const uint2*>(h_s + t * hidden_size + m * 128 + lane * 4);
        const float2 h01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 h23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
#pragma unroll
        for (int e = 0; e < kExpertsPerWarp; ++e) {
          acc[e][t] = fmaf(h01.x, w[e].x, acc[e][t]);
          acc[e][t] = fmaf(h01.y, w[e].y, acc[e][t]);
          acc[e][t] = fmaf(h23.x, w[e].z, acc[e][t]);
          acc[e][t] = fmaf(h23.y, w[e].w, acc[e][t]);
        }
      }
    };
    const int chunks = hidden_size / 128;
    int m = 0;
    // kUnroll chunks' weight loads in flight before their FMAs.
    for (; m + kUnroll - 1 < chunks; m += kUnroll) {
      float4 w[kUnroll][kExpertsPerWarp];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
        for (int e = 0; e < kExpertsPerWarp; ++e) {
          w[u][e] = __ldg(reinterpret_cast<const float4*>(w_rows[e] + (m + u) * 128) + lane);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        fma_chunk(m + u, w[u]);
      }
    }
    for (; m < chunks; ++m) {
      float4 w[kExpertsPerWarp];
#pragma unroll
      for (int e = 0; e < kExpertsPerWarp; ++e) {
        w[e] = __ldg(reinterpret_cast<const float4*>(w_rows[e] + m * 128) + lane);
      }
      fma_chunk(m, w);
    }
  } else {
    int k = lane;
    for (; k + kWarpSize * 7 < hidden_size; k += kWarpSize * 8) {
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        step(k + u * kWarpSize);
      }
    }
    for (; k < hidden_size; k += kWarpSize) {
      step(k);
    }
  }
#pragma unroll
  for (int e = 0; e < kExpertsPerWarp; ++e) {
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      acc[e][t] = warp_sum(acc[e][t]);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int e = 0; e < kExpertsPerWarp; ++e) {
#pragma unroll
      for (int t = 0; t < kTokens; ++t) {
        if (t < ntok && expert0 + e < num_experts) {
          float v = acc[e][t];
          if constexpr (kApplySigmoid) {
            v = __frcp_rn(1.0f + __expf(-v));
          }
          logits[static_cast<size_t>(token0 + t) * num_experts + expert0 + e] = v;
        }
      }
    }
  }
}

template <int kWarps, int kTokens, int kExpertsPerWarp, bool kApplySigmoid, int kUnroll>
void launch_route_logits_lane_major(
    const __nv_bfloat16* hidden, const float* gate_w, float* logits, uint8_t* x_q, uint8_t* x_sf,
    int rows, int hsize, int experts, cudaStream_t stream) {
  auto kernel = minimax_route_fp32_logits_tiled_kernel<kWarps, kTokens, kExpertsPerWarp, kApplySigmoid,
                                                       true, kUnroll>;
  const int smem = kTokens * hsize * static_cast<int>(sizeof(__nv_bfloat16));
  if (smem > 48 * 1024) {
    host::RuntimeCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) ==
                           cudaSuccess,
                       "MiniMax route logits: hidden size too large for the token tile");
  }
  const dim3 grid((rows + kTokens - 1) / kTokens,
                  (experts + kWarps * kExpertsPerWarp - 1) / (kWarps * kExpertsPerWarp));
  host::LaunchKernel(grid, kWarpSize * kWarps, stream, smem)
      .with_attr(host::minisgl_pdl_enabled())(kernel, hidden, gate_w, logits, x_q, x_sf, rows, hsize, experts);
}

// Block shape by token count (B200, 144 free SMs with the AG kernel on 4, cold L2, 256
// experts x 3072 FP32 gate): 36 / 72 / 108 / 144 / 216 rows took 18.3 / 18.7 / 19.3 / 20.9 /
// 26.4 us with one token per block and 9.2 / 12.0 / 12.1 / 14.2 / 20.4 us with these. Every
// shape yields logits bit-identical to one token per block.
template <bool kApplySigmoid>
void launch_route_logits_lane_major_auto(
    const __nv_bfloat16* hidden, const float* gate_w, float* logits, uint8_t* x_q, uint8_t* x_sf,
    int rows, int hsize, int experts, cudaStream_t stream) {
  if (rows <= 40) {
    launch_route_logits_lane_major<8, 2, 2, kApplySigmoid, 12>(hidden, gate_w, logits, x_q, x_sf, rows, hsize, experts, stream);
  } else if (rows <= 112) {
    launch_route_logits_lane_major<16, 3, 2, kApplySigmoid, 12>(hidden, gate_w, logits, x_q, x_sf, rows, hsize, experts, stream);
  } else if (rows <= 150) {
    launch_route_logits_lane_major<16, 4, 2, kApplySigmoid, 12>(hidden, gate_w, logits, x_q, x_sf, rows, hsize, experts, stream);
  } else {
    launch_route_logits_lane_major<16, 6, 2, kApplySigmoid, 8>(hidden, gate_w, logits, x_q, x_sf, rows, hsize, experts, stream);
  }
}

}  // namespace

struct MiniMaxRouteFusedKernel {
  static void run_logits(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto hidden_size = SymbolicSize{"H"};
    auto num_experts = SymbolicSize{"E"};
    auto device_ref = SymbolicDevice{};
    auto hidden_dtype = SymbolicDType{};

    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(logits);
    TensorMatcher({num_tokens, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(hidden_dtype)
        .verify(hidden);
    {
      const auto dt = hidden_dtype.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "MiniMax route logits hidden must be bf16");
    }
    TensorMatcher({num_experts, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(gate_weight);

    const auto rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto hsize = static_cast<int>(hidden_size.unwrap());
    if (rows == 0) return;
    RuntimeCheck(experts > 0 && experts <= kMaxExperts,
                 "MiniMax route logits supports 1..256 experts");

    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    constexpr int kWarps = 8;
    const dim3 grid_dot(rows, (experts + kWarps - 1) / kWarps);
    LaunchKernel(grid_dot, kWarpSize * kWarps, stream, hsize * sizeof(__nv_bfloat16))
        .with_attr(minisgl_pdl_enabled())(
        minimax_route_fp32_logits_tile_shared_kernel<kWarps>,
        static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        static_cast<const float*>(gate_weight.data_ptr()),
        static_cast<float*>(logits.data_ptr()),
        nullptr, nullptr,
        rows, hsize, experts);
  }

  static void run_logits4(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight) {
    run_logits_impl<4>(
        logits,
        hidden,
        gate_weight,
        tvm::ffi::Optional<tvm::ffi::TensorView>(),
        tvm::ffi::Optional<tvm::ffi::TensorView>());
  }

  static void run_logits16(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight) {
    run_logits_impl<16>(
        logits,
        hidden,
        gate_weight,
        tvm::ffi::Optional<tvm::ffi::TensorView>(),
        tvm::ffi::Optional<tvm::ffi::TensorView>());
  }

  static void run_logits4_quant(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_impl<4>(logits, hidden, gate_weight, x_q, x_sf);
  }

  static void run_logits16_quant(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_impl<16>(logits, hidden, gate_weight, x_q, x_sf);
  }

  static void run_logits16_quant_sigmoid(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_impl<16, true>(logits, hidden, gate_weight, x_q, x_sf);
  }

  // Lane-major gate (minimax_route_lane_major): token-tiled launch, float4 weight loads;
  // the logits (and the FP8 quant) are bit-identical to run_logits16 / run_logits16_quant*.
  static void run_logits_lm(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight_lane_major) {
    run_logits_lm_impl<false>(logits, hidden, gate_weight_lane_major,
                              tvm::ffi::Optional<tvm::ffi::TensorView>(),
                              tvm::ffi::Optional<tvm::ffi::TensorView>());
  }

  static void run_logits_lm_quant(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight_lane_major,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_lm_impl<false>(logits, hidden, gate_weight_lane_major, x_q, x_sf);
  }

  static void run_logits_lm_quant_sigmoid(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight_lane_major,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_lm_impl<true>(logits, hidden, gate_weight_lane_major, x_q, x_sf);
  }

 private:
  template <int kWarps, bool kApplySigmoid = false>
  static void run_logits_impl(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_q,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_sf) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto hidden_size = SymbolicSize{"H"};
    auto num_experts = SymbolicSize{"E"};
    auto device_ref = SymbolicDevice{};
    auto hidden_dtype = SymbolicDType{};

    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(logits);
    TensorMatcher({num_tokens, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(hidden_dtype)
        .verify(hidden);
    {
      const auto dt = hidden_dtype.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "MiniMax route logits hidden must be bf16");
    }
    TensorMatcher({num_experts, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(gate_weight);

    const auto rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto hsize = static_cast<int>(hidden_size.unwrap());
    if (rows == 0) return;
    RuntimeCheck(experts > 0 && experts <= kMaxExperts,
                 "MiniMax route logits supports 1..256 experts");
    RuntimeCheck(hsize % 128 == 0,
                 "MiniMax route external quant requires hidden_size % 128 == 0");

    uint8_t* x_q_ptr = nullptr;
    uint8_t* x_sf_ptr = nullptr;
    if (x_q.has_value()) {
      RuntimeCheck(x_sf.has_value(), "x_sf is required with x_q");
      x_q_ptr = static_cast<uint8_t*>(x_q.value().data_ptr());
      x_sf_ptr = static_cast<uint8_t*>(x_sf.value().data_ptr());
    }

    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    const dim3 grid_dot(rows, (experts + kWarps - 1) / kWarps);
    LaunchKernel(grid_dot, kWarpSize * kWarps, stream, hsize * sizeof(__nv_bfloat16))
        .with_attr(minisgl_pdl_enabled())(
        minimax_route_fp32_logits_tile_shared_kernel<kWarps, kApplySigmoid>,
        static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        static_cast<const float*>(gate_weight.data_ptr()),
        static_cast<float*>(logits.data_ptr()),
        x_q_ptr, x_sf_ptr,
        rows, hsize, experts);
  }

  template <bool kApplySigmoid>
  static void run_logits_lm_impl(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight_lane_major,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_q,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_sf) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto hidden_size = SymbolicSize{"H"};
    auto num_experts = SymbolicSize{"E"};
    auto device_ref = SymbolicDevice{};
    auto hidden_dtype = SymbolicDType{};

    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(logits);
    TensorMatcher({num_tokens, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(hidden_dtype)
        .verify(hidden);
    {
      const auto dt = hidden_dtype.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "MiniMax route logits hidden must be bf16");
    }
    TensorMatcher({num_experts, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(gate_weight_lane_major);

    const auto rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto hsize = static_cast<int>(hidden_size.unwrap());
    if (rows == 0) return;
    RuntimeCheck(experts > 0 && experts <= kMaxExperts,
                 "MiniMax route logits supports 1..256 experts");
    RuntimeCheck(hsize % 128 == 0, "MiniMax lane-major route logits require hidden_size % 128 == 0");

    uint8_t* x_q_ptr = nullptr;
    uint8_t* x_sf_ptr = nullptr;
    if (x_q.has_value()) {
      RuntimeCheck(x_sf.has_value(), "x_sf is required with x_q");
      x_q_ptr = static_cast<uint8_t*>(x_q.value().data_ptr());
      x_sf_ptr = static_cast<uint8_t*>(x_sf.value().data_ptr());
    }
    launch_route_logits_lane_major_auto<kApplySigmoid>(
        static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        static_cast<const float*>(gate_weight_lane_major.data_ptr()),
        static_cast<float*>(logits.data_ptr()), x_q_ptr, x_sf_ptr, rows, hsize, experts,
        LaunchKernel::resolve_device(device_ref.unwrap()));
  }
};
