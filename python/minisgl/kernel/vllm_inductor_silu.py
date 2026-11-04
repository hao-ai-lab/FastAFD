"""vLLM inductor kernel triton_poi_fused_mul_silu_slice_0 (role silu), captured from vLLM db9527a4's Qwen3.5-397B decode run:
<torchinductor cache>/rp/crpg5ytx5qbk4taqocrfvmlfbhjxbmgz6qr3rtufyoolulmidm6i.py (sha256_12 8b37eff5f3d8). The text from ``@triton.jit`` to the end is that file's, byte for byte
(sha256_12 b984283ec4be); inductor's decorator above it (size_hints {'x': 1048576}, autotune list) is
replaced by the compile contract below (inductor_launch.py) and the config vLLM ran:
  vLLM Triton entry QPFQRHESDLKP: TRACED (block 256, REG 26).
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice  # = torch._inductor.runtime.triton_helpers.libdevice

CAPTURED_FILE = 'rp/crpg5ytx5qbk4taqocrfvmlfbhjxbmgz6qr3rtufyoolulmidm6i.py'
CAPTURED_SHA12 = '8b37eff5f3d8'
JIT_TEXT_SHA12 = 'b984283ec4be'
VLLM_TRITON_ENTRY = 'QPFQRHESDLKP'
SIGNATURE = {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}
DIVISIBLE = (0, 1)
CONFIG = {'XBLOCK': 512, 'num_warps': 8, 'num_stages': 1}
CONFIG_PROVENANCE = 'TRACED (block 256, REG 26)'
KERNEL_NAME = 'triton_poi_fused_mul_silu_slice_0'


@triton.jit
def triton_poi_fused_mul_silu_slice_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x0 + ks1*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (ks0 + x0 + ks1*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + x1*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))), tmp9, xmask)
