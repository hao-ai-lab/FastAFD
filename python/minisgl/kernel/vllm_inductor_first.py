"""vLLM inductor kernel triton_red_fused__to_copy_add_rms_norm_0 (role first), captured from vLLM db9527a4's Qwen3.5-397B decode run:
<torchinductor cache>/5x/c5xygyeu53eaps65zpuv6izclg47pg37dbfjwalimmpckumpxgez.py (sha256_12 bffa025ec407). The text from ``@triton.jit`` to the end is that file's, byte for byte
(sha256_12 2483b96517e7); inductor's decorator above it (size_hints {'x': 1024, 'r0_': 4096}, autotune list) is
replaced by the compile contract below (inductor_launch.py) and the config vLLM ran:
  vLLM Triton entry VCYEJT4GOT3N: TRACED (block 256, REG 25).
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice  # = torch._inductor.runtime.triton_helpers.libdevice

CAPTURED_FILE = '5x/c5xygyeu53eaps65zpuv6izclg47pg37dbfjwalimmpckumpxgez.py'
CAPTURED_SHA12 = 'bffa025ec407'
JIT_TEXT_SHA12 = '2483b96517e7'
VLLM_TRITON_ENTRY = 'VCYEJT4GOT3N'
SIGNATURE = {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}
DIVISIBLE = (0, 1, 2, 4)
CONFIG = {'XBLOCK': 1, 'R0_BLOCK': 1024, 'num_warps': 8, 'num_stages': 1}
CONFIG_PROVENANCE = 'TRACED (block 256, REG 25)'
KERNEL_NAME = 'triton_red_fused__to_copy_add_rms_norm_0'


@triton.jit
def triton_red_fused__to_copy_add_rms_norm_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 4096.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp19, r0_mask & xmask)
