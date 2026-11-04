"""vLLM inductor kernel triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_0 (role post_gdn), captured from vLLM db9527a4's Qwen3.5-397B decode run:
<torchinductor cache>/oh/cohbndbaotpfjdoelrbwfxul6qc7b3wn3jyyahn77gtebzcdompi.py (sha256_12 44accf44c057). The text from ``@triton.jit`` to the end is that file's, byte for byte
(sha256_12 025ca6a53ef8); inductor's decorator above it (size_hints {'x': 1024, 'r0_': 4096}, autotune list) is
replaced by the compile contract below (inductor_launch.py) and the config vLLM ran:
  vLLM Triton entry 57PCQHLDPSNX: TRACED (n183 / 4K n245: block 256, REG 32).
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice  # = torch._inductor.runtime.triton_helpers.libdevice

CAPTURED_FILE = 'oh/cohbndbaotpfjdoelrbwfxul6qc7b3wn3jyyahn77gtebzcdompi.py'
CAPTURED_SHA12 = '44accf44c057'
JIT_TEXT_SHA12 = '025ca6a53ef8'
VLLM_TRITON_ENTRY = '57PCQHLDPSNX'
SIGNATURE = {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}
DIVISIBLE = (0, 1, 2, 3, 4, 5, 7)
CONFIG = {'XBLOCK': 1, 'R0_BLOCK': 1024, 'num_warps': 8, 'num_stages': 1}
CONFIG_PROVENANCE = 'TRACED (n183 / 4K n245: block 256, REG 32)'
KERNEL_NAME = 'triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_0'


@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_moe_forward_shared_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 4096.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp25, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp25, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 4096*x0), tmp25, r0_mask & xmask)
