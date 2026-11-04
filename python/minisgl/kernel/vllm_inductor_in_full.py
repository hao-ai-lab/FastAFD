"""vLLM inductor kernel triton_red_fused__to_copy_add_fused_add_rms_norm_per_token_group_fp8_quant_packed_1 (role in_full), captured from vLLM db9527a4's Qwen3.5-397B decode run:
<torchinductor cache>/jx/cjxbzgm2qydhlncxyrkhr6od5m5lojdiagp2ngtdni7g5ohruivt.py (sha256_12 f336a523aecc). The text from ``@triton.jit`` to the end is that file's, byte for byte
(sha256_12 85a07128c4b6); inductor's decorator above it (size_hints {'x': 1024, 'r0_': 4096}, autotune list) is
replaced by the compile contract below (inductor_launch.py) and the config vLLM ran:
  vLLM Triton entry FER3HHTSMTAF: TRACED (block 256, REG 44).
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice  # = torch._inductor.runtime.triton_helpers.libdevice

CAPTURED_FILE = 'jx/cjxbzgm2qydhlncxyrkhr6od5m5lojdiagp2ngtdni7g5ohruivt.py'
CAPTURED_SHA12 = 'f336a523aecc'
JIT_TEXT_SHA12 = '85a07128c4b6'
VLLM_TRITON_ENTRY = 'FER3HHTSMTAF'
SIGNATURE = {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}
DIVISIBLE = (0, 1, 2, 3, 4, 5, 6, 8)
CONFIG = {'XBLOCK': 1, 'R0_BLOCK': 1024, 'num_warps': 8, 'num_stages': 1}
CONFIG_PROVENANCE = 'TRACED (block 256, REG 44)'
KERNEL_NAME = 'triton_red_fused__to_copy_add_fused_add_rms_norm_per_token_group_fp8_quant_packed_1'


@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_per_token_group_fp8_quant_packed_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp3 + tmp10
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = tmp11.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp16, r0_mask & xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp20 + tmp27
        tmp29 = tl.full([1, 1], 4096.0, tl.float32)
        tmp30 = (tmp14 / tmp29)
        tmp31 = tl.full([1, 1], 1e-06, tl.float32)
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tl.full([1, 1], 1.0, tl.float32)
        tmp38 = tmp36 + tmp37
        tmp39 = tmp34 * tmp38
        tmp40 = tmp39.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp40, r0_mask & xmask)
