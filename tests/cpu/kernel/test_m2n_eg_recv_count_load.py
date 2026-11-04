"""The M2N EG pull loop must not peel sources until their counts add up to the expert total.

The AG count push stores each per-source count with a plain remote store and then bumps
the expert total with a relaxed system-scope atomic, so an EG rank can observe the total
before a per-source count. The pull loop's min-peeling then never terminates for slots past
the short sum: the EG rank silently misses the tag-2 barrier and every other rank reports a
300 s timeout there.

The kernel is JIT-built on GPU hosts only, so the CPU lane checks the load protocol in the
source itself: the block that loads an expert's per-source counts must re-read them until they
add up, with volatile loads, bounded below the NVLink barrier's 300 s so the EG side traps
first, without printf (it spills this warp role's 48-register budget), and with an acquire
fence before the source-index reads when a retry happened.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_KERNEL = pathlib.Path(
    "python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/sm100_mega_moe_m2n.cuh"
)
_BARRIER = pathlib.Path(
    "python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/comm/barrier.cuh"
)
_BLOCK_START = "// Load per-source-rank counts when expert changes"
_BLOCK_END = "// Round-robin source rank selection via iterative min-peeling"


def _code(text: str) -> str:
    """Drop `//` comments: the protocol must be in what the compiler sees."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


@pytest.fixture(scope="module")
def count_load_block() -> str:
    source = _KERNEL.read_text()
    eg_start = source.index("sm100_mega_moe_m2n_eg_impl(")
    start = source.index(_BLOCK_START, eg_start)
    end = source.index(_BLOCK_END, start)
    return _code(source[start:end])


def _timeout_seconds(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"(\d+)ll\s*\*\s*2000000000ll", text)]


def test_counts_are_reloaded_until_they_add_up_to_the_expert_total(count_load_block):
    assert "expert_end_idx - expert_start_idx" in count_load_block
    assert re.search(r"while\s*\(\s*true\s*\)", count_load_block)
    assert re.search(
        r"__reduce_add_sync\(0xffffffff,\s*sum_in_lane\)\s*==\s*expert_total\)\s*\n\s*break;",
        count_load_block,
    )


def test_every_count_read_bypasses_stale_copies(count_load_block):
    # A retry that re-reads a cached line would spin on the stale zero forever.
    assert "ptx::ld_volatile(workspace.get_expert_recv_count_ptr(" in count_load_block
    assert "*workspace.get_expert_recv_count_ptr(" not in count_load_block


def test_the_wait_is_bounded_and_fails_before_the_nvlink_barrier(count_load_block):
    assert "clock64()" in count_load_block
    assert re.search(r'asm\s+volatile\("trap;"\)', count_load_block)
    ours = _timeout_seconds(count_load_block)
    barrier = _timeout_seconds(_code(_BARRIER.read_text()))
    assert len(ours) == 1 and barrier
    assert ours[0] < min(barrier)


def test_the_dispatch_count_load_has_no_printf(count_load_block):
    # nvcc 13.0 for an EG instance: any printf here raised spill loads from
    # 116 to 220+ bytes; without it the spill profile is unchanged.
    assert "printf" not in count_load_block


def test_a_retry_fences_before_the_source_index_reads(count_load_block):
    assert re.search(r"if\s*\(\s*retried\s*\)\s*\n\s*fence_acquire_sys\(\);", count_load_block)
    assert re.search(r"retried\s*=\s*true;", count_load_block)
