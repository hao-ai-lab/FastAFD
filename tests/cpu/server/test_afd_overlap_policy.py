"""M2N lane-owned graph buffers do not need the DeepEP re-entry fallback."""

import pytest

from minisgl.server.args import _afd_multi_mb_graph_auto_serializes
from minisgl.server import args as server_args


def _arguments(**overrides):
    return {
        "mode": "afd-serve", "afd_num_mb": 2,
        "afd_enable_attention_decode_graph": True,
        "afd_enable_model_decode_graph": True,
        "cuda_graph_max_bs": 16, "afd_decode_graph_bs": (),
        "afd_disable_overlap": False, "afd_force_multi_mb_graph_overlap": False,
        **overrides,
    }


@pytest.mark.parametrize("num_mb", [1, 2])
def test_m2n_overlap_needs_no_force_switch(monkeypatch, num_mb):
    monkeypatch.setenv("MINISGL_AFD_MOE_BACKEND", "megamoe_m2n")
    assert not _afd_multi_mb_graph_auto_serializes(_arguments(afd_num_mb=num_mb))


def test_deepep_graph_reentry_protection_remains(monkeypatch):
    monkeypatch.setenv("MINISGL_AFD_MOE_BACKEND", "deepep")
    assert _afd_multi_mb_graph_auto_serializes(_arguments())
    assert not _afd_multi_mb_graph_auto_serializes(
        _arguments(afd_force_multi_mb_graph_overlap=True)
    )


def test_default_backend_still_has_deepep_guard(monkeypatch):
    monkeypatch.delenv("MINISGL_AFD_MOE_BACKEND", raising=False)
    assert _afd_multi_mb_graph_auto_serializes(_arguments())


def test_explicit_serialized_diagnostic_is_not_overridden(monkeypatch):
    monkeypatch.setenv("MINISGL_AFD_MOE_BACKEND", "megamoe_m2n")
    assert not _afd_multi_mb_graph_auto_serializes(_arguments(afd_disable_overlap=True))


@pytest.mark.parametrize(("backend", "mb", "disabled"), [
    ("megamoe_m2n", 1, False), ("megamoe_m2n", 2, False),
    ("deepep", 1, False), ("deepep", 2, True),
])
def test_public_cli_keeps_effective_backend_overlap_policy(monkeypatch, backend, mb, disabled):
    monkeypatch.setenv("MINISGL_AFD_MOE_BACKEND", backend)
    monkeypatch.setattr(server_args, "_detect_local_ip", lambda: "127.0.0.1")
    parsed, _ = server_args.parse_args([
        "--mode", "afd-serve", "--model", "/unused-checkpoint", "--dtype", "bfloat16",
        "--cache-type", "naive", "--attention-backend", "trtllm",
        "--afd-moe-runner-backend", "deep_gemm", "--afd-num-mb", str(mb),
        "--cuda-graph-max-bs", "8", "--afd-attn-dp-size", "4",
        "--afd-attn-tp-size", "1", "--afd-mlp-dp-size", "4",
        "--afd-mlp-tp-size", "1", "--afd-mlp-ep-size", "4",
    ])
    assert parsed.afd_disable_overlap is disabled
    assert not parsed.afd_force_multi_mb_graph_overlap
    assert parsed.afd_num_mb == mb
    assert parsed.afd_enable_attention_decode_graph
    assert parsed.afd_enable_model_decode_graph
