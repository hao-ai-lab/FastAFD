"""Compile + launch a captured vLLM inductor kernel exactly as inductor does .

Inductor (torch _inductor/runtime/triton_heuristics.py CachingAutotuner._precompile_config) compiles each config with
``triton.compile(ASTSource(fn, signature, constants, attrs), target, options)``: the signature types and the divisibility
attributes come from its triton_meta, NOT from the runtime values. A plain ``@triton.jit`` call would instead specialize on
the values (e.g. ``xnumel`` divisible by 16, or == 1 at one token; ``ks0`` / ``ks1`` would become i32), i.e. a different
kernel. Here each module of ``vllm_inductor_*`` carries inductor's SIGNATURE / DIVISIBLE and the CONFIG vLLM's traced run
used; the binary is compiled once per (kernel, device) and launched through the compiled kernel's runner (the launcher
inductor's Triton >= 3.2 path uses: every argument incl. the constexpr values, positional).
"""

from __future__ import annotations

import torch

_BINARIES: dict = {}


def compiled(mod):
    """The compiled binary of an vllm_inductor_<role>.py module for the current CUDA device."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    dev = torch.cuda.current_device()
    key = (mod.__name__, dev)
    b = _BINARIES.get(key)
    if b is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{mod.KERNEL_NAME}: first compile requested inside CUDA-graph capture")
        fn = getattr(mod, mod.KERNEL_NAME)
        cfg = mod.CONFIG
        constants = {k: v for k, v in cfg.items() if k not in ("num_warps", "num_stages")}
        attrs = {(i,): [["tt.divisibility", 16]] for i in mod.DIVISIBLE}
        major, minor = torch.cuda.get_device_capability(dev)
        target = GPUTarget("cuda", major * 10 + minor, 32)
        options = {"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"], "debug": False}
        b = triton.compile(ASTSource(fn, dict(mod.SIGNATURE), constants, attrs), target=target, options=options)
        _BINARIES[key] = b
    return b


def launch(mod, grid, *args):
    """Launch with inductor's argument list (all signature args in order; constexpr slots take the config values)."""
    cfg = mod.CONFIG
    full = []
    it = iter(args)
    for name, ty in mod.SIGNATURE.items():
        full.append(cfg[name] if ty == "constexpr" else next(it))
    compiled(mod)[grid](*full, stream=torch.cuda.current_stream().cuda_stream)


__all__ = ["compiled", "launch"]
