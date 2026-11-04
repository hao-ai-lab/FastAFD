"""CAD (core-attention disaggregation) decode mode for AFD: attention GPUs run only the core attention (KV store + decode
FMHA) and hold the KV cache; model GPUs run everything else per layer (norms, QKV / O projections, q/k norm + RoPE, router,
MoE). Prefill steps stay AFD steps. Off unless MINISGL_AFD_MODE=cad.

Package map: config / layout / window + kernel/cad_hop (the hop transport), program / runtime / attn_side (the step
runtime and the attention side), model_side / m2n_dual / model_worker / tokens (the model side).
"""
