"""Speculative decoding for AFD (EAGLE-3 drafts): the attention side runs the draft and the verify step; the MoE side
only sees wider decode steps. Off unless MINISGL_AFD_SPEC=eagle3.

Package map: config (configuration, greedy acceptance, the reply encoding), attn_side (the draft and the verify
bookkeeping), device_inputs (device-resident committed lengths), graph (CUDA graphs of a verify step).
"""
