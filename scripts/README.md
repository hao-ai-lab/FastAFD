# FastAFD Scripts

This directory is organized by workflow.

## Public Entry Points

Quickstart:

```bash
./scripts/quickstart/qwen3_30b_a3b_sample.sh
```

Correctness alignment:

```bash
./scripts/validate/qwen3_30b_a3b_alignment.sh
./scripts/validate/qwen3_30b_a3b_fastafd_alignment.sh
./scripts/validate/qwen35_397b_a17b_fp8_fastafd_alignment.sh
```

Large-scale AFD preset:

```bash
./scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh
```

Published Qwen3-235B and MiniMax M2.5 result presets live under
`scripts/experiments/afd/qwen3_235b/` and `scripts/experiments/afd/minimax_m25/`;
they dispatch through `scripts/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh`.
The Qwen3.5-397B-A17B-FP8 driver, `scripts/experiments/afd/qwen35/run_qwen35_fp8_afd_dynamicnode.sh`,
runs every role without tensor parallelism and dispatches through the same launcher.

## Layout

- `quickstart/`: mini-sgl sampling workflow.
- `serve/`: mini-sgl, FastAFD, and vLLM server launchers.
- `validate/`: mini-sgl/vLLM and FastAFD/vLLM alignment pipelines.
- `validate/check_alignment_report.py`: offline metric gate; recomputes rates
  from raw token ranks and checks them against the sampled continuations.
- `validate/reference_identity.py`: records which vLLM source, compiled
  extensions, torch and FlashInfer the vLLM reference imports (`reference_identity`).
- `reference/`: build script, patches and hash-pinned lock of the
  [reproducible vLLM reference](../README.md#reproducible-vllm-reference).
- `experiments/afd/qwen3_30b/`: the FastAFD launcher the presets dispatch to.
- `experiments/afd/qwen3_235b/`: Qwen3-235B published-result presets.
- `experiments/afd/minimax_m25/`: MiniMax M2.5 published-result presets.
- `experiments/afd/qwen35/`: the Qwen3.5-397B-A17B-FP8 driver.
- `experiments/vllm/`: the sharded vLLM alignment scorer.
- `benchmark/`: reruns the reference measurements (configurations, runner,
  prompt-pool builder, result check); see `benchmark/README.md`.
- `data_gen/`: prompt generation helper.
- `lib/`: shared shell helpers.

## Qwen3.5 Alignment Preset

The Qwen3.5 alignment preset is environment-configured: unsupported positional
arguments fail instead of being discarded. Use fresh `TMP_DIR` and `OUTPUT_JSON`
paths; the AFD aligner refuses existing artifacts. Fix `ALIGN_MIN_TOP10_RATE`
and the optional `ALIGN_MIN_TOP1_RATE` / `ALIGN_MIN_TOP100_RATE` before a run;
the top-10 default of 1.0 is not a universal model quality standard.
`VLLM_ENGINE_READY_TIMEOUT_S` (default 3600) bounds the vLLM reference's own
engine startup, which a cold FlashInfer JIT build can push past vLLM's 600 s.
