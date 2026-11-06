<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="FastAFD" src="assets/logo.svg" width="320">
  </picture>
</p>

<p align="center"><b>Fast Attention–FFN Disaggregation for MoE Serving</b></p>

<p align="center">
| <a href="https://arxiv.org/abs/2611.07149">arXiv</a> | <a href="https://haoailab.com/blogs/fastafd/">Blog</a> | <a href="#">X</a> |
</p>

---

FastAFD is an open-source serving system for **large-scale Attention–FFN Disaggregation (AFD)** of MoE models.

## News

- **[2026-11-04]** FastAFD now supports **EAGLE-3 speculative decoding** and the
  **Qwen3.5 hybrid-attention** model (Qwen3.5-397B-A17B-FP8).
- **[2026-07-07]** FastAFD is open-sourced — **1.3–1.5× per-GPU decode throughput**
  over colocated MoE serving on Blackwell NVL72.

## Contents

- [How FastAFD Works](#how-fastafd-works)
- [Key Techniques](#key-techniques)
  - [N2M/M2N MoE Megakernel](#n2mm2n-moe-megakernel)
  - [Micro-Batch Ping-Pong Overlap](#micro-batch-ping-pong-overlap)
  - [Zero-Overhead Cluster Coordination](#zero-overhead-cluster-coordination)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Install](#install)
  - [Supported Models](#supported-models)
  - [Quickstart](#quickstart)
  - [Correctness Alignment](#correctness-alignment)
  - [Reproducible vLLM Reference](#reproducible-vllm-reference)
  - [Large-Scale AFD Experiments](#large-scale-afd-experiments)
  - [Speculative Decoding](#speculative-decoding)
  - [Core-Attention Disaggregation](#core-attention-disaggregation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## How FastAFD Works

<p align="center">
  <img alt="FastAFD Attention-FFN disaggregation" src="assets/afd-token-flow.gif" width="86%">
</p>

FastAFD splits MoE decoding into **attention servers** and **MLP servers**.
Attention servers own KV-cache attention, routing, and sampling; MLP servers
aggregate routed hidden states from many attention servers and run the experts.
Micro-batches overlap dispatch, expert execution, and combine so the MLP side
stays dense while activations move.

## Key Techniques

### N2M/M2N MoE Megakernel

FastAFD maps the AFD boundary to an asymmetric DeepEP-style dispatch/combine
path. During N2M, attention ranks send routed activations and MLP ranks receive
expert inputs; during M2N, MLP ranks return combined outputs and attention ranks
receive layer results. On the MLP side, FastAFD runs the repeated receive, group,
expert, combine, and send-back loop as a role-specialized MoE path with fewer
launches and synchronization points.

### Micro-Batch Ping-Pong Overlap

<p align="center">
  <img alt="FastAFD micro-batch pipeline" src="assets/afd-microbatch-pipeline.png" width="92%">
</p>

FastAFD splits each decode batch into micro-batches so N2M dispatch, expert
execution, M2N return, and attention-side work overlap across layers instead of
serializing at every AFD boundary.

### Zero-Overhead Cluster Coordination

<p align="center">
  <img alt="FastAFD zero-overhead coordinator" src="assets/zero-overhead-coordinator.png" width="92%">
  <br>
  <img alt="FastAFD Nsight Systems trace" src="assets/zero-overhead-nsys.png" width="92%">
</p>

FastAFD prepares the next decode plan while GPUs execute the current one. The
coordinator publishes micro-batch order, buffer slots, request ownership, and
peer metadata one step ahead, so scheduling stays off the decode critical path.

## Results

<p align="center">
  <img alt="FastAFD throughput improvement over colocated MoE serving" src="assets/fig_win.png" width="82%">
</p>

FastAFD improves per-GPU decode throughput by **1.35–1.45×** over colocated
MoE serving across Qwen3-235B-A22B-FP8 and MiniMax-M2.5 on GB200 NVL72.

## Getting Started

FastAFD is currently validated on `aarch64` with CUDA 13.0. The conda
environment is named `minisgl-cuda130` for compatibility with the current
runtime scripts.

### Install

```bash
conda env create -f environment.cuda130.yml
./scripts/enter_clean.sh minisgl-cuda130
pip install -e .
```

Install vLLM in the same clean shell if you want to run alignment checks:

```bash
python -m pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

Then apply the runtime overrides used by the AFD/DeepEP path:

```bash
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "nvidia-nccl-cu13==2.30.4" \
  "setuptools==80.10.2" \
  "fsspec==2026.2.0"
```

Validated package versions are `torch 2.10.0+cu130`, `triton 3.6.0`,
`nvidia-nccl-cu13 2.30.4`, and `vllm 0.19.0+cu130` when alignment is enabled.
`environment.cuda130.yml` pins FlashInfer `0.6.11.post2` (`flashinfer-python`
and `flashinfer-cubin`). The vLLM 0.19.0 wheel pins FlashInfer 0.6.6, so install
vLLM into a separate env (for example a clone made before this step) and pass it
to the validator with `--vllm-env`; FastAFD itself does not import vLLM.

### Supported Models

- Quickstart and correctness: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Published scaling presets: Qwen3-235B-A22B-FP8 and MiniMax-M2.5-FP8
- Hybrid linear attention: Qwen3.5-397B-A17B-FP8 (block FP8), which interleaves
  gated DeltaNet with full attention at 3:1
- Speculative decoding with EAGLE-3 drafts for Qwen3-235B-A22B and MiniMax-M2.5

### Quickstart

```bash
./scripts/quickstart/qwen3_30b_a3b_sample.sh
```

This starts a local mini-sgl server, samples the Qwen3-30B-A3B prompt set, and
writes artifacts under `reports/`.

### Correctness Alignment

```bash
./scripts/validate/qwen3_30b_a3b_alignment.sh
./scripts/validate/qwen3_30b_a3b_fastafd_alignment.sh
```

The first script checks mini-sgl against vLLM (single node, 4 GPUs via the `mp`
backend). The second checks the FastAFD serve path against vLLM and needs a
running Ray cluster with at least 8 GPUs (attention TP 4 + MLP TP 4).

Both alignment commands are thin 30B presets that call one reusable validator,
`scripts/validate/fastafd_vllm_alignment.sh`, which aligns the FastAFD serve path
against vLLM for any model or AFD attention/MLP layout. To validate a different
model or a custom split, call it directly (run with no arguments for the full
option list):

```bash
./scripts/validate/fastafd_vllm_alignment.sh \
  --env minisgl-cuda130 --model <hf-id-or-path> \
  --prompt-file <utf8-prompts.txt> \
  --attn-tp-size 4 --mlp-tp-size 4 --afd-mlp-ep-size 4 \
  --afd-max-running-requests 64 --sample-concurrency 64 \
  --max-new-tokens 64
```

Qwen3.5-397B-A17B-FP8 has its own preset over the same validator; it pins the
checkpoint revision and needs a Ray cluster with 8 GPUs and the
[reproducible vLLM reference](#reproducible-vllm-reference), passed as
`VLLM_ENV_NAME` or `FASTAFD_VLLM_REFERENCE`
(`./scripts/validate/qwen35_fastafd_alignment.sh --help` lists every control):

```bash
./scripts/validate/qwen35_397b_a17b_fp8_fastafd_alignment.sh
```

### Reproducible vLLM Reference

The Qwen3.5 alignment preset scores FastAFD's samples in a separate vLLM
environment, pinned by content rather than by a version string:

| Component | Pin |
|---|---|
| vLLM source | commit `d4004455d2357985830af10e432709b42c820455` (0.21.1rc1.dev278) plus the two patches in [`scripts/reference/patches/`](scripts/reference/patches) |
| vLLM compiled extensions | vLLM's precompiled wheel for that commit (sha256 `067f4458fbab21c7...`) |
| torch, FlashInfer | `2.11.0+cu130`, `0.6.11.post2`; every dependency hash-pinned in [`scripts/reference/requirements.lock`](scripts/reference/requirements.lock) |
| Platform | Linux aarch64, CPython 3.12, CUDA 13.0 toolkit |

Build it once into a new venv, then point the preset at it:

```bash
./scripts/reference/build_vllm_reference.sh --prefix "$HOME/vllm-reference" \
  --cuda-home /usr/local/cuda-13.0
VLLM_ENV_NAME="$HOME/vllm-reference" \
VLLM_ENV_SETUP="$HOME/vllm-reference/vllm-reference.env" \
  ./scripts/validate/qwen35_397b_a17b_fp8_fastafd_alignment.sh
```

`VLLM_ENV_NAME` is a conda env name (run through `conda run -n`) or, when it
contains `/`, the path of a Python environment whose own `bin/vllm` is run.
`VLLM_ENV_SETUP` is a bash file sourced only by that path-type env's vLLM
process; the one the build writes selects the CUDA toolkit FlashInfer compiles
with. Every alignment report records `reference_identity` (vLLM version and
path, git HEAD and the sha256 of its diff, the sha256 of each compiled
extension, torch and FlashInfer versions, `CUDA_HOME`);
`scripts/validate/check_alignment_report.py` prints it. Both patches are inert
unless their environment variables are set (a MoE trace file; KV-cache sizing
without memory profiling).

### Large-Scale AFD Experiments

Presets live under `scripts/experiments/afd/` and assume a running Ray GPU cluster,
launched from the activated `minisgl-cuda130` environment. Nodes are
auto-discovered from `RAY_ADDRESS=auto` (last node = MLP, the rest = attention);
on multi-node clusters use a shared local `MODEL_PATH` snapshot to avoid
per-rank Hugging Face cache resolution.

The 512-prompt long-context caches ship under `prompts/`. Regenerate them only
if needed (the 16K cache uses `--target-tokens 16384` and adds `--min-source-tokens 20000`):

```bash
PYTHONPATH=python python scripts/data_gen/generate_realistic_long_prompts.py \
  --model Qwen/Qwen3-235B-A22B-FP8 --target-tokens 8192 --count 512 \
  --output prompts/prompts_512x8192_seed20260527.txt --seed 20260527
```

Run a preset (minimal 4-node, no vLLM scoring or Nsight profiling):

```bash
MODEL_PATH=/path/to/Qwen3-235B-A22B-FP8 AFD_TOTAL_NODES=4 \
RUN_VLLM_ALIGNMENT=0 NSYS=0 \
bash scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
```

Optional environment knobs:

| Env var | Effect |
|---|---|
| `MODEL_PATH` | Local weight snapshot (recommended for multi-node; required offline, e.g. MiniMax-M2.5). |
| `AFD_NODE_LIST` / `AFD_MLP_NODE` | Explicit node placement instead of Ray auto-discovery (e.g. `node0,node1,node2,node3` / `node3`). |
| `RUN_VLLM_ALIGNMENT=1` | Cross-check FastAFD outputs against an expert-parallel vLLM baseline (see below). |
| `NSYS=1` + `MINISGL_RAY_NSYS_BIN=/path/to/nsys` | Capture Nsight Systems traces. `nsys` must also be on the `PATH` of the Ray daemons (`ray start`) on every node that hosts a profiled worker; the driver checks this before starting workers. |
| `CLEAN_FIRST=1` / `CLEAN_AFTER=1` | Before / after the run, call `scripts/kill_afd_ray_tasks.sh`, which SIGKILLs AFD and vLLM processes by command-line pattern on **every** live Ray node, other people's jobs included. Both default to `0`; set them only on a Ray cluster you have to yourself. |

Available presets:

| Model | Context | Default workload | Script |
|---|---:|---|---|
| Qwen3-235B-A22B-FP8 | 8K | 96 requests / attention GPU | `scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh` |
| Qwen3-235B-A22B-FP8 | 16K | 48 requests / attention GPU | `scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_16k_b48_dynamicnode_mb2_nsys_alignment.sh` |
| MiniMax-M2.5-FP8 | 8K | 72 requests / attention GPU | `scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh` |
| MiniMax-M2.5-FP8 | 16K | 36 requests / attention GPU | `scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_16k_b36_dynamicnode_mb2_nsys_alignment.sh` |
| Qwen3.5-397B-A17B-FP8 | 4K-16K | `PER_ATTN_GPU_BSZ` requests / attention GPU | `scripts/experiments/afd/qwen35/run_qwen35_fp8_afd_dynamicnode.sh` |

The Qwen3.5 driver runs every role without tensor parallelism (attention DP 4 /
TP 1 per node, experts DP 4 / EP 4 on the last node) and needs `MODEL_PATH`,
`AFD_NUM_MB` and the batch to be set explicitly; the configurations in
`scripts/benchmark/` give complete settings.

**vLLM cross-check (`RUN_VLLM_ALIGNMENT=1`).** Scores through an expert-parallel
vLLM baseline (`--all2all-backend deepep_low_latency --moe-backend deep_gemm`),
which needs two optional kernel packages in the same environment: `deep_ep`
(all-to-all dispatch/combine) and `deep_gemm` (FP8 grouped-GEMM experts — vLLM
only enables its `BATCHED_DEEPGEMM` MoE backend when `import deep_gemm` succeeds,
otherwise the engine fails to start). Install both via vLLM's
[`tools/ep_kernels`](https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md),
or build `deep_gemm` from source (`pip install --no-deps .` in a
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) clone). Both are independent
of the FastAFD serve path, which uses its own vendored kernels. On memory-tight
nodes, lower `--gpu-memory-utilization` (e.g. to `0.85`) via `VLLM_EXTRA_ARGS`
to avoid OOM during CUDA-graph capture.

Each preset prints the selected topology and output directory at startup. See
`scripts/README.md` for the full script map.

### Benchmarks

[`scripts/benchmark/`](scripts/benchmark/README.md) reruns FastAFD's reference decode
measurements on up to 8 nodes and checks each one against its recorded result:
standard decoding (MiniMax-M2.5, Qwen3-235B, Qwen3.5-397B, 4K-16K, attention:FFN
1:1-7:1), speculative decoding, core-attention disaggregation, and runtime
variants (MoE backend, microbatch count, cross-step overlap). It builds the
exact prompt pools, runs one configuration per command and reports PASS within
+/-3% of the reference step time.

```bash
python scripts/benchmark/benchmark.py list --kind spec
bash scripts/benchmark/run.sh qwen3-235b-8k-7to1 <ip0>,...,<ip7>   # FFN node last
```

### Speculative Decoding

Set `MINISGL_AFD_SPEC=eagle3` and `MINISGL_AFD_SPEC_DRAFT=<EAGLE-3 checkpoint>` on an
AFD launch. The draft and the verify step run on the attention GPUs; the MoE GPUs
only see a decode step `MINISGL_AFD_SPEC_BLOCK` tokens wider per request.

- Drafts: EAGLE-3 checkpoints in the `LlamaForCausalLMEagle3` format, e.g.
  `lmsys/Qwen3-235B-A22B-EAGLE3` and `togethercomputer/Aurora-Spec-Minimax-M2.5`.
- Targets: full-attention MoE models; hybrid targets such as Qwen3.5 are rejected
  at startup. Verification is greedy (temperature 0).
- Launch with `--attention-backend trtllm` (CUDA graphs) and an explicit
  `--num-pages`: the draft's weights and KV are not part of the automatic estimate.

| Env var | Effect |
|---|---|
| `MINISGL_AFD_SPEC_BLOCK` | Verify block = bonus token + (`BLOCK` - 1) draft tokens (default 8). |
| `MINISGL_AFD_SPEC_DEVICE_INPUTS=1` | Keep committed lengths on the device (no host round trip per step). |
| `MINISGL_AFD_SPEC_SIM_ACCEPT` / `_SIM_SEED` | Force a mean acceptance length, for performance studies only; the output is then not the target model's. |
| `MINISGL_AFD_SPEC_TIMING=1` | CUDA-event breakdown of each verify step. |
| `MINISGL_AFD_SPEC_EAGLE3_LAYERS` | Override the draft's auxiliary target layers (e.g. `1,46,90`). |

Greedy speculative output matches decoding without it (MiniMax-M2.5, block 3: 63 of
64 outputs identical, the last diverges at character 65), and acceptance lengths
match vLLM on the same prompts (Qwen3-235B, 1 / 3 drafts: 1.41 / 1.54-1.59 tokens per
step vs vLLM's 1.42-1.43 / 1.56-1.61). The implementation is the
[`python/minisgl/spec/`](python/minisgl/spec/__init__.py) package.

### Core-Attention Disaggregation

`MINISGL_AFD_MODE=cad` keeps only the core attention (KV store and decode
attention) and the KV cache on the attention GPUs, and runs every other part of
each layer (norms, QKV / O projections, router, MoE) on the model GPUs. Prefill
steps stay AFD steps. Its settings are the `MINISGL_AFD_CAD_*` environment
variables, listed in [`python/minisgl/cad/config.py`](python/minisgl/cad/config.py).

## Acknowledgements

FastAFD started from a [mini-sglang](python/minisgl) serving codebase inspired by
[SGLang](https://github.com/sgl-project/sglang), and builds on ideas and
components from the broader open-source LLM systems community, including
[DeepEP](https://github.com/deepseek-ai/DeepEP),
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM),
[vLLM](https://github.com/vllm-project/vllm).

## Citation

If you find FastAFD useful, please cite:

```bibtex
@misc{fu2026fastafddoesattentionffndisaggregation,
      title={FastAFD: When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?},
      author={Yichao Fu and Yuxuan Zhang and Minshen Zhang and Ruitian Wang and Junda Chen and Hao Zhang},
      year={2026},
      eprint={2611.07149},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2611.07149},
}
```
