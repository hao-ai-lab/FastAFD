# Benchmarks

Reruns FastAFD's reference decode measurements on GB200 NVL72 and checks each one against its recorded result.
`configs.json` holds 101 configurations that fit in **8 nodes** (4 GPUs each):

| Kind | What runs | Configurations |
| --- | --- | --- |
| `decode` | standard decoding: MiniMax-M2.5, Qwen3-235B, Qwen3.5-397B, 4K-16K context, attention:FFN 1:1-7:1 | 36 |
| `spec` | EAGLE-3 speculative decoding, 3 or 7 draft tokens, acceptance fixed at 1.8 tokens per step | 36 |
| `cad` | core-attention disaggregation (`MINISGL_AFD_MODE=cad`) on MiniMax-M2.5, 4K-16K | 15 |
| `variant` | the decode configuration with one setting changed: DeepEP + DeepGEMM instead of the fused MegaMoE kernels, 1-4 microbatches, cross-step overlap off, memory ratio | 14 |

A configuration names its runner (`scripts/experiments/afd/...`), the environment it ran with, the prompt rows
it reads and its reference result. `python benchmark.py list` prints them.

## Requirements

- GB200 NVL72 (or another NVLink domain of 4-GPU nodes), joined into one Ray cluster.
- The FastAFD GPU environment of the top-level README. The reference results ran the AFD workers with
  **FlashInfer 0.7.0.post1** (and nvidia-cutlass-dsl 4.7.1); `environment.cuda130.yml` installs 0.6.11.post2.
  Install 0.7.0.post1 into a separate directory and put it in front of the workers' path; the driver keeps the
  environment's version:
  ```bash
  python -m pip install --target $FI flashinfer-python==0.7.0.post1 flashinfer-cubin==0.7.0.post1 \
      nvidia-cutlass-dsl==4.7.1
  export MINISGL_WORKER_PYTHONPATH_PREPEND=$FI:$FI/nvidia_cutlass_dsl/dsl_packages
  ```
  (`dsl_packages` is normally added by a `.pth` file, which a `PYTHONPATH` entry does not read.)
- Checkpoints (`models.json` gives repository and revision): `MiniMaxAI/MiniMax-M2.5`,
  `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, `Qwen/Qwen3.5-397B-A17B-FP8`, and for speculative decoding the drafts
  `togethercomputer/Aurora-Spec-Minimax-M2.5` and `lmsys/Qwen3-235B-A22B-EAGLE3`. The Qwen3 draft needs
  `max_position_embeddings` raised for 32K contexts:
  `python benchmark.py prepare-draft eagle3-qwen3-235b <downloaded dir> $FASTAFD_MODEL_ROOT/Qwen3-235B-A22B-EAGLE3-pos262k`.

## Steps

1. **Prompt pools** (CPU, minutes per pool). Each configuration reads fixed-length prompts cut from the text in
   `prompts/`; the builder reproduces the recorded files byte for byte and exits 1 if a sha256 in
   `prompt_pools.json` differs:
   ```bash
   python build_prompts.py --model minimax-m2.5 --model-path $FASTAFD_MODEL_ROOT/MiniMax-M2.5 --out $FASTAFD_POOL_DIR \
       --pool 4096:20480 --pool 8192:10240 --pool 16384:5120
   python build_prompts.py --model qwen3-235b --model-path $FASTAFD_MODEL_ROOT/Qwen3-235B-A22B-Instruct-2507-FP8 \
       --out $FASTAFD_POOL_DIR --pool 4096:20480 --pool 8192:10240 --pool 16384:5120 --pool 32768:2560
   python build_prompts.py --model qwen3.5-397b --model-path $FASTAFD_MODEL_ROOT/Qwen3.5-397B-A17B-FP8 \
       --out $FASTAFD_POOL_DIR --pool 4096:37376 --pool 8192:27392 --pool 16384:18176
   ```
2. **One configuration** (on the first of its nodes, with the Ray cluster reachable at `MINISGL_RAY_ADDRESS`):
   ```bash
   export FASTAFD_MODEL_ROOT=/models FASTAFD_POOL_DIR=/data/pools FASTAFD_OUT=/data/benchmark-runs
   bash run.sh qwen3-235b-8k-7to1 <ip0>,<ip1>,...,<ip7>      # 8 nodes, the FFN node last
   ```
   The run directory `$FASTAFD_OUT/runs/<config>-<UTC>/` keeps the logs, `driver_env.txt`, `code_head.txt`,
   `config.json` and `result.json` (verdict, measured and reference step time).
3. **All results**: `python benchmark.py summary $FASTAFD_OUT/runs`.

Each run loads its model from scratch: plan 10-20 minutes, and the first run on a fresh cache compiles the
DeepGEMM / FlashInfer / MegaMoE kernels. Configurations of the same node count can run side by side on disjoint
nodes.

## What is measured

- **Step time** (`step_mean`, decode and CAD): the mean decode step over the pure-decode window, from the
  coordinator's per-step timing lines (`MINISGL_AFD_STEP_TIMES=1`): decode steps after the last prefill step,
  every attention rank at its full batch, consecutive steps whose context grew by exactly the batch. Throughput
  per GPU, all GPUs counted: `b / step x r / (r + 1)`, `b` requests per attention GPU.
- **Speculative decoding** (`spec_step_mean`): the same window, where a verify step may advance a request by
  1-4 (3 draft tokens) or 1-8 (7 draft tokens) tokens; throughput multiplies by the measured acceptance.
- **Runtime variants** (`wall_period`): 1 / slope of decode step ids against their 1-s log timestamps, which
  also counts host time outside the timed step (needed when cross-step overlap is off).

`run_stats.py` implements the three.

## Pass criterion

A configuration passes when its measured step time (or wall period) is within **+/-3%** of its reference
(`tolerance` in `configs.json`). Repeated reference runs of the same configuration varied by at most 0.7 %; the
band leaves room for node-to-node variation. Speculative decoding draws acceptance on the device around a fixed
mean of 1.8, so its throughput also moves with the drawn acceptance; the step time is the compared quantity.
