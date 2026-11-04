#!/usr/bin/env python3
"""Decode statistics of one AFD run directory, measured the way the reference results in configs.json were.

Input: RUN_DIR/ray_logs/afd_coordinator.log ("[afd-step-time]" lines, MINISGL_AFD_STEP_TIMES=1) and
RUN_DIR/ray_logs/afd-coordinator.log ("[afd_step]" lines with 1-s timestamps, MINISGL_AFD_LOG_STEPS=1).

step_mean (decode and CAD configurations): the MEAN step time over the pure-decode window: decode steps after the
  last prefill step, every DP rank at the peak batch, consecutive step ids, and every rank's context grown
  by exactly its batch since the previous step (drops tail steps that emit no token). Page stalls stay in.
  tok/s/GPU = b / mean x r / (r + 1), b = requests per attention GPU, all GPUs counted.
spec_step_mean (speculative decoding configurations): as step_mean, but a verify step may grow a rank's context by 1..BLOCK
  tokens per request; accept_len = context growth / requests over the kept steps and
  tok/s/GPU = b x accept_len / mean x r / (r + 1).
wall_period (runtime variants): 1 / slope of a least-squares fit of the first dp0 decode step id logged in
  each second against that second, over the interior seconds of the pure-decode window. Unlike step_mean
  it includes host time outside the timed step (the variant without cross-step overlap needs it).

usage: run_stats.py RUN_DIR --ratio R [--metric step_mean|spec_step_mean|wall_period]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

STEP_TIME = re.compile(r"\[afd-step-time\] step=(\d+) phase=(\w+) bucket=\d+ num_mb=(\d+) ms=([0-9.]+) "
                       r"real=\[([^\]]*)\](?: ctx=\[([^\]]*)\])?")
AFD_STEP = re.compile(r"\[afd_step\] step=(\d+) dp=(\d+) phase=(\w+) bs=(\d+)")
STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d)\|(\d\d:\d\d:\d\d)\|afd-coordinator\]")


def _ints(text: str | None) -> list[int] | None:
    return None if text is None else [int(x) for x in text.split(",") if x.strip()]


def _read_steps(run: Path) -> tuple[list[tuple[int, list[int], list[int] | None, float, int]], int]:
    rows, last_prefill = [], -1
    with open(run / "ray_logs" / "afd_coordinator.log", errors="replace") as f:
        for line in f:
            m = STEP_TIME.search(line)
            if not m:
                continue
            step, phase = int(m.group(1)), m.group(2)
            if phase != "decode":
                last_prefill = max(last_prefill, step)
                continue
            rows.append((step, _ints(m.group(5)), _ints(m.group(6)), float(m.group(4)), int(m.group(3))))
    return rows, last_prefill


def decode_window(rows, last_prefill: int, block: int) -> tuple[list, dict, int]:
    """The pure-decode steps (see the module docstring); block > 1 for speculative verify steps."""
    if not rows:
        return [], {}, 0
    peak = max(sum(r[1]) for r in rows)
    dec = sorted(r for r in rows if r[0] > last_prefill)
    kept, drop = [], {"not_peak": 0, "gap": 0, "ctx": 0}
    for prev, cur in zip(dec, dec[1:]):
        if sum(cur[1]) != peak:
            drop["not_peak"] += 1
        elif cur[0] != prev[0] + 1:
            drop["gap"] += 1
        elif cur[2] is None or prev[2] is None or any(not (q <= c - p <= block * q) for c, p, q in zip(cur[2], prev[2], cur[1])):
            drop["ctx"] += 1
        else:
            kept.append((prev, cur))
    return kept, drop, peak


def step_stats(run: Path, ratio: int, block: int) -> dict:
    rows, last_prefill = _read_steps(run)
    kept, drop, peak = decode_window(rows, last_prefill, block)
    if len(kept) < 20:
        return {"ok": False, "why": f"only {len(kept)} pure-decode steps", "dropped": drop}
    ms = [cur[3] for _, cur in kept]
    grow = sum(sum(c - p for c, p in zip(cur[2], prev[2])) for prev, cur in kept)
    reqs = sum(sum(cur[1]) for _, cur in kept)
    accept = grow / reqs if block > 1 else 1.0
    b = peak / (4 * ratio)
    mean, q = st.fmean(ms), st.quantiles(ms, n=100)
    return {"ok": True, "ratio": ratio, "b": b, "n": len(ms), "dropped": drop, "mean_ms": round(mean, 3),
            "p50_ms": round(st.median(ms), 3), "p99_ms": round(q[98], 3), "accept_len": round(accept, 4),
            "tok_s_gpu": round(b * accept / mean * 1e3 * ratio / (ratio + 1), 1),
            "tok_s_attention_gpu": round(b * accept / mean * 1e3, 1)}


def wall_period(run: Path, ratio: int) -> dict:
    rows, last_prefill = _read_steps(run)
    if not rows:
        return {"ok": False, "why": "no decode step lines"}
    peak = max(sum(r[1]) for r in rows)
    window = {r[0] for r in rows if r[0] > last_prefill and sum(r[1]) == peak}
    first_in_second: dict[int, int] = {}
    with open(run / "ray_logs" / "afd-coordinator.log", errors="replace") as f:
        for line in f:
            m, t = AFD_STEP.search(line), STAMP.search(line)
            if not (m and t) or m.group(2) != "0" or m.group(3) != "decode" or int(m.group(1)) not in window:
                continue
            sec = int(datetime.strptime(f"{t.group(1)} {t.group(2)}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            first_in_second.setdefault(sec, int(m.group(1)))
    secs = sorted(first_in_second)
    inner = secs[1:-1] if len(secs) >= 4 else []
    if len(inner) < 3:
        return {"ok": False, "why": f"only {len(inner)} interior seconds with dp0 decode lines"}
    xs, ys = inner, [first_in_second[s] for s in inner]
    mx, my = st.fmean(xs), st.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    period = 1000.0 / slope
    b = peak / (4 * ratio)
    return {"ok": True, "ratio": ratio, "b": b, "period_ms": round(period, 3), "fit_seconds": len(inner),
            "tok_s_gpu": round(b / period * 1e3 * ratio / (ratio + 1), 1),
            "step_mean_ms": round(st.fmean(r[3] for r in rows if r[0] in window), 3)}


def measure(run: Path, ratio: int, metric: str, block: int = 4) -> dict:
    if metric == "wall_period":
        out = wall_period(run, ratio)
    else:
        out = step_stats(run, ratio, block if metric == "spec_step_mean" else 1)
    out.update(run=str(run), metric=metric)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--ratio", type=int, required=True, help="attention:FFN node ratio r")
    ap.add_argument("--metric", default="step_mean", choices=["step_mean", "spec_step_mean", "wall_period"])
    ap.add_argument("--spec-block", type=int, default=4, help="verified tokens per step (draft length + 1)")
    args = ap.parse_args()
    out = measure(args.run, args.ratio, args.metric, args.spec_block)
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
