#!/usr/bin/env python3
"""Build the fixed-length prompt pools of the benchmark configurations from the prompt text in prompts/.

Every benchmark configuration reads its requests from one pool: a configuration with N requests uses the first
N lines of `<model>_isl<ISL>_r<ROWS>.txt`, one prompt per line, and each line encodes to exactly
ISL tokens on the server's tokenization path. The pools are deterministic: given the same tokenizer
files, this script reproduces the frozen files byte for byte (`prompt_pools.json` holds their sha256).

Row rule:
  stream  the deduplicated, stripped lines of prompts/prompts_512x16384_seed20260527.txt, then
          prompts/prompts_512x8192_seed20260527.txt, each encoded with the model tokenizer
          (add_special_tokens=False) and followed by encode("\\n\\n");
  starts  stride = (len(stream) - ISL - TAIL) // (ROWS - 1); row i starts at i * stride, moved forward
          by one token until its first 16 tokens are new and its first token decodes without U+FFFD;
  text    " ".join(decode(stream[s : s + ISL + TAIL]).split());
  cut     the shortest edit of the token count around ISL whose re-encoding is exactly ISL tokens;
          a window without such a cut is re-derived one token later (no padding);
  unique  a row whose first 16 ids repeat an earlier row's is re-derived one token later.

usage: build_prompts.py --model minimax-m2.5 --model-path /models/MiniMax-M2.5 --out POOL_DIR --pool 8192:10240 [--pool ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402

SOURCES = [REPO / "prompts/prompts_512x16384_seed20260527.txt", REPO / "prompts/prompts_512x8192_seed20260527.txt"]
TAIL, UNIQ, SEARCH = 128, 16, 64
BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
G: dict = {}


def _tm():
    if "tm" not in G:
        from minisgl.hf_support import load_tokenizer
        from minisgl.tokenizer.tokenize import TokenizeManager

        G["tm"] = TokenizeManager(load_tokenizer(G["model_path"]))
    return G["tm"]


def server_ids(text: str) -> list[int]:
    """The ids the AFD server computes for one prompt line (TokenizeManager.tokenize, the serving path)."""
    return _tm().tokenize([types.SimpleNamespace(input_ids=None, text=text)])[0].tolist()


def decode(ids) -> str:
    return _tm().tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def norm(text: str) -> str:
    return " ".join(text.split())


def build_stream() -> tuple[np.ndarray, int]:
    tok = _tm().tokenizer
    seen, lines = set(), []
    for path in SOURCES:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                lines.append(line)
    sep = tok.encode("\n\n", add_special_tokens=False)
    parts = []
    for i in range(0, len(lines), 64):
        for ids in tok(lines[i:i + 64], add_special_tokens=False)["input_ids"]:
            parts.append(np.asarray(ids, dtype=np.int32))
            parts.append(np.asarray(sep, dtype=np.int32))
    return np.concatenate(parts), len(lines)


def starts_for(stream: np.ndarray, isl: int, rows: int) -> tuple[np.ndarray, int]:
    stride = (len(stream) - isl - TAIL) // (rows - 1)
    assert stride >= UNIQ, f"stream too short for {rows} rows of {isl}"
    used, starts = set(), []
    for i in range(rows):
        s0 = s = i * stride
        while tuple(stream[s:s + UNIQ].tolist()) in used or "\ufffd" in decode(stream[s:s + 1]):
            s += 1
        assert s - s0 < stride and s + isl + TAIL <= len(stream), f"row {i}: no distinct start within the stride"
        used.add(tuple(stream[s:s + UNIQ].tolist()))
        starts.append(s)
    return np.asarray(starts, dtype=np.int64), stride


def row_from_start(stream: np.ndarray, s: int, isl: int):
    """(line, ids, how) for the window at s, or (None, None, why)."""
    text = norm(decode(stream[s:s + isl + TAIL]))
    ids = server_ids(text)
    if len(ids) < isl:
        return None, None, "short"
    base = text.count("\ufffd")
    fallback = None
    for d in range(SEARCH + 1):
        for length in ((isl,) if d == 0 else (isl - d, isl + d)):
            if length <= 0 or length > len(ids):
                continue
            cut = norm(decode(ids[:length]))
            got = server_ids(cut)
            if len(got) == isl:
                if cut.count("\ufffd") <= base:
                    return cut, got, "cut" if d == 0 else f"cut{length - isl:+d}"
                fallback = fallback or (cut, got)
    if fallback:
        return fallback[0], fallback[1], "cut_fffd"
    return None, None, "none"


def _convert(idx):
    stream, starts, isl = G["stream"], G["starts"], G["isl"]
    return [(i, *row_from_start(stream, int(starts[i]), isl)) for i in idx]


def _check(args):
    i0, lines, digests = args
    bad = []
    for k, line in enumerate(lines):
        fmt = bool(line) and line == line.strip() and not any(c in BREAKS for c in line)
        ids = server_ids(line)
        if not (fmt and len(ids) == G["isl"] and hashlib.sha1(np.asarray(ids, np.int32).tobytes()).digest() == digests[k]):
            bad.append(i0 + k)
    return len(lines), bad


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def build(model: str, out: Path, isl: int, rows: int, workers: int, stream: np.ndarray) -> dict:
    t0 = time.time()
    name = f"{model}_isl{isl}_r{rows}"
    starts, stride = starts_for(stream, isl, rows)
    G.update(stream=stream, starts=starts, isl=isl)
    txt = out / f"{name}.txt"
    kinds, rederived, used16, digests = {}, 0, set(), []
    chunks = [list(range(s, min(s + 64, rows))) for s in range(0, rows, 64)]
    with open(str(txt) + ".tmp", "w", encoding="utf-8") as f, mp.get_context("fork").Pool(workers) as pool:
        for res in pool.imap(_convert, chunks):
            for i, line, ids, how in res:
                s = int(starts[i])
                s0 = s
                while line is None or tuple(ids[:UNIQ]) in used16:   # re-derive one token later (rare)
                    s += 1
                    assert s - i * stride < stride, f"row {i}: no usable start"
                    if "\ufffd" in decode(stream[s:s + 1]):
                        continue
                    line, ids, how = row_from_start(stream, s, isl)
                rederived += s != s0
                used16.add(tuple(ids[:UNIQ]))
                kinds[how] = kinds.get(how, 0) + 1
                digests.append(hashlib.sha1(np.asarray(ids, np.int32).tobytes()).digest())
                f.write(line + "\n")
    os.replace(str(txt) + ".tmp", txt)
    # proof: re-read the file line by line and re-encode on the server path; every row is exactly ISL tokens, unchanged
    lines = txt.read_text(encoding="utf-8").split("\n")[:-1]
    jobs = [(i, lines[i:i + 64], digests[i:i + 64]) for i in range(0, len(lines), 64)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = list(pool.imap(_check, jobs, chunksize=1))
    bad = [b for _, x in res for b in x]
    meta = {"pool": name, "model": model, "isl": isl, "rows": rows, "txt_sha256": sha256(txt), "stride": int(stride),
            "rederived_rows": rederived, "kinds": kinds, "rows_checked": sum(n for n, _ in res), "bad_rows": bad[:20],
            "ok": not bad and len(lines) == rows, "seconds": round(time.time() - t0, 1)}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["minimax-m2.5", "qwen3-235b", "qwen3.5-397b"], help="model key (prompt_pools.json)")
    ap.add_argument("--model-path", required=True, help="checkpoint directory (only the tokenizer files are read)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pool", action="append", required=True, help="ISL:ROWS, repeatable")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--expect", type=Path, default=Path(__file__).with_name("prompt_pools.json"),
                    help="frozen sha256 per pool; a mismatch exits 1")
    args = ap.parse_args()
    G["model_path"] = args.model_path
    args.out.mkdir(parents=True, exist_ok=True)
    expect = json.loads(args.expect.read_text()) if args.expect and args.expect.exists() else {}
    stream, nlines = build_stream()
    print(f"{args.model}: stream {len(stream)} tokens from {nlines} unique lines", flush=True)
    rc = 0
    for spec in args.pool:
        isl, rows = (int(x) for x in spec.split(":"))
        meta = build(args.model, args.out, isl, rows, args.workers, stream)
        want = (expect.get(meta["pool"]) or {}).get("txt_sha256")
        meta["frozen_match"] = None if want is None else meta["txt_sha256"] == want
        print(json.dumps(meta), flush=True)
        rc |= int(not meta["ok"] or meta["frozen_match"] is False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
