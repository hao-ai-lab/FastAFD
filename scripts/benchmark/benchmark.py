#!/usr/bin/env python3
"""The benchmark configurations (configs.json): list them, print one configuration's environment, check a run.

  benchmark.py list [--nodes N] [--kind decode|spec|cad|variant]
  benchmark.py env CONFIG_ID          shell exports for run.sh (needs FASTAFD_MODEL_ROOT, FASTAFD_POOL_DIR, FASTAFD_OUT)
  benchmark.py show CONFIG_ID [--nodes IPS]   the configuration as JSON (run.sh keeps it in the run directory)
  benchmark.py check RUN_DIR          measure a finished run (run_stats.py) against the configuration's reference
  benchmark.py summary RESULT_DIR     one line per checked configuration, first comparable result per configuration
  benchmark.py prepare-draft KEY SRC DST   write a draft checkpoint copy with its config override (models.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_stats  # noqa: E402

CONFIGS = {c["id"]: c for c in json.loads((HERE / "configs.json").read_text())["configs"]}
MODELS = json.loads((HERE / "models.json").read_text())


def _env_key(key: str) -> str:
    return "FASTAFD_MODEL_" + key.upper().replace("-", "_").replace(".", "_")


def model_dir(key: str) -> str:
    spec = MODELS["models"].get(key) or MODELS["drafts"][key]
    if os.environ.get(_env_key(key)):
        return os.environ[_env_key(key)]
    root = os.environ.get("FASTAFD_MODEL_ROOT")
    if not root:
        sys.exit(f"set FASTAFD_MODEL_ROOT (directory holding {spec['dir']}) or {_env_key(key)}")
    return str(Path(root) / spec["dir"])


def prompt_file(config: dict) -> Path:
    """The first N lines of the configuration's pool, materialized once under $FASTAFD_OUT/prompts."""
    pool, rows = config["workload"]["pool"], config["workload"]["rows"]
    src = Path(os.environ["FASTAFD_POOL_DIR"]) / f"{pool}.txt"
    if not src.is_file():
        sys.exit(f"missing pool {src}: build it with build_prompts.py")
    dst = Path(os.environ["FASTAFD_OUT"]) / "prompts" / f"{pool}_n{rows}.txt"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(src, encoding="utf-8") as f:
            lines = [next(f) for _ in range(rows)]
        tmp = dst.with_suffix(".tmp")
        tmp.write_text("".join(lines), encoding="utf-8")
        tmp.replace(dst)
    return dst


def cmd_list(args) -> int:
    for c in CONFIGS.values():
        if (args.nodes and c["nodes"] != args.nodes) or (args.kind and c["kind"] != args.kind):
            continue
        ref = c["reference"].get("step_ms") or c["reference"].get("period_ms")
        print(f"{c['id']:44} nodes={c['nodes']} {c['metric']:15} reference={ref} ms")
    return 0


def cmd_env(args) -> int:
    c = CONFIGS[args.config]
    env = dict(c["env"])
    env.update(PRESET=c["runner"], MODEL_PATH=model_dir(c["model"]), PROMPT_FILE=str(prompt_file(c)),
               PROMPT_ROWS=str(c["workload"]["rows"]), WORKLOAD_POOL=c["workload"]["pool"],
               FASTAFD_CONFIG=c["id"], FASTAFD_CONFIG_NODES=str(c["nodes"]), FASTAFD_CONFIG_RATIO=str(c["ratio"]))
    if c["model"] == "qwen3.5-397b":
        env["PROMPT_POOL"] = env["PROMPT_FILE"]   # the Qwen3.5 runner hashes its pool file; the configuration's rows are the pool
    if c["draft"]:
        env["MINISGL_AFD_SPEC_DRAFT"] = model_dir(c["draft"])
    for k, v in env.items():
        print(f"export {k}={shlex.quote(v)}")
    return 0


def cmd_show(args) -> int:
    c = dict(CONFIGS[args.config])
    if args.nodes:
        c["nodes_used"] = args.nodes.split(",")
    print(json.dumps(c, indent=1))
    return 0


def check(run: Path, config_id: str | None = None) -> dict:
    meta = json.loads((run / "config.json").read_text()) if (run / "config.json").is_file() else {}
    c = CONFIGS[config_id or meta["id"]]
    try:
        out = run_stats.measure(run, c["ratio"], c["metric"], int(c["env"].get("MINISGL_AFD_SPEC_BLOCK", "4")))
    except (OSError, ValueError) as e:   # the runner failed before writing its logs
        out = {"ok": False, "why": f"{type(e).__name__}: {e}", "run": str(run), "metric": c["metric"]}
    target = c["reference"].get("step_ms") or c["reference"].get("period_ms")
    got = out.get("mean_ms") or out.get("period_ms")
    rel = (got / target - 1) if (out["ok"] and got) else None
    verdict = "NOT_COMPARABLE" if rel is None else ("PASS" if abs(rel) <= c["tolerance"] else "FAIL")
    return {"config": c["id"], "verdict": verdict, "target_ms": target, "measured_ms": got,
            "rel": None if rel is None else round(rel, 4), "tolerance": c["tolerance"],
            "tok_s_gpu": out.get("tok_s_gpu"), "reference_tok_s_gpu": c["reference"].get("tok_s_gpu"), "stats": out}


def cmd_check(args) -> int:
    res = check(args.run, args.config)
    print(json.dumps(res))
    return 0 if res["verdict"] == "PASS" else 1


def cmd_summary(args) -> int:
    """One line per configuration: its FIRST comparable measurement (PASS / FAIL), else its latest attempt. A later run of
    the same configuration never replaces a comparable one (reruns are diagnostics, kept in their own directory)."""
    latest: dict[str, dict] = {}
    for p in sorted(args.results.glob("*/result.json")):   # run dirs sort by UTC start time
        try:
            r = json.loads(p.read_text())
        except ValueError:
            continue
        if latest.get(r["config"], {}).get("verdict") in ("PASS", "FAIL"):
            continue
        latest[r["config"]] = r | {"dir": p.parent.name}
    n = {"PASS": 0, "FAIL": 0, "NOT_COMPARABLE": 0}
    for cid in CONFIGS:
        r = latest.get(cid)
        if r is None:
            print(f"{cid:44} NOT_RUN")
            continue
        n[r["verdict"]] = n.get(r["verdict"], 0) + 1
        rel = "" if r["rel"] is None else f"{100 * r['rel']:+.2f}%"
        print(f"{cid:44} {r['verdict']:14} target {r['target_ms']} ms  measured {r['measured_ms']} ms  {rel}  {r['dir']}")
    print(json.dumps(n | {"NOT_RUN": len(CONFIGS) - len(latest)}))
    return 0


def cmd_prepare_draft(args) -> int:
    spec = MODELS["drafts"][args.key]
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    config.update(spec.get("config_override", {}))
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for f in src.iterdir():
        if f.name != "config.json" and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    print(f"{dst}: config sha256 {hashlib.sha256((dst / 'config.json').read_bytes()).hexdigest()[:12]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--nodes", type=int)
    p.add_argument("--kind")
    p = sub.add_parser("env")
    p.add_argument("config")
    p = sub.add_parser("show")
    p.add_argument("config")
    p.add_argument("--nodes")
    p = sub.add_parser("check")
    p.add_argument("run", type=Path)
    p.add_argument("--config")
    p = sub.add_parser("summary")
    p.add_argument("results", type=Path)
    p = sub.add_parser("prepare-draft")
    p.add_argument("key")
    p.add_argument("src")
    p.add_argument("dst")
    args = ap.parse_args()
    return {"list": cmd_list, "env": cmd_env, "show": cmd_show, "check": cmd_check, "summary": cmd_summary,
            "prepare-draft": cmd_prepare_draft}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
