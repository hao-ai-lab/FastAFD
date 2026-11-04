#!/usr/bin/env python3
"""Check alignment metrics against raw ranks and the exact sampled continuations.

This is an offline metric gate, not proof of runtime/source identity or general
model quality. A launcher must propagate its own failure before calling it.
It also prints the report's `reference_identity` (which vLLM scored it; see
reference_identity.py) and warns when that is missing or incomplete; identity
never changes the verdict, so reports written before it was recorded stay valid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def rate(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be finite and within [0, 1]")
    return float(value)


def validate_report(report, sample_bundle, *, min_top10, min_top1=None, min_top100=1.0):
    thresholds = {"overall_top10_hit_rate": rate(min_top10, "min_top10")}
    if min_top1 is not None:
        thresholds["overall_top1_agreement_rate"] = rate(min_top1, "min_top1")
    thresholds["overall_top100_hit_rate"] = rate(min_top100, "min_top100")
    samples = sample_bundle.get("samples")
    results = report.get("results")
    summary = report.get("summary")
    if not isinstance(samples, list) or not samples:
        raise ValueError("sample bundle needs non-empty samples")
    if not isinstance(results, list) or len(results) != len(samples):
        raise ValueError("report must score every sampled request exactly once")
    if not isinstance(summary, dict):
        raise ValueError("report needs a summary")

    ranks = []
    for index, (sample, result) in enumerate(zip(samples, results)):
        ids = sample.get("generated_token_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"sample {index} has no generated tokens")
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"sample {index} has invalid token IDs")
        if "prompt" not in sample or result.get("prompt") != sample["prompt"]:
            raise ValueError(f"sample {index} prompt identity mismatch")
        if result.get("generated_token_ids") != ids:
            raise ValueError(f"sample {index} continuation identity mismatch")
        if result.get("num_generated_tokens") != len(ids):
            raise ValueError(f"sample {index} token count mismatch")
        steps = result.get("steps")
        if not isinstance(steps, list) or len(steps) != len(ids):
            raise ValueError(f"sample {index} has missing scored steps")
        for position, (token, step) in enumerate(zip(ids, steps)):
            if step.get("position") != position or step.get("token_id") != token:
                raise ValueError(f"sample {index} step identity mismatch")
            rank = step.get("rank_in_vllm")
            if type(rank) is not int or rank <= 0:
                raise ValueError(f"sample {index} has an invalid token rank")
            ranks.append(rank)

    if summary.get("num_prompts") != len(samples):
        raise ValueError("summary prompt count mismatch")
    if summary.get("total_generated_tokens") != len(ranks):
        raise ValueError("summary token denominator mismatch")
    measured = {
        "overall_top1_agreement_rate": sum(rank == 1 for rank in ranks) / len(ranks),
        "overall_top10_hit_rate": sum(rank <= 10 for rank in ranks) / len(ranks),
        "overall_top100_hit_rate": sum(rank <= 100 for rank in ranks) / len(ranks),
    }
    for name, expected in measured.items():
        reported = rate(summary.get(name), name)
        if not math.isclose(reported, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{name} disagrees with raw token ranks")
    failures = [name for name, threshold in thresholds.items() if measured[name] < threshold]
    return {
        "status": "FAIL" if failures else "PASS",
        "scope": "sample-bound token-rank alignment; not end-to-end quality certification",
        "num_prompts": len(samples),
        "total_generated_tokens": len(ranks),
        "metrics": measured,
        "thresholds": thresholds,
        "failed_metrics": failures,
    }


def reference_identity_warnings(report):
    """List what the report does not say about the vLLM reference that scored it."""
    identity = report.get("reference_identity")
    if identity is None:
        return ["report has no reference_identity; the vLLM reference that scored it is unidentified"]
    if not isinstance(identity, dict):
        return ["reference_identity is not a JSON object"]
    vllm = identity.get("vllm") if isinstance(identity.get("vllm"), dict) else {}
    warnings = [f"reference_identity lacks vllm.{key}" for key in ("version", "file") if not vllm.get(key)]
    git = vllm.get("git")
    if not (isinstance(git, dict) and git.get("head")) and not vllm.get("tree_sha256"):
        warnings.append("reference_identity names no vLLM source (neither git.head nor tree_sha256)")
    if not vllm.get("extensions"):
        warnings.append("reference_identity lists no compiled vLLM extensions")
    for name in ("torch", "flashinfer"):
        entry = identity.get(name)
        if not (isinstance(entry, dict) and entry.get("version")):
            warnings.append(f"reference_identity lacks {name}.version")
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--min-top10", type=float, required=True)
    parser.add_argument("--min-top1", type=float)
    # Top-100 = 1.0 is the default model acceptance gate; it binds only when
    # --min-top10 is below 1.0.
    parser.add_argument("--min-top100", type=float, default=1.0)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        samples = json.loads(args.sample_json.read_text(encoding="utf-8"))
        result = validate_report(
            report,
            samples,
            min_top10=args.min_top10,
            min_top1=args.min_top1,
            min_top100=args.min_top100,
        )
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(json.dumps({"status": "INVALID", "reason": str(error)}))
        return 2
    result["reference_identity"] = report.get("reference_identity")
    result["warnings"] = reference_identity_warnings(report)
    for warning in result["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
