import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate.check_alignment_report import validate_report


def evidence():
    sample = {"prompt": "A short prompt", "generated_token_ids": [41, 42]}
    result = {
        **sample,
        "num_generated_tokens": 2,
        "steps": [
            {"position": 0, "token_id": 41, "rank_in_vllm": 1},
            {"position": 1, "token_id": 42, "rank_in_vllm": 4},
        ],
    }
    return {
        "results": [result],
        "summary": {
            "num_prompts": 1,
            "total_generated_tokens": 2,
            "overall_top1_agreement_rate": 0.5,
            "overall_top10_hit_rate": 1.0,
            "overall_top100_hit_rate": 1.0,
        },
    }, {"samples": [copy.deepcopy(sample)]}


def test_gate_recomputes_metrics_and_applies_declared_thresholds():
    report, samples = evidence()
    assert validate_report(report, samples, min_top10=1.0)["status"] == "PASS"
    rejected = validate_report(report, samples, min_top10=1.0, min_top1=0.75)
    assert rejected["status"] == "FAIL"
    assert rejected["failed_metrics"] == ["overall_top1_agreement_rate"]


@pytest.mark.parametrize("invalid", [None, True, "1.0", float("nan"), float("inf"), -1, 2])
def test_missing_or_nonfinite_metrics_cannot_pass(invalid):
    report, samples = evidence()
    report["summary"]["overall_top10_hit_rate"] = invalid
    with pytest.raises(ValueError):
        validate_report(report, samples, min_top10=0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_thresholds_cannot_disable_the_gate(invalid):
    report, samples = evidence()
    with pytest.raises(ValueError):
        validate_report(report, samples, min_top10=invalid)


def top100_miss():
    """Rank 150 misses the top 100: top-1, top-10 and top-100 are all 1/2."""
    report, samples = evidence()
    report["results"][0]["steps"][1]["rank_in_vllm"] = 150
    report["summary"]["overall_top10_hit_rate"] = 0.5
    report["summary"]["overall_top100_hit_rate"] = 0.5
    return report, samples


def test_top100_gate_defaults_to_one():
    report, samples = top100_miss()
    rejected = validate_report(report, samples, min_top10=0.5)
    assert rejected["status"] == "FAIL"
    assert rejected["failed_metrics"] == ["overall_top100_hit_rate"]
    assert rejected["thresholds"]["overall_top100_hit_rate"] == 1.0
    assert validate_report(report, samples, min_top10=0.5, min_top100=0.5)["status"] == "PASS"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True, None])
def test_invalid_top100_threshold_cannot_disable_the_gate(invalid):
    report, samples = evidence()
    with pytest.raises(ValueError):
        validate_report(report, samples, min_top10=1.0, min_top100=invalid)


def test_cli_applies_the_top100_default_and_override(tmp_path):
    report, samples = top100_miss()
    report_path, sample_path = tmp_path / "report.json", tmp_path / "sample.json"
    report_path.write_text(json.dumps(report))
    sample_path.write_text(json.dumps(samples))
    checker = Path(__file__).resolve().parents[3] / "scripts/validate/check_alignment_report.py"
    base = [sys.executable, str(checker), "--report", str(report_path), "--sample-json", str(sample_path),
            "--min-top10", "0.5"]
    default = subprocess.run(base, capture_output=True, text=True, timeout=10, check=False)
    assert default.returncode == 1
    assert json.loads(default.stdout)["failed_metrics"] == ["overall_top100_hit_rate"]
    relaxed = subprocess.run(
        [*base, "--min-top100", "0.5"], capture_output=True, text=True, timeout=10, check=False
    )
    assert relaxed.returncode == 0
    assert json.loads(relaxed.stdout)["status"] == "PASS"


def test_stale_continuation_does_not_match_current_sample():
    report, samples = evidence()
    samples["samples"][0]["generated_token_ids"][0] = 99
    with pytest.raises(ValueError, match="identity"):
        validate_report(report, samples, min_top10=1.0)


@pytest.mark.parametrize("field,value", [("num_prompts", 2), ("total_generated_tokens", 256)])
def test_wrong_denominator_is_rejected(field, value):
    report, samples = evidence()
    report["summary"][field] = value
    with pytest.raises(ValueError, match="count|denominator"):
        validate_report(report, samples, min_top10=1.0)


def test_green_summary_cannot_hide_failed_raw_ranks():
    report, samples = evidence()
    report["results"][0]["steps"][1]["rank_in_vllm"] = 11
    with pytest.raises(ValueError, match="raw token ranks"):
        validate_report(report, samples, min_top10=1.0)


def test_empty_report_is_not_a_pass():
    with pytest.raises(ValueError, match="non-empty"):
        validate_report({"results": [], "summary": {}}, {"samples": []}, min_top10=0)


def test_cli_propagates_metric_failure(tmp_path):
    report, samples = evidence()
    report_path, sample_path = tmp_path / "report.json", tmp_path / "sample.json"
    report_path.write_text(json.dumps(report))
    sample_path.write_text(json.dumps(samples))
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/validate/check_alignment_report.py"),
            "--report", str(report_path), "--sample-json", str(sample_path),
            "--min-top10", "1", "--min-top1", "0.75",
        ],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "FAIL"


def test_qwen_wrapper_rejects_silently_discarded_cli_arguments():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["bash", str(root / "scripts/validate/qwen35_fastafd_alignment.sh"), "--attn-tp-size", "1"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    assert result.returncode == 2
    assert "unexpected arguments" in result.stderr
