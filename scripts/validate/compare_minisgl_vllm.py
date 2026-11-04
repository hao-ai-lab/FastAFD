#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from minisgl.hf_support import load_tokenizer


@dataclass
class MiniSGLSample:
    prompt: str
    generated_text: str
    generated_token_ids: list[int]
    ttft_seconds: float | None = None
    latency_seconds: float | None = None


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text. Can be provided multiple times.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 text file with one prompt per line. Empty lines and lines starting with '#' are ignored.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Repeat the loaded prompt set this many times in memory. Default: 1.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic continuations with mini-sgl and score the "
            "resulting tokens under a vLLM server via prompt_logprobs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Generate deterministic outputs from a mini-sgl server and optionally save them.",
    )
    sample_parser.add_argument("--model", required=True, help="Tokenizer model path or Hugging Face repo.")
    sample_parser.add_argument(
        "--minisgl-url",
        default="http://127.0.0.1:19193",
        help="mini-sgl base URL.",
    )
    add_prompt_args(sample_parser)
    sample_parser.add_argument("--max-tokens", type=int, default=64, help="Maximum decode tokens.")
    sample_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    sample_parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    sample_parser.add_argument("--top-k", type=int, default=1, help="Sampling top-k.")
    sample_parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS during sampling.")
    sample_parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help="How many prompts to sample from mini-sgl concurrently.",
    )
    sample_parser.add_argument(
        "--batch-submit",
        action="store_true",
        help="Submit all prompts to mini-sgl with one batch request.",
    )
    sample_parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds.")
    sample_parser.add_argument("--sample-json", type=Path, help="Optional path to write sampled outputs.")

    vllm_sample_parser = subparsers.add_parser(
        "vllm-sample",
        help="Generate deterministic outputs from a vLLM server and optionally save them.",
    )
    vllm_sample_parser.add_argument("--model", required=True, help="Tokenizer model path or Hugging Face repo.")
    vllm_sample_parser.add_argument("--vllm-url", default="http://127.0.0.1:8000", help="vLLM base URL.")
    add_prompt_args(vllm_sample_parser)
    vllm_sample_parser.add_argument("--max-tokens", type=int, default=64, help="Maximum decode tokens.")
    vllm_sample_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    vllm_sample_parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    vllm_sample_parser.add_argument("--top-k", type=int, default=-1, help="Sampling top-k.")
    vllm_sample_parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS during sampling.")
    vllm_sample_parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help="How many prompts to sample from vLLM concurrently.",
    )
    vllm_sample_parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds.")
    vllm_sample_parser.add_argument("--sample-json", type=Path, help="Optional path to write sampled outputs.")

    score_parser = subparsers.add_parser(
        "score",
        help="Score previously sampled mini-sgl outputs under a vLLM server.",
    )
    score_parser.add_argument("--sample-json", required=True, type=Path, help="Sample bundle produced by 'sample'.")
    score_parser.add_argument(
        "--model",
        help="Tokenizer model path or Hugging Face repo. Defaults to the value stored in sample-json.",
    )
    score_parser.add_argument("--vllm-url", default="http://127.0.0.1:8000", help="vLLM base URL.")
    score_parser.add_argument(
        "--prompt-logprobs",
        type=int,
        default=5,
        help="How many prompt logprobs to request from vLLM. Use 0 for chosen-token-only.",
    )
    score_parser.add_argument(
        "--score-concurrency",
        type=int,
        default=1,
        help="How many vLLM prompt-logprob scoring requests to keep in flight.",
    )
    score_parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds.")
    score_parser.add_argument("--output-json", type=Path, help="Optional path to write the scored report JSON.")
    score_parser.add_argument(
        "--reference-identity-json",
        type=Path,
        help="Reference identity written by vllm_server.sh --identity-json; copied into the report.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Run sampling against mini-sgl and scoring against vLLM in one step.",
    )
    compare_parser.add_argument("--model", required=True, help="Tokenizer model path or Hugging Face repo.")
    compare_parser.add_argument("--minisgl-url", default="http://127.0.0.1:19193", help="mini-sgl base URL.")
    compare_parser.add_argument("--vllm-url", default="http://127.0.0.1:8000", help="vLLM base URL.")
    add_prompt_args(compare_parser)
    compare_parser.add_argument("--max-tokens", type=int, default=64, help="Maximum decode tokens.")
    compare_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    compare_parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    compare_parser.add_argument("--top-k", type=int, default=1, help="Sampling top-k.")
    compare_parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS during sampling.")
    compare_parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help="How many prompts to sample from mini-sgl concurrently.",
    )
    compare_parser.add_argument(
        "--batch-submit",
        action="store_true",
        help="Submit all prompts to mini-sgl with one batch request.",
    )
    compare_parser.add_argument(
        "--prompt-logprobs",
        type=int,
        default=5,
        help="How many prompt logprobs to request from vLLM. Use 0 for chosen-token-only.",
    )
    compare_parser.add_argument(
        "--score-concurrency",
        type=int,
        default=1,
        help="How many vLLM prompt-logprob scoring requests to keep in flight.",
    )
    compare_parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds.")
    compare_parser.add_argument("--sample-json", type=Path, help="Optional path to write sampled outputs.")
    compare_parser.add_argument("--output-json", type=Path, help="Optional path to write the scored report JSON.")

    return parser.parse_args()


def load_prompts(prompt_list: list[str], prompt_file: Path | None, prompt_repeat: int) -> list[str]:
    if prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be >= 1.")
    prompts = list(prompt_list)
    if prompt_file is not None:
        for raw_line in prompt_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        raise ValueError("No prompts found after loading --prompt and/or --prompt-file.")
    return prompts * prompt_repeat


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def stream_json_sse(url: str, payload: dict[str, Any], timeout: float) -> Iterable[dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                yield json.loads(data)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def get_first_served_model(base_url: str, timeout: float) -> str:
    response = get_json(f"{base_url.rstrip('/')}/v1/models", timeout)
    data = response.get("data") or []
    if not data:
        raise RuntimeError(f"{base_url} did not return any served models.")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError(f"{base_url} returned a model card without an id.")
    return str(model_id)


def generate_with_minisgl(
    base_url: str,
    served_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout: float,
    ignore_eos: bool = False,
) -> MiniSGLSample:
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stream": True,
        "return_token_ids": True,
        "ignore_eos": ignore_eos,
    }
    text_parts: list[str] = []
    token_ids: list[int] = []
    started_at = time.perf_counter()
    first_token_at: float | None = None
    for chunk in stream_json_sse(f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout):
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            text_parts.append(content)
        chunk_token_ids = choice.get("token_ids") or []
        if chunk_token_ids and first_token_at is None:
            first_token_at = time.perf_counter()
        token_ids.extend(int(token_id) for token_id in chunk_token_ids)
    finished_at = time.perf_counter()
    return MiniSGLSample(
        prompt=prompt,
        generated_text="".join(text_parts),
        generated_token_ids=token_ids,
        ttft_seconds=(first_token_at - started_at) if first_token_at is not None else None,
        latency_seconds=finished_at - started_at,
    )


def generate_batch_with_minisgl(
    base_url: str,
    served_model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout: float,
    ignore_eos: bool = False,
) -> list[MiniSGLSample]:
    payload = {
        "model": served_model,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "return_token_ids": True,
        "ignore_eos": ignore_eos,
    }
    response = post_json(f"{base_url.rstrip('/')}/v1/chat/completions/batch", payload, timeout)
    raw_choices = response.get("choices") or []
    if len(raw_choices) != len(prompts):
        raise RuntimeError(
            f"Batch mini-sgl response returned {len(raw_choices)} choices for {len(prompts)} prompts."
        )
    choices = sorted(raw_choices, key=lambda choice: int(choice["index"]))
    samples: list[MiniSGLSample] = []
    for prompt, choice in zip(prompts, choices, strict=True):
        message = choice.get("message") or {}
        samples.append(
            MiniSGLSample(
                prompt=prompt,
                generated_text=str(message.get("content") or ""),
                generated_token_ids=[int(token_id) for token_id in (choice.get("token_ids") or [])],
            )
        )
    return samples


def generate_with_vllm(
    base_url: str,
    served_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout: float,
    ignore_eos: bool = False,
) -> MiniSGLSample:
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        "return_token_ids": True,
        "ignore_eos": ignore_eos,
    }
    if top_k >= 0:
        payload["top_k"] = top_k
    response = post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("vLLM response did not contain any choices.")
    choice = choices[0]
    token_ids = choice.get("token_ids")
    if token_ids is None:
        raise RuntimeError("vLLM response did not include token_ids; return_token_ids may be unsupported.")
    return MiniSGLSample(
        prompt=prompt,
        generated_text=str(choice.get("text") or ""),
        generated_token_ids=[int(token_id) for token_id in token_ids],
    )


def sample_prompts_with_minisgl(
    prompts: list[str],
    base_url: str,
    served_model: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout: float,
    sample_concurrency: int,
    batch_submit: bool = False,
    ignore_eos: bool = False,
) -> list[MiniSGLSample]:
    if batch_submit:
        return generate_batch_with_minisgl(
            base_url=base_url,
            served_model=served_model,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            timeout=timeout,
            ignore_eos=ignore_eos,
        )

    if sample_concurrency <= 1 or len(prompts) <= 1:
        return [
            generate_with_minisgl(
                base_url=base_url,
                served_model=served_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                timeout=timeout,
                ignore_eos=ignore_eos,
            )
            for prompt in prompts
        ]

    samples: list[MiniSGLSample | None] = [None] * len(prompts)
    max_workers = min(sample_concurrency, len(prompts))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                generate_with_minisgl,
                base_url=base_url,
                served_model=served_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                timeout=timeout,
                ignore_eos=ignore_eos,
            ): index
            for index, prompt in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            samples[index] = future.result()

    return [sample for sample in samples if sample is not None]


def sample_prompts_with_vllm(
    prompts: list[str],
    base_url: str,
    served_model: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout: float,
    sample_concurrency: int,
    ignore_eos: bool = False,
) -> list[MiniSGLSample]:
    if sample_concurrency <= 1 or len(prompts) <= 1:
        return [
            generate_with_vllm(
                base_url=base_url,
                served_model=served_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                timeout=timeout,
                ignore_eos=ignore_eos,
            )
            for prompt in prompts
        ]

    samples: list[MiniSGLSample | None] = [None] * len(prompts)
    max_workers = min(sample_concurrency, len(prompts))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                generate_with_vllm,
                base_url=base_url,
                served_model=served_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                timeout=timeout,
                ignore_eos=ignore_eos,
            ): index
            for index, prompt in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                samples[index] = future.result()
            except Exception as exc:
                raise RuntimeError(f"vLLM sampling failed for prompt index {index}") from exc

    return [sample for sample in samples if sample is not None]


def score_with_vllm(
    base_url: str,
    served_model: str,
    prompt_token_ids: list[int],
    prompt_logprobs: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": served_model,
        "prompt": prompt_token_ids,
        "max_tokens": 0,
        "echo": True,
        "stream": False,
        "prompt_logprobs": prompt_logprobs,
        "return_token_ids": True,
    }
    response = post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("vLLM response did not contain any choices.")
    return choices[0]


def normalize_logprob_entry(entry: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not entry:
        return {}
    return {int(raw_token_id): value for raw_token_id, value in entry.items()}


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def safe_exp_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(values) / len(values))


def hit_rate_at_k(ranks: list[int], k: int) -> float | None:
    if not ranks:
        return None
    return sum(1 for rank in ranks if rank <= k) / len(ranks)


def compare_prompt(
    tokenizer,
    sample: MiniSGLSample,
    scored_choice: dict[str, Any],
    requested_topk: int,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=True)
    combined_ids = prompt_ids + sample.generated_token_ids

    returned_prompt_ids = scored_choice.get("prompt_token_ids")
    if returned_prompt_ids is not None and list(returned_prompt_ids) != combined_ids:
        raise RuntimeError("vLLM prompt_token_ids did not match local prompt+generation token ids.")

    prompt_logprobs = scored_choice.get("prompt_logprobs")
    if prompt_logprobs is None:
        raise RuntimeError("vLLM response did not include prompt_logprobs.")
    if len(prompt_logprobs) != len(combined_ids):
        raise RuntimeError(
            f"Expected {len(combined_ids)} prompt_logprobs entries, got {len(prompt_logprobs)}."
        )

    generated_steps: list[dict[str, Any]] = []
    top1_hits = 0
    requested_topk_hits = 0
    ranks: list[int] = []
    chosen_logprobs: list[float] = []
    first_non_top1_position: int | None = None

    for index, token_id in enumerate(sample.generated_token_ids, start=len(prompt_ids)):
        entry = normalize_logprob_entry(prompt_logprobs[index])
        token_info = entry.get(token_id)
        if token_info is None:
            raise RuntimeError(
                f"Prompt logprobs entry at position {index} did not include token {token_id}."
            )

        rank = int(token_info["rank"])
        logprob = float(token_info["logprob"])
        is_top1 = rank == 1
        if is_top1:
            top1_hits += 1
        elif first_non_top1_position is None:
            first_non_top1_position = index - len(prompt_ids)

        if requested_topk > 0 and rank <= requested_topk:
            requested_topk_hits += 1

        best_token_id = next(
            (candidate_token_id for candidate_token_id, value in entry.items() if int(value["rank"]) == 1),
            None,
        )

        generated_steps.append(
            {
                "position": index - len(prompt_ids),
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "rank_in_vllm": rank,
                "logprob_in_vllm": logprob,
                "is_top1_in_vllm": is_top1,
                "vllm_top1_token_id": best_token_id,
                "vllm_top1_token_text": tokenizer.decode([best_token_id]) if best_token_id is not None else None,
                "vllm_candidates": [
                    {
                        "token_id": candidate_token_id,
                        "token_text": value.get("decoded_token"),
                        "rank": int(value["rank"]),
                        "logprob": float(value["logprob"]),
                    }
                    for candidate_token_id, value in sorted(entry.items(), key=lambda kv: int(kv[1]["rank"]))
                ],
            }
        )

        ranks.append(rank)
        chosen_logprobs.append(logprob)

    num_generated = len(sample.generated_token_ids)
    return {
        "prompt": sample.prompt,
        "prompt_token_ids": prompt_ids,
        "generated_text": sample.generated_text,
        "generated_token_ids": sample.generated_token_ids,
        "num_generated_tokens": num_generated,
        "top1_agreement_rate": (top1_hits / num_generated) if num_generated else None,
        "top10_hit_rate": hit_rate_at_k(ranks, 10),
        "top100_hit_rate": hit_rate_at_k(ranks, 100),
        "requested_topk": requested_topk if requested_topk > 0 else None,
        "requested_topk_hit_rate": (
            (requested_topk_hits / num_generated) if num_generated and requested_topk > 0 else None
        ),
        "avg_rank_in_vllm": safe_mean([float(rank) for rank in ranks]),
        "max_rank_in_vllm": max(ranks) if ranks else None,
        "mean_logprob_in_vllm": safe_mean(chosen_logprobs),
        "perplexity_under_vllm": safe_exp_mean([-logprob for logprob in chosen_logprobs]) if chosen_logprobs else None,
        "first_non_top1_position": first_non_top1_position,
        "steps": generated_steps,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts = [int(result["num_generated_tokens"]) for result in results]
    top1_numerators = [
        sum(1 for step in result["steps"] if step["is_top1_in_vllm"])
        for result in results
    ]
    total_tokens = sum(token_counts)
    total_top1_hits = sum(top1_numerators)
    all_ranks = [
        float(step["rank_in_vllm"])
        for result in results
        for step in result["steps"]
    ]
    all_logprobs = [
        float(step["logprob_in_vllm"])
        for result in results
        for step in result["steps"]
    ]
    return {
        "num_prompts": len(results),
        "total_generated_tokens": total_tokens,
        "overall_top1_agreement_rate": (total_top1_hits / total_tokens) if total_tokens else None,
        "overall_top10_hit_rate": hit_rate_at_k([int(rank) for rank in all_ranks], 10),
        "overall_top100_hit_rate": hit_rate_at_k([int(rank) for rank in all_ranks], 100),
        "overall_avg_rank_in_vllm": safe_mean(all_ranks),
        "overall_max_rank_in_vllm": max(all_ranks) if all_ranks else None,
        "overall_mean_logprob_in_vllm": safe_mean(all_logprobs),
        "overall_perplexity_under_vllm": safe_exp_mean([-logprob for logprob in all_logprobs]) if all_logprobs else None,
    }


def print_sample_summary(bundle: dict[str, Any]) -> None:
    print("Sample Summary")
    print(json.dumps(bundle["summary"], ensure_ascii=False, indent=2))
    print("")
    for idx, sample in enumerate(bundle["samples"], start=1):
        print(f"Prompt {idx}")
        print(f"  text: {sample['prompt']!r}")
        print(f"  generated_tokens: {len(sample['generated_token_ids'])}")
        print(f"  generated_text: {sample['generated_text']!r}")
        print("")


def print_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print("Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("")
    for idx, result in enumerate(results, start=1):
        print(f"Prompt {idx}")
        print(f"  text: {result['prompt']!r}")
        print(f"  generated_tokens: {result['num_generated_tokens']}")
        print(f"  generated_text: {result['generated_text']!r}")
        print(f"  top1_agreement_rate: {result['top1_agreement_rate']}")
        print(f"  top10_hit_rate: {result['top10_hit_rate']}")
        print(f"  top100_hit_rate: {result['top100_hit_rate']}")
        print(f"  requested_topk_hit_rate: {result['requested_topk_hit_rate']}")
        print(f"  avg_rank_in_vllm: {result['avg_rank_in_vllm']}")
        print(f"  max_rank_in_vllm: {result['max_rank_in_vllm']}")
        print(f"  mean_logprob_in_vllm: {result['mean_logprob_in_vllm']}")
        print(f"  first_non_top1_position: {result['first_non_top1_position']}")
        print("")


def sample_bundle_to_json(
    samples: list[MiniSGLSample],
    tokenizer_model: str,
    *,
    workload_elapsed_seconds: float | None = None,
    sample_concurrency: int | None = None,
    batch_submit: bool | None = None,
) -> dict[str, Any]:
    total_generated_tokens = sum(len(sample.generated_token_ids) for sample in samples)
    ttfts = [float(sample.ttft_seconds) for sample in samples if sample.ttft_seconds is not None]
    latencies = [
        float(sample.latency_seconds)
        for sample in samples
        if sample.latency_seconds is not None
    ]
    tpots = [
        max(0.0, float(sample.latency_seconds) - float(sample.ttft_seconds))
        / (len(sample.generated_token_ids) - 1)
        for sample in samples
        if sample.latency_seconds is not None
        and sample.ttft_seconds is not None
        and len(sample.generated_token_ids) > 1
    ]
    elapsed = (
        float(workload_elapsed_seconds)
        if workload_elapsed_seconds is not None
        else None
    )
    performance = {
        "measurement_mode": (
            "batch_non_streaming" if batch_submit else "streaming_per_request"
        ) if batch_submit is not None else None,
        "sample_concurrency": sample_concurrency,
        "workload_elapsed_seconds": elapsed,
        "generated_throughput_tokens_per_second": (
            total_generated_tokens / elapsed
            if elapsed is not None and elapsed > 0
            else None
        ),
        "mean_ttft_seconds": safe_mean(ttfts),
        "mean_tpot_seconds": safe_mean(tpots),
        "mean_request_latency_seconds": safe_mean(latencies),
        "timed_request_count": len(latencies),
        "timed_tpot_request_count": len(tpots),
    }
    return {
        "tokenizer_model": tokenizer_model,
        "samples": [
            {
                "prompt": sample.prompt,
                "generated_text": sample.generated_text,
                "generated_token_ids": sample.generated_token_ids,
                "timing": {
                    "ttft_seconds": sample.ttft_seconds,
                    "latency_seconds": sample.latency_seconds,
                    "tpot_seconds": (
                        max(0.0, sample.latency_seconds - sample.ttft_seconds)
                        / (len(sample.generated_token_ids) - 1)
                        if sample.latency_seconds is not None
                        and sample.ttft_seconds is not None
                        and len(sample.generated_token_ids) > 1
                        else None
                    ),
                },
            }
            for sample in samples
        ],
        "summary": {
            "num_prompts": len(samples),
            "total_generated_tokens": total_generated_tokens,
        },
        "performance": performance,
    }


def sample_bundle_from_json(path: Path) -> dict[str, Any]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if "samples" not in bundle or not isinstance(bundle["samples"], list):
        raise RuntimeError(f"{path} is not a valid sample bundle.")
    return bundle


def run_sample(args: argparse.Namespace) -> int:
    prompts = load_prompts(args.prompt, args.prompt_file, args.prompt_repeat)
    minisgl_model = get_first_served_model(args.minisgl_url, args.timeout)
    workload_started_at = time.perf_counter()
    samples = sample_prompts_with_minisgl(
        prompts=prompts,
        base_url=args.minisgl_url,
        served_model=minisgl_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout=args.timeout,
        sample_concurrency=args.sample_concurrency,
        batch_submit=args.batch_submit,
        ignore_eos=args.ignore_eos,
    )
    workload_elapsed_seconds = time.perf_counter() - workload_started_at
    bundle = sample_bundle_to_json(
        samples,
        tokenizer_model=args.model,
        workload_elapsed_seconds=workload_elapsed_seconds,
        sample_concurrency=args.sample_concurrency,
        batch_submit=args.batch_submit,
    )
    if args.sample_json is not None:
        args.sample_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print_sample_summary(bundle)
    return 0


def run_vllm_sample(args: argparse.Namespace) -> int:
    prompts = load_prompts(args.prompt, args.prompt_file, args.prompt_repeat)
    vllm_model = get_first_served_model(args.vllm_url, args.timeout)
    samples = sample_prompts_with_vllm(
        prompts=prompts,
        base_url=args.vllm_url,
        served_model=vllm_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout=args.timeout,
        sample_concurrency=args.sample_concurrency,
        ignore_eos=args.ignore_eos,
    )
    bundle = sample_bundle_to_json(samples, tokenizer_model=args.model)
    if args.sample_json is not None:
        args.sample_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print_sample_summary(bundle)
    return 0


def load_reference_identity(path: Path | None) -> dict[str, Any] | None:
    """Read the identity vllm_server.sh --identity-json recorded; warn and return None if unusable."""
    if path is None:
        return None
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"WARNING: no reference identity recorded ({path}: {error})", file=sys.stderr)
        return None
    if not isinstance(identity, dict):
        print(f"WARNING: reference identity in {path} is not a JSON object", file=sys.stderr)
        return None
    return identity


def run_score(args: argparse.Namespace) -> int:
    bundle = sample_bundle_from_json(args.sample_json)
    tokenizer_model = args.model or bundle.get("tokenizer_model")
    if not tokenizer_model:
        raise RuntimeError("Tokenizer model must be supplied via --model or stored in sample-json.")
    tokenizer = load_tokenizer(tokenizer_model)
    vllm_model = get_first_served_model(args.vllm_url, args.timeout)

    raw_samples = list(bundle["samples"])
    results: list[dict[str, Any] | None] = [None] * len(raw_samples)

    def score_one(index: int, raw_sample: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        sample = MiniSGLSample(
            prompt=raw_sample["prompt"],
            generated_text=raw_sample["generated_text"],
            generated_token_ids=[int(token_id) for token_id in raw_sample["generated_token_ids"]],
        )
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=True)
        scored_choice = score_with_vllm(
            base_url=args.vllm_url,
            served_model=vllm_model,
            prompt_token_ids=prompt_ids + sample.generated_token_ids,
            prompt_logprobs=args.prompt_logprobs,
            timeout=args.timeout,
        )
        return index, compare_prompt(tokenizer, sample, scored_choice, args.prompt_logprobs)

    max_workers = max(1, min(int(args.score_concurrency), len(raw_samples)))
    if max_workers == 1:
        for index, raw_sample in enumerate(raw_samples):
            result_index, result = score_one(index, raw_sample)
            results[result_index] = result
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(score_one, index, raw_sample): index
                for index, raw_sample in enumerate(raw_samples)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result_index, result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"vLLM scoring failed for sample index {index}") from exc
                results[result_index] = result

    finalized_results = [result for result in results if result is not None]
    summary = summarize(finalized_results)
    report = {
        "tokenizer_model": tokenizer_model,
        "prompt_logprobs": args.prompt_logprobs,
        "summary": summary,
        "fastafd_performance": bundle.get("performance"),
    }
    reference_identity = load_reference_identity(args.reference_identity_json)
    if reference_identity is not None:
        report["reference_identity"] = reference_identity
    report["results"] = finalized_results
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary, finalized_results)
    return 0


def run_compare(args: argparse.Namespace) -> int:
    prompts = load_prompts(args.prompt, args.prompt_file, args.prompt_repeat)
    minisgl_model = get_first_served_model(args.minisgl_url, args.timeout)
    vllm_model = get_first_served_model(args.vllm_url, args.timeout)
    tokenizer = load_tokenizer(args.model)

    samples = sample_prompts_with_minisgl(
        prompts=prompts,
        base_url=args.minisgl_url,
        served_model=minisgl_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout=args.timeout,
        sample_concurrency=args.sample_concurrency,
        batch_submit=args.batch_submit,
        ignore_eos=args.ignore_eos,
    )

    bundle = sample_bundle_to_json(samples, tokenizer_model=args.model)
    if args.sample_json is not None:
        args.sample_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    results: list[dict[str, Any] | None] = [None] * len(samples)

    def score_one(index: int, sample: MiniSGLSample) -> tuple[int, dict[str, Any]]:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=True)
        scored_choice = score_with_vllm(
            base_url=args.vllm_url,
            served_model=vllm_model,
            prompt_token_ids=prompt_ids + sample.generated_token_ids,
            prompt_logprobs=args.prompt_logprobs,
            timeout=args.timeout,
        )
        return index, compare_prompt(tokenizer, sample, scored_choice, args.prompt_logprobs)

    max_workers = max(1, min(int(args.score_concurrency), len(samples)))
    if max_workers == 1:
        for index, sample in enumerate(samples):
            result_index, result = score_one(index, sample)
            results[result_index] = result
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(score_one, index, sample): index
                for index, sample in enumerate(samples)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result_index, result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"vLLM scoring failed for sample index {index}") from exc
                results[result_index] = result

    finalized_results = [result for result in results if result is not None]
    summary = summarize(finalized_results)
    report = {
        "tokenizer_model": args.model,
        "prompt_logprobs": args.prompt_logprobs,
        "summary": summary,
        "results": finalized_results,
    }
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary, finalized_results)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "sample":
        return run_sample(args)
    if args.command == "vllm-sample":
        return run_vllm_sample(args)
    if args.command == "score":
        return run_score(args)
    if args.command == "compare":
        return run_compare(args)
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
