import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoConfig, PretrainedConfig

from minisgl.hf_support import DisabledTqdm, load_tokenizer, local_files_only, resolve_local_model_dir


@functools.cache
def _load_hf_config_cached(resolved_model_path: str, local_files_only: bool) -> Any:
    try:
        return AutoConfig.from_pretrained(
            resolved_model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except ValueError:
        # Transformers 4.x predates Qwen3.5, while vLLM ships a compatible
        # config implementation. mini-sgl only needs the serialized fields, so
        # retain them in a generic PretrainedConfig instead of requiring a
        # process-wide Transformers upgrade.
        if os.path.isdir(resolved_model_path):
            config_path = os.path.join(resolved_model_path, "config.json")
            if not os.path.isfile(config_path):
                raise
        else:
            config_path = hf_hub_download(
                repo_id=resolved_model_path,
                filename="config.json",
                local_files_only=local_files_only,
            )
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        if config_dict.get("model_type") not in {"qwen3_5", "qwen3_5_moe"}:
            raise
        return PretrainedConfig(**config_dict)


def _load_hf_config(model_path: str) -> Any:
    local_files = local_files_only()
    resolved_model_path = resolve_local_model_dir(model_path, local_only=local_files)
    return _load_hf_config_cached(resolved_model_path, local_files)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
