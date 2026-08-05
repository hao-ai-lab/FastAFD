from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


def _hf_config_uses_modelopt_nvfp4(config) -> bool:
    def value(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    for candidate in (config, value(config, "text_config", None)):
        if candidate is None:
            continue
        quant = value(candidate, "quantization_config", None)
        if quant is None:
            continue
        if not isinstance(quant, dict):
            quant = getattr(quant, "to_dict", lambda: {})()
        if quant.get("quant_method") == "modelopt" and quant.get("quant_algo") == "NVFP4":
            return True
    return False


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    mode: str = "serve"
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    launch_backend: str = "mp"
    ray_address: str = "auto"
    ray_log_dir: str = ""
    ray_nsys: bool = False
    ray_nsys_output_prefix: str = ""
    ray_node_scoped_cache: bool = False
    control_host: str = field(default_factory=lambda: _detect_local_ip())
    afd_attn_dp_size: int = 1
    afd_mlp_dp_size: int = 1
    afd_attn_tp_size: int = 4
    afd_mlp_tp_size: int = 4
    afd_mlp_ep_size: int = 1
    afd_moe_a2a_backend: str = "none"
    afd_moe_runner_backend: str = "auto"
    afd_prompts: tuple[str, ...] = field(default_factory=tuple)
    afd_prompt_file: str = ""
    afd_prompt_repeat: int = 1
    afd_batch_size: int = 8
    afd_max_new_tokens: int = 1
    afd_decode_graph_bs: tuple[int, ...] = field(default_factory=tuple)
    afd_enable_attention_decode_graph: bool = True
    afd_enable_model_decode_graph: bool = True
    afd_max_batched_tokens: int = 8192
    afd_max_running_req: int = 0
    afd_sample_json: str = ""
    afd_report_dir: str = ""
    afd_device_comm_num_sms: int = 1
    afd_disable_overlap: bool = False
    afd_async_pending_steps: int = 3
    afd_force_multi_mb_graph_overlap: bool = False
    afd_num_mb: int = 1

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def use_tcp_control(self) -> bool:
        return self.launch_backend == "ray"

    @property
    def zmq_frontend_addr(self) -> str:
        if self.use_tcp_control:
            return f"tcp://{self.control_host}:{self.server_port + 5}"
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        if self.use_tcp_control:
            result = f"tcp://{self.control_host}:{self.server_port + 6}"
            assert result != self.zmq_detokenizer_addr
            return result
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        host = self.control_host if self.use_tcp_control else "127.0.0.1"
        return f"tcp://{host}:{self.server_port + 1}"

    @property
    def zmq_backend_addr(self) -> str:
        if self.use_tcp_control:
            return f"tcp://{self.control_host}:{self.server_port + 2}"
        return super().zmq_backend_addr

    @property
    def zmq_detokenizer_addr(self) -> str:
        if self.use_tcp_control:
            return f"tcp://{self.control_host}:{self.server_port + 3}"
        return super().zmq_detokenizer_addr

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        if self.use_tcp_control:
            return f"tcp://{self.control_host}:{self.server_port + 4}"
        return super().zmq_scheduler_broadcast_addr


def _detect_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        pass

    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _mutually_divisible(lhs: int, rhs: int) -> bool:
    small, large = sorted((int(lhs), int(rhs)))
    return small >= 1 and large % small == 0


def _validate_afd_parallel_args(
    parser: argparse.ArgumentParser,
    kwargs: dict[str, object],
) -> None:
    attn_dp = int(kwargs["afd_attn_dp_size"])
    mlp_dp = int(kwargs["afd_mlp_dp_size"])
    attn_tp = int(kwargs["afd_attn_tp_size"])
    mlp_tp = int(kwargs["afd_mlp_tp_size"])
    ep_size = int(kwargs["afd_mlp_ep_size"])
    for name, value in (
        ("--afd-attn-dp-size", attn_dp),
        ("--afd-mlp-dp-size", mlp_dp),
        ("--afd-attn-tp-size", attn_tp),
        ("--afd-mlp-tp-size", mlp_tp),
        ("--afd-mlp-ep-size", ep_size),
    ):
        if value < 1:
            parser.error(f"{name} must be >= 1, got {value}")

    if not _mutually_divisible(attn_dp, mlp_dp):
        parser.error(
            "--afd-attn-dp-size and --afd-mlp-dp-size must be mutually "
            f"divisible, got attn_dp={attn_dp} mlp_dp={mlp_dp}"
        )
    if not _mutually_divisible(attn_tp, mlp_tp):
        parser.error(
            "--afd-attn-tp-size and --afd-mlp-tp-size must be mutually "
            f"divisible, got attn_tp={attn_tp} mlp_tp={mlp_tp}"
        )

    mlp_world = mlp_dp * mlp_tp
    if ep_size > mlp_world or mlp_world % ep_size != 0:
        parser.error(
            "--afd-mlp-ep-size must be a divisor of "
            "--afd-mlp-dp-size * --afd-mlp-tp-size, got "
            f"ep_size={ep_size} mlp_world={mlp_world} "
            f"(mlp_dp={mlp_dp} mlp_tp={mlp_tp})"
        )

    if ep_size not in {1, mlp_tp, mlp_world}:
        parser.error(
            "This AFD runtime currently supports --afd-mlp-ep-size in "
            "{1, --afd-mlp-tp-size, --afd-mlp-dp-size*--afd-mlp-tp-size}: "
            "1 disables EP, mlp_tp_size is per-MLP-DP EP, and mlp_world is "
            f"full-world EP. Got ep_size={ep_size} mlp_world={mlp_world}."
        )


def _afd_multi_mb_graph_requires_serial_steps(kwargs: dict) -> bool:
    """DeepEP graph instances are not safe with multiple steps in flight.

    This only serializes the coordinator/worker command lead. The per-step
    multi-MB AG/EG paper pipeline still overlaps its compute and communication
    lanes.
    """
    graph_requested = bool(kwargs.get("afd_decode_graph_bs")) or int(
        kwargs.get("cuda_graph_max_bs") or 0
    ) > 0
    return (
        kwargs.get("mode") == "afd-serve"
        and int(kwargs.get("afd_num_mb") or 1) > 1
        and bool(kwargs.get("afd_enable_attention_decode_graph"))
        and bool(kwargs.get("afd_enable_model_decode_graph"))
        and graph_requested
    )


def _afd_multi_mb_graph_auto_serializes(kwargs: dict) -> bool:
    return (
        _afd_multi_mb_graph_requires_serial_steps(kwargs)
        and not bool(kwargs.get("afd_disable_overlap"))
        and not bool(kwargs.get("afd_force_multi_mb_graph_overlap"))
    )


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--mode",
        choices=["serve", "afd-serve"],
        default=ServerArgs.mode,
        help=(
            "Launch mode. 'serve' starts the normal API server; "
            "'afd-serve' starts the online API server backed by the AFD runtime."
        ),
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    parser.add_argument(
        "--distributed-timeout",
        type=float,
        default=ServerArgs.distributed_timeout,
        help="Timeout in seconds for distributed collectives and rendezvous.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--launch-backend",
        choices=["mp", "ray"],
        default=ServerArgs.launch_backend,
        help="Backend process launcher. 'mp' uses local multiprocessing; 'ray' uses a Ray supervisor actor.",
    )

    parser.add_argument(
        "--ray-address",
        default=ServerArgs.ray_address,
        help="Ray cluster address used when --launch-backend=ray. Default: auto",
    )

    parser.add_argument(
        "--ray-log-dir",
        default=ServerArgs.ray_log_dir,
        help="Optional directory for per-rank Ray scheduler logs.",
    )

    parser.add_argument(
        "--ray-nsys",
        action="store_true",
        default=ServerArgs.ray_nsys,
        help="Enable Nsight Systems profiling for each Ray scheduler actor.",
    )

    parser.add_argument(
        "--ray-nsys-output-prefix",
        default=ServerArgs.ray_nsys_output_prefix,
        help=(
            "Optional Nsight Systems output prefix for Ray scheduler actors. "
            "Defaults to <ray-log-dir>/nsys/minisgl when --ray-log-dir is set, "
            "otherwise minisgl-ray."
        ),
    )

    parser.add_argument(
        "--control-host",
        default=_detect_local_ip(),
        help="Driver node IP used for Ray backend control sockets and distributed rendezvous.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--afd-attn-dp-size",
        type=int,
        default=ServerArgs.afd_attn_dp_size,
        help=(
            "AFD runtime attention-side DP size. Must be mutually divisible "
            "with --afd-mlp-dp-size."
        ),
    )

    parser.add_argument(
        "--afd-mlp-dp-size",
        type=int,
        default=ServerArgs.afd_mlp_dp_size,
        help=(
            "AFD runtime model/mlp-side DP size. This is also the dense QKVO "
            "DP size and must be mutually divisible with --afd-attn-dp-size."
        ),
    )

    # AFD parallel topology contract:
    #
    #   Attention side:
    #     attn_world = afd_attn_dp_size * afd_attn_tp_size
    #
    #   MLP/model side:
    #     mlp_world = afd_mlp_dp_size * afd_mlp_tp_size
    #
    #   Dense QKVO parallelism is exactly the MLP factorization:
    #     qkvo_dp_size = afd_mlp_dp_size
    #     qkvo_tp_size = afd_mlp_tp_size
    #
    #   Supported MoE EP modes:
    #     afd_mlp_ep_size = 1
    #       No EP.
    #     afd_mlp_ep_size = afd_mlp_tp_size
    #       Per-MLP-DP EP: each MLP DP replica owns a full EP shard group across
    #       its local TP ranks; experts are replicated across MLP DP replicas.
    #     afd_mlp_ep_size = afd_mlp_dp_size * afd_mlp_tp_size
    #       Full-world EP: one expert-parallel group spans the whole MLP world.
    #
    # Target-valid combinations:
    #   - attn_dp and mlp_dp may be <, ==, or > each other, but must be
    #     mutually divisible so request ownership can project cleanly.
    #   - attn_tp and mlp_tp may be <, ==, or > each other, but must be
    #     mutually divisible so Q/K/V/O head shards can fan in/out cleanly.
    #   - afd_mlp_ep_size is restricted to the three supported modes above;
    #     num_experts must also be divisible by afd_mlp_ep_size at model load
    #     time when EP is enabled.
    #
    # Current runtime subset:
    #   - Dense QKVO TP communication supports both mutually-divisible
    #     directions, including attn_tp > mlp_tp head fan-out.
    #   - Partial EP/MoE-TP splits such as mlp_tp_size < ep_size < mlp_world
    #     are intentionally unsupported.
    #   - Ray placement puts each DP's TP group on one node.  If used nodes
    #     have unequal real AFD PE counts, the coordinator may append inert
    #     dummy PEs so each used node has an equal PE count.
    parser.add_argument(
        "--afd-attn-tp-size",
        type=int,
        default=ServerArgs.afd_attn_tp_size,
        help=(
            "AFD runtime attention-side TP size. Target topology requires mutual "
            "divisibility with --afd-mlp-tp-size."
        ),
    )

    parser.add_argument(
        "--afd-mlp-tp-size",
        type=int,
        default=ServerArgs.afd_mlp_tp_size,
        help=(
            "AFD runtime model/mlp-side TP size. This is also the dense QKVO "
            "TP size."
        ),
    )

    parser.add_argument(
        "--afd-mlp-ep-size",
        type=int,
        default=ServerArgs.afd_mlp_ep_size,
        help=(
            "AFD runtime model/mlp-side EP size. Supported values are 1 "
            "(disable EP), mlp_tp_size (per-MLP-DP EP), or "
            "mlp_dp_size*mlp_tp_size (full-world EP)."
        ),
    )

    parser.add_argument(
        "--afd-moe-a2a-backend",
        choices=["none", "deepep"],
        default=ServerArgs.afd_moe_a2a_backend,
        help=(
            "MoE token all-to-all dispatch backend. 'none' uses the standard "
            "dispatcher without DeepEP. 'deepep' uses DeepEP V2 ElasticBuffer "
            "for full-world AFD EP eager dispatch/combine."
        ),
    )

    parser.add_argument(
        "--afd-moe-runner-backend",
        choices=[
            "auto",
            "triton",
            "triton_fp8",
            "triton_kernel",
            "deep_gemm",
            "flashinfer_nvfp4",
        ],
        default=ServerArgs.afd_moe_runner_backend,
        help=(
            "MoE compute runner backend. 'triton_fp8' uses the FP8-native "
            "Triton fallback; 'deep_gemm' uses minisgl's vendored grouped "
            "DeepGEMM kernels for DeepEP V2 expanded layout; "
            "'flashinfer_nvfp4' uses native ModelOpt NVFP4 routed experts with "
            "DeepEP non-expanded token/top-k transport."
        ),
    )

    parser.add_argument(
        "--afd-prompt",
        action="append",
        default=[],
        dest="afd_prompts",
        help="AFD runtime prompt text. Can be provided multiple times.",
    )

    parser.add_argument(
        "--afd-prompt-file",
        default=ServerArgs.afd_prompt_file,
        help="AFD runtime UTF-8 prompt file with one prompt per line.",
    )

    parser.add_argument(
        "--afd-prompt-repeat",
        type=int,
        default=ServerArgs.afd_prompt_repeat,
        help="AFD runtime prompt repeat factor.",
    )

    parser.add_argument(
        "--afd-batch-size",
        type=int,
        default=ServerArgs.afd_batch_size,
        help="AFD runtime prefill batch size.",
    )

    parser.add_argument(
        "--afd-max-new-tokens",
        type=int,
        default=ServerArgs.afd_max_new_tokens,
        help="AFD runtime max new tokens per request.",
    )

    parser.add_argument(
        "--afd-decode-graph-bs",
        default="",
        help=(
            "Optional comma-separated static decode graph buckets for AFD, "
            "for example '1,2,4,8,16'. If omitted, AFD derives the list from --cuda-graph-max-bs "
            "using the vLLM/sglang-aligned pattern: [1,2,4] + range(8, min(max,256)+1, 8) + "
            "range(272, max+1, 16)."
        ),
    )

    parser.add_argument(
        "--afd-disable-attention-decode-graph",
        action="store_false",
        dest="afd_enable_attention_decode_graph",
        default=ServerArgs.afd_enable_attention_decode_graph,
        help="Disable AFD attention-side decode graph while leaving decode graph buckets/config intact.",
    )

    parser.add_argument(
        "--afd-disable-model-decode-graph",
        action="store_false",
        dest="afd_enable_model_decode_graph",
        default=ServerArgs.afd_enable_model_decode_graph,
        help="Disable AFD model-side decode graph while leaving decode graph buckets/config intact.",
    )

    parser.add_argument(
        "--afd-max-batched-tokens",
        type=int,
        default=ServerArgs.afd_max_batched_tokens,
        help=(
            "Static total-token budget for a single eager AFD prefill batch. "
            "This also drives communication buffer sizing for arbitrary requests."
        ),
    )

    parser.add_argument(
        "--afd-max-running-requests",
        type=int,
        dest="afd_max_running_req",
        default=ServerArgs.afd_max_running_req,
        help=(
            "Static maximum number of live AFD requests. 0 means auto: "
            "max(afd-batch-size, max(afd-decode-graph-bs))."
        ),
    )

    parser.add_argument(
        "--afd-sample-json",
        default=ServerArgs.afd_sample_json,
        help="Optional path to write the AFD runtime sample bundle.",
    )

    parser.add_argument(
        "--afd-report-dir",
        default=ServerArgs.afd_report_dir,
        help="Optional explicit report directory for the AFD runtime.",
    )

    parser.add_argument(
        "--afd-device-comm-num-sms",
        type=int,
        default=ServerArgs.afd_device_comm_num_sms,
        help="AFD eager device-comm segmented sender grid size. 1 uses the current stable single-segment device wrapper.",
    )

    parser.add_argument(
        "--afd-disable-overlap",
        action="store_true",
        default=ServerArgs.afd_disable_overlap,
        help="Disable AFD overlap scheduling on both attention and model workers (overlap is on by default).",
    )

    parser.add_argument(
        "--afd-async-pending-steps",
        type=int,
        default=ServerArgs.afd_async_pending_steps,
        help=(
            "Maximum number of outstanding global AFD steps in overlap mode. "
            "The workers launch this window before retiring the oldest reply."
        ),
    )

    parser.add_argument(
        "--afd-force-multi-mb-graph-overlap",
        action="store_true",
        default=ServerArgs.afd_force_multi_mb_graph_overlap,
        help=(
            "Diagnostic opt-in: allow cross-step overlap for multi-MB decode "
            "graphs despite the collective re-entry safety guard."
        ),
    )

    parser.add_argument(
        "--afd-num-mb",
        type=int,
        default=ServerArgs.afd_num_mb,
        help=(
            "Number of micro-batches per AFD decode step for ping-pong pipelining. "
            "1 disables microbatching (default). Decode graph captures the full split step when enabled."
        ),
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    hf_config = None
    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        hf_config = cached_load_hf_config(kwargs["model_path"])
        dtype_str = hf_config.dtype

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]
    kwargs["afd_prompts"] = tuple(kwargs["afd_prompts"])
    kwargs["afd_decode_graph_bs"] = tuple(
        int(value)
        for value in str(kwargs["afd_decode_graph_bs"]).split(",")
        if str(value).strip()
    )
    if kwargs["mode"] == "afd-serve":
        if kwargs["cache_type"] != "naive":
            parser.error(
                "AFD serve centralized scheduler currently supports only --cache-type naive"
            )
        if kwargs["afd_moe_runner_backend"] == "auto":
            if hf_config is None:
                from minisgl.utils import cached_load_hf_config

                hf_config = cached_load_hf_config(kwargs["model_path"])
            # ModelOpt NVFP4 checkpoints need FlashInfer's native W4A4 MoE
            # runner and DeepEP's non-expanded token/top-k transport. Existing
            # BF16/FP8 AFD models retain the expanded DeepGEMM contract.
            kwargs["afd_moe_runner_backend"] = (
                "flashinfer_nvfp4"
                if _hf_config_uses_modelopt_nvfp4(hf_config)
                else "deep_gemm"
            )
    if kwargs["cuda_graph_max_bs"] is None and not kwargs["afd_decode_graph_bs"]:
        cap = (
            int(kwargs["afd_max_running_req"])
            if int(kwargs["afd_max_running_req"]) > 0
            else int(kwargs["afd_batch_size"])
        )
        if cap >= 1:
            kwargs["cuda_graph_max_bs"] = min(cap, 512)
    if int(kwargs["afd_async_pending_steps"]) < 2:
        parser.error("--afd-async-pending-steps must be >= 2")
    if bool(kwargs["afd_force_multi_mb_graph_overlap"]) and bool(
        kwargs["afd_disable_overlap"]
    ):
        parser.error(
            "--afd-force-multi-mb-graph-overlap conflicts with --afd-disable-overlap"
        )
    auto_serialized_graph_steps = _afd_multi_mb_graph_auto_serializes(kwargs)
    if auto_serialized_graph_steps:
        kwargs["afd_disable_overlap"] = True
    _validate_afd_parallel_args(parser, kwargs)

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    if auto_serialized_graph_steps:
        logger.warning(
            "AFD multi-MB decode graphs serialize cross-step command launch to "
            "avoid re-entering captured DeepEP collectives; within-step "
            "microbatch compute/communication overlap remains enabled."
        )
    elif (
        bool(kwargs["afd_force_multi_mb_graph_overlap"])
        and _afd_multi_mb_graph_requires_serial_steps(kwargs)
    ):
        logger.warning(
            "AFD multi-MB decode graph cross-step overlap is FORCE ENABLED for "
            "diagnostic profiling; async_pending_steps=%d",
            int(kwargs["afd_async_pending_steps"]),
        )
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
