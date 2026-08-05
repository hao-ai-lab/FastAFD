from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


def _prepare_user_msg(
    *,
    msg: UserMsg,
    max_seq_len: int,
    warn,
) -> bool:
    input_len = len(msg.input_ids)
    max_output_len = max_seq_len - input_len
    if max_output_len <= 0:
        warn(
            f"Input sequence length {input_len} exceeds {max_seq_len}, "
            f"request {msg.uid} is dropped."
        )
        return False
    if msg.sampling_params.max_tokens > max_output_len:
        msg.sampling_params.max_tokens = max_output_len
        warn(f"Adjust max_tokens to {max_output_len} for request {msg.uid}.")
    return True


def _abort_req(
    *,
    uid: int,
    prefill_manager: PrefillManager,
    decode_manager: DecodeManager,
    free_req_resources,
) -> None:
    req_to_free = prefill_manager.abort_req(uid)
    req_to_free = req_to_free or decode_manager.abort_req(uid)
    if req_to_free is not None:
        free_req_resources(req_to_free)


def _schedule_next_batch(
    *,
    prefill_manager: PrefillManager,
    decode_manager: DecodeManager,
    prefill_budget: int,
) -> Batch | None:
    return (
        prefill_manager.schedule_next_batch(prefill_budget)
        or decode_manager.schedule_next_batch()
    )


def _complete_batch_results(
    *,
    batch: Batch,
    next_tokens_cpu: torch.Tensor,
    eos_token_id: int | list[int] | tuple[int, ...] | set[int] | None,
    finished_reqs: Set[Req],
    decode_manager: DecodeManager,
    cache_manager: CacheManager,
    free_req_resources,
) -> tuple[list[DetokenizeMsg], set[Req]]:
    reply: list[DetokenizeMsg] = []
    new_finished_reqs: set[Req] = set()
    with cache_manager.lazy_free_region():
        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue
            next_token = next_tokens_cpu[i]
            req.append_host(next_token.unsqueeze(0))
            next_token_id = int(next_token.item())
            finished = len(req.input_ids) >= req.max_device_len
            if not req.sampling_params.ignore_eos:
                if eos_token_id is None:
                    pass
                elif isinstance(eos_token_id, (list, tuple, set)):
                    finished |= next_token_id in {int(v) for v in eos_token_id}
                else:
                    finished |= next_token_id == int(eos_token_id)
            reply.append(
                DetokenizeMsg(uid=req.uid, next_token=next_token_id, finished=finished)
            )

            # NOTE: overlap scheduling may make the request freed twice, skip second free
            if finished and req not in finished_reqs:
                decode_manager.remove_req(req)
                free_req_resources(req)
                new_finished_reqs.add(req)
            elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                cache_manager.cache_req(req, finished=False)

    return reply, new_finished_reqs


def _materialize_batch_metadata(
    *,
    batch: Batch,
    device: torch.device,
    page_table: torch.Tensor,
    token_pool: torch.Tensor | None = None,
) -> tuple[Indice2D, Indice2D]:
    positions_host, positions_device = _make_positions(batch, device)
    batch.positions = positions_device
    batch.positions_host = positions_host
    input_mapping = _make_input_tuple(batch, device)
    batch.state_indices = input_mapping[0]
    write_mapping = _make_write_tuple(batch, device)
    batch.out_loc = page_table[input_mapping]
    if token_pool is not None:
        batch.input_ids = token_pool[input_mapping]
    return input_mapping, write_mapping


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        if config.model_config.is_hybrid_attention and config.cache_type != "naive":
            raise RuntimeError(
                "Hybrid GDN models require cache_type='naive' until GDN prefix "
                "state snapshots are implemented"
            )
        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        logger.info("Scheduler init stage: tokenizer:start")
        eos_token_id = -1
        self.tokenizer = None
        if config.tp_info.is_primary():
            self.tokenizer = load_tokenizer(config.model_path)
            eos_token_id = int(self.tokenizer.eos_token_id)
        eos_token_tensor = torch.tensor([eos_token_id], dtype=torch.int64)
        self.engine.tp_cpu_group.broadcast(eos_token_tensor, root=0).wait()
        self.eos_token_id = int(eos_token_tensor.item())
        logger.info("Scheduler init stage: tokenizer:done eos_token_id=%d", self.eos_token_id)
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config
        self._ray_nsys_enabled = bool(getattr(config, "ray_nsys", False))
        self._nsys_warmup_complete = not self._ray_nsys_enabled
        self._nsys_runtime_started = False

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)
        logger.info("Scheduler init complete")

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        if self._nsys_runtime_started:
            torch.cuda.synchronize(self.device)
            torch.cuda.profiler.stop()
            logger.info_rank0("Stopped CUDA profiler after Ray worker runtime capture")
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _maybe_start_nsys_runtime_capture(self) -> None:
        if not self._ray_nsys_enabled:
            return
        if not self._nsys_warmup_complete:
            return
        if self._nsys_runtime_started:
            return
        logger.info_rank0("Starting CUDA profiler for Ray worker runtime capture")
        torch.cuda.profiler.start()
        self._nsys_runtime_started = True

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply, new_finished_reqs = _complete_batch_results(
            batch=batch,
            next_tokens_cpu=next_tokens_cpu.to(dtype=torch.int32),
            eos_token_id=self.eos_token_id,
            finished_reqs=self.finished_reqs,
            decode_manager=self.decode_manager,
            cache_manager=self.cache_manager,
            free_req_resources=self._free_req_resources,
        )
        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            if not _prepare_user_msg(
                msg=msg,
                max_seq_len=self.engine.max_seq_len,
                warn=logger.warning_rank0,
            ):
                return
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            _abort_req(
                uid=msg.uid,
                prefill_manager=self.prefill_manager,
                decode_manager=self.decode_manager,
                free_req_resources=self._free_req_resources,
            )
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        input_mapping, write_mapping = _materialize_batch_metadata(
            batch=batch,
            device=self.device,
            page_table=self.engine.page_table,
        )
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = _schedule_next_batch(
            prefill_manager=self.prefill_manager,
            decode_manager=self.decode_manager,
            prefill_budget=self.prefill_budget,
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        self._maybe_start_nsys_runtime_capture()
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        if self._ray_nsys_enabled and not self._nsys_warmup_complete:
            self._nsys_warmup_complete = True
            logger.info_rank0(
                "Completed Ray worker Nsight warmup batch; runtime capture will start on the next batch"
            )
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host, indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
