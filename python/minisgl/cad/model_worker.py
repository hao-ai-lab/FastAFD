"""CAD model worker: an AFD expert (EG) worker that, under MINISGL_AFD_MODE=cad, also runs the model side of every decode
step (prefill steps stay AFD steps; this worker is their EG).

At the CAD init command (AfdCadInitCmd: buckets, t_max, table rows, num_mb; the same moment on every rank):
  1. the AG stage on this GPU (_load_afd_ag_stage: dense layers, router, embedding, final norm, LM head)
  2. the CAD groups (minisgl.cad.groups, collective) and this GPU's window (CadWindow.allocate, collective)
  3. the dual-role MegaMoE over the model node, on the prefill EG adapter's own FP8 expert tensors (m2n_dual)
  4. the token buffers, the task executor (model_side.CadModelTasks) and the runtime, warmed per bucket
Per CAD decode step (AfdRunCadStepCmd): runtime.run_step on the worker's stream, then the sampled tokens to the host and
the reply. The loop serves EG steps (prefill) and CAD steps in arrival order (synchronous, as afd_normal_loop).

The protocol names (AfdCadInitCmd / AfdRunCadStepCmd / AfdCadStepReply, cad/program.py, cad/runtime.py) are resolved
at run time.
"""

from __future__ import annotations

from minisgl.afd_expert_worker import AfdExpertWorker
from minisgl.afd_protocol import AfdFlushStepCmd, AfdProfilerCmd, AfdRunEGStepCmd, AfdStopCmd
from minisgl.afd_support import log_line

from .config import CadConfig
from .groups import CadRanks


class CadUnsupported(RuntimeError):
    pass


def check_cad_model(cfg: CadConfig, arch: str, adapter_problem_list: list[str]) -> None:
    """The CAD v1 model side: MiniMax-M2.5, dual-role MegaMoE on the prefill adapter's FP8 experts."""
    if not cfg.enabled:
        raise CadUnsupported("CAD init without MINISGL_AFD_MODE=cad")
    if arch != "MiniMaxM2ForCausalLM":
        raise CadUnsupported(f"CAD v1 model side is MiniMax-M2.5 only, got {arch!r}")
    if cfg.moe != "megamoe_dual":
        raise CadUnsupported(f"MINISGL_AFD_CAD_MOE={cfg.moe!r}: only megamoe_dual is built (deepep: a later preset)")
    if adapter_problem_list:
        raise CadUnsupported("the prefill EG adapter cannot lend its experts: " + "; ".join(adapter_problem_list))


class CadModelWorker(AfdExpertWorker):
    """AfdExpertWorker + the CAD model side (see the module docstring)."""

    def _run_hot_rpc_loop(self) -> None:
        self._prepare_hot_rpc_runtime()
        self.cad_loop()

    def cad_loop(self) -> None:
        while True:
            cmd = self._recv_hot_command(blocking=True)
            if cmd is None or isinstance(cmd, AfdFlushStepCmd):
                continue
            if isinstance(cmd, AfdStopCmd):
                return
            if isinstance(cmd, AfdProfilerCmd):
                self.handle_profiler_cmd(cmd)
                continue
            if isinstance(cmd, AfdRunEGStepCmd):
                self._process_last_afd_eg(self._run_afd_eg_step(cmd.plan, sent_ns=cmd.sent_ns))
                continue
            kind = type(cmd).__name__
            if kind == "AfdCadInitCmd":
                self._init_cad(cmd)
                continue
            if kind == "AfdRunCadStepCmd":
                self._run_cad_step(cmd)
                continue
            raise RuntimeError(f"Unsupported CAD model-worker command: {kind}")

    # ------------------------------------------------------------------ init
    def _cad_ranks(self) -> CadRanks:
        topo = self._topology
        return CadRanks(rank=int(self.worker_rank), num_attn=int(topo.total_attn_workers),
                        num_model=int(topo.mlp_world_size))

    def _init_cad(self, cmd) -> None:
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        from minisgl.kernel import megamoe_m2n_mega as mm

        from minisgl.cad import program as cad_program
        from minisgl.cad import runtime as cad_runtime
        from minisgl.afd_protocol import AfdCadReadyReply

        from .groups import cad_groups
        from .layout import CadLayout
        from .m2n_dual import M2NDual, adapter_problems, check_even_rounds, layers_from_adapter, make_dual_buffers
        from .model_side import CadModelTasks
        from .tokens import CadTokenInputs
        from .window import CadWindow

        cfg = CadConfig.from_env()
        rs = self.runtime_state
        mc = rs.config.model_config
        ranks = self._cad_ranks()
        if ranks.role != "model":
            raise CadUnsupported(f"rank {ranks.rank} is not a model rank of {ranks}")
        L, m, t_max = int(mc.num_layers), int(cmd.num_mb), int(cmd.t_max)
        E, H, inter = int(mc.num_experts), int(mc.hidden_size), int(mc.moe_intermediate_size)
        adapter = self._afd_adapters[0] if self._afd_adapters else None
        probs = adapter_problems(adapter, range(L), ranks.local, num_experts=E, hidden=H, intermediate=inter,
                                 size=ranks.num_model)
        check_cad_model(cfg, (getattr(mc, "architectures", None) or [""])[0], probs)
        check_even_rounds(L, m)
        if getattr(self, "_afd_ag_model", None) is None:
            self._load_afd_ag_stage()
        tag = f"[cad-model{ranks.local}] init:"

        def stage(what: str) -> None:   # the coordinator's CAD-init watchdog names each rank's last stage from these lines
            log_line(self.log_path, f"{tag} {what}", flush=True)

        # Collective order, the SAME on every rank up to the window (the attention side: groups, window, warm). torch's
        # symmetric memory keys every rendezvous's store exchange by ONE per-process sequence number (c10d::symmetric_memory
        # StoreExchange seq_id_, torch/include/.../symm_mem/CUDASymmetricMemoryUtils.hpp:65,68,104), whatever the group:
        # a model-group rendezvous before the WORLD window would shift this rank's sequence and the window would wait on
        # keys the attention ranks never write. Model-group rendezvous only after the window.
        stage("groups (dist.new_group over the default group)")
        model_group, window_group = cad_groups(dist, ranks)
        lay = CadLayout.for_model(num_layers=L, num_mb=m, r=ranks.r, t_max=t_max, q_heads=int(mc.num_qo_heads),
                                  kv_heads=int(mc.num_kv_heads), head_dim=int(mc.head_dim))
        stage("window (symm_mem rendezvous + barrier over WORLD)")
        win = CadWindow.allocate(lay, "model", ranks.rank, ranks.peers, window_group, rs.device, cfg.timeout_ns)
        stage("dual MegaMoE buffers (2 symm_mem rendezvous + barrier over the model group)")
        ag_buf, eg_buf, self._cad_m2n_handles = make_dual_buffers(
            mm, symm_mem, dist, group=model_group, local=ranks.local, size=ranks.num_model, max_tokens=ranks.r * t_max,
            num_experts=E, num_topk=int(mc.num_experts_per_tok), hidden=H, intermediate=inter, device=rs.device)
        moe = M2NDual(ag_buf=ag_buf, eg_buf=eg_buf, layers=layers_from_adapter(adapter, range(L), rs.device),
                      local=ranks.local, size=ranks.num_model, num_mb=m, experts_per_rank=E // ranks.num_model,
                      expected_tokens=lambda n: adapter._expected_tokens_for_eg(int(n), dual_lane_decode=False))
        # the EG JIT here, before the warm-up: no rendezvous (compile_only); the attention ranks meanwhile wait in their
        # warm-up's first flag wait, bounded by the CAD hang timeout (300 s; a cold JIT build takes ~10 s)
        stage("EG precompile (compile only, no collective)")
        for b in sorted({int(x) for x in cmd.buckets}):   # every bucket's EG instance (the expected-token heuristic picks it)
            moe.precompile(range(L), ranks.r * b)
        tokens = CadTokenInputs.allocate(r=ranks.r, num_mb=m, t_max=t_max, table_rows=int(cmd.table_rows),
                                         device=rs.device)
        tasks = CadModelTasks(window=win, model=self._afd_ag_model, moe=moe, tokens=tokens, num_layers=L, hidden=H,
                              device=rs.device)
        backend = cad_runtime.TorchCadBackend(device=rs.device)
        runtime = cad_runtime.CadNodeRuntime(tasks, cad_program.program("model", L, m), win, backend=backend,
                                             log=lambda s: log_line(self.log_path, s, flush=True),
                                             label=f"cad-model{ranks.local}")
        stage("warm (paired eager + capture over the CAD world)")
        with torch.cuda.stream(rs.stream):   # the attention ranks warm the same sorted order (paired eager + capture)
            captured = runtime.warm(sorted(int(b) for b in cmd.buckets), graphs=bool(getattr(cmd, "use_graph", True)))
        self._cad = {"cfg": cfg, "ranks": ranks, "window": win, "tasks": tasks, "tokens": tokens, "runtime": runtime,
                     "out_host": torch.zeros(tokens.out.numel(), dtype=torch.int32, pin_memory=True)}
        log_line(self.log_path, f"[cad-model{ranks.local}] ready: layout {lay.describe()}, peers {ranks.peers}, "
                                f"graphs {captured}", flush=True)
        self._send_reply(AfdCadReadyReply(worker_rank=int(self.worker_rank), captured=tuple(captured)))

    # ------------------------------------------------------------------ steps
    def _run_cad_step(self, cmd) -> None:
        import torch

        from minisgl.afd_protocol import AfdCadStepReply

        from .runtime import CadStepAborted
        from .tokens import reply_tokens

        st = getattr(self, "_cad", None)
        if st is None:
            raise RuntimeError("AfdRunCadStepCmd before AfdCadInitCmd")
        tasks, tok, out_host = st["tasks"], st["tokens"], st["out_host"]
        stream = self.runtime_state.stream
        with torch.cuda.stream(stream):
            done, finish = st["runtime"].run_step(cmd)
            n = tok.rows
            out_host[:n].copy_(tok.out[:n], non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(stream)
        sent_ns = int(getattr(cmd, "sent_ns", 0))
        try:
            info = finish()                               # the step's event + its abort word
        except CadStepAborted as exc:                     # as the attention ranks: reply the abort word, keep serving
            copied.synchronize()
            err = int(st["window"].err.item()) or 1
            log_line(self.log_path, f"[cad-model{st['ranks'].local}] step {int(cmd.step_id)} aborted: {exc}", flush=True)
            self._send_reply(AfdCadStepReply(worker_rank=int(self.worker_rank), step_id=int(cmd.step_id), tokens={},
                                             err=err, sent_ns=sent_ns))
            return
        copied.synchronize()                              # the token copy, enqueued after the step
        got = reply_tokens(out_host[:n].tolist(), list(cmd.model_rows), tasks.where)
        self._send_reply(AfdCadStepReply(worker_rank=int(self.worker_rank), step_id=int(cmd.step_id), tokens=got,
                                         err=int(info["err"]), sent_ns=sent_ns))


__all__ = ["CadModelWorker", "CadUnsupported", "check_cad_model"]
