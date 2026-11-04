"""CAD mode switches: MINISGL_AFD_MODE and the MINISGL_AFD_CAD_* keys (the only source of the
CAD config). Pure Python, no torch.

  MINISGL_AFD_MODE                 afd (default, also unset / "") | cad
  MINISGL_AFD_CAD_O_DTYPE          bf16 (o comes back in BF16, 12,288 B per token for MiniMax-M2.5)
  MINISGL_AFD_CAD_MOE              megamoe_dual (MegaMoE M2N, each model GPU AG + EG of a node-local union) | deepep
  MINISGL_AFD_CAD_HOP              window (symmetric-memory window, put / wait kernels of kernel/cad_hop.py)
  MINISGL_AFD_CAD_HANG_TIMEOUT_NS  every flag wait's timeout; a wait that times out sets the window's abort word
  MINISGL_AFD_CAD_INIT_TIMEOUT_S   the coordinator's wait for every rank's AfdCadReadyReply (default 900 s)

Unknown values are refused (never silently mapped to a default). With MINISGL_AFD_MODE != cad the other keys must be
unset: a CAD key without CAD mode is a job-script mistake.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

MODE_KEY = "MINISGL_AFD_MODE"
O_DTYPE_KEY = "MINISGL_AFD_CAD_O_DTYPE"
MOE_KEY = "MINISGL_AFD_CAD_MOE"
HOP_KEY = "MINISGL_AFD_CAD_HOP"
TIMEOUT_KEY = "MINISGL_AFD_CAD_HANG_TIMEOUT_NS"
INIT_TIMEOUT_KEY = "MINISGL_AFD_CAD_INIT_TIMEOUT_S"
# every key a CAD job may set; forwarded on both Ray hops (ray_env_keys.py includes this tuple)
CAD_ENV_KEYS = (MODE_KEY, O_DTYPE_KEY, MOE_KEY, HOP_KEY, TIMEOUT_KEY, INIT_TIMEOUT_KEY)

MODES = ("afd", "cad")
O_DTYPES = ("bf16",)
MOES = ("megamoe_dual", "deepep")
HOPS = ("window",)
DEFAULT_TIMEOUT_NS = 300_000_000_000
DEFAULT_INIT_TIMEOUT_S = 900


class CadConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CadConfig:
    enabled: bool = False
    o_dtype: str = "bf16"
    moe: str = "megamoe_dual"
    hop: str = "window"
    timeout_ns: int = DEFAULT_TIMEOUT_NS
    init_timeout_s: int = DEFAULT_INIT_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.o_dtype not in O_DTYPES:
            raise CadConfigError(f"{O_DTYPE_KEY}={self.o_dtype!r}: one of {O_DTYPES}")
        if self.moe not in MOES:
            raise CadConfigError(f"{MOE_KEY}={self.moe!r}: one of {MOES}")
        if self.hop not in HOPS:
            raise CadConfigError(f"{HOP_KEY}={self.hop!r}: one of {HOPS}")
        if not isinstance(self.timeout_ns, int) or self.timeout_ns <= 0:
            raise CadConfigError(f"{TIMEOUT_KEY}={self.timeout_ns!r}: a positive integer (ns)")
        if not isinstance(self.init_timeout_s, int) or self.init_timeout_s <= 0:
            raise CadConfigError(f"{INIT_TIMEOUT_KEY}={self.init_timeout_s!r}: a positive integer (s)")

    @property
    def o_elt_bytes(self) -> int:
        """Bytes per element of the o rows the attention GPU writes back."""
        return 2

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "CadConfig":
        env = os.environ if environ is None else environ
        mode = (env.get(MODE_KEY) or "afd").strip()
        if mode not in MODES:
            raise CadConfigError(f"{MODE_KEY}={mode!r}: one of {MODES}")
        if mode != "cad":
            stray = [k for k in CAD_ENV_KEYS[1:] if env.get(k)]
            if stray:
                raise CadConfigError(f"{stray} set without {MODE_KEY}=cad")
            return cls(enabled=False)
        raw = env.get(TIMEOUT_KEY) or str(DEFAULT_TIMEOUT_NS)
        try:
            timeout = int(raw)
        except ValueError as exc:
            raise CadConfigError(f"{TIMEOUT_KEY}={raw!r}: a positive integer (ns)") from exc
        raw_init = env.get(INIT_TIMEOUT_KEY) or str(DEFAULT_INIT_TIMEOUT_S)
        try:
            init_timeout = int(raw_init)
        except ValueError as exc:
            raise CadConfigError(f"{INIT_TIMEOUT_KEY}={raw_init!r}: a positive integer (s)") from exc
        return cls(enabled=True, o_dtype=env.get(O_DTYPE_KEY) or "bf16", moe=env.get(MOE_KEY) or "megamoe_dual",
                   hop=env.get(HOP_KEY) or "window", timeout_ns=timeout, init_timeout_s=init_timeout)

    def env(self) -> dict[str, str]:
        """The keys a render writes for this config (empty when CAD is off)."""
        if not self.enabled:
            return {}
        return {MODE_KEY: "cad", O_DTYPE_KEY: self.o_dtype, MOE_KEY: self.moe, HOP_KEY: self.hop,
                TIMEOUT_KEY: str(self.timeout_ns), INIT_TIMEOUT_KEY: str(self.init_timeout_s)}


__all__ = ["CAD_ENV_KEYS", "CadConfig", "CadConfigError", "DEFAULT_INIT_TIMEOUT_S", "DEFAULT_TIMEOUT_NS", "HOPS", "MODES",
           "MOES", "O_DTYPES"]
