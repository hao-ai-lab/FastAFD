"""minisgl: imports env_override first so its defaults precede every other import (as vLLM does)."""

import minisgl.env_override  # noqa: F401
