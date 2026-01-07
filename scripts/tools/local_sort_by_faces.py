from __future__ import annotations

"""
Thin CLI wrapper.

Core implementation lives in `photosorter.pipeline.local_sort`.
This file remains for backward compatibility and for Web UI which runs it via `.venv-face`.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from photosorter.pipeline.local_sort import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())





