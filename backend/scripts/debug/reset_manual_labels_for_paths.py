from __future__ import annotations

r"""
Reset mistaken manual labels in SQLite for specific file paths.

Example:
  python scripts/debug/reset_manual_labels_for_paths.py --path "local:C:\tmp\Photo\IMG-20240625-WA0001.jpg"

This script is intentionally in scripts/debug (операционная утилита).
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable (so `from common.db import ...` works even when running from scripts/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.db import DedupStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-run-id", type=int, default=None, help="If set, reset run-scoped labels for that run")
    ap.add_argument("--path", action="append", default=[], help="File path in DB (local:... or disk:...)")
    args = ap.parse_args()

    paths = [str(p).strip() for p in (args.path or []) if str(p).strip()]
    if not paths:
        raise SystemExit("Provide at least one --path")

    ds = DedupStore()
    try:
        updated = 0
        for p in paths:
            if args.pipeline_run_id is not None:
                ds.delete_run_manual_labels(pipeline_run_id=int(args.pipeline_run_id), path=p)
            else:
                # legacy fallback
                ds.set_faces_manual_label(path=p, label=None)
                ds.set_people_no_face_manual(path=p, is_people_no_face=False, person=None)
            updated += 1
    finally:
        ds.close()

    print(f"OK: reset manual labels for {updated} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


