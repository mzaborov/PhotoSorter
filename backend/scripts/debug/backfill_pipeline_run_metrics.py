from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python backend/scripts/debug/backfill_pipeline_run_metrics.py ...
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = _REPO_ROOT / "backend"
for p in (str(_BACKEND_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.db import PipelineStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Manual backfill of pipeline_run_metrics for old runs where live recompute is impossible "
            "(because files.* was overwritten). Use only if you have trusted numbers."
        )
    )
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--face-run-id", type=int, default=None)
    ap.add_argument("--step2-total", type=int, default=None)
    ap.add_argument("--step2-processed", type=int, default=None)
    ap.add_argument("--cats-total", type=int, required=True)
    ap.add_argument("--cats-mism", type=int, required=True)
    ap.add_argument("--faces-total", type=int, required=True)
    ap.add_argument("--faces-mism", type=int, required=True)
    ap.add_argument("--no-faces-total", type=int, required=True)
    ap.add_argument("--no-faces-mism", type=int, required=True)
    args = ap.parse_args()

    ps = PipelineStore()
    try:
        rid = int(args.pipeline_run_id)
        pr = ps.get_run_by_id(run_id=rid)
        if not pr:
            raise SystemExit(f"pipeline_run_id not found: {rid}")

        frid = int(args.face_run_id) if args.face_run_id is not None else (int(pr.get("face_run_id") or 0) or None)

        ps.upsert_metrics(
            pipeline_run_id=rid,
            metrics={
                # We intentionally write computed_at now. This is a manual backfill.
                "face_run_id": frid,
                "step2_total": int(args.step2_total) if args.step2_total is not None else None,
                "step2_processed": int(args.step2_processed) if args.step2_processed is not None else None,
                "cats_total": int(args.cats_total),
                "cats_mism": int(args.cats_mism),
                "faces_total": int(args.faces_total),
                "faces_mism": int(args.faces_mism),
                "no_faces_total": int(args.no_faces_total),
                "no_faces_mism": int(args.no_faces_mism),
            },
        )
    finally:
        ps.close()

    print(f"OK: backfilled pipeline_run_metrics for pipeline_run_id={int(args.pipeline_run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

