from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize quarantine reasons for a given pipeline_run_id.")
    ap.add_argument("--db", default="data/photosorter.db")
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--limit-samples", type=int, default=5)
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = _connect(db)
    try:
        cur = conn.cursor()
        pr = cur.execute(
            "SELECT id, kind, root_path, face_run_id FROM pipeline_runs WHERE id=? LIMIT 1",
            (int(args.pipeline_run_id),),
        ).fetchone()
        if not pr:
            raise SystemExit(f"pipeline_run_id not found: {args.pipeline_run_id}")

        face_run_id = pr["face_run_id"]
        if not face_run_id:
            raise SystemExit("face_run_id is not set on this pipeline run yet")

        fr = cur.execute("SELECT id, processed_files, total_files, faces_found, last_error FROM face_runs WHERE id=? LIMIT 1", (int(face_run_id),)).fetchone()

        print("pipeline_run:", dict(pr))
        print("face_run:", dict(fr) if fr else None)

        # Only files seen by this run (faces_run_id == face_run_id)
        rows = cur.execute(
            """
            SELECT
              COALESCE(faces_auto_quarantine, 0) AS qflag,
              lower(trim(COALESCE(faces_quarantine_reason, ''))) AS qreason,
              COALESCE(animals_auto, 0) AS animals,
              lower(trim(COALESCE(faces_manual_label, ''))) AS manual_label,
              COALESCE(faces_count, 0) AS faces_count,
              path
            FROM files
            WHERE faces_run_id = ?
              AND status != 'deleted'
            """,
            (int(face_run_id),),
        ).fetchall()

        total = len(rows)
        print("rows_total:", total)
        if total == 0:
            return 0

        # bucket keys: (qflag, qreason)
        cnt = Counter()
        samples: dict[tuple[int, str], list[str]] = defaultdict(list)
        faces_hist_q = Counter()
        faces_hist_nq = Counter()

        for r in rows:
            qflag = int(r["qflag"] or 0)
            qreason = str(r["qreason"] or "")
            key = (qflag, qreason)
            cnt[key] += 1
            if len(samples[key]) < int(args.limit_samples):
                samples[key].append(str(r["path"] or ""))
            fc = int(r["faces_count"] or 0)
            if qflag:
                faces_hist_q[fc] += 1
            else:
                faces_hist_nq[fc] += 1

        def _pct(n: int) -> str:
            return f"{(100.0 * n / float(total)):.1f}%"

        print("\nTOP quarantine buckets (qflag,qreason):")
        for (qflag, qreason), n in cnt.most_common(20):
            if qflag != 1:
                continue
            print(f"  q=1 reason='{qreason or ''}': {n} ({_pct(n)})")
            for s in samples[(qflag, qreason)]:
                print(f"    - {s}")

        print("\nTOP non-quarantine buckets (qflag,qreason):")
        for (qflag, qreason), n in cnt.most_common(10):
            if qflag != 0:
                continue
            print(f"  q=0 reason='{qreason or ''}': {n} ({_pct(n)})")

        # Simple histograms: just show most common faces_count values
        print("\nfaces_count histogram (quarantine):")
        for fc, n in faces_hist_q.most_common(15):
            print(f"  faces_count={fc}: {n} ({_pct(n)})")

        print("\nfaces_count histogram (non-quarantine):")
        for fc, n in faces_hist_nq.most_common(15):
            print(f"  faces_count={fc}: {n} ({_pct(n)})")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




