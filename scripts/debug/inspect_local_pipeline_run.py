from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect latest local_sort pipeline run + related face_run.")
    ap.add_argument("--db", default="data/photosorter.db", help="Path to SQLite DB (default: data/photosorter.db)")
    ap.add_argument("--run-id", type=int, default=None, help="Pipeline run id to inspect (default: latest local_sort)")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = _connect(db)
    try:
        cur = conn.cursor()

        if args.run_id is None:
            pr = cur.execute(
                "SELECT * FROM pipeline_runs WHERE kind='local_sort' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            pr = cur.execute(
                "SELECT * FROM pipeline_runs WHERE kind='local_sort' AND id=? LIMIT 1",
                (int(args.run_id),),
            ).fetchone()

        if not pr:
            print("pipeline_run: <none>")
            return 0

        prd = dict(pr)
        keys = [
            "id",
            "kind",
            "status",
            "root_path",
            "apply",
            "skip_dedup",
            "step_num",
            "step_total",
            "step_title",
            "dedup_run_id",
            "face_run_id",
            "started_at",
            "updated_at",
            "finished_at",
            "last_error",
        ]
        print("pipeline_run:")
        for k in keys:
            if k in prd:
                print(f"  {k}: {prd.get(k)}")

        face_run_id = prd.get("face_run_id")
        if face_run_id:
            fr = cur.execute("SELECT * FROM face_runs WHERE id=? LIMIT 1", (int(face_run_id),)).fetchone()
            print("face_run:")
            if not fr:
                print(f"  <missing id={face_run_id}>")
            else:
                frd = dict(fr)
                fkeys = [
                    "id",
                    "status",
                    "root_path",
                    "total_files",
                    "processed_files",
                    "faces_found",
                    "errors_count",
                    "last_path",
                    "last_error",
                    "started_at",
                    "updated_at",
                    "finished_at",
                ]
                for k in fkeys:
                    if k in frd:
                        print(f"  {k}: {frd.get(k)}")
        else:
            print("face_run: <none>")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




