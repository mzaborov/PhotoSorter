from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    ap = argparse.ArgumentParser(description="List recent pipeline runs from SQLite (helps pick pipeline_run_id).")
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--limit", type=int, default=15, help="How many rows to print")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = _connect(db)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT
              id,
              kind,
              root_path,
              status,
              apply,
              face_run_id,
              dedup_run_id,
              started_at,
              updated_at,
              finished_at
            FROM pipeline_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(args.limit),),
        ).fetchall()

        if not rows:
            print("No pipeline_runs rows.")
            return 0

        # Simple aligned table
        headers = ["id", "kind", "status", "apply", "face_run_id", "root_path", "started_at", "finished_at"]
        vals = []
        for r in rows:
            vals.append(
                {
                    "id": str(r["id"]),
                    "kind": str(r["kind"] or ""),
                    "status": str(r["status"] or ""),
                    "apply": str(int(r["apply"] or 0)),
                    "face_run_id": str(r["face_run_id"] or ""),
                    "root_path": str(r["root_path"] or ""),
                    "started_at": str(r["started_at"] or ""),
                    "finished_at": str(r["finished_at"] or ""),
                }
            )

        widths = {h: max(len(h), max(len(v[h]) for v in vals)) for h in headers}
        print(" | ".join(h.ljust(widths[h]) for h in headers))
        print("-+-".join("-" * widths[h] for h in headers))
        for v in vals:
            print(" | ".join(v[h].ljust(widths[h]) for h in headers))

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

