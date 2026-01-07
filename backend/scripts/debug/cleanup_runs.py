from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Plan:
    keep_pipeline_ids: set[int]
    delete_pipeline_ids: set[int]
    keep_face_run_ids: set[int]
    delete_face_run_ids: set[int]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row[0] or 0) if row else 0


def _plan(conn: sqlite3.Connection, *, keep_per_kind: int) -> Plan:
    cur = conn.cursor()
    kinds = [r[0] for r in cur.execute("SELECT DISTINCT kind FROM pipeline_runs").fetchall()]
    keep_pipeline_ids: set[int] = set()
    for kind in kinds:
        rows = cur.execute(
            "SELECT id FROM pipeline_runs WHERE kind=? ORDER BY id DESC LIMIT ?",
            (str(kind), int(keep_per_kind)),
        ).fetchall()
        keep_pipeline_ids.update(int(r[0]) for r in rows)

    all_pipeline_ids = {int(r[0]) for r in cur.execute("SELECT id FROM pipeline_runs").fetchall()}
    delete_pipeline_ids = all_pipeline_ids - keep_pipeline_ids

    # Face runs: keep anything referenced by kept pipeline runs + keep last N local face runs as safety.
    keep_face_run_ids: set[int] = set()
    for r in cur.execute(
        "SELECT DISTINCT face_run_id FROM pipeline_runs WHERE face_run_id IS NOT NULL AND id IN (%s)"
        % ",".join("?" for _ in keep_pipeline_ids),
        tuple(sorted(keep_pipeline_ids)),
    ).fetchall():
        keep_face_run_ids.add(int(r[0]))

    # also keep last N face_runs (any scope) as additional safety net
    rows = cur.execute("SELECT id FROM face_runs ORDER BY id DESC LIMIT ?", (int(keep_per_kind),)).fetchall()
    keep_face_run_ids.update(int(r[0]) for r in rows)

    all_face_run_ids = {int(r[0]) for r in cur.execute("SELECT id FROM face_runs").fetchall()}
    delete_face_run_ids = all_face_run_ids - keep_face_run_ids

    return Plan(
        keep_pipeline_ids=keep_pipeline_ids,
        delete_pipeline_ids=delete_pipeline_ids,
        keep_face_run_ids=keep_face_run_ids,
        delete_face_run_ids=delete_face_run_ids,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Cleanup old pipeline/face runs in data/photosorter.db")
    ap.add_argument("--db", default="data/photosorter.db", help="Path to SQLite DB (default: data/photosorter.db)")
    ap.add_argument("--keep-per-kind", type=int, default=5, help="How many latest pipeline_runs to keep per kind (default: 5)")
    ap.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = _connect(db)
    try:
        before = {
            "pipeline_runs": _count(conn, "SELECT COUNT(*) FROM pipeline_runs"),
            "face_runs": _count(conn, "SELECT COUNT(*) FROM face_runs"),
            "face_rectangles": _count(conn, "SELECT COUNT(*) FROM face_rectangles"),
        }
        print("BEFORE:", before)

        plan = _plan(conn, keep_per_kind=int(args.keep_per_kind))
        print(
            "PLAN:",
            {
                "keep_pipeline_runs": len(plan.keep_pipeline_ids),
                "delete_pipeline_runs": len(plan.delete_pipeline_ids),
                "keep_face_runs": len(plan.keep_face_run_ids),
                "delete_face_runs": len(plan.delete_face_run_ids),
            },
        )

        if not args.apply:
            print("DRY RUN (no deletions). Use --apply to delete.")
            return 0

        cur = conn.cursor()

        # Delete old pipeline runs
        if plan.delete_pipeline_ids:
            cur.execute(
                "DELETE FROM pipeline_runs WHERE id IN (%s)" % ",".join("?" for _ in plan.delete_pipeline_ids),
                tuple(sorted(plan.delete_pipeline_ids)),
            )

        # Delete old face rectangles for deleted face runs first (FK-less safety)
        if plan.delete_face_run_ids:
            cur.execute(
                "DELETE FROM face_rectangles WHERE run_id IN (%s)" % ",".join("?" for _ in plan.delete_face_run_ids),
                tuple(sorted(plan.delete_face_run_ids)),
            )
            cur.execute(
                "DELETE FROM face_runs WHERE id IN (%s)" % ",".join("?" for _ in plan.delete_face_run_ids),
                tuple(sorted(plan.delete_face_run_ids)),
            )

        conn.commit()

        after = {
            "pipeline_runs": _count(conn, "SELECT COUNT(*) FROM pipeline_runs"),
            "face_runs": _count(conn, "SELECT COUNT(*) FROM face_runs"),
            "face_rectangles": _count(conn, "SELECT COUNT(*) FROM face_rectangles"),
        }
        print("AFTER:", after)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




