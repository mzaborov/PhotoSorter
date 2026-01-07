from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "photosorter.db"


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    return cur.fetchone() is not None


def _count_rows(cur: sqlite3.Cursor, name: str) -> int | None:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {name}")
        return int(cur.fetchone()[0] or 0)
    except Exception:
        return None


def _columns(cur: sqlite3.Cursor, name: str) -> list[str]:
    try:
        cur.execute(f"PRAGMA table_info({name})")
        return [str(r[1]) for r in cur.fetchall()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug helper: show SQLite tables + row counts for yd_files/files")
    ap.add_argument("--db", default=str(_default_db_path()), help="Path to photosorter.db")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [str(r[0]) for r in cur.fetchall()]
        print("db:", str(db))
        print("tables:", ", ".join(tables))

        for t in ("yd_files", "files"):
            ex = _table_exists(cur, t)
            cnt = _count_rows(cur, t) if ex else None
            cols = _columns(cur, t) if ex else []
            print(f"{t}: exists={ex} rows={cnt} cols={len(cols)}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





