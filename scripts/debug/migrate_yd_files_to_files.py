from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "photosorter.db"


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    return cur.fetchone() is not None


def _count(cur: sqlite3.Cursor, name: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {name}")
    return int(cur.fetchone()[0] or 0)


def _columns(cur: sqlite3.Cursor, name: str) -> list[str]:
    cur.execute(f"PRAGMA table_info({name})")
    # row: (cid, name, type, notnull, dflt_value, pk)
    return [str(r[1]) for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate data from legacy table yd_files to new table files")
    ap.add_argument("--db", default=str(_default_db_path()), help="Path to photosorter.db")
    ap.add_argument("--drop-old", action="store_true", help="Drop legacy yd_files after successful copy")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()

        yd_ex = _table_exists(cur, "yd_files")
        files_ex = _table_exists(cur, "files")
        if not yd_ex and not files_ex:
            print("Nothing to do: neither yd_files nor files exists.")
            return 0
        if not yd_ex and files_ex:
            print("Nothing to do: yd_files does not exist.")
            return 0
        if yd_ex and not files_ex:
            print("files does not exist; renaming yd_files -> files")
            cur.execute("ALTER TABLE yd_files RENAME TO files;")
            con.commit()
            return 0

        # Both exist: copy rows (best-effort) by intersection of columns (exclude id so autoincrement works).
        yd_cols = [c for c in _columns(cur, "yd_files") if c != "id"]
        files_cols = [c for c in _columns(cur, "files") if c != "id"]
        common = [c for c in yd_cols if c in set(files_cols)]
        if not common:
            raise SystemExit("No common columns between yd_files and files (unexpected).")

        before_files = _count(cur, "files")
        before_yd = _count(cur, "yd_files")
        print("db:", str(db))
        print("before: files rows =", before_files, "; yd_files rows =", before_yd)
        print("copy columns:", ", ".join(common))

        cols_sql = ", ".join(common)
        cur.execute(f"INSERT OR IGNORE INTO files ({cols_sql}) SELECT {cols_sql} FROM yd_files;")
        con.commit()

        after_files = _count(cur, "files")
        after_yd = _count(cur, "yd_files")
        print("after:  files rows =", after_files, "; yd_files rows =", after_yd)
        print("inserted:", max(0, after_files - before_files))

        if args.drop_old:
            print("dropping legacy table yd_files...")
            cur.execute("DROP TABLE yd_files;")
            con.commit()
            print("dropped.")

        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())





