from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _is_readonly_sql(sql: str) -> bool:
    s = (sql or "").strip().lstrip("\ufeff").strip()
    if not s:
        return False
    # Disallow multiple statements (best-effort): allow at most one trailing ';'
    if ";" in s[:-1]:
        return False
    first = s.split(None, 1)[0].lower()
    return first in ("select", "with", "pragma")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Execute a READ-ONLY SQL query from a file against photosorter.db.\n"
            "Allowed starters: SELECT / WITH / PRAGMA. Disallows multi-statement SQL."
        )
    )
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--sql-file", required=True, help="Path to .sql file with a single SELECT/WITH/PRAGMA statement")
    ap.add_argument("--limit", type=int, default=200, help="Max rows to print (safety)")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    sql_path = Path(args.sql_file)
    if not sql_path.exists():
        raise SystemExit(f"SQL file not found: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8", errors="replace")
    if not _is_readonly_sql(sql):
        raise SystemExit("Only single-statement SELECT/WITH/PRAGMA from file is allowed.")

    conn = _connect(db)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(int(args.limit))
        if not rows:
            print("(no rows)")
            return 0

        cols = list(rows[0].keys())
        print("\t".join(cols))
        for r in rows:
            print("\t".join("" if r[c] is None else str(r[c]) for c in cols))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

