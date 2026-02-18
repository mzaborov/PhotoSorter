#!/usr/bin/env python3
"""Проверка целостности бекапа БД: PRAGMA integrity_check."""
import sqlite3
import sys
from pathlib import Path

def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("backup_path", type=Path, help="Путь к файлу бекапа .db")
    args = ap.parse_args()
    p = args.backup_path.resolve()
    if not p.exists():
        print(f"Файл не найден: {p}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(str(p))
    try:
        r = conn.execute("PRAGMA integrity_check").fetchone()
        ok = r and r[0] == "ok"
        print("integrity_check:", r[0] if r else "?")
        return 0 if ok else 1
    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(main())
