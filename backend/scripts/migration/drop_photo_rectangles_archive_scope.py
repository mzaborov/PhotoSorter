#!/usr/bin/env python3
"""
Удаление колонки archive_scope из photo_rectangles (3NF).

Статус архива берётся из files.inventory_scope, дублирование в photo_rectangles нарушает 3NF.
"""
import sys
import sqlite3
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection


def main() -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(photo_rectangles)")
    columns = {row[1] for row in cur.fetchall()}
    if "archive_scope" not in columns:
        print("✅ Колонка archive_scope уже удалена из photo_rectangles")
        conn.close()
        return 0

    version = sqlite3.sqlite_version_info
    if version < (3, 35, 0):
        print(f"❌ DROP COLUMN требует SQLite >= 3.35.0, текущая версия: {version}")
        conn.close()
        return 1

    # Удаляем индекс
    cur.execute("DROP INDEX IF EXISTS idx_photo_rect_archive_scope")
    conn.commit()

    cur.execute("ALTER TABLE photo_rectangles DROP COLUMN archive_scope")
    conn.commit()

    print("✅ Колонка archive_scope удалена из photo_rectangles (3NF)")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
