#!/usr/bin/env python3
"""
Миграция: удаление таблицы person_rectangles.

Таблица person_rectangles была пуста (подтверждено analyze_person_rectangles.py).
Миграция данных не требуется. Код и схема в db.py уже не создают таблицу.

Перед запуском:
1. Сделать бекап: python backend/scripts/tools/backup_database.py
2. Проверить целостность бекапа: python backend/scripts/debug/verify_backup_integrity.py <путь_к_бекапу.db>
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection, DB_PATH


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DROP TABLE person_rectangles")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет выполнено")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_rectangles'")
        if not cur.fetchone():
            print("Таблица person_rectangles уже отсутствует.")
            return 0
        if args.dry_run:
            print("[DRY RUN] Будет выполнено: DROP TABLE IF EXISTS person_rectangles;")
            return 0
        cur.execute("DROP TABLE IF EXISTS person_rectangles")
        conn.commit()
        print("Таблица person_rectangles удалена.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
