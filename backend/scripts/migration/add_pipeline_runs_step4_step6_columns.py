#!/usr/bin/env python3
"""
DDL-миграция: добавить колонки step4_processed, step4_total, step4_phase, step5_done, step6_done в pipeline_runs.

Нужны для прогресса шагов 5/6 на главной («Переместить локально» / «Перенести в архив»).
DDL только через миграции, не в runtime.

Запуск из корня репозитория:
  python backend/scripts/migration/add_pipeline_runs_step4_step6_columns.py
  python backend/scripts/migration/add_pipeline_runs_step4_step6_columns.py --dry-run
"""

import sys
import sqlite3
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

try:
    from backend.common.db import get_connection, DB_PATH
except ImportError:
    from common.db import get_connection, DB_PATH


TABLE_NAME = "pipeline_runs"
COLUMNS_DDL = {
    "step4_processed": "step4_processed INTEGER",
    "step4_total": "step4_total INTEGER",
    "step4_phase": "step4_phase TEXT",
    "step5_done": "step5_done INTEGER",
    "step6_done": "step6_done INTEGER",
}


def get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def apply_migration(conn: sqlite3.Connection, dry_run: bool) -> int:
    """
    Добавляет недостающие колонки в pipeline_runs.
    Возвращает количество добавленных колонок.
    """
    existing = get_columns(conn, TABLE_NAME)
    added = 0
    cur = conn.cursor()
    for col, ddl in COLUMNS_DDL.items():
        if col in existing:
            continue
        if dry_run:
            print(f"[DRY RUN] Будет выполнено: ALTER TABLE {TABLE_NAME} ADD COLUMN {ddl}")
            added += 1
        else:
            cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {ddl}")
            conn.commit()
            added += 1
    return added


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Добавить колонки step4/5/6 в pipeline_runs (прогресс шагов «Переместить локально» / «Перенести в архив»)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Ошибка: БД не найдена: {DB_PATH}", file=sys.stderr)
        return 1

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
        if not cur.fetchone():
            print(f"Таблица {TABLE_NAME} не найдена. Пропуск миграции.", file=sys.stderr)
            return 0
        added = apply_migration(conn, dry_run=args.dry_run)
        if args.dry_run:
            print("Завершено (dry-run). Запустите без --dry-run для применения.")
            return 0
        if added:
            print(f"Добавлено колонок в {TABLE_NAME}: {added}.")
        else:
            print(f"Все колонки уже присутствуют в {TABLE_NAME}.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
