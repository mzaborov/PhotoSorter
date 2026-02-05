#!/usr/bin/env python3
"""
DDL-миграция: добавить колонку processed в таблицу files.

Колонка processed=1 означает, что файл имеет привязки лиц к персонам (ручные или через кластеры)
и не должен пересканироваться при досчёте лиц — иначе привязки станут невидимыми.

После добавления колонки устанавливает processed=1 для всех файлов,
у которых есть photo_rectangles с manual_person_id или cluster_id (привязки к персонам).

Запуск из корня репозитория:
  python backend/scripts/migration/add_processed_column.py
  python backend/scripts/migration/add_processed_column.py --dry-run
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


COLUMN_NAME = "processed"
COLUMN_DDL = "processed INTEGER NOT NULL DEFAULT 0"
TABLE_NAME = "files"


def get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def apply_migration(conn: sqlite3.Connection, dry_run: bool) -> tuple[bool, int]:
    """
    Добавляет колонку processed в files (если её нет),
    затем устанавливает processed=1 для файлов с привязками.
    Возвращает (changed_ddl, updated_count).
    """
    existing = get_columns(conn, TABLE_NAME)
    changed_ddl = False
    if COLUMN_NAME not in existing:
        if dry_run:
            print(f"[DRY RUN] Будет выполнено: ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_DDL}")
            changed_ddl = True
        else:
            cur = conn.cursor()
            cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_DDL}")
            conn.commit()
            changed_ddl = True

    # Устанавливаем processed=1 для файлов с привязками
    # Привязки: photo_rectangles.manual_person_id IS NOT NULL или cluster_id IS NOT NULL
    cur = conn.cursor()
    if dry_run:
        cur.execute(
            """
            SELECT COUNT(DISTINCT f.id)
            FROM files f
            JOIN photo_rectangles pr ON pr.file_id = f.id
            WHERE pr.manual_person_id IS NOT NULL OR pr.cluster_id IS NOT NULL
            """
        )
        updated_count = int(cur.fetchone()[0] or 0)
        if updated_count:
            print(f"[DRY RUN] Будет установлено processed=1 для {updated_count} файлов с привязками")
    else:
        cur.execute(
            """
            SELECT DISTINCT f.id
            FROM files f
            JOIN photo_rectangles pr ON pr.file_id = f.id
            WHERE (pr.manual_person_id IS NOT NULL OR pr.cluster_id IS NOT NULL)
              AND COALESCE(f.processed, 0) = 0
            """
        )
        file_ids = [row[0] for row in cur.fetchall()]
        updated_count = 0
        if file_ids:
            placeholders = ",".join(["?"] * len(file_ids))
            cur.execute(
                f"UPDATE {TABLE_NAME} SET processed = 1 WHERE id IN ({placeholders})",
                file_ids,
            )
            updated_count = cur.rowcount
            conn.commit()

    return changed_ddl, updated_count


def verify(conn: sqlite3.Connection) -> bool:
    cols = get_columns(conn, TABLE_NAME)
    return COLUMN_NAME in cols


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Добавить колонку processed в files и установить её для файлов с привязками"
    )
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Ошибка: БД не найдена: {DB_PATH}", file=sys.stderr)
        return 1

    conn = get_connection()
    try:
        changed_ddl, updated_count = apply_migration(conn, dry_run=args.dry_run)
        if args.dry_run:
            print("Завершено (dry-run). Запустите без --dry-run для применения.")
            return 0
        if changed_ddl:
            print(f"Выполнено: ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_DDL}")
        if updated_count:
            print(f"Установлено processed=1 для {updated_count} файлов с привязками.")
        if not changed_ddl and not updated_count:
            print(f"Колонка {COLUMN_NAME} уже есть, все файлы с привязками уже обработаны.")

        if not verify(conn):
            print(f"Ошибка проверки: колонка {COLUMN_NAME} не найдена в {TABLE_NAME}.", file=sys.stderr)
            return 1
        print(f"Проверка: колонка {COLUMN_NAME} присутствует в {TABLE_NAME}.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
