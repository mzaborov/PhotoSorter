#!/usr/bin/env python3
"""
Удаление лишних записей в file_groups: для каждой тройки (pipeline_run_id, file_id, group_path)
оставляем одну запись (с минимальным id), остальные удаляем.

Запуск:
  python backend/scripts/migration/cleanup_file_groups_duplicates.py --dry-run   # только показать
  python backend/scripts/migration/cleanup_file_groups_duplicates.py              # выполнить
"""

import sys
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
    import argparse
    ap = argparse.ArgumentParser(description="Удалить дубликаты в file_groups (оставить по одной записи на тройку)")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет удалено")
    args = ap.parse_args()

    conn = get_connection()
    try:
        cur = conn.cursor()

        # id дубликатов: для каждой (pipeline_run_id, file_id, group_path) оставляем MIN(id), остальные — на удаление
        cur.execute(
            """
            SELECT fg.id
            FROM file_groups fg
            JOIN (
                SELECT pipeline_run_id, file_id, group_path, MIN(id) AS keep_id
                FROM file_groups
                GROUP BY pipeline_run_id, file_id, group_path
                HAVING COUNT(*) > 1
            ) dup ON dup.pipeline_run_id = fg.pipeline_run_id
                  AND dup.file_id = fg.file_id
                  AND dup.group_path = fg.group_path
                  AND fg.id != dup.keep_id
            ORDER BY fg.id
            """
        )
        ids_to_delete = [row[0] for row in cur.fetchall()]

        if not ids_to_delete:
            print("Дубликатов в file_groups нет. Ничего не удаляем.")
            return 0

        print(f"Будет удалено записей: {len(ids_to_delete)} (id: {ids_to_delete})")

        if args.dry_run:
            print("\n[DRY RUN] Изменения не применены. Запустите без --dry-run для удаления.")
            return 0

        placeholders = ",".join("?" * len(ids_to_delete))
        cur.execute(f"DELETE FROM file_groups WHERE id IN ({placeholders})", ids_to_delete)
        deleted = cur.rowcount
        conn.commit()
        print(f"\nУдалено записей: {deleted}. Готово.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
