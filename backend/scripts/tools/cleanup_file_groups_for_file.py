#!/usr/bin/env python3
"""
Очистка записей file_groups для одного файла: оставить только одну группу.

Использование:
  python backend/scripts/tools/cleanup_file_groups_for_file.py --file-id 27269 --group-path "2025 Гончарка Москва"
  python backend/scripts/tools/cleanup_file_groups_for_file.py --path "local:C:\\tmp\\Photo\\_no_faces\\VID-20250511-WA0009.mp4" --group-path "2025 Гончарка Москва" --dry-run
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection, _get_file_id_from_path


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description="Оставить для файла только одну группу в file_groups")
    ap.add_argument("--file-id", type=int, help="id файла в таблице files")
    ap.add_argument("--path", type=str, help="path файла (local: или disk:) — альтернатива --file-id")
    ap.add_argument("--group-path", type=str, required=True, help="Единственная группа, которую оставить (например: 2025 Гончарка Москва)")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    args = ap.parse_args()

    if args.file_id is None and not (args.path or "").strip():
        ap.error("Укажите --file-id или --path")
    if args.file_id is not None and (args.path or "").strip():
        ap.error("Укажите только один из --file-id или --path")

    conn = get_connection()
    try:
        cur = conn.cursor()
        if args.file_id is not None:
            file_id = args.file_id
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (file_id,))
            row = cur.fetchone()
            file_path = row[0] if row else None
        else:
            path = (args.path or "").strip()
            file_id = _get_file_id_from_path(conn, path)
            file_path = path
            if file_id is None:
                print(f"Файл не найден: {path}")
                return 1

        # Текущие записи
        cur.execute(
            "SELECT id, pipeline_run_id, group_path, created_at FROM file_groups WHERE file_id = ? ORDER BY id",
            (file_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print(f"У файла file_id={file_id} нет записей в file_groups.")
            return 0

        group_path = (args.group_path or "").strip()
        if not group_path:
            print("--group-path не может быть пустым.")
            return 1

        pipeline_run_id = int(rows[0]["pipeline_run_id"])

        to_delete = [r for r in rows if (dict(r).get("group_path") or "").strip() != group_path]
        # Также удалим лишние дубликаты с тем же group_path (оставим одну)
        same_group = [r for r in rows if (dict(r).get("group_path") or "").strip() == group_path]
        if len(same_group) > 1:
            to_delete.extend(same_group[1:])  # оставить первую, остальные удалить
        ids_to_delete = [dict(r)["id"] for r in to_delete]

        print(f"Файл: file_id={file_id}, path={file_path}")
        print(f"pipeline_run_id: {pipeline_run_id}")
        print(f"Оставляем группу: {group_path!r}")
        print(f"Текущих записей: {len(rows)}")
        print(f"Будет удалено записей: {len(ids_to_delete)} (id: {ids_to_delete})")

        if args.dry_run:
            print("\n[DRY RUN] Изменения не применены.")
            return 0

        if not ids_to_delete:
            # Уже одна запись с нужной группой
            print("\nУже одна запись с указанной группой. Ничего не меняем.")
            return 0

        placeholders = ",".join("?" * len(ids_to_delete))
        cur.execute(f"DELETE FROM file_groups WHERE id IN ({placeholders})", ids_to_delete)
        deleted = cur.rowcount

        # Если не осталось ни одной записи с нужной группой — вставляем одну
        cur.execute(
            "SELECT 1 FROM file_groups WHERE file_id = ? AND pipeline_run_id = ? AND group_path = ? LIMIT 1",
            (file_id, pipeline_run_id, group_path),
        )
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO file_groups (pipeline_run_id, file_id, group_path, created_at) VALUES (?, ?, ?, ?)",
                (pipeline_run_id, file_id, group_path, _now_utc_iso()),
            )
            print("Добавлена одна запись с указанной группой.")

        conn.commit()
        print(f"\nУдалено записей: {deleted}. Готово.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
