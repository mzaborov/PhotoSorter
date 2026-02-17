#!/usr/bin/env python3
"""
Инициализация trip_files из папок YD поездок.

Для каждой поездки с yd_folder_path добавляет в trip_files все файлы из files,
чьи пути под yd_folder_path, со status='included', source='folder'.
Пропускает файлы, уже имеющие запись в trip_files (сохраняет excluded/manual).

Запуск один раз после настройки yd_folder_path: python backend/scripts/tools/init_trip_folder_files.py [--dry-run]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from common.db import get_connection, init_db, _now_utc_iso


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Инициализация trip_files из папок YD поездок (status=included, source=folder)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Не писать в БД, только вывести список")
    args = parser.parse_args()

    init_db()

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, yd_folder_path FROM trips WHERE yd_folder_path IS NOT NULL AND yd_folder_path != ''"
        )
        trips = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    if not trips:
        print("Нет поездок с yd_folder_path.")
        return

    total_added = 0
    for t in trips:
        trip_id = int(t["id"])
        name = (t.get("name") or "").strip() or f"trip_{trip_id}"
        yd_folder = (t.get("yd_folder_path") or "").strip().rstrip("/")
        if not yd_folder:
            continue
        yd_prefix = yd_folder + "/"
        # На ЯД в путях иногда папка с пробелом в конце: "2019 Лиссабон /file" — учитываем оба варианта
        yd_prefix_with_space = yd_folder + " /"

        conn = get_connection()
        try:
            cur = conn.cursor()
            # Файлы в папке, ещё не в trip_files
            cur.execute(
                """
                SELECT f.id AS file_id
                FROM files f
                WHERE (f.path LIKE ? OR f.path LIKE ? OR f.path = ?)
                  AND (f.status IS NULL OR f.status != 'deleted')
                  AND f.id NOT IN (SELECT file_id FROM trip_files WHERE trip_id = ?)
                """,
                (yd_prefix + "%", yd_prefix_with_space + "%", yd_folder, trip_id),
            )
            to_add = [row["file_id"] for row in cur.fetchall()]
        finally:
            conn.close()

        if not to_add:
            continue

        print(f"{name} (id={trip_id}): {len(to_add)} файлов в {yd_folder}")

        if not args.dry_run:
            now = _now_utc_iso()
            conn = get_connection()
            try:
                cur = conn.cursor()
                for file_id in to_add:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO trip_files (trip_id, file_id, status, source, created_at)
                        VALUES (?, ?, 'included', 'folder', ?)
                        """,
                        (trip_id, file_id, now),
                    )
                conn.commit()
            finally:
                conn.close()
        total_added += len(to_add)

    if args.dry_run:
        print("\n[dry-run] Изменения не применены.")
    else:
        print(f"\nДобавлено в trip_files: {total_added} записей.")


if __name__ == "__main__":
    main()
