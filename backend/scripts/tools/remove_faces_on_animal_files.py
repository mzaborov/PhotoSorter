#!/usr/bin/env python3
"""
Удаляет лица на всех сортируемых фото, где определилось животное (animals_auto=1 или
animals_manual=1). Помечает прямоугольники ignore_flag=1 и удаляет привязки к кластерам
и ручные привязки к персонам — по сути то же, что кнопка «Кот» на /faces, но пакетно.

Только run-scope: обрабатываются photo_rectangles с run_id IS NOT NULL и не архив.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import init_db, FaceStore


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Удалить лица на сортируемых фото с животными (animals_auto/manual=1)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет сделано, не менять БД",
    )
    args = ap.parse_args()
    dry_run = args.dry_run

    init_db()
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()

        # 1. file_ids, где животное: files.animals_auto=1 ИЛИ files_manual_labels.animals_manual=1
        cur.execute(
            """
            SELECT id AS file_id FROM files
            WHERE COALESCE(animals_auto, 0) = 1
            UNION
            SELECT file_id FROM files_manual_labels
            WHERE COALESCE(animals_manual, 0) = 1
            """
        )
        animal_file_ids = [r["file_id"] for r in cur.fetchall()]
        if not animal_file_ids:
            print("Нет файлов с животными (animals_auto=1 или animals_manual=1). Выход.")
            return

        placeholders = ",".join("?" * len(animal_file_ids))

        # 2. Прямоугольники run-scope для этих файлов
        cur.execute(
            f"""
            SELECT pr.id, pr.file_id, pr.run_id
            FROM photo_rectangles pr
            JOIN files f ON f.id = pr.file_id
            WHERE pr.file_id IN ({placeholders})
              AND pr.run_id IS NOT NULL
              AND (f.inventory_scope IS NULL OR f.inventory_scope != 'archive')
            """,
            tuple(animal_file_ids),
        )
        rows = cur.fetchall()
        rect_ids = [r["id"] for r in rows]
        files_touched = len({r["file_id"] for r in rows})

        if not rect_ids:
            print(
                f"Найдено {len(animal_file_ids)} файлов с животными, "
                "но ни одного run-scope прямоугольника. Выход."
            )
            return

        print(f"Файлов с животными: {len(animal_file_ids)}")
        print(f"Файлов с run-scope прямоугольниками: {files_touched}")
        print(f"Прямоугольников к обработке: {len(rect_ids)}")

        # 3. Сколько привязок удалим
        ph = ",".join("?" * len(rect_ids))
        cur.execute(
            f"SELECT COUNT(*) AS c FROM person_rectangle_manual_assignments WHERE rectangle_id IN ({ph})",
            tuple(rect_ids),
        )
        n_manual = cur.fetchone()["c"]
        cur.execute(
            f"SELECT COUNT(*) AS c FROM face_cluster_members WHERE rectangle_id IN ({ph})",
            tuple(rect_ids),
        )
        n_cluster = cur.fetchone()["c"]
        print(f"Ручных привязок к удалению: {n_manual}")
        print(f"Привязок к кластерам к удалению: {n_cluster}")

        if dry_run:
            print("\n[--dry-run] Изменения не применены.")
            return

        # 4. Применяем: ignore_flag=1, удалить привязки
        cur.execute(
            f"UPDATE photo_rectangles SET ignore_flag = 1 WHERE id IN ({ph})",
            tuple(rect_ids),
        )
        cur.execute(
            f"DELETE FROM person_rectangle_manual_assignments WHERE rectangle_id IN ({ph})",
            tuple(rect_ids),
        )
        cur.execute(
            f"DELETE FROM face_cluster_members WHERE rectangle_id IN ({ph})",
            tuple(rect_ids),
        )
        conn.commit()
        print("\nГотово: ignore_flag=1, привязки удалены.")
    finally:
        fs.close()


if __name__ == "__main__":
    main()
