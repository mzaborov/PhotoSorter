"""
Очищает thumb_jpeg только у сортируемых и ручных одновременно: записи из прогона
(run_id IS NOT NULL), привязанные к персоне через кластер или manual_person_id.
Архивные лица (run_id NULL) не трогаем.
После запуска миниатюры пересчитаются при следующей загрузке страницы персоны.

Пример:
  python backend/scripts/tools/clear_face_thumbs_for_person.py --person-id 9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Корень репозитория (PhotoSorter/) — для импорта backend
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.common.db import get_connection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Очистить thumb_jpeg у лиц указанной персоны (кропы пересчитаются при открытии страницы персоны)."
    )
    parser.add_argument(
        "--person-id",
        type=int,
        default=9,
        help="ID персоны (по умолчанию 9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, сколько записей будет обновлено, без изменений в БД",
    )
    args = parser.parse_args()
    person_id = args.person_id

    conn = get_connection()
    try:
        cur = conn.cursor()
        # Только из прогона (run_id IS NOT NULL) и привязка к персоне (кластер или manual); архив не трогаем
        where_clause = """
            run_id IS NOT NULL
            AND (cluster_id IN (SELECT id FROM face_clusters WHERE person_id = ?) OR manual_person_id = ?)
            AND thumb_jpeg IS NOT NULL
        """
        cur.execute(
            f"SELECT COUNT(*) FROM photo_rectangles WHERE {where_clause}",
            (person_id, person_id),
        )
        count = cur.fetchone()[0]
        if count == 0:
            print(f"У персоны {person_id} нет сортируемых записей с thumb_jpeg (всё уже пусто или лиц нет).")
            return
        if args.dry_run:
            print(f"Будет обнулено thumb_jpeg у {count} записей персоны {person_id} (сортируемые + ручные). Запустите без --dry-run для применения.")
            return
        cur.execute(
            """
            UPDATE photo_rectangles SET thumb_jpeg = NULL
            WHERE run_id IS NOT NULL
              AND (cluster_id IN (SELECT id FROM face_clusters WHERE person_id = ?) OR manual_person_id = ?)
              AND thumb_jpeg IS NOT NULL
            """,
            (person_id, person_id),
        )
        updated = cur.rowcount
        conn.commit()
        print(f"Обнулён thumb_jpeg у {updated} записей персоны {person_id}. Откройте страницу /persons/{person_id} — кропы пересчитаются.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
