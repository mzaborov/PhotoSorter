#!/usr/bin/env python3
"""
Миграция архивных лиц из прогонов в архивный режим (без привязки к run_id).

Находит все face_runs с scope='yadisk' и root_path начинающимся с 'disk:/Фото',
и перемещает связанные face_clusters в архивный режим (archive_scope='archive').
Для photo_rectangles: статус архива берётся из files.inventory_scope (3NF).

ПРИМЕЧАНИЕ: photo_rectangles.archive_scope удалён (миграция drop_photo_rectangles_archive_scope).
Вместо UPDATE photo_rectangles — обновляем files.inventory_scope='archive' для файлов.

TODO: При переезде файлов в архив нужно копировать people_no_face_person
из files_manual_labels в files.people_no_face_person, чтобы привязка сохранилась.
См. комментарий в backend/common/db.py около строки 243.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

# Загружаем secrets.env/.env
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection


def main() -> int:
    print("=" * 60)
    print("МИГРАЦИЯ АРХИВНЫХ ЛИЦ В АРХИВНЫЙ РЕЖИМ")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Находим все архивные face_runs (scope='yadisk', root_path начинается с 'disk:/Фото')
    cur.execute(
        """
        SELECT id, scope, root_path, status, faces_found
        FROM face_runs
        WHERE scope = 'yadisk' AND root_path LIKE 'disk:/Фото/%'
        ORDER BY started_at DESC
        """
    )
    
    archive_runs = cur.fetchall()
    
    if not archive_runs:
        print("Архивные face_runs не найдены. Миграция не требуется.")
        return 0
    
    print(f"Найдено архивных прогонов: {len(archive_runs)}")
    for run in archive_runs:
        print(f"  - run_id={run['id']}, root_path={run['root_path']}, faces={run['faces_found']}")
    print()
    
    # Проверяем, есть ли уже лица с файлами в архиве (files.inventory_scope='archive')
    cur.execute(
        """
        SELECT COUNT(*) as cnt
        FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        WHERE f.inventory_scope = 'archive'
        """
    )
    existing_archive = cur.fetchone()["cnt"]
    
    if existing_archive > 0:
        print(f"ВНИМАНИЕ: Найдено {existing_archive} лиц в файлах с inventory_scope='archive'.")
        print("Миграция может создать дубликаты. Рекомендуется сначала очистить архивные данные.")
        response = input("Продолжить миграцию? (yes/no): ")
        if response.lower() != "yes":
            print("Миграция отменена.")
            return 1
        print()
    
    # Статистика перед миграцией
    total_faces = 0
    total_clusters = 0
    total_labels = 0
    
    for run in archive_runs:
        run_id = run["id"]
        
        # Подсчитываем лица
        cur.execute(
            "SELECT COUNT(*) as cnt FROM photo_rectangles WHERE run_id = ? AND is_face = 1",
            (run_id,)
        )
        faces_count = cur.fetchone()["cnt"]
        total_faces += faces_count
        
        # Подсчитываем кластеры
        cur.execute(
            "SELECT COUNT(*) as cnt FROM face_clusters WHERE run_id = ?",
            (run_id,)
        )
        clusters_count = cur.fetchone()["cnt"]
        total_clusters += clusters_count
        
        # Подсчитываем метки (через кластеры)
        if clusters_count > 0:
            cur.execute(
                """
                SELECT COUNT(*) as cnt
                FROM person_rectangle_manual_assignments prma
                JOIN face_cluster_members fcm ON fcm.rectangle_id = prma.rectangle_id
                JOIN face_clusters fc ON fc.id = fcm.cluster_id
                WHERE fc.run_id = ?
                """,
                (run_id,)
            )
            labels_count = cur.fetchone()["cnt"]
            total_labels += labels_count
    
    print(f"Статистика для миграции:")
    print(f"  - Лиц: {total_faces}")
    print(f"  - Кластеров: {total_clusters}")
    print(f"  - Меток через кластеры: {total_labels}")
    print()
    
    # Выполняем миграцию
    print("Начинаем миграцию...")
    print()
    
    migrated_faces = 0
    migrated_clusters = 0
    deduplicated_faces = 0
    
    for run in archive_runs:
        run_id = run["id"]
        root_path = run["root_path"]
        
        print(f"Обработка run_id={run_id} ({root_path})...")
        
        # 1. Обновляем files.inventory_scope='archive' для файлов с лицами из этого run
        # (photo_rectangles.archive_scope удалён — статус в files.inventory_scope)
        cur.execute(
            """
            UPDATE files
            SET inventory_scope = 'archive'
            WHERE id IN (
                SELECT DISTINCT file_id FROM photo_rectangles
                WHERE run_id = ? AND is_face = 1
            )
            AND (inventory_scope IS NULL OR inventory_scope != 'archive')
            """,
            (run_id,),
        )
        migrated_files = cur.rowcount
        cur.execute(
            "SELECT COUNT(*) as cnt FROM photo_rectangles WHERE run_id = ? AND is_face = 1",
            (run_id,),
        )
        migrated_in_run = cur.fetchone()["cnt"]
        migrated_faces += migrated_in_run
        
        # Подсчитываем дубликаты (файлы уже в архиве)
        cur.execute(
            """
            SELECT COUNT(*) as cnt
            FROM photo_rectangles pr
            JOIN files f ON f.id = pr.file_id
            WHERE pr.run_id = ? AND pr.is_face = 1
              AND f.inventory_scope = 'archive'
            """,
            (run_id,),
        )
        duplicates_count = cur.fetchone()["cnt"]
        
        if duplicates_count > 0:
            print(f"  - Обновлено файлов: {migrated_files}, лиц в run: {migrated_in_run}, уже в архиве: {duplicates_count}")
            deduplicated_faces += duplicates_count
            
            # Обновляем face_cluster_members для дубликатов:
            # находим архивные лица (файл с inventory_scope='archive') и переназначаем связи
            cur.execute(
                """
                UPDATE face_cluster_members
                SET rectangle_id = (
                    SELECT pr2.id
                    FROM photo_rectangles pr2
                    JOIN files f2 ON f2.id = pr2.file_id
                    WHERE f2.inventory_scope = 'archive'
                      AND pr2.file_id = (
                          SELECT file_id FROM photo_rectangles WHERE id = face_cluster_members.rectangle_id
                      )
                      AND pr2.bbox_x = (
                          SELECT bbox_x FROM photo_rectangles WHERE id = face_cluster_members.rectangle_id
                      )
                      AND pr2.bbox_y = (
                          SELECT bbox_y FROM photo_rectangles WHERE id = face_cluster_members.rectangle_id
                      )
                      AND pr2.bbox_w = (
                          SELECT bbox_w FROM photo_rectangles WHERE id = face_cluster_members.rectangle_id
                      )
                      AND pr2.bbox_h = (
                          SELECT bbox_h FROM photo_rectangles WHERE id = face_cluster_members.rectangle_id
                      )
                    LIMIT 1
                )
                WHERE rectangle_id IN (
                    SELECT id FROM photo_rectangles
                    WHERE run_id = ? AND archive_scope IS NULL
                )
                AND EXISTS (
                    SELECT 1 FROM photo_rectangles pr3
                    LEFT JOIN files f3 ON f3.id = pr3.file_id
                    WHERE pr3.id = face_cluster_members.rectangle_id
                      AND (f3.inventory_scope IS NULL OR f3.inventory_scope != 'archive')
                      AND EXISTS (
                          SELECT 1 FROM photo_rectangles pr4
                          JOIN files f4 ON f4.id = pr4.file_id
                          WHERE f4.inventory_scope = 'archive'
                            AND pr4.file_id = pr3.file_id
                            AND pr4.bbox_x = pr3.bbox_x
                            AND pr4.bbox_y = pr3.bbox_y
                            AND pr4.bbox_w = pr3.bbox_w
                            AND pr4.bbox_h = pr3.bbox_h
                      )
                )
                """,
                (run_id,)
            )
            
            # Аналогично для person_rectangle_manual_assignments
            cur.execute(
                """
                UPDATE person_rectangle_manual_assignments
                SET rectangle_id = (
                    SELECT pr2.id
                    FROM photo_rectangles pr2
                    JOIN files f2 ON f2.id = pr2.file_id
                    WHERE f2.inventory_scope = 'archive'
                      AND pr2.file_id = (
                          SELECT file_id FROM photo_rectangles WHERE id = person_rectangle_manual_assignments.rectangle_id
                      )
                      AND pr2.bbox_x = (
                          SELECT bbox_x FROM photo_rectangles WHERE id = person_rectangle_manual_assignments.rectangle_id
                      )
                      AND pr2.bbox_y = (
                          SELECT bbox_y FROM photo_rectangles WHERE id = person_rectangle_manual_assignments.rectangle_id
                      )
                      AND pr2.bbox_w = (
                          SELECT bbox_w FROM photo_rectangles WHERE id = person_rectangle_manual_assignments.rectangle_id
                      )
                      AND pr2.bbox_h = (
                          SELECT bbox_h FROM photo_rectangles WHERE id = person_rectangle_manual_assignments.rectangle_id
                      )
                    LIMIT 1
                )
                WHERE rectangle_id IN (
                    SELECT pr.id FROM photo_rectangles pr
                    LEFT JOIN files f ON f.id = pr.file_id
                    WHERE pr.run_id = ? AND (f.inventory_scope IS NULL OR f.inventory_scope != 'archive')
                )
                AND EXISTS (
                    SELECT 1 FROM photo_rectangles pr3
                    LEFT JOIN files f3 ON f3.id = pr3.file_id
                    WHERE pr3.id = person_rectangle_manual_assignments.rectangle_id
                      AND (f3.inventory_scope IS NULL OR f3.inventory_scope != 'archive')
                      AND EXISTS (
                          SELECT 1 FROM photo_rectangles pr4
                          JOIN files f4 ON f4.id = pr4.file_id
                          WHERE f4.inventory_scope = 'archive'
                            AND pr4.file_id = pr3.file_id
                            AND pr4.bbox_x = pr3.bbox_x
                            AND pr4.bbox_y = pr3.bbox_y
                            AND pr4.bbox_w = pr3.bbox_w
                            AND pr4.bbox_h = pr3.bbox_h
                      )
                )
                """,
                (run_id,)
            )
        else:
            print(f"  - Мигрировано лиц: {migrated_in_run}")
        
        # 2. Мигрируем face_clusters (устанавливаем archive_scope='archive', run_id оставляем для совместимости)
        cur.execute(
            """
            UPDATE face_clusters
            SET archive_scope = 'archive'
            WHERE run_id = ?
            """,
            (run_id,),
        )
        clusters_in_run = cur.rowcount
        migrated_clusters += clusters_in_run
        print(f"  - Мигрировано кластеров: {clusters_in_run}")
        
        conn.commit()
        print()
    
    # Финальная статистика (лица в файлах с inventory_scope='archive')
    cur.execute(
        """
        SELECT COUNT(*) as cnt
        FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        WHERE f.inventory_scope = 'archive' AND pr.is_face = 1
        """
    )
    final_faces = cur.fetchone()["cnt"]
    
    cur.execute(
        """
        SELECT COUNT(*) as cnt
        FROM face_clusters
        WHERE archive_scope = 'archive'
        """
    )
    final_clusters = cur.fetchone()["cnt"]
    
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Итоговая статистика:")
    print(f"  - Лиц в архиве: {final_faces} (мигрировано: {migrated_faces}, пропущено дубликатов: {deduplicated_faces})")
    print(f"  - Кластеров в архиве: {final_clusters} (мигрировано: {migrated_clusters})")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
