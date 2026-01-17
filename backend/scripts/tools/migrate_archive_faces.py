#!/usr/bin/env python3
"""
Миграция архивных лиц из прогонов в архивный режим (без привязки к run_id).

Находит все face_runs с scope='yadisk' и root_path начинающимся с 'disk:/Фото',
и перемещает связанные face_rectangles и face_clusters в архивный режим
(archive_scope='archive', run_id=NULL для кластеров).

Сохраняет все связи: face_cluster_members, face_labels.
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
    
    # Проверяем, есть ли уже лица с archive_scope='archive'
    cur.execute(
        """
        SELECT COUNT(*) as cnt
        FROM face_rectangles
        WHERE archive_scope = 'archive'
        """
    )
    existing_archive = cur.fetchone()["cnt"]
    
    if existing_archive > 0:
        print(f"ВНИМАНИЕ: Найдено {existing_archive} лиц с archive_scope='archive'.")
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
            "SELECT COUNT(*) as cnt FROM face_rectangles WHERE run_id = ?",
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
                FROM face_labels fl
                JOIN face_clusters fc ON fl.cluster_id = fc.id
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
        
        # 1. Мигрируем face_rectangles
        # Проверяем дубликаты перед миграцией (по file_path + bbox)
        cur.execute(
            """
            UPDATE face_rectangles
            SET archive_scope = 'archive'
            WHERE run_id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM face_rectangles fr2
                  WHERE fr2.archive_scope = 'archive'
                    AND fr2.file_path = face_rectangles.file_path
                    AND fr2.bbox_x = face_rectangles.bbox_x
                    AND fr2.bbox_y = face_rectangles.bbox_y
                    AND fr2.bbox_w = face_rectangles.bbox_w
                    AND fr2.bbox_h = face_rectangles.bbox_h
              )
            """,
            (run_id,),
        )
        migrated_in_run = cur.rowcount
        migrated_faces += migrated_in_run
        
        # Подсчитываем дубликаты (которые не были мигрированы из-за EXISTS)
        cur.execute(
            """
            SELECT COUNT(*) as cnt
            FROM face_rectangles
            WHERE run_id = ?
              AND archive_scope IS NULL
              AND EXISTS (
                  SELECT 1 FROM face_rectangles fr2
                  WHERE fr2.archive_scope = 'archive'
                    AND fr2.file_path = face_rectangles.file_path
                    AND fr2.bbox_x = face_rectangles.bbox_x
                    AND fr2.bbox_y = face_rectangles.bbox_y
                    AND fr2.bbox_w = face_rectangles.bbox_w
                    AND fr2.bbox_h = face_rectangles.bbox_h
              )
            """,
            (run_id,),
        )
        duplicates_count = cur.fetchone()["cnt"]
        
        if duplicates_count > 0:
            print(f"  - Мигрировано лиц: {migrated_in_run}, пропущено дубликатов: {duplicates_count}")
            deduplicated_faces += duplicates_count
            
            # Обновляем face_cluster_members для дубликатов:
            # находим архивные лица-дубликаты и переназначаем связи
            cur.execute(
                """
                UPDATE face_cluster_members
                SET face_rectangle_id = (
                    SELECT fr2.id
                    FROM face_rectangles fr2
                    WHERE fr2.archive_scope = 'archive'
                      AND fr2.file_path = (
                          SELECT file_path FROM face_rectangles WHERE id = face_cluster_members.face_rectangle_id
                      )
                      AND fr2.bbox_x = (
                          SELECT bbox_x FROM face_rectangles WHERE id = face_cluster_members.face_rectangle_id
                      )
                      AND fr2.bbox_y = (
                          SELECT bbox_y FROM face_rectangles WHERE id = face_cluster_members.face_rectangle_id
                      )
                      AND fr2.bbox_w = (
                          SELECT bbox_w FROM face_rectangles WHERE id = face_cluster_members.face_rectangle_id
                      )
                      AND fr2.bbox_h = (
                          SELECT bbox_h FROM face_rectangles WHERE id = face_cluster_members.face_rectangle_id
                      )
                    LIMIT 1
                )
                WHERE face_rectangle_id IN (
                    SELECT id FROM face_rectangles
                    WHERE run_id = ? AND archive_scope IS NULL
                )
                AND EXISTS (
                    SELECT 1 FROM face_rectangles fr3
                    WHERE fr3.id = face_cluster_members.face_rectangle_id
                      AND fr3.archive_scope IS NULL
                      AND EXISTS (
                          SELECT 1 FROM face_rectangles fr4
                          WHERE fr4.archive_scope = 'archive'
                            AND fr4.file_path = fr3.file_path
                            AND fr4.bbox_x = fr3.bbox_x
                            AND fr4.bbox_y = fr3.bbox_y
                            AND fr4.bbox_w = fr3.bbox_w
                            AND fr4.bbox_h = fr3.bbox_h
                      )
                )
                """,
                (run_id,)
            )
            
            # Аналогично для face_labels
            cur.execute(
                """
                UPDATE face_labels
                SET face_rectangle_id = (
                    SELECT fr2.id
                    FROM face_rectangles fr2
                    WHERE fr2.archive_scope = 'archive'
                      AND fr2.file_path = (
                          SELECT file_path FROM face_rectangles WHERE id = face_labels.face_rectangle_id
                      )
                      AND fr2.bbox_x = (
                          SELECT bbox_x FROM face_rectangles WHERE id = face_labels.face_rectangle_id
                      )
                      AND fr2.bbox_y = (
                          SELECT bbox_y FROM face_rectangles WHERE id = face_labels.face_rectangle_id
                      )
                      AND fr2.bbox_w = (
                          SELECT bbox_w FROM face_rectangles WHERE id = face_labels.face_rectangle_id
                      )
                      AND fr2.bbox_h = (
                          SELECT bbox_h FROM face_rectangles WHERE id = face_labels.face_rectangle_id
                      )
                    LIMIT 1
                )
                WHERE face_rectangle_id IN (
                    SELECT id FROM face_rectangles
                    WHERE run_id = ? AND archive_scope IS NULL
                )
                AND EXISTS (
                    SELECT 1 FROM face_rectangles fr3
                    WHERE fr3.id = face_labels.face_rectangle_id
                      AND fr3.archive_scope IS NULL
                      AND EXISTS (
                          SELECT 1 FROM face_rectangles fr4
                          WHERE fr4.archive_scope = 'archive'
                            AND fr4.file_path = fr3.file_path
                            AND fr4.bbox_x = fr3.bbox_x
                            AND fr4.bbox_y = fr3.bbox_y
                            AND fr4.bbox_w = fr3.bbox_w
                            AND fr4.bbox_h = fr3.bbox_h
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
    
    # Финальная статистика
    cur.execute(
        """
        SELECT COUNT(*) as cnt
        FROM face_rectangles
        WHERE archive_scope = 'archive'
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
