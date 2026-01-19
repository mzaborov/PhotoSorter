#!/usr/bin/env python3
"""
Скрипт для проверки дублей лиц в кластерах.

Находит лица (face_rectangle_id), которые находятся в нескольких кластерах одновременно.
Может фильтровать по person_id для проверки конкретной персоны.

Использование:
    python backend/scripts/tools/check_duplicate_faces_in_clusters.py --person-id 10
    python backend/scripts/tools/check_duplicate_faces_in_clusters.py  # все персоны
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def check_duplicate_faces(person_id: int | None = None) -> None:
    """
    Проверяет дубликаты лиц в кластерах.
    
    Args:
        person_id: опционально, фильтр по person_id (только для кластеров конкретной персоны)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие для фильтрации по персоне
    person_filter = ""
    person_params = []
    if person_id is not None:
        person_filter = """
            AND EXISTS (
                SELECT 1 FROM face_labels fl2 
                WHERE fl2.cluster_id = fcm.cluster_id 
                AND fl2.person_id = ?
            )
        """
        person_params = [person_id]
    
    # Находим все лица, которые находятся в нескольких кластерах
    sql_query = f"""
        SELECT 
            fcm.face_rectangle_id,
            COUNT(DISTINCT fcm.cluster_id) as cluster_count,
            GROUP_CONCAT(DISTINCT fcm.cluster_id) as cluster_ids,
            fr.file_path,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM face_cluster_members fcm
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE COALESCE(fr.ignore_flag, 0) = 0
        {person_filter}
        GROUP BY fcm.face_rectangle_id
        HAVING cluster_count > 1
        ORDER BY cluster_count DESC, fcm.face_rectangle_id
    """
    
    cur.execute(sql_query, person_params)
    duplicates = cur.fetchall()
    
    if len(duplicates) == 0:
        if person_id is not None:
            print(f"Дубликатов лиц в кластерах персоны {person_id} не найдено.")
        else:
            print("Дубликатов лиц в кластерах не найдено.")
        return
    
    print(f"Найдено лиц с дубликатами: {len(duplicates)}")
    if person_id is not None:
        print(f"Для персоны: {person_id}")
    print()
    
    # Группируем по кластерам для более удобного отображения
    cluster_duplicates: dict[int, list[dict]] = defaultdict(list)
    
    for dup in duplicates:
        face_id = dup["face_rectangle_id"]
        cluster_ids_str = dup["cluster_ids"]
        cluster_ids = [int(cid) for cid in cluster_ids_str.split(",") if cid.strip()]
        
        for cluster_id in cluster_ids:
            cluster_duplicates[cluster_id].append({
                "face_id": face_id,
                "file_path": dup["file_path"],
                "bbox": (dup["bbox_x"], dup["bbox_y"], dup["bbox_w"], dup["bbox_h"]),
                "all_clusters": cluster_ids,
            })
    
    # Выводим информацию по кластерам
    print("=" * 80)
    print("Дубликаты по кластерам:")
    print("=" * 80)
    
    for cluster_id in sorted(cluster_duplicates.keys()):
        faces_in_cluster = cluster_duplicates[cluster_id]
        print(f"\nКластер #{cluster_id}: {len(faces_in_cluster)} лиц с дубликатами")
        
        # Получаем информацию о персоне для этого кластера
        cur.execute("""
            SELECT p.id, p.name
            FROM face_labels fl
            JOIN persons p ON fl.person_id = p.id
            WHERE fl.cluster_id = ?
            LIMIT 1
        """, (cluster_id,))
        person_row = cur.fetchone()
        if person_row:
            print(f"  Персона: {person_row['name']} (ID: {person_row['id']})")
        
        # Получаем общее количество лиц в кластере
        cur.execute("""
            SELECT COUNT(DISTINCT fcm.face_rectangle_id) as total_faces
            FROM face_cluster_members fcm
            JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
            WHERE fcm.cluster_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        """, (cluster_id,))
        total_row = cur.fetchone()
        total_faces = total_row["total_faces"] if total_row else 0
        print(f"  Всего лиц в кластере: {total_faces}")
        
        # Показываем первые 10 дубликатов
        for i, face_info in enumerate(faces_in_cluster[:10], 1):
            other_clusters = [cid for cid in face_info["all_clusters"] if cid != cluster_id]
            print(f"  {i}. Лицо #{face_info['face_id']}: также в кластерах {other_clusters}")
            print(f"     Файл: {face_info['file_path']}")
            print(f"     BBox: x={face_info['bbox'][0]}, y={face_info['bbox'][1]}, w={face_info['bbox'][2]}, h={face_info['bbox'][3]}")
        
        if len(faces_in_cluster) > 10:
            print(f"  ... и еще {len(faces_in_cluster) - 10} лиц с дубликатами")
    
    # Статистика
    print("\n" + "=" * 80)
    print("Статистика:")
    print("=" * 80)
    
    # Подсчитываем общее количество уникальных лиц с дубликатами
    unique_duplicate_faces = len(duplicates)
    total_duplicate_entries = sum(dup["cluster_count"] for dup in duplicates)
    
    print(f"Уникальных лиц с дубликатами: {unique_duplicate_faces}")
    print(f"Всего записей в face_cluster_members для дубликатов: {total_duplicate_entries}")
    print(f"Избыточных записей (которые можно удалить): {total_duplicate_entries - unique_duplicate_faces}")
    
    # Подсчитываем кластеры с дубликатами
    clusters_with_duplicates = len(cluster_duplicates)
    print(f"Кластеров с дубликатами: {clusters_with_duplicates}")


def main():
    parser = argparse.ArgumentParser(description="Проверка дублей лиц в кластерах")
    parser.add_argument("--person-id", type=int, default=None, help="ID персоны для фильтрации (опционально)")
    
    args = parser.parse_args()
    
    check_duplicate_faces(person_id=args.person_id)


if __name__ == "__main__":
    main()
