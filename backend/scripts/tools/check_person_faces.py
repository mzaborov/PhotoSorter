#!/usr/bin/env python3
"""
Скрипт для проверки количества кластеров и лиц у персоны.

Использование:
    python backend/scripts/tools/check_person_faces.py --person-id 12
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def check_person_faces(person_id: int) -> None:
    """Проверяет количество кластеров и лиц у персоны."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем информацию о персоне
    cur.execute(
        """
        SELECT id, name
        FROM persons
        WHERE id = ?
        """,
        (person_id,),
    )
    
    person_row = cur.fetchone()
    if not person_row:
        print(f"Персона с ID {person_id} не найдена.")
        return
    
    person_name = person_row["name"]
    print(f"Персона: {person_name} (ID: {person_id})")
    print("=" * 60)
    
    # Подсчитываем количество кластеров
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.cluster_id) as clusters_count
        FROM face_labels fl
        JOIN face_cluster_members fcm ON fl.cluster_id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    
    clusters_row = cur.fetchone()
    clusters_count = clusters_row["clusters_count"] if clusters_row else 0
    
    # Подсчитываем количество лиц
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as faces_count
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    
    faces_row = cur.fetchone()
    faces_count = faces_row["faces_count"] if faces_row else 0
    
    print(f"Кластеров: {clusters_count}")
    print(f"Лиц: {faces_count}")
    print()
    
    # Показываем детали по кластерам
    cur.execute(
        """
        SELECT 
            fl.cluster_id,
            COUNT(DISTINCT fl.face_rectangle_id) as faces_in_cluster
        FROM face_labels fl
        JOIN face_cluster_members fcm ON fl.cluster_id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY fl.cluster_id
        ORDER BY fl.cluster_id
        """,
        (person_id,),
    )
    
    print("Детали по кластерам:")
    print("-" * 60)
    total_faces_in_clusters = 0
    for row in cur.fetchall():
        cluster_id = row["cluster_id"]
        faces_in_cluster = row["faces_in_cluster"]
        total_faces_in_clusters += faces_in_cluster
        print(f"  Кластер #{cluster_id}: {faces_in_cluster} лиц")
    
    print("-" * 60)
    print(f"Итого лиц в кластерах: {total_faces_in_clusters}")
    print()
    
    # Проверяем, есть ли лица без кластеров
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as faces_without_cluster
        FROM face_labels fl
        LEFT JOIN face_cluster_members fcm ON fl.face_rectangle_id = fcm.face_rectangle_id
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND fcm.face_rectangle_id IS NULL
        """,
        (person_id,),
    )
    
    without_cluster_row = cur.fetchone()
    faces_without_cluster = without_cluster_row["faces_without_cluster"] if without_cluster_row else 0
    
    if faces_without_cluster > 0:
        print(f"⚠ Лиц без кластеров: {faces_without_cluster}")
    
    # Проверяем дубликаты (лица в нескольких кластерах)
    cur.execute(
        """
        SELECT 
            fl.face_rectangle_id,
            COUNT(DISTINCT fl.cluster_id) as cluster_count,
            GROUP_CONCAT(DISTINCT fl.cluster_id) as cluster_ids
        FROM face_labels fl
        JOIN face_cluster_members fcm ON fl.cluster_id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY fl.face_rectangle_id
        HAVING cluster_count > 1
        """,
        (person_id,),
    )
    
    duplicates = cur.fetchall()
    if duplicates:
        print(f"⚠ Лиц в нескольких кластерах: {len(duplicates)}")
        for dup in duplicates[:10]:  # Показываем первые 10
            print(f"  Face ID {dup['face_rectangle_id']}: в кластерах {dup['cluster_ids']}")
        if len(duplicates) > 10:
            print(f"  ... и еще {len(duplicates) - 10} лиц")


def main():
    parser = argparse.ArgumentParser(description="Проверка количества кластеров и лиц у персоны")
    parser.add_argument("--person-id", type=int, required=True, help="ID персоны")
    
    args = parser.parse_args()
    
    check_person_faces(person_id=args.person_id)


if __name__ == "__main__":
    main()
