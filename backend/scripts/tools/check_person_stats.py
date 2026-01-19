#!/usr/bin/env python3
"""
Скрипт для проверки статистики персоны: количество лиц и кластеров.

Использование:
    python backend/scripts/tools/check_person_stats.py --person-id 12
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def check_person_stats(person_id: int) -> None:
    """Проверяет статистику персоны."""
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
    
    # Подсчитываем количество лиц через face_labels
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as faces_count
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    
    faces_row = cur.fetchone()
    faces_count = faces_row["faces_count"] if faces_row else 0
    
    # Подсчитываем количество кластеров
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.cluster_id) as clusters_count
        FROM face_labels fl
        WHERE fl.person_id = ? AND fl.cluster_id IS NOT NULL
        """,
        (person_id,),
    )
    
    clusters_row = cur.fetchone()
    clusters_count = clusters_row["clusters_count"] if clusters_row else 0
    
    # Подсчитываем количество лиц в кластерах (через face_cluster_members)
    cur.execute(
        """
        SELECT COUNT(DISTINCT fcm.face_rectangle_id) as faces_in_clusters_count
        FROM face_labels fl
        JOIN face_cluster_members fcm ON fl.cluster_id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND fl.cluster_id IS NOT NULL
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    
    faces_in_clusters_row = cur.fetchone()
    faces_in_clusters_count = faces_in_clusters_row["faces_in_clusters_count"] if faces_in_clusters_row else 0
    
    # Подсчитываем количество лиц, которые есть и в face_labels, и в кластерах
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as faces_with_cluster_count
        FROM face_labels fl
        JOIN face_cluster_members fcm ON fl.face_rectangle_id = fcm.face_rectangle_id 
            AND fl.cluster_id = fcm.cluster_id
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND fl.cluster_id IS NOT NULL
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    
    faces_with_cluster_row = cur.fetchone()
    faces_with_cluster_count = faces_with_cluster_row["faces_with_cluster_count"] if faces_with_cluster_row else 0
    
    # Получаем список кластеров с количеством лиц
    cur.execute(
        """
        SELECT 
            fl.cluster_id,
            COUNT(DISTINCT fcm.face_rectangle_id) as faces_count
        FROM face_labels fl
        JOIN face_cluster_members fcm ON fl.cluster_id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND fl.cluster_id IS NOT NULL
          AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY fl.cluster_id
        ORDER BY faces_count DESC, fl.cluster_id
        """,
        (person_id,),
    )
    
    clusters_list = cur.fetchall()
    
    print(f"\nСтатистика:")
    print(f"  Лиц через face_labels: {faces_count}")
    print(f"  Кластеров: {clusters_count}")
    print(f"  Лиц в кластерах (через face_cluster_members): {faces_in_clusters_count}")
    print(f"  Лиц с корректной связью (face_labels + face_cluster_members): {faces_with_cluster_count}")
    
    if faces_count != faces_in_clusters_count:
        print(f"\n⚠ ВНИМАНИЕ: Несоответствие!")
        print(f"  Лиц в face_labels: {faces_count}")
        print(f"  Лиц в кластерах: {faces_in_clusters_count}")
        print(f"  Разница: {abs(faces_count - faces_in_clusters_count)}")
    
    print(f"\nКластеры ({len(clusters_list)}):")
    total_faces_in_clusters = 0
    for cluster_row in clusters_list:
        cluster_id = cluster_row["cluster_id"]
        cluster_faces = cluster_row["faces_count"]
        total_faces_in_clusters += cluster_faces
        print(f"  Кластер #{cluster_id}: {cluster_faces} лиц")
    
    print(f"\nИтого лиц в кластерах (сумма): {total_faces_in_clusters}")
    
    # Проверяем, есть ли лица без кластеров
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as faces_without_cluster
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND (fl.cluster_id IS NULL OR fl.cluster_id NOT IN (
              SELECT DISTINCT cluster_id FROM face_cluster_members
          ))
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    
    faces_without_cluster_row = cur.fetchone()
    faces_without_cluster = faces_without_cluster_row["faces_without_cluster"] if faces_without_cluster_row else 0
    
    if faces_without_cluster > 0:
        print(f"\n⚠ Лица без кластеров: {faces_without_cluster}")


def main():
    parser = argparse.ArgumentParser(description="Проверка статистики персоны")
    parser.add_argument("--person-id", type=int, required=True, help="ID персоны")
    
    args = parser.parse_args()
    
    check_person_stats(person_id=args.person_id)


if __name__ == "__main__":
    main()
