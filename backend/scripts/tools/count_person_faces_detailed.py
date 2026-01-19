#!/usr/bin/env python3
"""
Скрипт для детального подсчета лиц персоны разными способами.

Показывает, сколько лиц считается разными методами:
1. COUNT(DISTINCT face_rectangle_id) из face_labels
2. COUNT(DISTINCT face_rectangle_id) из face_labels с фильтром ignore_flag
3. Количество записей в face_labels (может быть больше из-за дубликатов)
4. Количество записей в face_labels для разных кластеров

Использование:
    python backend/scripts/tools/count_person_faces_detailed.py --person-id 1
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def count_person_faces_detailed(person_id: int) -> None:
    """Детальный подсчет лиц персоны разными способами."""
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
    print("=" * 80)
    print()
    
    # Метод 1: COUNT(DISTINCT face_rectangle_id) из face_labels (как в /api/persons/stats)
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as faces_count
        FROM face_labels fl
        LEFT JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    row1 = cur.fetchone()
    count1 = row1["faces_count"] if row1 else 0
    print(f"1. COUNT(DISTINCT face_rectangle_id) из face_labels (с фильтром ignore_flag=0): {count1}")
    print("   (Этот метод используется в /api/persons/stats)")
    print()
    
    # Метод 2: COUNT(DISTINCT face_rectangle_id) из face_labels БЕЗ фильтра ignore_flag
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as faces_count
        FROM face_labels fl
        WHERE fl.person_id = ?
        """,
        (person_id,),
    )
    row2 = cur.fetchone()
    count2 = row2["faces_count"] if row2 else 0
    print(f"2. COUNT(DISTINCT face_rectangle_id) из face_labels (БЕЗ фильтра ignore_flag): {count2}")
    print()
    
    # Метод 3: Количество записей в face_labels (может быть больше из-за дубликатов)
    cur.execute(
        """
        SELECT COUNT(*) as labels_count
        FROM face_labels fl
        LEFT JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    row3 = cur.fetchone()
    count3 = row3["labels_count"] if row3 else 0
    print(f"3. Количество записей в face_labels (с фильтром ignore_flag=0): {count3}")
    print(f"   Разница с методом 1: {count3 - count1} (это лица в нескольких кластерах)")
    print()
    
    # Метод 4: Проверяем, есть ли лица в нескольких кластерах
    cur.execute(
        """
        SELECT 
            fl.face_rectangle_id,
            COUNT(DISTINCT fl.cluster_id) as cluster_count,
            GROUP_CONCAT(DISTINCT fl.cluster_id) as cluster_ids
        FROM face_labels fl
        LEFT JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY fl.face_rectangle_id
        HAVING cluster_count > 1
        """,
        (person_id,),
    )
    faces_in_multiple_clusters = cur.fetchall()
    print(f"4. Лиц в нескольких кластерах: {len(faces_in_multiple_clusters)}")
    if faces_in_multiple_clusters:
        print("   Примеры:")
        for face in faces_in_multiple_clusters[:5]:
            print(f"     Face ID {face['face_rectangle_id']}: в кластерах {face['cluster_ids']}")
        if len(faces_in_multiple_clusters) > 5:
            print(f"     ... и еще {len(faces_in_multiple_clusters) - 5} лиц")
    print()
    
    # Метод 5: Проверяем лица с ignore_flag=1
    cur.execute(
        """
        SELECT COUNT(DISTINCT fl.face_rectangle_id) as ignored_faces_count
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND fr.ignore_flag = 1
        """,
        (person_id,),
    )
    row5 = cur.fetchone()
    count5 = row5["ignored_faces_count"] if row5 else 0
    print(f"5. Лиц с ignore_flag=1: {count5}")
    print()
    
    # Итоговая статистика
    print("=" * 80)
    print("ИТОГО:")
    print(f"  Уникальных лиц (метод 1, используется в интерфейсе): {count1}")
    print(f"  Уникальных лиц (включая ignore): {count2}")
    print(f"  Записей face_labels: {count3}")
    print(f"  Лиц в нескольких кластерах: {len(faces_in_multiple_clusters)}")
    print(f"  Лиц с ignore_flag=1: {count5}")
    print()
    print(f"  Разница между записями и уникальными: {count3 - count1} (лица в нескольких кластерах)")


def main():
    parser = argparse.ArgumentParser(description="Детальный подсчет лиц персоны")
    parser.add_argument("--person-id", type=int, required=True, help="ID персоны")
    
    args = parser.parse_args()
    
    count_person_faces_detailed(person_id=args.person_id)


if __name__ == "__main__":
    main()
