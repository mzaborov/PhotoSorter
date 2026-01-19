#!/usr/bin/env python3
"""
Скрипт для проверки рассинхронизации между face_labels и face_cluster_members.

Показывает случаи, когда:
1. В face_labels одно лицо находится в нескольких кластерах
2. В face_cluster_members это лицо находится только в одном кластере (или не находится вообще)
3. cluster_id в face_labels не совпадает с cluster_id в face_cluster_members

Использование:
    python backend/scripts/tools/check_face_labels_vs_cluster_members.py --person-id 1
    python backend/scripts/tools/check_face_labels_vs_cluster_members.py  # все персоны
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def check_sync(person_id: int | None = None) -> None:
    """
    Проверяет рассинхронизацию между face_labels и face_cluster_members.
    
    Args:
        person_id: опционально, фильтр по person_id
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие для фильтрации по персоне
    person_filter = ""
    person_params = []
    if person_id is not None:
        person_filter = "AND fl.person_id = ?"
        person_params = [person_id]
    
    # Находим лица, которые в face_labels находятся в нескольких кластерах
    sql_query = f"""
        SELECT 
            fl.face_rectangle_id,
            fl.person_id,
            COUNT(DISTINCT fl.cluster_id) as labels_cluster_count,
            GROUP_CONCAT(DISTINCT fl.cluster_id) as labels_cluster_ids,
            COUNT(DISTINCT fcm.cluster_id) as fcm_cluster_count,
            GROUP_CONCAT(DISTINCT fcm.cluster_id) as fcm_cluster_ids
        FROM face_labels fl
        LEFT JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        LEFT JOIN face_cluster_members fcm ON fl.face_rectangle_id = fcm.face_rectangle_id
        WHERE COALESCE(fr.ignore_flag, 0) = 0
        {person_filter}
        GROUP BY fl.face_rectangle_id, fl.person_id
        HAVING labels_cluster_count > 1 OR fcm_cluster_count != 1 OR fcm_cluster_count IS NULL
        ORDER BY labels_cluster_count DESC, fl.face_rectangle_id
    """
    
    cur.execute(sql_query, person_params)
    issues = cur.fetchall()
    
    if len(issues) == 0:
        if person_id is not None:
            print(f"Рассинхронизации для персоны {person_id} не найдено.")
        else:
            print("Рассинхронизации не найдено.")
        return
    
    print(f"Найдено проблем: {len(issues)}")
    if person_id is not None:
        print(f"Для персоны: {person_id}")
    print()
    print("=" * 100)
    print("Детальная информация:")
    print("=" * 100)
    print()
    
    # Группируем по типам проблем
    multiple_labels = []
    missing_in_fcm = []
    mismatch_cluster = []
    
    for issue in issues:
        face_id = issue["face_rectangle_id"]
        person_id_val = issue["person_id"]
        labels_count = issue["labels_cluster_count"]
        labels_clusters = issue["labels_cluster_ids"]
        fcm_count = issue["fcm_cluster_count"] or 0
        fcm_clusters = issue["fcm_cluster_ids"] or "NULL"
        
        if labels_count > 1:
            multiple_labels.append(issue)
        elif fcm_count == 0:
            missing_in_fcm.append(issue)
        else:
            mismatch_cluster.append(issue)
    
    # Показываем лица в нескольких кластерах в face_labels
    if multiple_labels:
        print(f"1. Лица в нескольких кластерах в face_labels ({len(multiple_labels)}):")
        print("-" * 100)
        for issue in multiple_labels[:10]:
            print(f"   Face ID {issue['face_rectangle_id']} (Person {issue['person_id']}):")
            print(f"     face_labels: {issue['labels_cluster_ids']} ({issue['labels_cluster_count']} кластеров)")
            print(f"     face_cluster_members: {issue['fcm_cluster_ids']} ({issue['fcm_cluster_count'] or 0} кластеров)")
        if len(multiple_labels) > 10:
            print(f"   ... и еще {len(multiple_labels) - 10} лиц")
        print()
    
    # Показываем лица, которых нет в face_cluster_members
    if missing_in_fcm:
        print(f"2. Лица, которых нет в face_cluster_members ({len(missing_in_fcm)}):")
        print("-" * 100)
        for issue in missing_in_fcm[:10]:
            print(f"   Face ID {issue['face_rectangle_id']} (Person {issue['person_id']}):")
            print(f"     face_labels: {issue['labels_cluster_ids']} ({issue['labels_cluster_count']} кластеров)")
            print(f"     face_cluster_members: отсутствует")
        if len(missing_in_fcm) > 10:
            print(f"   ... и еще {len(missing_in_fcm) - 10} лиц")
        print()
    
    # Показываем несоответствие кластеров
    if mismatch_cluster:
        print(f"3. Несоответствие кластеров ({len(mismatch_cluster)}):")
        print("-" * 100)
        for issue in mismatch_cluster[:10]:
            print(f"   Face ID {issue['face_rectangle_id']} (Person {issue['person_id']}):")
            print(f"     face_labels: {issue['labels_cluster_ids']}")
            print(f"     face_cluster_members: {issue['fcm_cluster_ids']}")
        if len(mismatch_cluster) > 10:
            print(f"   ... и еще {len(mismatch_cluster) - 10} лиц")
        print()
    
    # Статистика
    print("=" * 100)
    print("Статистика:")
    print("=" * 100)
    print(f"Всего проблемных лиц: {len(issues)}")
    print(f"  - В нескольких кластерах в face_labels: {len(multiple_labels)}")
    print(f"  - Отсутствуют в face_cluster_members: {len(missing_in_fcm)}")
    print(f"  - Несоответствие кластеров: {len(mismatch_cluster)}")
    
    # Показываем примеры записей face_labels для проблемных лиц
    if multiple_labels:
        print()
        print("=" * 100)
        print("Примеры записей face_labels для лиц в нескольких кластерах:")
        print("=" * 100)
        example_face_id = multiple_labels[0]["face_rectangle_id"]
        cur.execute("""
            SELECT id, face_rectangle_id, person_id, cluster_id, source, confidence, created_at
            FROM face_labels
            WHERE face_rectangle_id = ?
            ORDER BY created_at DESC
        """, (example_face_id,))
        example_labels = cur.fetchall()
        print(f"\nFace ID {example_face_id}:")
        for label in example_labels:
            print(f"  ID={label['id']}, cluster_id={label['cluster_id']}, person_id={label['person_id']}, "
                  f"source={label['source']}, created_at={label['created_at']}")


def main():
    parser = argparse.ArgumentParser(description="Проверка рассинхронизации face_labels и face_cluster_members")
    parser.add_argument("--person-id", type=int, default=None, help="ID персоны для фильтрации (опционально)")
    
    args = parser.parse_args()
    
    check_sync(person_id=args.person_id)


if __name__ == "__main__":
    main()
