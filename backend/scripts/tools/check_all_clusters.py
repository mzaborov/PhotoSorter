#!/usr/bin/env python3
"""
Проверка всех кластеров в БД и их связи с персонами.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    conn = get_connection()
    cur = conn.cursor()
    
    print("=" * 80)
    print("Проверка всех кластеров в БД")
    print("=" * 80)
    print()
    
    # 1. Общая статистика по кластерам
    print("1. Общая статистика:")
    cur.execute("""
        SELECT 
            COUNT(*) AS total_clusters,
            COUNT(DISTINCT run_id) AS unique_runs,
            COUNT(DISTINCT archive_scope) AS unique_scopes
        FROM face_clusters
    """)
    stats = cur.fetchone()
    print(f"   Всего кластеров в БД: {stats['total_clusters'] or 0}")
    print(f"   Кластеров из разных run_id: {stats['unique_runs'] or 0}")
    print(f"   Кластеров из разных archive_scope: {stats['unique_scopes'] or 0}")
    print()
    
    # 2. Кластеры по run_id
    print("2. Кластеры по run_id:")
    cur.execute("""
        SELECT 
            run_id,
            COUNT(*) AS clusters_count,
            COUNT(DISTINCT fcm.face_rectangle_id) AS faces_count
        FROM face_clusters fc
        LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        WHERE run_id IS NOT NULL
        GROUP BY run_id
        ORDER BY run_id DESC
    """)
    runs = cur.fetchall()
    for row in runs:
        print(f"   run_id={row['run_id']}: {row['clusters_count']} кластеров, {row['faces_count']} лиц")
    print()
    
    # 3. Кластеры по archive_scope
    print("3. Кластеры по archive_scope:")
    cur.execute("""
        SELECT 
            archive_scope,
            COUNT(*) AS clusters_count,
            COUNT(DISTINCT fcm.face_rectangle_id) AS faces_count
        FROM face_clusters fc
        LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        WHERE archive_scope IS NOT NULL
        GROUP BY archive_scope
    """)
    scopes = cur.fetchall()
    if scopes:
        for row in scopes:
            print(f"   archive_scope={row['archive_scope']}: {row['clusters_count']} кластеров, {row['faces_count']} лиц")
    else:
        print("   Нет кластеров с archive_scope")
    print()
    
    # 4. Связь кластеров с персонами
    print("4. Связь кластеров с персонами:")
    cur.execute("""
        SELECT 
            COUNT(DISTINCT fc.id) AS clusters_with_person,
            COUNT(DISTINCT fl.person_id) AS unique_persons,
            COUNT(DISTINCT fcm.face_rectangle_id) AS faces_with_person
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        JOIN face_labels fl ON fl.face_rectangle_id = fcm.face_rectangle_id
    """)
    person_stats = cur.fetchone()
    clusters_with_person = person_stats['clusters_with_person'] or 0
    unique_persons = person_stats['unique_persons'] or 0
    faces_with_person = person_stats['faces_with_person'] or 0
    
    cur.execute("SELECT COUNT(*) FROM face_clusters")
    total_clusters = cur.fetchone()[0] or 0
    
    print(f"   Кластеров с привязанными персонами: {clusters_with_person} из {total_clusters}")
    print(f"   Уникальных персон в кластерах: {unique_persons}")
    print(f"   Лиц с привязанными персонами: {faces_with_person}")
    print()
    
    # 5. Топ персон по количеству лиц
    if unique_persons > 0:
        print("5. Топ персон по количеству лиц:")
        cur.execute("""
            SELECT 
                p.id,
                p.name,
                COUNT(DISTINCT fcm.face_rectangle_id) AS faces_count,
                COUNT(DISTINCT fc.id) AS clusters_count
            FROM persons p
            JOIN face_labels fl ON fl.person_id = p.id
            JOIN face_cluster_members fcm ON fcm.face_rectangle_id = fl.face_rectangle_id
            JOIN face_clusters fc ON fc.id = fcm.cluster_id
            GROUP BY p.id, p.name
            ORDER BY faces_count DESC
            LIMIT 10
        """)
        persons = cur.fetchall()
        for i, row in enumerate(persons, 1):
            print(f"   {i}. {row['name']} (ID: {row['id']}): {row['faces_count']} лиц в {row['clusters_count']} кластерах")
        print()
    
    # 6. Структура связи
    print("6. Структура связи кластеров с персонами:")
    print("   face_clusters -> face_cluster_members -> face_rectangle_id")
    print("   face_rectangle_id -> face_labels -> person_id -> persons")
    print("   (Один кластер может содержать много лиц, все лица кластера привязываются к одной персоне)")
    print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
