#!/usr/bin/env python3
"""
Анализ распределения кластеров и лиц по персонам.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Анализ распределения по персонам")
    parser.add_argument("--pipeline-run-id", type=int, default=26, help="ID прогона pipeline")
    args = parser.parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем face_run_id
    cur.execute("SELECT face_run_id FROM pipeline_runs WHERE id = ?", (args.pipeline_run_id,))
    run_info = cur.fetchone()
    if not run_info or not run_info['face_run_id']:
        print(f"❌ Прогон {args.pipeline_run_id} не найден или нет face_run_id!")
        return
    
    face_run_id = run_info['face_run_id']
    
    print("=" * 80)
    print(f"Распределение кластеров и лиц по персонам для прогона {args.pipeline_run_id} (face_run_id={face_run_id})")
    print("=" * 80)
    print()
    
    # 1. Общая статистика по кластерам
    print("1. Общая статистика по кластерам:")
    cur.execute("""
        SELECT 
            COUNT(DISTINCT fc.id) AS total_clusters,
            COUNT(DISTINCT fcm.face_rectangle_id) AS total_faces_in_clusters
        FROM face_clusters fc
        LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        WHERE fc.run_id = ?
    """, (face_run_id,))
    
    stats = cur.fetchone()
    total_clusters = stats['total_clusters'] or 0
    total_faces = stats['total_faces_in_clusters'] or 0
    print(f"   Всего кластеров: {total_clusters}")
    print(f"   Всего лиц в кластерах: {total_faces}")
    print()
    
    # 2. Кластеры с привязкой к персонам
    print("2. Кластеры с привязкой к персонам:")
    # Кластер считается привязанным к персоне, если хотя бы одно лицо в кластере имеет face_labels для этой персоны
    # Считаем только кластеры, которые содержат лица из ТЕКУЩЕГО прогона
    cur.execute("""
        SELECT 
            COUNT(DISTINCT fc.id) AS clusters_with_person
        FROM face_clusters fc
        WHERE (fc.run_id = ? OR fc.archive_scope = 'archive')
          AND EXISTS (
              SELECT 1
              FROM face_cluster_members fcm
              JOIN face_labels fl ON fl.face_rectangle_id = fcm.face_rectangle_id
              JOIN face_rectangles fr ON fr.id = fcm.face_rectangle_id
              WHERE fcm.cluster_id = fc.id
                AND fr.run_id = ?
          )
    """, (face_run_id, face_run_id))
    
    clusters_with_person = cur.fetchone()['clusters_with_person'] or 0
    # Кластеры без персон - только те, которые содержат лица из текущего прогона
    cur.execute("""
        SELECT COUNT(DISTINCT fc.id) AS clusters_without_person
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        JOIN face_rectangles fr ON fr.id = fcm.face_rectangle_id
        WHERE (fc.run_id = ? OR fc.archive_scope = 'archive')
          AND fr.run_id = ?
          AND NOT EXISTS (
              SELECT 1
              FROM face_labels fl
              WHERE fl.face_rectangle_id = fcm.face_rectangle_id
          )
    """, (face_run_id, face_run_id))
    
    clusters_without_person = cur.fetchone()['clusters_without_person'] or 0
    print(f"   Кластеров с привязанными персонами: {clusters_with_person}")
    print(f"   Кластеров без персон (с лицами из текущего прогона): {clusters_without_person}")
    print()
    
    # 3. Распределение по персонам (по файлам)
    print("3. Распределение файлов по персонам:")
    # Считаем файлы из ТЕКУЩЕГО прогона, где хотя бы одно лицо привязано к персоне
    # Файл считается привязанным к персоне, если:
    # 1. Лицо имеет face_labels для персоны (прямая привязка)
    # 2. ИЛИ лицо в кластере, где есть другие лица с face_labels для этой персоны (через JOIN)
    cur.execute("""
        SELECT 
            p.id AS person_id,
            p.name AS person_name,
            p."group" AS person_group,
            COUNT(DISTINCT fr.file_path) AS files_count,
            COUNT(DISTINCT fr.id) AS faces_count
        FROM persons p
        JOIN face_labels fl ON fl.person_id = p.id
        -- Находим кластеры, где есть лица с face_labels для этой персоны
        JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl.face_rectangle_id
        JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
        -- Находим ВСЕ лица в этих кластерах (включая новые, без face_labels)
        JOIN face_cluster_members fcm_all ON fcm_all.cluster_id = fc.id
        JOIN face_rectangles fr ON fr.id = fcm_all.face_rectangle_id
        WHERE fr.run_id = ?
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND (fc.run_id = ? OR fc.archive_scope = 'archive')
        GROUP BY p.id, p.name, p."group"
        ORDER BY files_count DESC, p.name
    """, (face_run_id, face_run_id))
    
    persons = cur.fetchall()
    
    if len(persons) == 0:
        print("   Нет файлов, привязанных к персонам")
    else:
        print(f"   Всего персон с привязанными файлами: {len(persons)}")
        print()
        print("   Распределение по файлам:")
        print(f"   {'Персона':<30} {'Группа':<20} {'Файлов':<12} {'Лиц':<10}")
        print("   " + "-" * 72)
        for person in persons:
            person_name = person['person_name'] or '(без имени)'
            person_group = person['person_group'] or '(без группы)'
            files_count = person['files_count'] or 0
            faces_count = person['faces_count'] or 0
            print(f"   {person_name:<30} {person_group:<20} {files_count:<12} {faces_count:<10}")
    print()
    
    # 4. Кластеры без персон (топ-10 по размеру)
    print("4. Кластеры без привязки к персонам (топ-10 по размеру):")
    cur.execute("""
        SELECT 
            fc.id AS cluster_id,
            COUNT(fcm.face_rectangle_id) AS faces_count
        FROM face_clusters fc
        LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        LEFT JOIN face_labels fl ON fl.face_rectangle_id = fcm.face_rectangle_id
        WHERE fc.run_id = ?
          AND fl.id IS NULL
        GROUP BY fc.id
        ORDER BY faces_count DESC
        LIMIT 10
    """, (face_run_id,))
    
    unassigned_clusters = cur.fetchall()
    
    if len(unassigned_clusters) == 0:
        print("   Все кластеры привязаны к персонам")
    else:
        print(f"   Найдено {len(unassigned_clusters)} кластеров без персон (показаны топ-10):")
        for cluster in unassigned_clusters:
            print(f"   Кластер {cluster['cluster_id']}: {cluster['faces_count']} лиц")
    print()
    
    # 5. Лица без кластеров (шум)
    print("5. Лица без кластеров (шум):")
    cur.execute("""
        SELECT COUNT(*) AS noise_count
        FROM face_rectangles fr
        LEFT JOIN face_cluster_members fcm ON fcm.face_rectangle_id = fr.id
        WHERE fr.run_id = ?
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND fcm.face_rectangle_id IS NULL
    """, (face_run_id,))
    
    noise_count = cur.fetchone()['noise_count'] or 0
    print(f"   Лиц без кластеров: {noise_count}")
    print()
    
    # 6. Итоговая статистика
    print("6. Итоговая статистика:")
    total_faces_in_run = cur.execute("""
        SELECT COUNT(*) FROM face_rectangles 
        WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
    """, (face_run_id,)).fetchone()[0] or 0
    
    faces_in_clusters = total_faces
    faces_assigned_to_persons = sum(p['faces_count'] for p in persons) if persons else 0
    files_assigned_to_persons = sum(p['files_count'] for p in persons) if persons else 0
    
    # Подсчет файлов в прогоне
    cur.execute("""
        SELECT COUNT(DISTINCT file_path) AS total_files
        FROM face_rectangles 
        WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
    """, (face_run_id,))
    total_files_in_run = cur.fetchone()['total_files'] or 0
    
    print(f"   Всего лиц в прогоне: {total_faces_in_run}")
    print(f"   Всего файлов с лицами: {total_files_in_run}")
    print(f"   Лиц в кластерах: {faces_in_clusters} ({faces_in_clusters/total_faces_in_run*100:.1f}%)")
    print(f"   Файлов привязано к персонам: {files_assigned_to_persons} ({files_assigned_to_persons/total_files_in_run*100:.1f}% от файлов)")
    print(f"   Лиц привязано к персонам: {faces_assigned_to_persons} ({faces_assigned_to_persons/total_faces_in_run*100:.1f}% от лиц)")
    print(f"   Лиц без кластеров (шум): {noise_count} ({noise_count/total_faces_in_run*100:.1f}%)")
    print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
