#!/usr/bin/env python3
"""
Анализ привязки лиц к персонам и статистика по видео для прогона.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Анализ персон и видео для прогона")
    parser.add_argument("--pipeline-run-id", type=int, default=26, help="ID прогона pipeline")
    args = parser.parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем информацию о прогоне
    cur.execute("""
        SELECT id, face_run_id, root_path, status
        FROM pipeline_runs
        WHERE id = ?
    """, (args.pipeline_run_id,))
    
    run_info = cur.fetchone()
    if not run_info:
        print(f"❌ Прогон {args.pipeline_run_id} не найден!")
        return
    
    face_run_id = run_info['face_run_id']
    if not face_run_id:
        print(f"❌ У прогона {args.pipeline_run_id} нет face_run_id!")
        return
    
    print("=" * 80)
    print(f"Анализ персон и видео для прогона {args.pipeline_run_id}")
    print(f"  Face run ID: {face_run_id}")
    print(f"  Root path: {run_info['root_path']}")
    print("=" * 80)
    print()
    
    # 1. Общая статистика по лицам
    print("1. Общая статистика по лицам:")
    cur.execute("""
        SELECT 
            COUNT(*) AS total_faces,
            COUNT(DISTINCT file_path) AS total_files_with_faces
        FROM face_rectangles
        WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
    """, (face_run_id,))
    
    faces_stats = cur.fetchone()
    print(f"   Всего лиц: {faces_stats['total_faces']}")
    print(f"   Всего файлов с лицами: {faces_stats['total_files_with_faces']}")
    print()
    
    # 2. Статистика по привязке к персонам
    print("2. Статистика по привязке лиц к персонам:")
    cur.execute("""
        SELECT 
            COUNT(DISTINCT fl.face_rectangle_id) AS faces_with_person,
            COUNT(DISTINCT fl.person_id) AS unique_persons
        FROM face_labels fl
        JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
        WHERE fr.run_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
    """, (face_run_id,))
    
    persons_stats = cur.fetchone()
    faces_with_person = persons_stats['faces_with_person'] or 0
    unique_persons = persons_stats['unique_persons'] or 0
    faces_without_person = faces_stats['total_faces'] - faces_with_person
    
    print(f"   Лиц привязано к персонам: {faces_with_person}")
    print(f"   Лиц без привязки к персонам: {faces_without_person}")
    print(f"   Уникальных персон в прогоне: {unique_persons}")
    if faces_stats['total_faces'] > 0:
        percent = (faces_with_person / faces_stats['total_faces']) * 100
        print(f"   Процент привязанных лиц: {percent:.1f}%")
    print()
    
    # 3. Статистика по персонам (топ персоны)
    print("3. Топ персоны по количеству лиц:")
    cur.execute("""
        SELECT 
            p.id,
            p.name,
            COUNT(DISTINCT fl.face_rectangle_id) AS faces_count,
            COUNT(DISTINCT fr.file_path) AS files_count
        FROM persons p
        JOIN face_labels fl ON fl.person_id = p.id
        JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
        WHERE fr.run_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY p.id, p.name
        ORDER BY faces_count DESC
        LIMIT 20
    """, (face_run_id,))
    
    top_persons = cur.fetchall()
    if top_persons:
        for i, row in enumerate(top_persons, 1):
            print(f"   {i}. {row['name']} (ID: {row['id']}): {row['faces_count']} лиц в {row['files_count']} файлах")
    else:
        print("   Нет персон с привязанными лицами")
    print()
    
    # 4. Статистика по видео
    print("4. Статистика по видео файлам:")
    cur.execute("""
        SELECT 
            COUNT(DISTINCT f.path) AS total_video_files,
            COUNT(DISTINCT CASE WHEN fr.id IS NOT NULL THEN f.path END) AS videos_with_faces,
            COUNT(DISTINCT fr.id) AS faces_in_videos
        FROM files f
        LEFT JOIN face_rectangles fr ON fr.file_path = f.path 
            AND fr.run_id = ? 
            AND COALESCE(fr.ignore_flag, 0) = 0
        WHERE f.faces_run_id = ?
          AND f.status != 'deleted'
          AND (
            f.path LIKE '%.mp4' 
            OR f.path LIKE '%.mov' 
            OR f.path LIKE '%.mkv' 
            OR f.path LIKE '%.avi' 
            OR f.path LIKE '%.wmv' 
            OR f.path LIKE '%.m4v' 
            OR f.path LIKE '%.webm' 
            OR f.path LIKE '%.3gp'
          )
    """, (face_run_id, face_run_id))
    
    video_stats = cur.fetchone()
    total_videos = video_stats['total_video_files'] or 0
    videos_with_faces = video_stats['videos_with_faces'] or 0
    faces_in_videos = video_stats['faces_in_videos'] or 0
    
    print(f"   Всего видео файлов в прогоне: {total_videos}")
    print(f"   Видео с обнаруженными лицами: {videos_with_faces}")
    print(f"   Всего лиц в видео: {faces_in_videos}")
    if total_videos > 0:
        percent = (videos_with_faces / total_videos) * 100
        print(f"   Процент видео с лицами: {percent:.1f}%")
    print()
    
    # 5. Примеры видео с лицами
    if videos_with_faces > 0:
        print("5. Примеры видео с обнаруженными лицами (первые 10):")
        cur.execute("""
            SELECT DISTINCT fr.file_path, COUNT(*) AS faces_count
            FROM face_rectangles fr
            WHERE fr.run_id = ?
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND (
                fr.file_path LIKE '%.mp4' 
                OR fr.file_path LIKE '%.mov' 
                OR fr.file_path LIKE '%.mkv' 
                OR fr.file_path LIKE '%.avi' 
                OR fr.file_path LIKE '%.wmv' 
                OR fr.file_path LIKE '%.m4v' 
                OR fr.file_path LIKE '%.webm' 
                OR fr.file_path LIKE '%.3gp'
              )
            GROUP BY fr.file_path
            ORDER BY faces_count DESC
            LIMIT 10
        """, (face_run_id,))
        
        video_examples = cur.fetchall()
        for i, row in enumerate(video_examples, 1):
            print(f"   {i}. {row['file_path']}: {row['faces_count']} лиц")
        print()
    
    # 6. Сводная таблица
    print("6. Сводная таблица:")
    print(f"   Всего лиц: {faces_stats['total_faces']}")
    print(f"   ├─ Привязано к персонам: {faces_with_person} ({((faces_with_person/faces_stats['total_faces'])*100):.1f}%)")
    print(f"   └─ Без привязки: {faces_without_person} ({((faces_without_person/faces_stats['total_faces'])*100):.1f}%)")
    print()
    print(f"   Всего файлов с лицами: {faces_stats['total_files_with_faces']}")
    print(f"   ├─ Видео файлов: {total_videos}")
    print(f"   │  └─ Видео с лицами: {videos_with_faces}")
    print(f"   └─ Остальные (изображения): {faces_stats['total_files_with_faces'] - videos_with_faces}")
    print()
    
    print("=" * 80)
    print("Анализ завершен")
    print("=" * 80)

if __name__ == "__main__":
    main()
