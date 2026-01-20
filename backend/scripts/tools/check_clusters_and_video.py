#!/usr/bin/env python3
"""
Проверка кластеров и видео для прогона.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Проверка кластеров и видео")
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
    print(f"Проверка кластеров и видео для прогона {args.pipeline_run_id} (face_run_id={face_run_id})")
    print("=" * 80)
    print()
    
    # 1. Проверка кластеров
    print("1. Кластеры лиц:")
    cur.execute("""
        SELECT 
            COUNT(*) AS total_clusters,
            COUNT(DISTINCT fcm.face_rectangle_id) AS faces_in_clusters,
            COUNT(DISTINCT fcm.cluster_id) AS unique_clusters
        FROM face_clusters fc
        LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        WHERE fc.run_id = ?
    """, (face_run_id,))
    
    clusters_stats = cur.fetchone()
    total_clusters = clusters_stats['total_clusters'] or 0
    faces_in_clusters = clusters_stats['faces_in_clusters'] or 0
    
    print(f"   Всего кластеров: {total_clusters}")
    print(f"   Лиц в кластерах: {faces_in_clusters}")
    print()
    
    if total_clusters > 0:
        print("   Примеры кластеров (первые 10):")
        cur.execute("""
            SELECT 
                fc.id,
                COUNT(fcm.face_rectangle_id) AS faces_count
            FROM face_clusters fc
            LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
            WHERE fc.run_id = ?
            GROUP BY fc.id
            ORDER BY faces_count DESC
            LIMIT 10
        """, (face_run_id,))
        
        clusters = cur.fetchall()
        for i, row in enumerate(clusters, 1):
            print(f"   {i}. Кластер {row['id']}: {row['faces_count']} лиц")
        print()
    
    # 2. Проверка привязки кластеров к персонам
    print("2. Привязка кластеров к персонам:")
    cur.execute("""
        SELECT 
            COUNT(DISTINCT fc.id) AS clusters_with_person
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        JOIN face_labels fl ON fl.face_rectangle_id = fcm.face_rectangle_id
        WHERE fc.run_id = ?
    """, (face_run_id,))
    
    clusters_with_person = cur.fetchone()['clusters_with_person'] or 0
    print(f"   Кластеров с привязанными персонами: {clusters_with_person}")
    print(f"   Кластеров без персон: {total_clusters - clusters_with_person}")
    print()
    
    # 3. Проверка видео файлов в БД
    print("3. Видео файлы в БД:")
    cur.execute("""
        SELECT 
            COUNT(*) AS total_video_files
        FROM files
        WHERE faces_run_id = ?
          AND status != 'deleted'
          AND (
            path LIKE '%.mp4' 
            OR path LIKE '%.mov' 
            OR path LIKE '%.mkv' 
            OR path LIKE '%.avi' 
            OR path LIKE '%.wmv' 
            OR path LIKE '%.m4v' 
            OR path LIKE '%.webm' 
            OR path LIKE '%.3gp'
            OR path LIKE '%.MP4'
            OR path LIKE '%.MOV'
            OR path LIKE '%.MKV'
            OR path LIKE '%.AVI'
        )
    """, (face_run_id,))
    
    video_files = cur.fetchone()['total_video_files'] or 0
    print(f"   Всего видео файлов в БД: {video_files}")
    
    # Проверяем, есть ли face_rectangles для видео
    cur.execute("""
        SELECT 
            COUNT(DISTINCT fr.file_path) AS videos_with_faces,
            COUNT(fr.id) AS faces_in_videos
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
            OR fr.file_path LIKE '%.MP4'
            OR fr.file_path LIKE '%.MOV'
            OR fr.file_path LIKE '%.MKV'
            OR fr.file_path LIKE '%.AVI'
          )
    """, (face_run_id,))
    
    video_faces = cur.fetchone()
    videos_with_faces = video_faces['videos_with_faces'] or 0
    faces_in_videos = video_faces['faces_in_videos'] or 0
    
    print(f"   Видео с обнаруженными лицами: {videos_with_faces}")
    print(f"   Всего лиц в видео: {faces_in_videos}")
    if video_files > 0:
        print(f"   Процент видео с лицами: {(videos_with_faces/video_files)*100:.1f}%")
    print()
    
    # 4. Проверка embeddings
    print("4. Embeddings (для кластеризации):")
    cur.execute("""
        SELECT 
            COUNT(*) AS total_faces,
            COUNT(embedding) AS faces_with_embedding
        FROM face_rectangles
        WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
    """, (face_run_id,))
    
    embeddings_stats = cur.fetchone()
    total_faces = embeddings_stats['total_faces'] or 0
    faces_with_embedding = embeddings_stats['faces_with_embedding'] or 0
    
    print(f"   Всего лиц: {total_faces}")
    print(f"   Лиц с embeddings: {faces_with_embedding}")
    if total_faces > 0:
        print(f"   Процент с embeddings: {(faces_with_embedding/total_faces)*100:.1f}%")
    print()
    
    # 5. Выводы
    print("5. Выводы:")
    if total_clusters == 0:
        print("   ⚠️  Кластеризация НЕ была выполнена или завершилась с ошибкой")
        print("      Нужно запустить кластеризацию вручную")
    else:
        print(f"   ✓ Кластеризация выполнена: {total_clusters} кластеров, {faces_in_clusters} лиц")
    
    if video_files > 0 and videos_with_faces == 0:
        print(f"   ⚠️  Видео файлы есть в БД ({video_files}), но лица в них не обнаружены")
        print("      Возможно, видео не обрабатывались (video_samples=0 по умолчанию)")
    elif video_files == 0:
        print("   ℹ️  Видео файлов в прогоне нет")
    else:
        print(f"   ✓ Видео обработаны: {videos_with_faces} из {video_files} с лицами")
    
    if faces_with_embedding < total_faces * 0.9:
        print(f"   ⚠️  Не все лица имеют embeddings (только {faces_with_embedding} из {total_faces})")
        print("      Это может быть проблемой для кластеризации")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
