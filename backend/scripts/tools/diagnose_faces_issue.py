#!/usr/bin/env python3
"""
Диагностика проблемы детекции лиц для прогона 26.
Проверяет расхождение между files.faces_count и face_rectangles.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    pipeline_run_id = 26
    face_run_id = 27
    
    conn = get_connection()
    cur = conn.cursor()
    
    print("=" * 80)
    print(f"Диагностика проблемы детекции лиц для pipeline_run_id={pipeline_run_id}, face_run_id={face_run_id}")
    print("=" * 80)
    print()
    
    # 1. Проверяем информацию о прогоне
    print("1. Информация о прогоне:")
    cur.execute("""
        SELECT 
            pr.id AS pipeline_run_id,
            pr.face_run_id,
            pr.root_path,
            pr.status,
            pr.started_at,
            pr.finished_at,
            (SELECT COUNT(*) FROM face_rectangles WHERE run_id = pr.face_run_id AND COALESCE(ignore_flag, 0) = 0) AS faces_count,
            (SELECT COUNT(DISTINCT file_path) FROM face_rectangles WHERE run_id = pr.face_run_id AND COALESCE(ignore_flag, 0) = 0) AS files_with_faces_count
        FROM pipeline_runs pr
        WHERE pr.id = ?
    """, (pipeline_run_id,))
    
    run_info = cur.fetchone()
    if run_info:
        print(f"   Pipeline run ID: {run_info['pipeline_run_id']}")
        print(f"   Face run ID: {run_info['face_run_id']}")
        print(f"   Root path: {run_info['root_path']}")
        print(f"   Status: {run_info['status']}")
        print(f"   Started at: {run_info['started_at']}")
        print(f"   Finished at: {run_info['finished_at']}")
        print(f"   Face rectangles count: {run_info['faces_count']}")
        print(f"   Files with faces (from face_rectangles): {run_info['files_with_faces_count']}")
    else:
        print(f"   [WARNING] Прогон {pipeline_run_id} не найден!")
        return
    print()
    
    # 2. Сравнение files.faces_count и face_rectangles
    print("2. Сравнение files.faces_count и face_rectangles:")
    cur.execute("""
        SELECT 
            'files.faces_count > 0' AS source,
            COUNT(*) AS total_files,
            SUM(COALESCE(faces_count, 0)) AS total_faces_counted
        FROM files
        WHERE faces_run_id = ? AND COALESCE(faces_count, 0) > 0 AND status != 'deleted'
    """, (face_run_id,))
    
    files_stats = cur.fetchone()
    
    cur.execute("""
        SELECT 
            'face_rectangles' AS source,
            COUNT(DISTINCT file_path) AS total_files,
            COUNT(*) AS total_faces_counted
        FROM face_rectangles
        WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
    """, (face_run_id,))
    
    rectangles_stats = cur.fetchone()
    
    print(f"   Файлы с faces_count > 0:")
    print(f"     - Всего файлов: {files_stats['total_files']}")
    print(f"     - Всего лиц (сумма faces_count): {files_stats['total_faces_counted']}")
    print()
    print(f"   Face rectangles в БД:")
    print(f"     - Всего файлов: {rectangles_stats['total_files']}")
    print(f"     - Всего лиц: {rectangles_stats['total_faces_counted']}")
    print()
    
    if files_stats['total_files'] != rectangles_stats['total_files']:
        diff = files_stats['total_files'] - rectangles_stats['total_files']
        print(f"   [WARNING] РАСХОЖДЕНИЕ: {diff} файлов имеют faces_count > 0, но нет записей в face_rectangles!")
    else:
        print(f"   [OK] Количество файлов совпадает")
    print()
    
    # 3. Файлы с faces_count, но без face_rectangles
    print("3. Файлы с faces_count > 0, но без face_rectangles (первые 20):")
    cur.execute("""
        SELECT f.path, f.faces_count, f.faces_scanned_at, f.faces_run_id
        FROM files f
        LEFT JOIN face_rectangles fr ON fr.file_path = f.path AND fr.run_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        WHERE f.faces_run_id = ? 
          AND COALESCE(f.faces_count, 0) > 0 
          AND f.status != 'deleted'
          AND fr.id IS NULL
        LIMIT 20
    """, (face_run_id, face_run_id))
    
    problem_files = cur.fetchall()
    if problem_files:
        print(f"   Найдено проблемных файлов (показано первых {len(problem_files)}):")
        for i, row in enumerate(problem_files, 1):
            print(f"   {i}. {row['path']}")
            print(f"      faces_count={row['faces_count']}, faces_scanned_at={row['faces_scanned_at']}, faces_run_id={row['faces_run_id']}")
    else:
        print("   [OK] Проблемных файлов не найдено")
    print()
    
    # 4. Файлы с faces_scanned_at, но без face_rectangles
    print("4. Файлы с faces_scanned_at, но без face_rectangles (первые 20):")
    cur.execute("""
        SELECT f.path, f.faces_count, f.faces_scanned_at, f.faces_run_id
        FROM files f
        LEFT JOIN face_rectangles fr ON fr.file_path = f.path AND fr.run_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        WHERE f.faces_run_id = ? 
          AND f.faces_scanned_at IS NOT NULL
          AND f.status != 'deleted'
          AND fr.id IS NULL
        LIMIT 20
    """, (face_run_id, face_run_id))
    
    scanned_without_rectangles = cur.fetchall()
    if scanned_without_rectangles:
        print(f"   Найдено файлов с faces_scanned_at, но без face_rectangles (показано первых {len(scanned_without_rectangles)}):")
        for i, row in enumerate(scanned_without_rectangles, 1):
            print(f"   {i}. {row['path']}")
            print(f"      faces_count={row['faces_count']}, faces_scanned_at={row['faces_scanned_at']}, faces_run_id={row['faces_run_id']}")
    else:
        print("   [OK] Все файлы с faces_scanned_at имеют face_rectangles")
    print()
    
    # 5. Статистика по faces_scanned_at
    print("5. Статистика по faces_scanned_at:")
    cur.execute("""
        SELECT 
            COUNT(*) AS total_files,
            COUNT(CASE WHEN faces_scanned_at IS NOT NULL THEN 1 END) AS files_with_scanned_at,
            COUNT(CASE WHEN faces_scanned_at IS NULL THEN 1 END) AS files_without_scanned_at
        FROM files
        WHERE faces_run_id = ? AND status != 'deleted'
    """, (face_run_id,))
    
    scan_stats = cur.fetchone()
    print(f"   Всего файлов в прогоне: {scan_stats['total_files']}")
    print(f"   С faces_scanned_at: {scan_stats['files_with_scanned_at']}")
    print(f"   Без faces_scanned_at: {scan_stats['files_without_scanned_at']}")
    print()
    
    # Дополнительная статистика: файлы в исключаемых папках
    cur.execute("""
        SELECT 
            COUNT(*) AS total_files_in_excluded_dirs
        FROM files
        WHERE faces_run_id = ? 
          AND status != 'deleted'
          AND (
            path LIKE '%/_faces/%' 
            OR path LIKE '%/_no_faces/%'
            OR path LIKE '%/_duplicates/%'
            OR path LIKE '%/_animals/%'
            OR path LIKE '%/_quarantine/%'
            OR path LIKE '%/_people_no_face/%'
          )
    """, (face_run_id,))
    
    excluded_stats = cur.fetchone()
    print(f"   Файлов в исключаемых папках (_faces, _no_faces и т.д.): {excluded_stats['total_files_in_excluded_dirs']}")
    print()
    
    # 6. Подсчет проблемных файлов
    print("6. Итоговая статистика проблемных файлов:")
    cur.execute("""
        SELECT COUNT(*) AS count
        FROM files f
        LEFT JOIN face_rectangles fr ON fr.file_path = f.path AND fr.run_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        WHERE f.faces_run_id = ? 
          AND COALESCE(f.faces_count, 0) > 0 
          AND f.status != 'deleted'
          AND fr.id IS NULL
    """, (face_run_id, face_run_id))
    
    total_problem_files = cur.fetchone()['count']
    print(f"   Всего файлов с faces_count > 0, но без face_rectangles: {total_problem_files}")
    
    cur.execute("""
        SELECT COUNT(*) AS count
        FROM files f
        LEFT JOIN face_rectangles fr ON fr.file_path = f.path AND fr.run_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        WHERE f.faces_run_id = ? 
          AND f.faces_scanned_at IS NOT NULL
          AND f.status != 'deleted'
          AND fr.id IS NULL
    """, (face_run_id, face_run_id))
    
    total_scanned_without_rectangles = cur.fetchone()['count']
    print(f"   Всего файлов с faces_scanned_at, но без face_rectangles: {total_scanned_without_rectangles}")
    print()
    
    print("=" * 80)
    print("Диагностика завершена")
    print("=" * 80)

if __name__ == "__main__":
    main()
