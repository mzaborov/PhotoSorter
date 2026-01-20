#!/usr/bin/env python3
"""
Скрипт для очистки faces_scanned_at для прогона, чтобы перезапустить детекцию лиц.
Используется для исправления проблем, когда faces_scanned_at установлен, но face_rectangles не сохранены.
"""

import sys
import argparse
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    parser = argparse.ArgumentParser(
        description="Очищает faces_scanned_at для прогона, чтобы перезапустить детекцию лиц"
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=int,
        required=True,
        help="ID прогона pipeline (pipeline_run_id)"
    )
    parser.add_argument(
        "--clear-rectangles",
        action="store_true",
        help="Также удалить face_rectangles для face_run_id прогона (полное пересканирование)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет сделано, без изменений в БД"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Подтвердить выполнение (без этого будет только dry-run)"
    )
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
    print(f"Информация о прогоне:")
    print(f"  Pipeline run ID: {run_info['id']}")
    print(f"  Face run ID: {face_run_id}")
    print(f"  Root path: {run_info['root_path']}")
    print(f"  Status: {run_info['status']}")
    print()
    
    # Статистика до изменений
    cur.execute("""
        SELECT 
            COUNT(*) AS total_files,
            COUNT(CASE WHEN faces_scanned_at IS NOT NULL THEN 1 END) AS files_with_scanned_at,
            COUNT(CASE WHEN COALESCE(faces_count, 0) > 0 THEN 1 END) AS files_with_faces_count
        FROM files
        WHERE faces_run_id = ? AND status != 'deleted'
    """, (face_run_id,))
    
    stats_before = cur.fetchone()
    
    cur.execute("""
        SELECT COUNT(*) AS count
        FROM face_rectangles
        WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
    """, (face_run_id,))
    
    rectangles_before = cur.fetchone()['count']
    
    print("Статистика ДО изменений:")
    print(f"  Всего файлов в прогоне: {stats_before['total_files']}")
    print(f"  Файлов с faces_scanned_at: {stats_before['files_with_scanned_at']}")
    print(f"  Файлов с faces_count > 0: {stats_before['files_with_faces_count']}")
    print(f"  Face rectangles в БД: {rectangles_before}")
    print()
    
    # Проверяем, есть ли файлы с faces_scanned_at, но без face_rectangles
    cur.execute("""
        SELECT COUNT(*) AS count
        FROM files f
        LEFT JOIN face_rectangles fr ON fr.file_path = f.path AND fr.run_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        WHERE f.faces_run_id = ? 
          AND f.faces_scanned_at IS NOT NULL
          AND f.status != 'deleted'
          AND fr.id IS NULL
    """, (face_run_id, face_run_id))
    
    problem_files_count = cur.fetchone()['count']
    
    print(f"Файлов с faces_scanned_at, но без face_rectangles: {problem_files_count}")
    print()
    
    if not args.confirm:
        print("⚠️  Это dry-run. Для выполнения используйте --confirm")
        print()
        print("Что будет сделано:")
        print(f"  1. Очистить faces_scanned_at для {stats_before['files_with_scanned_at']} файлов")
        if args.clear_rectangles:
            print(f"  2. Удалить {rectangles_before} записей из face_rectangles для run_id={face_run_id}")
        else:
            print(f"  2. Face rectangles НЕ будут удалены (используйте --clear-rectangles для полного пересканирования)")
        return
    
    if args.dry_run:
        print("⚠️  Режим dry-run: изменения НЕ будут применены")
        print()
        print("Что будет сделано:")
        print(f"  1. Очистить faces_scanned_at для {stats_before['files_with_scanned_at']} файлов")
        if args.clear_rectangles:
            print(f"  2. Удалить {rectangles_before} записей из face_rectangles для run_id={face_run_id}")
        return
    
    # Выполняем очистку
    print("Выполняем очистку...")
    
    # 1. Очищаем faces_scanned_at
    cur.execute("""
        UPDATE files
        SET faces_scanned_at = NULL
        WHERE faces_run_id = ? AND faces_scanned_at IS NOT NULL AND status != 'deleted'
    """, (face_run_id,))
    
    files_cleared = cur.rowcount
    
    # 2. Очищаем faces_count (чтобы не было расхождений)
    cur.execute("""
        UPDATE files
        SET faces_count = NULL
        WHERE faces_run_id = ? AND faces_count IS NOT NULL AND status != 'deleted'
    """, (face_run_id,))
    
    # Примечание: faces_run_id НЕ очищаем, чтобы сохранить информацию о принадлежности файлов к прогону.
    # Очистка faces_scanned_at и faces_count достаточна для пересканирования, так как логика resume
    # проверяет именно faces_scanned_at (и теперь также проверяет наличие face_rectangles).
    
    # 4. Удаляем face_rectangles, если нужно
    rectangles_deleted = 0
    if args.clear_rectangles:
        cur.execute("""
            DELETE FROM face_rectangles
            WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
        """, (face_run_id,))
        rectangles_deleted = cur.rowcount
    
    conn.commit()
    
    print()
    print("=" * 80)
    print("Изменения применены:")
    print(f"  Очищено faces_scanned_at для {files_cleared} файлов")
    if args.clear_rectangles:
        print(f"  Удалено {rectangles_deleted} записей из face_rectangles")
    print()
    print("Теперь можно перезапустить детекцию лиц для этого прогона.")
    print("=" * 80)

if __name__ == "__main__":
    main()
