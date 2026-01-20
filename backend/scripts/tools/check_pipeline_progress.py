#!/usr/bin/env python3
"""
Проверка прогресса прогона pipeline и последних ошибок.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Проверка прогресса прогона")
    parser.add_argument("--pipeline-run-id", type=int, required=True, help="ID прогона")
    args = parser.parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Информация о прогоне
    cur.execute("""
        SELECT id, status, step_num, step_total, step_title, last_path, last_error, updated_at
        FROM pipeline_runs
        WHERE id = ?
    """, (args.pipeline_run_id,))
    
    run = cur.fetchone()
    if not run:
        print(f"❌ Прогон {args.pipeline_run_id} не найден")
        return
    
    print("=" * 80)
    print(f"Прогон {run['id']}:")
    print(f"  Статус: {run['status']}")
    print(f"  Шаг: {run['step_num']}/{run['step_total']} - {run['step_title']}")
    print(f"  Обновлен: {run['updated_at']}")
    if run['last_path']:
        print(f"  Последний файл: {run['last_path']}")
    if run['last_error']:
        print(f"  Последняя ошибка: {run['last_error']}")
    print()
    
    # Информация о face_run
    if run['status'] == 'running':
        cur.execute("""
            SELECT pr.face_run_id, fr.processed_files, fr.faces_found, fr.last_path, fr.last_error
            FROM pipeline_runs pr
            LEFT JOIN face_runs fr ON fr.id = pr.face_run_id
            WHERE pr.id = ?
        """, (args.pipeline_run_id,))
        
        face_run = cur.fetchone()
        if face_run and face_run['face_run_id']:
            print(f"Face run {face_run['face_run_id']}:")
            print(f"  Обработано файлов: {face_run['processed_files']}")
            print(f"  Найдено лиц: {face_run['faces_found']}")
            if face_run['last_path']:
                print(f"  Последний файл: {face_run['last_path']}")
            if face_run['last_error']:
                print(f"  Последняя ошибка: {face_run['last_error']}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
