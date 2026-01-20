#!/usr/bin/env python3
"""
Скрипт для сброса статуса прогона pipeline на 'running', чтобы можно было перезапустить.
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
        description="Сбрасывает статус прогона pipeline на 'running'"
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=int,
        required=True,
        help="ID прогона pipeline"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Подтвердить изменение (без этого будет только dry-run)"
    )
    args = parser.parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем информацию о прогоне
    cur.execute("""
        SELECT id, kind, root_path, status, face_run_id
        FROM pipeline_runs
        WHERE id = ?
    """, (args.pipeline_run_id,))
    
    run_info = cur.fetchone()
    if not run_info:
        print(f"❌ Прогон {args.pipeline_run_id} не найден!")
        return
    
    print("=" * 80)
    print(f"Информация о прогоне:")
    print(f"  ID: {run_info['id']}")
    print(f"  Kind: {run_info['kind']}")
    print(f"  Root path: {run_info['root_path']}")
    print(f"  Текущий статус: {run_info['status']}")
    print(f"  Face run ID: {run_info['face_run_id']}")
    print()
    
    if run_info['status'] == 'running':
        print("⚠️  Прогон уже в статусе 'running'")
        return
    
    if not args.confirm:
        print("⚠️  Это dry-run. Для выполнения используйте --confirm")
        print()
        print(f"Что будет сделано:")
        print(f"  Изменить статус с '{run_info['status']}' на 'running'")
        return
    
    # Изменяем статус
    cur.execute("""
        UPDATE pipeline_runs
        SET status = 'running', last_error = ''
        WHERE id = ?
    """, (args.pipeline_run_id,))
    
    conn.commit()
    
    print()
    print("=" * 80)
    print(f"Статус прогона {args.pipeline_run_id} изменен на 'running'")
    print(f"Теперь можно перезапустить pipeline с --pipeline-run-id {args.pipeline_run_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()
