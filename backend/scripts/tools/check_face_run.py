#!/usr/bin/env python3
"""
Проверка информации о face_run.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем face_runs
    cur.execute("SELECT * FROM face_runs WHERE id = ?", (args.run_id,))
    run = cur.fetchone()
    
    if run:
        print(f"Run {args.run_id} найден в face_runs:")
        print(f"  scope: {run['scope']}")
        print(f"  root_path: {run['root_path']}")
        print(f"  status: {run['status']}")
    else:
        print(f"Run {args.run_id} не найден в face_runs")
        
        # Проверяем, есть ли face_rectangles с этим run_id
        cur.execute("SELECT COUNT(*) FROM face_rectangles WHERE run_id = ?", (args.run_id,))
        count = cur.fetchone()[0]
        print(f"Но есть {count} face_rectangles с run_id={args.run_id}")
        
        # Проверяем pipeline_runs
        cur.execute("SELECT id, face_run_id FROM pipeline_runs WHERE face_run_id = ?", (args.run_id,))
        pipeline = cur.fetchone()
        if pipeline:
            print(f"Найден pipeline_run_id={pipeline['id']} с face_run_id={args.run_id}")
    
    # Показываем последние face_runs
    print("\nПоследние 10 face_runs:")
    cur.execute("SELECT id, scope, root_path, status FROM face_runs ORDER BY id DESC LIMIT 10")
    for row in cur.fetchall():
        print(f"  ID: {row['id']}, scope: {row['scope']}, root: {row['root_path']}, status: {row['status']}")

if __name__ == "__main__":
    main()
