"""Список последних прогонов pipeline.

Использование:
    python backend/scripts/debug/list_pipeline_runs.py
    python backend/scripts/debug/list_pipeline_runs.py --limit 15
    python backend/scripts/debug/list_pipeline_runs.py --limit 5 --include-face-runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection


def main() -> int:
    ap = argparse.ArgumentParser(description="Список последних прогонов pipeline")
    ap.add_argument("--limit", type=int, default=15, help="Количество прогонов для отображения (по умолчанию 15)")
    ap.add_argument("--include-face-runs", action="store_true", help="Включить информацию о face_runs")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    limit = max(1, min(100, args.limit))

    # Получаем последние прогоны
    cur.execute(
        """
        SELECT id, kind, status, root_path, face_run_id, started_at, finished_at, step_num, step_total, step_title
        FROM pipeline_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    runs = cur.fetchall()
    if not runs:
        print("Нет прогонов в БД")
        return 0

    print("=" * 100)
    print(f"Последние {len(runs)} прогонов pipeline:")
    print("=" * 100)

    for run in runs:
        print(f"\nПрогон #{run['id']}:")
        print(f"  Вид: {run['kind']}")
        print(f"  Статус: {run['status']}")
        print(f"  Путь: {run['root_path']}")
        if run['face_run_id']:
            print(f"  Face run ID: {run['face_run_id']}")
        print(f"  Начало: {run['started_at']}")
        if run['finished_at']:
            print(f"  Завершение: {run['finished_at']}")
        if run['step_num'] and run['step_total']:
            print(f"  Прогресс: {run['step_num']}/{run['step_total']} - {run['step_title'] or 'N/A'}")

        # Если нужно, получаем информацию о face_run
        if args.include_face_runs and run['face_run_id']:
            cur.execute(
                """
                SELECT id, scope, status, total_files, processed_files, faces_found, started_at, finished_at
                FROM face_runs
                WHERE id = ?
                """,
                (run['face_run_id'],),
            )
            face_run = cur.fetchone()
            if face_run:
                print(f"  Face Run #{face_run['id']}:")
                print(f"    Статус: {face_run['status']}")
                print(f"    Файлов: {face_run['processed_files']}/{face_run['total_files']}")
                print(f"    Лиц найдено: {face_run['faces_found']}")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
