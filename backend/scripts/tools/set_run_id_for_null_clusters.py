#!/usr/bin/env python3
"""
Ставит face_run_id прогона кластерам с run_id IS NULL, кроме архивных (archive_scope = 'archive').
Прогон задаётся pipeline_run_id; берётся его face_run_id из pipeline_runs.

Запуск из корня репо:
  python backend/scripts/tools/set_run_id_for_null_clusters.py --pipeline-run-id 26 --dry-run   # только показать количество
  python backend/scripts/tools/set_run_id_for_null_clusters.py --pipeline-run-id 26            # выполнить UPDATE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection


def main() -> int:
    ap = argparse.ArgumentParser(description="Установить run_id прогона всем кластерам с run_id IS NULL")
    ap.add_argument("--pipeline-run-id", type=int, default=26, help="pipeline_run_id (по умолчанию 26)")
    ap.add_argument("--dry-run", action="store_true", help="Только показать количество, не обновлять")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    # face_run_id прогона
    cur.execute(
        "SELECT id, face_run_id, root_path FROM pipeline_runs WHERE id = ?",
        (args.pipeline_run_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"[FAIL] Прогон pipeline_run_id={args.pipeline_run_id} не найден.")
        return 1
    face_run_id = row[1] if isinstance(row, (tuple, list)) else row["face_run_id"]
    if face_run_id is None:
        print(f"[FAIL] У прогона {args.pipeline_run_id} не задан face_run_id.")
        return 1
    face_run_id = int(face_run_id)

    # Архивные не трогаем: run_id IS NULL AND archive_scope = 'archive'
    cur.execute(
        "SELECT COUNT(*) FROM face_clusters WHERE run_id IS NULL AND COALESCE(TRIM(archive_scope), '') = 'archive'"
    )
    archive_cnt = cur.fetchone()[0]
    # К обновлению: run_id IS NULL и не архивные
    cur.execute(
        """SELECT COUNT(*) FROM face_clusters
           WHERE run_id IS NULL AND (COALESCE(TRIM(archive_scope), '') != 'archive')"""
    )
    cnt = cur.fetchone()[0]

    print(f"Кластеров с run_id IS NULL всего: {cnt + archive_cnt}")
    print(f"  из них архивных (archive_scope='archive'): {archive_cnt}, не трогаем")
    print(f"  к обновлению (не архивные): {cnt}")
    print(f"Прогон pipeline_run_id={args.pipeline_run_id} -> face_run_id={face_run_id}")

    if cnt == 0:
        print("Нечего обновлять.")
        conn.close()
        return 0

    if args.dry_run:
        print("[DRY-RUN] Выполнили бы: UPDATE ... SET run_id = ? WHERE run_id IS NULL AND не архивные")
        conn.close()
        return 0

    cur.execute(
        """UPDATE face_clusters SET run_id = ?
           WHERE run_id IS NULL AND (COALESCE(TRIM(archive_scope), '') != 'archive')""",
        (face_run_id,),
    )
    updated = cur.rowcount
    conn.commit()
    conn.close()
    print(f"Обновлено кластеров: {updated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
