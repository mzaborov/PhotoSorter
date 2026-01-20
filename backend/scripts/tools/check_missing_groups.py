"""
Скрипт для проверки, какие группы отсутствуют в БД для данного pipeline_run_id.

Показывает разницу между группами, которые есть в БД, и группами из старого предопределенного списка.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys
from datetime import datetime, timezone


def check_missing_groups(db_path: Path, pipeline_run_id: int) -> None:
    """Проверяет, какие группы отсутствуют в БД."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    try:
        cur = conn.cursor()
        
        # Получаем существующие группы для данного pipeline_run_id
        cur.execute("""
            SELECT DISTINCT group_path, COUNT(*) as files_count
            FROM file_groups
            WHERE pipeline_run_id = ?
            GROUP BY group_path
            ORDER BY group_path
        """, (pipeline_run_id,))
        
        existing_groups = {row["group_path"]: row["files_count"] for row in cur.fetchall()}
        
        print(f"Группы в БД для pipeline_run_id={pipeline_run_id}:")
        print(f"{'Группа':<30} {'Файлов':<10}")
        print("-" * 40)
        if existing_groups:
            for group, count in sorted(existing_groups.items()):
                print(f"{group:<30} {count:<10}")
        else:
            print("(нет групп)")
        
        print(f"\nВсего групп в БД: {len(existing_groups)}")
        
        # Старый предопределенный список (для справки)
        old_predefined = [
            "2023 Турция",
            "2024 Турция",
            "Мемы",
            "Здоровье",
            "Чеки",
            "Дом и ремонт",
            "Артефакты людей"
        ]
        
        missing = [g for g in old_predefined if g not in existing_groups]
        if missing:
            print(f"\nГруппы из старого предопределенного списка, которых нет в БД:")
            for g in missing:
                print(f"  - {g}")
            print(f"\n⚠️  ВНИМАНИЕ: Группы без файлов не будут показываться в выпадашке!")
            print(f"Чтобы группа появилась в UI, нужно назначить ей хотя бы один файл.")
        else:
            print(f"\n✅ Все группы из старого списка присутствуют в БД (или были удалены из предопределенных)")
        
    finally:
        conn.close()


def main() -> int:
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Проверяет, какие группы отсутствуют в БД"
    )
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--pipeline-run-id", type=int, required=True, help="Pipeline run ID")
    
    args = ap.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ БД не найдена: {db_path}", file=sys.stderr)
        return 1
    
    check_missing_groups(db_path, args.pipeline_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
