"""
Скрипт для добавления недостающих групп в таблицу file_groups.

Группы добавляются без привязки к файлам (только для отображения в выпадашке).
Но на самом деле, если в группе нет файлов, она не должна показываться в выпадашке.

Этот скрипт может быть полезен для миграции или для добавления групп,
которые должны быть доступны, но еще не имеют файлов.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys


def add_missing_groups(db_path: Path, pipeline_run_id: int, groups: list[str], dry_run: bool = True) -> None:
    """Добавляет недостающие группы в БД."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    try:
        cur = conn.cursor()
        
        # Получаем существующие группы для данного pipeline_run_id
        cur.execute("""
            SELECT DISTINCT group_path
            FROM file_groups
            WHERE pipeline_run_id = ?
        """, (pipeline_run_id,))
        
        existing_groups = {row["group_path"] for row in cur.fetchall()}
        print(f"Существующие группы для pipeline_run_id={pipeline_run_id}: {sorted(existing_groups)}")
        
        # Определяем, какие группы нужно добавить
        groups_to_add = [g for g in groups if g not in existing_groups]
        
        if not groups_to_add:
            print(f"Все группы уже существуют в БД.")
            return
        
        print(f"\nГруппы для добавления: {groups_to_add}")
        
        if dry_run:
            print(f"\n[DRY RUN] Будет добавлено групп: {len(groups_to_add)}")
            for group in groups_to_add:
                print(f"  - {group}")
            print(f"\n⚠️  Это был DRY RUN. Для реального добавления запустите с --apply")
            print(f"⚠️  ВНИМАНИЕ: Группы без файлов не будут показываться в выпадашке!")
            return
        
        # Добавляем группы (но без файлов они не будут видны в UI)
        # На самом деле, если в группе нет файлов, она не нужна в БД
        # Но если пользователь хочет их добавить для будущего использования, можно добавить "заглушку"
        
        print(f"\n⚠️  ВНИМАНИЕ: Группы без файлов не будут показываться в выпадашке!")
        print(f"Группы добавляются только для справки. Чтобы они появились в UI, нужно назначить им файлы.")
        
        response = input("Продолжить? (yes/no): ")
        if response.lower() != 'yes':
            print("Отменено.")
            return
        
        # На самом деле, добавлять группы без файлов не имеет смысла,
        # так как они не будут показываться в выпадашке
        # Но если нужно, можно добавить запись с пустым file_path (но это нарушит схему)
        
        print(f"\n❌ Нельзя добавить группы без файлов - они не будут видны в UI.")
        print(f"Чтобы группа появилась в выпадашке, нужно назначить ей хотя бы один файл.")
        
    finally:
        conn.close()


def main() -> int:
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Добавляет недостающие группы в БД (но они не будут видны без файлов)"
    )
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--pipeline-run-id", type=int, required=True, help="Pipeline run ID")
    ap.add_argument("--groups", nargs="+", help="Список групп для добавления")
    ap.add_argument("--apply", action="store_true", help="Применить изменения (по умолчанию dry-run)")
    
    args = ap.parse_args()
    
    if not args.groups:
        # Группы из старого предопределенного списка (если нужно)
        groups = [
            "2023 Турция",
            "Здоровье",
            "Чеки",
            "Дом и ремонт",
        ]
        print(f"Используются группы по умолчанию: {groups}")
        print(f"Используйте --groups для указания своих групп")
    else:
        groups = args.groups
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ БД не найдена: {db_path}", file=sys.stderr)
        return 1
    
    add_missing_groups(db_path, args.pipeline_run_id, groups, dry_run=not args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
