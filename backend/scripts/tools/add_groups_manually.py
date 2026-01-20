"""
Скрипт для ручного добавления групп в БД.

Группы добавляются через создание записи в file_groups с "заглушечным" файлом.
Это нужно для того, чтобы группы были видны в выпадашке даже если в них еще нет реальных файлов.

ВНИМАНИЕ: После добавления реальных файлов в эти группы, заглушечную запись можно удалить.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys
from datetime import datetime, timezone


def add_groups_manually(db_path: Path, pipeline_run_id: int, groups: list[str], dry_run: bool = True) -> None:
    """Добавляет группы в БД через заглушечную запись."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    try:
        cur = conn.cursor()
        
        # Проверяем, существует ли pipeline_run_id
        cur.execute("SELECT id FROM pipeline_runs WHERE id = ?", (pipeline_run_id,))
        if not cur.fetchone():
            print(f"❌ Pipeline run {pipeline_run_id} не найден в БД")
            return
        
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
            print(f"\n✅ Все группы уже существуют в БД.")
            return
        
        print(f"\nГруппы для добавления: {groups_to_add}")
        
        # Заглушечный путь файла (не должен существовать в реальности)
        dummy_file_path = f"__dummy_group_marker__"
        now = datetime.now(timezone.utc).isoformat()
        
        if dry_run:
            print(f"\n[DRY RUN] Будет добавлено групп: {len(groups_to_add)}")
            for group in groups_to_add:
                print(f"  - {group} (с заглушечным файлом '{dummy_file_path}')")
            print(f"\n⚠️  Это был DRY RUN. Для реального добавления запустите с --apply")
            print(f"⚠️  ВНИМАНИЕ: После добавления реальных файлов в эти группы, заглушечные записи можно удалить")
            return
        
        print(f"\n⚠️  ВНИМАНИЕ: Группы будут добавлены с заглушечным файлом.")
        print(f"После назначения реальных файлов в эти группы, заглушечные записи можно удалить.")
        response = input("Продолжить? (yes/no): ")
        if response.lower() != 'yes':
            print("Отменено.")
            return
        
        added_count = 0
        for group in groups_to_add:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO file_groups (
                        pipeline_run_id, file_path, group_path, created_at
                    )
                    VALUES (?, ?, ?, ?)
                """, (pipeline_run_id, dummy_file_path, group, now))
                if cur.rowcount > 0:
                    added_count += 1
                    print(f"  ✅ Добавлена группа: {group}")
                else:
                    print(f"  ⚠️  Группа уже существует: {group}")
            except Exception as e:
                print(f"  ❌ Ошибка при добавлении группы '{group}': {e}")
        
        conn.commit()
        
        print(f"\n{'='*60}")
        print(f"Итого добавлено групп: {added_count}")
        print(f"\n💡 После назначения реальных файлов в эти группы, можно удалить заглушечные записи:")
        print(f"   DELETE FROM file_groups WHERE file_path = '{dummy_file_path}' AND pipeline_run_id = {pipeline_run_id}")
        
    finally:
        conn.close()


def main() -> int:
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Добавляет группы в БД вручную (через заглушечную запись)"
    )
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--pipeline-run-id", type=int, required=True, help="Pipeline run ID")
    ap.add_argument("--groups", nargs="+", help="Список групп для добавления")
    ap.add_argument("--apply", action="store_true", help="Применить изменения (по умолчанию dry-run)")
    
    args = ap.parse_args()
    
    if not args.groups:
        # Группы по умолчанию
        groups = [
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
    
    add_groups_manually(db_path, args.pipeline_run_id, groups, dry_run=not args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
