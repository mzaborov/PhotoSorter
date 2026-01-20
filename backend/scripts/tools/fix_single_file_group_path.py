"""
Скрипт для исправления одного неправильного пути в таблице file_groups.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys


def fix_path(db_path: Path, pipeline_run_id: int, broken_path: str, correct_path: str, dry_run: bool = True) -> None:
    """Исправляет путь в file_groups."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    try:
        cur = conn.cursor()
        
        # Проверяем, есть ли запись с неправильным путем
        cur.execute("""
            SELECT id, file_path, group_path
            FROM file_groups
            WHERE pipeline_run_id = ? AND file_path = ?
        """, (pipeline_run_id, broken_path))
        
        records = cur.fetchall()
        
        if not records:
            print(f"❌ Запись с путем '{broken_path}' не найдена для pipeline_run_id={pipeline_run_id}")
            return
        
        print(f"Найдено записей: {len(records)}")
        
        # Проверяем, существует ли правильный путь в таблице files
        cur.execute("""
            SELECT path FROM files WHERE path = ?
        """, (correct_path,))
        
        if not cur.fetchone():
            print(f"⚠️  ВНИМАНИЕ: Правильный путь '{correct_path}' не найден в таблице files!")
            response = input("Продолжить обновление? (yes/no): ")
            if response.lower() != 'yes':
                print("Отменено.")
                return
        
        # Обновляем каждую запись
        for record in records:
            fg_id = record["id"]
            group_path = record["group_path"]
            
            print(f"\nЗапись ID={fg_id}:")
            print(f"  Старый путь: {broken_path}")
            print(f"  Новый путь:  {correct_path}")
            print(f"  Группа:      {group_path}")
            
            if not dry_run:
                # Проверяем, нет ли уже записи с правильным путем и той же группой
                cur.execute("""
                    SELECT id FROM file_groups
                    WHERE pipeline_run_id = ? AND file_path = ? AND group_path = ?
                """, (pipeline_run_id, correct_path, group_path))
                
                existing = cur.fetchone()
                if existing:
                    print(f"  ⚠️  Запись с правильным путем уже существует (ID={existing['id']})")
                    # Удаляем старую запись с неправильным путем
                    cur.execute("DELETE FROM file_groups WHERE id = ?", (fg_id,))
                    print(f"  ✅ Удалена старая запись с неправильным путем")
                else:
                    # Обновляем путь
                    cur.execute("""
                        UPDATE file_groups
                        SET file_path = ?
                        WHERE id = ?
                    """, (correct_path, fg_id))
                    print(f"  ✅ Обновлено в БД")
                
                conn.commit()
            else:
                print(f"  [DRY RUN] Будет обновлено")
        
        print(f"\n{'='*60}")
        if dry_run:
            print(f"⚠️  Это был DRY RUN. Для реального обновления запустите с --apply")
        else:
            print(f"✅ Изменения применены")
        
    finally:
        conn.close()


def main() -> int:
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Исправляет неправильный путь в таблице file_groups"
    )
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--pipeline-run-id", type=int, required=True, help="Pipeline run ID")
    ap.add_argument("--broken-path", required=True, help="Неправильный путь (как сохранен в БД)")
    ap.add_argument("--correct-path", required=True, help="Правильный путь (как должен быть)")
    ap.add_argument("--apply", action="store_true", help="Применить изменения (по умолчанию dry-run)")
    
    args = ap.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ БД не найдена: {db_path}", file=sys.stderr)
        return 1
    
    fix_path(
        db_path, 
        args.pipeline_run_id, 
        args.broken_path, 
        args.correct_path, 
        dry_run=not args.apply
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
