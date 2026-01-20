"""
Скрипт для исправления неправильных путей в таблице file_groups.

Проблема: из-за неправильного экранирования путей в JavaScript некоторые пути
в file_groups сохранены с обрезанными/искаженными символами (например, обратные слэши).

Скрипт находит записи в file_groups, где путь не найден в таблице files,
и пытается найти правильный путь по имени файла, затем обновляет file_groups.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys


def fix_broken_paths(db_path: Path, pipeline_run_id: int, dry_run: bool = True) -> None:
    """Исправляет неправильные пути в file_groups."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    try:
        cur = conn.cursor()
        
        # Находим записи в file_groups, где путь не найден в files
        cur.execute("""
            SELECT 
              fg.id,
              fg.file_path as broken_path,
              fg.group_path,
              fg.pipeline_run_id
            FROM file_groups fg
            LEFT JOIN files f ON f.path = fg.file_path
            WHERE fg.pipeline_run_id = ?
              AND f.path IS NULL
        """, (pipeline_run_id,))
        
        broken_records = cur.fetchall()
        
        if not broken_records:
            print(f"Нет записей с неправильными путями для pipeline_run_id={pipeline_run_id}")
            return
        
        print(f"Найдено записей с неправильными путями: {len(broken_records)}")
        
        fixed_count = 0
        not_found_count = 0
        
        for record in broken_records:
            broken_path = record["broken_path"]
            group_path = record["group_path"]
            fg_id = record["id"]
            
            print(f"\nЗапись ID={fg_id}:")
            print(f"  Неправильный путь: {broken_path}")
            print(f"  Группа: {group_path}")
            
            # Пытаемся извлечь имя файла из обрезанного пути
            # Например: "local:C:        mpPhoto_no_faces41017_125832.jpg" -> "20241017_125832.jpg"
            file_name = None
            if broken_path:
                import re
                # Ищем паттерн даты в пути (YYYYMMDD - 8 цифр)
                date_match = re.search(r'(\d{8})', broken_path)
                if not date_match:
                    # Если не нашли 8 цифр, ищем более короткие паттерны (6-7 цифр)
                    # Например, "41017" или "125832" из "20241017_125832"
                    date_match = re.search(r'(\d{6,7})', broken_path)
                
                if date_match:
                    date_part = date_match.group(1)
                    print(f"  Извлечена дата из пути: {date_part}")
                    
                    # Ищем файл с этой датой в имени или пути
                    cur.execute("""
                        SELECT path, name
                        FROM files
                        WHERE name LIKE ? OR path LIKE ?
                        LIMIT 10
                    """, (f'%{date_part}%', f'%{date_part}%'))
                    candidates = cur.fetchall()
                    
                    if candidates:
                        # Если несколько кандидатов, выбираем наиболее подходящий
                        # (например, по длине пути или по наличию полного имени файла)
                        if len(candidates) == 1:
                            correct_path = candidates[0]["path"]
                        else:
                            # Пытаемся найти файл, который содержит полное имя из обрезанного пути
                            # Извлекаем имя файла из обрезанного пути (последняя часть после последнего слэша или обратного слэша)
                            broken_name = broken_path.split('\\')[-1].split('/')[-1]
                            # Ищем кандидата с похожим именем
                            correct_path = None
                            for cand in candidates:
                                if broken_name.lower() in cand["name"].lower() or broken_name.lower() in cand["path"].lower():
                                    correct_path = cand["path"]
                                    break
                            if not correct_path:
                                # Берем первый кандидат
                                correct_path = candidates[0]["path"]
                        
                        print(f"  Найден правильный путь: {correct_path}")
                        
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
                                # Обновляем путь в file_groups
                                cur.execute("""
                                    UPDATE file_groups
                                    SET file_path = ?
                                    WHERE id = ?
                                """, (correct_path, fg_id))
                                print(f"  ✅ Обновлено в БД")
                            conn.commit()
                            fixed_count += 1
                        else:
                            print(f"  [DRY RUN] Будет обновлено на: {correct_path}")
                            fixed_count += 1
                    else:
                        print(f"  ❌ Не удалось найти правильный путь по дате {date_part}")
                        not_found_count += 1
                else:
                    print(f"  ❌ Не удалось извлечь дату из пути (искали паттерн 6-8 цифр)")
                    not_found_count += 1
        
        print(f"\n{'='*60}")
        print(f"Итого:")
        print(f"  Исправлено: {fixed_count}")
        print(f"  Не найдено: {not_found_count}")
        if dry_run:
            print(f"\n⚠️  Это был DRY RUN. Для реального обновления запустите с --apply")
        
    finally:
        conn.close()


def main() -> int:
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Исправляет неправильные пути в таблице file_groups"
    )
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--pipeline-run-id", type=int, required=True, help="Pipeline run ID")
    ap.add_argument("--apply", action="store_true", help="Применить изменения (по умолчанию dry-run)")
    
    args = ap.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ БД не найдена: {db_path}", file=sys.stderr)
        return 1
    
    fix_broken_paths(db_path, args.pipeline_run_id, dry_run=not args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
