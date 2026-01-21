#!/usr/bin/env python3
"""
Исправление file_id для записей с NULL - поиск файлов по имени в таблице files.

Файлы были перемещены в процессе сортировки (из корня в _faces, _delete и т.д.),
но метки остались со старыми путями. Находим файлы по имени и обновляем file_id.
"""

import sys
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection


def get_basename(path: str) -> str:
    """Получить имя файла из пути."""
    clean_path = path.replace("local:", "")
    return os.path.basename(clean_path)


def find_file_by_name(conn, basename: str) -> list[dict]:
    """Найти файлы с таким же именем в таблице files."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, path, inventory_scope, status
        FROM files
        WHERE path LIKE ? OR path LIKE ?
        ORDER BY 
            CASE WHEN status = 'deleted' THEN 1 ELSE 0 END,  -- Приоритет не удаленным
            path
        LIMIT 10
    """, (f"%/{basename}", f"%\\{basename}"))
    return [dict(row) for row in cur.fetchall()]


def fix_table(conn, table_name: str, path_column: str, file_id_column: str = "file_id", dry_run: bool = False):
    """Исправить file_id для записей с NULL, найдя файлы по имени."""
    cur = conn.cursor()
    
    # Получаем все уникальные пути с NULL file_id
    cur.execute(f"""
        SELECT DISTINCT {path_column}
        FROM {table_name}
        WHERE {file_id_column} IS NULL
    """)
    paths = [row[path_column] for row in cur.fetchall()]
    
    if not paths:
        print(f"  ✅ {table_name}: нет записей с NULL {file_id_column}")
        return 0, 0, 0
    
    print(f"\n  📋 {table_name}: {len(paths)} уникальных путей с NULL {file_id_column}")
    
    fixed_count = 0
    not_found_count = 0
    multiple_matches = 0
    
    for i, path in enumerate(paths):
        basename = get_basename(path)
        matches = find_file_by_name(conn, basename)
        
        if not matches:
            not_found_count += 1
            if not_found_count <= 3:
                print(f"    ❌ Не найден: {basename}")
            continue
        
        # Если несколько совпадений, берем первое (не удаленное, если есть)
        file_id = matches[0]["id"]
        if len(matches) > 1:
            multiple_matches += 1
            if multiple_matches <= 3:
                print(f"    ⚠️  Несколько совпадений для {basename}, берем первое:")
                print(f"       Выбран: ID={file_id}, Path={matches[0]['path']}")
        
        # Обновляем file_id
        if not dry_run:
            cur.execute(f"""
                UPDATE {table_name}
                SET {file_id_column} = ?
                WHERE {file_id_column} IS NULL AND {path_column} = ?
            """, (file_id, path))
            updated = cur.rowcount
            fixed_count += updated
        else:
            fixed_count += 1
            if fixed_count <= 3:
                print(f"    ✅ Будет обновлено: {basename} -> file_id={file_id} (Path: {matches[0]['path']})")
        
        if (i + 1) % 100 == 0:
            print(f"    Обработано: {i + 1}/{len(paths)}... (исправлено: {fixed_count}, не найдено: {not_found_count})")
    
    print(f"  Результаты:")
    print(f"    Исправлено: {fixed_count}")
    print(f"    Не найдено: {not_found_count}")
    if multiple_matches > 0:
        print(f"    С несколькими совпадениями: {multiple_matches}")
    
    return fixed_count, not_found_count, multiple_matches


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Исправить file_id для записей с NULL, найдя файлы по имени")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ИСПРАВЛЕНИЕ file_id ДЛЯ ЗАПИСЕЙ С NULL")
    print("=" * 70)
    print("Поиск файлов в таблице files по имени (в разных папках)")
    print(f"Режим: {'DRY RUN' if args.dry_run else 'ВЫПОЛНЕНИЕ'}")
    
    if not args.dry_run:
        response = input("\n⚠️  ВНИМАНИЕ: Будут обновлены file_id для записей с NULL!\nПродолжить? (yes/no): ")
        if response.lower() != "yes":
            print("Отменено.")
            return 1
    
    conn = get_connection()
    try:
        tables = [
            ("file_groups", "file_path"),
            ("files_manual_labels", "path"),
        ]
        
        total_fixed = 0
        total_not_found = 0
        
        for table_name, path_column in tables:
            print(f"\n{'='*70}")
            print(f"Таблица: {table_name}")
            print(f"{'='*70}")
            fixed, not_found, multiple = fix_table(conn, table_name, path_column, dry_run=args.dry_run)
            total_fixed += fixed
            total_not_found += not_found
        
        if not args.dry_run:
            conn.commit()
            print(f"\n{'='*70}")
            print("ИТОГО")
            print(f"{'='*70}")
            print(f"  Исправлено записей: {total_fixed}")
            print(f"  Не найдено файлов: {total_not_found}")
            print(f"\n✅ Исправление завершено!")
        else:
            print(f"\n[DRY RUN] Исправление не выполнено. Используйте без --dry-run для выполнения.")
            print(f"  Будет исправлено записей: {total_fixed}")
            print(f"  Не найдено файлов: {total_not_found}")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
