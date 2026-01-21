#!/usr/bin/env python3
"""
Очистка записей с file_id = NULL для файлов, которых нет физически.

Эти записи остались после удаления файлов и не несут полезной информации.
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


def strip_local_prefix(path: str) -> str:
    """Убрать префикс local: из пути."""
    if path.startswith("local:"):
        return path[6:]
    return path


def cleanup_table(conn, table_name: str, path_column: str, file_id_column: str = "file_id", dry_run: bool = False):
    """Удалить записи с NULL file_id для несуществующих файлов."""
    cur = conn.cursor()
    
    # Получаем все пути с NULL file_id
    cur.execute(f"""
        SELECT DISTINCT {path_column}
        FROM {table_name}
        WHERE {file_id_column} IS NULL
    """)
    paths = [row[path_column] for row in cur.fetchall()]
    
    if not paths:
        print(f"  ✅ {table_name}: нет записей с NULL {file_id_column}")
        return 0
    
    # Проверяем, какие файлы существуют
    existing_paths = []
    missing_paths = []
    
    for path in paths:
        abs_path = strip_local_prefix(path)
        if os.path.isfile(abs_path):
            existing_paths.append(path)
        else:
            missing_paths.append(path)
    
    if existing_paths:
        print(f"  ⚠️  {table_name}: найдено {len(existing_paths)} файлов, которые существуют, но отсутствуют в files")
        print(f"     Эти записи НЕ будут удалены (требуют ручной проверки)")
    
    if not missing_paths:
        print(f"  ✅ {table_name}: все файлы существуют (нечего удалять)")
        return 0
    
    # Подсчитываем количество записей для удаления
    placeholders = ",".join(["?"] * len(missing_paths))
    cur.execute(f"""
        SELECT COUNT(*) as cnt
        FROM {table_name}
        WHERE {file_id_column} IS NULL AND {path_column} IN ({placeholders})
    """, missing_paths)
    count = cur.fetchone()["cnt"]
    
    print(f"  🗑️  {table_name}: будет удалено {count} записей для {len(missing_paths)} несуществующих файлов")
    
    if not dry_run:
        cur.execute(f"""
            DELETE FROM {table_name}
            WHERE {file_id_column} IS NULL AND {path_column} IN ({placeholders})
        """, missing_paths)
        deleted = cur.rowcount
        print(f"     Удалено: {deleted} записей")
        return deleted
    else:
        print(f"     [DRY RUN] Будет удалено: {count} записей")
        return 0


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Очистка записей с file_id = NULL для несуществующих файлов")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    parser.add_argument("--yes", action="store_true", help="Автоматически подтвердить удаление (без интерактивного запроса)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ОЧИСТКА ЗАПИСЕЙ С file_id = NULL")
    print("=" * 70)
    print(f"Режим: {'DRY RUN' if args.dry_run else 'ВЫПОЛНЕНИЕ'}")
    
    if not args.dry_run:
        if not args.yes:
            response = input("\n⚠️  ВНИМАНИЕ: Будут удалены записи для несуществующих файлов!\nПродолжить? (yes/no): ")
            if response.lower() != "yes":
                print("Отменено.")
                return 1
        else:
            print("\n⚠️  ВНИМАНИЕ: Будут удалены записи для несуществующих файлов!")
            print("Продолжаем (--yes указан)...")
    
    conn = get_connection()
    try:
        tables = [
            ("file_groups", "file_path"),
            ("files_manual_labels", "path"),
        ]
        
        total_deleted = 0
        for table_name, path_column in tables:
            print(f"\n📋 Таблица: {table_name}")
            deleted = cleanup_table(conn, table_name, path_column, dry_run=args.dry_run)
            total_deleted += deleted
        
        if not args.dry_run:
            conn.commit()
            print(f"\n✅ Очистка завершена. Удалено записей: {total_deleted}")
        else:
            print(f"\n[DRY RUN] Очистка не выполнена. Используйте без --dry-run для выполнения.")
        
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
