#!/usr/bin/env python3
"""
Миграция данных: заполнение file_id во всех таблицах из files по file_path/path.

ЭТАП 1.2: Миграция данных file_path → file_id

Для каждой таблицы:
1. Найти все уникальные file_path/path
2. Найти соответствующий file_id в files
3. Если файла нет - пометить как проблемную запись (логирование)
4. Обновить file_id для всех записей
"""

import sys
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


def migrate_table(
    cur,
    table_name: str,
    path_column: str,
    file_id_column: str = "file_id",
    dry_run: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Мигрирует данные в одной таблице: заполняет file_id из files по path.
    
    Returns:
        (updated_count, missing_count, missing_paths)
    """
    print(f"\nМиграция таблицы: {table_name}")
    print(f"  Колонка пути: {path_column}")
    print(f"  Колонка file_id: {file_id_column}")
    
    # 1. Получаем все уникальные пути из таблицы
    cur.execute(f"SELECT DISTINCT {path_column} FROM {table_name} WHERE {path_column} IS NOT NULL AND {path_column} != ''")
    paths = [row[0] for row in cur.fetchall()]
    
    if not paths:
        print(f"  Нет записей для миграции")
        return (0, 0, [])
    
    print(f"  Найдено уникальных путей: {len(paths)}")
    
    # 2. Находим соответствующие file_id в files
    path_to_file_id = {}
    missing_paths = []
    
    for path in paths:
        cur.execute("SELECT id FROM files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if row:
            path_to_file_id[path] = row[0]
        else:
            missing_paths.append(path)
    
    print(f"  Найдено в files: {len(path_to_file_id)}")
    print(f"  Отсутствует в files: {len(missing_paths)}")
    
    if missing_paths:
        print(f"  ⚠️  ПРОБЛЕМНЫЕ ПУТИ (первые 10):")
        for path in missing_paths[:10]:
            print(f"      {path}")
        if len(missing_paths) > 10:
            print(f"      ... и еще {len(missing_paths) - 10}")
    
    # 3. Обновляем file_id для всех записей (batch)
    updated_count = 0
    if not dry_run and path_to_file_id:
        # Группируем обновления по file_id для эффективности
        updates_by_file_id = {}
        for path, file_id in path_to_file_id.items():
            if file_id not in updates_by_file_id:
                updates_by_file_id[file_id] = []
            updates_by_file_id[file_id].append(path)
        
        # Обновляем batch'ами по file_id
        for file_id, paths_list in updates_by_file_id.items():
            placeholders = ",".join(["?"] * len(paths_list))
            cur.execute(
                f"UPDATE {table_name} SET {file_id_column} = ? WHERE {path_column} IN ({placeholders})",
                (file_id, *paths_list)
            )
            updated_count += cur.rowcount
    
    if dry_run:
        # Подсчитываем количество записей, которые будут обновлены
        # Для этого нужно посчитать количество записей с каждым file_path
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {path_column} IN ({','.join(['?'] * len(path_to_file_id))})", list(path_to_file_id.keys()))
        estimated_count = cur.fetchone()[0] if path_to_file_id else 0
        print(f"  [DRY RUN] Будет обновлено записей: ~{estimated_count}")
    
    return (updated_count, len(missing_paths), missing_paths)


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Миграция file_path → file_id")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано, без изменений")
    args = parser.parse_args()
    
    print("=" * 60)
    print("МИГРАЦИЯ ДАННЫХ: заполнение file_id во всех таблицах")
    print("=" * 60)
    if args.dry_run:
        print("[DRY RUN MODE - изменения не будут применены]")
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что колонки file_id существуют
    tables_with_file_path = [
        ("face_rectangles", "file_path"),
        ("person_rectangles", "file_path"),
        ("file_persons", "file_path"),
        ("file_groups", "file_path"),
        ("file_group_persons", "file_path"),
        ("files_manual_labels", "path"),
        ("video_manual_frames", "path"),
    ]
    
    print("Проверка наличия колонок file_id...")
    missing_columns = []
    for table_name, path_col in tables_with_file_path:
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cur.fetchall()}
        if "file_id" not in columns:
            missing_columns.append(table_name)
    
    if missing_columns:
        print(f"❌ ОШИБКА: Колонка file_id отсутствует в таблицах: {', '.join(missing_columns)}")
        print("   Сначала выполните ЭТАП 1.1: добавление колонок file_id")
        return 1
    
    print("✅ Все колонки file_id присутствуют")
    print()
    
    # Мигрируем каждую таблицу
    total_updated = 0
    total_missing = 0
    all_missing_paths = []
    
    for table_name, path_col in tables_with_file_path:
        updated, missing, missing_paths = migrate_table(
            cur, table_name, path_col, file_id_column="file_id", dry_run=args.dry_run
        )
        total_updated += updated
        total_missing += missing
        all_missing_paths.extend(missing_paths)
        
        # Для dry-run нужно подсчитать примерное количество записей
        if args.dry_run and updated == 0:
            # Подсчитываем количество записей, которые будут обновлены
            cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {path_col} IS NOT NULL AND {path_col} != ''")
            count = cur.fetchone()[0]
            if count > 0:
                # Примерная оценка: все записи минус проблемные
                total_updated += count - missing
    
    # Коммитим изменения (если не dry-run)
    if not args.dry_run:
        conn.commit()
        print(f"\n✅ Миграция завершена")
        print(f"   Обновлено записей: {total_updated}")
        print(f"   Пропущено (нет в files): {total_missing}")
    else:
        print(f"\n[DRY RUN] Итого:")
        print(f"   Будет обновлено записей: ~{total_updated}")
        print(f"   Пропущено (нет в files): {total_missing}")
    
    # Список уникальных проблемных путей
    if all_missing_paths:
        unique_missing = sorted(set(all_missing_paths))
        print(f"\n⚠️  ВСЕГО УНИКАЛЬНЫХ ПРОБЛЕМНЫХ ПУТЕЙ: {len(unique_missing)}")
        print("   Эти пути есть в таблицах, но отсутствуют в files")
        print("   Они останутся с file_id = NULL")
    
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
