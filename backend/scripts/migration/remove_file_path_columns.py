#!/usr/bin/env python3
"""
Удаление избыточных колонок file_path/path из таблиц (приведение к 3NF).

Таблицы для обработки:
- face_rectangles: file_path
- person_rectangles: file_path
- file_persons: file_path
- file_groups: file_path
- file_group_persons: file_path (если есть)
- files_manual_labels: path
- video_manual_frames: path

Эти колонки дублируют данные из files.path через file_id и нарушают 3NF.
"""

import sys
import sqlite3
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection, DB_PATH


def get_table_schema(conn, table_name: str):
    """Получает схему таблицы."""
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = {}
    for row in cur.fetchall():
        col_name = row[1]
        col_type = row[2]
        notnull = row[3] == 1
        pk = row[5] == 1
        columns[col_name] = {
            "type": col_type,
            "notnull": notnull,
            "pk": pk
        }
    return columns


def remove_column_from_table(conn, table_name: str, column_name: str, dry_run: bool = False):
    """Удаляет колонку из таблицы (пересоздает таблицу)."""
    cur = conn.cursor()
    
    # Проверяем, есть ли колонка
    schema = get_table_schema(conn, table_name)
    if column_name not in schema:
        print(f"  ✅ {table_name}.{column_name}: колонка уже удалена")
        return 0
    
    if dry_run:
        print(f"  [DRY RUN] {table_name}.{column_name}: будет удалена")
        return 1
    
    # Получаем CREATE TABLE SQL для извлечения PRIMARY KEY и других constraints
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    create_sql_row = cur.fetchone()
    create_sql = create_sql_row[0] if create_sql_row else None
    
    # Проверяем, есть ли составной PRIMARY KEY в CREATE TABLE
    has_composite_pk = False
    composite_pk_cols = []
    if create_sql and "PRIMARY KEY" in create_sql.upper():
        import re
        # Ищем PRIMARY KEY (col1, col2, ...)
        pk_match = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', create_sql, re.IGNORECASE)
        if pk_match:
            has_composite_pk = True
            composite_pk_cols = [c.strip() for c in pk_match.group(1).split(',')]
            # Убираем удаляемую колонку из составного PK
            composite_pk_cols = [c for c in composite_pk_cols if c != column_name]
    
    # Получаем полное определение таблицы из PRAGMA
    cur.execute("PRAGMA table_info({})".format(table_name))
    table_info = cur.fetchall()
    
    # Строим CREATE TABLE для новой таблицы
    columns_def = []
    pk_columns = []
    has_autoinc = False
    
    for row in table_info:
        col_name = row[1]
        if col_name == column_name:
            continue  # Пропускаем удаляемую колонку
        
        col_type = row[2]
        notnull = row[3] == 1
        default_val = row[4]
        pk = row[5] == 1
        
        col_def = f"{col_name} {col_type}"
        if notnull:
            col_def += " NOT NULL"
        if default_val is not None:
            col_def += f" DEFAULT {default_val}"
        
        # Добавляем PRIMARY KEY только если нет составного PK
        if pk and not has_composite_pk:
            pk_columns.append(col_name)
            if "AUTOINCREMENT" in col_type.upper():
                has_autoinc = True
                col_def += " PRIMARY KEY AUTOINCREMENT"
            elif len(pk_columns) == 1:  # Только если это первый PK
                col_def += " PRIMARY KEY"
        
        columns_def.append(col_def)
    
    # Если есть составной PRIMARY KEY, добавляем его в конец
    if has_composite_pk and composite_pk_cols:
        columns_def.append(f"PRIMARY KEY ({', '.join(composite_pk_cols)})")
    
    # Создаем временную таблицу без удаляемой колонки
    temp_table = f"{table_name}_new"
    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
    
    create_new_sql = f"""
        CREATE TABLE {temp_table} (
            {', '.join(columns_def)}
        )
    """
    
    cur.execute(create_new_sql)
    
    # Копируем данные (только колонки, которые остаются)
    all_columns = [col[1] for col in table_info if col[1] != column_name]
    columns_list = ", ".join(all_columns)
    cur.execute(f"""
        INSERT INTO {temp_table} ({columns_list})
        SELECT {columns_list}
        FROM {table_name}
    """)
    copied = cur.rowcount
    
    # Отключаем проверку FOREIGN KEY для удаления таблицы
    cur.execute("PRAGMA foreign_keys = OFF")
    
    # Удаляем старую таблицу
    cur.execute(f"DROP TABLE {table_name}")
    
    # Переименовываем новую таблицу
    cur.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
    
    # Включаем обратно проверку FOREIGN KEY
    cur.execute("PRAGMA foreign_keys = ON")
    
    # Восстанавливаем индексы (сохраняем список ДО удаления таблицы)
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table_name,))
    old_indexes = [(row[0], row[1]) for row in cur.fetchall()]
    
    for idx_name, idx_sql in old_indexes:
        if idx_sql:
            # Заменяем имя таблицы в SQL индекса
            idx_sql_new = idx_sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS").replace("CREATE UNIQUE INDEX", "CREATE UNIQUE INDEX IF NOT EXISTS")
            # Таблица уже переименована
            try:
                cur.execute(idx_sql_new)
            except Exception as e:
                print(f"    ⚠️  Ошибка создания индекса {idx_name}: {e}")
    
    print(f"  ✅ {table_name}.{column_name}: удалена ({copied} записей скопировано)")
    return 1


def remove_file_path_columns(conn, dry_run: bool = False):
    """Удаляет колонки file_path/path из всех таблиц."""
    tables_to_process = [
        ("photo_rectangles", "file_path"),
        ("person_rectangles", "file_path"),
        ("file_persons", "file_path"),
        ("file_groups", "file_path"),
        ("file_group_persons", "file_path"),  # Проверим, есть ли она
        ("files_manual_labels", "path"),
        ("video_manual_frames", "path"),
    ]
    
    total_removed = 0
    for table_name, column_name in tables_to_process:
        # Проверяем, существует ли таблица
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cur.fetchone():
            print(f"  ⚠️  {table_name}: таблица не найдена, пропускаем")
            continue
        
        removed = remove_column_from_table(conn, table_name, column_name, dry_run=dry_run)
        total_removed += removed
    
    return total_removed


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Удаление избыточных колонок file_path/path из таблиц")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    parser.add_argument("--yes", action="store_true", help="Автоматически подтвердить удаление")
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ БД не найдена: {DB_PATH}")
        return 1
    
    print("=" * 70)
    print("УДАЛЕНИЕ ИЗБЫТОЧНЫХ КОЛОНОК file_path/path ИЗ ТАБЛИЦ")
    print("=" * 70)
    print(f"\nБД: {DB_PATH}")
    print(f"Режим: {'DRY RUN' if args.dry_run else 'ВЫПОЛНЕНИЕ'}")
    
    if not args.dry_run:
        if not args.yes:
            response = input("\n⚠️  ВНИМАНИЕ: Это пересоздаст 7 таблиц. Убедитесь, что есть резервная копия БД!\nПродолжить? (yes/no): ")
            if response.lower() != "yes":
                print("Отменено.")
                return 1
        else:
            print("\n⚠️  ВНИМАНИЕ: Это пересоздаст 7 таблиц. Убедитесь, что есть резервная копия БД!")
            print("Продолжаем (--yes указан)...")
    
    # Создаем резервную копию
    if not args.dry_run:
        backup_path = DB_PATH.parent / "backups" / f"photosorter_backup_before_remove_file_path_{Path(__file__).stem}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        print(f"\n✅ Резервная копия создана: {backup_path}")
    
    conn = get_connection()
    # Отключаем FOREIGN KEY на уровне соединения перед началом работы
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        removed = remove_file_path_columns(conn, dry_run=args.dry_run)
        
        if not args.dry_run:
            conn.commit()
            print(f"\n✅ Миграция завершена. Удалено колонок: {removed}")
        else:
            print(f"\n[DRY RUN] Будет удалено колонок: {removed}")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Включаем обратно проверку FOREIGN KEY
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
