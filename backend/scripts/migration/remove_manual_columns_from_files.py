#!/usr/bin/env python3
"""
Удаление избыточных колонок *_manual_* из таблицы files (приведение к 3NF).

Колонки для удаления:
- faces_manual_label
- faces_manual_at
- people_no_face_manual
- animals_manual
- animals_manual_kind
- animals_manual_at

Эти колонки дублируют данные из files_manual_labels и нарушают 3NF.
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


def remove_columns_from_files(conn, dry_run: bool = False):
    """Удаляет колонки *_manual_* из таблицы files."""
    cur = conn.cursor()
    
    # Колонки для удаления
    columns_to_remove = [
        "faces_manual_label",
        "faces_manual_at",
        "people_no_face_manual",
        "animals_manual",
        "animals_manual_kind",
        "animals_manual_at",
    ]
    
    # Проверяем, какие колонки реально есть
    schema = get_table_schema(conn, "files")
    existing_columns = [col for col in columns_to_remove if col in schema]
    
    if not existing_columns:
        print("✅ Все колонки уже удалены")
        return 0
    
    print(f"\nКолонки для удаления: {len(existing_columns)}")
    for col in existing_columns:
        print(f"  - {col}")
    
    if dry_run:
        print("\n[DRY RUN] Колонки будут удалены (пересоздание таблицы)")
        return len(existing_columns)
    
    # SQLite не поддерживает DROP COLUMN, нужно пересоздать таблицу
    # Получаем все колонки кроме удаляемых
    all_columns = list(schema.keys())
    columns_to_keep = [col for col in all_columns if col not in existing_columns]
    
    # Получаем полное определение таблицы из PRAGMA
    cur.execute("PRAGMA table_info(files)")
    table_info = cur.fetchall()
    
    # Строим CREATE TABLE для новой таблицы
    columns_def = []
    pk_columns = []
    
    for row in table_info:
        col_name = row[1]
        if col_name in existing_columns:
            continue  # Пропускаем удаляемые колонки
        
        col_type = row[2]
        notnull = row[3] == 1
        default_val = row[4]
        pk = row[5] == 1
        
        col_def = f"{col_name} {col_type}"
        if notnull:
            col_def += " NOT NULL"
        if default_val is not None:
            col_def += f" DEFAULT {default_val}"
        if pk:
            pk_columns.append(col_name)
            if "AUTOINCREMENT" in col_type.upper() or col_type.upper() == "INTEGER":
                col_def += " PRIMARY KEY AUTOINCREMENT"
            else:
                col_def += " PRIMARY KEY"
        
        columns_def.append(col_def)
    
    # Создаем временную таблицу без удаляемых колонок
    temp_table = "files_new"
    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
    
    create_new_sql = f"""
        CREATE TABLE {temp_table} (
            {', '.join(columns_def)}
        )
    """
    
    cur.execute(create_new_sql)
    
    # Копируем данные (только колонки, которые остаются)
    columns_list = ", ".join(columns_to_keep)
    cur.execute(f"""
        INSERT INTO {temp_table} ({columns_list})
        SELECT {columns_list}
        FROM files
    """)
    copied = cur.rowcount
    print(f"\nСкопировано записей: {copied}")
    
    # Удаляем старую таблицу (FOREIGN KEY уже отключен на уровне соединения)
    cur.execute("DROP TABLE files")
    
    # Переименовываем новую таблицу
    cur.execute(f"ALTER TABLE {temp_table} RENAME TO files")
    
    # Восстанавливаем индексы
    cur.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='files'")
    indexes = [row[0] for row in cur.fetchall() if row[0]]
    for idx_sql in indexes:
        # Заменяем имя таблицы в SQL индекса
        idx_sql_new = idx_sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS")
        try:
            cur.execute(idx_sql_new)
        except Exception as e:
            print(f"  ⚠️  Ошибка создания индекса: {e}")
    
    print(f"\n✅ Удалено колонок: {len(existing_columns)}")
    return len(existing_columns)


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Удаление избыточных колонок *_manual_* из files")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    parser.add_argument("--yes", action="store_true", help="Автоматически подтвердить удаление")
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ БД не найдена: {DB_PATH}")
        return 1
    
    print("=" * 70)
    print("УДАЛЕНИЕ ИЗБЫТОЧНЫХ КОЛОНОК *_manual_* ИЗ files")
    print("=" * 70)
    print(f"\nБД: {DB_PATH}")
    print(f"Режим: {'DRY RUN' if args.dry_run else 'ВЫПОЛНЕНИЕ'}")
    
    if not args.dry_run:
        if not args.yes:
            response = input("\n⚠️  ВНИМАНИЕ: Это пересоздаст таблицу files. Убедитесь, что есть резервная копия БД!\nПродолжить? (yes/no): ")
            if response.lower() != "yes":
                print("Отменено.")
                return 1
        else:
            print("\n⚠️  ВНИМАНИЕ: Это пересоздаст таблицу files. Убедитесь, что есть резервная копия БД!")
            print("Продолжаем (--yes указан)...")
    
    # Создаем резервную копию
    if not args.dry_run:
        backup_path = DB_PATH.parent / "backups" / f"photosorter_backup_before_remove_manual_{Path(__file__).stem}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        print(f"\n✅ Резервная копия создана: {backup_path}")
    
    conn = get_connection()
    # Отключаем FOREIGN KEY на уровне соединения перед началом работы
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        removed = remove_columns_from_files(conn, dry_run=args.dry_run)
        
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
