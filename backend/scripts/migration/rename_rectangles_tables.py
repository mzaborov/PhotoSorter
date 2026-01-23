#!/usr/bin/env python3
"""
Скрипт миграции: переименование таблиц rectangles.

Переименования:
- face_rectangles → photo_rectangles
- face_person_manual_assignments → person_rectangle_manual_assignments
- face_rectangle_id → rectangle_id (в таблицах привязок)
- Добавление колонки is_face в photo_rectangles (DEFAULT 1, NOT NULL)

ВАЖНО: Перед выполнением убедитесь, что:
1. Создана резервная копия БД
2. Все данные экспортированы в JSON (выполняется автоматически)
"""

import sqlite3
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# Путь к БД
DB_PATH = Path(__file__).resolve().parents[3] / "data" / "photosorter.db"
BACKUP_DIR = Path(__file__).resolve().parents[3] / "data" / "backups"


def get_connection():
    """Получить подключение к БД."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Включаем поддержку FOREIGN KEY constraints
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def calculate_file_hash(file_path: Path) -> str:
    """Вычислить SHA256 хеш файла."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def export_table_to_json(conn: sqlite3.Connection, table_name: str, output_file: Path) -> int:
    """Экспортировать таблицу в JSON файл."""
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    rows = cur.fetchall()
    
    # Конвертируем Row объекты в словари
    data = []
    for row in rows:
        row_dict = {}
        for key in row.keys():
            value = row[key]
            # Конвертируем BLOB в base64 для JSON
            if isinstance(value, bytes):
                import base64
                value = base64.b64encode(value).decode('utf-8')
                row_dict[key] = value
            else:
                row_dict[key] = value
        data.append(row_dict)
    
    # Сохраняем в JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return len(data)


def create_backup_and_export(conn: sqlite3.Connection, dry_run: bool = False) -> tuple[Path, Path]:
    """Создать резервную копию БД и экспортировать данные в JSON."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_DIR / f"migration_rename_rectangles_{timestamp}"
    
    if dry_run:
        print(f"\n[DRY RUN] Будет создана папка: {backup_dir}")
        return backup_dir, backup_dir
    
    # Создаем папку для backup
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Резервное копирование БД
    backup_db_path = backup_dir / f"photosorter.db.backup_{timestamp}"
    print(f"\n📦 Создание резервной копии БД...")
    shutil.copy2(DB_PATH, backup_db_path)
    backup_size = backup_db_path.stat().st_size
    print(f"   ✅ Backup создан: {backup_db_path} ({backup_size / 1024 / 1024:.2f} MB)")
    
    # 2. Экспорт данных в JSON
    print(f"\n📤 Экспорт данных в JSON...")
    metadata = {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "tables": {}
    }
    
    # Экспортируем face_rectangles
    face_rectangles_file = backup_dir / "face_rectangles.json"
    count = export_table_to_json(conn, "face_rectangles", face_rectangles_file)
    file_hash = calculate_file_hash(face_rectangles_file)
    metadata["tables"]["face_rectangles"] = {
        "count": count,
        "file": str(face_rectangles_file.name),
        "sha256": file_hash
    }
    print(f"   ✅ face_rectangles: {count} записей → {face_rectangles_file.name}")
    
    # Экспортируем face_person_manual_assignments
    if table_exists(conn, "face_person_manual_assignments"):
        assignments_file = backup_dir / "face_person_manual_assignments.json"
        count = export_table_to_json(conn, "face_person_manual_assignments", assignments_file)
        file_hash = calculate_file_hash(assignments_file)
        metadata["tables"]["face_person_manual_assignments"] = {
            "count": count,
            "file": str(assignments_file.name),
            "sha256": file_hash
        }
        print(f"   ✅ face_person_manual_assignments: {count} записей → {assignments_file.name}")
    
    # Экспортируем face_cluster_members
    if table_exists(conn, "face_cluster_members"):
        cluster_members_file = backup_dir / "face_cluster_members.json"
        count = export_table_to_json(conn, "face_cluster_members", cluster_members_file)
        file_hash = calculate_file_hash(cluster_members_file)
        metadata["tables"]["face_cluster_members"] = {
            "count": count,
            "file": str(cluster_members_file.name),
            "sha256": file_hash
        }
        print(f"   ✅ face_cluster_members: {count} записей → {cluster_members_file.name}")
    
    # Сохраняем метаданные
    metadata_file = backup_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Метаданные сохранены: {metadata_file.name}")
    
    return backup_dir, backup_db_path


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Проверить существование таблицы."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def check_integrity(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
    """Проверить целостность БД."""
    cur = conn.cursor()
    errors = []
    
    # PRAGMA integrity_check
    cur.execute("PRAGMA integrity_check")
    integrity_result = cur.fetchall()
    if integrity_result and integrity_result[0][0] != "ok":
        errors.append(f"Integrity check failed: {integrity_result[0][0]}")
    
    # PRAGMA foreign_key_check
    cur.execute("PRAGMA foreign_key_check")
    fk_errors = cur.fetchall()
    if fk_errors:
        for error in fk_errors:
            errors.append(f"Foreign key error: {error}")
    
    return len(errors) == 0, errors


def get_table_count(conn: sqlite3.Connection, table_name: str) -> int:
    """Получить количество записей в таблице."""
    if not table_exists(conn, table_name):
        return 0
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
    row = cur.fetchone()
    return row["cnt"] if row else 0


def validate_before_migration(conn: sqlite3.Connection) -> tuple[bool, dict]:
    """Валидация перед миграцией."""
    print("\n🔍 Валидация перед миграцией...")
    
    results = {}
    
    # Проверка существования таблиц
    tables_to_check = ["face_rectangles", "face_person_manual_assignments", "face_cluster_members"]
    for table in tables_to_check:
        exists = table_exists(conn, table)
        results[f"{table}_exists"] = exists
        if exists:
            count = get_table_count(conn, table)
            results[f"{table}_count"] = count
            print(f"   ✅ {table}: {count} записей")
        else:
            print(f"   ⚠️  {table}: таблица не существует")
    
    # Проверка целостности
    integrity_ok, errors = check_integrity(conn)
    results["integrity_ok"] = integrity_ok
    if integrity_ok:
        print(f"   ✅ Целостность БД: OK")
    else:
        print(f"   ❌ Целостность БД: ОШИБКИ")
        for error in errors:
            print(f"      {error}")
    
    return integrity_ok, results


def validate_after_migration(conn: sqlite3.Connection, before_counts: dict) -> bool:
    """Валидация после миграции."""
    print("\n🔍 Валидация после миграции...")
    
    # Проверка количества записей
    photo_rectangles_count = get_table_count(conn, "photo_rectangles")
    expected_count = before_counts.get("face_rectangles_count", 0)
    if photo_rectangles_count == expected_count:
        print(f"   ✅ photo_rectangles: {photo_rectangles_count} записей (ожидалось: {expected_count})")
    else:
        print(f"   ❌ photo_rectangles: {photo_rectangles_count} записей (ожидалось: {expected_count})")
        return False
    
    # Проверка is_face
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM photo_rectangles WHERE is_face IS NULL")
    null_count = cur.fetchone()["cnt"]
    if null_count == 0:
        print(f"   ✅ Все записи имеют is_face (NULL: {null_count})")
    else:
        print(f"   ❌ Найдено {null_count} записей с NULL is_face")
        return False
    
    # Проверка, что все rectangles в кластерах имеют is_face = 1
    cur.execute("""
        SELECT COUNT(*) as cnt FROM face_cluster_members fcm
        JOIN photo_rectangles pr ON pr.id = fcm.rectangle_id
        WHERE pr.is_face != 1
    """)
    invalid_cluster_count = cur.fetchone()["cnt"]
    if invalid_cluster_count == 0:
        print(f"   ✅ Все rectangles в кластерах имеют is_face = 1")
    else:
        print(f"   ❌ Найдено {invalid_cluster_count} rectangles в кластерах с is_face != 1")
        return False
    
    # Проверка целостности
    integrity_ok, errors = check_integrity(conn)
    if integrity_ok:
        print(f"   ✅ Целостность БД: OK")
    else:
        print(f"   ❌ Целостность БД: ОШИБКИ")
        for error in errors:
            print(f"      {error}")
        return False
    
    return True


def migrate(conn: sqlite3.Connection, dry_run: bool = False) -> bool:
    """Выполнить миграцию."""
    cur = conn.cursor()
    
    if dry_run:
        print("\n[DRY RUN] План миграции:")
        print("  1. Добавить колонку is_face в face_rectangles (DEFAULT 1)")
        print("  2. Переименовать face_rectangles → photo_rectangles")
        print("  3. Переименовать face_person_manual_assignments → person_rectangle_manual_assignments")
        print("  4. Переименовать face_rectangle_id → rectangle_id в person_rectangle_manual_assignments")
        print("  5. Переименовать face_rectangle_id → rectangle_id в face_cluster_members")
        print("  6. Обновить индексы")
        return True
    
    try:
        # Начинаем транзакцию
        conn.execute("BEGIN TRANSACTION")
        
        # Шаг 1: Добавить колонку is_face в face_rectangles (DEFAULT 1)
        print("\n📝 Шаг 1: Добавление колонки is_face в face_rectangles...")
        cur.execute("ALTER TABLE face_rectangles ADD COLUMN is_face INTEGER DEFAULT 1")
        
        # Убеждаемся, что все существующие записи имеют is_face = 1
        cur.execute("UPDATE face_rectangles SET is_face = 1 WHERE is_face IS NULL")
        updated = cur.rowcount
        print(f"   ✅ Колонка добавлена, обновлено записей: {updated}")
        
        # Шаг 2: Переименовать face_rectangles → photo_rectangles
        print("\n📝 Шаг 2: Переименование face_rectangles → photo_rectangles...")
        cur.execute("ALTER TABLE face_rectangles RENAME TO photo_rectangles")
        print(f"   ✅ Таблица переименована")
        
        # Шаг 3: Переименовать face_person_manual_assignments → person_rectangle_manual_assignments
        if table_exists(conn, "face_person_manual_assignments"):
            print("\n📝 Шаг 3: Переименование face_person_manual_assignments → person_rectangle_manual_assignments...")
            cur.execute("ALTER TABLE face_person_manual_assignments RENAME TO person_rectangle_manual_assignments")
            print(f"   ✅ Таблица переименована")
        else:
            print("\n⚠️  Шаг 3: Таблица face_person_manual_assignments не существует, пропускаем")
        
        # Шаг 4: Переименовать face_rectangle_id → rectangle_id в person_rectangle_manual_assignments
        if table_exists(conn, "person_rectangle_manual_assignments"):
            print("\n📝 Шаг 4: Переименование face_rectangle_id → rectangle_id в person_rectangle_manual_assignments...")
            # SQLite не поддерживает ALTER COLUMN, нужно пересоздать таблицу
            cur.execute("""
                CREATE TABLE person_rectangle_manual_assignments_new (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    rectangle_id        INTEGER NOT NULL,
                    person_id           INTEGER NOT NULL,
                    source              TEXT NOT NULL,
                    confidence          REAL,
                    created_at          TEXT NOT NULL
                )
            """)
            
            cur.execute("""
                INSERT INTO person_rectangle_manual_assignments_new (
                    id, rectangle_id, person_id, source, confidence, created_at
                )
                SELECT 
                    id, face_rectangle_id, person_id, source, confidence, created_at
                FROM person_rectangle_manual_assignments
            """)
            copied = cur.rowcount
            print(f"   ✅ Скопировано записей: {copied}")
            
            cur.execute("DROP TABLE person_rectangle_manual_assignments")
            cur.execute("ALTER TABLE person_rectangle_manual_assignments_new RENAME TO person_rectangle_manual_assignments")
            print(f"   ✅ Таблица пересоздана с новым именем колонки")
        
        # Шаг 5: Переименовать face_rectangle_id → rectangle_id в face_cluster_members
        if table_exists(conn, "face_cluster_members"):
            print("\n📝 Шаг 5: Переименование face_rectangle_id → rectangle_id в face_cluster_members...")
            
            # Проверка: все rectangles в кластерах должны иметь is_face = 1
            cur.execute("""
                SELECT COUNT(*) as cnt FROM face_cluster_members fcm
                JOIN photo_rectangles pr ON pr.id = fcm.face_rectangle_id
                WHERE pr.is_face != 1
            """)
            invalid_count = cur.fetchone()["cnt"]
            if invalid_count > 0:
                raise ValueError(f"Найдено {invalid_count} rectangles в кластерах с is_face != 1. Это не должно быть возможно!")
            
            # SQLite не поддерживает ALTER COLUMN, нужно пересоздать таблицу
            cur.execute("""
                CREATE TABLE face_cluster_members_new (
                    cluster_id          INTEGER NOT NULL,
                    rectangle_id       INTEGER NOT NULL,
                    PRIMARY KEY (cluster_id, rectangle_id)
                )
            """)
            
            cur.execute("""
                INSERT INTO face_cluster_members_new (cluster_id, rectangle_id)
                SELECT cluster_id, face_rectangle_id
                FROM face_cluster_members
            """)
            copied = cur.rowcount
            print(f"   ✅ Скопировано записей: {copied}")
            
            cur.execute("DROP TABLE face_cluster_members")
            cur.execute("ALTER TABLE face_cluster_members_new RENAME TO face_cluster_members")
            print(f"   ✅ Таблица пересоздана с новым именем колонки")
        
        # Шаг 6: Обновить индексы
        print("\n📝 Шаг 6: Обновление индексов...")
        
        # Удаляем старые индексы
        old_indexes = [
            "idx_face_rect_run",
            "idx_face_rect_file",
            "idx_face_rect_file_id",
            "idx_face_rect_archive_scope",
            "idx_face_person_manual_assignments_face",
            "idx_face_person_manual_assignments_person",
            "idx_face_person_manual_assignments_unique",
            "idx_face_cluster_members_face",
        ]
        for idx_name in old_indexes:
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
        
        # Создаем новые индексы
        new_indexes = [
            ("idx_photo_rect_run", "CREATE INDEX idx_photo_rect_run ON photo_rectangles(run_id)"),
            ("idx_photo_rect_file", "CREATE INDEX idx_photo_rect_file ON photo_rectangles(file_id)"),
            ("idx_photo_rect_file_id", "CREATE INDEX idx_photo_rect_file_id ON photo_rectangles(file_id)"),
            ("idx_photo_rect_archive_scope", "CREATE INDEX idx_photo_rect_archive_scope ON photo_rectangles(archive_scope)"),
            ("idx_photo_rect_is_face", "CREATE INDEX idx_photo_rect_is_face ON photo_rectangles(is_face)"),
            ("idx_person_rectangle_manual_assignments_rect", "CREATE INDEX idx_person_rectangle_manual_assignments_rect ON person_rectangle_manual_assignments(rectangle_id)"),
            ("idx_person_rectangle_manual_assignments_person", "CREATE INDEX idx_person_rectangle_manual_assignments_person ON person_rectangle_manual_assignments(person_id)"),
            ("idx_person_rectangle_manual_assignments_unique", "CREATE UNIQUE INDEX idx_person_rectangle_manual_assignments_unique ON person_rectangle_manual_assignments(rectangle_id, person_id)"),
            ("idx_face_cluster_members_rect", "CREATE INDEX idx_face_cluster_members_rect ON face_cluster_members(rectangle_id)"),
        ]
        
        for idx_name, idx_sql in new_indexes:
            cur.execute(idx_sql)
            print(f"   ✅ Создан индекс: {idx_name}")
        
        # Коммитим транзакцию
        conn.commit()
        print("\n✅ Миграция завершена успешно!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Ошибка при миграции: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Переименование таблиц rectangles")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    parser.add_argument("--yes", action="store_true", help="Автоматически подтвердить миграцию")
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ БД не найдена: {DB_PATH}")
        return 1
    
    print("=" * 70)
    print("МИГРАЦИЯ: Переименование таблиц rectangles")
    print("=" * 70)
    print(f"\nБД: {DB_PATH}")
    print(f"Режим: {'DRY RUN' if args.dry_run else 'ВЫПОЛНЕНИЕ'}")
    
    conn = get_connection()
    try:
        # Валидация перед миграцией
        integrity_ok, before_results = validate_before_migration(conn)
        if not integrity_ok:
            print("\n❌ Валидация не прошла. Исправьте ошибки перед миграцией.")
            return 1
        
        # Сохраняем количество записей для валидации после миграции
        before_counts = {
            "face_rectangles_count": before_results.get("face_rectangles_count", 0),
        }
        
        if not args.dry_run:
            if not args.yes:
                print("\n⚠️  ВНИМАНИЕ: Это переименует таблицы и добавит колонку is_face.")
                print("   Убедитесь, что есть резервная копия БД!")
                response = input("\nПродолжить? (yes/no): ")
                if response.lower() != "yes":
                    print("Отменено.")
                    return 1
            else:
                print("\n⚠️  ВНИМАНИЕ: Это переименует таблицы и добавит колонку is_face.")
                print("   Продолжаем (--yes указан)...")
        
        # Создание backup и экспорт данных
        backup_dir, backup_db_path = create_backup_and_export(conn, dry_run=args.dry_run)
        
        # Выполнение миграции
        success = migrate(conn, dry_run=args.dry_run)
        
        if success and not args.dry_run:
            # Валидация после миграции
            validation_ok = validate_after_migration(conn, before_counts)
            if not validation_ok:
                print("\n❌ Валидация после миграции не прошла!")
                print(f"   Восстановите БД из backup: {backup_db_path}")
                return 1
            
            print(f"\n✅ Миграция завершена успешно!")
            print(f"   Backup сохранен в: {backup_dir}")
        elif args.dry_run:
            print(f"\n[DRY RUN] Миграция не выполнена. Используйте без --dry-run для выполнения.")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
