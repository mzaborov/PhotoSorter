#!/usr/bin/env python3
"""
Скрипт миграции: добавление NOT NULL и FOREIGN KEY constraints для file_id.

SQLite не поддерживает ALTER TABLE для добавления NOT NULL или FOREIGN KEY к существующим колонкам,
поэтому пересоздаем таблицы с нужными constraints.

ВАЖНО: Перед выполнением убедитесь, что:
1. Все file_id заполнены (кроме проблемных записей, которые останутся с NULL)
2. Создана резервная копия БД
"""

import sqlite3
import sys
from pathlib import Path

# Путь к БД
DB_PATH = Path(__file__).resolve().parents[3] / "data" / "photosorter.db"


def get_connection():
    """Получить подключение к БД."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def check_null_file_ids(conn: sqlite3.Connection, table: str, file_id_column: str = "file_id") -> int:
    """Проверить количество NULL значений в file_id."""
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) as cnt FROM {table} WHERE {file_id_column} IS NULL")
    row = cur.fetchone()
    return row["cnt"] if row else 0


def migrate_table(
    conn: sqlite3.Connection,
    table_name: str,
    create_table_sql: str,
    copy_data_sql: str,
    dry_run: bool = False,
) -> None:
    """Пересоздать таблицу с NOT NULL и FOREIGN KEY constraints."""
    cur = conn.cursor()
    
    # Проверяем количество NULL значений
    null_count = check_null_file_ids(conn, table_name)
    if null_count > 0:
        print(f"  ⚠️  В таблице {table_name} найдено {null_count} записей с NULL file_id")
        print(f"     Эти записи будут пропущены при копировании")
    
    if dry_run:
        print(f"  [DRY RUN] Будет пересоздана таблица {table_name}")
        print(f"  [DRY RUN] SQL создания: {create_table_sql[:100]}...")
        return
    
    # Создаем временную таблицу с новыми constraints
    temp_table = f"{table_name}_new"
    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
    # Заменяем имя таблицы в SQL создания
    create_sql_for_temp = create_table_sql.replace(f"CREATE TABLE {table_name}", f"CREATE TABLE {temp_table}")
    cur.execute(create_sql_for_temp)
    
    # Копируем данные (только записи с заполненным file_id)
    # Заменяем {table_name} в SQL копирования на temp_table
    copy_sql_for_temp = copy_data_sql.replace("{table_name}", temp_table)
    cur.execute(copy_sql_for_temp)
    copied = cur.rowcount
    print(f"  Скопировано записей: {copied}")
    
    # Удаляем старую таблицу
    cur.execute(f"DROP TABLE {table_name}")
    
    # Переименовываем новую таблицу
    cur.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
    
    # Восстанавливаем индексы (они удаляются при DROP TABLE)
    # Индексы будут созданы в create_table_sql через CREATE INDEX IF NOT EXISTS


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Добавить NOT NULL и FOREIGN KEY constraints для file_id")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    parser.add_argument("--yes", action="store_true", help="Автоматически подтвердить миграцию (без интерактивного запроса)")
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ БД не найдена: {DB_PATH}")
        return 1
    
    print("=" * 70)
    print("МИГРАЦИЯ: Добавление NOT NULL и FOREIGN KEY constraints для file_id")
    print("=" * 70)
    print(f"\nБД: {DB_PATH}")
    print(f"Режим: {'DRY RUN' if args.dry_run else 'ВЫПОЛНЕНИЕ'}")
    
    if not args.dry_run:
        if not args.yes:
            response = input("\n⚠️  ВНИМАНИЕ: Это пересоздаст таблицы. Убедитесь, что есть резервная копия БД!\nПродолжить? (yes/no): ")
            if response.lower() != "yes":
                print("Отменено.")
                return 1
        else:
            print("\n⚠️  ВНИМАНИЕ: Это пересоздаст таблицы. Убедитесь, что есть резервная копия БД!")
            print("Продолжаем (--yes указан)...")
    
    conn = get_connection()
    try:
        # Таблицы для миграции
        tables = [
            {
                "name": "face_rectangles",
                "create_sql": """
                    CREATE TABLE face_rectangles (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id              INTEGER,
                        file_path           TEXT NOT NULL,
                        file_id             INTEGER NOT NULL,
                        face_index          INTEGER NOT NULL,
                        bbox_x              INTEGER NOT NULL,
                        bbox_y              INTEGER NOT NULL,
                        bbox_w              INTEGER NOT NULL,
                        bbox_h              INTEGER NOT NULL,
                        confidence          REAL,
                        presence_score      REAL,
                        thumb_jpeg          BLOB,
                        manual_person       TEXT,
                        ignore_flag         INTEGER DEFAULT 0,
                        created_at          TEXT NOT NULL,
                        is_manual           INTEGER DEFAULT 0,
                        manual_created_at  TEXT,
                        archive_scope       TEXT,
                        FOREIGN KEY (file_id) REFERENCES files(id)
                    );
                """,
                "copy_sql": """
                    INSERT INTO {table_name} (
                        id, run_id, file_path, file_id, face_index,
                        bbox_x, bbox_y, bbox_w, bbox_h,
                        confidence, presence_score, thumb_jpeg, manual_person,
                        ignore_flag, created_at, is_manual, manual_created_at, archive_scope
                    )
                    SELECT 
                        id, run_id, file_path, file_id, face_index,
                        bbox_x, bbox_y, bbox_w, bbox_h,
                        confidence, presence_score, thumb_jpeg, manual_person,
                        ignore_flag, created_at, is_manual, manual_created_at, archive_scope
                    FROM face_rectangles
                    WHERE file_id IS NOT NULL
                """,
            },
            {
                "name": "person_rectangles",
                "create_sql": """
                    CREATE TABLE person_rectangles (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        pipeline_run_id     INTEGER NOT NULL,
                        file_path           TEXT NOT NULL,
                        file_id             INTEGER NOT NULL,
                        frame_idx           INTEGER,
                        bbox_x              INTEGER NOT NULL,
                        bbox_y              INTEGER NOT NULL,
                        bbox_w              INTEGER NOT NULL,
                        bbox_h              INTEGER NOT NULL,
                        person_id           INTEGER NOT NULL,
                        created_at          TEXT NOT NULL,
                        FOREIGN KEY (file_id) REFERENCES files(id),
                        FOREIGN KEY (person_id) REFERENCES persons(id)
                    );
                """,
                "copy_sql": """
                    INSERT INTO {table_name} (
                        id, pipeline_run_id, file_path, file_id, frame_idx,
                        bbox_x, bbox_y, bbox_w, bbox_h, person_id, created_at
                    )
                    SELECT 
                        id, pipeline_run_id, file_path, file_id, frame_idx,
                        bbox_x, bbox_y, bbox_w, bbox_h, person_id, created_at
                    FROM person_rectangles
                    WHERE file_id IS NOT NULL
                """,
            },
            {
                "name": "file_persons",
                "create_sql": """
                    CREATE TABLE file_persons (
                        pipeline_run_id     INTEGER NOT NULL,
                        file_path           TEXT NOT NULL,
                        file_id             INTEGER NOT NULL,
                        person_id           INTEGER NOT NULL,
                        created_at          TEXT NOT NULL,
                        PRIMARY KEY (pipeline_run_id, file_id, person_id),
                        FOREIGN KEY (file_id) REFERENCES files(id),
                        FOREIGN KEY (person_id) REFERENCES persons(id)
                    );
                """,
                "copy_sql": """
                    INSERT INTO {table_name} (
                        pipeline_run_id, file_path, file_id, person_id, created_at
                    )
                    SELECT 
                        pipeline_run_id, file_path, file_id, person_id, created_at
                    FROM file_persons
                    WHERE file_id IS NOT NULL
                """,
            },
            {
                "name": "file_groups",
                "create_sql": """
                    CREATE TABLE file_groups (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        pipeline_run_id     INTEGER NOT NULL,
                        file_path           TEXT NOT NULL,
                        file_id             INTEGER NOT NULL,
                        group_path          TEXT NOT NULL,
                        created_at          TEXT NOT NULL,
                        UNIQUE(pipeline_run_id, file_id, group_path),
                        FOREIGN KEY (file_id) REFERENCES files(id)
                    );
                """,
                "copy_sql": """
                    INSERT INTO {table_name} (
                        id, pipeline_run_id, file_path, file_id, group_path, created_at
                    )
                    SELECT 
                        id, pipeline_run_id, file_path, file_id, group_path, created_at
                    FROM file_groups
                    WHERE file_id IS NOT NULL
                """,
            },
            {
                "name": "file_group_persons",
                "create_sql": """
                    CREATE TABLE file_group_persons (
                        pipeline_run_id     INTEGER NOT NULL,
                        file_path           TEXT NOT NULL,
                        file_id             INTEGER NOT NULL,
                        group_path          TEXT NOT NULL,
                        person_id           INTEGER NOT NULL,
                        created_at          TEXT NOT NULL,
                        PRIMARY KEY (pipeline_run_id, file_id, group_path, person_id),
                        FOREIGN KEY (file_id) REFERENCES files(id),
                        FOREIGN KEY (person_id) REFERENCES persons(id)
                    );
                """,
                "copy_sql": """
                    INSERT INTO {table_name} (
                        pipeline_run_id, file_path, file_id, group_path, person_id, created_at
                    )
                    SELECT 
                        pipeline_run_id, file_path, file_id, group_path, person_id, created_at
                    FROM file_group_persons
                    WHERE file_id IS NOT NULL
                """,
            },
            {
                "name": "files_manual_labels",
                "create_sql": """
                    CREATE TABLE files_manual_labels (
                        pipeline_run_id       INTEGER NOT NULL,
                        path                  TEXT,
                        file_id               INTEGER NOT NULL,
                        faces_manual_label    TEXT,
                        faces_manual_at       TEXT,
                        people_no_face_manual INTEGER NOT NULL DEFAULT 0,
                        people_no_face_person TEXT,
                        animals_manual        INTEGER NOT NULL DEFAULT 0,
                        animals_manual_kind   TEXT,
                        animals_manual_at     TEXT,
                        quarantine_manual      INTEGER NOT NULL DEFAULT 0,
                        quarantine_manual_at  TEXT,
                        PRIMARY KEY (pipeline_run_id, file_id),
                        FOREIGN KEY (file_id) REFERENCES files(id)
                    );
                """,
                "copy_sql": """
                    INSERT INTO {table_name} (
                        pipeline_run_id, path, file_id, faces_manual_label, faces_manual_at,
                        people_no_face_manual, people_no_face_person,
                        animals_manual, animals_manual_kind, animals_manual_at,
                        quarantine_manual, quarantine_manual_at
                    )
                    SELECT 
                        pipeline_run_id, path, file_id, faces_manual_label, faces_manual_at,
                        people_no_face_manual, people_no_face_person,
                        animals_manual, animals_manual_kind, animals_manual_at,
                        quarantine_manual, quarantine_manual_at
                    FROM files_manual_labels
                    WHERE file_id IS NOT NULL
                """,
            },
            {
                "name": "video_manual_frames",
                "create_sql": """
                    CREATE TABLE video_manual_frames (
                        pipeline_run_id     INTEGER NOT NULL,
                        path                TEXT NOT NULL,
                        file_id             INTEGER NOT NULL,
                        frame_idx           INTEGER NOT NULL,
                        t_sec               REAL,
                        rects_json          TEXT,
                        updated_at          TEXT NOT NULL,
                        PRIMARY KEY (pipeline_run_id, file_id, frame_idx),
                        FOREIGN KEY (file_id) REFERENCES files(id)
                    );
                """,
                "copy_sql": """
                    INSERT INTO {table_name} (
                        pipeline_run_id, path, file_id, frame_idx, t_sec, rects_json, updated_at
                    )
                    SELECT 
                        pipeline_run_id, path, file_id, frame_idx, t_sec, rects_json, updated_at
                    FROM video_manual_frames
                    WHERE file_id IS NOT NULL;
                """,
            },
        ]
        
        # Создаем индексы после пересоздания таблиц
        indexes = [
            ("face_rectangles", "CREATE INDEX IF NOT EXISTS idx_face_rect_file_id ON face_rectangles(file_id);"),
            ("face_rectangles", "CREATE INDEX IF NOT EXISTS idx_face_rect_file ON face_rectangles(file_path);"),
            ("face_rectangles", "CREATE INDEX IF NOT EXISTS idx_face_rect_run ON face_rectangles(run_id);"),
            ("person_rectangles", "CREATE INDEX IF NOT EXISTS idx_person_rect_file_id ON person_rectangles(file_id);"),
            ("person_rectangles", "CREATE INDEX IF NOT EXISTS idx_person_rect_file ON person_rectangles(file_path);"),
            ("person_rectangles", "CREATE INDEX IF NOT EXISTS idx_person_rect_run ON person_rectangles(pipeline_run_id);"),
            ("file_persons", "CREATE INDEX IF NOT EXISTS idx_file_persons_file_id ON file_persons(file_id);"),
            ("file_persons", "CREATE INDEX IF NOT EXISTS idx_file_persons_file ON file_persons(file_path);"),
            ("file_persons", "CREATE INDEX IF NOT EXISTS idx_file_persons_run ON file_persons(pipeline_run_id);"),
            ("file_persons", "CREATE INDEX IF NOT EXISTS idx_file_persons_person ON file_persons(person_id);"),
            ("file_groups", "CREATE INDEX IF NOT EXISTS idx_file_groups_file_id ON file_groups(file_id);"),
            ("file_groups", "CREATE INDEX IF NOT EXISTS idx_file_groups_file ON file_groups(file_path);"),
            ("file_groups", "CREATE INDEX IF NOT EXISTS idx_file_groups_run ON file_groups(pipeline_run_id);"),
            ("file_group_persons", "CREATE INDEX IF NOT EXISTS idx_file_group_persons_file_id ON file_group_persons(file_id);"),
            ("file_group_persons", "CREATE INDEX IF NOT EXISTS idx_file_group_persons_file ON file_group_persons(file_path);"),
            ("file_group_persons", "CREATE INDEX IF NOT EXISTS idx_file_group_persons_run ON file_group_persons(pipeline_run_id);"),
            ("file_group_persons", "CREATE INDEX IF NOT EXISTS idx_file_group_persons_person ON file_group_persons(person_id);"),
            ("files_manual_labels", "CREATE INDEX IF NOT EXISTS idx_files_manual_labels_file_id ON files_manual_labels(file_id);"),
            ("files_manual_labels", "CREATE INDEX IF NOT EXISTS idx_files_manual_labels_run ON files_manual_labels(pipeline_run_id);"),
            ("video_manual_frames", "CREATE INDEX IF NOT EXISTS idx_video_manual_frames_file_id ON video_manual_frames(file_id);"),
            ("video_manual_frames", "CREATE INDEX IF NOT EXISTS idx_video_manual_frames_path ON video_manual_frames(path);"),
        ]
        
        print(f"\n📋 Таблицы для миграции: {len(tables)}")
        for table_info in tables:
            print(f"\n🔄 Таблица: {table_info['name']}")
            migrate_table(
                conn,
                table_info["name"],
                table_info["create_sql"],
                table_info["copy_sql"],
                dry_run=args.dry_run,
            )
        
        if not args.dry_run:
            # Создаем индексы
            print(f"\n📇 Создание индексов...")
            for table_name, index_sql in indexes:
                cur = conn.cursor()
                cur.execute(index_sql)
                print(f"  ✅ {index_sql.split()[-1]}")
            
            conn.commit()
            print(f"\n✅ Миграция завершена успешно!")
        else:
            print(f"\n[DRY RUN] Миграция не выполнена. Используйте без --dry-run для выполнения.")
        
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
