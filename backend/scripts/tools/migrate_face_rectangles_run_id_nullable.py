#!/usr/bin/env python3
"""
Миграция: делает run_id опциональным (NULL) в face_rectangles для поддержки архивных лиц.
SQLite не поддерживает ALTER TABLE для изменения NOT NULL, поэтому пересоздаём таблицу.
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


def main() -> int:
    print("=" * 60)
    print("МИГРАЦИЯ: делаем run_id опциональным в face_rectangles")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем текущее состояние
    cur.execute("PRAGMA table_info(face_rectangles)")
    columns = cur.fetchall()
    run_id_col = next((c for c in columns if c["name"] == "run_id"), None)
    
    if run_id_col and run_id_col["notnull"] == 0:
        print("run_id уже опциональный. Миграция не требуется.")
        return 0
    
    print("Пересоздаём таблицу face_rectangles с опциональным run_id...")
    print()
    
    # 1. Создаём временную таблицу с новой структурой
    cur.execute("""
        CREATE TABLE face_rectangles_new (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id         INTEGER,                    -- Теперь опциональный
            archive_scope  TEXT,
            file_path      TEXT NOT NULL,
            face_index     INTEGER NOT NULL,
            bbox_x         INTEGER NOT NULL,
            bbox_y         INTEGER NOT NULL,
            bbox_w         INTEGER NOT NULL,
            bbox_h         INTEGER NOT NULL,
            confidence     REAL,
            presence_score REAL,
            thumb_jpeg     BLOB,
            manual_person  TEXT,
            ignore_flag    INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT NOT NULL,
            is_manual      INTEGER NOT NULL DEFAULT 0,
            manual_created_at TEXT,
            embedding      BLOB
        )
    """)
    
    # 2. Копируем данные
    print("Копируем данные...")
    cur.execute("""
        INSERT INTO face_rectangles_new (
            id, run_id, archive_scope, file_path, face_index,
            bbox_x, bbox_y, bbox_w, bbox_h,
            confidence, presence_score, thumb_jpeg,
            manual_person, ignore_flag, created_at,
            is_manual, manual_created_at, embedding
        )
        SELECT 
            id, run_id, archive_scope, file_path, face_index,
            bbox_x, bbox_y, bbox_w, bbox_h,
            confidence, presence_score, thumb_jpeg,
            manual_person, ignore_flag, created_at,
            COALESCE(is_manual, 0), manual_created_at, embedding
        FROM face_rectangles
    """)
    
    copied = cur.rowcount
    print(f"Скопировано записей: {copied}")
    
    # 3. Удаляем старую таблицу и переименовываем новую
    print("Переименовываем таблицы...")
    cur.execute("DROP TABLE face_rectangles")
    cur.execute("ALTER TABLE face_rectangles_new RENAME TO face_rectangles")
    
    # 4. Восстанавливаем индексы
    print("Восстанавливаем индексы...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_rect_run ON face_rectangles(run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_rect_file ON face_rectangles(file_path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_rect_archive_scope ON face_rectangles(archive_scope)")
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Скопировано записей: {copied}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
