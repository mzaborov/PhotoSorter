#!/usr/bin/env python3
"""
Миграция: делает run_id опциональным (NULL) в face_clusters для поддержки архивных кластеров.
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
    print("МИГРАЦИЯ: делаем run_id опциональным в face_clusters")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем текущее состояние
    cur.execute("PRAGMA table_info(face_clusters)")
    columns = cur.fetchall()
    run_id_col = next((c for c in columns if c["name"] == "run_id"), None)
    
    if run_id_col and run_id_col["notnull"] == 0:
        print("run_id уже опциональный. Миграция не требуется.")
        return 0
    
    print("Пересоздаём таблицу face_clusters с опциональным run_id...")
    print()
    
    # 1. Создаём временную таблицу
    cur.execute("""
        CREATE TABLE face_clusters_new (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER,                    -- Теперь опциональный
            archive_scope   TEXT,
            method          TEXT NOT NULL,
            params_json     TEXT,
            created_at      TEXT NOT NULL
        )
    """)
    
    # 2. Копируем данные
    print("Копируем данные...")
    cur.execute("""
        INSERT INTO face_clusters_new (
            id, run_id, archive_scope, method, params_json, created_at
        )
        SELECT 
            id, run_id, archive_scope, method, params_json, created_at
        FROM face_clusters
    """)
    
    copied = cur.rowcount
    print(f"Скопировано записей: {copied}")
    
    # 3. Удаляем старую таблицу и переименовываем новую
    print("Переименовываем таблицы...")
    cur.execute("DROP TABLE face_clusters")
    cur.execute("ALTER TABLE face_clusters_new RENAME TO face_clusters")
    
    # 4. Восстанавливаем индексы
    print("Восстанавливаем индексы...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_run ON face_clusters(run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_archive_scope ON face_clusters(archive_scope)")
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Скопировано записей: {copied}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
