#!/usr/bin/env python3
"""
Миграция: добавление person_id в face_clusters.
Этап 2: Добавляем колонку person_id и индексы.
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
    print("МИГРАЦИЯ: Добавление person_id в face_clusters")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, есть ли уже колонка person_id
    cur.execute("PRAGMA table_info(face_clusters)")
    columns = cur.fetchall()
    has_person_id = any(col['name'] == 'person_id' for col in columns)
    
    if has_person_id:
        print("person_id уже существует в face_clusters. Миграция не требуется.")
        return 0
    
    print("Добавляем колонку person_id...")
    cur.execute("ALTER TABLE face_clusters ADD COLUMN person_id INTEGER NULL")
    
    print("Создаём индекс...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_person ON face_clusters(person_id)")
    
    # Добавляем FOREIGN KEY (SQLite поддерживает только при создании таблицы, но проверим)
    # В SQLite нельзя добавить FOREIGN KEY через ALTER TABLE, но можно проверить целостность вручную
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print("Добавлено:")
    print("  - Колонка person_id INTEGER NULL")
    print("  - Индекс idx_face_clusters_person")
    print()
    print("Следующий шаг: миграция данных (перенос person_id из face_labels)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
