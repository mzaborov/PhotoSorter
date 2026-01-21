#!/usr/bin/env python3
"""
Миграция: переименование face_labels → face_person_manual_assignments.
Этап 4: Создаём новую таблицу, копируем только source='manual', удаляем старую.
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
    print("МИГРАЦИЯ: Переименование face_labels → face_person_manual_assignments")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, существует ли уже новая таблица
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='face_person_manual_assignments'
    """)
    if cur.fetchone():
        print("Таблица face_person_manual_assignments уже существует. Миграция не требуется.")
        return 0
    
    # Проверяем количество записей с source='manual'
    cur.execute("SELECT COUNT(*) as count FROM face_labels WHERE source = 'manual'")
    manual_count = cur.fetchone()['count']
    print(f"Записей с source='manual' для копирования: {manual_count}")
    
    # Проверяем другие source (которые не копируем)
    cur.execute("SELECT source, COUNT(*) as count FROM face_labels WHERE source != 'manual' GROUP BY source")
    other_sources = cur.fetchall()
    if other_sources:
        print("Записи с другими source (не копируются):")
        for row in other_sources:
            print(f"   {row['source']}: {row['count']}")
    print()
    
    # 1. Создаём новую таблицу
    print("1. Создаём таблицу face_person_manual_assignments...")
    cur.execute("""
        CREATE TABLE face_person_manual_assignments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            face_rectangle_id   INTEGER NOT NULL,
            person_id           INTEGER NOT NULL,
            source              TEXT NOT NULL,
            confidence          REAL,
            created_at          TEXT NOT NULL
        )
    """)
    
    # 2. Копируем только записи с source='manual'
    print("2. Копируем записи с source='manual'...")
    cur.execute("""
        INSERT INTO face_person_manual_assignments (
            id, face_rectangle_id, person_id, source, confidence, created_at
        )
        SELECT 
            id, face_rectangle_id, person_id, source, confidence, created_at
        FROM face_labels
        WHERE source = 'manual'
    """)
    copied = cur.rowcount
    print(f"   Скопировано записей: {copied}")
    
    # 3. Создаём индексы
    print("3. Создаём индексы...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_person_manual_assignments_face ON face_person_manual_assignments(face_rectangle_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_person_manual_assignments_person ON face_person_manual_assignments(person_id)")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_face_person_manual_assignments_unique ON face_person_manual_assignments(face_rectangle_id, person_id)")
    
    # 4. Добавляем FOREIGN KEY (SQLite не поддерживает через ALTER, но проверим целостность)
    print("4. Проверка целостности данных...")
    cur.execute("""
        SELECT COUNT(*) as count
        FROM face_person_manual_assignments fpma
        WHERE NOT EXISTS (
            SELECT 1 FROM face_rectangles fr WHERE fr.id = fpma.face_rectangle_id
        )
    """)
    invalid_faces = cur.fetchone()['count']
    if invalid_faces > 0:
        print(f"   ⚠️  Найдено {invalid_faces} записей с несуществующими face_rectangle_id")
    else:
        print("   ✅ Все face_rectangle_id валидны")
    
    cur.execute("""
        SELECT COUNT(*) as count
        FROM face_person_manual_assignments fpma
        WHERE NOT EXISTS (
            SELECT 1 FROM persons p WHERE p.id = fpma.person_id
        )
    """)
    invalid_persons = cur.fetchone()['count']
    if invalid_persons > 0:
        print(f"   ⚠️  Найдено {invalid_persons} записей с несуществующими person_id")
    else:
        print("   ✅ Все person_id валидны")
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print("Создана таблица: face_person_manual_assignments")
    print(f"Скопировано записей: {copied}")
    print()
    print("ВАЖНО: Старая таблица face_labels пока не удалена.")
    print("Следующий шаг: обновить код, затем удалить face_labels.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
