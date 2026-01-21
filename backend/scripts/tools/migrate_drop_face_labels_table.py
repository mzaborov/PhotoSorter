#!/usr/bin/env python3
"""
Миграция: удаление старой таблицы face_labels.
Этап 9: Удаляем таблицу face_labels после завершения миграции.
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
    print("МИГРАЦИЯ: Удаление старой таблицы face_labels")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, существует ли таблица
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='face_labels'
    """)
    if not cur.fetchone():
        print("Таблица face_labels не найдена. Миграция не требуется.")
        return 0
    
    # Проверяем, есть ли данные в таблице
    cur.execute("SELECT COUNT(*) as count FROM face_labels")
    count = cur.fetchone()['count']
    if count > 0:
        print(f"⚠️  ВНИМАНИЕ: В таблице face_labels осталось {count} записей!")
        print("Убедитесь, что все данные мигрированы в face_person_manual_assignments")
        print("и что все места в коде обновлены на использование новой схемы.")
        response = input("Продолжить удаление? (yes/no): ")
        if response.lower() != 'yes':
            print("Отменено.")
            return 1
    
    # Удаляем индексы
    print("Удаляем индексы...")
    cur.execute("DROP INDEX IF EXISTS idx_face_labels_face")
    cur.execute("DROP INDEX IF EXISTS idx_face_labels_person")
    cur.execute("DROP INDEX IF EXISTS idx_face_labels_unique")
    
    # Удаляем таблицу
    print("Удаляем таблицу face_labels...")
    cur.execute("DROP TABLE face_labels")
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print("Таблица face_labels удалена.")
    print()
    print("✅ Миграция полностью завершена!")
    print("   - face_clusters.person_id используется для привязки кластеров к персонам")
    print("   - face_person_manual_assignments используется для ручных привязок")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
