#!/usr/bin/env python3
"""
Миграция: перенос оставшихся записей из face_labels в face_person_manual_assignments.
Переносим записи с source='manual' и source='restored_from_gold'.
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
    print("МИГРАЦИЯ: Перенос оставшихся записей из face_labels")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем количество оставшихся записей
    cur.execute("SELECT source, COUNT(*) as count FROM face_labels GROUP BY source")
    remaining = cur.fetchall()
    print("Оставшиеся записи в face_labels:")
    total = 0
    for row in remaining:
        print(f"  {row['source']}: {row['count']}")
        total += row['count']
    print(f"  Всего: {total}")
    print()
    
    if total == 0:
        print("Записей для переноса не найдено.")
        return 0
    
    # Проверяем, есть ли уже записи в face_person_manual_assignments
    cur.execute("SELECT COUNT(*) as count FROM face_person_manual_assignments")
    existing = cur.fetchone()['count']
    print(f"Записей в face_person_manual_assignments: {existing}")
    print()
    
    # Переносим все оставшиеся записи (manual и restored_from_gold)
    print("Перенос записей...")
    cur.execute("""
        INSERT OR IGNORE INTO face_person_manual_assignments (
            id, face_rectangle_id, person_id, source, confidence, created_at
        )
        SELECT 
            id, face_rectangle_id, person_id, source, confidence, created_at
        FROM face_labels
    """)
    copied = cur.rowcount
    print(f"Скопировано записей: {copied}")
    
    # Проверяем результат
    cur.execute("SELECT COUNT(*) as count FROM face_person_manual_assignments")
    final_count = cur.fetchone()['count']
    print(f"Всего записей в face_person_manual_assignments: {final_count}")
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Скопировано записей: {copied}")
    print()
    print("ВАЖНО: Старая таблица face_labels пока не удалена.")
    print("После полного обновления кода можно будет удалить face_labels.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
