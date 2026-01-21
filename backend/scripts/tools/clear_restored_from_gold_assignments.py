#!/usr/bin/env python3
"""
Удаление всех записей с source='restored_from_gold' из face_person_manual_assignments.
Эти записи создаются скриптом restore_lost_faces_from_gold.py и конфликтуют с привязками через кластеры.
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

conn = get_connection()
cur = conn.cursor()

# Проверяем количество записей
cur.execute("SELECT COUNT(*) as count FROM face_person_manual_assignments WHERE source = 'restored_from_gold'")
count = cur.fetchone()['count']

if count == 0:
    print("Нет записей с source='restored_from_gold' для удаления.")
    sys.exit(0)

print(f"Найдено записей с source='restored_from_gold': {count}")

# Удаляем записи, которые конфликтуют с кластерами (лицо уже в кластере с правильной персоной)
cur.execute("""
    DELETE FROM face_person_manual_assignments
    WHERE source = 'restored_from_gold'
      AND EXISTS (
          SELECT 1
          FROM face_cluster_members fcm
          JOIN face_clusters fc ON fc.id = fcm.cluster_id
          WHERE fcm.face_rectangle_id = face_person_manual_assignments.face_rectangle_id
            AND fc.person_id = face_person_manual_assignments.person_id
      )
""")
conflict_count = cur.rowcount

# Удаляем все оставшиеся записи с source='restored_from_gold'
cur.execute("DELETE FROM face_person_manual_assignments WHERE source = 'restored_from_gold'")
total_deleted = cur.rowcount

conn.commit()

print(f"Удалено записей, конфликтующих с кластерами: {conflict_count}")
print(f"Всего удалено записей: {total_deleted}")
print("\n✅ Очистка завершена!")
