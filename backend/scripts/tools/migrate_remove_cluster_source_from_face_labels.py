#!/usr/bin/env python3
"""
Миграция: удаление записей с source='cluster' из face_labels.
Этап 5: Записи с source='cluster' больше не нужны, т.к. кластеры привязаны через face_clusters.person_id.
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
    print("МИГРАЦИЯ: Удаление записей с source='cluster' из face_labels")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем количество записей с source='cluster'
    cur.execute("SELECT COUNT(*) as count FROM face_labels WHERE source = 'cluster'")
    cluster_count = cur.fetchone()['count']
    print(f"Записей с source='cluster': {cluster_count}")
    
    if cluster_count == 0:
        print("Записей с source='cluster' не найдено. Миграция не требуется.")
        return 0
    
    # Проверяем другие source
    cur.execute("SELECT source, COUNT(*) as count FROM face_labels GROUP BY source")
    other_sources = cur.fetchall()
    print("\nРаспределение по source:")
    for row in other_sources:
        print(f"  {row['source']}: {row['count']}")
    print()
    
    # Удаляем записи с source='cluster'
    print("Удаление записей с source='cluster'...")
    cur.execute("DELETE FROM face_labels WHERE source = 'cluster'")
    deleted = cur.rowcount
    print(f"Удалено записей: {deleted}")
    
    conn.commit()
    
    # Проверяем результат
    cur.execute("SELECT source, COUNT(*) as count FROM face_labels GROUP BY source")
    remaining = cur.fetchall()
    print("\nОсталось записей по source:")
    for row in remaining:
        print(f"  {row['source']}: {row['count']}")
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Удалено записей с source='cluster': {deleted}")
    print()
    print("ВАЖНО: Старая таблица face_labels пока не удалена.")
    print("После полного обновления кода можно будет удалить face_labels.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
