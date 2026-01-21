#!/usr/bin/env python3
"""
Скрипт для удаления всех ручных привязок из face_person_manual_assignments.
Ручные привязки считаются мусором и должны быть удалены.
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
    print("УДАЛЕНИЕ ВСЕХ РУЧНЫХ ПРИВЯЗОК")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем количество записей
    cur.execute("SELECT COUNT(*) as count FROM face_person_manual_assignments")
    count_row = cur.fetchone()
    total_count = count_row["count"] if count_row else 0
    
    print(f"Всего записей в face_person_manual_assignments: {total_count}")
    print()
    
    if total_count == 0:
        print("Записей для удаления не найдено.")
        return 0
    
    # Показываем распределение по source
    cur.execute("SELECT source, COUNT(*) as count FROM face_person_manual_assignments GROUP BY source")
    sources = cur.fetchall()
    print("Распределение по source:")
    for row in sources:
        print(f"  {row['source']}: {row['count']}")
    print()
    
    # Подтверждение
    response = input(f"Удалить все {total_count} записей? (yes/no): ")
    if response.lower() != 'yes':
        print("Отменено.")
        return 0
    
    # Удаляем все записи
    print("Удаление всех записей...")
    cur.execute("DELETE FROM face_person_manual_assignments")
    deleted = cur.rowcount
    print(f"Удалено записей: {deleted}")
    
    conn.commit()
    
    # Проверяем результат
    cur.execute("SELECT COUNT(*) as count FROM face_person_manual_assignments")
    remaining_row = cur.fetchone()
    remaining = remaining_row["count"] if remaining_row else 0
    
    print()
    print("=" * 60)
    print("УДАЛЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Удалено записей: {deleted}")
    print(f"Осталось записей: {remaining}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
