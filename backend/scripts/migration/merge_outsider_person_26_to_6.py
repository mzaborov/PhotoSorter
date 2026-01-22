#!/usr/bin/env python3
"""
Миграция: перенос данных с персоны ID 26 ("Посторонние") на персону ID 6 ("Посторонний")
и удаление персоны ID 26.
"""
import sqlite3
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("data/photosorter.db")

SOURCE_PERSON_ID = 26  # "Посторонние" - удаляем
TARGET_PERSON_ID = 6   # "Посторонний" - правильная персона

def main():
    parser = argparse.ArgumentParser(description='Объединение персон Посторонний')
    parser.add_argument('--yes', action='store_true', help='Автоматическое подтверждение')
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ База данных не найдена: {DB_PATH}")
        return 1
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    print("=" * 60)
    print("МИГРАЦИЯ: Перенос данных с персоны 26 на персону 6")
    print("=" * 60)
    
    # Проверяем, что целевая персона существует
    cur.execute("SELECT id, name FROM persons WHERE id = ?", (TARGET_PERSON_ID,))
    target_person = cur.fetchone()
    if not target_person:
        print(f"❌ ОШИБКА: Персона ID {TARGET_PERSON_ID} не найдена!")
        return 1
    
    print(f"✅ Целевая персона найдена: ID={target_person['id']}, Name='{target_person['name']}'")
    
    # Проверяем, что исходная персона существует
    cur.execute("SELECT id, name FROM persons WHERE id = ?", (SOURCE_PERSON_ID,))
    source_person = cur.fetchone()
    if not source_person:
        print(f"⚠️  Персона ID {SOURCE_PERSON_ID} не найдена, миграция не требуется")
        return 0
    
    print(f"📋 Исходная персона: ID={source_person['id']}, Name='{source_person['name']}'")
    
    # Подсчитываем данные для переноса
    tables_to_check = [
        ("face_person_manual_assignments", "person_id"),
        ("face_clusters", "person_id"),
        ("person_rectangles", "person_id"),
        ("file_persons", "person_id"),
    ]
    
    total_records = 0
    for table_name, column_name in tables_to_check:
        cur.execute(f"SELECT COUNT(*) as cnt FROM {table_name} WHERE {column_name} = ?", (SOURCE_PERSON_ID,))
        count = cur.fetchone()["cnt"]
        if count > 0:
            print(f"  - {table_name}: {count} записей")
            total_records += count
    
    if total_records == 0:
        print("⚠️  Нет данных для переноса")
    else:
        print(f"\n📊 Всего записей для переноса: {total_records}")
    
    # Переносим данные
    print("\n🔄 Перенос данных...")
    for table_name, column_name in tables_to_check:
        cur.execute(f"""
            UPDATE {table_name}
            SET {column_name} = ?
            WHERE {column_name} = ?
        """, (TARGET_PERSON_ID, SOURCE_PERSON_ID))
        updated = cur.rowcount
        if updated > 0:
            print(f"  ✅ {table_name}: перенесено {updated} записей")
    
    # Удаляем исходную персону
    print(f"\n🗑️  Удаление персоны ID {SOURCE_PERSON_ID}...")
    cur.execute("DELETE FROM persons WHERE id = ?", (SOURCE_PERSON_ID,))
    deleted = cur.rowcount
    if deleted > 0:
        print(f"  ✅ Персона ID {SOURCE_PERSON_ID} удалена")
    else:
        print(f"  ⚠️  Персона ID {SOURCE_PERSON_ID} не была удалена")
    
    conn.commit()
    conn.close()
    print("\n✅ Миграция завершена успешно!")
    return 0

if __name__ == "__main__":
    exit(main())
