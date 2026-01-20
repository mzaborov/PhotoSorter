#!/usr/bin/env python3
"""
Проверка групп персон в БД.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем структуру таблицы
    cur.execute("PRAGMA table_info(persons)")
    columns = cur.fetchall()
    print("Колонки таблицы persons:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    print()
    
    # Получаем всех персон с группами
    cur.execute("""
        SELECT id, name, "group", group_order 
        FROM persons 
        ORDER BY COALESCE(group_order, 999) ASC, name ASC
    """)
    
    rows = cur.fetchall()
    print(f"Всего персон: {len(rows)}")
    print()
    
    # Группируем по группам
    grouped = {}
    no_group = []
    
    for row in rows:
        group_name = row["group"]
        if group_name:
            if group_name not in grouped:
                grouped[group_name] = []
            grouped[group_name].append(row)
        else:
            no_group.append(row)
    
    # Выводим по группам
    for group_name in sorted(grouped.keys(), key=lambda g: grouped[g][0]["group_order"] if grouped[g] else 999):
        persons = grouped[group_name]
        print(f"Группа: {group_name} (order: {persons[0]['group_order']})")
        for p in persons:
            print(f"  - {p['id']}: {p['name']}")
        print()
    
    if no_group:
        print("Без группы:")
        for p in no_group:
            print(f"  - {p['id']}: {p['name']}")
        print()

if __name__ == "__main__":
    main()
