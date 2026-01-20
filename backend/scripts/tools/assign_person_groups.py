"""
Скрипт для массового назначения групп существующим персонам по именам.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection
from datetime import datetime, timezone

# Группы персон с порядком сортировки (дублируем из face_clusters.py чтобы избежать зависимости от FastAPI)
PERSON_GROUPS = {
    "Я и Супруга": {"order": 1},
    "Дети": {"order": 2},
    "Родственники": {"order": 3},
    "Синяя диагональ": {"order": 4},
    "Работа": {"order": 5},
}

def get_group_order(group_name: str | None) -> int | None:
    """Возвращает порядок группы по её названию. Если группы нет в PERSON_GROUPS, возвращает None."""
    if not group_name:
        return None
    group_info = PERSON_GROUPS.get(group_name)
    return group_info["order"] if group_info else None

# Маппинг имен персон на группы
PERSON_NAME_TO_GROUP = {
    # Я и Супруга
    "Заборов Михаил": "Я и Супруга",
    "Рид Анна": "Я и Супруга",
    
    # Дети
    "Агата": "Дети",
    "Санек": "Дети",
    "Санёк": "Дети",  # Вариант написания
    "Нюся": "Дети",
    "Темка": "Дети",
    "Артём": "Дети",  # Возможно это Темка
    
    # Родственники
    "Заборов Андрей": "Родственники",
    "Света Заборова": "Родственники",
    "Сергеев Александр": "Родственники",
    "Сергеева Наталья": "Родственники",
    "Бабушка Ева": "Родственники",
    "Мама": "Родственники",
    "Папа": "Родственники",
    "Заборова Юля": "Родственники",
    "Тимофей": "Родственники",
    "Богдан": "Родственники",
    "Ася": "Родственники",
    
    # Синяя диагональ
    "Бульба": "Синяя диагональ",
    "Шатин": "Синяя диагональ",
    "Аксельрод": "Синяя диагональ",
    "Тоха": "Синяя диагональ",
    "Леня Штишевский": "Синяя диагональ",
    "Лёня Штишевский": "Синяя диагональ",  # Вариант написания
    
    # Работа
    "Герман Алексеев": "Работа",
}


def assign_groups(dry_run=True):
    """
    Назначает группы персон по именам.
    
    Args:
        dry_run: Если True, только показывает что будет сделано, не изменяет БД
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем всех персон
    cur.execute(
        """
        SELECT id, name, "group", group_order
        FROM persons
        ORDER BY name
        """
    )
    
    persons = cur.fetchall()
    updates = []
    
    for person in persons:
        person_id = person["id"]
        person_name = person["name"]
        current_group = person["group"]
        target_group = PERSON_NAME_TO_GROUP.get(person_name)
        
        if target_group:
            if current_group != target_group:
                group_order = get_group_order(target_group)
                updates.append({
                    "id": person_id,
                    "name": person_name,
                    "current_group": current_group,
                    "target_group": target_group,
                    "group_order": group_order,
                })
        # Персоны не в маппинге оставляем без изменений
    
    if not updates:
        print("Нет персон для обновления групп.")
        return
    
    print(f"\nНайдено персон для обновления: {len(updates)}\n")
    print("Будут обновлены:")
    for u in updates:
        current = u["current_group"] or "(нет группы)"
        print(f"  {u['name']} (ID: {u['id']}): {current} -> {u['target_group']}")
    
    if dry_run:
        print("\n[DRY RUN] Режим проверки. Для применения изменений запустите с --apply")
        return
    
    # Применяем изменения
    now = datetime.now(timezone.utc).isoformat()
    updated_count = 0
    
    for u in updates:
        try:
            cur.execute(
                """
                UPDATE persons
                SET "group" = ?, group_order = ?, updated_at = ?
                WHERE id = ?
                """,
                (u["target_group"], u["group_order"], now, u["id"]),
            )
            updated_count += 1
        except Exception as e:
            print(f"Ошибка при обновлении персоны {u['name']} (ID: {u['id']}): {e}")
    
    conn.commit()
    print(f"\n✓ Обновлено персон: {updated_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Назначает группы персон по именам")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Применить изменения (по умолчанию только dry-run)",
    )
    
    args = parser.parse_args()
    
    assign_groups(dry_run=not args.apply)
