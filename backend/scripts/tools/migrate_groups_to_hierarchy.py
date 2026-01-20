#!/usr/bin/env python3
"""
Миграция групп: нормализация данных и создание иерархии через parent_id.

1. Нормализует существующие group_path в file_groups (убирает префикс "Поездки/")
2. Создает таблицу groups с иерархией
3. Мигрирует данные из file_groups в groups
4. Обновляет file_groups для использования group_id вместо group_path
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common.db import FaceStore


def normalize_group_paths():
    """Нормализует group_path в file_groups: убирает префикс 'Поездки/' или 'Поездки\\'"""
    fs = FaceStore()
    try:
        cur = fs.conn.cursor()
        
        # Получаем все уникальные group_path
        cur.execute("SELECT DISTINCT group_path FROM file_groups")
        all_paths = [row[0] for row in cur.fetchall()]
        
        print(f"Найдено уникальных group_path: {len(all_paths)}")
        
        # Нормализуем каждый group_path
        updates = []
        for old_path in all_paths:
            if not old_path:
                continue
            
            # Убираем префикс "Поездки/" или "Поездки\"
            new_path = old_path
            if old_path.startswith("Поездки/"):
                new_path = old_path[8:]  # Убираем "Поездки/" (8 символов)
            elif old_path.startswith("Поездки\\"):
                new_path = old_path[9:]  # Убираем "Поездки\" (9 символов, т.к. обратный слэш экранируется)
            
            if new_path != old_path:
                updates.append((new_path, old_path))
                print(f"  {old_path} -> {new_path}")
        
        # Обновляем записи
        for new_path, old_path in updates:
            cur.execute(
                "UPDATE file_groups SET group_path = ? WHERE group_path = ?",
                (new_path, old_path)
            )
        
        fs.conn.commit()
        print(f"\nОбновлено записей: {len(updates)}")
        
    finally:
        fs.close()


def create_groups_table():
    """Создает таблицу groups с иерархией"""
    fs = FaceStore()
    try:
        cur = fs.conn.cursor()
        
        # Создаем таблицу groups
        cur.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_run_id     INTEGER NOT NULL,
                name                TEXT NOT NULL,
                parent_id           INTEGER NULL,
                created_at          TEXT NOT NULL,
                UNIQUE(pipeline_run_id, name, parent_id),
                FOREIGN KEY (parent_id) REFERENCES groups(id)
            )
        """)
        
        # Создаем индексы
        cur.execute("CREATE INDEX IF NOT EXISTS idx_groups_run ON groups(pipeline_run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_groups_parent ON groups(parent_id)")
        
        fs.conn.commit()
        print("Таблица groups создана")
        
    finally:
        fs.close()


def migrate_to_groups():
    """Мигрирует данные из file_groups в groups и обновляет file_groups"""
    fs = FaceStore()
    try:
        cur = fs.conn.cursor()
        
        # Получаем все уникальные group_path с pipeline_run_id
        cur.execute("""
            SELECT DISTINCT pipeline_run_id, group_path
            FROM file_groups
            ORDER BY pipeline_run_id, group_path
        """)
        all_groups = cur.fetchall()
        
        print(f"Найдено групп для миграции: {len(all_groups)}")
        
        # Создаем группы
        group_map = {}  # (pipeline_run_id, group_path) -> group_id
        parent_map = {}  # (pipeline_run_id, parent_name) -> parent_id
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        for pipeline_run_id, group_path in all_groups:
            # Определяем, является ли группа поездкой (начинается с года)
            import re
            is_trip = re.match(r'^\d{3,4}\s+', group_path)
            
            if is_trip:
                # Это поездка - создаем родительскую группу "Поездки" если её нет
                parent_key = (pipeline_run_id, "Поездки")
                if parent_key not in parent_map:
                    cur.execute("""
                        INSERT INTO groups (pipeline_run_id, name, parent_id, created_at)
                        VALUES (?, ?, NULL, ?)
                    """, (pipeline_run_id, "Поездки", now))
                    parent_id = cur.lastrowid
                    parent_map[parent_key] = parent_id
                    print(f"  Создана группа 'Поездки' (id={parent_id}) для pipeline_run_id={pipeline_run_id}")
                else:
                    parent_id = parent_map[parent_key]
            else:
                parent_id = None
            
            # Создаем группу
            cur.execute("""
                INSERT INTO groups (pipeline_run_id, name, parent_id, created_at)
                VALUES (?, ?, ?, ?)
            """, (pipeline_run_id, group_path, parent_id, now))
            group_id = cur.lastrowid
            group_map[(pipeline_run_id, group_path)] = group_id
            print(f"  Создана группа '{group_path}' (id={group_id}, parent_id={parent_id})")
        
        fs.conn.commit()
        print(f"\nСоздано групп: {len(group_map)}")
        
        # Добавляем колонку group_id в file_groups (если её нет)
        try:
            cur.execute("ALTER TABLE file_groups ADD COLUMN group_id INTEGER")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise
            print("Колонка group_id уже существует")
        
        # Обновляем file_groups: заполняем group_id
        for (pipeline_run_id, group_path), group_id in group_map.items():
            cur.execute("""
                UPDATE file_groups
                SET group_id = ?
                WHERE pipeline_run_id = ? AND group_path = ?
            """, (group_id, pipeline_run_id, group_path))
        
        fs.conn.commit()
        print(f"Обновлено записей в file_groups: {sum(1 for _ in group_map)}")
        
    finally:
        fs.close()


if __name__ == "__main__":
    print("=== Шаг 1: Нормализация group_path ===")
    normalize_group_paths()
    
    print("\n=== Шаг 2: Создание таблицы groups ===")
    create_groups_table()
    
    print("\n=== Шаг 3: Миграция данных ===")
    migrate_to_groups()
    
    print("\n=== Миграция завершена ===")
