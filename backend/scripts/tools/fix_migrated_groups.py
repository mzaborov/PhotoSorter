#!/usr/bin/env python3
"""
Исправление уже мигрированных данных: "024 Минск" -> "2024 Минск"
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common.db import FaceStore


def fix_cut_group_names():
    """Исправляет обрезанные названия групп типа "024 Минск" -> "2024 Минск" """
    fs = FaceStore()
    try:
        cur = fs.conn.cursor()
        
        # Получаем все group_path, которые начинаются с "0XX "
        cur.execute("SELECT DISTINCT group_path FROM file_groups WHERE group_path LIKE '0__ %'")
        cut_paths = [row[0] for row in cur.fetchall()]
        
        print(f"Найдено обрезанных названий: {len(cut_paths)}")
        
        fixes = []
        for old_path in cut_paths:
            # Проверяем паттерн "0XX Название" -> "20XX Название"
            import re
            match = re.match(r'^0(\d{2})\s+(.+)$', old_path)
            if match:
                short_year = match.group(1)
                place = match.group(2)
                new_path = f"20{short_year} {place}"
                fixes.append((new_path, old_path))
                print(f"  {old_path} -> {new_path}")
        
        # Обновляем в file_groups
        for new_path, old_path in fixes:
            cur.execute(
                "UPDATE file_groups SET group_path = ? WHERE group_path = ?",
                (new_path, old_path)
            )
        
        # Обновляем в groups (если таблица существует)
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='groups'")
            if cur.fetchone():
                for new_path, old_path in fixes:
                    # Обновляем название группы
                    cur.execute(
                        "UPDATE groups SET name = ? WHERE name = ?",
                        (new_path, old_path)
                    )
                    # Обновляем group_id в file_groups, если он был связан со старой группой
                    cur.execute("""
                        UPDATE file_groups 
                        SET group_id = (
                            SELECT id FROM groups WHERE name = ?
                        )
                        WHERE group_path = ? AND group_id = (
                            SELECT id FROM groups WHERE name = ?
                        )
                    """, (new_path, new_path, old_path))
                print(f"  Обновлено в таблице groups: {len(fixes)} записей")
                print(f"  Обновлены связи group_id в file_groups")
            else:
                print("  Таблица groups не найдена, пропускаем")
        except Exception as e:
            print(f"  Предупреждение: не удалось обновить groups: {e}")
        
        fs.conn.commit()
        print(f"\nИсправлено записей: {len(fixes)}")
        
    finally:
        fs.close()


if __name__ == "__main__":
    print("=== Исправление обрезанных названий групп ===")
    fix_cut_group_names()
    print("\n=== Готово ===")
