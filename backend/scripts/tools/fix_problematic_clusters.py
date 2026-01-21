#!/usr/bin/env python3
"""
Исправление проблемных кластеров: удаление ошибочных записей face_labels.
Кластеры 528 и 733 - все лица Санёк, но были ошибочные записи для Агаты.
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
    print("ИСПРАВЛЕНИЕ ПРОБЛЕМНЫХ КЛАСТЕРОВ")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    problematic_clusters = [528, 733]
    correct_person_id = 4  # Санёк
    wrong_person_id = 9    # Агата
    
    for cluster_id in problematic_clusters:
        print(f"Кластер {cluster_id}:")
        
        # Проверяем текущий person_id в face_clusters
        cur.execute("SELECT person_id FROM face_clusters WHERE id = ?", (cluster_id,))
        row = cur.fetchone()
        if row:
            current_person_id = row['person_id']
            print(f"  Текущий person_id в face_clusters: {current_person_id}")
            
            # Если неправильный, исправляем
            if current_person_id != correct_person_id:
                print(f"  ⚠️  Исправляем person_id: {current_person_id} -> {correct_person_id}")
                cur.execute(
                    "UPDATE face_clusters SET person_id = ? WHERE id = ?",
                    (correct_person_id, cluster_id)
                )
            else:
                print(f"  ✅ person_id уже правильный ({correct_person_id})")
        
        # Находим ошибочные записи face_labels
        cur.execute("""
            SELECT COUNT(*) as count
            FROM face_cluster_members fcm
            JOIN face_labels fl ON fcm.face_rectangle_id = fl.face_rectangle_id
            WHERE fcm.cluster_id = ? 
            AND fl.source = 'cluster'
            AND fl.person_id = ?
        """, (cluster_id, wrong_person_id))
        
        wrong_count = cur.fetchone()['count']
        print(f"  Ошибочных записей face_labels (person_id={wrong_person_id}): {wrong_count}")
        
        if wrong_count > 0:
            # Удаляем ошибочные записи
            cur.execute("""
                DELETE FROM face_labels
                WHERE face_rectangle_id IN (
                    SELECT fcm.face_rectangle_id
                    FROM face_cluster_members fcm
                    WHERE fcm.cluster_id = ?
                )
                AND source = 'cluster'
                AND person_id = ?
            """, (cluster_id, wrong_person_id))
            
            deleted = cur.rowcount
            print(f"  ✅ Удалено ошибочных записей: {deleted}")
        else:
            print(f"  ✅ Ошибочных записей не найдено")
        
        # Проверяем результат
        cur.execute("""
            SELECT COUNT(DISTINCT fl.person_id) as person_count
            FROM face_cluster_members fcm
            JOIN face_labels fl ON fcm.face_rectangle_id = fl.face_rectangle_id
            WHERE fcm.cluster_id = ? 
            AND fl.source = 'cluster'
        """, (cluster_id,))
        
        person_count = cur.fetchone()['person_count']
        if person_count == 1:
            print(f"  ✅ Теперь только одна персона в кластере")
        else:
            print(f"  ⚠️  Все еще {person_count} персон в кластере")
        
        print()
    
    conn.commit()
    
    print("=" * 60)
    print("ИСПРАВЛЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
