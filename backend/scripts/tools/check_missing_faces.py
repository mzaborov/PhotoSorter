#!/usr/bin/env python3
"""
Скрипт для проверки, существуют ли лица в БД (включая помеченные как ignore).

Использование:
    python backend/scripts/tools/check_missing_faces.py --face-ids 167606,167608,166554
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def check_faces(face_ids: list[int]) -> None:
    """Проверяет, существуют ли лица в БД."""
    conn = get_connection()
    cur = conn.cursor()
    
    placeholders = ",".join(["?"] * len(face_ids))
    
    # Проверяем без фильтра по ignore_flag
    cur.execute(
        f"""
        SELECT 
            fr.id as face_id,
            fr.file_path,
            fr.ignore_flag,
            fcm.cluster_id,
            fl.person_id,
            p.name as person_name
        FROM face_rectangles fr
        LEFT JOIN face_cluster_members fcm ON fr.id = fcm.face_rectangle_id
        LEFT JOIN face_labels fl ON fr.id = fl.face_rectangle_id
        LEFT JOIN persons p ON fl.person_id = p.id
        WHERE fr.id IN ({placeholders})
        ORDER BY fr.id
        """,
        tuple(face_ids),
    )
    
    found_faces = {}
    for row in cur.fetchall():
        face_id = row["face_id"]
        found_faces[face_id] = {
            "face_id": face_id,
            "file_path": row["file_path"],
            "ignore_flag": row["ignore_flag"] or 0,
            "cluster_id": row["cluster_id"],
            "person_id": row["person_id"],
            "person_name": row["person_name"],
        }
    
    print(f"Проверка {len(face_ids)} лиц:")
    print("=" * 80)
    
    found_count = 0
    not_found_count = 0
    ignored_count = 0
    
    for face_id in face_ids:
        if face_id in found_faces:
            face_info = found_faces[face_id]
            found_count += 1
            
            status = []
            if face_info["ignore_flag"]:
                status.append("IGNORE")
                ignored_count += 1
            if face_info["cluster_id"]:
                status.append(f"Cluster: {face_info['cluster_id']}")
            if face_info["person_id"]:
                status.append(f"Person: {face_info['person_name']} (ID: {face_info['person_id']})")
            
            status_str = ", ".join(status) if status else "Без кластера и персоны"
            print(f"✓ Face ID {face_id}: {face_info['file_path']}")
            print(f"  Статус: {status_str}")
        else:
            not_found_count += 1
            print(f"✗ Face ID {face_id}: НЕ НАЙДЕНО В БД")
    
    print()
    print("=" * 80)
    print(f"Итого: найдено {found_count}, не найдено {not_found_count}, помечено как ignore {ignored_count}")


def main():
    parser = argparse.ArgumentParser(description="Проверка наличия лиц в БД")
    parser.add_argument("--face-ids", type=str, required=True, help="Список ID лиц через запятую (например: 167606,167608,166554)")
    
    args = parser.parse_args()
    
    face_ids = [int(fid.strip()) for fid in args.face_ids.split(",") if fid.strip()]
    
    if not face_ids:
        print("Ошибка: не указаны ID лиц")
        return
    
    check_faces(face_ids)


if __name__ == "__main__":
    main()
