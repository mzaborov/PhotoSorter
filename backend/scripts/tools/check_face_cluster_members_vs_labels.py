#!/usr/bin/env python3
"""
Скрипт для сравнения face_cluster_members и face_labels.

Проверяет, есть ли расхождения между этими таблицами:
- face_cluster_members: одно лицо должно быть в одном кластере
- face_labels: одно лицо может быть в нескольких кластерах (если есть несколько записей)

Использование:
    python backend/scripts/tools/check_face_cluster_members_vs_labels.py --person-id 1
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def check_face_cluster_members_vs_labels(person_id: int) -> None:
    """Сравнивает face_cluster_members и face_labels для персоны."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем информацию о персоне
    cur.execute(
        """
        SELECT id, name
        FROM persons
        WHERE id = ?
        """,
        (person_id,),
    )
    
    person_row = cur.fetchone()
    if not person_row:
        print(f"Персона с ID {person_id} не найдена.")
        return
    
    person_name = person_row["name"]
    print(f"Персона: {person_name} (ID: {person_id})")
    print("=" * 80)
    print()
    
    # Получаем все лица персоны из face_labels
    cur.execute(
        """
        SELECT 
            fl.face_rectangle_id as face_id,
            fl.cluster_id as label_cluster_id
        FROM face_labels fl
        LEFT JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        ORDER BY fl.face_rectangle_id, fl.cluster_id
        """,
        (person_id,),
    )
    
    labels_by_face = defaultdict(list)
    for row in cur.fetchall():
        face_id = row["face_id"]
        cluster_id = row["label_cluster_id"]
        labels_by_face[face_id].append(cluster_id)
    
    # Получаем все лица персоны из face_cluster_members
    cur.execute(
        """
        SELECT DISTINCT
            fcm.face_rectangle_id as face_id,
            fcm.cluster_id as member_cluster_id
        FROM face_cluster_members fcm
        JOIN face_labels fl ON fl.face_rectangle_id = fcm.face_rectangle_id
        LEFT JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        ORDER BY fcm.face_rectangle_id, fcm.cluster_id
        """,
        (person_id,),
    )
    
    members_by_face = defaultdict(list)
    for row in cur.fetchall():
        face_id = row["face_id"]
        cluster_id = row["member_cluster_id"]
        members_by_face[face_id].append(cluster_id)
    
    print(f"Лиц в face_labels: {len(labels_by_face)}")
    print(f"Лиц в face_cluster_members: {len(members_by_face)}")
    print()
    
    # Проверяем расхождения
    all_face_ids = set(labels_by_face.keys()) | set(members_by_face.keys())
    
    faces_with_multiple_labels = []
    faces_with_multiple_members = []
    faces_mismatch = []
    faces_only_in_labels = []
    faces_only_in_members = []
    
    for face_id in all_face_ids:
        label_clusters = labels_by_face.get(face_id, [])
        member_clusters = members_by_face.get(face_id, [])
        
        if len(label_clusters) > 1:
            faces_with_multiple_labels.append({
                "face_id": face_id,
                "label_clusters": label_clusters,
                "member_clusters": member_clusters,
            })
        
        if len(member_clusters) > 1:
            faces_with_multiple_members.append({
                "face_id": face_id,
                "label_clusters": label_clusters,
                "member_clusters": member_clusters,
            })
        
        if not label_clusters:
            faces_only_in_members.append(face_id)
        elif not member_clusters:
            faces_only_in_labels.append(face_id)
        else:
            # Проверяем, совпадают ли кластеры
            label_set = set(label_clusters)
            member_set = set(member_clusters)
            if label_set != member_set:
                faces_mismatch.append({
                    "face_id": face_id,
                    "label_clusters": label_clusters,
                    "member_clusters": member_clusters,
                })
    
    print("=" * 80)
    print("АНАЛИЗ:")
    print("=" * 80)
    print()
    
    if faces_with_multiple_members:
        print(f"⚠ КРИТИЧНО: Лиц в нескольких кластерах через face_cluster_members: {len(faces_with_multiple_members)}")
        print("   Это НЕ должно быть! Одно лицо должно быть только в одном кластере.")
        print("   Примеры:")
        for face in faces_with_multiple_members[:5]:
            print(f"     Face ID {face['face_id']}: в кластерах {face['member_clusters']}")
        if len(faces_with_multiple_members) > 5:
            print(f"     ... и еще {len(faces_with_multiple_members) - 5} лиц")
        print()
    else:
        print("✓ Лиц в нескольких кластерах через face_cluster_members: 0 (правильно)")
        print()
    
    if faces_with_multiple_labels:
        print(f"ℹ Лиц в нескольких кластерах через face_labels: {len(faces_with_multiple_labels)}")
        print("   Это может быть нормально, если лицо было в нескольких кластерах до объединения.")
        print("   Примеры:")
        for face in faces_with_multiple_labels[:5]:
            print(f"     Face ID {face['face_id']}:")
            print(f"       face_labels: {face['label_clusters']}")
            print(f"       face_cluster_members: {face['member_clusters']}")
        if len(faces_with_multiple_labels) > 5:
            print(f"     ... и еще {len(faces_with_multiple_labels) - 5} лиц")
        print()
    
    if faces_mismatch:
        print(f"⚠ Расхождения между face_labels и face_cluster_members: {len(faces_mismatch)}")
        print("   Примеры:")
        for face in faces_mismatch[:5]:
            print(f"     Face ID {face['face_id']}:")
            print(f"       face_labels: {face['label_clusters']}")
            print(f"       face_cluster_members: {face['member_clusters']}")
        if len(faces_mismatch) > 5:
            print(f"     ... и еще {len(faces_mismatch) - 5} лиц")
        print()
    
    if faces_only_in_labels:
        print(f"⚠ Лиц только в face_labels (нет в face_cluster_members): {len(faces_only_in_labels)}")
        print("   Это проблема - лицо назначено персоне, но не находится ни в одном кластере.")
        print()
    
    if faces_only_in_members:
        print(f"⚠ Лиц только в face_cluster_members (нет в face_labels): {len(faces_only_in_members)}")
        print("   Это проблема - лицо в кластере, но не назначено персоне.")
        print()
    
    if not faces_with_multiple_members and not faces_mismatch and not faces_only_in_labels and not faces_only_in_members:
        print("✓ Все в порядке! Нет расхождений между face_cluster_members и face_labels.")


def main():
    parser = argparse.ArgumentParser(description="Сравнение face_cluster_members и face_labels")
    parser.add_argument("--person-id", type=int, required=True, help="ID персоны")
    
    args = parser.parse_args()
    
    check_face_cluster_members_vs_labels(person_id=args.person_id)


if __name__ == "__main__":
    main()
