#!/usr/bin/env python3
"""
Скрипт для сравнения снимка состояния лиц персоны с текущим состоянием в БД.

Находит лица, которые были в snapshot, но отсутствуют в текущем состоянии.

Использование:
    python backend/scripts/tools/compare_person_snapshot.py --snapshot-file backend/data/person_1_snapshot.json
"""

import sys
import argparse
import json
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def compare_snapshot(snapshot_file: Path) -> None:
    """
    Сравнивает snapshot с текущим состоянием в БД и находит пропавшие лица.
    
    Args:
        snapshot_file: путь к JSON файлу со snapshot
    """
    # Читаем snapshot
    with open(snapshot_file, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    
    person_id = snapshot["person_id"]
    person_name = snapshot["person_name"]
    snapshot_faces = snapshot["faces"]
    snapshot_face_ids = {face["face_id"] for face in snapshot_faces}
    
    print(f"Персона: {person_name} (ID: {person_id})")
    print(f"Snapshot создан: {snapshot['snapshot_created_at']}")
    print(f"Записей face_labels в snapshot: {len(snapshot_faces)}")
    print(f"Уникальных лиц в snapshot: {len(snapshot_face_ids)}")
    print(f"Ожидалось лиц (из total_faces): {snapshot.get('total_faces', 'N/A')}")
    print()
    
    # Получаем текущее состояние из БД
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT 
            fl.face_rectangle_id as face_id,
            fl.cluster_id
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
    )
    
    current_faces = []
    for row in cur.fetchall():
        current_faces.append({
            "face_id": row["face_id"],
            "cluster_id": row["cluster_id"],
        })
    
    current_face_ids = {face["face_id"] for face in current_faces}
    
    print(f"Лиц в БД сейчас: {len(current_face_ids)}")
    print()
    
    # Находим пропавшие записи face_labels (не только уникальные лица)
    # Создаем множество пар (face_id, cluster_id) для snapshot
    snapshot_face_cluster_pairs = {(face["face_id"], face["cluster_id"]) for face in snapshot_faces}
    
    # Создаем множество пар (face_id, cluster_id) для текущего состояния
    current_face_cluster_pairs = {(face["face_id"], face["cluster_id"]) for face in current_faces}
    
    # Находим пропавшие пары (старые кластеры)
    missing_pairs = snapshot_face_cluster_pairs - current_face_cluster_pairs
    
    # Находим пропавшие уникальные лица (те, которые вообще не назначены персоне)
    missing_face_ids = snapshot_face_ids - current_face_ids
    
    # Проверяем, действительно ли пропали лица или они просто в других кластерах
    # Для каждого пропавшего лица проверяем, есть ли оно у персоны в другом кластере
    faces_in_other_clusters = set()
    truly_missing_face_ids = set()
    
    # Получаем все face_id из пропавших пар
    missing_face_ids_from_pairs = {face_id for face_id, _ in missing_pairs}
    
    if missing_face_ids_from_pairs:
        placeholders = ",".join(["?"] * len(missing_face_ids_from_pairs))
        cur.execute(
            f"""
            SELECT DISTINCT fl.face_rectangle_id as face_id, fl.cluster_id
            FROM face_labels fl
            WHERE fl.face_rectangle_id IN ({placeholders})
              AND fl.person_id = ?
            """,
            tuple(missing_face_ids_from_pairs) + (person_id,),
        )
        faces_found_in_other_clusters = {}
        for row in cur.fetchall():
            face_id = row["face_id"]
            cluster_id = row["cluster_id"]
            if face_id not in faces_found_in_other_clusters:
                faces_found_in_other_clusters[face_id] = []
            faces_found_in_other_clusters[face_id].append(cluster_id)
        
        faces_in_other_clusters = set(faces_found_in_other_clusters.keys())
        truly_missing_face_ids = missing_face_ids_from_pairs - faces_in_other_clusters
    
    # Также проверяем уникальные пропавшие лица
    if missing_face_ids:
        placeholders = ",".join(["?"] * len(missing_face_ids))
        cur.execute(
            f"""
            SELECT DISTINCT fl.face_rectangle_id as face_id
            FROM face_labels fl
            WHERE fl.face_rectangle_id IN ({placeholders})
              AND fl.person_id = ?
            """,
            tuple(missing_face_ids) + (person_id,),
        )
        truly_missing_face_ids = truly_missing_face_ids | (missing_face_ids - {row["face_id"] for row in cur.fetchall()})
    
    if not missing_pairs and not truly_missing_face_ids:
        print("✓ Все лица на месте!")
        return
    
    if missing_pairs:
        print(f"⚠ Пропало записей face_labels для старых кластеров: {len(missing_pairs)}")
        if faces_in_other_clusters:
            print(f"  ✓ Но {len(faces_in_other_clusters)} из них находятся в других кластерах - это нормально после объединения")
            # Показываем примеры перемещений
            if faces_found_in_other_clusters:
                print(f"\n  Примеры перемещений:")
                count = 0
                for face_id, new_clusters in list(faces_found_in_other_clusters.items())[:5]:
                    # Находим старый кластер из snapshot
                    old_cluster = next((c for f_id, c in missing_pairs if f_id == face_id), None)
                    if old_cluster:
                        print(f"    Face ID {face_id}: кластер {old_cluster} → {new_clusters[0]}")
                        count += 1
                if len(faces_found_in_other_clusters) > 5:
                    print(f"    ... и еще {len(faces_found_in_other_clusters) - 5} перемещений")
    
    if truly_missing_face_ids:
        print(f"⚠ Пропало уникальных лиц (не назначены персоне): {len(truly_missing_face_ids)}")
    print()
    
    # Получаем детальную информацию о пропавших записях
    # Включаем только те лица, которые действительно потеряли связь с персоной
    missing_faces_info = []
    for face in snapshot_faces:
        face_id = face["face_id"]
        pair = (face_id, face["cluster_id"])
        # Включаем только если пара пропала И лицо не найдено в других кластерах
        if pair in missing_pairs and face_id in truly_missing_face_ids:
            missing_faces_info.append(face)
    
    # Если все лица найдены в других кластерах, выходим
    if not truly_missing_face_ids and missing_pairs:
        print("✓ Все лица на месте! Они просто переместились в другие кластеры после объединения.")
        return
    
    # Группируем по кластерам
    missing_by_cluster = {}
    for face in missing_faces_info:
        cluster_id = face["cluster_id"]
        if cluster_id not in missing_by_cluster:
            missing_by_cluster[cluster_id] = []
        missing_by_cluster[cluster_id].append(face)
    
    print("Пропавшие лица по кластерам:")
    print("-" * 60)
    for cluster_id in sorted(missing_by_cluster.keys()):
        faces_in_cluster = missing_by_cluster[cluster_id]
        print(f"Кластер #{cluster_id}: {len(faces_in_cluster)} лиц")
        for face in faces_in_cluster[:5]:  # Показываем первые 5
            print(f"  - Face ID: {face['face_id']}, File: {face['file_path']}")
        if len(faces_in_cluster) > 5:
            print(f"  ... и еще {len(faces_in_cluster) - 5} лиц")
        print()
    
    # Проверяем, существуют ли эти лица в БД (может быть, они просто потеряли связь с персоной)
    print("Проверяем, существуют ли эти лица в БД...")
    placeholders = ",".join(["?"] * len(missing_face_ids))
    # Сначала проверяем без фильтра по ignore_flag, чтобы увидеть все лица
    cur.execute(
        f"""
        SELECT 
            fr.id as face_id,
            fr.file_path,
            fr.ignore_flag,
            fcm.cluster_id
        FROM face_rectangles fr
        LEFT JOIN face_cluster_members fcm ON fr.id = fcm.face_rectangle_id
        WHERE fr.id IN ({placeholders})
        """,
        tuple(missing_face_ids),
    )
    
    existing_faces = {}
    ignored_faces = []
    for row in cur.fetchall():
        face_id = row["face_id"]
        ignore_flag = row.get("ignore_flag", 0) or 0
        if ignore_flag:
            ignored_faces.append(face_id)
        existing_faces[face_id] = {
            "face_id": face_id,
            "file_path": row["file_path"],
            "cluster_id": row["cluster_id"],
            "ignore_flag": ignore_flag,
        }
    
    # Проверяем, есть ли face_labels для этих лиц (но с другой персоной или без персоны)
    cur.execute(
        f"""
        SELECT 
            fl.face_rectangle_id as face_id,
            fl.person_id,
            fl.cluster_id,
            p.name as person_name
        FROM face_labels fl
        LEFT JOIN persons p ON fl.person_id = p.id
        WHERE fl.face_rectangle_id IN ({placeholders})
        """,
        tuple(missing_face_ids),
    )
    
    faces_with_labels = {}
    for row in cur.fetchall():
        face_id = row["face_id"]
        if face_id not in faces_with_labels:
            faces_with_labels[face_id] = []
        faces_with_labels[face_id].append({
            "person_id": row["person_id"],
            "person_name": row["person_name"],
            "cluster_id": row["cluster_id"],
        })
    
    print()
    print("Статус пропавших лиц:")
    print("-" * 60)
    
    faces_to_restore = []
    faces_not_in_db = []
    faces_without_cluster = []
    faces_with_other_person = []
    processed_pairs = set()  # Для отслеживания уже обработанных пар (face_id, cluster_id)
    
    # Обрабатываем каждую пропавшую запись из snapshot
    for face_info in missing_faces_info:
        face_id = face_info["face_id"]
        expected_cluster_id = face_info["cluster_id"]
        pair_key = (face_id, expected_cluster_id)
        
        # Пропускаем, если уже обработали эту пару
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        if face_id not in existing_faces:
            if face_id not in faces_not_in_db:
                faces_not_in_db.append(face_id)
            continue
        
        db_face_info = existing_faces[face_id]
        current_cluster_id = db_face_info["cluster_id"]
        
        # Если лицо помечено как ignore, пропускаем его
        if db_face_info.get("ignore_flag", 0):
            continue
        
        if not current_cluster_id:
            if face_id not in faces_without_cluster:
                faces_without_cluster.append(face_id)
            continue
        
        # Проверяем, есть ли face_label для этого лица с правильной персоной и кластером
        has_correct_label = False
        if face_id in faces_with_labels:
            labels = faces_with_labels[face_id]
            # Проверяем, есть ли label с правильной персоной и кластером
            has_correct_label = any(
                label["person_id"] == person_id and label["cluster_id"] == expected_cluster_id 
                for label in labels
            )
            
            if not has_correct_label:
                # Проверяем, есть ли label с правильной персоной, но другим кластером
                has_correct_person_wrong_cluster = any(
                    label["person_id"] == person_id and label["cluster_id"] != expected_cluster_id
                    for label in labels
                )
                
                if has_correct_person_wrong_cluster:
                    # Персона правильная, но кластер другой - нужно восстановить для правильного кластера
                    faces_to_restore.append({
                        "face_id": face_id,
                        "file_path": face_info["file_path"],
                        "cluster_id": expected_cluster_id,  # Используем ожидаемый кластер из snapshot
                    })
                else:
                    # Персона неправильная или нет персоны
                    faces_with_other_person.append({
                        "face_id": face_id,
                        "file_path": face_info["file_path"],
                        "cluster_id": expected_cluster_id,
                        "current_cluster_id": current_cluster_id,
                        "current_labels": labels,
                    })
        
        if not has_correct_label and not any(f["face_id"] == face_id and f["cluster_id"] == expected_cluster_id for f in faces_to_restore):
            # Лицо существует, есть кластер, но нет правильного face_label - нужно восстановить
            # Используем ожидаемый кластер из snapshot
            faces_to_restore.append({
                "face_id": face_id,
                "file_path": face_info["file_path"],
                "cluster_id": expected_cluster_id,
            })
    
    if ignored_faces:
        print(f"⚠ Лиц помечено как ignore: {len(ignored_faces)}")
        for face_id in ignored_faces[:5]:
            print(f"  - Face ID: {face_id}")
        if len(ignored_faces) > 5:
            print(f"  ... и еще {len(ignored_faces) - 5} лиц")
        print()
    
    if faces_not_in_db:
        print(f"⚠ Лиц не найдено в БД (возможно, удалены): {len(faces_not_in_db)}")
        for face_id in faces_not_in_db[:5]:
            # Пытаемся найти информацию о лице из snapshot
            face_info_from_snapshot = next((f for f in missing_faces_info if f["face_id"] == face_id), None)
            if face_info_from_snapshot:
                print(f"  - Face ID: {face_id}, File: {face_info_from_snapshot['file_path']}, Cluster: {face_info_from_snapshot['cluster_id']}")
            else:
                print(f"  - Face ID: {face_id}")
        if len(faces_not_in_db) > 5:
            print(f"  ... и еще {len(faces_not_in_db) - 5} лиц")
        print()
    
    if faces_without_cluster:
        print(f"⚠ Лиц без кластеров: {len(faces_without_cluster)}")
        for face_id in faces_without_cluster[:5]:
            print(f"  - Face ID: {face_id}")
        if len(faces_without_cluster) > 5:
            print(f"  ... и еще {len(faces_without_cluster) - 5} лиц")
        print()
    
    if faces_with_other_person:
        print(f"⚠ Лиц с другими персонами: {len(faces_with_other_person)}")
        for face_info in faces_with_other_person[:3]:
            print(f"  - Face ID: {face_info['face_id']}, File: {face_info['file_path']}")
            for label in face_info["current_labels"]:
                print(f"    Текущая персона: {label['person_name']} (ID: {label['person_id']}), Cluster: {label['cluster_id']}")
        if len(faces_with_other_person) > 3:
            print(f"  ... и еще {len(faces_with_other_person) - 3} лиц")
        print()
    
    if faces_to_restore:
        print(f"✓ Лиц для восстановления: {len(faces_to_restore)}")
        print("  Эти лица существуют в БД, имеют кластеры, но потеряли связь с персоной.")
        print()
        print("Первые 10 лиц для восстановления:")
        for face_info in faces_to_restore[:10]:
            print(f"  - Face ID: {face_info['face_id']}, Cluster ID: {face_info['cluster_id']}, File: {face_info['file_path']}")
        if len(faces_to_restore) > 10:
            print(f"  ... и еще {len(faces_to_restore) - 10} лиц")
        print()
        
        # Сохраняем список для восстановления
        restore_file = snapshot_file.parent / f"person_{person_id}_restore.json"
        restore_data = {
            "person_id": person_id,
            "person_name": person_name,
            "faces_to_restore": faces_to_restore,
            "total": len(faces_to_restore),
        }
        
        with open(restore_file, "w", encoding="utf-8") as f:
            json.dump(restore_data, f, ensure_ascii=False, indent=2)
        
        print(f"Список для восстановления сохранен: {restore_file}")
        print(f"Для восстановления запустите:")
        print(f"  python backend/scripts/tools/restore_faces_from_snapshot.py --restore-file {restore_file}")


def main():
    parser = argparse.ArgumentParser(description="Сравнение snapshot с текущим состоянием БД")
    parser.add_argument("--snapshot-file", type=str, required=True, help="Путь к JSON файлу со snapshot")
    
    args = parser.parse_args()
    
    snapshot_file = Path(args.snapshot_file)
    if not snapshot_file.exists():
        print(f"Ошибка: файл не найден: {snapshot_file}")
        return
    
    compare_snapshot(snapshot_file)


if __name__ == "__main__":
    main()
