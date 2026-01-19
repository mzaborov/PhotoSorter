#!/usr/bin/env python3
"""
Скрипт для сохранения снимка состояния лиц персоны в файл.

Сохраняет полную информацию о всех лицах персоны (ID лиц, кластеры, файлы) в JSON файл
для последующей проверки, что лица не пропали после операций (например, объединения кластеров).

Использование:
    python backend/scripts/tools/save_person_faces_snapshot.py --person-id 12
    python backend/scripts/tools/save_person_faces_snapshot.py --person-id 12 --output person_12_snapshot.json
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def save_person_faces_snapshot(person_id: int, output_file: Path | None = None) -> None:
    """
    Сохраняет снимок состояния лиц персоны в файл.
    
    Args:
        person_id: ID персоны
        output_file: путь к файлу для сохранения (если None, используется person_{person_id}_snapshot.json)
    """
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
    
    # Получаем все лица персоны с полной информацией
    cur.execute(
        """
        SELECT 
            fl.face_rectangle_id as face_id,
            fl.cluster_id,
            fr.file_path,
            fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
            fr.confidence,
            fl.source,
            fl.created_at as label_created_at
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? 
          AND COALESCE(fr.ignore_flag, 0) = 0
        ORDER BY fl.cluster_id, fl.face_rectangle_id
        """,
        (person_id,),
    )
    
    faces = []
    for row in cur.fetchall():
        faces.append({
            "face_id": row["face_id"],
            "cluster_id": row["cluster_id"],
            "file_path": row["file_path"],
            "face_index": row["face_index"],
            "bbox": {
                "x": row["bbox_x"],
                "y": row["bbox_y"],
                "w": row["bbox_w"],
                "h": row["bbox_h"],
            },
            "confidence": row["confidence"],
            "source": row["source"],
            "label_created_at": row["label_created_at"],
        })
    
    # Группируем по кластерам для статистики
    clusters_stats = {}
    for face in faces:
        cluster_id = face["cluster_id"]
        if cluster_id not in clusters_stats:
            clusters_stats[cluster_id] = {
                "cluster_id": cluster_id,
                "faces_count": 0,
                "face_ids": [],
            }
        clusters_stats[cluster_id]["faces_count"] += 1
        clusters_stats[cluster_id]["face_ids"].append(face["face_id"])
    
    # Формируем итоговый снимок
    snapshot = {
        "person_id": person_id,
        "person_name": person_name,
        "snapshot_created_at": datetime.now(timezone.utc).isoformat(),
        "total_faces": len(faces),
        "total_clusters": len(clusters_stats),
        "clusters_stats": list(clusters_stats.values()),
        "faces": faces,
    }
    
    # Определяем путь к файлу
    if output_file is None:
        output_file = project_root / "backend" / "data" / f"person_{person_id}_snapshot.json"
    else:
        output_file = Path(output_file)
    
    # Создаем директорию, если её нет
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем в JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    
    print(f"Снимок сохранен: {output_file}")
    print(f"Персона: {person_name} (ID: {person_id})")
    print(f"Лиц: {len(faces)}")
    print(f"Кластеров: {len(clusters_stats)}")
    print()
    print("Статистика по кластерам:")
    for cluster_stat in sorted(clusters_stats.values(), key=lambda x: x["cluster_id"]):
        print(f"  Кластер #{cluster_stat['cluster_id']}: {cluster_stat['faces_count']} лиц")


def main():
    parser = argparse.ArgumentParser(description="Сохранение снимка состояния лиц персоны")
    parser.add_argument("--person-id", type=int, required=True, help="ID персоны")
    parser.add_argument("--output", type=str, default=None, help="Путь к файлу для сохранения (по умолчанию: backend/data/person_{person_id}_snapshot.json)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    save_person_faces_snapshot(person_id=args.person_id, output_file=output_path)


if __name__ == "__main__":
    main()
