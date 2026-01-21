#!/usr/bin/env python3
"""
Скрипт для восстановления потерянных связей лиц с персонами на основе Gold-данных.

Сравнивает текущее состояние БД с Gold-данными (faces_manual_rects_gold.ndjson) и
восстанавливает связи лиц с персонами, которые были потеряны (например, при объединении кластеров).

Использование:
    python backend/scripts/tools/restore_lost_faces_from_gold.py --person-id 10  # для конкретной персоны
    python backend/scripts/tools/restore_lost_faces_from_gold.py  # для всех персон
    python backend/scripts/tools/restore_lost_faces_from_gold.py --dry-run  # только проверка, без изменений
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection
from backend.logic.gold.store import (
    gold_faces_manual_rects_path,
    gold_read_ndjson_by_path,
)


def find_face_in_db(cur, file_path: str, x: int, y: int, w: int, h: int) -> dict | None:
    """
    Находит лицо в БД по file_path и bbox (с небольшой погрешностью).
    
    Returns:
        dict с информацией о лице или None, если не найдено
    """
    cur.execute(
        """
        SELECT 
            fr.id as face_id,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
            fr.ignore_flag
        FROM face_rectangles fr
        WHERE fr.file_path = ? 
          AND ABS(fr.bbox_x - ?) <= 10
          AND ABS(fr.bbox_y - ?) <= 10
          AND ABS(fr.bbox_w - ?) <= 10
          AND ABS(fr.bbox_h - ?) <= 10
          AND COALESCE(fr.ignore_flag, 0) = 0
        ORDER BY 
          (ABS(fr.bbox_x - ?) + ABS(fr.bbox_y - ?) + ABS(fr.bbox_w - ?) + ABS(fr.bbox_h - ?)) ASC,
          fr.id ASC
        LIMIT 1
        """,
        (file_path, x, y, w, h, x, y, w, h),
    )
    
    row = cur.fetchone()
    if row:
        return {
            "face_id": row["face_id"],
            "bbox_x": row["bbox_x"],
            "bbox_y": row["bbox_y"],
            "bbox_w": row["bbox_w"],
            "bbox_h": row["bbox_h"],
        }
    return None


def get_face_cluster_and_person(cur, face_id: int) -> dict | None:
    """
    Получает информацию о кластере для лица.
    
    Returns:
        dict с cluster_id или None, если кластер не найден
    """
    cur.execute(
        """
        SELECT 
            fcm.cluster_id
        FROM face_cluster_members fcm
        WHERE fcm.face_rectangle_id = ?
        LIMIT 1
        """,
        (face_id,),
    )
    
    row = cur.fetchone()
    if row and "cluster_id" in row.keys() and row["cluster_id"]:
        return {
            "cluster_id": row["cluster_id"],
        }
    return None


def restore_lost_faces(person_id: int | None = None, dry_run: bool = False) -> None:
    """
    Восстанавливает потерянные связи лиц с персонами на основе Gold-данных.
    
    Args:
        person_id: опционально, фильтр по person_id (только для одной персоны)
        dry_run: если True, только проверяет, но не вносит изменения
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Читаем Gold-данные
    gold_path = gold_faces_manual_rects_path()
    if not gold_path.exists():
        print(f"Ошибка: Gold-файл не найден: {gold_path}")
        return
    
    gold_data = gold_read_ndjson_by_path(gold_path)
    if not gold_data:
        print("Gold-данные пусты.")
        return
    
    print(f"Загружено {len(gold_data)} файлов из Gold-данных")
    
    # Группируем лица из Gold по персонам
    # Используем логику, аналогичную api_gold_faces_by_persons:
    # для каждого лица в Gold находим соответствующее лицо в БД и определяем персону
    gold_faces_by_person: dict[int, list[dict]] = defaultdict(list)
    
    gold_file_paths = list(gold_data.keys())
    if not gold_file_paths:
        print("Нет файлов в Gold-данных.")
        return
    
    # Сначала получаем все персоны, у которых есть лица на файлах из gold
    # Используем новую схему: face_person_manual_assignments + face_clusters.person_id
    placeholders = ",".join(["?"] * len(gold_file_paths))
    cur.execute(
        f"""
        SELECT DISTINCT
            p.id as person_id,
            p.name as person_name
        FROM (
            -- Ручные привязки
            SELECT fpma.person_id
            FROM face_person_manual_assignments fpma
            JOIN face_rectangles fr ON fr.id = fpma.face_rectangle_id
            WHERE fr.file_path IN ({placeholders})
            
            UNION
            
            -- Привязки через кластеры
            SELECT fc.person_id
            FROM face_cluster_members fcm
            JOIN face_rectangles fr ON fr.id = fcm.face_rectangle_id
            JOIN face_clusters fc ON fc.id = fcm.cluster_id
            WHERE fr.file_path IN ({placeholders})
              AND fc.person_id IS NOT NULL
        ) person_ids
        JOIN persons p ON p.id = person_ids.person_id
        WHERE COALESCE((SELECT ignore_flag FROM face_rectangles fr WHERE fr.file_path IN ({placeholders}) LIMIT 1), 0) = 0
        ORDER BY 
          CASE WHEN p.name = ? THEN 1 ELSE 0 END,
          p.name
        """,
        gold_file_paths + gold_file_paths + gold_file_paths + ["Посторонние"],
    )
    
    persons_dict = {}
    for row in cur.fetchall():
        person_id_val = row["person_id"]
        person_name = row["person_name"] or f"Person {person_id_val}"
        persons_dict[person_id_val] = person_name
        
        # Фильтруем по person_id, если указан
        if person_id is not None and person_id_val != person_id:
            continue
    
    if not persons_dict:
        print("Не найдено персон для файлов из Gold-данных.")
        return
    
    print(f"Найдено персон: {len(persons_dict)}")
    
    # Для каждого файла и прямоугольника в Gold находим соответствующее лицо в БД
    for file_path, gold_entry in gold_data.items():
        rects = gold_entry.get("rects", [])
        if not isinstance(rects, list):
            continue
        
        for rect in rects:
            if not isinstance(rect, dict):
                continue
            
            try:
                x = int(rect.get("x", 0))
                y = int(rect.get("y", 0))
                w = int(rect.get("w", 0))
                h = int(rect.get("h", 0))
            except (ValueError, TypeError):
                continue
            
            if w <= 0 or h <= 0:
                continue
            
            # Находим лицо в БД по file_path и bbox (с небольшой погрешностью)
            # Используем новую схему: face_person_manual_assignments + face_clusters.person_id
            cur.execute(
                """
                SELECT 
                    fr.id as face_id,
                    fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                    COALESCE(fpma.person_id, fc.person_id) as person_id,
                    COALESCE(p_manual.name, p_cluster.name) as person_name,
                    fcm.cluster_id
                FROM face_rectangles fr
                LEFT JOIN face_person_manual_assignments fpma ON fpma.face_rectangle_id = fr.id
                LEFT JOIN persons p_manual ON p_manual.id = fpma.person_id
                LEFT JOIN face_cluster_members fcm ON fcm.face_rectangle_id = fr.id
                LEFT JOIN face_clusters fc ON fc.id = fcm.cluster_id
                LEFT JOIN persons p_cluster ON p_cluster.id = fc.person_id
                WHERE fr.file_path = ? 
                  AND ABS(fr.bbox_x - ?) <= 10
                  AND ABS(fr.bbox_y - ?) <= 10
                  AND ABS(fr.bbox_w - ?) <= 10
                  AND ABS(fr.bbox_h - ?) <= 10
                  AND COALESCE(fr.ignore_flag, 0) = 0
                ORDER BY 
                  CASE WHEN COALESCE(fpma.person_id, fc.person_id) IS NOT NULL THEN 0 ELSE 1 END,
                  (ABS(fr.bbox_x - ?) + ABS(fr.bbox_y - ?) + ABS(fr.bbox_w - ?) + ABS(fr.bbox_h - ?)) ASC,
                  fr.id ASC
                LIMIT 1
                """,
                (file_path, x, y, w, h, x, y, w, h),
            )
            
            face_row = cur.fetchone()
            
            # Если не нашли точное совпадение, пробуем найти любую персону на этом файле
            if not face_row or "person_id" not in face_row.keys() or not face_row["person_id"] or "face_id" not in face_row.keys() or not face_row["face_id"]:
                cur.execute(
                    """
                    SELECT 
                        fl.person_id,
                        p.name as person_name
                    FROM face_labels fl
                    JOIN persons p ON fl.person_id = p.id
                    JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
                    WHERE fr.file_path = ?
                      AND COALESCE(fr.ignore_flag, 0) = 0
                    LIMIT 1
                    """,
                    (file_path,),
                )
                fallback_row = cur.fetchone()
                if fallback_row:
                    # Находим лицо без персоны
                    face_info = find_face_in_db(cur, file_path, x, y, w, h)
                    if face_info and face_info.get("face_id"):
                        person_id_from_gold = fallback_row["person_id"]
                        person_name = fallback_row["person_name"]
                        
                        # Фильтруем по person_id, если указан
                        if person_id is not None and person_id_from_gold != person_id:
                            continue
                        
                        # Дедупликация: проверяем, нет ли уже этого лица
                        is_duplicate = False
                        for existing_face in gold_faces_by_person[person_id_from_gold]:
                            if existing_face["face_id"] == face_info["face_id"]:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            gold_faces_by_person[person_id_from_gold].append({
                                "face_id": face_info["face_id"],
                                "file_path": file_path,
                                "bbox": {"x": x, "y": y, "w": w, "h": h},
                                "person_name": person_name,
                            })
            else:
                # Нашли лицо с персоной
                face_id_val = face_row["face_id"] if "face_id" in face_row.keys() else None
                if not face_id_val:
                    continue
                
                person_id_from_gold = face_row["person_id"] if "person_id" in face_row.keys() else None
                person_name = face_row["person_name"] if "person_name" in face_row.keys() else None
                
                if not person_id_from_gold:
                    continue
                
                # Фильтруем по person_id, если указан
                if person_id is not None and person_id_from_gold != person_id:
                    continue
                
                # Дедупликация: проверяем, нет ли уже этого лица
                is_duplicate = False
                for existing_face in gold_faces_by_person[person_id_from_gold]:
                    if existing_face["face_id"] == face_id_val:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    gold_faces_by_person[person_id_from_gold].append({
                        "face_id": face_id_val,
                        "file_path": file_path,
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "person_name": person_name,
                    })
    
    if not gold_faces_by_person:
        print("Не найдено лиц в Gold-данных для восстановления.")
        return
    
    print(f"\nНайдено персон в Gold: {len(gold_faces_by_person)}")
    
    # Для каждой персоны проверяем, какие лица потеряли связь
    total_restored = 0
    total_checked = 0
    
    for person_id_val, gold_faces in gold_faces_by_person.items():
        person_name = gold_faces[0]["person_name"] if gold_faces else f"Person {person_id_val}"
        print(f"\nПерсона: {person_name} (ID: {person_id_val})")
        print(f"  Лиц в Gold: {len(gold_faces)}")
        
        # Проверяем текущее состояние в БД (ручные привязки + через кластеры)
        cur.execute(
            """
            SELECT COUNT(DISTINCT face_id) as faces_count
            FROM (
                SELECT fr.id as face_id
                FROM face_person_manual_assignments fpma
                JOIN face_rectangles fr ON fr.id = fpma.face_rectangle_id
                WHERE fpma.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
                
                UNION
                
                SELECT fr.id as face_id
                FROM face_cluster_members fcm
                JOIN face_rectangles fr ON fr.id = fcm.face_rectangle_id
                JOIN face_clusters fc ON fc.id = fcm.cluster_id
                WHERE fc.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
            )
            """,
            (person_id_val, person_id_val),
        )
        db_row = cur.fetchone()
        db_faces_count = db_row["faces_count"] if db_row else 0
        
        print(f"  Лиц в БД: {db_faces_count}")
        
        if db_faces_count >= len(gold_faces):
            print(f"  ✓ Все лица на месте (или больше)")
            continue
        
        # Находим лица, которые есть в Gold, но потеряли связь с персоной
        faces_to_restore = []
        
        for gold_face in gold_faces:
            face_id = gold_face["face_id"]
            total_checked += 1
            
            # Проверяем, есть ли привязка для этого лица и персоны (ручная или через кластер)
            cur.execute(
                """
                SELECT 
                    COALESCE(fpma.id, 1) as id,
                    fcm.cluster_id
                FROM face_rectangles fr
                LEFT JOIN face_person_manual_assignments fpma ON fpma.face_rectangle_id = fr.id AND fpma.person_id = ?
                LEFT JOIN face_cluster_members fcm ON fcm.face_rectangle_id = fr.id
                LEFT JOIN face_clusters fc ON fc.id = fcm.cluster_id AND fc.person_id = ?
                WHERE fr.id = ?
                LIMIT 1
                """,
                (person_id_val, person_id_val, face_id),
            )
            
            existing_label = cur.fetchone()
            
            if not existing_label:
                # Лицо потеряло связь с персоной
                # Нужно найти кластер для этого лица и восстановить связь
                cluster_info = get_face_cluster_and_person(cur, face_id)
                
                if cluster_info and "cluster_id" in cluster_info and cluster_info["cluster_id"]:
                    faces_to_restore.append({
                        "face_id": face_id,
                        "cluster_id": cluster_info["cluster_id"],
                        "file_path": gold_face["file_path"],
                        "bbox": gold_face["bbox"],
                    })
                else:
                    # Лицо не в кластере - тоже можно восстановить (создать ручную привязку)
                    faces_to_restore.append({
                        "face_id": face_id,
                        "cluster_id": None,
                        "file_path": gold_face["file_path"],
                        "bbox": gold_face["bbox"],
                    })
        
        if faces_to_restore:
            print(f"  ⚠ Найдено потерянных лиц: {len(faces_to_restore)}")
            
            if dry_run:
                print(f"  [DRY RUN] Будет восстановлено {len(faces_to_restore)} лиц:")
                for face_info in faces_to_restore[:10]:  # Показываем первые 10
                    print(f"    - Face ID: {face_info['face_id']}, Cluster ID: {face_info['cluster_id']}, File: {face_info['file_path']}")
                if len(faces_to_restore) > 10:
                    print(f"    ... и еще {len(faces_to_restore) - 10} лиц")
            else:
                # Восстанавливаем связи
                now = datetime.now(timezone.utc).isoformat()
                restored_count = 0
                
                for face_info in faces_to_restore:
                    face_id = face_info["face_id"]
                    cluster_id = face_info.get("cluster_id")
                    
                    # ВАЖНО: НЕ создаем записи для лиц, которые уже в кластерах с правильной персоной
                    # Проверяем, есть ли уже привязка через кластер
                    cur.execute(
                        """
                        SELECT fc.person_id
                        FROM face_cluster_members fcm
                        JOIN face_clusters fc ON fc.id = fcm.cluster_id
                        WHERE fcm.face_rectangle_id = ? AND fc.person_id = ?
                        """,
                        (face_id, person_id_val),
                    )
                    cluster_assignment = cur.fetchone()
                    
                    if cluster_assignment:
                        # Лицо уже в кластере с правильной персоной - НЕ создаем ручную привязку
                        continue
                    
                    # Проверяем, нет ли уже ручной привязки
                    cur.execute(
                        """
                        SELECT id FROM face_person_manual_assignments
                        WHERE face_rectangle_id = ? AND person_id = ?
                        """,
                        (face_id, person_id_val),
                    )
                    existing = cur.fetchone()
                    
                    if existing:
                        # Запись уже существует, пропускаем
                        continue
                    
                    # Создаем ручную привязку ТОЛЬКО если лицо НЕ в кластере с правильной персоной
                    try:
                        cur.execute(
                            """
                            INSERT INTO face_person_manual_assignments (face_rectangle_id, person_id, source, confidence, created_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (face_id, person_id_val, "restored_from_gold", 0.9, now),
                        )
                        restored_count += 1
                    except Exception as e:
                        # Ошибка при создании (например, нарушение ограничений БД)
                        print(f"    ⚠ Не удалось восстановить Face ID {face_id}: {e}")
                
                conn.commit()
                total_restored += restored_count
                print(f"  ✓ Восстановлено: {restored_count} лиц")
        else:
            print(f"  ✓ Все лица имеют связи с персоной")
    
    print(f"\n{'=' * 60}")
    print(f"Итого проверено лиц: {total_checked}")
    if not dry_run:
        print(f"Восстановлено связей: {total_restored}")
    else:
        print(f"[DRY RUN] Изменения не внесены")


def main():
    parser = argparse.ArgumentParser(description="Восстановление потерянных связей лиц с персонами из Gold-данных")
    parser.add_argument("--person-id", type=int, default=None, help="ID персоны для восстановления (опционально)")
    parser.add_argument("--dry-run", action="store_true", help="Только проверка, без внесения изменений")
    
    args = parser.parse_args()
    
    restore_lost_faces(person_id=args.person_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
