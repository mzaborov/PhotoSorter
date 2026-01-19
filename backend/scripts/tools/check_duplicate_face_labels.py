#!/usr/bin/env python3
"""
Скрипт для проверки дублей записей в face_labels.

Находит случаи, когда одно лицо имеет несколько записей face_labels для одного и того же кластера и персоны.
Это может быть проблемой, которая приводит к неправильному подсчету лиц.

Использование:
    python backend/scripts/tools/check_duplicate_face_labels.py --person-id 1
    python backend/scripts/tools/check_duplicate_face_labels.py  # все персоны
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def check_duplicate_face_labels(person_id: int | None = None) -> None:
    """
    Проверяет дубликаты записей в face_labels.
    
    Args:
        person_id: опционально, фильтр по person_id
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие для фильтрации по персоне
    person_filter = ""
    person_params = []
    if person_id is not None:
        person_filter = "AND fl.person_id = ?"
        person_params = [person_id]
    
    # Находим все записи face_labels, где одно лицо имеет несколько записей для одного кластера и персоны
    sql_query = f"""
        SELECT 
            fl.face_rectangle_id,
            fl.person_id,
            fl.cluster_id,
            COUNT(*) as label_count,
            GROUP_CONCAT(fl.id) as label_ids,
            fr.file_path,
            p.name as person_name
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        LEFT JOIN persons p ON fl.person_id = p.id
        WHERE COALESCE(fr.ignore_flag, 0) = 0
          {person_filter}
        GROUP BY fl.face_rectangle_id, fl.person_id, fl.cluster_id
        HAVING label_count > 1
        ORDER BY label_count DESC, fl.face_rectangle_id
    """
    
    cur.execute(sql_query, person_params)
    duplicates = cur.fetchall()
    
    if len(duplicates) == 0:
        if person_id is not None:
            print(f"Дубликатов записей в face_labels для персоны {person_id} не найдено.")
        else:
            print("Дубликатов записей в face_labels не найдено.")
        return
    
    print(f"Найдено дубликатов записей в face_labels: {len(duplicates)}")
    if person_id is not None:
        print(f"Для персоны: {person_id}")
    print()
    
    # Группируем по персонам для более удобного отображения
    persons_duplicates: dict[int, list[dict]] = defaultdict(list)
    
    for dup in duplicates:
        person_id_val = dup["person_id"]
        persons_duplicates[person_id_val].append({
            "face_id": dup["face_rectangle_id"],
            "cluster_id": dup["cluster_id"],
            "label_count": dup["label_count"],
            "label_ids": dup["label_ids"],
            "file_path": dup["file_path"],
            "person_name": dup["person_name"],
        })
    
    # Выводим информацию по персонам
    print("=" * 80)
    print("Дубликаты по персонам:")
    print("=" * 80)
    
    for person_id_val in sorted(persons_duplicates.keys()):
        faces_in_person = persons_duplicates[person_id_val]
        person_name = faces_in_person[0]["person_name"] if faces_in_person else f"Person {person_id_val}"
        
        print(f"\nПерсона: {person_name} (ID: {person_id_val})")
        print(f"  Лиц с дубликатами записей: {len(faces_in_person)}")
        
        # Показываем первые 10 дубликатов
        for i, face_info in enumerate(faces_in_person[:10], 1):
            label_ids = [int(lid) for lid in face_info["label_ids"].split(",") if lid.strip()]
            print(f"  {i}. Face ID: {face_info['face_id']}, Cluster ID: {face_info['cluster_id']}, "
                  f"Записей: {face_info['label_count']}, Label IDs: {label_ids}")
            print(f"     Файл: {face_info['file_path']}")
        
        if len(faces_in_person) > 10:
            print(f"  ... и еще {len(faces_in_person) - 10} лиц с дубликатами")
    
    # Статистика
    print("\n" + "=" * 80)
    print("Статистика:")
    print("=" * 80)
    
    total_duplicate_labels = sum(dup["label_count"] for dup in duplicates)
    total_excess_labels = total_duplicate_labels - len(duplicates)
    
    print(f"Уникальных комбинаций (face_id, person_id, cluster_id) с дубликатами: {len(duplicates)}")
    print(f"Всего записей face_labels для дубликатов: {total_duplicate_labels}")
    print(f"Избыточных записей (которые можно удалить): {total_excess_labels}")
    print(f"Персон с дубликатами: {len(persons_duplicates)}")


def main():
    parser = argparse.ArgumentParser(description="Проверка дублей записей в face_labels")
    parser.add_argument("--person-id", type=int, default=None, help="ID персоны для фильтрации (опционально)")
    
    args = parser.parse_args()
    
    check_duplicate_face_labels(person_id=args.person_id)


if __name__ == "__main__":
    main()
