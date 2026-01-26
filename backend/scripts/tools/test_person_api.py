#!/usr/bin/env python3
"""
Скрипт для тестирования API персон без UI.

Использование:
    python backend/scripts/tools/test_person_api.py --person-id 1
    python backend/scripts/tools/test_person_api.py --stats
"""

import sys
import argparse
import json
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import get_connection


def test_persons_stats():
    """Тестирует API /api/persons/stats"""
    print("=" * 80)
    print("Тест: /api/persons/stats")
    print("=" * 80)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем ID персоны "Посторонний"
    cur.execute("SELECT id FROM persons WHERE name = 'Посторонний' LIMIT 1")
    outsider_row = cur.fetchone()
    outsider_person_id = outsider_row["id"] if outsider_row else None
    
    # Тестируем запросы из api_persons_stats
    print("\n1. Проверка person_rectangles с разделением на архив/прогон:")
    cur.execute("""
        SELECT 
            pr.person_id,
            COUNT(DISTINCT CASE WHEN f.path LIKE 'disk:/Фото%' THEN pr.file_id END) as files_archive,
            COUNT(DISTINCT CASE WHEN f.path NOT LIKE 'disk:/Фото%' AND pr.pipeline_run_id IS NOT NULL THEN pr.file_id END) as files_run,
            COUNT(DISTINCT pr.file_id) as files_total
        FROM person_rectangles pr
        LEFT JOIN files f ON pr.file_id = f.id
        GROUP BY pr.person_id
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  Person {row['person_id']}: архив={row['files_archive']}, прогон={row['files_run']}, всего={row['files_total']}")
    
    print("\n2. Проверка file_persons с разделением на архив/прогон:")
    cur.execute("""
        SELECT 
            fp.person_id,
            COUNT(DISTINCT CASE WHEN f.path LIKE 'disk:/Фото%' THEN fp.file_id END) as files_archive,
            COUNT(DISTINCT CASE WHEN f.path NOT LIKE 'disk:/Фото%' AND fp.pipeline_run_id IS NOT NULL THEN fp.file_id END) as files_run,
            COUNT(DISTINCT fp.file_id) as files_total
        FROM file_persons fp
        LEFT JOIN files f ON fp.file_id = f.id
        GROUP BY fp.person_id
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  Person {row['person_id']}: архив={row['files_archive']}, прогон={row['files_run']}, всего={row['files_total']}")
    
    print("\n3. Проверка лиц через кластеры:")
    cur.execute("""
        SELECT 
            fc.person_id,
            COUNT(DISTINCT CASE WHEN fr.archive_scope = 'archive' THEN fr.id END) as faces_archive,
            COUNT(DISTINCT CASE WHEN (fr.archive_scope IS NULL OR fr.archive_scope = '') AND fr.run_id IS NOT NULL THEN fr.id END) as faces_run,
            COUNT(DISTINCT fr.id) as faces_total
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        JOIN photo_rectangles fr ON fcm.rectangle_id = fr.id
        WHERE fc.person_id IS NOT NULL 
          AND fr.is_face = 1
          AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY fc.person_id
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  Person {row['person_id']}: архив={row['faces_archive']}, прогон={row['faces_run']}, всего={row['faces_total']}")
    
    print("\n✅ Тест завершен")


def test_person_detail(person_id: int):
    """Тестирует API /api/persons/{person_id}"""
    print("=" * 80)
    print(f"Тест: /api/persons/{person_id}")
    print("=" * 80)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем существование персоны
    cur.execute("SELECT id, name FROM persons WHERE id = ?", (person_id,))
    person_row = cur.fetchone()
    if not person_row:
        print(f"❌ Персона с ID {person_id} не найдена")
        return
    
    print(f"\nПерсона: {person_row['name']} (ID: {person_id})")
    
    # Тестируем запросы из api_person_detail
    print("\n1. Элементы через кластеры:")
    cur.execute("""
        SELECT DISTINCT
            fr.id as face_id,
            fr.archive_scope,
            f.path as file_path,
            f.id as file_id,
            'cluster' as assignment_type
        FROM face_cluster_members fcm
        JOIN face_clusters fc ON fcm.cluster_id = fc.id
        JOIN photo_rectangles fr ON fcm.rectangle_id = fr.id
        LEFT JOIN files f ON fr.file_id = f.id
        WHERE fc.person_id = ? 
          AND fr.is_face = 1
          AND COALESCE(fr.ignore_flag, 0) = 0
        LIMIT 5
    """, (person_id,))
    cluster_items = cur.fetchall()
    print(f"  Найдено элементов: {len(cluster_items)}")
    for item in cluster_items[:3]:
        print(f"    - face_id={item['face_id']}, type={item['assignment_type']}, archive_scope={item['archive_scope']}, path={item['file_path'][:50] if item['file_path'] else 'None'}...")
    
    print("\n2. Элементы через ручные привязки:")
    cur.execute("""
        SELECT DISTINCT
            fr.id as face_id,
            fr.archive_scope,
            f.path as file_path,
            f.id as file_id,
            'manual_face' as assignment_type
        FROM person_rectangle_manual_assignments fpma
        JOIN photo_rectangles fr ON fpma.rectangle_id = fr.id
        LEFT JOIN files f ON fr.file_id = f.id
        WHERE fpma.person_id = ?
          AND fr.is_face = 1
          AND COALESCE(fr.ignore_flag, 0) = 0
        LIMIT 5
    """, (person_id,))
    manual_items = cur.fetchall()
    print(f"  Найдено элементов: {len(manual_items)}")
    for item in manual_items[:3]:
        print(f"    - face_id={item['face_id']}, type={item['assignment_type']}, archive_scope={item['archive_scope']}, path={item['file_path'][:50] if item['file_path'] else 'None'}...")
    
    print("\n3. Элементы через person_rectangles:")
    cur.execute("""
        SELECT DISTINCT
            pr.id as person_rectangle_id,
            CASE WHEN f.path LIKE 'disk:/Фото%' THEN 'archive' ELSE NULL END as archive_scope,
            f.path as file_path,
            pr.file_id,
            'person_rectangle' as assignment_type
        FROM person_rectangles pr
        LEFT JOIN files f ON pr.file_id = f.id
        WHERE pr.person_id = ?
        LIMIT 5
    """, (person_id,))
    rect_items = cur.fetchall()
    print(f"  Найдено элементов: {len(rect_items)}")
    for item in rect_items[:3]:
        print(f"    - person_rectangle_id={item['person_rectangle_id']}, type={item['assignment_type']}, archive_scope={item['archive_scope']}, path={item['file_path'][:50] if item['file_path'] else 'None'}...")
    
    print("\n4. Элементы через file_persons:")
    cur.execute("""
        SELECT DISTINCT
            NULL as face_id,
            CASE WHEN f.path LIKE 'disk:/Фото%' THEN 'archive' ELSE NULL END as archive_scope,
            f.path as file_path,
            fp.file_id,
            'file_person' as assignment_type
        FROM file_persons fp
        LEFT JOIN files f ON fp.file_id = f.id
        WHERE fp.person_id = ?
        LIMIT 5
    """, (person_id,))
    file_items = cur.fetchall()
    print(f"  Найдено элементов: {len(file_items)}")
    for item in file_items[:3]:
        print(f"    - file_id={item['file_id']}, type={item['assignment_type']}, archive_scope={item['archive_scope']}, path={item['file_path'][:50] if item['file_path'] else 'None'}...")
    
    print("\n✅ Тест завершен")


def main():
    parser = argparse.ArgumentParser(description='Тестирование API персон без UI')
    parser.add_argument('--person-id', type=int, help='ID персоны для тестирования api_person_detail')
    parser.add_argument('--stats', action='store_true', help='Тестировать api_persons_stats')
    
    args = parser.parse_args()
    
    if args.stats:
        test_persons_stats()
    elif args.person_id:
        test_person_detail(args.person_id)
    else:
        parser.print_help()
        print("\nПримеры использования:")
        print("  python backend/scripts/tools/test_person_api.py --stats")
        print("  python backend/scripts/tools/test_person_api.py --person-id 1")


if __name__ == '__main__':
    main()
