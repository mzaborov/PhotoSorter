#!/usr/bin/env python3
"""
Скрипт для очистки дубликатов в face_labels.

Находит все дубликаты (несколько записей с одинаковыми face_rectangle_id и person_id)
и оставляет только одну запись (самую новую по created_at или с максимальным id).

ВАЖНО: После очистки будет создан UNIQUE индекс для предотвращения дубликатов в будущем.

Использование:
    python backend/scripts/tools/cleanup_duplicate_face_labels.py --dry-run  # только показать дубликаты
    python backend/scripts/tools/cleanup_duplicate_face_labels.py  # очистить дубликаты
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(project_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(project_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection


def find_duplicates() -> list[dict]:
    """Находит все дубликаты в face_labels."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Находим все комбинации (face_rectangle_id, person_id) с несколькими записями
    cur.execute("""
        SELECT 
            face_rectangle_id,
            person_id,
            COUNT(*) as duplicate_count,
            GROUP_CONCAT(id) as label_ids,
            GROUP_CONCAT(created_at) as created_ats
        FROM face_labels
        GROUP BY face_rectangle_id, person_id
        HAVING duplicate_count > 1
        ORDER BY duplicate_count DESC, face_rectangle_id
    """)
    
    duplicates = []
    for row in cur.fetchall():
        duplicates.append({
            "face_rectangle_id": row["face_rectangle_id"],
            "person_id": row["person_id"],
            "duplicate_count": row["duplicate_count"],
            "label_ids": [int(lid) for lid in row["label_ids"].split(",") if lid.strip()],
        })
    
    return duplicates


def cleanup_duplicates(dry_run: bool = False) -> None:
    """
    Очищает дубликаты в face_labels.
    
    Args:
        dry_run: Если True, только показывает дубликаты без удаления
    """
    conn = get_connection()
    cur = conn.cursor()
    
    duplicates = find_duplicates()
    
    if len(duplicates) == 0:
        print("Дубликатов не найдено.")
        return
    
    print(f"Найдено дубликатов: {len(duplicates)}")
    print(f"Всего избыточных записей: {sum(d['duplicate_count'] - 1 for d in duplicates)}")
    print()
    
    if dry_run:
        print("=" * 80)
        print("РЕЖИМ ПРОСМОТРА (dry-run) - изменения не будут применены")
        print("=" * 80)
        print()
        
        # Показываем первые 20 дубликатов
        for i, dup in enumerate(duplicates[:20], 1):
            print(f"{i}. Face ID: {dup['face_rectangle_id']}, Person ID: {dup['person_id']}, "
                  f"Дубликатов: {dup['duplicate_count']}, Label IDs: {dup['label_ids']}")
        
        if len(duplicates) > 20:
            print(f"... и еще {len(duplicates) - 20} дубликатов")
        
        print()
        print("Для очистки запустите скрипт без флага --dry-run")
        return
    
    print("=" * 80)
    print("ОЧИСТКА ДУБЛИКАТОВ")
    print("=" * 80)
    print()
    
    total_deleted = 0
    
    for dup in duplicates:
        face_id = dup["face_rectangle_id"]
        person_id = dup["person_id"]
        label_ids = dup["label_ids"]
        
        # Оставляем запись с максимальным id (самую новую)
        keep_id = max(label_ids)
        delete_ids = [lid for lid in label_ids if lid != keep_id]
        
        # Удаляем дубликаты
        placeholders = ",".join("?" * len(delete_ids))
        cur.execute(
            f"DELETE FROM face_labels WHERE id IN ({placeholders})",
            delete_ids
        )
        
        deleted_count = cur.rowcount
        total_deleted += deleted_count
        
        if len(duplicates) <= 50:  # Показываем детали только для небольшого количества
            print(f"Face ID: {face_id}, Person ID: {person_id}: "
                  f"удалено {deleted_count} дубликатов, оставлен ID {keep_id}")
    
    conn.commit()
    
    print()
    print("=" * 80)
    print("ОЧИСТКА ЗАВЕРШЕНА")
    print("=" * 80)
    print(f"Удалено избыточных записей: {total_deleted}")
    print()
    
    # Проверяем, что дубликатов больше нет
    remaining = find_duplicates()
    if len(remaining) == 0:
        print("✓ Дубликатов не осталось")
    else:
        print(f"⚠ ВНИМАНИЕ: Осталось {len(remaining)} дубликатов!")
    
    print()
    print("ВАЖНО: Убедитесь, что в схеме БД создан UNIQUE индекс:")
    print("  CREATE UNIQUE INDEX IF NOT EXISTS idx_face_labels_unique ON face_labels(face_rectangle_id, person_id);")


def ensure_unique_index() -> None:
    """Создает UNIQUE индекс, если его еще нет."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, существует ли индекс
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='index' AND name='idx_face_labels_unique'
    """)
    
    if cur.fetchone():
        print("UNIQUE индекс idx_face_labels_unique уже существует.")
        return
    
    print("Создаём UNIQUE индекс idx_face_labels_unique...")
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_face_labels_unique 
        ON face_labels(face_rectangle_id, person_id)
    """)
    conn.commit()
    print("✓ UNIQUE индекс создан")


def main() -> int:
    parser = argparse.ArgumentParser(description="Очистка дубликатов в face_labels")
    parser.add_argument("--dry-run", action="store_true", help="Только показать дубликаты без удаления")
    parser.add_argument("--create-index", action="store_true", help="Создать UNIQUE индекс после очистки")
    
    args = parser.parse_args()
    
    cleanup_duplicates(dry_run=args.dry_run)
    
    if args.create_index and not args.dry_run:
        print()
        ensure_unique_index()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
