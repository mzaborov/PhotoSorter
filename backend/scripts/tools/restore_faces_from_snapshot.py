#!/usr/bin/env python3
"""
Скрипт для восстановления лиц персоны на основе snapshot или restore файла.

Использование:
    python backend/scripts/tools/restore_faces_from_snapshot.py --restore-file backend/data/person_1_restore.json
    python backend/scripts/tools/restore_faces_from_snapshot.py --snapshot-file backend/data/person_1_snapshot.json --dry-run
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


def restore_faces_from_restore_file(restore_file: Path, dry_run: bool = False) -> None:
    """
    Восстанавливает лица из restore файла.
    
    Args:
        restore_file: путь к JSON файлу со списком лиц для восстановления
        dry_run: если True, только проверяет, но не вносит изменения
    """
    # Читаем restore файл
    with open(restore_file, "r", encoding="utf-8") as f:
        restore_data = json.load(f)
    
    person_id = restore_data["person_id"]
    person_name = restore_data["person_name"]
    faces_to_restore = restore_data["faces_to_restore"]
    
    print(f"Персона: {person_name} (ID: {person_id})")
    print(f"Лиц для восстановления: {len(faces_to_restore)}")
    print()
    
    if dry_run:
        print("[DRY RUN] Будет восстановлено:")
        for face_info in faces_to_restore[:10]:
            print(f"  - Face ID: {face_info['face_id']}, Cluster ID: {face_info['cluster_id']}, File: {face_info['file_path']}")
        if len(faces_to_restore) > 10:
            print(f"  ... и еще {len(faces_to_restore) - 10} лиц")
        return
    
    # Восстанавливаем связи
    conn = get_connection()
    cur = conn.cursor()
    
    now = datetime.now(timezone.utc).isoformat()
    restored_count = 0
    errors = []
    
    for face_info in faces_to_restore:
        face_id = face_info["face_id"]
        cluster_id = face_info["cluster_id"]
        
        # Проверяем, нет ли уже такой записи
        cur.execute(
            """
            SELECT id FROM face_labels
            WHERE face_rectangle_id = ? AND person_id = ? AND cluster_id = ?
            """,
            (face_id, person_id, cluster_id),
        )
        existing = cur.fetchone()
        
        if existing:
            # Запись уже существует, пропускаем
            continue
        
        # Создаем face_label для этого лица
        try:
            cur.execute(
                """
                INSERT INTO face_labels (face_rectangle_id, person_id, cluster_id, source, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (face_id, person_id, cluster_id, "restored_from_snapshot", 0.9, now),
            )
            restored_count += 1
        except Exception as e:
            errors.append({
                "face_id": face_id,
                "error": str(e),
            })
            print(f"  ⚠ Ошибка при восстановлении Face ID {face_id}: {e}")
    
    conn.commit()
    
    print(f"✓ Восстановлено: {restored_count} лиц")
    if errors:
        print(f"⚠ Ошибок: {len(errors)}")


def restore_faces_from_snapshot(snapshot_file: Path, dry_run: bool = False) -> None:
    """
    Восстанавливает лица из snapshot файла (сначала сравнивает с текущим состоянием).
    
    Args:
        snapshot_file: путь к JSON файлу со snapshot
        dry_run: если True, только проверяет, но не вносит изменения
    """
    # Импортируем функцию сравнения
    from backend.scripts.tools.compare_person_snapshot import compare_snapshot
    
    # Сначала сравниваем
    compare_snapshot(snapshot_file)
    
    # Если есть restore файл, восстанавливаем из него
    snapshot_path = Path(snapshot_file)
    restore_file = snapshot_path.parent / f"person_{snapshot_path.stem.split('_')[1]}_restore.json"
    
    if restore_file.exists():
        print()
        print("Найден файл для восстановления. Восстанавливаем...")
        restore_faces_from_restore_file(restore_file, dry_run=dry_run)
    else:
        print()
        print("Файл для восстановления не найден. Запустите compare_person_snapshot.py сначала.")


def main():
    parser = argparse.ArgumentParser(description="Восстановление лиц персоны из snapshot")
    parser.add_argument("--restore-file", type=str, default=None, help="Путь к JSON файлу со списком лиц для восстановления")
    parser.add_argument("--snapshot-file", type=str, default=None, help="Путь к JSON файлу со snapshot (сначала выполнит сравнение)")
    parser.add_argument("--dry-run", action="store_true", help="Только проверка, без внесения изменений")
    
    args = parser.parse_args()
    
    if args.restore_file:
        restore_file = Path(args.restore_file)
        if not restore_file.exists():
            print(f"Ошибка: файл не найден: {restore_file}")
            return
        restore_faces_from_restore_file(restore_file, dry_run=args.dry_run)
    elif args.snapshot_file:
        snapshot_file = Path(args.snapshot_file)
        if not snapshot_file.exists():
            print(f"Ошибка: файл не найден: {snapshot_file}")
            return
        restore_faces_from_snapshot(snapshot_file, dry_run=args.dry_run)
    else:
        print("Ошибка: необходимо указать --restore-file или --snapshot-file")
        parser.print_help()


if __name__ == "__main__":
    main()
