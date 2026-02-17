#!/usr/bin/env python3
"""
Синхронизация списка поездок с папками на Я.Диске.

Читает корневую папку поездок (TRIPS_YD_ROOT, по умолчанию disk:/Фото/Путешествия),
получает список подпапок на ЯД; для каждой подпапки создаёт запись в trips, если такой
yd_folder_path ещё нет. Имя поездки = имя папки.

После этого импорт из Excel/CSV используется только для обогащения (даты, страна).

Запуск: python backend/scripts/tools/sync_trips_from_yadisk.py [--root путь] [--clear] [--dry-run]
  --clear  перед синхронизацией удалить все поездки и привязки (trip_files, trips).
Переменная окружения: TRIPS_YD_ROOT (например disk:/Фото/Путешествия).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

# После добавления backend в path
from common.db import get_connection, get_trip_by_yd_folder_path, init_db, trip_create_from_folder
from common.yadisk_client import get_disk


def _normalize_yadisk_path(path: str) -> str:
    p = (path or "").strip()
    if p.startswith("disk:"):
        p = p[len("disk:") :]
    if not p.startswith("/"):
        p = "/" + p
    return p


def _as_disk_path(path: str) -> str:
    p = (path or "").strip()
    if p.startswith("disk:"):
        return p
    if not p.startswith("/"):
        p = "/" + p
    return "disk:" + p


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Синхронизация поездок с папками Я.Диска")
    parser.add_argument(
        "--root",
        default=os.getenv("TRIPS_YD_ROOT", "disk:/Фото/Путешествия"),
        help="Корневая папка поездок на ЯД (по умолчанию TRIPS_YD_ROOT или disk:/Фото/Путешествия)",
    )
    parser.add_argument("--clear", action="store_true", help="Удалить все поездки и привязки перед синхронизацией")
    parser.add_argument("--dry-run", action="store_true", help="Не писать в БД, только вывести список папок")
    args = parser.parse_args()

    init_db()

    if args.clear and not args.dry_run:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM trip_files")
            cur.execute("DELETE FROM trips")
            conn.commit()
            print("Удалены все записи из trip_files и trips.")
        finally:
            conn.close()
    elif args.clear and args.dry_run:
        print("[dry-run] С --clear были бы удалены все поездки и привязки.")

    root = (args.root or "").strip() or "disk:/Фото/Путешествия"
    root_norm = _normalize_yadisk_path(root)

    disk = get_disk()
    try:
        items = list(disk.listdir(root_norm))
    except Exception as e:
        print(f"Ошибка при чтении папки {root}: {e}", file=sys.stderr)
        sys.exit(1)

    created = 0
    skipped = 0
    for item in items:
        if getattr(item, "type", None) != "dir":
            continue
        name = (getattr(item, "name", None) or "").strip() or None
        path_yd = getattr(item, "path", None)
        if not name or not path_yd:
            continue
        path_disk = _as_disk_path(str(path_yd))

        existing = get_trip_by_yd_folder_path(path_disk)
        if existing:
            skipped += 1
            if args.dry_run:
                print(f"[dry-run] Уже есть: {name!r} -> {path_disk}")
            continue
        if args.dry_run:
            print(f"[dry-run] Создано бы: {name!r} -> {path_disk}")
            created += 1
            continue
        trip_id = trip_create_from_folder(name=name, yd_folder_path=path_disk)
        print(f"Создана поездка id={trip_id}: {name!r} -> {path_disk}")
        created += 1

    print(f"Итого: создано {created}, уже существовало {skipped}")


if __name__ == "__main__":
    main()
