#!/usr/bin/env python3
"""
Переименовать папку поездки на Я.Диске так, чтобы она совпадала с текущим именем поездки в БД (trips.name).
Обновляет trips.yd_folder_path и пути в files для всех файлов в этой папке.

Запуск: python backend/scripts/tools/trip_rename_folder_to_match_name.py --trip-id 91 [--dry-run]
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from common.db import get_connection, init_db, get_trip, trip_update, DedupStore
from common.yadisk_client import get_disk

# Подстановка имени для известных поездок (обход кодировки консоли Windows)
PREFERRED_TRIP_NAMES = {91: "2019 Италия 2"}
# Если папка на ЯД была создана с неправильным именем — указать фактическое имя папки на диске
PREFERRED_OLD_FOLDER_NAMES = {91: "2019 отпуск 2"}


def _normalize_yadisk_path(path: str) -> str:
    p = (path or "").strip().replace("\\", "/")
    if p.lower().startswith("disk:"):
        p = p[5:].lstrip("/")
    if not p.startswith("/"):
        p = "/" + p
    return p


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Переименовать папку поездки на ЯД в соответствии с trips.name")
    parser.add_argument("--trip-id", type=int, required=True)
    parser.add_argument(
        "--name",
        default=None,
        help="Новое имя поездки (и папки). Если не задано, используется текущее trips.name.",
    )
    parser.add_argument(
        "--old-folder-name",
        default=None,
        help="Фактическое имя папки на ЯД (если переименовали ошибочно). Для поездки 91 можно не указывать.",
    )
    parser.add_argument(
        "--root",
        default=os.getenv("TRIPS_YD_ROOT", "disk:/Фото/Путешествия"),
        help="Корневая папка поездок на ЯД",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    init_db()
    trip = get_trip(args.trip_id)
    if not trip:
        print(f"Поездка {args.trip_id} не найдена.", file=sys.stderr)
        sys.exit(1)

    # Имя: --name, env TRIP_RENAME_NEW_NAME, PREFERRED_TRIP_NAMES, или текущее в БД
    name = (
        (args.name or os.getenv("TRIP_RENAME_NEW_NAME") or PREFERRED_TRIP_NAMES.get(args.trip_id) or trip.get("name") or "").strip()
        or f"Поездка {args.trip_id}"
    )
    if name != (trip.get("name") or "").strip() and not args.dry_run:
        trip_update(args.trip_id, name=name)
    root = (args.root or "").strip().rstrip("/") or "disk:/Фото/Путешествия"
    new_yd = root + "/" + name
    # Фактическая папка на ЯД: из --old-folder-name, PREFERRED_OLD_FOLDER_NAMES или из БД
    old_folder_name = (
        args.old_folder_name
        or os.getenv("TRIP_RENAME_OLD_FOLDER_NAME")
        or PREFERRED_OLD_FOLDER_NAMES.get(args.trip_id)
    )
    if old_folder_name:
        old_yd = (root + "/" + old_folder_name.strip()).rstrip("/")
    else:
        old_yd = (trip.get("yd_folder_path") or "").strip().rstrip("/")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"  old_yd={old_yd!r}, new_yd={new_yd!r}")
    if not old_yd:
        print("ERROR: trip has no yd_folder_path and --old-folder-name not set", file=sys.stderr)
        sys.exit(1)
    if old_yd == new_yd:
        print("OK: folder name already matches, nothing to do.")
        return

    print(f"Trip {args.trip_id}: name={name!r}")

    if args.dry_run:
        print("[dry-run] Выполните без --dry-run для переименования.")
        return

    disk = get_disk()
    src_norm = _normalize_yadisk_path(old_yd)
    dst_norm = _normalize_yadisk_path(new_yd)
    moved_on_disk = False
    try:
        disk.move(src_norm, dst_norm)
        moved_on_disk = True
    except Exception as e:
        err_str = str(e).lower()
        if "not found" in err_str or "404" in err_str or "disknotfound" in err_str:
            print("Папка на ЯД не найдена (уже переименована или другой путь). Обновляю только БД.", file=sys.stderr)
        else:
            print(f"Ошибка YaDisk move: {e}", file=sys.stderr)
            sys.exit(1)

    trip_update(args.trip_id, yd_folder_path=new_yd)

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, path, name, parent_path FROM files WHERE path LIKE ?",
            (old_yd + "/%",),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    ds = DedupStore()
    try:
        for row in rows:
            path = row["path"]
            if path == old_yd:
                continue
            name_f = row["name"] or Path(path).name
            new_path = new_yd + "/" + (name_f or "")
            ds.update_path(
                old_path=path,
                new_path=new_path,
                new_name=name_f,
                new_parent_path=new_yd,
            )
    finally:
        ds.close()

    print(f"Готово. На ЯД переименовано: {moved_on_disk}. Обновлено путей в files: {len(rows)}.")


if __name__ == "__main__":
    main()
