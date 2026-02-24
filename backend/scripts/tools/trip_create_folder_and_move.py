#!/usr/bin/env python3
"""
Создать папку поездки на Я.Диске, перенести в неё файлы из папок НЕ связанных с людьми (не target),
обновить пути в БД.

«Папки людей» = папки из folders с role='target'. Переносятся только файлы, чей path
не лежит под ни одной такой папкой.

Запуск: python backend/scripts/tools/trip_create_folder_and_move.py --trip-id 91 [--dry-run]
Переменная окружения: TRIPS_YD_ROOT (по умолчанию disk:/Фото/Путешествия).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from common.db import (
    DedupStore,
    get_connection,
    init_db,
    list_folders,
    list_trip_files_included,
    get_trip,
    trip_update,
)
from common.yadisk_client import get_disk

# yadisk exceptions for PathExistsError
try:
    import yadisk.exceptions as yadisk_exceptions
except Exception:
    yadisk_exceptions = None


def _normalize_yadisk_path(path: str) -> str:
    """Для API ЯД: без disk:, с ведущим /."""
    p = (path or "").strip().replace("\\", "/")
    if p.lower().startswith("disk:"):
        p = p[5:].lstrip("/")
    if not p.startswith("/"):
        p = "/" + p
    return p


def _ensure_yadisk_folder(disk, folder_disk_path: str) -> None:
    """Создать папку на ЯД и цепочку родительских. folder_disk_path: disk:/Фото/..."""
    norm = _normalize_yadisk_path(folder_disk_path)
    parts = [x for x in norm.split("/") if x]
    for i in range(1, len(parts) + 1):
        sub = "/" + "/".join(parts[:i])
        try:
            disk.mkdir(sub)
        except Exception as e:
            if yadisk_exceptions and isinstance(e, yadisk_exceptions.PathExistsError):
                pass
            elif "PathExistsError" in type(e).__name__ or "already exists" in str(e).lower():
                pass
            else:
                raise


def _path_under_any(file_path: str, target_prefixes: list[str]) -> bool:
    """True если file_path лежит под одной из target папок (path или path/)."""
    p = (file_path or "").strip().replace("\\", "/").rstrip("/")
    for t in target_prefixes:
        t = (t or "").strip().replace("\\", "/").rstrip("/")
        if not t:
            continue
        if p == t or p.startswith(t + "/"):
            return True
    return False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Создать папку поездки на ЯД и перенести файлы не из папок людей"
    )
    parser.add_argument("--trip-id", type=int, required=True, help="ID поездки (например 91)")
    parser.add_argument(
        "--root",
        default=os.getenv("TRIPS_YD_ROOT", "disk:/Фото/Путешествия"),
        help="Корневая папка поездок на ЯД",
    )
    parser.add_argument("--dry-run", action="store_true", help="Не создавать папку, не переносить, не менять БД")
    args = parser.parse_args()

    init_db()
    trip_id = args.trip_id
    trip = get_trip(trip_id)
    if not trip:
        print(f"Поездка {trip_id} не найдена.", file=sys.stderr)
        sys.exit(1)

    name = (trip.get("name") or "").strip() or f"Поездка {trip_id}"
    yd_existing = (trip.get("yd_folder_path") or "").strip().rstrip("/")
    if yd_existing:
        print(f"У поездки {trip_id} уже задана папка: {yd_existing}. Скрипт не меняет существующую папку.", file=sys.stderr)
        sys.exit(1)

    root = (args.root or "").strip().rstrip("/") or "disk:/Фото/Путешествия"
    yd_folder_path = root + "/" + name

    # «Папки людей» = target-папки, кроме Путешествия и Технологии (файлы оттуда переносим в папку поездки)
    folders = list_folders(role="target")
    exclude_codes = ("puteshestviya", "tehnologii")
    target_prefixes = []
    for f in folders:
        path = (f.get("path") or "").strip().rstrip("/")
        code = (f.get("code") or "").strip().lower()
        if not path or any(ex in code for ex in exclude_codes):
            continue
        target_prefixes.append(path)

    files = list_trip_files_included(trip_id)
    # Только disk: и не из папок людей
    to_move = []
    for f in files:
        path = (f.get("path") or "").strip()
        if not path.startswith("disk:"):
            continue
        if _path_under_any(path, target_prefixes):
            continue
        to_move.append(f)

    skipped = len(files) - len(to_move)
    print(f"Поездка {trip_id}: {name!r}")
    print(f"Папка поездки: {yd_folder_path}")
    print(f"Target-папок (люди): {len(target_prefixes)}")
    print(f"Файлов в поездке: {len(files)}, к переносу (не из папок людей): {len(to_move)}, пропуск: {skipped}")

    if args.dry_run:
        for i, f in enumerate(to_move[:30]):
            print(f"  [dry-run] перенос: {f.get('path')} -> {yd_folder_path}/{f.get('name') or '?'}")
        if len(to_move) > 30:
            print(f"  ... и ещё {len(to_move) - 30} файлов")
        print("[dry-run] Завершено. Запустите без --dry-run для выполнения.")
        return

    disk = get_disk()
    _ensure_yadisk_folder(disk, yd_folder_path)
    trip_update(trip_id, yd_folder_path=yd_folder_path)
    print(f"Папка создана и записана в trips.yd_folder_path.")

    moved = 0
    errors = []
    ds = DedupStore()
    try:
        for f in to_move:
            path = (f.get("path") or "").strip()
            name_f = (f.get("name") or "").strip() or Path(path).name or "file"
            new_path = yd_folder_path.rstrip("/") + "/" + name_f
            src_norm = _normalize_yadisk_path(path)
            dst_norm = _normalize_yadisk_path(new_path)
            try:
                try:
                    disk.move(src_norm, dst_norm, overwrite=False)
                except TypeError:
                    disk.move(src_norm, dst_norm)
            except Exception as e:
                errors.append((path, str(e)))
                continue
            ds.update_path(
                old_path=path,
                new_path=new_path,
                new_name=name_f,
                new_parent_path=yd_folder_path.rstrip("/"),
            )
            moved += 1
    finally:
        ds.close()

    if errors:
        print(f"Ошибки при переносе ({len(errors)}):", file=sys.stderr)
        for p, err in errors[:10]:
            print(f"  {p}: {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... и ещё {len(errors) - 10}", file=sys.stderr)
    print(f"Перенесено файлов: {moved}.")


if __name__ == "__main__":
    main()
