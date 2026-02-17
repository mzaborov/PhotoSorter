#!/usr/bin/env python3
"""
Бэкфилл start_date/end_date для поездок без дат.

Вычисляет даты по taken_at входящих в поездку файлов и записывает в trips.
Запуск:
    python backend/scripts/tools/backfill_trip_dates.py
    python backend/scripts/tools/backfill_trip_dates.py --dry-run
    python backend/scripts/tools/backfill_trip_dates.py --all  # перезаписать даты и для поездок с уже заданными
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.common.db import list_trips, trip_dates_from_files, trip_update


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Заполнить start_date/end_date поездок по датам съёмки файлов"
    )
    parser.add_argument("--dry-run", action="store_true", help="Не писать в БД, только вывести")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Обновить даты и для поездок, у которых уже заданы start_date/end_date",
    )
    args = parser.parse_args()

    trips = list_trips()
    updated = 0
    skipped = 0
    no_dates = 0

    for t in trips:
        tid = int(t["id"])
        name = (t.get("name") or "").strip() or "?"
        start = (t.get("start_date") or "").strip()
        end = (t.get("end_date") or "").strip()

        if start and not args.all:
            skipped += 1
            continue

        ds, de = trip_dates_from_files(tid) or (None, None)
        if not ds:
            no_dates += 1
            if args.dry_run:
                print(f"[dry-run] {name} (id={tid}): нет дат в файлах — пропуск")
            continue

        de = de or ds

        if args.dry_run:
            print(f"[dry-run] {name} (id={tid}): было {start or '-'} - {end or '-'} -> {ds} - {de}")
            updated += 1
            continue

        if trip_update(tid, start_date=ds, end_date=de):
            print(f"Обновлено: {name} (id={tid}) -> {ds} - {de}")
            updated += 1

    print(f"\nОбновлено: {updated}")
    print(f"Пропущено (даты уже заданы): {skipped}")
    print(f"Пропущено (нет дат в файлах): {no_dates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
