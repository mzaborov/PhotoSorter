#!/usr/bin/env python3
"""
Импорт/обогащение поездок из CSV (data/trips.csv).

Рекомендуемый порядок: сначала sync_trips_from_yadisk.py (список поездок с папок ЯД),
затем этот скрипт — он обновляет у существующих поездок даты и страну по совпадению имени
(например "2013 Израиль"). Если поездки с таким именем нет — создаётся новая запись
(как раньше), если в диапазоне дат есть фото.

Формат CSV: разделитель ";", первые две строки — заголовки.
Колонки: 0 — страна, 2 — дата въезда (DD.MM.YYYY), 3 — дата выезда.
Запуск: python backend/scripts/tools/import_trips_from_csv.py [--csv путь] [--dry-run]
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

# путь к репозиторию и к БД
REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "photosorter.db"
DEFAULT_CSV = REPO_ROOT / "data" / "trips.csv"


def _now_utc_iso() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_date_str(s: str) -> str | None:
    """Нормализует строку даты DD.MM.YYYY, исправляет типичные опечатки."""
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    # 16.052023 -> 16.05.2023; 24 04.2023 -> 24.04.2023
    s = re.sub(r"\.(\d{4})$", r".\1", s)
    s = re.sub(r"(\d{1,2})\s+(\d{1,2})\.", r"\1.\2.", s)
    s = re.sub(r"(\d{1,2})\.(\d{2})(\d{4})", r"\1.\2.\3", s)
    try:
        dt = datetime.strptime(s, "%d.%m.%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_row(row: list[str]) -> tuple[str | None, str | None, str | None, str | None]:
    """Возвращает (place_country, start_date_iso, end_date_iso, name)."""
    if len(row) < 4:
        return None, None, None, None
    country = (row[0] or "").strip() or None
    entry = _normalize_date_str(row[2])
    exit_ = _normalize_date_str(row[3])
    if not entry or not country:
        return country, None, None, None
    if not exit_:
        exit_ = entry
    start, end = entry, exit_
    if start > end:
        start, end = end, start
    year = start[:4] if start else ""
    name = f"{year} {country}" if year and country else (country or "Поездка")
    return country, start, end, name


def _count_files_in_date_range(conn, start_iso: str, end_iso: str) -> int:
    """Количество файлов в files с taken_at в [start_iso, end_iso] (включительно)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) AS cnt FROM files
        WHERE taken_at IS NOT NULL AND taken_at != ''
          AND date(taken_at) >= date(?) AND date(taken_at) <= date(?)
        """,
        (start_iso, end_iso),
    )
    row = cur.fetchone()
    return row[0] if row else 0


def main() -> None:
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description="Импорт поездок из CSV (только поездки с фото)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Путь к CSV (по умолчанию data/trips.csv)")
    parser.add_argument("--dry-run", action="store_true", help="Не писать в БД, только вывести, что было бы добавлено")
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    if not csv_path.exists():
        print(f"Файл не найден: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Инициализация БД (создаст таблицы trips при необходимости)
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from common.db import init_db

    init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows_added = 0
    rows_updated = 0
    rows_skipped_no_photos = 0
    rows_skipped_bad_dates = 0

    def _find_trip_by_name(cur, name: str) -> dict | None:
        cur.execute("SELECT id, name, start_date, end_date, place_country FROM trips WHERE trim(name) = ?", (name.strip(),))
        row = cur.fetchone()
        return dict(row) if row else None

    try:
        f = open(csv_path, "r", encoding="utf-8-sig")
        f.read(1)
        f.seek(0)
    except UnicodeDecodeError:
        f = open(csv_path, "r", encoding="cp1251")
    with f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)
        next(reader, None)
        for row in reader:
            country, start, end, name = _parse_row(row)
            if not start:
                rows_skipped_bad_dates += 1
                continue
            cur = conn.cursor()
            existing = _find_trip_by_name(cur, name or "")
            if existing:
                if args.dry_run:
                    print(f"[dry-run] Обновлено бы: {name!r} -> {start}–{end}, {country}")
                    rows_updated += 1
                else:
                    now = _now_utc_iso()
                    cur.execute(
                        "UPDATE trips SET start_date=?, end_date=?, place_country=?, updated_at=? WHERE id=?",
                        (start, end, country, now, existing["id"]),
                    )
                    rows_updated += 1
                continue
            cnt = _count_files_in_date_range(conn, start, end)
            if cnt == 0:
                rows_skipped_no_photos += 1
                continue
            if args.dry_run:
                print(f"[dry-run] Добавлено бы: {name!r} {start}–{end} ({cnt} фото)")
                rows_added += 1
                continue
            now = _now_utc_iso()
            cur.execute(
                """
                INSERT INTO trips (name, start_date, end_date, place_country, place_city, yd_folder_path, cover_file_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (name, start, end, country, now, now),
            )
            rows_added += 1

    if not args.dry_run:
        conn.commit()
    conn.close()

    print(f"Добавлено поездок: {rows_added}")
    print(f"Обновлено поездок (даты/страна): {rows_updated}")
    print(f"Пропущено (нет фото в диапазоне): {rows_skipped_no_photos}")
    print(f"Пропущено (нет дат): {rows_skipped_bad_dates}")


if __name__ == "__main__":
    main()
