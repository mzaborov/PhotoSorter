#!/usr/bin/env python3
"""
Сбрасывает ignore_flag в основной БД: все photo_rectangles с ignore_flag=1 получают 0.
Убирает последствия экспериментов (и любые ручные «Не лицо» — не различаем источник).

  python backend/scripts/tools/reset_ignore_flag_in_main_db.py --dry-run   # только показать количество
  python backend/scripts/tools/reset_ignore_flag_in_main_db.py             # выполнить
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.common.db import get_connection


def main() -> int:
    parser = argparse.ArgumentParser(description="Сброс ignore_flag в основной БД (убрать эксперименты)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, сколько записей будет сброшено")
    args = parser.parse_args()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM photo_rectangles WHERE COALESCE(ignore_flag, 0) = 1")
    n = cur.fetchone()[0]
    conn.close()

    if n == 0:
        print("Нет записей с ignore_flag=1. Ничего делать не нужно.")
        return 0

    if args.dry_run:
        print(f"Будет сброшено ignore_flag → 0 для {n} записей в основной БД. Запустите без --dry-run для применения.")
        return 0

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE photo_rectangles SET ignore_flag = 0 WHERE COALESCE(ignore_flag, 0) = 1")
    conn.commit()
    updated = cur.rowcount
    conn.close()
    print(f"В основной БД сброшено ignore_flag для {updated} записей.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
