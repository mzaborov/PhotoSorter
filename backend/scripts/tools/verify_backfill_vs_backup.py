#!/usr/bin/env python3
"""
Верификация backfill: сравнивает текущую БД с бекапом (до backfill).

Проверяет, что manual_person_id из бекапа совпадает с person_id кластера в текущей БД
для всех лиц в ручных кластерах архива.

Использование:
  python backend/scripts/tools/verify_backfill_vs_backup.py --backup data/photosorter.db.backup
  python backend/scripts/tools/verify_backfill_vs_backup.py --backup C:/backup/photosorter_20250209.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сравнение текущей БД с бекапом для верификации backfill"
    )
    parser.add_argument(
        "--backup",
        required=True,
        help="Путь к файлу бекапа БД (до backfill)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Сколько несовпадений вывести (по умолчанию 10)",
    )
    args = parser.parse_args()

    backup_path = Path(args.backup)
    if not backup_path.is_absolute():
        backup_path = _PROJECT_ROOT / backup_path
    if not backup_path.exists():
        print(f"Ошибка: бекап не найден: {backup_path}", file=sys.stderr)
        return 1

    from backend.common.db import DB_PATH

    current_path = Path(DB_PATH)
    if not current_path.exists():
        print(f"Ошибка: текущая БД не найдена: {current_path}", file=sys.stderr)
        return 1

    conn_backup = sqlite3.connect(str(backup_path))
    conn_backup.row_factory = sqlite3.Row
    conn_current = sqlite3.connect(str(current_path))
    conn_current.row_factory = sqlite3.Row

    cur_current = conn_current.cursor()
    cur_backup = conn_backup.cursor()

    # Лица в ручных кластерах архива в текущей БД
    cur_current.execute("""
        SELECT pr.id AS rect_id, pr.cluster_id, fc.person_id AS cluster_person_id
        FROM photo_rectangles pr
        JOIN face_clusters fc ON fc.id = pr.cluster_id
        WHERE fc.archive_scope = 'archive' AND COALESCE(fc.method, '') = 'manual'
        ORDER BY pr.id
    """)
    current_rows = cur_current.fetchall()

    if not current_rows:
        print("В текущей БД нет лиц в ручных кластерах архива.")
        conn_backup.close()
        conn_current.close()
        return 0

    mismatch_count = 0
    mismatch_samples = []
    not_in_backup = 0
    compared = 0

    for row in current_rows:
        rect_id = row["rect_id"]
        cluster_person_id = row["cluster_person_id"]

        cur_backup.execute(
            "SELECT manual_person_id FROM photo_rectangles WHERE id = ?",
            (rect_id,),
        )
        backup_row = cur_backup.fetchone()
        if backup_row is None:
            not_in_backup += 1
            continue

        backup_manual_id = backup_row["manual_person_id"]
        if backup_manual_id is None:
            continue

        compared += 1
        if int(backup_manual_id) != int(cluster_person_id):
            mismatch_count += 1
            if len(mismatch_samples) < args.limit:
                cur_backup.execute("SELECT name FROM persons WHERE id = ?", (int(backup_manual_id),))
                backup_name = cur_backup.fetchone()
                backup_name = backup_name["name"] if backup_name else "?"
                cur_current.execute("SELECT name FROM persons WHERE id = ?", (int(cluster_person_id),))
                current_name = cur_current.fetchone()
                current_name = current_name["name"] if current_name else "?"
                mismatch_samples.append({
                    "rect_id": rect_id,
                    "backup_manual_id": int(backup_manual_id),
                    "backup_name": backup_name,
                    "cluster_person_id": int(cluster_person_id),
                    "cluster_name": current_name,
                })

    conn_backup.close()
    conn_current.close()

    total = len(current_rows)
    print(f"Лиц в ручных кластерах (текущая БД): {total}")
    print(f"Сравнено (есть manual_person_id в бекапе): {compared}")
    if not_in_backup > 0:
        print(f"  (не найдено в бекапе: {not_in_backup})")
    print(f"\nНесовпадений (manual_person_id ≠ person_id кластера): {mismatch_count}")

    if mismatch_count > 0:
        print(f"\nПримеры (первые {len(mismatch_samples)}):")
        for s in mismatch_samples:
            print(f"  rect_id={s['rect_id']}: бекап {s['backup_name']} (id={s['backup_manual_id']}) "
                  f"→ текущий {s['cluster_name']} (id={s['cluster_person_id']})")
        return 1
    else:
        print("\nOK: все проверенные привязки совпадают.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
