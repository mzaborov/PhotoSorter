#!/usr/bin/env python3
"""
Удаление ошибочной привязки 205158+Ася и опциональное восстановление 205158 в кластер Санька.

Использование:
  python backend/scripts/tools/fix_rect_205158_duplicate.py --delete-erroneous
  python backend/scripts/tools/fix_rect_205158_duplicate.py --backup data/backups/photosorter_backup_YYYYMMDD_HHMMSS.db --restore-cluster
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import DB_PATH, get_connection


RECT_ID = 205158
ASYA_NAME = "Ася"


def get_person_id_by_name(conn: sqlite3.Connection, name: str) -> int | None:
    cur = conn.cursor()
    cur.execute("SELECT id FROM persons WHERE name = ? LIMIT 1", (name,))
    row = cur.fetchone()
    return row[0] if row else None


def run_delete_erroneous(conn: sqlite3.Connection, dry_run: bool) -> None:
    cur = conn.cursor()
    asya_id = get_person_id_by_name(conn, ASYA_NAME)
    if asya_id is None:
        print(f"  [SKIP] Персона '{ASYA_NAME}' не найдена в persons")
        return

    cur.execute(
        """
        SELECT fpma.id, fpma.rectangle_id, fpma.person_id, p.name
        FROM person_rectangle_manual_assignments fpma
        JOIN persons p ON p.id = fpma.person_id
        WHERE fpma.rectangle_id = ? AND fpma.person_id = ?
        """,
        (RECT_ID, asya_id),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"  [OK] Записей (rect={RECT_ID}, {ASYA_NAME}) не найдено, нечего удалять")
        return

    print(f"  Найдено записей (rect={RECT_ID}, {ASYA_NAME}): {len(rows)}")
    for r in rows:
        print(f"    id={r[0]} rectangle_id={r[1]} person_id={r[2]} name={r[3]}")

    if dry_run:
        print("  [DRY-RUN] Удаление не выполнено")
        return

    cur.execute(
        """
        DELETE FROM person_rectangle_manual_assignments
        WHERE rectangle_id = ? AND person_id = ?
        """,
        (RECT_ID, asya_id),
    )
    deleted = cur.rowcount
    conn.commit()
    print(f"  [OK] Удалено записей: {deleted}")


def _rect_column_in_fcm(cur: sqlite3.Cursor) -> str:
    cur.execute("PRAGMA table_info(face_cluster_members)")
    cols = [r[1] for r in cur.fetchall()]
    if "rectangle_id" in cols:
        return "rectangle_id"
    if "face_rectangle_id" in cols:
        return "face_rectangle_id"
    raise RuntimeError("face_cluster_members: ни rectangle_id, ни face_rectangle_id не найдено")


def find_cluster_in_backup(backup_path: Path) -> dict | None:
    conn = sqlite3.connect(str(backup_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rect_col = _rect_column_in_fcm(cur)
    cur.execute(
        f"""
        SELECT fcm.cluster_id, fc.person_id, fc.run_id, p.name AS person_name
        FROM face_cluster_members fcm
        JOIN face_clusters fc ON fc.id = fcm.cluster_id
        LEFT JOIN persons p ON p.id = fc.person_id
        WHERE fcm.{rect_col} = ?
        """,
        (RECT_ID,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def run_restore_cluster(conn: sqlite3.Connection, backup_path: Path, dry_run: bool) -> None:
    info = find_cluster_in_backup(backup_path)
    if not info:
        print(f"  [WARN] В бекапе {backup_path} для rectangle_id={RECT_ID} запись в face_cluster_members не найдена")
        return

    cluster_id = info["cluster_id"]
    person_name = info.get("person_name") or "?"
    run_id = info.get("run_id")
    print(f"  В бекапе: rect={RECT_ID} -> cluster_id={cluster_id} (person: {person_name}, run_id={run_id})")

    cur = conn.cursor()
    cur.execute("SELECT id, run_id, person_id FROM face_clusters WHERE id = ?", (cluster_id,))
    exists = cur.fetchone()
    if not exists:
        print(f"  [WARN] Кластер {cluster_id} в текущей БД не найден, восстановление в кластер невозможно")
        return

    if dry_run:
        print("  [DRY-RUN] Восстановление в кластер не выполнено")
        return

    cur.execute(
        "DELETE FROM person_rectangle_manual_assignments WHERE rectangle_id = ?",
        (RECT_ID,),
    )
    cur.execute(
        "DELETE FROM face_cluster_members WHERE rectangle_id = ?",
        (RECT_ID,),
    )
    cur.execute(
        "INSERT OR IGNORE INTO face_cluster_members (cluster_id, rectangle_id) VALUES (?, ?)",
        (cluster_id, RECT_ID),
    )
    conn.commit()
    print(f"  [OK] Удалены ручные привязки для rect={RECT_ID}, добавлен в кластер {cluster_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Исправление дубликата привязки 205158 (Ася/Санёк)")
    ap.add_argument("--delete-erroneous", action="store_true", help="Удалить ошибочную запись 205158+Ася")
    ap.add_argument("--backup", type=Path, help="Путь к бекапу БД для поиска кластера")
    ap.add_argument("--restore-cluster", action="store_true", help="Вернуть 205158 в кластер из бекапа (требуется --backup)")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    args = ap.parse_args()

    if not args.delete_erroneous and not args.restore_cluster:
        ap.error("Укажите --delete-erroneous и/или --restore-cluster")

    if args.restore_cluster and not args.backup:
        ap.error("Для --restore-cluster нужен --backup")

    conn = get_connection()
    try:
        if args.delete_erroneous:
            print("1. Удаление ошибочной записи (205158, Ася)...")
            run_delete_erroneous(conn, args.dry_run)

        if args.restore_cluster and args.backup:
            if not args.backup.exists():
                print(f"[ERROR] Бекап не найден: {args.backup}")
                return 1
            print("2. Восстановление 205158 в кластер из бекапа...")
            run_restore_cluster(conn, args.backup, args.dry_run)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
