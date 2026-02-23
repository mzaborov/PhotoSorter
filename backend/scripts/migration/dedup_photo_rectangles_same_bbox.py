#!/usr/bin/env python3
"""
Одноразовая миграция: удаление дубликатов в photo_rectangles (одинаковые file_id, frame_idx, bbox).
Удаляем: (1) записи без привязок; (2) дубликаты с той же персоной, что и оставляемая; (3) записи с привязкой на «Посторонний».
Перед удалением переносим persons.avatar_face_id на оставляемый id.

Запуск (из корня репо):
  python backend/scripts/migration/dedup_photo_rectangles_same_bbox.py --dry-run  # только отчёт
  python backend/scripts/migration/dedup_photo_rectangles_same_bbox.py             # выполнить
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_REPO_ROOT / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(_REPO_ROOT / ".env"), override=False)
except Exception:
    pass

from backend.common.db import DB_PATH, get_outsider_person_id


def _group_key(row: dict) -> tuple:
    fidx = row["frame_idx"] if row["frame_idx"] is not None else -1
    return (
        int(row["file_id"]),
        fidx,
        int(row["bbox_x"]),
        int(row["bbox_y"]),
        int(row["bbox_w"]),
        int(row["bbox_h"]),
    )


def _choose_keep_id(rows: list[dict]) -> int:
    """Кандидат на сохранение: приоритет manual_person_id, затем cluster_id, иначе min id."""
    with_manual = [r for r in rows if r.get("manual_person_id") is not None]
    if with_manual:
        return min(int(r["id"]) for r in with_manual)
    with_cluster = [r for r in rows if r.get("cluster_id") is not None]
    if with_cluster:
        return min(int(r["id"]) for r in with_cluster)
    return min(int(r["id"]) for r in rows)


def _delete_ids_for_group(
    rows: list[dict], keep_id: int, outsider_person_id: int | None
) -> list[int]:
    """
    Удаляем: (а) без привязок; (б) с той же персоной, что и кандидат; (в) с персоной Посторонний.
    Не удаляем записи с другой (не Посторонний) персоной.
    """
    keep_row = next((r for r in rows if r["id"] == keep_id), None)
    keep_person = (
        int(keep_row["effective_person_id"])
        if keep_row and keep_row.get("effective_person_id") is not None
        else None
    )
    delete_ids = []
    for r in rows:
        if r["id"] == keep_id:
            continue
        eff = r.get("effective_person_id")
        eff_id = int(eff) if eff is not None else None
        if eff_id is None:
            delete_ids.append(int(r["id"]))
        elif outsider_person_id is not None and eff_id == outsider_person_id:
            delete_ids.append(int(r["id"]))
        elif keep_person is not None and eff_id == keep_person:
            delete_ids.append(int(r["id"]))
    return delete_ids


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def run(conn: sqlite3.Connection, dry_run: bool) -> tuple[int, int, int]:
    """
    Возвращает (groups_processed, avatar_updates, rows_deleted).
    """
    cur = conn.cursor()

    cur.execute("""
        WITH dup_groups AS (
            SELECT file_id, COALESCE(frame_idx, -1) AS fidx, bbox_x, bbox_y, bbox_w, bbox_h
            FROM photo_rectangles
            GROUP BY file_id, COALESCE(frame_idx, -1), bbox_x, bbox_y, bbox_w, bbox_h
            HAVING COUNT(*) > 1
        )
        SELECT pr.id, pr.file_id, pr.frame_idx, pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h,
               pr.manual_person_id, pr.cluster_id,
               COALESCE(pr.manual_person_id, fc.person_id) AS effective_person_id
        FROM photo_rectangles pr
        LEFT JOIN face_clusters fc ON fc.id = pr.cluster_id
        JOIN dup_groups d
          ON d.file_id = pr.file_id AND COALESCE(pr.frame_idx, -1) = d.fidx
         AND d.bbox_x = pr.bbox_x AND d.bbox_y = pr.bbox_y
         AND d.bbox_w = pr.bbox_w AND d.bbox_h = pr.bbox_h
        ORDER BY pr.file_id, COALESCE(pr.frame_idx, -1), pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h, pr.id
    """)
    all_rows = [dict(r) for r in cur.fetchall()]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        groups[_group_key(r)].append(r)

    outsider_person_id = get_outsider_person_id(conn, create_if_missing=False)
    if outsider_person_id is not None:
        outsider_person_id = int(outsider_person_id)

    has_face_labels = _table_exists(cur, "face_labels")
    has_prma = _table_exists(cur, "person_rectangle_manual_assignments")
    has_fcm = _table_exists(cur, "face_cluster_members")

    total_avatar_updates = 0
    total_deleted = 0
    groups_processed = 0

    for key, rows in groups.items():
        keep_id = _choose_keep_id(rows)
        delete_ids = _delete_ids_for_group(rows, keep_id, outsider_person_id)
        if not delete_ids:
            continue
        groups_processed += 1
        if dry_run and delete_ids:
            placeholders = ",".join("?" * len(delete_ids))
            cur.execute(
                f"SELECT COUNT(*) FROM persons WHERE avatar_face_id IN ({placeholders})",
                delete_ids,
            )
            total_avatar_updates += cur.fetchone()[0] or 0
        for rid in delete_ids:
            if not dry_run:
                cur.execute(
                    "UPDATE persons SET avatar_face_id = ? WHERE avatar_face_id = ?",
                    (keep_id, rid),
                )
                if cur.rowcount and cur.rowcount > 0:
                    total_avatar_updates += 1
                if has_face_labels:
                    cur.execute(
                        "UPDATE face_labels SET face_rectangle_id = ? WHERE face_rectangle_id = ?",
                        (keep_id, rid),
                    )
                if has_prma:
                    cur.execute(
                        "UPDATE person_rectangle_manual_assignments SET rectangle_id = ? WHERE rectangle_id = ?",
                        (keep_id, rid),
                    )
                if has_fcm:
                    cur.execute(
                        "UPDATE face_cluster_members SET rectangle_id = ? WHERE rectangle_id = ?",
                        (keep_id, rid),
                    )
            total_deleted += 1

        if not dry_run and delete_ids:
            placeholders = ",".join("?" * len(delete_ids))
            cur.execute(
                f"DELETE FROM photo_rectangles WHERE id IN ({placeholders})",
                delete_ids,
            )

    return groups_processed, total_avatar_updates, total_deleted


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Удалить дубликаты photo_rectangles (одинаковый bbox). Записи с привязками не трогаем."
    )
    ap.add_argument("--db", type=Path, default=DB_PATH, help="Путь к photosorter.db")
    ap.add_argument("--dry-run", action="store_true", help="Только отчёт, без изменений")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"БД не найдена: {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        groups_processed, avatar_updates, rows_deleted = run(conn, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[DRY-RUN] Групп к обработке: {groups_processed}")
            print(f"[DRY-RUN] Записей к удалению (без привязок, та же персона или Посторонний): {rows_deleted}")
            print(f"[DRY-RUN] Обновлений persons.avatar_face_id: {avatar_updates}")
            print("Запустите без --dry-run для выполнения.")
            return 0
        conn.commit()
        print(f"Групп обработано: {groups_processed}")
        print(f"Записей удалено: {rows_deleted}")
        print(f"Обновлений persons.avatar_face_id: {avatar_updates}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
