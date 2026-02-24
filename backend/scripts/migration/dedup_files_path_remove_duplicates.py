#!/usr/bin/env python3
"""
Одноразовая миграция: удаление дубликатов по files.path и создание UNIQUE индекса.

Для каждого path оставляем одну запись (MIN(id)), привязки (лица, поездки, группы и т.д.)
переносим на неё; дубли привязок не создаём. Пути .tonfotos.ini удаляются из инвентаря полностью.
После миграции создаётся индекс idx_files_path, без него upsert_file падает с ON CONFLICT.

Запуск (из корня репо):
  python backend/scripts/migration/dedup_files_path_remove_duplicates.py --dry-run  # только отчёт
  python backend/scripts/migration/dedup_files_path_remove_duplicates.py             # выполнить
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_REPO_ROOT / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(_REPO_ROOT / ".env"), override=False)
except Exception:
    pass

from backend.common.db import DB_PATH


def _migrate_refs_and_remove_duplicates(conn: sqlite3.Connection) -> int:
    """
    Переносит привязки с удаляемых file_id на оставляемый (keep_id) по каждой группе дубликатов path.
    .tonfotos.ini — удаляем из инвентаря полностью. Возвращает число удалённых строк из files.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT path, MIN(id) AS keep_id, GROUP_CONCAT(id) AS all_ids
        FROM files
        GROUP BY path
        HAVING COUNT(*) > 1
    """)
    groups = cur.fetchall()
    if not groups:
        return 0
    deleted_total = 0
    for row in groups:
        path = row["path"]
        keep_id = int(row["keep_id"])
        all_ids = [int(x) for x in str(row["all_ids"]).split(",")]
        delete_ids = [i for i in all_ids if i != keep_id]
        if not delete_ids:
            continue
        if path and str(path).strip().lower().endswith(".tonfotos.ini"):
            all_ids_list = ",".join("?" * len(all_ids))
            cur.execute(f"UPDATE trips SET cover_file_id = NULL WHERE cover_file_id IN ({all_ids_list})", all_ids)
            cur.execute(f"DELETE FROM trip_files WHERE file_id IN ({all_ids_list})", all_ids)
            cur.execute(f"DELETE FROM file_perceptual_hashes WHERE file_id IN ({all_ids_list})", all_ids)
            cur.execute(f"DELETE FROM files_manual_labels WHERE file_id IN ({all_ids_list})", all_ids)
            cur.execute(f"DELETE FROM photo_rectangles WHERE file_id IN ({all_ids_list})", all_ids)
            cur.execute(f"DELETE FROM file_persons WHERE file_id IN ({all_ids_list})", all_ids)
            cur.execute(f"DELETE FROM file_groups WHERE file_id IN ({all_ids_list})", all_ids)
            cur.execute(f"DELETE FROM file_group_persons WHERE file_id IN ({all_ids_list})", all_ids)
            cur.execute("DELETE FROM files WHERE path = ?", (path,))
            deleted_total += len(all_ids)
            continue
        del_list = ",".join("?" * len(delete_ids))
        cur.execute(f"UPDATE trips SET cover_file_id = ? WHERE cover_file_id IN ({del_list})", (keep_id, *delete_ids))
        for fid in delete_ids:
            cur.execute(
                "DELETE FROM trip_files WHERE file_id = ? AND trip_id IN (SELECT trip_id FROM trip_files WHERE file_id = ?)",
                (fid, keep_id),
            )
            cur.execute("UPDATE trip_files SET file_id = ? WHERE file_id = ?", (keep_id, fid))
        cur.execute(
            "DELETE FROM trip_files WHERE rowid NOT IN (SELECT MIN(rowid) FROM trip_files GROUP BY trip_id, file_id)"
        )
        for fid in delete_ids:
            cur.execute(
                "DELETE FROM file_perceptual_hashes WHERE file_id = ? AND algorithm IN (SELECT algorithm FROM file_perceptual_hashes WHERE file_id = ?)",
                (fid, keep_id),
            )
            cur.execute("UPDATE file_perceptual_hashes SET file_id = ? WHERE file_id = ?", (keep_id, fid))
        cur.execute(
            "DELETE FROM file_perceptual_hashes WHERE rowid NOT IN (SELECT MIN(rowid) FROM file_perceptual_hashes GROUP BY file_id, algorithm)"
        )
        for fid in delete_ids:
            cur.execute(
                "DELETE FROM files_manual_labels WHERE file_id = ? AND pipeline_run_id IN (SELECT pipeline_run_id FROM files_manual_labels WHERE file_id = ?)",
                (fid, keep_id),
            )
            cur.execute("UPDATE files_manual_labels SET file_id = ? WHERE file_id = ?", (keep_id, fid))
        cur.execute(
            "DELETE FROM files_manual_labels WHERE rowid NOT IN (SELECT MIN(rowid) FROM files_manual_labels GROUP BY pipeline_run_id, file_id)"
        )
        cur.execute("SELECT DISTINCT run_id FROM photo_rectangles WHERE file_id IN (" + del_list + ")", delete_ids)
        run_ids = [r["run_id"] for r in cur.fetchall() if r["run_id"] is not None]
        for run_id in run_ids:
            cur.execute(
                "SELECT COALESCE(MAX(face_index), -1) + 1 AS next_idx FROM photo_rectangles WHERE file_id = ? AND run_id = ?",
                (keep_id, run_id),
            )
            next_idx = int(cur.fetchone()[0])
            cur.execute(
                "SELECT id, face_index FROM photo_rectangles WHERE file_id IN (" + del_list + ") AND run_id = ? ORDER BY face_index",
                (*delete_ids, run_id),
            )
            for pr_row in cur.fetchall():
                cur.execute(
                    "UPDATE photo_rectangles SET file_id = ?, face_index = ? WHERE id = ?",
                    (keep_id, next_idx, pr_row["id"]),
                )
                next_idx += 1
        for fid in delete_ids:
            cur.execute(
                """DELETE FROM file_persons WHERE file_id = ? AND (pipeline_run_id || '.' || person_id) IN (
                    SELECT pipeline_run_id || '.' || person_id FROM file_persons WHERE file_id = ?
                )""",
                (fid, keep_id),
            )
            cur.execute("UPDATE file_persons SET file_id = ? WHERE file_id = ?", (keep_id, fid))
        cur.execute(
            "DELETE FROM file_persons WHERE rowid NOT IN (SELECT MIN(rowid) FROM file_persons GROUP BY pipeline_run_id, file_id, person_id)"
        )
        for fid in delete_ids:
            cur.execute(
                "DELETE FROM file_groups WHERE file_id = ? AND (pipeline_run_id || '.' || group_path) IN (SELECT pipeline_run_id || '.' || group_path FROM file_groups WHERE file_id = ?)",
                (fid, keep_id),
            )
            cur.execute("UPDATE file_groups SET file_id = ? WHERE file_id = ?", (keep_id, fid))
        cur.execute(
            "DELETE FROM file_groups WHERE id NOT IN (SELECT MIN(id) FROM file_groups GROUP BY pipeline_run_id, file_id, group_path)"
        )
        for fid in delete_ids:
            cur.execute(
                """DELETE FROM file_group_persons WHERE file_id = ? AND (pipeline_run_id || '.' || group_path || '.' || person_id) IN (
                    SELECT pipeline_run_id || '.' || group_path || '.' || person_id FROM file_group_persons WHERE file_id = ?
                )""",
                (fid, keep_id),
            )
            cur.execute("UPDATE file_group_persons SET file_id = ? WHERE file_id = ?", (keep_id, fid))
        cur.execute(
            "DELETE FROM file_group_persons WHERE rowid NOT IN (SELECT MIN(rowid) FROM file_group_persons GROUP BY pipeline_run_id, file_id, group_path, person_id)"
        )
    cur.execute("DELETE FROM files WHERE id NOT IN (SELECT MIN(id) FROM files GROUP BY path)")
    deleted_total += cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    return deleted_total


def main() -> int:
    ap = argparse.ArgumentParser(description="Удалить дубликаты files.path, перенести привязки, создать idx_files_path")
    ap.add_argument("--db", type=Path, default=DB_PATH, help="Путь к photosorter.db")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    args = ap.parse_args()
    if not args.db.exists():
        print(f"БД не найдена: {args.db}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_files_path'")
        if cur.fetchone():
            print("Индекс idx_files_path уже есть. Дубликатов по path быть не должно.")
            cur.execute("SELECT path, COUNT(*) AS c FROM files GROUP BY path HAVING COUNT(*) > 1")
            dups = cur.fetchall()
            if dups:
                print(f"Но найдено групп дубликатов: {len(dups)}. Запустите без --dry-run для исправления.")
            return 0
        cur.execute("SELECT path, COUNT(*) AS cnt FROM files GROUP BY path HAVING COUNT(*) > 1")
        groups = cur.fetchall()
        if not groups:
            print("Дубликатов по path нет. Создаём индекс.")
            if not args.dry_run:
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_files_path ON files(path)")
                conn.commit()
                print("Индекс idx_files_path создан.")
            return 0
        # Для .tonfotos.ini удаляются все записи по path; для остальных — все кроме одной (MIN(id))
        def _rows_to_delete(g: dict) -> int:
            cnt = int(g["cnt"])
            path = (g.get("path") or "").strip().lower()
            return cnt if path.endswith(".tonfotos.ini") else cnt - 1
        to_delete = sum(_rows_to_delete(dict(r)) for r in groups)
        print(f"Групп дубликатов по path: {len(groups)}. Строк будет удалено из files: {to_delete}")
        if args.dry_run:
            print("Запустите без --dry-run, чтобы выполнить миграцию и создать индекс.")
            return 0
        deleted = _migrate_refs_and_remove_duplicates(conn)
        cur = conn.cursor()
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_files_path ON files(path)")
        conn.commit()
        print(f"Готово. Удалено лишних строк из files: {deleted}. Индекс idx_files_path создан.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
