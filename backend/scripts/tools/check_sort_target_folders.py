"""
Проверка сортировки по правилам: для файлов текущего отчёта шага 4 выводит таблицу
file_name, old_target_folder, new_target_folder (без изменений в БД).

Пример:
  python backend/scripts/tools/check_sort_target_folders.py --pipeline-run-id 26
  python backend/scripts/tools/check_sort_target_folders.py --latest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Чтобы common.sort_rules импортировал common.db
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.common.db import DedupStore, PipelineStore, get_connection, list_folders
from backend.common.sort_rules import (
    determine_target_folder,
    resolve_target_folder_for_faces,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Таблица file_name, old_target_folder, new_target_folder по правилам сортировки (dry-run)."
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=int,
        default=None,
        help="ID прогона pipeline (если не указан, используется --latest)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Взять последний прогон по id",
    )
    args = parser.parse_args()

    ps = PipelineStore()
    try:
        if args.latest or args.pipeline_run_id is None:
            cur = ps.conn.cursor()
            cur.execute("SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                print("Нет прогонов в БД.", file=sys.stderr)
                sys.exit(1)
            pipeline_run_id = int(row[0])
            print(f"Используется последний прогон: pipeline_run_id={pipeline_run_id}", file=sys.stderr)
        else:
            pipeline_run_id = args.pipeline_run_id

        pr = ps.get_run_by_id(run_id=pipeline_run_id)
    finally:
        ps.close()

    if not pr:
        print(f"Прогон pipeline_run_id={pipeline_run_id} не найден.", file=sys.stderr)
        sys.exit(1)

    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        print("У прогона не задан face_run_id (шаг 3 не запускался).", file=sys.stderr)
        sys.exit(1)
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "").strip()
    if not root_path:
        print("У прогона не задан root_path.", file=sys.stderr)
        sys.exit(1)

    # preclean_map: path -> kind
    ps = PipelineStore()
    try:
        cur = ps.conn.cursor()
        cur.execute(
            "SELECT src_path, kind FROM preclean_moves WHERE pipeline_run_id = ?",
            (pipeline_run_id,),
        )
        preclean_map = {str(r[0] or ""): str(r[1] or "") for r in cur.fetchall() if r[0] and r[1]}
    finally:
        ps.close()

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        cur.execute(
            """
            SELECT
              f.id, f.path, f.name, f.target_folder,
              COALESCE(m.faces_manual_label, '') AS faces_manual_label,
              COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
              COALESCE(f.faces_auto_quarantine, 0) AS faces_auto_quarantine,
              COALESCE(f.faces_count, 0) AS faces_count,
              COALESCE(m.animals_manual, 0) AS animals_manual,
              COALESCE(f.animals_auto, 0) AS animals_auto,
              COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual
            FROM files f
            LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE f.faces_run_id = ? AND f.status != 'deleted'
              AND f.target_folder IS NOT NULL AND trim(f.target_folder) != ''
            ORDER BY f.target_folder, f.path
            """,
            (pipeline_run_id, face_run_id_i),
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        ds.close()

    target_folders = list_folders(role="target")
    conn = get_connection()

    # Таблица: file_name, old_target_folder, new_target_folder
    table_rows: list[tuple[str, str, str]] = []

    try:
        for r in rows:
            path = str(r.get("path") or "")
            if not path:
                continue
            file_id = r.get("id")
            if file_id is None:
                continue
            file_name = str(r.get("name") or "").strip() or Path(path).name
            old_target = (r.get("target_folder") or "").strip()
            if not old_target:
                continue  # только файлы с заполненным target_folder

            preclean_kind = preclean_map.get(path)
            effective_tab = "no_faces"
            if preclean_kind:
                effective_tab = None
            elif r.get("people_no_face_manual"):
                effective_tab = "people_no_face"
            elif (r.get("faces_manual_label") or "").lower().strip() == "faces":
                effective_tab = "faces"
            elif (r.get("faces_manual_label") or "").lower().strip() == "no_faces":
                effective_tab = "no_faces"
            elif r.get("quarantine_manual") and (r.get("faces_count") or 0) > 0:
                effective_tab = "quarantine"
            elif r.get("animals_manual") or r.get("animals_auto"):
                effective_tab = "animals"
            elif (r.get("faces_auto_quarantine") or 0) and (r.get("faces_count") or 0) > 0:
                effective_tab = "quarantine"
            elif (r.get("faces_count") or 0) > 0:
                effective_tab = "faces"

            person_name = None
            if effective_tab == "faces":
                try:
                    person_name = resolve_target_folder_for_faces(
                        conn,
                        file_id=int(file_id),
                        pipeline_run_id=pipeline_run_id,
                        face_run_id=face_run_id_i,
                        target_folders=target_folders,
                    )
                except Exception as e:
                    new_target = f"(ошибка: {e!r})"
                    table_rows.append((file_name, old_target, new_target))
                    continue

            new_target_raw = determine_target_folder(
                path=path,
                effective_tab=effective_tab or "no_faces",
                root_path=root_path,
                preclean_kind=preclean_kind,
                person_name=person_name,
                target_folders=target_folders,
            )
            new_target = (new_target_raw or "").strip() or "—"
            table_rows.append((file_name, old_target, new_target))
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Сортировка по old_target_folder, затем по file_name
    table_rows.sort(key=lambda r: (r[1], r[0]))

    # Вывод таблицы с выравниванием колонок
    if not table_rows:
        print("Нет файлов с заполненным target_folder в прогоне.", file=sys.stderr)
        return

    col0 = max(3, len(str(len(table_rows))))
    col1 = max(len("file_name"), max(len(r[0]) for r in table_rows), 20)
    col2 = max(len("old_target_folder"), max(len(r[1]) for r in table_rows), 24)
    col3 = max(len("new_target_folder"), max(len(r[2]) for r in table_rows), 24)
    fmt = f"{{0:>{col0}}}  {{1:<{col1}}}  {{2:<{col2}}}  {{3:<{col3}}}"
    print(fmt.format("#", "file_name", "old_target_folder", "new_target_folder"))
    print("-" * (col0 + col1 + col2 + col3 + 6))
    for i, (file_name, old_target, new_target) in enumerate(table_rows, start=1):
        print(fmt.format(i, file_name, old_target, new_target))


if __name__ == "__main__":
    main()
