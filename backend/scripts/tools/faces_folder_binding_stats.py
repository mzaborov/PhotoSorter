#!/usr/bin/env python3
"""
Статистика привязок по файлам в папке _faces.

1) По прямоугольникам (photo_rectangles): сколько прямоугольников, сколько с привязкой к персоне
   (manual_person_id или cluster_id → person).
2) По file_persons: сколько файлов с 0 / 1 / 2+ привязками файл→персона.
3) По персонам (итог): объединение прямоугольников и file_persons — 0 / 1 / 2+ персон.

Примеры:
  python backend/scripts/tools/faces_folder_binding_stats.py
  python backend/scripts/tools/faces_folder_binding_stats.py --videos-only
  python backend/scripts/tools/faces_folder_binding_stats.py --pipeline-run-id 26
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.common.db import get_connection
from backend.common.sort_rules import get_all_person_names_for_file

# Расширения видео (как в local_sort)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".3gp")


def _path_in_faces(path: str) -> bool:
    """Путь лежит в папке _faces и не в _no_faces."""
    p = (path or "").replace("\\", "/")
    if "_no_faces" in p:
        return False
    return "/_faces/" in p or p.rstrip("/").endswith("/_faces") or "\\_faces\\" in p or p.rstrip("\\").endswith("\\_faces")


def _is_video(path: str) -> bool:
    ext = (Path(path).suffix or "").lower()
    return ext in VIDEO_EXTENSIONS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Статистика привязок по файлам в папке _faces (0 / 1 / 2+ персон)."
    )
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Учитывать только видео",
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=int,
        default=None,
        help="Ограничить файлами прогона (по faces_run_id); иначе все файлы из _faces",
    )
    args = parser.parse_args()

    conn = get_connection()
    try:
        cur = conn.cursor()

        # Файлы в _faces (path содержит _faces, не _no_faces)
        cur.execute(
            """
            SELECT id, path, name, faces_run_id
            FROM files
            WHERE path LIKE '%_faces%' AND path NOT LIKE '%_no_faces%'
              AND (status IS NULL OR status != 'deleted')
            ORDER BY path
            """
        )
        rows = cur.fetchall()
        # Доп. фильтр по пути: точно в папке _faces
        files = [dict(r) for r in rows if _path_in_faces(r["path"])]

        if args.videos_only:
            files = [f for f in files if _is_video(f["path"])]

        if args.pipeline_run_id is not None:
            cur.execute(
                "SELECT face_run_id FROM pipeline_runs WHERE id = ? LIMIT 1",
                (args.pipeline_run_id,),
            )
            run_row = cur.fetchone()
            if not run_row:
                print(f"Прогон pipeline_run_id={args.pipeline_run_id} не найден.", file=sys.stderr)
                sys.exit(1)
            face_run_id_filter = int(run_row[0])
            files = [f for f in files if f.get("faces_run_id") == face_run_id_filter]

        if not files:
            print("Нет файлов в папке _faces (с учётом фильтров).", file=sys.stderr)
            return

        # Для каждого файла нужен pipeline_run_id по face_run_id
        cur.execute(
            """
            SELECT face_run_id, MAX(id) AS pipeline_run_id
            FROM pipeline_runs
            WHERE face_run_id IS NOT NULL
            GROUP BY face_run_id
            """
        )
        face_to_pipeline = {int(r[0]): int(r[1]) for r in cur.fetchall() if r[0] is not None and r[1] is not None}

        # --- 1) Статистика по прямоугольникам (photo_rectangles) ---
        files_with_run = [(f["id"], int(f["faces_run_id"])) for f in files if f.get("faces_run_id") is not None]
        rect_total_by_file: dict[int, int] = defaultdict(int)
        rect_assigned_by_file: dict[int, int] = defaultdict(int)

        if files_with_run:
            placeholders = ",".join(["(?,?)"] * len(files_with_run))
            params = []
            for fid, rid in files_with_run:
                params.extend([fid, rid])
            cur.execute(
                """
                SELECT pr.file_id, pr.manual_person_id, fc.person_id AS cluster_person_id
                FROM photo_rectangles pr
                LEFT JOIN face_clusters fc ON fc.id = pr.cluster_id
                WHERE (pr.file_id, pr.run_id) IN ("""
                + placeholders
                + """)
                """,
                params,
            )
            for row in cur.fetchall():
                fid = row[0]
                rect_total_by_file[fid] += 1
                if row[1] is not None or row[2] is not None:
                    rect_assigned_by_file[fid] += 1

        rect_0 = sum(1 for f in files if rect_total_by_file.get(f["id"], 0) == 0)
        rect_unassigned = sum(
            1 for f in files if rect_total_by_file.get(f["id"], 0) > 0 and rect_assigned_by_file.get(f["id"], 0) == 0
        )
        rect_assigned = sum(1 for f in files if rect_assigned_by_file.get(f["id"], 0) > 0)

        # --- 2) Статистика по file_persons ---
        fp_pairs = [
            (f["id"], face_to_pipeline[int(f["faces_run_id"])])
            for f in files
            if f.get("faces_run_id") is not None and face_to_pipeline.get(int(f["faces_run_id"])) is not None
        ]
        fp_count_by_file: dict[int, int] = {}
        if fp_pairs:
            placeholders_fp = ",".join(["(?,?)"] * len(fp_pairs))
            params_fp = []
            for fid, prid in fp_pairs:
                params_fp.extend([fid, prid])
            cur.execute(
                """
                SELECT file_id, pipeline_run_id, COUNT(*) AS cnt
                FROM file_persons
                WHERE (file_id, pipeline_run_id) IN ("""
                + placeholders_fp
                + """)
                GROUP BY file_id, pipeline_run_id
                """,
                params_fp,
            )
            for row in cur.fetchall():
                fp_count_by_file[row[0]] = int(row[2])

        fp_0 = sum(1 for f in files if fp_count_by_file.get(f["id"], 0) == 0)
        fp_1 = sum(1 for f in files if fp_count_by_file.get(f["id"], 0) == 1)
        fp_many = sum(1 for f in files if fp_count_by_file.get(f["id"], 0) >= 2)

        # --- 3) Подсчёт персон по файлу (итог: прямоугольники + file_persons) ---
        bucket_0: list[str] = []
        bucket_1: dict[str, list[str]] = {}  # person_name -> [paths]
        bucket_mixed: list[str] = []

        for f in files:
            file_id = f["id"]
            path = f["path"]
            face_run_id = f.get("faces_run_id")
            if face_run_id is None:
                bucket_0.append(path)
                continue
            pipeline_run_id = face_to_pipeline.get(int(face_run_id))
            if pipeline_run_id is None:
                bucket_0.append(path)
                continue
            try:
                names = get_all_person_names_for_file(
                    conn,
                    file_id=int(file_id),
                    pipeline_run_id=pipeline_run_id,
                    face_run_id=int(face_run_id),
                )
            except Exception:
                bucket_0.append(path)
                continue
            n = len(names)
            if n == 0:
                bucket_0.append(path)
            elif n == 1:
                name = names[0]
                bucket_1.setdefault(name, []).append(path)
            else:
                bucket_mixed.append(path)

        # Вывод
        total = len(files)
        n0, n1, nm = len(bucket_0), sum(len(v) for v in bucket_1.values()), len(bucket_mixed)
        title = "Видео" if args.videos_only else "Файлы"
        print(f"Папка _faces: {title} всего = {total}")
        print()
        print("1) По прямоугольникам (photo_rectangles):")
        print(f"   файлов без прямоугольников:     {rect_0}")
        print(f"   файлов с прямоугольниками, но 0 привязаны к персоне: {rect_unassigned}")
        print(f"   файлов с ≥1 привязанным прямоугольником: {rect_assigned}")
        print()
        print("2) По file_persons (привязка файл→персона):")
        print(f"   файлов без записей file_persons: {fp_0}")
        print(f"   файлов с 1 записью:             {fp_1}")
        print(f"   файлов с 2+ записями:            {fp_many}")
        print()
        print("3) По персонам (итог: прямоугольники + file_persons):")
        print(f"   0 персон (_unassigned): {n0}")
        print(f"   1 персона:              {n1}")
        print(f"   2+ персон (_mixed):     {nm}")
        print()
        if bucket_1:
            print("По имени персоны (1 персона на файл):")
            for name in sorted(bucket_1.keys()):
                count = len(bucket_1[name])
                print(f"  {name}: {count}")
        if bucket_0 and total <= 30:
            print()
            print("Файлы без привязки (0 персон):")
            for p in bucket_0[:20]:
                print(f"  {p}")
            if len(bucket_0) > 20:
                print(f"  ... и ещё {len(bucket_0) - 20}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
