#!/usr/bin/env python3
"""
Считает эмбеддинги альтернативной моделью для лиц архива (список из основной БД),
сохраняет в БД экспериментов (experiments.db) в face_embeddings_alt. Основную БД не меняет.

Использование:
  python backend/scripts/tools/backfill_archive_embeddings_alt.py --model-path models/face_recognition/antelopev2/w600k_r50.onnx --model-key antelopev2
  python backend/scripts/tools/backfill_archive_embeddings_alt.py --model-path ... --model-key antelopev2 --limit 50 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
except Exception:
    pass


def _path_to_local(path: str) -> str | None:
    if not path or not isinstance(path, str):
        return None
    path = path.strip()
    if path.startswith("local:"):
        p = path[6:].strip()
        return os.path.abspath(p) if p else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill эмбеддингов альтернативной моделью для архива")
    parser.add_argument("--model-path", type=Path, required=True, help="Путь к ONNX модели (например antelopev2/w600k_r50.onnx)")
    parser.add_argument("--model-key", type=str, required=True, help="Ключ модели для face_embeddings_alt (например antelopev2)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, сколько лиц будет обработано")
    parser.add_argument("--limit", type=int, default=None, help="Максимум файлов")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    model_path = args.model_path if args.model_path.is_absolute() else (_PROJECT_ROOT / args.model_path)
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        return 1

    from backend.common.db import get_connection
    from backend.common.experiments_db import get_experiments_connection, ensure_experiments_tables
    from backend.logic.face_recognition import _load_embedding_model_from_path, _extract_embedding_from_crop

    main_conn = get_connection()
    main_conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
    main_cur = main_conn.cursor()

    main_cur.execute(
        """
        SELECT f.id AS file_id, f.path AS file_path
        FROM files f
        INNER JOIN photo_rectangles pr ON pr.file_id = f.id
        WHERE (f.inventory_scope = 'archive' OR TRIM(COALESCE(f.inventory_scope, '')) = 'archive')
          AND pr.embedding IS NOT NULL
          AND pr.is_face = 1
          AND COALESCE(pr.ignore_flag, 0) = 0
        GROUP BY f.id
        ORDER BY f.id
        """
    )
    file_rows = main_cur.fetchall()
    main_conn.close()

    exp_conn = get_experiments_connection()
    ensure_experiments_tables(exp_conn)
    exp_cur = exp_conn.cursor()

    if not file_rows:
        print("Нет файлов архива с лицами (embedding уже есть).")
        return 0

    if args.limit:
        file_rows = file_rows[: args.limit]
    total_rects = 0
    main_conn = get_connection()
    main_cur = main_conn.cursor()
    for r in file_rows:
        main_cur.execute(
            """
            SELECT pr.id, pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h, pr.frame_t_sec
            FROM photo_rectangles pr
            WHERE pr.file_id = ? AND pr.embedding IS NOT NULL AND pr.is_face = 1 AND COALESCE(pr.ignore_flag, 0) = 0
            """,
            (r["file_id"],),
        )
        total_rects += len(main_cur.fetchall())
    main_conn.close()
    print(f"Файлов: {len(file_rows)}, лиц всего: {total_rects}")
    if args.dry_run:
        return 0

    model = _load_embedding_model_from_path(model_path)
    if not model:
        print("Не удалось загрузить модель", file=sys.stderr)
        return 1
    print(f"Модель загружена: {model_path}")

    disk = None
    try:
        from backend.common.yadisk_client import get_disk
        disk = get_disk()
    except Exception as e:
        print(f"YaDisk: {e}")

    try:
        import cv2
    except ImportError:
        print("Нужен opencv-python (cv2)", file=sys.stderr)
        return 1

    done = 0
    errors = 0
    main_conn = get_connection()
    main_cur = main_conn.cursor()
    for i, row in enumerate(file_rows):
        file_id = row["file_id"]
        db_path = row["file_path"] or ""
        local_path = _path_to_local(db_path)
        tmp_path = None
        if local_path and os.path.isfile(local_path):
            pass
        elif db_path.startswith("disk:") and disk:
            try:
                remote = db_path.replace("disk:/", "/")
                suf = Path(db_path).suffix or ".jpg"
                fd, tmp_path = tempfile.mkstemp(suffix=suf)
                os.close(fd)
                disk.download(remote, tmp_path)
                local_path = tmp_path
            except Exception as e:
                if args.verbose:
                    print(f"  file_id={file_id}: {e}")
                errors += 1
                continue
        else:
            if args.verbose:
                print(f"  file_id={file_id}: путь недоступен")
            errors += 1
            continue

        main_cur.execute(
            """
            SELECT pr.id, pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h, pr.frame_t_sec
            FROM photo_rectangles pr
            WHERE pr.file_id = ? AND pr.embedding IS NOT NULL AND pr.is_face = 1 AND COALESCE(pr.ignore_flag, 0) = 0
            """,
            (file_id,),
        )
        rects = main_cur.fetchall()
        is_video = local_path.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".3gp"))
        cap = None
        if is_video:
            cap = cv2.VideoCapture(local_path)
            if not cap or not cap.isOpened():
                if tmp_path and os.path.isfile(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                continue
        else:
            frame = cv2.imread(local_path)
            if frame is None:
                if tmp_path and os.path.isfile(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                continue

        for r in rects:
            rect_id = r[0]
            bbox = (r[1], r[2], r[3], r[4])
            frame_t_sec = r[5]
            try:
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(frame_t_sec or 0)) * 1000.0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        errors += 1
                        continue
                h, w = frame.shape[:2]
                x, y, bw, bh = bbox
                pad = int(round(max(bw, bh) * 0.18))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    errors += 1
                    continue
                emb_bytes = _extract_embedding_from_crop(crop, model)
                if emb_bytes is None:
                    errors += 1
                    continue
                exp_cur.execute(
                    """
                    INSERT INTO face_embeddings_alt (rectangle_id, model_key, embedding)
                    VALUES (?, ?, ?)
                    ON CONFLICT(rectangle_id, model_key) DO UPDATE SET embedding = excluded.embedding
                    """,
                    (rect_id, args.model_key, emb_bytes),
                )
                exp_conn.commit()
                done += 1
            except Exception as e:
                errors += 1
                if args.verbose:
                    print(f"  rect_id={rect_id}: {e}")
        if is_video and cap:
            cap.release()
        if (i + 1) % 20 == 0 or (i + 1) == len(file_rows):
            print(f"  [{i + 1}/{len(file_rows)}] файлов, эмбеддингов записано: {done}, ошибок: {errors}")

        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    main_conn.close()
    exp_conn.close()
    print(f"Готово. Записано в experiments.db: {done} эмбеддингов, ошибок: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
