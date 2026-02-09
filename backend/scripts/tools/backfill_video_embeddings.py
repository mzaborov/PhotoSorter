#!/usr/bin/env python3
"""
Извлекает embeddings для лиц на видео (photo_rectangles с frame_idx IS NOT NULL).

Выбирает записи с embedding IS NULL, загружает кадр видео в frame_t_sec,
кропает лицо по bbox, извлекает embedding и обновляет photo_rectangles.

Использование:
  python backend/scripts/tools/backfill_video_embeddings.py --face-run-id 123
  python backend/scripts/tools/backfill_video_embeddings.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/scripts/tools -> project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore[import-untyped]
except ImportError:
    ort = None  # type: ignore[assignment]


def _path_to_abs(path: str) -> str | None:
    r"""local:C:\... или local:/path -> абсолютный путь."""
    if not path or not isinstance(path, str):
        return None
    p = path.replace("local:", "").strip()
    if not p:
        return None
    return os.path.abspath(p)


def load_recognition_model() -> dict | None:
    """Загружает модель распознавания лиц (ArcFace)."""
    if ort is None:
        print("onnxruntime не установлен")
        return None
    try:
        repo_root = Path(__file__).resolve().parents[3]  # backend/scripts/tools -> project root
        candidates = [
            repo_root / "models" / "face_recognition" / "w600k_r50.onnx",
            repo_root / "models" / "face_recognition" / "arcface_r50_v1.onnx",
            repo_root.parent / "models" / "face_recognition" / "w600k_r50.onnx",
        ]
        onnx_path = None
        for p in candidates:
            if p.exists():
                onnx_path = p
                break
        if onnx_path is None:
            try:
                home = Path.home()
                for sub in (".insightface/models/buffalo_l", ".insightface/models/buffalo_s"):
                    cand = home / sub / "w600k_r50.onnx"
                    if cand.exists():
                        onnx_path = cand
                        break
            except Exception:
                pass
        if onnx_path is None:
            print(f"Модель не найдена. Проверьте models/face_recognition/w600k_r50.onnx")
            return None

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        return {
            "session": sess,
            "input_name": inp.name,
            "output_name": sess.get_outputs()[0].name,
            "input_shape": inp.shape,
        }
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None


def extract_embedding(face_bgr: np.ndarray, model: dict) -> bytes | None:
    """Извлекает embedding из кропа лица (BGR)."""
    try:
        sess = model["session"]
        inp_name = model["input_name"]
        out_name = model["output_name"]
        shape = model["input_shape"]
        target = (int(shape[3]), int(shape[2])) if len(shape) == 4 else (112, 112)

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target)
        norm = (resized.astype(np.float32) - 127.5) / 128.0
        batch = np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0)

        out = sess.run([out_name], {inp_name: batch})
        emb = out[0][0]
        if emb is None or emb.size == 0:
            return None
        norm_val = np.linalg.norm(emb)
        if norm_val > 0:
            emb = emb / norm_val
        return json.dumps(emb.tolist()).encode("utf-8")
    except Exception:
        return None


def process_faces(conn, cur, model: dict, run_id: int | None) -> tuple[int, int]:
    """Обрабатывает все видео-лица без embedding. Возвращает (обработано, ошибок)."""
    if run_id is not None:
        cur.execute(
            """
            SELECT pr.id, pr.file_id, pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h,
                   pr.frame_t_sec, f.path AS file_path
            FROM photo_rectangles pr
            JOIN files f ON f.id = pr.file_id
            WHERE pr.run_id = ?
              AND pr.embedding IS NULL
              AND pr.frame_idx IS NOT NULL
              AND pr.frame_t_sec IS NOT NULL
              AND COALESCE(pr.ignore_flag, 0) = 0
            ORDER BY pr.id
            """,
            (run_id,),
        )
    else:
        cur.execute(
            """
            SELECT pr.id, pr.file_id, pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h,
                   pr.frame_t_sec, f.path AS file_path
            FROM photo_rectangles pr
            JOIN files f ON f.id = pr.file_id
            WHERE pr.embedding IS NULL
              AND pr.frame_idx IS NOT NULL
              AND pr.frame_t_sec IS NOT NULL
              AND COALESCE(pr.ignore_flag, 0) = 0
            ORDER BY pr.id
            """,
        )

    rows = cur.fetchall()
    total = len(rows)
    if total == 0:
        return 0, 0

    processed = 0
    errors = 0
    last_path: str | None = None
    cap: cv2.VideoCapture | None = None

    for i, row in enumerate(rows):
        rect_id = row["id"]
        bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
        t_sec = float(row["frame_t_sec"] or 0)
        file_path = row["file_path"] or ""

        abs_path = _path_to_abs(file_path)
        if not abs_path or not os.path.isfile(abs_path):
            errors += 1
            continue

        try:
            # Переиспользуем VideoCapture, если тот же файл
            if last_path != abs_path:
                if cap is not None:
                    cap.release()
                    cap = None
                cap = cv2.VideoCapture(abs_path)
                last_path = abs_path

            if cap is None or not cap.isOpened():
                errors += 1
                continue

            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                errors += 1
                continue

            h, w = frame.shape[:2]
            x, y, bw, bh = bbox
            pad = int(round(max(bw, bh) * 0.18))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            crop = frame[y0:y1, x0:x1]

            if crop.size == 0:
                errors += 1
                continue

            emb = extract_embedding(crop, model)
            if emb is None:
                errors += 1
                continue

            cur.execute("UPDATE photo_rectangles SET embedding = ? WHERE id = ?", (emb, rect_id))
            processed += 1
            conn.commit()

            if (i + 1) % 50 == 0:
                print(f"  Обработано: {i + 1}/{total}, успешно: {processed}, ошибок: {errors}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Ошибка rect_id={rect_id}: {type(e).__name__}: {e}")

    if cap is not None:
        cap.release()

    return processed, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Извлекает embeddings для лиц на видео (photo_rectangles с frame_idx)"
    )
    parser.add_argument(
        "--face-run-id",
        type=int,
        default=None,
        help="Обработать только лица из указанного face_run_id",
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=int,
        default=None,
        help="Взять face_run_id из pipeline_runs (альтернатива --face-run-id)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Обработать все видео-лица без embedding (игнорирует --face-run-id)",
    )
    args = parser.parse_args()

    run_id: int | None = None
    if not args.all:
        run_id = args.face_run_id
        if run_id is None and args.pipeline_run_id is not None:
            from backend.common.db import PipelineStore
            ps = PipelineStore()
            try:
                pr = ps.get_run_by_id(run_id=args.pipeline_run_id)
                if pr:
                    run_id = pr.get("face_run_id")
                    if run_id is not None:
                        run_id = int(run_id)
                if run_id is None:
                    print(f"У прогона {args.pipeline_run_id} нет face_run_id", file=sys.stderr)
                    return 1
            finally:
                ps.close()
        if run_id is None:
            print("Укажите --face-run-id <id>, --pipeline-run-id <id> или --all", file=sys.stderr)
            return 1

    print("Загрузка модели распознавания...")
    model = load_recognition_model()
    if not model:
        print("✗ Модель не найдена", file=sys.stderr)
        return 1
    print("✓ Модель загружена")

    from backend.common.db import get_connection

    conn = get_connection()
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
    cur = conn.cursor()

    try:
        if run_id is not None:
            print(f"\nОбработка face_run_id={run_id}...")
        else:
            print("\nОбработка всех видео-лиц без embedding...")

        processed, errors = process_faces(conn, cur, model, run_id)
        print(f"\nГотово. Обработано: {processed}, ошибок: {errors}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
