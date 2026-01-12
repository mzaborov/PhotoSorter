from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
from PIL import Image, ExifTags, ImageOps  # type: ignore[import-untyped]

from common.db import DedupStore, FaceStore, PipelineStore
from logic.gold.store import gold_file_map, gold_normalize_path, gold_read_lines


EXCLUDE_DIR_NAMES_DEFAULT = ("_faces", "_no_faces", "_duplicates", "_animals", "_non_media", "_broken_media")

# Step 1 (предочистка) folders (до дедупликации):
# - _non_media: files that are not image/video (pdf/txt/zip/...)
# - _broken_media: looks like media by extension, but cannot be decoded/opened
NON_MEDIA_DIRNAME_DEFAULT = "_non_media"
BROKEN_MEDIA_DIRNAME_DEFAULT = "_broken_media"

# OpenCV иногда нестабилен на некоторых системах при многопоточности.
# Для конвейера нам важнее устойчивость, чем скорость.
try:
    cv2.setNumThreads(1)
except Exception:
    pass


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pipe_log(pipeline: PipelineStore | None, pipeline_run_id: int | None, line: str) -> None:
    s = (line or "").replace("\r\n", "\n")
    print(s, end="" if s.endswith("\n") else "\n")
    if pipeline is None or pipeline_run_id is None:
        return
    try:
        pipeline.append_log(run_id=int(pipeline_run_id), line=s if s.endswith("\n") else (s + "\n"))
    except Exception:
        # Лог — best-effort. Конвейер не должен падать из-за сбоя записи лога.
        pass


def _as_local_path(p: str) -> str:
    # Храним локальные пути в общей таблице с префиксом, чтобы не конфликтовать с disk:/...
    return "local:" + str(p)


def _strip_local_prefix(p: str) -> str:
    return p[len("local:") :] if (p or "").startswith("local:") else p


def _sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_files(root_dir: str, *, exclude_dir_names: tuple[str, ...]) -> Iterable[str]:
    root = os.path.abspath(root_dir)
    for dirpath, dirnames, filenames in os.walk(root):
        # не заходим в служебные папки
        dirnames[:] = [d for d in dirnames if d not in set(exclude_dir_names)]
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def _is_image(path: str) -> bool:
    mime_type, _enc = mimetypes.guess_type(path)
    if mime_type and mime_type.startswith("image/"):
        return True
    ext = (Path(path).suffix or "").lower()
    return ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic")


def _is_video(path: str) -> bool:
    mime_type, _enc = mimetypes.guess_type(path)
    if mime_type and mime_type.startswith("video/"):
        return True
    ext = (Path(path).suffix or "").lower()
    return ext in (".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".3gp")


def _is_media(path: str) -> bool:
    return _is_image(path) or _is_video(path)


def _safe_move_unique(src: str, dst: str) -> str:
    """
    Move src -> dst, avoiding overwrites by adding suffix if needed.
    Returns final destination path.
    """
    d = Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        shutil.move(src, str(d))
        return str(d)
    stem = d.stem
    suf = d.suffix
    parent = d.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}__dup{i}{suf}"
        if not cand.exists():
            shutil.move(src, str(cand))
            return str(cand)
    raise RuntimeError(f"failed_to_choose_unique_name_for:{dst}")


def _is_broken_image_file(path: str) -> bool:
    """
    ВАЖНО: никаких "полных" декодирований через PIL здесь не делаем.
    На некоторых повреждённых файлах декодер может падать нативно (access violation),
    и это убьёт весь процесс конвейера.

    Поэтому используем безопасные эвристики по заголовкам/магическим байтам.
    Это не идеально, но надёжно и быстро.
    """
    try:
        ext = (Path(path).suffix or "").lower()
        with open(path, "rb") as f:
            head = f.read(64)
            if ext in (".jpg", ".jpeg"):
                if len(head) < 2 or head[0:2] != b"\xff\xd8":
                    return True
                try:
                    f.seek(-2, os.SEEK_END)
                    tail = f.read(2)
                except Exception:
                    return True
                return tail != b"\xff\xd9"
            if ext == ".png":
                return not (len(head) >= 8 and head[0:8] == b"\x89PNG\r\n\x1a\n")
            if ext == ".webp":
                return not (len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP")
            if ext == ".bmp":
                return not (len(head) >= 2 and head[0:2] == b"BM")
            if ext in (".tif", ".tiff"):
                return not (
                    len(head) >= 4
                    and (head[0:4] == b"II*\x00" or head[0:4] == b"MM\x00*")
                )
        # неизвестный формат: не считаем битым (пусть дальше обрабатывается обычным путём)
        return False
    except Exception:
        return True


def _is_broken_video_file(path: str) -> bool:
    try:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return True
            ok, _frame = cap.read()
            return not bool(ok)
        finally:
            try:
                cap.release()
            except Exception:
                pass
    except Exception:
        return True


def _preclean_nonmedia_and_broken(
    *,
    root_dir: str,
    dry_run: bool,
    exclude_dir_names: tuple[str, ...],
    non_media_dirname: str,
    broken_media_dirname: str,
    pipeline: PipelineStore | None,
    pipeline_run_id: int | None,
) -> dict[str, int]:
    """
    Step 1 (предочистка): до дедупликации перемещаем:
    - non-media (not image/video) -> _non_media
    - broken media (ext looks like image/video but cannot be decoded) -> _broken_media
    Returns counts.
    """
    root = os.path.abspath(root_dir)
    non_media_dir = os.path.join(root, non_media_dirname)
    broken_dir = os.path.join(root, broken_media_dirname)

    moved_non_media = 0
    moved_broken = 0
    checked = 0

    # --- resume state / results in DB (best-effort) ---
    # Храним в preclean_state + preclean_moves по pipeline_run_id.
    # Для resume порядок файлов фиксируем сортировкой (по нормализованному пути).
    last_seen: str | None = None
    if pipeline_run_id is not None:
        try:
            cur = pipeline.conn.cursor() if pipeline is not None else None
            if cur is not None:
                cur.execute(
                    "SELECT checked, non_media, broken_media, last_path, dry_run FROM preclean_state WHERE pipeline_run_id=? LIMIT 1",
                    (int(pipeline_run_id),),
                )
                row = cur.fetchone()
                if row:
                    checked = int(row[0] or 0)
                    moved_non_media = int(row[1] or 0)
                    moved_broken = int(row[2] or 0)
                    last_seen = str(row[3]) if row[3] else None
                else:
                    cur.execute(
                        "INSERT OR REPLACE INTO preclean_state(pipeline_run_id, root_path, dry_run, checked, non_media, broken_media, last_path, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                        (int(pipeline_run_id), str(root), 1 if dry_run else 0, 0, 0, 0, None, _now_utc_iso()),
                    )
                    pipeline.conn.commit()
        except Exception:
            last_seen = None

    def _save_state(*, last_path: str | None) -> None:
        if pipeline is None or pipeline_run_id is None:
            return
        try:
            pipeline.conn.execute(
                "INSERT OR REPLACE INTO preclean_state(pipeline_run_id, root_path, dry_run, checked, non_media, broken_media, last_path, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (int(pipeline_run_id), str(root), 1 if dry_run else 0, int(checked), int(moved_non_media), int(moved_broken), last_path, _now_utc_iso()),
            )
            pipeline.conn.commit()
        except Exception:
            pass

    def _add_move(*, kind: str, src_abs: str, dst_abs: str, is_applied: bool) -> None:
        if pipeline is None or pipeline_run_id is None:
            return
        try:
            pipeline.conn.execute(
                "INSERT INTO preclean_moves(pipeline_run_id, kind, src_path, dst_path, is_applied, created_at) VALUES(?,?,?,?,?,?)",
                (int(pipeline_run_id), str(kind), _as_local_path(os.path.normcase(src_abs)), _as_local_path(os.path.normcase(dst_abs)), 1 if is_applied else 0, _now_utc_iso()),
            )
        except Exception:
            pass

    # Стабильный порядок для resume.
    files = list(_iter_files(root, exclude_dir_names=exclude_dir_names))
    files.sort(key=lambda p: _as_local_path(os.path.normcase(p)))

    for abspath in files:
        norm_abs = os.path.normcase(abspath)
        db_abs = _as_local_path(norm_abs)
        if last_seen and db_abs <= last_seen:
            continue
        checked += 1
        # Skip if already under target dirs (defensive)
        if abspath.startswith(non_media_dir + os.sep) or abspath.startswith(broken_dir + os.sep):
            continue

        rel = os.path.relpath(abspath, root)
        if not _is_media(abspath):
            dst = os.path.join(non_media_dir, rel)
            if dry_run:
                _pipe_log(pipeline, pipeline_run_id, f"preclean: non_media -> {dst}\n")
                _add_move(kind="non_media", src_abs=abspath, dst_abs=dst, is_applied=False)
            else:
                dst_final = _safe_move_unique(abspath, dst)
                _pipe_log(pipeline, pipeline_run_id, f"preclean: moved non_media -> {dst_final}\n")
                _add_move(kind="non_media", src_abs=abspath, dst_abs=dst_final, is_applied=True)
            moved_non_media += 1
            continue

        # media by ext -> validate
        is_broken = False
        if _is_image(abspath):
            is_broken = _is_broken_image_file(abspath)
        elif _is_video(abspath):
            is_broken = _is_broken_video_file(abspath)
        if is_broken:
            dst = os.path.join(broken_dir, rel)
            if dry_run:
                _pipe_log(pipeline, pipeline_run_id, f"preclean: broken_media -> {dst}\n")
                _add_move(kind="broken_media", src_abs=abspath, dst_abs=dst, is_applied=False)
            else:
                dst_final = _safe_move_unique(abspath, dst)
                _pipe_log(pipeline, pipeline_run_id, f"preclean: moved broken_media -> {dst_final}\n")
                _add_move(kind="broken_media", src_abs=abspath, dst_abs=dst_final, is_applied=True)
            moved_broken += 1

        # прогресс/резюме (редко)
        if checked % 200 == 0:
            _pipe_log(
                pipeline,
                pipeline_run_id,
                f"preclean: checked={checked} moved_non_media={moved_non_media} moved_broken_media={moved_broken}\n",
            )
            _save_state(last_path=db_abs)

    _save_state(last_path=(_as_local_path(os.path.normcase(files[-1])) if files else last_seen))
    _pipe_log(
        pipeline,
        pipeline_run_id,
        f"preclean: checked={checked} moved_non_media={moved_non_media} moved_broken_media={moved_broken}\n",
    )

    return {"checked": checked, "moved_non_media": moved_non_media, "moved_broken_media": moved_broken}


def _load_preclean_exclude_paths(*, pipeline: PipelineStore | None, pipeline_run_id: int | None) -> set[str]:
    """
    Возвращает множество local:-путей, которые шаг 1 (предочистка) пометил к переносу в _non_media/_broken_media.
    Используем это в DRY-RUN, чтобы последующие шаги считали/обрабатывали файлы "как после очистки",
    не выполняя реальных перемещений.
    """
    if pipeline is None or pipeline_run_id is None:
        return set()
    try:
        cur = pipeline.conn.cursor()
        cur.execute(
            """
            SELECT src_path
            FROM preclean_moves
            WHERE pipeline_run_id = ?
              AND kind IN ('non_media', 'broken_media')
            """,
            (int(pipeline_run_id),),
        )
        rows = cur.fetchall()
        out: set[str] = set()
        for r in rows:
            p = str(r[0] or "").strip()
            if not p:
                continue
            # preclean записывает нормализованный local:-путь; держим его как ключ для сравнения.
            out.add(p)
        return out
    except Exception:
        return set()


def _effective_tab_sql_for_metrics() -> str:
    # aliases: f (files), m (files_manual_labels)
    return """
    CASE
      WHEN COALESCE(m.people_no_face_manual, 0) = 1 THEN 'people_no_face'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'faces' THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'no_faces' THEN 'no_faces'
      WHEN COALESCE(m.quarantine_manual, 0) = 1
           AND COALESCE(f.faces_count, 0) > 0
        THEN 'quarantine'
      WHEN COALESCE(m.animals_manual, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.animals_auto, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.faces_auto_quarantine, 0) = 1
           AND COALESCE(f.faces_count, 0) > 0
           AND lower(trim(coalesce(f.faces_quarantine_reason, ''))) != 'many_small_faces'
        THEN 'quarantine'
      ELSE (CASE WHEN COALESCE(f.faces_count, 0) > 0 THEN 'faces' ELSE 'no_faces' END)
    END
    """


def _count_misses_for_gold(
    *,
    ds: DedupStore,
    pipeline_run_id: int,
    face_run_id: int | None,
    root_like: str | None,
    gold_name: str,
    expected_tab: str,
) -> tuple[int, int]:
    fp = gold_file_map().get(gold_name)
    if not fp:
        return 0, 0
    lines = gold_read_lines(fp)
    paths: list[str] = []
    for raw in lines:
        nm = gold_normalize_path(raw)
        pp = str(nm.get("path") or "")
        if not pp or not (pp.startswith("local:") or pp.startswith("disk:")):
            continue
        if root_like and root_like.endswith("%"):
            pref = root_like[:-1]
            if not pp.startswith(pref):
                continue
        paths.append(pp)
    # unique
    uniq: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    if not uniq:
        return 0, 0
    placeholders = ",".join(["?"] * len(uniq))
    eff_sql = _effective_tab_sql_for_metrics()
    sql = f"""
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN ({eff_sql}) != ? THEN 1 ELSE 0 END) AS mism
        FROM files f
        LEFT JOIN files_manual_labels m
          ON m.pipeline_run_id = ? AND m.path = f.path
        WHERE f.status != 'deleted'
          AND (? IS NULL OR COALESCE(f.faces_run_id, -1) = ?)
          AND f.path IN ({placeholders})
    """
    frid = int(face_run_id) if face_run_id else None
    row = ds.conn.execute(sql, [expected_tab, int(pipeline_run_id), frid, frid, *uniq]).fetchone()
    total = int(row["total"] or 0) if row else 0
    mism = int(row["mism"] or 0) if row else 0
    return total, mism


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def ensure_yunet_model(*, model_path: Path, model_url: str) -> Path:
    if model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    # маленький и безопасный download через urllib в face_scan.py; здесь переиспользуем shutil+urlretrieve не хотим
    import urllib.request

    _ensure_parent_dir(model_path)
    tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    print(f"Downloading YuNet model to: {model_path}")
    urllib.request.urlretrieve(model_url, tmp)  # noqa: S310
    tmp.replace(model_path)
    return model_path


def ensure_yolo_model(*, model_path: Path, model_urls: list[str]) -> Path:
    """
    Скачивает YOLO ONNX модель (COCO) для детекта кошки, если она отсутствует.
    """
    if model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    _ensure_parent_dir(model_path)
    tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    last_err: Exception | None = None
    for url in model_urls:
        try:
            print(f"Downloading YOLO model to: {model_path} ({url})")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            tmp.replace(model_path)
            return model_path
        except Exception as e:  # noqa: BLE001
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            continue
    raise RuntimeError(f"Failed to download YOLO model to {model_path}: {last_err}")


def _letterbox(img_bgr: np.ndarray, *, new_size: int = 640, color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, int, int]:
    """
    Letterbox resize для YOLO: возвращает (img, r, dw, dh),
    где r — scale, dw/dh — паддинги слева/сверху.
    """
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return img_bgr, 1.0, 0, 0
    r = min(float(new_size) / float(h), float(new_size) / float(w))
    nh = int(round(h * r))
    nw = int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR)
    pad_w = new_size - nw
    pad_h = new_size - nh
    dw = int(round(pad_w / 2.0))
    dh = int(round(pad_h / 2.0))
    out = cv2.copyMakeBorder(resized, dh, pad_h - dh, dw, pad_w - dw, cv2.BORDER_CONSTANT, value=color)
    return out, r, dw, dh


def _detect_cat_yolo(
    *,
    sess,
    input_name: str,
    img_bgr_full: np.ndarray,
    conf_threshold: float,
    nms_iou: float,
    min_box_area_ratio: float,
) -> bool:
    """
    Детект кошки через YOLO (COCO) ONNX. Возвращает True, если нашли хотя бы одну кошку.

    Важно: некоторые ONNX модели ожидают input dtype float16 (например, yolov5s.onnx из релиза),
    поэтому подбираем dtype по `sess.get_inputs()[0].type`.
    """
    h0, w0 = img_bgr_full.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return False
    lb, r, dw, dh = _letterbox(img_bgr_full, new_size=640)
    # BGR->RGB, normalize, CHW
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    try:
        in_type = (sess.get_inputs()[0].type or "").lower()
        if "float16" in in_type:
            x = x.astype(np.float16)
    except Exception:
        pass
    outs = sess.run(None, {input_name: x})
    if not outs:
        return False
    out = outs[0]
    # Нормализуем форму в (N, D)
    if out.ndim == 3:
        o = out[0]
        if o.shape[0] in (84, 85):
            o = o.transpose(1, 0)  # (N, D)
        # иначе предполагаем (N, D) уже
    elif out.ndim == 2:
        o = out
    else:
        return False
    if o is None or len(o.shape) != 2:
        return False
    d = int(o.shape[1])
    # COCO: cat class index = 15 (0-based)
    cat_idx = 15
    boxes_xywh: list[list[int]] = []
    scores: list[float] = []
    for row in o:
        if d == 84:
            # [x,y,w,h] + 80 class scores
            x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            conf = float(row[4 + cat_idx])
        elif d == 85:
            # [x,y,w,h,obj] + 80 class scores
            x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            obj = float(row[4])
            conf = obj * float(row[5 + cat_idx])
        else:
            continue
        if conf < float(conf_threshold):
            continue
        # xywh -> xyxy (letterboxed)
        x1 = x_c - bw / 2.0
        y1 = y_c - bh / 2.0
        x2 = x_c + bw / 2.0
        y2 = y_c + bh / 2.0
        # unletterbox -> original
        x1 = (x1 - float(dw)) / float(r)
        y1 = (y1 - float(dh)) / float(r)
        x2 = (x2 - float(dw)) / float(r)
        y2 = (y2 - float(dh)) / float(r)
        # clip
        x1i = max(0, min(int(round(x1)), w0 - 1))
        y1i = max(0, min(int(round(y1)), h0 - 1))
        x2i = max(0, min(int(round(x2)), w0 - 1))
        y2i = max(0, min(int(round(y2)), h0 - 1))
        if x2i <= x1i or y2i <= y1i:
            continue
        area = float((x2i - x1i) * (y2i - y1i))
        if area / float(w0 * h0) < float(min_box_area_ratio):
            continue
        boxes_xywh.append([x1i, y1i, x2i - x1i, y2i - y1i])
        scores.append(conf)
    if not boxes_xywh:
        return False
    try:
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=float(conf_threshold), nms_threshold=float(nms_iou))
        if idxs is None:
            return False
        return len(idxs) > 0
    except Exception:
        # best-effort: если NMS недоступен/упал, считаем по первому боксу
        return True


def _detect_class_yolo(
    *,
    sess,
    input_name: str,
    img_bgr_full: np.ndarray,
    class_idx: int,
    conf_threshold: float,
    nms_iou: float,
    min_box_area_ratio: float,
) -> bool:
    """
    Детект класса COCO через YOLO ONNX. Возвращает True, если нашли хотя бы один бокс данного класса.
    """
    h0, w0 = img_bgr_full.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return False
    lb, r, dw, dh = _letterbox(img_bgr_full, new_size=640)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    try:
        in_type = (sess.get_inputs()[0].type or "").lower()
        if "float16" in in_type:
            x = x.astype(np.float16)
    except Exception:
        pass
    outs = sess.run(None, {input_name: x})
    if not outs:
        return False
    out = outs[0]
    # Нормализуем форму в (N, D)
    if out.ndim == 3:
        o = out[0]
        if o.shape[0] in (84, 85):
            o = o.transpose(1, 0)  # (N, D)
    elif out.ndim == 2:
        o = out
    else:
        return False
    if o is None or len(o.shape) != 2:
        return False
    d = int(o.shape[1])
    boxes_xywh: list[list[int]] = []
    scores: list[float] = []
    for row in o:
        if d == 84:
            x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            conf = float(row[4 + int(class_idx)])
        elif d == 85:
            x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            obj = float(row[4])
            conf = obj * float(row[5 + int(class_idx)])
        else:
            continue
        if conf < float(conf_threshold):
            continue
        # xywh -> xyxy (letterboxed)
        x1 = x_c - bw / 2.0
        y1 = y_c - bh / 2.0
        x2 = x_c + bw / 2.0
        y2 = y_c + bh / 2.0
        # unletterbox -> original
        x1 = (x1 - float(dw)) / float(r)
        y1 = (y1 - float(dh)) / float(r)
        x2 = (x2 - float(dw)) / float(r)
        y2 = (y2 - float(dh)) / float(r)
        x1i = max(0, min(int(round(x1)), w0 - 1))
        y1i = max(0, min(int(round(y1)), h0 - 1))
        x2i = max(0, min(int(round(x2)), w0 - 1))
        y2i = max(0, min(int(round(y2)), h0 - 1))
        if x2i <= x1i or y2i <= y1i:
            continue
        area = float((x2i - x1i) * (y2i - y1i))
        if area / float(w0 * h0) < float(min_box_area_ratio):
            continue
        boxes_xywh.append([x1i, y1i, x2i - x1i, y2i - y1i])
        scores.append(conf)
    if not boxes_xywh:
        return False
    try:
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=float(conf_threshold), nms_threshold=float(nms_iou))
        if idxs is None:
            return False
        return len(idxs) > 0
    except Exception:
        return True


def _looks_like_screen_ui(img_bgr_full: np.ndarray) -> bool:
    """
    Best-effort признак "это экран/скрин/страница с UI/текстом".

    Цель: отправлять такие кадры в карантин (техничка), не засоряя "Есть лица".
    """
    h0, w0 = img_bgr_full.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return False
    ar = float(max(h0, w0)) / float(min(h0, w0))
    # Экраны/страницы на фото бывают и 4:3, и "вытянутые" (телефон).
    if not (1.00 <= ar <= 2.60):
        return False

    # Downscale for stable metrics
    max_dim = 720
    img = img_bgr_full
    m = max(h0, w0)
    if m > max_dim:
        sc = float(max_dim) / float(m)
        nh = max(1, int(round(h0 * sc)))
        nw = max(1, int(round(w0 * sc)))
        img = cv2.resize(img_bgr_full, (nw, nh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # brightness stats help to detect screens/pages (white cards, black bezels)
    white_ratio = float(np.count_nonzero(g >= 235)) / float(g.size)
    black_ratio = float(np.count_nonzero(g <= 20)) / float(g.size)
    # edges
    v = float(np.median(g))
    lo = max(0, int(0.66 * v))
    hi = min(255, int(1.33 * v))
    e = cv2.Canny(g, lo, hi)
    er_total = float(np.count_nonzero(e)) / float(e.size)
    h = int(e.shape[0])
    if h <= 0:
        return False
    top = e[: max(1, int(round(h * 0.12))), :]
    bot = e[h - max(1, int(round(h * 0.12))) :, :]
    er_top = float(np.count_nonzero(top)) / float(top.size)
    er_bot = float(np.count_nonzero(bot)) / float(bot.size)

    # ВАЖНО: по реальным данным прежняя эвристика была слишком агрессивной и ловила обычные фото как "screen_like".
    # Делаем более строгую версию: реагируем в первую очередь на "много текста"/страницу,
    # либо на ярко выраженную "страницу/экран" (много белого и есть границы), либо на тёмный UI.
    text_heavy = _looks_like_text_heavy(g)
    if text_heavy and er_total >= 0.015:
        return True
    # Белая "страница"/слайд/документ с рамками/текстом
    if white_ratio >= 0.18 and er_total >= 0.030:
        return True
    # Экран/страница с контрастными рамками (например, телефон/ноутбук в кадре)
    if (white_ratio >= 0.10 and black_ratio >= 0.08) and er_total >= 0.030:
        return True
    # Тёмный интерфейс (мессенджер/музыка): много чёрных пикселей + заметные edges
    if black_ratio >= 0.20 and er_total >= 0.055:
        return True
    # Широкий формат + высокий общий edge + тёмные рамки
    if ar >= 1.60 and black_ratio >= 0.06 and er_total >= 0.085:
        return True
    return False


def _looks_like_text_heavy(gray: np.ndarray) -> bool:
    """
    Оценка "много текста" без OCR:
    - бинаризация (adaptive threshold)
    - считаем количество мелких компонент (буквы/элементы UI)
    """
    try:
        if gray.ndim != 2:
            return False
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return False
        # adaptive threshold: white text blobs on black
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            5,
        )
        # remove speckles
        bw = cv2.medianBlur(bw, 3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        small = 0
        # stats: [x, y, w, h, area]
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 10 or area > 400:
                continue
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if ww <= 0 or hh <= 0:
                continue
            ar = float(ww) / float(hh)
            if 0.15 <= ar <= 8.0:
                small += 1
        # Порог подобран для даунскейла ~720px: на письмах/чатах обычно сотни компонент.
        return small >= 140
    except Exception:
        return False


def _qr_found_best_effort(img_bgr_full: np.ndarray, *, qr, qr_max_dim: int) -> bool:
    """
    QR detection tends to be flaky; try a few representations/sizes.
    """
    if qr is None:
        return False
    h0, w0 = img_bgr_full.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return False

    def _try_one(qimg: np.ndarray) -> bool:
        try:
            if hasattr(qr, "detectAndDecodeMulti"):
                _ok, decoded, _points, _ = qr.detectAndDecodeMulti(qimg)
                return bool(decoded and any(str(x or "").strip() for x in decoded))
            decoded, _points, _ = qr.detectAndDecode(qimg)
            return bool(str(decoded or "").strip())
        except Exception:
            return False

    # try at a couple of scales
    for target in (qr_max_dim, 900):
        qimg = img_bgr_full
        mh = max(int(h0), int(w0))
        if mh > target and mh > 0:
            sc = float(target) / float(mh)
            qh = max(1, int(round(h0 * sc)))
            qw = max(1, int(round(w0 * sc)))
            qimg = cv2.resize(img_bgr_full, (qw, qh), interpolation=cv2.INTER_AREA)

        # 1) raw BGR
        if _try_one(qimg):
            return True
        # 2) grayscale
        try:
            g = cv2.cvtColor(qimg, cv2.COLOR_BGR2GRAY)
        except Exception:
            g = None
        if g is not None and _try_one(g):
            return True
        # 3) binarized (helps on screen photos)
        if g is not None:
            try:
                th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
                if _try_one(th):
                    return True
            except Exception:
                pass
    return False

def _create_face_detector(model_path: str, *, score_threshold: float):
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(model_path, "", (320, 320), score_threshold, 0.3, 5000)
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)


def _detect_faces(detector, img_bgr) -> list[tuple[int, int, int, int, float]]:
    h, w = img_bgr.shape[:2]
    try:
        detector.setInputSize((w, h))
    except Exception:
        pass

    res = detector.detect(img_bgr)
    if isinstance(res, tuple) and len(res) >= 2:
        faces = res[1]
    else:
        faces = res
    if faces is None:
        return []

    out: list[tuple[int, int, int, int, float]] = []
    for row in faces:
        x, y, ww, hh, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        # OpenCV иногда отдаёт мусор/inf (видели OverflowError при int(round(inf))).
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(ww) and math.isfinite(hh) and math.isfinite(score)):
            continue
        # YuNet score expected ~[0..1], but we clamp for robustness (and to avoid garbage values in DB).
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        xi = max(0, int(round(x)))
        yi = max(0, int(round(y)))
        wi = max(0, int(round(ww)))
        hi = max(0, int(round(hh)))
        if xi >= w or yi >= h:
            continue
        if xi + wi > w:
            wi = max(0, w - xi)
        if yi + hi > h:
            hi = max(0, h - yi)
        if wi <= 1 or hi <= 1:
            continue
        out.append((xi, yi, wi, hi, float(score)))
    return out


def _rotate_bgr(img_bgr, *, rot: int):
    """
    rot: 90|180|270 (clockwise degrees).
    """
    if rot == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr


def _map_point_from_rotated_to_base(*, x: float, y: float, rot: int, w0: int, h0: int) -> tuple[float, float]:
    """
    Map a point from rotated image coordinates back to base image coordinates.

    Base image: width=w0, height=h0.
    Rotated image is produced by rotating base clockwise by rot degrees.
    """
    if rot == 90:
        # base -> rot: x' = h0-1-y, y' = x
        # rot -> base: x = y', y = h0-1-x'
        return (float(y), float(h0 - 1) - float(x))
    if rot == 180:
        return (float(w0 - 1) - float(x), float(h0 - 1) - float(y))
    if rot == 270:
        # base -> rot: x' = y, y' = w0-1-x
        # rot -> base: x = w0-1-y', y = x'
        return (float(w0 - 1) - float(y), float(x))
    return (float(x), float(y))


def _unrotate_bbox_to_base(*, x: int, y: int, w: int, h: int, rot: int, w0: int, h0: int) -> tuple[int, int, int, int]:
    """
    Convert bbox from rotated image coordinates back to base image coordinates.
    bbox: x,y,w,h (rotated coords).
    """
    x0 = float(x)
    y0 = float(y)
    x1 = float(x + w)
    y1 = float(y + h)
    pts = [
        _map_point_from_rotated_to_base(x=x0, y=y0, rot=rot, w0=w0, h0=h0),
        _map_point_from_rotated_to_base(x=x1, y=y0, rot=rot, w0=w0, h0=h0),
        _map_point_from_rotated_to_base(x=x0, y=y1, rot=rot, w0=w0, h0=h0),
        _map_point_from_rotated_to_base(x=x1, y=y1, rot=rot, w0=w0, h0=h0),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bx0 = max(0, int(math.floor(min(xs))))
    by0 = max(0, int(math.floor(min(ys))))
    bx1 = min(int(w0), int(math.ceil(max(xs))))
    by1 = min(int(h0), int(math.ceil(max(ys))))
    bw = max(0, bx1 - bx0)
    bh = max(0, by1 - by0)
    if bx0 >= w0 or by0 >= h0 or bw <= 1 or bh <= 1:
        return (0, 0, 0, 0)
    if bx0 + bw > w0:
        bw = max(0, w0 - bx0)
    if by0 + bh > h0:
        bh = max(0, h0 - by0)
    return (bx0, by0, bw, bh)


def _crop_thumb_jpeg(*, img: Image.Image, bbox: tuple[int, int, int, int], thumb_size: int, pad_ratio: float = 0.18) -> bytes:
    from io import BytesIO

    x, y, w, h = bbox
    iw, ih = img.size
    pad = int(round(max(w, h) * pad_ratio))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(iw, x + w + pad)
    y1 = min(ih, y + h + pad)
    crop = img.crop((x0, y0, x1, y1)).convert("RGB")
    crop.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=78, optimize=True)
    return buf.getvalue()


@dataclass
class Stats:
    total_files: int = 0
    processed: int = 0
    hashed: int = 0
    errors: int = 0
    duplicates_moved: int = 0
    images_scanned: int = 0
    videos_scanned: int = 0
    video_frames_scanned: int = 0
    faces_found: int = 0
    faces_files: int = 0
    no_faces_files: int = 0
    moved_faces: int = 0
    moved_no_faces: int = 0
    moved_quarantine: int = 0
    moved_animals: int = 0
    moved_people_no_face: int = 0
    moved_faces_named: int = 0
    moved_faces_unassigned: int = 0
    moved_no_faces_by_geo_year: int = 0


# region agent log (debug mode)
def _dbg_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    """
    Writes one NDJSON line to .cursor/debug.log (repo-relative). Best-effort; never raises.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        p = root / ".cursor" / "debug.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": str(hypothesis_id),
            "location": str(location),
            "message": str(message),
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion agent log (debug mode)


def _move_file(*, src: str, dst: str, dry_run: bool) -> None:
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if dry_run:
        return
    _ensure_parent_dir(Path(dst))
    shutil.move(src, dst)


def dedup_local(
    *,
    root_dir: str,
    dry_run: bool,
    move_duplicates: bool,
    duplicates_dirname: str,
    exclude_dir_names: tuple[str, ...],
    exclude_paths: set[str] | None = None,
    run_id: int | None = None,
    pipeline: PipelineStore | None = None,
    pipeline_run_id: int | None = None,
) -> Stats:
    stats = Stats()
    root = os.path.abspath(root_dir)

    store = DedupStore()
    try:
        if run_id is None:
            scope = f"pipeline:{int(pipeline_run_id)}:source" if pipeline_run_id else "source"
            run_id = store.create_run(scope=scope, root_path=root, max_download_bytes=None)
        run_id_i = int(run_id)
        if pipeline is not None and pipeline_run_id is not None:
            pipeline.update_run(run_id=int(pipeline_run_id), dedup_run_id=run_id_i)

        # total
        total = 0
        for _ in _iter_files(root, exclude_dir_names=exclude_dir_names):
            if exclude_paths is not None:
                try:
                    # Сравниваем по нормализованному local:-пути (Windows case-insensitive).
                    key = _as_local_path(os.path.normcase(_))
                    if key in exclude_paths:
                        continue
                except Exception:
                    pass
            total += 1
        store.update_run_progress(run_id=run_id_i, total_files=total)
        stats.total_files = total

        # 1) hash + upsert
        hash_to_first: dict[str, str] = {}
        for abspath in _iter_files(root, exclude_dir_names=exclude_dir_names):
            # В DRY-RUN после предочистки исключаем мусор/битые медиа (см. preclean_moves),
            # чтобы дедуп работал "как после очистки".
            if exclude_paths is not None:
                try:
                    key = _as_local_path(os.path.normcase(abspath))
                    if key in exclude_paths:
                        continue
                except Exception:
                    pass
            stats.processed += 1
            fn = os.path.basename(abspath)
            parent = os.path.dirname(abspath)
            db_path = _as_local_path(abspath)
            parent_path = _as_local_path(parent)
            mime_type, _enc = mimetypes.guess_type(abspath)
            media_type: Optional[str] = None
            if mime_type:
                if mime_type.startswith("image/"):
                    media_type = "image"
                elif mime_type.startswith("video/"):
                    media_type = "video"
            size: Optional[int] = None
            modified: Optional[str] = None
            try:
                st = os.stat(abspath)
                size = int(st.st_size)
                modified = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
            except OSError:
                pass

            try:
                # Resume-friendly: если уже есть валидный hash в БД и метаданные совпадают — не пересчитываем.
                sha: str | None = None
                row = store.get_row_by_path(path=db_path)
                if row and row.get("hash_alg") and row.get("hash_value") and row.get("status") == "hashed":
                    same_size = (row.get("size") is None) or (size is None) or (int(row.get("size")) == int(size))
                    same_modified = (row.get("modified") is None) or (modified is None) or (str(row.get("modified")) == str(modified))
                    if same_size and same_modified:
                        sha = str(row.get("hash_value"))
                if not sha:
                    sha = _sha256_file(abspath)
                    stats.hashed += 1
                store.upsert_file(
                    run_id=run_id_i,
                    path=db_path,
                    inventory_scope="source",
                    name=fn,
                    parent_path=parent_path,
                    size=size,
                    created=None,
                    modified=modified,
                    mime_type=mime_type,
                    media_type=media_type,
                    hash_alg="sha256",
                    hash_value=sha,
                    hash_source="local",
                    status="hashed",
                    error=None,
                    scanned_at=None,
                    hashed_at=_now_utc_iso(),
                )
            except Exception as e:  # noqa: BLE001
                stats.errors += 1
                store.upsert_file(
                    run_id=run_id_i,
                    path=db_path,
                    inventory_scope="source",
                    name=fn,
                    parent_path=parent_path,
                    size=size,
                    created=None,
                    modified=modified,
                    mime_type=mime_type,
                    media_type=media_type,
                    hash_alg=None,
                    hash_value=None,
                    hash_source=None,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    scanned_at=None,
                    hashed_at=None,
                )
                continue

            # 2) move duplicates (optional)
            if not move_duplicates:
                continue
            first = hash_to_first.get(sha)
            if first is None:
                hash_to_first[sha] = abspath
                continue

            # duplicate: move into _duplicates keeping relative path from root
            rel = os.path.relpath(abspath, root)
            dst = os.path.join(root, duplicates_dirname, rel)
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path=abspath, last_dst_path=dst)
            _move_file(src=abspath, dst=dst, dry_run=dry_run)
            stats.duplicates_moved += 1

            # update DB path to new location (ONLY if реально переместили файл)
            if not dry_run:
                store.update_path(
                    old_path=_as_local_path(abspath),
                    new_path=_as_local_path(dst),
                    new_name=os.path.basename(dst),
                    new_parent_path=_as_local_path(os.path.dirname(dst)),
                )
                # ВАЖНО: переносим run-scoped manual метки вместе с файлом, иначе они "теряются" после move.
                if pipeline_run_id is not None:
                    try:
                        store.update_run_manual_labels_path(
                            pipeline_run_id=int(pipeline_run_id),
                            old_path=_as_local_path(abspath),
                            new_path=_as_local_path(dst),
                        )
                    except Exception:
                        pass
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path="", last_dst_path="")

            # Прогресс печатаем редко, чтобы UI мог показывать шаг "идёт".
            if stats.processed % 250 == 0:
                # И в БД тоже двигаем прогресс, иначе UI видит "0%".
                store.update_run_progress(
                    run_id=run_id_i,
                    processed_files=stats.processed,
                    hashed_files=stats.hashed,
                    errors_count=stats.errors,
                    last_path=db_path,
                )
                _pipe_log(
                    pipeline,
                    pipeline_run_id,
                    f"dedup_progress processed={stats.processed} total={stats.total_files} "
                    f"dup_moved={stats.duplicates_moved} errors={stats.errors}\n",
                )

        # финальный прогресс
        store.update_run_progress(
            run_id=run_id_i,
            processed_files=stats.total_files,
            hashed_files=stats.hashed,
            errors_count=stats.errors,
            last_path="",
        )
        store.finish_run(run_id=run_id_i, status="completed")
    finally:
        store.close()

    return stats


def scan_faces_local(
    *,
    root_dir: str,
    score_threshold: float,
    thumb_size: int,
    model_path: Path,
    model_url: str,
    exclude_dir_names: tuple[str, ...],
    exclude_paths: set[str] | None = None,
    enable_qr: bool = False,
    video_samples: int = 0,
    video_max_dim: int = 640,
    video_bad_mean_luma: float = 25.0,
    video_bad_std_luma: float = 10.0,
    video_bad_black_ratio: float = 0.85,
    video_black_luma_threshold: int = 16,
    run_id: int | None = None,
    pipeline: PipelineStore | None = None,
    pipeline_run_id: int | None = None,
) -> tuple[int, Stats]:
    stats = Stats()
    root = os.path.abspath(root_dir)

    model = ensure_yunet_model(model_path=model_path, model_url=model_url)
    detector = _create_face_detector(str(model), score_threshold=float(score_threshold))
    detector_recreate_every = 200  # снижает шанс нативных крэшей внутри detect()
    max_dim = 2000  # safety: не гоняем гигантские картинки через YuNet
    quarantine_max_face_dim_px = 44
    many_faces_min = 8  # >=8 лиц считаем "много лиц" (2-й уровень вкладок /faces)
    quarantine_max_face_area_ratio = 0.003  # ~0.3% кадра: типично "лицо на экране/миниатюра"
    qr_max_dim = 1200  # для QR детектора не нужен full-res

    store = FaceStore()
    dedup = DedupStore()
    # --- Yandex geocoder (country/city) for no_faces UI sorting ---
    y_key = (os.getenv("YANDEX_GEOCODER_API_KEY") or "").strip()
    y_url = (os.getenv("YANDEX_GEOCODER_URL") or "https://geocode-maps.yandex.ru/1.x/").strip()
    y_lang = (os.getenv("YANDEX_GEOCODER_LANG") or "ru_RU").strip()
    try:
        y_rps = float(os.getenv("YANDEX_GEOCODER_RPS") or 3.0)
    except Exception:
        y_rps = 3.0
    try:
        y_round = int(os.getenv("YANDEX_GEOCODER_ROUND") or 4)
    except Exception:
        y_round = 4
    y_last_call = 0.0

    def _geo_cache_key(lat: float, lon: float) -> str:
        rr = int(max(0, min(6, y_round)))
        return f"{float(lat):.{rr}f},{float(lon):.{rr}f}"

    def _rate_limit_geocode() -> None:
        nonlocal y_last_call
        if not y_key or y_rps <= 0:
            return
        now = time.time()
        min_dt = 1.0 / float(y_rps)
        dt = now - float(y_last_call)
        if dt < min_dt:
            time.sleep(float(min_dt - dt))
        y_last_call = time.time()

    def _yandex_country_city(lat: float, lon: float) -> tuple[str | None, str | None, str | None]:
        """
        Returns (country, city, raw_json_str) best-effort.
        """
        if not y_key:
            return (None, None, None)
        try:
            q = {
                "apikey": y_key,
                "geocode": f"{float(lon)},{float(lat)}",
                "format": "json",
                "results": "1",
                "lang": y_lang,
            }
            url = str(y_url).rstrip("?") + "?" + urllib.parse.urlencode(q)
            _rate_limit_geocode()
            req = urllib.request.Request(url, headers={"User-Agent": "PhotoSorter/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:  # noqa: S310
                raw = resp.read()
            txt = raw.decode("utf-8", errors="replace")
            obj = json.loads(txt)
            comps = (
                obj.get("response", {})
                .get("GeoObjectCollection", {})
                .get("featureMember", [{}])[0]
                .get("GeoObject", {})
                .get("metaDataProperty", {})
                .get("GeocoderMetaData", {})
                .get("Address", {})
                .get("Components", [])
            )
            country = None
            city = None
            if isinstance(comps, list):
                for c in comps:
                    if not isinstance(c, dict):
                        continue
                    kind = str(c.get("kind") or "").strip().lower()
                    name = str(c.get("name") or "").strip()
                    if not name:
                        continue
                    if kind == "country" and not country:
                        country = name
                    if kind == "locality" and not city:
                        city = name
                    if kind == "province" and not city:
                        city = name
            return (country, city, txt)
        except Exception:
            return (None, None, None)
    qr = None
    # Важно: QR-детектор OpenCV иногда "подвисает" на отдельных изображениях (внутри нативного вызова),
    # из-за чего прогресс может застрять на 0/.. надолго. Поэтому по умолчанию отключаем QR-эвристику.
    if bool(enable_qr):
        try:
            qr = cv2.QRCodeDetector()
        except Exception:
            qr = None
    # YOLO (COCO) for cat detection: init lazily, only when faces==0.
    # Важно: используем yolov5s.onnx из релизов (URL стабильнее, чем assets), и он хорошо ловит кошек в профиль.
    yolo_model_path = Path("data") / "models" / "yolov5s.onnx"
    yolo_urls = [
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
    ]
    yolo_sess = None
    yolo_in = None
    yolo_err: str | None = None
    try:
        ensure_yolo_model(model_path=yolo_model_path, model_urls=yolo_urls)
    except Exception as e:  # noqa: BLE001
        yolo_err = f"{type(e).__name__}: {e}"
    try:
        v_samples = max(0, int(video_samples or 0))
        if v_samples > 3:
            v_samples = 3
        media_files: list[str] = []
        for p in _iter_files(root, exclude_dir_names=exclude_dir_names):
            if not (_is_image(p) or (v_samples > 0 and _is_video(p))):
                continue
            if exclude_paths is not None:
                try:
                    key = _as_local_path(os.path.normcase(p))
                    if key in exclude_paths:
                        continue
                except Exception:
                    # best-effort: если не смогли нормализовать путь — не исключаем
                    pass
            media_files.append(p)
        if run_id is None:
            run_id = store.create_run(scope="local", root_path=_as_local_path(root), total_files=len(media_files))
        run_id_i = int(run_id)
        def _rectangles_count_db() -> int:
            try:
                cur0 = store.conn.cursor()
                cur0.execute("SELECT COUNT(*) FROM face_rectangles WHERE run_id = ?", (int(run_id_i),))
                v = cur0.fetchone()
                return int(v[0] or 0) if v else 0
            except Exception:
                return 0
        if pipeline is not None and pipeline_run_id is not None:
            pipeline.update_run(run_id=int(pipeline_run_id), face_run_id=run_id_i)
        total_items = len(media_files)

        _dbg_log(
            hypothesis_id="H0",
            location="photosorter/pipeline/local_sort.py:scan_faces_local",
            message="scan_faces_local_start",
            data={
                "run_id": int(run_id_i),
                "pipeline_run_id": int(pipeline_run_id) if pipeline_run_id is not None else None,
                "root_dir": str(root_dir),
                "total_images": int(len([x for x in media_files if _is_image(x)])),
                "total_videos": int(len([x for x in media_files if _is_video(x)])),
                "video_samples": int(v_samples),
                "enable_qr": bool(qr is not None),
                "python_exe": str(getattr(sys, "executable", "")),
                "module_file": str(__file__),
            },
        )

        def _video_duration_seconds_from_cap(cap: cv2.VideoCapture) -> float | None:
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if fps > 0.0 and n > 0.0:
                    return float(n) / float(fps)
            except Exception:
                pass
            return None

        def _pick_video_times(dur: float, *, samples: int) -> list[float]:
            d = float(max(0.0, dur))
            if d <= 0.0:
                return [0.0]
            if d < 2.0 or samples <= 1:
                return [min(0.5 * d, max(0.0, d - 0.5))]
            t1 = min(max(1.0, 0.05 * d), max(0.0, d - 0.5))
            t2 = min(0.5 * d, max(0.0, d - 0.5))
            if samples == 2:
                return [t1, t2]
            t3 = min(0.95 * d, max(0.0, d - 0.5))
            # preserve order but unique-ish
            out: list[float] = []
            for t in (t1, t2, t3):
                if not out or abs(out[-1] - t) > 0.05:
                    out.append(t)
            return out

        def _is_bad_frame_bgr(frame_bgr: np.ndarray) -> bool:
            try:
                if frame_bgr is None or frame_bgr.size == 0:
                    return True
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                mean = float(np.mean(gray))
                std = float(np.std(gray))
                black_ratio = float(np.mean(gray <= int(video_black_luma_threshold)))
                if mean < float(video_bad_mean_luma):
                    return True
                if std < float(video_bad_std_luma):
                    return True
                if black_ratio > float(video_bad_black_ratio):
                    return True
                return False
            except Exception:
                return True

        def _read_video_frame_at(cap: cv2.VideoCapture, t_sec: float, *, max_dim_px: int) -> np.ndarray | None:
            try:
                # best-effort seek (depends on backend/codecs)
                cap.set(cv2.CAP_PROP_POS_MSEC, float(max(0.0, t_sec)) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    return None
                h, w = frame.shape[:2]
                m = max(int(h), int(w))
                if max_dim_px and m and m > int(max_dim_px):
                    sc = float(max_dim_px) / float(m)
                    nh = max(1, int(round(h * sc)))
                    nw = max(1, int(round(w * sc)))
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                return frame
            except Exception:
                return None

        for abspath in media_files:
            stats.images_scanned += 1  # "items scanned" (images + optional videos)
            db_path = _as_local_path(abspath)
            stage = "loop_start"

            if stats.images_scanned <= 3:
                _dbg_log(
                    hypothesis_id="H2",
                    location="photosorter/pipeline/local_sort.py:scan_faces_local",
                    message="scan_loop_begin",
                    data={"i": int(stats.images_scanned), "abspath": str(abspath), "db_path": str(db_path)},
                )

            # В самом начале прогресс обновляем ДО тяжёлой обработки изображения,
            # иначе UI может долго показывать 0/.. и выглядеть как "зависло" (особенно если первый файл проблемный).
            if stats.images_scanned <= 20:
                try:
                    stage = "early_progress_count"
                    faces_found_db0 = _rectangles_count_db()
                    # Resume-friendly: не даём прогрессу уменьшаться после рестарта.
                    # Иначе UI кратковременно "падает" на 1..20/total и выглядит как баг.
                    existing_proc = 0
                    try:
                        fr0 = store.get_run_by_id(run_id=int(run_id_i))
                        existing_proc = int((fr0 or {}).get("processed_files") or 0)
                    except Exception:
                        existing_proc = 0
                    proc_to_write = max(int(existing_proc), int(stats.images_scanned))
                    stage = "early_progress_update"
                    store.update_run_progress(
                        run_id=run_id_i,
                        processed_files=proc_to_write,
                        faces_found=faces_found_db0,
                        last_path=db_path,
                    )
                    _pipe_log(
                        pipeline,
                        pipeline_run_id,
                        f"faces_progress images={proc_to_write} total={total_items} "
                        f"faces={faces_found_db0} errors={stats.errors}\n",
                    )
                except Exception:
                    pass

            # --- store taken_at + gps + place (best-effort) ---
            try:
                taken_at: str | None = None
                # mtime fallback for all media
                try:
                    st = os.stat(abspath)
                    taken_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
                except Exception:
                    taken_at = None
                # Записываем mtime сразу, а EXIF/GPS (и геокодинг) сделаем позже внутри image_open,
                # чтобы не открывать файл дважды.
                dedup.set_taken_at_and_gps(path=db_path, taken_at=taken_at, gps_lat=None, gps_lon=None)
            except Exception:
                pass

            # Resume-friendly: если уже есть сводка лиц для этого файла под этим run_id — пропускаем.
            stage = "dedup_get_row"
            row = dedup.get_row_by_path(path=db_path)
            if stats.images_scanned <= 3:
                _dbg_log(
                    hypothesis_id="H1",
                    location="photosorter/pipeline/local_sort.py:scan_faces_local",
                    message="dedup_row_loaded",
                    data={"i": int(stats.images_scanned), "has_row": bool(row), "row_keys": sorted(list(row.keys())) if row else None},
                )
            if row and row.get("faces_run_id") is not None and int(row.get("faces_run_id")) == run_id_i and row.get("faces_scanned_at"):
                continue
            try:
                # --- VIDEO: sample 1–3 frames and aggregate ---
                if v_samples > 0 and _is_video(abspath):
                    stats.videos_scanned += 1
                    stage = "video_open"
                    cap = cv2.VideoCapture(abspath)
                    try:
                        if not cap.isOpened():
                            raise RuntimeError("video_open_failed")
                        dur = _video_duration_seconds_from_cap(cap) or 0.0
                        times = _pick_video_times(float(dur), samples=int(v_samples))
                        best_frame: np.ndarray | None = None
                        # pick first non-bad frame among candidates
                        for t in times:
                            fr = _read_video_frame_at(cap, float(t), max_dim_px=int(video_max_dim))
                            if fr is None:
                                continue
                            stats.video_frames_scanned += 1
                            if not _is_bad_frame_bgr(fr):
                                best_frame = fr
                                break
                            # keep the first frame as fallback
                            if best_frame is None:
                                best_frame = fr
                        if best_frame is None:
                            raise RuntimeError("no_video_frames")
                    finally:
                        try:
                            cap.release()
                        except Exception:
                            pass

                    # run detectors on chosen frame
                    stage = "video_detect_faces"
                    faces_det_v = _detect_faces(detector, best_frame)
                    # best-effort: second pass (so we don't miss obvious faces in video)
                    if not faces_det_v:
                        try:
                            detector2v = _create_face_detector(str(model), score_threshold=0.65)
                            faces_det_v2 = _detect_faces(detector2v, best_frame)
                            hv, wv = best_frame.shape[:2]
                            img_area_v = float(max(1, int(wv) * int(hv)))
                            faces_det_v2 = [
                                (x, y, w, h, s)
                                for (x, y, w, h, s) in faces_det_v2
                                if max(int(w), int(h)) >= 60 and (float(int(w) * int(h)) / img_area_v) >= 0.005
                            ]
                            if faces_det_v2:
                                faces_det_v = faces_det_v2
                        except Exception:
                            pass

                    faces_count_v = int(len(faces_det_v or []))

                    # For video we do NOT use screen_like quarantine.
                    is_quarantine = False
                    quarantine_reason = None

                    is_animal = False
                    animal_kind = None
                    if faces_count_v <= 0:
                        try:
                            if yolo_err is None:
                                if yolo_sess is None:
                                    import onnxruntime as ort  # type: ignore[import-untyped]

                                    yolo_sess = ort.InferenceSession(str(yolo_model_path), providers=["CPUExecutionProvider"])
                                    yolo_in = yolo_sess.get_inputs()[0].name
                                if yolo_sess is not None and yolo_in:
                                    stage = "video_detect_cat"
                                    is_animal = _detect_cat_yolo(
                                        sess=yolo_sess,
                                        input_name=str(yolo_in),
                                        img_bgr_full=best_frame,
                                        conf_threshold=0.35,
                                        nms_iou=0.45,
                                        min_box_area_ratio=0.004,
                                    )
                                    if is_animal:
                                        animal_kind = "cat"
                        except Exception:
                            pass

                    # update per-file aggregates in DB (no rectangles for video yet)
                    stage = "video_db_write"
                    try:
                        store.clear_run_auto_rectangles_for_file(run_id=run_id_i, file_path=db_path)
                    except Exception:
                        pass
                    try:
                        dedup.set_faces_summary(path=db_path, faces_run_id=run_id_i, faces_count=int(faces_count_v))
                        dedup.set_faces_auto_group(path=db_path, group=("many_faces" if int(faces_count_v) >= int(many_faces_min) else None))
                        dedup.set_faces_auto_quarantine(path=db_path, is_quarantine=False, reason=None)
                        dedup.set_animals_auto(path=db_path, is_animal=bool(is_animal and int(faces_count_v) == 0), kind=animal_kind)
                    except Exception:
                        pass

                    if faces_count_v > 0:
                        stats.faces_files += 1
                    else:
                        stats.no_faces_files += 1
                    continue

                # Важно: браузер показывает JPEG с учётом EXIF orientation.
                # Значит и детект, и сохранённые bbox должны быть в той же системе координат.
                stage = "image_open"
                with Image.open(abspath) as pil0:
                    # EXIF meta extraction (best-effort): taken_at + GPS + geocoding.
                    try:
                        exif_taken = _try_exif_datetime_iso(pil0)
                        gps = _try_exif_gps_latlon(pil0)
                        gps_lat = float(gps[0]) if gps else None
                        gps_lon = float(gps[1]) if gps else None
                        if exif_taken or (gps_lat is not None and gps_lon is not None):
                            dedup.set_taken_at_and_gps(path=db_path, taken_at=exif_taken, gps_lat=gps_lat, gps_lon=gps_lon)
                        if gps_lat is not None and gps_lon is not None and y_key:
                            k = _geo_cache_key(gps_lat, gps_lon)
                            cached = dedup.geocode_cache_get(key=k)
                            if cached and (cached.get("country") or cached.get("city")):
                                dedup.set_place(
                                    path=db_path,
                                    country=str(cached.get("country") or "") or None,
                                    city=str(cached.get("city") or "") or None,
                                    source=str(cached.get("source") or "yandex"),
                                )
                            else:
                                ctry, city, rawj = _yandex_country_city(gps_lat, gps_lon)
                                if ctry or city:
                                    dedup.geocode_cache_upsert(
                                        key=k,
                                        lat=gps_lat,
                                        lon=gps_lon,
                                        country=ctry,
                                        city=city,
                                        source="yandex",
                                        raw_json=rawj,
                                    )
                                    dedup.set_place(path=db_path, country=ctry, city=city, source="yandex")
                    except Exception:
                        pass
                    try:
                        pil = ImageOps.exif_transpose(pil0)
                    except Exception:
                        pil = pil0
                    pil = pil.convert("RGB")
                    try:
                        pil.load()
                    except Exception:
                        pass

                    stage = "np_convert"
                    img_rgb = np.asarray(pil)
                    # safety: гарантируем ожидаемый формат (H,W,3) uint8
                    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
                        raise RuntimeError(f"bad_image_shape:{getattr(img_rgb, 'shape', None)}")
                    if img_rgb.dtype != np.uint8:
                        img_rgb = img_rgb.astype(np.uint8, copy=False)
                    # RGB -> BGR
                    img_bgr_full = img_rgb[:, :, ::-1].copy()

                    # safety: ограничим размер (иногда именно на огромных кадрах detect() падает нативно)
                    stage = "resize_if_needed"
                    h0, w0 = img_bgr_full.shape[:2]
                    m = max(h0, w0)
                    scale = 1.0
                    img_bgr = img_bgr_full
                    if m > max_dim and m > 0:
                        scale = float(max_dim) / float(m)
                        nh = max(1, int(round(h0 * scale)))
                        nw = max(1, int(round(w0 * scale)))
                        img_bgr = cv2.resize(img_bgr_full, (nw, nh), interpolation=cv2.INTER_AREA)
                # периодически пересоздаём детектор (best-effort от нативных крэшей/утечек)
                if detector_recreate_every > 0 and (stats.images_scanned % detector_recreate_every == 0):
                    stage = "detector_recreate"
                    detector = _create_face_detector(str(model), score_threshold=float(score_threshold))

                stage = "detect_faces"
                faces_det = _detect_faces(detector, img_bgr)
                # Второй проход (только если 0 лиц): снижаем порог, чтобы уменьшить ложные "нет лиц".
                if not faces_det:
                    try:
                        # По диагностике на реальных "очевидных" селфи: thr=0.74 всё ещё даёт 0,
                        # а thr≈0.65 начинает стабильно находить лицо.
                        detector2 = _create_face_detector(str(model), score_threshold=0.65)
                        faces_det2 = _detect_faces(detector2, img_bgr)
                        # Фильтр от мусора: на втором проходе игнорируем совсем маленькие боксы.
                        h2, w2 = img_bgr.shape[:2]
                        img_area2 = float(max(1, int(w2) * int(h2)))
                        faces_det2 = [
                            (x, y, w, h, s)
                            for (x, y, w, h, s) in faces_det2
                            if max(int(w), int(h)) >= 60 and (float(int(w) * int(h)) / img_area2) >= 0.005
                        ]
                        if faces_det2:
                            faces_det = faces_det2
                    except Exception:
                        pass

                # Fallback: некоторые "перевёрнутые" фото не имеют EXIF orientation.
                # Если после базового + пониженного порога всё ещё 0 лиц — пробуем повороты 90/180/270.
                if not faces_det:
                    try:
                        # Используем тот же "мягкий" порог, что и на втором проходе.
                        detectorR = locals().get("detector2")
                        if detectorR is None:
                            detectorR = _create_face_detector(str(model), score_threshold=0.65)

                        hb, wb = img_bgr.shape[:2]
                        best: list[tuple[int, int, int, int, float]] = []
                        best_rot = 0
                        best_score = -1.0

                        for rot in (90, 180, 270):
                            img_rot = _rotate_bgr(img_bgr, rot=rot)
                            det = _detect_faces(detectorR, img_rot)
                            # тот же фильтр от мусора
                            hr, wr = img_rot.shape[:2]
                            img_area_r = float(max(1, int(wr) * int(hr)))
                            det = [
                                (x, y, w, h, s)
                                for (x, y, w, h, s) in det
                                if max(int(w), int(h)) >= 60 and (float(int(w) * int(h)) / img_area_r) >= 0.005
                            ]
                            if not det:
                                continue
                            max_s = max(float(s) for (_x, _y, _w, _h, s) in det)
                            if (len(det) > len(best)) or (len(det) == len(best) and max_s > best_score):
                                best = det
                                best_rot = rot
                                best_score = max_s

                        if best and best_rot:
                            # переводим bbox обратно в координаты base img_bgr (до поворота)
                            unrot: list[tuple[int, int, int, int, float]] = []
                            for (x, y, w, h, s) in best:
                                bx, by, bw, bh = _unrotate_bbox_to_base(x=int(x), y=int(y), w=int(w), h=int(h), rot=int(best_rot), w0=int(wb), h0=int(hb))
                                if bw > 1 and bh > 1:
                                    unrot.append((bx, by, bw, bh, float(s)))
                            if unrot:
                                faces_det = unrot
                    except Exception:
                        pass
                # Если детектили на уменьшенной копии — пересчитываем bbox обратно к размерам полного изображения.
                faces: list[tuple[int, int, int, int, float]] = []
                if scale != 1.0 and scale > 0.0:
                    inv = 1.0 / float(scale)
                    for (x, y, w, h, s) in faces_det:
                        xi = max(0, int(round(float(x) * inv)))
                        yi = max(0, int(round(float(y) * inv)))
                        wi = max(0, int(round(float(w) * inv)))
                        hi = max(0, int(round(float(h) * inv)))
                        # clip к размерам полного изображения
                        if xi >= w0 or yi >= h0:
                            continue
                        if xi + wi > w0:
                            wi = max(0, w0 - xi)
                        if yi + hi > h0:
                            hi = max(0, h0 - yi)
                        if wi <= 1 or hi <= 1:
                            continue
                        faces.append((xi, yi, wi, hi, float(s)))
                else:
                    faces = faces_det
                areas = [float(w * h) for (_x, _y, w, h, _s) in faces]
                denom = float(sum(areas)) if areas else 0.0

                # --- auto quarantine / animals ---
                is_quarantine = False
                quarantine_reason: str | None = None
                is_animal = False
                animal_kind: str | None = None
                cat_overrides_faces = False
                try:
                    # 1) QR code -> карантин
                    qr_found = False
                    if qr is not None:
                        qr_found = _qr_found_best_effort(img_bgr_full, qr=qr, qr_max_dim=qr_max_dim)
                    if qr_found:
                        is_quarantine = True
                        quarantine_reason = "qr"
                    else:
                        # ВАЖНО: для кадров БЕЗ лиц сначала пробуем найти кошку (MVP),
                        # и только если кошки нет — применяем эвристику screen_like.
                        # Это критично для "кот ↔ нет лиц": многие коты выглядят как "скрин" (мессенджер/галерея),
                        # и раньше из-за раннего screen_like мы вообще не запускали YOLO на кота.
                        if not faces:
                            try:
                                if yolo_err is None:
                                    if yolo_sess is None:
                                        import onnxruntime as ort  # type: ignore[import-untyped]

                                        yolo_sess = ort.InferenceSession(
                                            str(yolo_model_path),
                                            providers=["CPUExecutionProvider"],
                                        )
                                        yolo_in = yolo_sess.get_inputs()[0].name
                                    if yolo_sess is not None and yolo_in:
                                        is_animal = _detect_cat_yolo(
                                            sess=yolo_sess,
                                            input_name=str(yolo_in),
                                            img_bgr_full=img_bgr_full,
                                            conf_threshold=0.35,
                                            nms_iou=0.45,
                                            min_box_area_ratio=0.004,  # ~0.4% кадра, чтобы не ловить мелкие картинки/иконки
                                        )
                                        if is_animal:
                                            animal_kind = "cat"
                                            # если это кот — не считаем это screen_like карантином
                                            is_quarantine = False
                                            quarantine_reason = None
                            except Exception:
                                pass

                        # screen_like: применяем только если лиц НЕТ и кота тоже НЕТ,
                        # иначе слишком много ложных карантинов на обычных фото/котах.
                        if (not faces) and (not is_animal):
                            try:
                                screen_like = _looks_like_screen_ui(img_bgr_full)
                            except Exception:
                                screen_like = False
                            if screen_like:
                                is_quarantine = True
                                quarantine_reason = "screen_like"

                        # 2) Строгий "tiny_face" (качество лица слишком низкое):
                        # карантин только если лицо реально крошечное (чтобы не вернуть старые ложные карантины).
                        # Подходит для случаев "размытый маленький кусочек лица".
                        if faces:
                            max_face_dim = max(int(max(w, h)) for (_x, _y, w, h, _s) in faces)
                            max_area = max(float(w * h) for (_x, _y, w, h, _s) in faces) if faces else 0.0
                            area_ratio = (max_area / float(w0 * h0)) if (w0 > 0 and h0 > 0) else 0.0
                            if max_face_dim <= 32 and area_ratio <= 0.0012:
                                is_quarantine = True
                                quarantine_reason = "tiny_face"
                        # 3b) Если YuNet нашёл "лица", но это может быть кошачья морда:
                        # запускаем YOLO для cat/person и, если cat==True и person==False — считаем это животным, а не лицом.
                        if faces and (len(faces) <= 2):
                            try:
                                if yolo_err is None:
                                    if yolo_sess is None:
                                        import onnxruntime as ort  # type: ignore[import-untyped]

                                        yolo_sess = ort.InferenceSession(
                                            str(yolo_model_path),
                                            providers=["CPUExecutionProvider"],
                                        )
                                        yolo_in = yolo_sess.get_inputs()[0].name
                                    if yolo_sess is not None and yolo_in:
                                        has_cat = _detect_class_yolo(
                                            sess=yolo_sess,
                                            input_name=str(yolo_in),
                                            img_bgr_full=img_bgr_full,
                                            class_idx=15,  # COCO cat
                                            conf_threshold=0.35,
                                            nms_iou=0.45,
                                            min_box_area_ratio=0.002,
                                        )
                                        has_person = _detect_class_yolo(
                                            sess=yolo_sess,
                                            input_name=str(yolo_in),
                                            img_bgr_full=img_bgr_full,
                                            class_idx=0,  # COCO person
                                            conf_threshold=0.35,
                                            nms_iou=0.45,
                                            min_box_area_ratio=0.002,
                                        )
                                        if has_cat and not has_person:
                                            is_animal = True
                                            animal_kind = "cat"
                                            cat_overrides_faces = True
                                            # Считаем, что "лица" на самом деле не люди -> обнулим позже при записи.
                            except Exception:
                                pass
                except Exception:
                    # best-effort: не ломаем конвейер из-за эвристик
                    is_quarantine = False
                    quarantine_reason = None
                    is_animal = False
                    animal_kind = None
                    cat_overrides_faces = False

                # Важно: при пересканах не затираем ручную разметку.
                stage = "db_clear_auto_rects"
                store.clear_run_auto_rectangles_for_file(run_id=run_id_i, file_path=db_path)

                # Если YOLO подтвердил кота и не нашёл человека — не считаем YuNet-детект "лицами".
                if cat_overrides_faces:
                    faces = []

                if faces:
                    stats.faces_files += 1
                else:
                    stats.no_faces_files += 1

                for i, (x, y, w, h, score) in enumerate(faces):
                    pres = (areas[i] / denom) if (denom > 0.0 and i < len(areas)) else None
                    thumb = _crop_thumb_jpeg(img=pil, bbox=(x, y, w, h), thumb_size=int(thumb_size))
                    stage = "db_insert_detection"
                    store.insert_detection(
                        run_id=run_id_i,
                        file_path=db_path,
                        face_index=i,
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                        confidence=score,
                        presence_score=pres,
                        thumb_jpeg=thumb,
                    )
                    stats.faces_found += 1

                # Пишем сводку "лица/не лица" в files так, чтобы faces_count совпадал
                # с количеством прямоугольников (auto+manual) в face_rectangles.
                stage = "db_faces_count_recompute"
                try:
                    cur1 = store.conn.cursor()
                    cur1.execute(
                        """
                        SELECT COUNT(*) FROM face_rectangles
                        WHERE run_id = ? AND file_path = ? AND COALESCE(ignore_flag, 0) = 0
                        """,
                        (int(run_id_i), db_path),
                    )
                    v = cur1.fetchone()
                    faces_count_db_file = int(v[0] or 0) if v else int(len(faces))
                except Exception:
                    faces_count_db_file = int(len(faces))

                stage = "db_set_faces_summary"
                dedup.set_faces_summary(path=db_path, faces_run_id=run_id_i, faces_count=int(faces_count_db_file))

                # 2-й уровень (variant 2): авто-группа "many_faces" НЕ через карантин.
                stage = "db_set_faces_auto_group"
                try:
                    dedup.set_faces_auto_group(
                        path=db_path,
                        group=("many_faces" if int(faces_count_db_file) >= int(many_faces_min) else None),
                    )
                except Exception:
                    pass
                # Пишем авто-карантин (если включился)
                try:
                    dedup.set_faces_auto_quarantine(path=db_path, is_quarantine=bool(is_quarantine), reason=quarantine_reason)
                except Exception:
                    pass
                # Пишем авто-животных (только когда нет лиц)
                try:
                    dedup.set_animals_auto(path=db_path, is_animal=bool(is_animal and (int(faces_count_db_file) == 0)), kind=animal_kind)
                except Exception:
                    pass

                # Прогресс: после первых 20 (которые апдейтим до обработки) продолжаем апдейты пачками.
                step = 10 if stats.images_scanned <= 200 else 50
                if stats.images_scanned > 20 and stats.images_scanned % step == 0:
                    # Важно для resume: не затираем счётчик лиц нулём, если файлы были пропущены как уже просканированные.
                    faces_found_db = _rectangles_count_db()
                    store.update_run_progress(
                        run_id=run_id_i,
                        processed_files=stats.images_scanned,
                        faces_found=faces_found_db,
                        last_path=db_path,
                    )
                    _pipe_log(
                        pipeline,
                        pipeline_run_id,
                        f"faces_progress images={stats.images_scanned} total={total_items} "
                        f"faces={faces_found_db} errors={stats.errors}\n",
                    )
            except Exception as e:  # noqa: BLE001
                store.update_run_progress(run_id=run_id_i, last_path=db_path, last_error=f"{type(e).__name__}: {e}")
                stats.errors += 1
                if stats.images_scanned <= 50:
                    _dbg_log(
                        hypothesis_id="H1",
                        location="photosorter/pipeline/local_sort.py:scan_faces_local",
                        message="scan_loop_exception",
                        data={
                            "i": int(stats.images_scanned),
                            "stage": str(stage),
                            "exc_type": type(e).__name__,
                            "exc_msg": str(e)[:400],
                            "db_path": str(db_path),
                            "abspath": str(abspath),
                        },
                    )

        faces_found_db = _rectangles_count_db()
        stats.faces_found = faces_found_db
        store.update_run_progress(run_id=run_id_i, processed_files=stats.images_scanned, faces_found=faces_found_db)
        store.finish_run(run_id=run_id_i, status="completed")
        return run_id_i, stats
    finally:
        dedup.close()
        store.close()


def sort_by_faces(
    *,
    root_dir: str,
    run_id: int,
    dry_run: bool,
    faces_dirname: str,
    faces_quarantine_dirname: str,
    faces_people_no_face_dirname: str,
    animals_dirname: str,
    no_faces_dirname: str,
    exclude_dir_names: tuple[str, ...],
    pipeline: PipelineStore | None = None,
    pipeline_run_id: int | None = None,
) -> Stats:
    stats = Stats()
    root = os.path.abspath(root_dir)

    store = FaceStore()
    dedup = DedupStore()
    try:
        # Берём список из files по faces_run_id — чтобы включить и "0 лиц".
        cur = dedup.conn.cursor()
        cur.execute(
            """
            SELECT
              path,
              COALESCE(faces_count, 0) AS faces_count,
              COALESCE(faces_manual_label, '') AS faces_manual_label,
              COALESCE(faces_auto_quarantine, 0) AS faces_auto_quarantine,
              COALESCE(animals_auto, 0) AS animals_auto,
              COALESCE(animals_kind, '') AS animals_kind,
              COALESCE(people_no_face_manual, 0) AS people_no_face_manual
            FROM files
            WHERE
              faces_run_id = ?
              AND path LIKE 'local:%'
              AND status != 'deleted'
            ORDER BY path ASC
            """,
            (int(run_id),),
        )
        rows = [
            (
                str(r[0]),
                int(r[1]),
                str(r[2] or ""),
                int(r[3] or 0),
                int(r[4] or 0),
                str(r[5] or ""),
                int(r[6] or 0),
            )
            for r in cur.fetchall()
        ]

        for file_path, cnt_auto, manual_label, auto_quarantine, animals_auto, animals_kind, people_no_face_manual in rows:
            src_abs = _strip_local_prefix(file_path)
            if not os.path.exists(src_abs):
                continue
            # safety: skip if under exclude dirs
            rel = os.path.relpath(src_abs, root)
            rel_parts = Path(rel).parts
            if rel_parts and rel_parts[0] in set(exclude_dir_names):
                continue

            # Эффективная категория (manual приоритетнее):
            # 0=no_faces, 1=quarantine, 2=faces, 3=animals, 4=people_no_face
            cat = 2 if int(cnt_auto) > 0 else 0
            ml = (manual_label or "").strip().lower()
            if ml == "faces":
                cat = 2
            elif ml == "no_faces":
                cat = 0
            else:
                if int(people_no_face_manual or 0) == 1:
                    cat = 4
                elif int(cnt_auto) <= 0 and int(animals_auto or 0) == 1 and (animals_kind or "").strip():
                    cat = 3
                if int(auto_quarantine or 0) == 1 and int(cnt_auto) > 0:
                    cat = 1

            if cat == 2:
                target_root = faces_dirname
            elif cat == 1:
                target_root = os.path.join(faces_dirname, faces_quarantine_dirname)
            elif cat == 4:
                target_root = os.path.join(faces_dirname, faces_people_no_face_dirname)
            elif cat == 3:
                target_root = animals_dirname
            else:
                target_root = no_faces_dirname
            dst_abs = os.path.join(root, target_root, rel)

            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path=src_abs, last_dst_path=dst_abs)
            _move_file(src=src_abs, dst=dst_abs, dry_run=dry_run)

            # update DB paths (files + face_rectangles) ONLY when не dry-run
            if not dry_run:
                new_db_path = _as_local_path(dst_abs)
                dedup.update_path(
                    old_path=file_path,
                    new_path=new_db_path,
                    new_name=os.path.basename(dst_abs),
                    new_parent_path=_as_local_path(os.path.dirname(dst_abs)),
                )
                if pipeline_run_id is not None:
                    try:
                        dedup.update_run_manual_labels_path(
                            pipeline_run_id=int(pipeline_run_id),
                            old_path=file_path,
                            new_path=new_db_path,
                        )
                    except Exception:
                        pass
                store.update_file_path(old_file_path=file_path, new_file_path=new_db_path)
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path="", last_dst_path="")

            if cat == 2:
                stats.moved_faces += 1
            elif cat == 1:
                stats.moved_quarantine += 1
            elif cat == 3:
                stats.moved_animals += 1
            elif cat == 4:
                stats.moved_people_no_face += 1
            else:
                stats.moved_no_faces += 1
    finally:
        dedup.close()
        store.close()

    return stats


def _sanitize_segment(s: str) -> str:
    out = []
    for ch in (s or "").strip():
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            out.append(ch)
        else:
            out.append("_")
    seg = "".join(out).strip().strip(".")
    seg = seg.replace(" ", "_")
    return seg or "unknown"


def _try_exif_datetime_year(img: Image.Image) -> Optional[str]:
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    # DateTimeOriginal = 36867
    dt = exif.get(36867) or exif.get(306)
    if not dt or not isinstance(dt, str):
        return None
    # "YYYY:MM:DD HH:MM:SS"
    if len(dt) >= 4 and dt[:4].isdigit():
        return dt[:4]
    return None


def _try_exif_datetime_iso(img: Image.Image) -> Optional[str]:
    """
    Returns ISO-like timestamp 'YYYY-MM-DDTHH:MM:SSZ' (best-effort).
    EXIF DateTimeOriginal does not contain timezone; we still suffix 'Z' for stable format/sorting.
    """
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    dt = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime
    if not dt or not isinstance(dt, str):
        return None
    s = dt.strip()
    # "YYYY:MM:DD HH:MM:SS"
    try:
        if len(s) >= 19 and s[4] == ":" and s[7] == ":" and s[10] in (" ", "T"):
            yyyy = s[0:4]
            mm = s[5:7]
            dd = s[8:10]
            hh = s[11:13]
            mi = s[14:16]
            ss = s[17:19]
            if yyyy.isdigit() and mm.isdigit() and dd.isdigit() and hh.isdigit() and mi.isdigit() and ss.isdigit():
                return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}Z"
    except Exception:
        return None
    return None


def _gps_to_deg(v) -> Optional[float]:
    try:
        d = float(v[0])
        m = float(v[1])
        s = float(v[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def _try_exif_gps_segment(img: Image.Image) -> Optional[str]:
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    # Получаем GPS IFD через get_ifd (правильный способ в Pillow)
    try:
        gps = exif.get_ifd(0x8825)  # GPS IFD tag (34853 в десятичной системе)
    except Exception:
        # Fallback: пробуем через get
        gps = exif.get(34853)
    if not gps:
        return None
    # gps is a dict-like with numeric keys (Pillow)
    try:
        lat = gps.get(2)
        lat_ref = gps.get(1)
        lon = gps.get(4)
        lon_ref = gps.get(3)
    except Exception:
        return None
    if not lat or not lon:
        return None
    lat_deg = _gps_to_deg(lat)
    lon_deg = _gps_to_deg(lon)
    if lat_deg is None or lon_deg is None:
        return None
    if isinstance(lat_ref, str) and lat_ref.upper() == "S":
        lat_deg = -lat_deg
    if isinstance(lon_ref, str) and lon_ref.upper() == "W":
        lon_deg = -lon_deg
    return f"gps_{lat_deg:.4f}_{lon_deg:.4f}"


def _try_exif_gps_latlon(img: Image.Image) -> tuple[float, float] | None:
    """
    Returns (lat, lon) in degrees (best-effort), applying N/S/E/W refs.
    """
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    # Получаем GPS IFD через get_ifd (правильный способ в Pillow)
    try:
        gps = exif.get_ifd(0x8825)  # GPS IFD tag (34853 в десятичной системе)
    except Exception:
        # Fallback: пробуем через get
        gps = exif.get(34853)
    if not gps:
        return None
    try:
        lat = gps.get(2)      # GPSLatitude
        lat_ref = gps.get(1)   # GPSLatitudeRef: 'N' or 'S'
        lon = gps.get(4)       # GPSLongitude
        lon_ref = gps.get(3)   # GPSLongitudeRef: 'E' or 'W'
    except Exception:
        return None
    if not lat or not lon:
        return None
    lat_deg = _gps_to_deg(lat)
    lon_deg = _gps_to_deg(lon)
    if lat_deg is None or lon_deg is None:
        return None
    if isinstance(lat_ref, str) and lat_ref.upper() == "S":
        lat_deg = -lat_deg
    if isinstance(lon_ref, str) and lon_ref.upper() == "W":
        lon_deg = -lon_deg
    return (float(lat_deg), float(lon_deg))


def sort_faces_into_named_folders(
    *,
    root_dir: str,
    run_id: int,
    dry_run: bool,
    faces_dirname: str,
    exclude_dir_names: tuple[str, ...],
    pipeline: PipelineStore | None = None,
    pipeline_run_id: int | None = None,
) -> Stats:
    """
    Шаг 3: внутри faces раскладываем по папкам с именами.

    MVP-логика: используем ручные метки `manual_person` в face_rectangles.
    Если меток нет -> _unassigned.
    """
    stats = Stats()
    root = os.path.abspath(root_dir)
    faces_root = os.path.join(root, faces_dirname)
    store = FaceStore()
    dedup = DedupStore()
    try:
        # Работаем ТОЛЬКО внутри _faces: это делает шаг идемпотентным и не конфликтует с exclude_dir_names.
        if not os.path.isdir(faces_root):
            return stats

        def _people_for_file(db_path: str) -> list[str]:
            cur = store.conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT manual_person
                FROM face_rectangles
                WHERE run_id = ? AND file_path = ? AND manual_person IS NOT NULL AND TRIM(manual_person) != ''
                """,
                (int(run_id), str(db_path)),
            )
            return sorted({str(r[0]).strip() for r in cur.fetchall() if r and r[0]})

        for dirpath, _dirnames, filenames in os.walk(faces_root):
            for fn in filenames:
                src_abs = os.path.join(dirpath, fn)
                if not os.path.exists(src_abs):
                    continue
                rel_inside = os.path.relpath(src_abs, faces_root)
                db_path = _as_local_path(src_abs)
                people = _people_for_file(db_path)
            if len(people) == 1:
                person = _sanitize_segment(people[0])
                desired_prefix = person
                stats.moved_faces_named += 1
            elif len(people) == 0:
                desired_prefix = "_unassigned"
                stats.moved_faces_unassigned += 1
            else:
                desired_prefix = "_mixed"
                stats.moved_faces_unassigned += 1

            # Идемпотентность: если файл уже лежит в нужной подпапке — ничего не делаем (не создаём вложенность).
            if rel_inside == desired_prefix or rel_inside.startswith(desired_prefix + os.sep):
                continue

            dst_abs = os.path.join(faces_root, desired_prefix, rel_inside)
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path=src_abs, last_dst_path=dst_abs)
            _move_file(src=src_abs, dst=dst_abs, dry_run=dry_run)

            if not dry_run:
                new_db_path = _as_local_path(dst_abs)
                dedup.update_path(
                    old_path=db_path,
                    new_path=new_db_path,
                    new_name=os.path.basename(dst_abs),
                    new_parent_path=_as_local_path(os.path.dirname(dst_abs)),
                )
                if pipeline_run_id is not None:
                    try:
                        dedup.update_run_manual_labels_path(
                            pipeline_run_id=int(pipeline_run_id),
                            old_path=db_path,
                            new_path=new_db_path,
                        )
                    except Exception:
                        pass
                store.update_file_path(old_file_path=db_path, new_file_path=new_db_path)
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path="", last_dst_path="")
    finally:
        dedup.close()
        store.close()
    return stats


def sort_no_faces_by_geo_year(
    *,
    root_dir: str,
    run_id: int,
    dry_run: bool,
    no_faces_dirname: str,
    exclude_dir_names: tuple[str, ...],
    pipeline: PipelineStore | None = None,
    pipeline_run_id: int | None = None,
) -> Stats:
    """
    Шаг 4: внутри no-faces раскладываем по (локация + год).

    MVP:
    - год: EXIF DateTimeOriginal, иначе mtime
    - локация: EXIF GPS -> gps_lat_lon, иначе unknown
    """
    stats = Stats()
    root = os.path.abspath(root_dir)
    no_faces_root = os.path.join(root, no_faces_dirname)
    dedup = DedupStore()
    store = FaceStore()
    try:
        # Работаем ТОЛЬКО внутри _no_faces: шаг идемпотентный и не ломается exclude_dir_names.
        if not os.path.isdir(no_faces_root):
            return stats

        for dirpath, _dirnames, filenames in os.walk(no_faces_root):
            for fn in filenames:
                src_abs = os.path.join(dirpath, fn)
                if not os.path.exists(src_abs):
                    continue
                rel_inside = os.path.relpath(src_abs, no_faces_root)

            year: Optional[str] = None
            loc: Optional[str] = None
            if _is_image(src_abs):
                try:
                    img = Image.open(src_abs)
                    year = _try_exif_datetime_year(img)
                    loc = _try_exif_gps_segment(img)
                except Exception:
                    pass

            if not year:
                try:
                    st = os.stat(src_abs)
                    year = time.strftime("%Y", time.gmtime(st.st_mtime))
                except OSError:
                    year = "unknown"
            if not loc:
                loc = "unknown"

            loc_seg = _sanitize_segment(loc)
            year_seg = _sanitize_segment(year)
            desired_prefix = os.path.join(loc_seg, year_seg)
            # Идемпотентность: если уже в правильной подпапке — не перемещаем снова.
            if rel_inside == desired_prefix or rel_inside.startswith(desired_prefix + os.sep):
                continue

            dst_abs = os.path.join(no_faces_root, desired_prefix, rel_inside)
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path=src_abs, last_dst_path=dst_abs)
            _move_file(src=src_abs, dst=dst_abs, dry_run=dry_run)

            if not dry_run:
                new_db_path = _as_local_path(dst_abs)
                dedup.update_path(
                    old_path=_as_local_path(src_abs),
                    new_path=new_db_path,
                    new_name=os.path.basename(dst_abs),
                    new_parent_path=_as_local_path(os.path.dirname(dst_abs)),
                )
                if pipeline_run_id is not None:
                    try:
                        dedup.update_run_manual_labels_path(
                            pipeline_run_id=int(pipeline_run_id),
                            old_path=_as_local_path(src_abs),
                            new_path=new_db_path,
                        )
                    except Exception:
                        pass
                store.update_file_path(old_file_path=_as_local_path(src_abs), new_file_path=new_db_path)
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path="", last_dst_path="")
            stats.moved_no_faces_by_geo_year += 1
    finally:
        store.close()
        dedup.close()
    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description="Local folder pipeline: 1) dedup 2) faces/no-faces (scan + split)"
    )
    p.add_argument("--root", required=True, help="Local folder root, e.g. C:\\\\Photos")
    p.add_argument("--apply", action="store_true", help="Apply moves (default: dry-run)")
    p.add_argument("--skip-dedup", action="store_true", help="Skip step 1 (dedup) and use existing inventory")
    p.add_argument("--pipeline-run-id", type=int, default=None, help="Pipeline run id in SQLite (for resume)")
    p.add_argument("--no-dedup-move", action="store_true", help="Do NOT move duplicates into _duplicates")
    p.add_argument("--enable-qr", action="store_true", help="Enable QR-based quarantine heuristic (disabled by default)")
    p.add_argument("--duplicates-dirname", default="_duplicates")
    p.add_argument("--faces-dirname", default="_faces")
    p.add_argument("--faces-quarantine-dirname", default="_quarantine")
    p.add_argument("--faces-people-no-face-dirname", default="_people_no_face")
    p.add_argument("--animals-dirname", default="_animals")
    p.add_argument("--no-faces-dirname", default="_no_faces")
    p.add_argument("--exclude-dirname", action="append", default=list(EXCLUDE_DIR_NAMES_DEFAULT))
    p.add_argument("--score-threshold", type=float, default=0.85)
    p.add_argument("--thumb-size", type=int, default=160)
    # Video (optional): sample 0..3 frames and run the same detectors.
    # Default 0 => do not process videos at all (keeps existing behavior and avoids slowing down runs).
    p.add_argument("--video-samples", type=int, default=0, help="0..3 frames per video (default 0 = disabled)")
    p.add_argument("--video-max-dim", type=int, default=640, help="Max frame dimension for video sampling (px)")
    p.add_argument("--video-bad-mean-luma", type=float, default=25.0, help="Reject frame if mean luma < threshold")
    p.add_argument("--video-bad-std-luma", type=float, default=10.0, help="Reject frame if std(luma) < threshold")
    p.add_argument("--video-bad-black-ratio", type=float, default=0.85, help="Reject frame if black pixel ratio > threshold")
    p.add_argument("--video-black-luma-threshold", type=int, default=16, help="Luma <= this counts as 'black' for ratio")
    p.add_argument(
        "--model-path",
        default=str(Path("data") / "models" / "face_detection_yunet_2023mar.onnx"),
    )
    p.add_argument(
        "--model-url",
        default="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    )
    args = p.parse_args()

    root = str(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Root dir not found: {root}")

    dry_run = not bool(args.apply)
    exclude = tuple(dict.fromkeys([str(x) for x in args.exclude_dirname]))  # preserve order, unique

    pipeline: PipelineStore | None = None
    pipeline_run_id: int | None = int(args.pipeline_run_id) if args.pipeline_run_id else None
    if pipeline_run_id is not None:
        pipeline = PipelineStore()
        try:
            pr = pipeline.get_run_by_id(run_id=pipeline_run_id)
            if not pr:
                raise SystemExit(f"pipeline_run_id not found in DB: {pipeline_run_id}")
            if str(pr.get("status")) == "completed":
                _pipe_log(pipeline, pipeline_run_id, f"Pipeline already completed: run_id={pipeline_run_id}\n")
                return 0
            pipeline.update_run(run_id=pipeline_run_id, status="running", last_error="")
        except Exception:
            if pipeline is not None:
                pipeline.close()
            raise

        # Если упали между move и update_path — добиваем последнюю транзакцию (best-effort).
        # В dry-run мы не двигаем файлы и не должны менять пути в БД.
        try:
            if dry_run:
                raise RuntimeError("skip_fixup_in_dry_run")
            src0 = str(pr.get("last_src_path") or "")
            dst0 = str(pr.get("last_dst_path") or "")
            if src0 and dst0 and (not os.path.exists(src0)) and os.path.exists(dst0):
                d = DedupStore()
                f = FaceStore()
                try:
                    d.update_path(
                        old_path=_as_local_path(src0),
                        new_path=_as_local_path(dst0),
                        new_name=os.path.basename(dst0),
                        new_parent_path=_as_local_path(os.path.dirname(dst0)),
                    )
                    try:
                        d.update_run_manual_labels_path(
                            pipeline_run_id=int(pipeline_run_id),
                            old_path=_as_local_path(src0),
                            new_path=_as_local_path(dst0),
                        )
                    except Exception:
                        pass
                    f.update_file_path(old_file_path=_as_local_path(src0), new_file_path=_as_local_path(dst0))
                    pipeline.update_run(run_id=pipeline_run_id, last_src_path="", last_dst_path="")
                finally:
                    d.close()
                    f.close()
        except Exception:
            pass

    _pipe_log(pipeline, pipeline_run_id, f"DRY_RUN={dry_run} root={os.path.abspath(root)}\n")

    try:
        # resume point
        start_step = 1
        existing_dedup_run_id: int | None = None
        existing_face_run_id: int | None = None
        if pipeline is not None and pipeline_run_id is not None:
            pr2 = pipeline.get_run_by_id(run_id=pipeline_run_id) or {}
            start_step = int(pr2.get("step_num") or 1)
            # backward compat: ранее было 4 шага (2=scan,3=split,4=inside groups). Сейчас шагов 2.
            if start_step > 2:
                start_step = 2
            existing_dedup_run_id = int(pr2["dedup_run_id"]) if pr2.get("dedup_run_id") else None
            existing_face_run_id = int(pr2["face_run_id"]) if pr2.get("face_run_id") else None

        # 1) предочистка (до дедупликации): не‑медиа и битые медиа уводим в служебные папки
        step0_counts: dict[str, int] | None = None
        if start_step <= 1:
            if pipeline is not None and pipeline_run_id is not None:
                # Внутренне у конвейера пока 2 шага (dedup + faces), но в UI мы показываем план 1..6.
                pipeline.update_run(run_id=pipeline_run_id, step_num=1, step_total=2, step_title="предочистка")
            _pipe_log(pipeline, pipeline_run_id, "\n== шаг 1: предочистка (не-медиа / битые медиа) ==\n")
            pre = _preclean_nonmedia_and_broken(
                root_dir=root,
                dry_run=dry_run,
                exclude_dir_names=exclude,
                non_media_dirname=NON_MEDIA_DIRNAME_DEFAULT,
                broken_media_dirname=BROKEN_MEDIA_DIRNAME_DEFAULT,
                pipeline=pipeline,
                pipeline_run_id=pipeline_run_id,
            )
            step0_counts = {k: int(pre.get(k) or 0) for k in ("checked", "moved_non_media", "moved_broken_media")}
            _pipe_log(
                pipeline,
                pipeline_run_id,
                f"preclean: checked={pre.get('checked')} moved_non_media={pre.get('moved_non_media')} moved_broken_media={pre.get('moved_broken_media')}\n",
            )

        # В DRY-RUN (и вообще для консистентности) считаем "эффективный" набор файлов после предочистки:
        # исключаем всё, что было помечено к переносу в _non_media/_broken_media.
        preclean_exclude_paths: set[str] | None = None
        try:
            if pipeline is not None and pipeline_run_id is not None:
                ex = _load_preclean_exclude_paths(pipeline=pipeline, pipeline_run_id=pipeline_run_id)
                preclean_exclude_paths = ex if ex else None
        except Exception:
            preclean_exclude_paths = None

        # 2) дедупликация
        if start_step <= 1:
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=pipeline_run_id, step_num=1, step_total=2, step_title="дедупликация")
            if args.skip_dedup:
                _pipe_log(pipeline, pipeline_run_id, "\n== шаг 2: дедупликация (SKIPPED) ==\n")
            else:
                _pipe_log(pipeline, pipeline_run_id, "\n== шаг 2: дедупликация ==\n")
                d = dedup_local(
                    root_dir=root,
                    dry_run=dry_run,
                    move_duplicates=not bool(args.no_dedup_move),
                    duplicates_dirname=str(args.duplicates_dirname),
                    exclude_dir_names=exclude,
                    exclude_paths=preclean_exclude_paths,
                    run_id=existing_dedup_run_id,
                    pipeline=pipeline,
                    pipeline_run_id=pipeline_run_id,
                )
                _pipe_log(
                    pipeline,
                    pipeline_run_id,
                    f"dedup: total={d.total_files} processed={d.processed} hashed={d.hashed} "
                    f"errors={d.errors} duplicates_moved={d.duplicates_moved}\n",
                )

        # 3) лица / нет лиц = скан + разложение
        if pipeline is not None and pipeline_run_id is not None:
            pipeline.update_run(run_id=pipeline_run_id, step_num=2, step_total=2, step_title="лица/нет лиц: скан")
        _pipe_log(pipeline, pipeline_run_id, "\n== шаг 3: лица / нет лиц ==\n")
        _pipe_log(pipeline, pipeline_run_id, "подшаг 3.1: скан лиц (face rectangles)\n")
        run_id2, s = scan_faces_local(
            root_dir=root,
            score_threshold=float(args.score_threshold),
            thumb_size=int(args.thumb_size),
            model_path=Path(str(args.model_path)),
            model_url=str(args.model_url),
            exclude_dir_names=exclude,
            exclude_paths=preclean_exclude_paths,
            enable_qr=bool(args.enable_qr),
            video_samples=int(args.video_samples or 0),
            video_max_dim=int(args.video_max_dim or 0),
            video_bad_mean_luma=float(args.video_bad_mean_luma or 0.0),
            video_bad_std_luma=float(args.video_bad_std_luma or 0.0),
            video_bad_black_ratio=float(args.video_bad_black_ratio or 0.0),
            video_black_luma_threshold=int(args.video_black_luma_threshold or 0),
            run_id=existing_face_run_id,
            pipeline=pipeline,
            pipeline_run_id=pipeline_run_id,
        )
        _pipe_log(
            pipeline,
            pipeline_run_id,
            f"faces: run_id={run_id2} images={s.images_scanned} faces_files={s.faces_files} "
            f"no_faces_files={s.no_faces_files} faces={s.faces_found} errors={s.errors}\n",
        )

        if pipeline is not None and pipeline_run_id is not None:
            pipeline.update_run(run_id=pipeline_run_id, step_num=2, step_total=2, step_title="лица/нет лиц: разложение")
        _pipe_log(pipeline, pipeline_run_id, "подшаг 2.2: разложение файлов в _faces / _no_faces\n")
        r = sort_by_faces(
            root_dir=root,
            run_id=run_id2,
            dry_run=dry_run,
            faces_dirname=str(args.faces_dirname),
            faces_quarantine_dirname=str(args.faces_quarantine_dirname),
            faces_people_no_face_dirname=str(args.faces_people_no_face_dirname),
            animals_dirname=str(args.animals_dirname),
            no_faces_dirname=str(args.no_faces_dirname),
            exclude_dir_names=exclude,
            pipeline=pipeline,
            pipeline_run_id=pipeline_run_id,
        )
        _pipe_log(
            pipeline,
            pipeline_run_id,
            f"split: moved_faces={r.moved_faces} moved_quarantine={r.moved_quarantine} moved_people_no_face={r.moved_people_no_face} moved_animals={r.moved_animals} moved_no_faces={r.moved_no_faces}\n",
        )

        if dry_run:
            _pipe_log(pipeline, pipeline_run_id, "\nNOTE: This was a dry-run. Re-run with --apply to actually move files.\n")

        if pipeline is not None and pipeline_run_id is not None:
            # Snapshot metrics to DB so they don't "float" after next run overwrites files.*
            try:
                dsM = DedupStore()
                try:
                    prM = pipeline.get_run_by_id(run_id=int(pipeline_run_id)) or {}
                    fridM = int(prM.get("face_run_id") or 0) or None
                    root_likeM = None
                    try:
                        root_likeM = "local:" + (os.path.abspath(root).rstrip("\\/") + "\\") + "%"
                    except Exception:
                        root_likeM = None
                    cats_total, cats_mism = _count_misses_for_gold(
                        ds=dsM,
                        pipeline_run_id=int(pipeline_run_id),
                        face_run_id=fridM,
                        root_like=root_likeM,
                        gold_name="cats_gold",
                        expected_tab="animals",
                    )
                    faces_total, faces_mism = _count_misses_for_gold(
                        ds=dsM,
                        pipeline_run_id=int(pipeline_run_id),
                        face_run_id=fridM,
                        root_like=root_likeM,
                        gold_name="faces_gold",
                        expected_tab="faces",
                    )
                    no_faces_total, no_faces_mism = _count_misses_for_gold(
                        ds=dsM,
                        pipeline_run_id=int(pipeline_run_id),
                        face_run_id=fridM,
                        root_like=root_likeM,
                        gold_name="no_faces_gold",
                        expected_tab="no_faces",
                    )
                finally:
                    dsM.close()
                # best-effort: also capture step2 totals from face_runs via FaceStore
                step2_total = step2_processed = None
                try:
                    if fridM:
                        fsM = FaceStore()
                        try:
                            frM = fsM.get_run_by_id(run_id=int(fridM)) or {}
                            step2_total = frM.get("total_files")
                            step2_processed = frM.get("processed_files")
                        finally:
                            fsM.close()
                except Exception:
                    step2_total = step2_processed = None
                pipeline.upsert_metrics(
                    pipeline_run_id=int(pipeline_run_id),
                    metrics={
                        "computed_at": _now_utc_iso(),
                        "face_run_id": int(fridM) if fridM else None,
                        "step0_checked": (step0_counts or {}).get("checked") if step0_counts is not None else None,
                        "step0_non_media": (step0_counts or {}).get("moved_non_media") if step0_counts is not None else None,
                        "step0_broken_media": (step0_counts or {}).get("moved_broken_media") if step0_counts is not None else None,
                        "step2_total": step2_total,
                        "step2_processed": step2_processed,
                        "cats_total": cats_total,
                        "cats_mism": cats_mism,
                        "faces_total": faces_total,
                        "faces_mism": faces_mism,
                        "no_faces_total": no_faces_total,
                        "no_faces_mism": no_faces_mism,
                    },
                )
            except Exception:
                pass
            pipeline.update_run(run_id=pipeline_run_id, status="completed", finished_at=_now_utc_iso())
        return 0
    except Exception as e:  # noqa: BLE001
        if pipeline is not None and pipeline_run_id is not None:
            try:
                pipeline.update_run(run_id=pipeline_run_id, status="failed", last_error=f"{type(e).__name__}: {e}", finished_at=_now_utc_iso())
            except Exception:
                pass
        raise
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())


