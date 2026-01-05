from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
from PIL import Image, ExifTags  # type: ignore[import-untyped]

from DB.db import DedupStore, FaceStore, PipelineStore


EXCLUDE_DIR_NAMES_DEFAULT = ("_faces", "_no_faces", "_duplicates")

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
    faces_found: int = 0
    faces_files: int = 0
    no_faces_files: int = 0
    moved_faces: int = 0
    moved_no_faces: int = 0
    moved_faces_named: int = 0
    moved_faces_unassigned: int = 0
    moved_no_faces_by_geo_year: int = 0


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
            total += 1
        store.update_run_progress(run_id=run_id_i, total_files=total)
        stats.total_files = total

        # 1) hash + upsert
        hash_to_first: dict[str, str] = {}
        for abspath in _iter_files(root, exclude_dir_names=exclude_dir_names):
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

    store = FaceStore()
    dedup = DedupStore()
    try:
        image_files = [p for p in _iter_files(root, exclude_dir_names=exclude_dir_names) if _is_image(p)]
        if run_id is None:
            run_id = store.create_run(scope="local", root_path=_as_local_path(root), total_files=len(image_files))
        run_id_i = int(run_id)
        if pipeline is not None and pipeline_run_id is not None:
            pipeline.update_run(run_id=int(pipeline_run_id), face_run_id=run_id_i)
        total_images = len(image_files)

        for abspath in image_files:
            stats.images_scanned += 1
            db_path = _as_local_path(abspath)
            # Resume-friendly: если уже есть сводка лиц для этого файла под этим run_id — пропускаем.
            row = dedup.get_row_by_path(path=db_path)
            if row and row.get("faces_run_id") is not None and int(row.get("faces_run_id")) == run_id_i and row.get("faces_scanned_at"):
                continue
            try:
                img_bgr = cv2.imread(abspath)
                if img_bgr is None:
                    raise RuntimeError("imdecode_failed")
                # safety: гарантируем ожидаемый формат (H,W,3) uint8
                if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
                    raise RuntimeError(f"bad_image_shape:{getattr(img_bgr, 'shape', None)}")
                if img_bgr.dtype != np.uint8:
                    img_bgr = img_bgr.astype(np.uint8, copy=False)
                # safety: ограничим размер (иногда именно на огромных кадрах detect() падает нативно)
                h, w = img_bgr.shape[:2]
                m = max(h, w)
                if m > max_dim and m > 0:
                    scale = float(max_dim) / float(m)
                    nh = max(1, int(round(h * scale)))
                    nw = max(1, int(round(w * scale)))
                    img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
                # периодически пересоздаём детектор (best-effort от нативных крэшей/утечек)
                if detector_recreate_every > 0 and (stats.images_scanned % detector_recreate_every == 0):
                    detector = _create_face_detector(str(model), score_threshold=float(score_threshold))

                faces = _detect_faces(detector, img_bgr)
                areas = [float(w * h) for (_x, _y, w, h, _s) in faces]
                denom = float(sum(areas)) if areas else 0.0

                store.clear_run_detections_for_file(run_id=run_id_i, file_path=db_path)
                with Image.open(abspath) as pil:

                    if faces:
                        stats.faces_files += 1
                    else:
                        stats.no_faces_files += 1

                    for i, (x, y, w, h, score) in enumerate(faces):
                        pres = (areas[i] / denom) if (denom > 0.0 and i < len(areas)) else None
                        thumb = _crop_thumb_jpeg(img=pil, bbox=(x, y, w, h), thumb_size=int(thumb_size))
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

                # Пишем сводку "лица/не лица" в yd_files, чтобы сортировка могла работать без повторного скана.
                dedup.set_faces_summary(path=db_path, faces_run_id=run_id_i, faces_count=len(faces))

                if stats.images_scanned % 50 == 0:
                    store.update_run_progress(
                        run_id=run_id_i,
                        processed_files=stats.images_scanned,
                        faces_found=stats.faces_found,
                        last_path=db_path,
                    )
                    _pipe_log(
                        pipeline,
                        pipeline_run_id,
                        f"faces_progress images={stats.images_scanned} total={total_images} "
                        f"faces={stats.faces_found} errors={stats.errors}\n",
                    )
            except Exception as e:  # noqa: BLE001
                store.update_run_progress(run_id=run_id_i, last_path=db_path, last_error=f"{type(e).__name__}: {e}")
                stats.errors += 1

        store.update_run_progress(run_id=run_id_i, processed_files=stats.images_scanned, faces_found=stats.faces_found)
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
        # Берём список из yd_files по faces_run_id — чтобы включить и "0 лиц".
        cur = dedup.conn.cursor()
        cur.execute(
            """
            SELECT
              path,
              COALESCE(faces_count, 0) AS faces_count,
              COALESCE(faces_manual_label, '') AS faces_manual_label
            FROM yd_files
            WHERE
              faces_run_id = ?
              AND path LIKE 'local:%'
              AND status != 'deleted'
            ORDER BY path ASC
            """,
            (int(run_id),),
        )
        rows = [(str(r[0]), int(r[1]), str(r[2] or "")) for r in cur.fetchall()]

        for file_path, cnt_auto, manual_label in rows:
            src_abs = _strip_local_prefix(file_path)
            if not os.path.exists(src_abs):
                continue
            # safety: skip if under exclude dirs
            rel = os.path.relpath(src_abs, root)
            rel_parts = Path(rel).parts
            if rel_parts and rel_parts[0] in set(exclude_dir_names):
                continue

            # manual override: приоритетнее автоматического faces_count
            cnt_effective = int(cnt_auto)
            ml = (manual_label or "").strip().lower()
            if ml == "faces":
                cnt_effective = 1
            elif ml == "no_faces":
                cnt_effective = 0

            target_root = faces_dirname if cnt_effective > 0 else no_faces_dirname
            dst_abs = os.path.join(root, target_root, rel)

            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path=src_abs, last_dst_path=dst_abs)
            _move_file(src=src_abs, dst=dst_abs, dry_run=dry_run)

            # update DB paths (yd_files + face_rectangles) ONLY when не dry-run
            if not dry_run:
                new_db_path = _as_local_path(dst_abs)
                dedup.update_path(
                    old_path=file_path,
                    new_path=new_db_path,
                    new_name=os.path.basename(dst_abs),
                    new_parent_path=_as_local_path(os.path.dirname(dst_abs)),
                )
                store.update_file_path(old_file_path=file_path, new_file_path=new_db_path)
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=int(pipeline_run_id), last_src_path="", last_dst_path="")

            if cnt_effective > 0:
                stats.moved_faces += 1
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
    p.add_argument("--duplicates-dirname", default="_duplicates")
    p.add_argument("--faces-dirname", default="_faces")
    p.add_argument("--no-faces-dirname", default="_no_faces")
    p.add_argument("--exclude-dirname", action="append", default=list(EXCLUDE_DIR_NAMES_DEFAULT))
    p.add_argument("--score-threshold", type=float, default=0.85)
    p.add_argument("--thumb-size", type=int, default=160)
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

        # 1) dedup
        if start_step <= 1:
            if pipeline is not None and pipeline_run_id is not None:
                pipeline.update_run(run_id=pipeline_run_id, step_num=1, step_total=2, step_title="дедупликация")
            if args.skip_dedup:
                _pipe_log(pipeline, pipeline_run_id, "\n== шаг 1/2: дедупликация (SKIPPED) ==\n")
            else:
                _pipe_log(pipeline, pipeline_run_id, "\n== шаг 1/2: дедупликация ==\n")
                d = dedup_local(
                    root_dir=root,
                    dry_run=dry_run,
                    move_duplicates=not bool(args.no_dedup_move),
                    duplicates_dirname=str(args.duplicates_dirname),
                    exclude_dir_names=exclude,
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

        # 2) лица / нет лиц = скан + разложение
        if pipeline is not None and pipeline_run_id is not None:
            pipeline.update_run(run_id=pipeline_run_id, step_num=2, step_total=2, step_title="лица/нет лиц: скан")
        _pipe_log(pipeline, pipeline_run_id, "\n== шаг 2/2: лица / нет лиц ==\n")
        _pipe_log(pipeline, pipeline_run_id, "подшаг 2.1: скан лиц (face rectangles)\n")
        run_id2, s = scan_faces_local(
            root_dir=root,
            score_threshold=float(args.score_threshold),
            thumb_size=int(args.thumb_size),
            model_path=Path(str(args.model_path)),
            model_url=str(args.model_url),
            exclude_dir_names=exclude,
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
            no_faces_dirname=str(args.no_faces_dirname),
            exclude_dir_names=exclude,
            pipeline=pipeline,
            pipeline_run_id=pipeline_run_id,
        )
        _pipe_log(pipeline, pipeline_run_id, f"split: moved_faces={r.moved_faces} moved_no_faces={r.moved_no_faces}\n")

        if dry_run:
            _pipe_log(pipeline, pipeline_run_id, "\nNOTE: This was a dry-run. Re-run with --apply to actually move files.\n")

        if pipeline is not None and pipeline_run_id is not None:
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


