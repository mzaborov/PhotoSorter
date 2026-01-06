from __future__ import annotations

import argparse
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import cv2  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

from DB.db import FaceStore
from yadisk_client import get_disk


def _get(item: Any, key: str) -> Optional[Any]:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _norm_yadisk_path(path: str) -> str:
    p = path or ""
    if p.startswith("disk:"):
        p = p[len("disk:") :]
    if not p.startswith("/"):
        p = "/" + p
    return p


def _as_disk_path(path: str) -> str:
    p = path or ""
    if p.startswith("disk:"):
        return p
    p2 = p if p.startswith("/") else ("/" + p)
    return "disk:" + p2


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path) -> None:
    _ensure_parent_dir(dest)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dest)


def ensure_yunet_model(*, model_path: Path, model_url: str) -> Path:
    if model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    print(f"Downloading YuNet model to: {model_path}")
    _download(model_url, model_path)
    return model_path


def _create_face_detector(model_path: str, *, score_threshold: float) -> Any:
    # OpenCV python API varies slightly by version.
    # YuNet model expects FaceDetectorYN.
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(model_path, "", (320, 320), score_threshold, 0.3, 5000)
    # Fallback
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)


def _detect_faces(detector: Any, img_bgr) -> list[tuple[int, int, int, int, float]]:
    h, w = img_bgr.shape[:2]
    try:
        detector.setInputSize((w, h))
    except Exception:
        # старые биндинги могли не иметь setInputSize
        pass

    res = detector.detect(img_bgr)
    # OpenCV bindings differ: sometimes returns (ok, faces), sometimes only faces.
    if isinstance(res, tuple) and len(res) >= 2:
        faces = res[1]
    else:
        faces = res
    if faces is None:
        return []
    out: list[tuple[int, int, int, int, float]] = []
    for row in faces:
        # [x, y, w, h, score, ...landmarks]
        x, y, ww, hh, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        xi = max(0, int(round(x)))
        yi = max(0, int(round(y)))
        wi = max(0, int(round(ww)))
        hi = max(0, int(round(hh)))
        # clip
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


def _crop_thumb_jpeg(
    *,
    img: Image.Image,
    bbox: tuple[int, int, int, int],
    thumb_size: int,
    pad_ratio: float = 0.18,
) -> bytes:
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
class ScanStats:
    processed_files: int = 0
    faces_found: int = 0


def list_image_files_recursive_yadisk(disk, root_path: str) -> list[dict[str, Any]]:
    root = _norm_yadisk_path(root_path)
    stack = [root]
    visited: set[str] = set()
    out: list[dict[str, Any]] = []

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        items = list(disk.listdir(cur))
        for it in items:
            t = _get(it, "type")
            if t == "dir":
                p = _get(it, "path")
                if p:
                    stack.append(_norm_yadisk_path(str(p)))
                continue
            if t != "file":
                continue
            mime = str(_get(it, "mime_type") or "").lower()
            name = str(_get(it, "name") or "")
            if mime.startswith("image/"):
                out.append(
                    {
                        "path": str(_get(it, "path") or ""),
                        "name": name,
                        "mime_type": mime,
                    }
                )
                continue
            # fallback по расширению
            ext = (Path(name).suffix or "").lower()
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"):
                out.append({"path": str(_get(it, "path") or ""), "name": name, "mime_type": mime or None})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Face scan for a YaDisk folder (photo-only). Writes results to SQLite.")
    parser.add_argument("--path", required=True, help="YaDisk path like disk:/Фото/Агата")
    parser.add_argument("--limit-files", type=int, default=0, help="Limit files (0 = no limit)")
    parser.add_argument("--thumb-size", type=int, default=160, help="Face thumbnail max size (pixels)")
    parser.add_argument("--score-threshold", type=float, default=0.85, help="Face detector score threshold")
    parser.add_argument(
        "--model-path",
        default=str(Path("data") / "models" / "face_detection_yunet_2023mar.onnx"),
        help="Path to YuNet onnx model (downloaded if missing)",
    )
    parser.add_argument(
        "--model-url",
        default="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        help="URL to download YuNet model if model-path does not exist",
    )
    args = parser.parse_args()

    root_path = str(args.path)
    if not root_path.startswith("disk:"):
        print("ERROR: only YaDisk disk:/... paths are supported for now")
        return 2

    disk = get_disk()
    files = list_image_files_recursive_yadisk(disk, root_path)
    if args.limit_files and args.limit_files > 0:
        files = files[: int(args.limit_files)]

    model_path = ensure_yunet_model(model_path=Path(args.model_path), model_url=str(args.model_url))
    detector = _create_face_detector(str(model_path), score_threshold=float(args.score_threshold))

    store = FaceStore()
    run_id = store.create_run(scope="yadisk", root_path=_as_disk_path(root_path), total_files=len(files))
    stats = ScanStats()

    print(f"RUN id={run_id} root={root_path} files={len(files)}")

    try:
        for f in files:
            p_disk = str(f.get("path") or "")
            if not p_disk:
                continue
            p_disk = _as_disk_path(p_disk)
            remote = _norm_yadisk_path(p_disk)

            tmp_path: str | None = None
            try:
                # Важно: tmp-файл всегда удаляем (не оставляем мусор).
                suffix = Path(str(f.get("name") or "")).suffix or ".bin"
                with tempfile.NamedTemporaryFile(prefix="photosorter_face_", suffix=suffix, delete=False) as tmp:
                    tmp_path = tmp.name
                disk.download(remote, tmp_path)

                img_bgr = cv2.imread(tmp_path)
                if img_bgr is None:
                    raise RuntimeError("imdecode_failed")

                faces = _detect_faces(detector, img_bgr)
                # presence_score = доля среди лиц (по площади bbox)
                areas = [float(w * h) for (_x, _y, w, h, _s) in faces]
                denom = float(sum(areas)) if areas else 0.0

                # Очищаем результаты для файла в рамках текущего прогона (на случай повтора).
                # Важно: не удаляем ручные прямоугольники (если они когда-нибудь появятся для этого run_id).
                store.clear_run_auto_rectangles_for_file(run_id=run_id, file_path=p_disk)

                # Для thumb используем PIL (качество JPEG + быстрый resize).
                pil = Image.open(tmp_path)

                for i, (x, y, w, h, score) in enumerate(faces):
                    pres = (areas[i] / denom) if (denom > 0.0 and i < len(areas)) else None
                    thumb = _crop_thumb_jpeg(img=pil, bbox=(x, y, w, h), thumb_size=int(args.thumb_size))
                    store.insert_detection(
                        run_id=run_id,
                        file_path=p_disk,
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

                stats.processed_files += 1
                if stats.processed_files % 20 == 0:
                    store.update_run_progress(
                        run_id=run_id,
                        processed_files=stats.processed_files,
                        faces_found=stats.faces_found,
                        last_path=p_disk,
                    )
                    print(f"progress: {stats.processed_files}/{len(files)} files, faces={stats.faces_found}")

            except Exception as e:  # noqa: BLE001
                store.update_run_progress(run_id=run_id, last_path=p_disk, last_error=f"{type(e).__name__}: {e}")
                # продолжаем скан, но печатаем ошибку
                print(f"ERROR: {p_disk}: {type(e).__name__}: {e}")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        store.update_run_progress(
            run_id=run_id,
            processed_files=stats.processed_files,
            faces_found=stats.faces_found,
            last_path=_as_disk_path(root_path),
        )
        store.finish_run(run_id=run_id, status="completed")
        print(f"DONE: files={stats.processed_files}, faces={stats.faces_found}")
        return 0
    except Exception as e:  # noqa: BLE001
        store.finish_run(run_id=run_id, status="failed", last_error=f"{type(e).__name__}: {e}")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())


