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

import sys
from pathlib import Path

# Добавляем корень проекта в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

# Загружаем secrets.env/.env
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

import cv2  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]
import json
import numpy as np

from backend.common.db import FaceStore
from backend.common.yadisk_client import get_disk

# #region agent log
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = "A") -> None:
    """Логирование для отладки проблемы с EXIF orientation."""
    log_path = repo_root / ".cursor" / "debug.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion


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

    # Определяем, является ли путь архивным (disk:/Фото/...)
    is_archive = root_path.startswith("disk:/Фото/") or root_path == "disk:/Фото"

    disk = get_disk()
    files = list_image_files_recursive_yadisk(disk, root_path)
    if args.limit_files and args.limit_files > 0:
        files = files[: int(args.limit_files)]

    model_path = ensure_yunet_model(model_path=Path(args.model_path), model_url=str(args.model_url))
    detector = _create_face_detector(str(model_path), score_threshold=float(args.score_threshold))

    store = FaceStore()
    # Для архива создаём run_id только для статистики (опционально)
    # Для сортируемых папок run_id обязателен
    run_id: int | None = None
    if not is_archive:
        run_id = store.create_run(scope="yadisk", root_path=_as_disk_path(root_path), total_files=len(files))
    
    stats = ScanStats()
    
    if is_archive:
        print(f"ARCHIVE MODE: root={root_path} files={len(files)}")
    else:
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

                # Загружаем изображение через PIL для применения EXIF transpose
                # Это критично: детектор и кроп должны работать с одним и тем же изображением
                pil = Image.open(tmp_path)
                pil_size_before = pil.size
                # #region agent log
                try:
                    from PIL.ExifTags import ORIENTATION
                    exif = pil.getexif()
                    orientation_tag = exif.get(ORIENTATION) if exif else None
                except Exception:
                    orientation_tag = None
                _debug_log(
                    "face_scan.py:before_exif_transpose",
                    "PIL image before EXIF transpose",
                    {
                        "file_path": p_disk,
                        "pil_size_before": list(pil_size_before),
                        "exif_orientation": orientation_tag,
                    },
                    hypothesis_id="A",
                )
                # #endregion
                
                # Применяем EXIF transpose для правильной ориентации
                try:
                    from PIL import ImageOps
                    pil = ImageOps.exif_transpose(pil)
                    pil_size_after = pil.size
                    # #region agent log
                    _debug_log(
                        "face_scan.py:after_exif_transpose",
                        "PIL image after EXIF transpose",
                        {
                            "file_path": p_disk,
                            "pil_size_before": list(pil_size_before),
                            "pil_size_after": list(pil_size_after),
                            "size_changed": pil_size_before != pil_size_after,
                        },
                        hypothesis_id="B",
                    )
                    # #endregion
                except Exception as e:
                    # #region agent log
                    _debug_log(
                        "face_scan.py:exif_transpose_error",
                        "EXIF transpose failed",
                        {
                            "file_path": p_disk,
                            "error": str(e),
                        },
                        hypothesis_id="A",
                    )
                    # #endregion
                    pass
                
                pil = pil.convert("RGB")
                
                # Конвертируем PIL в numpy array для OpenCV (BGR формат)
                pil_array = np.array(pil)
                # PIL использует RGB, OpenCV использует BGR
                img_bgr = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)
                
                # #region agent log
                _debug_log(
                    "face_scan.py:after_cv2_conversion",
                    "Image converted from PIL to OpenCV BGR (with EXIF applied)",
                    {
                        "file_path": p_disk,
                        "img_bgr_shape": list(img_bgr.shape),
                        "pil_size": list(pil.size),
                        "sizes_match": (img_bgr.shape[1], img_bgr.shape[0]) == pil.size,
                    },
                    hypothesis_id="A",
                )
                # #endregion

                faces = _detect_faces(detector, img_bgr)
                # presence_score = доля среди лиц (по площади bbox)
                areas = [float(w * h) for (_x, _y, w, h, _s) in faces]
                denom = float(sum(areas)) if areas else 0.0

                # Для прогонов очищаем результаты для файла (на случай повтора).
                # Для архива не очищаем - используем append без дублирования.
                # Важно: не удаляем ручные прямоугольники (если они когда-нибудь появятся для этого run_id).
                if not is_archive and run_id is not None:
                    store.clear_run_auto_rectangles_for_file(run_id=run_id, file_path=p_disk)

                # PIL изображение уже загружено и повернуто выше (используем его для кропов)

                # Face recognition model (опционально, для извлечения embeddings)
                recognition_model = None
                try:
                    import onnxruntime as ort
                    import json
                    
                    # Проверяем наличие модели
                    repo_root = Path(__file__).resolve().parents[3]
                    model_path = repo_root / "models" / "face_recognition" / "w600k_r50.onnx"
                    
                    if model_path.exists():
                        # Создаём модель напрямую
                        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                        input_name = sess.get_inputs()[0].name
                        output_name = sess.get_outputs()[0].name
                        input_shape = sess.get_inputs()[0].shape
                        
                        recognition_model = {
                            "session": sess,
                            "input_name": input_name,
                            "output_name": output_name,
                            "input_shape": input_shape,
                        }
                except Exception as e:
                    # embeddings опциональны
                    if stats.processed_files == 0:
                        print(f"Примечание: модель распознавания не найдена, embeddings не будут извлекаться: {type(e).__name__}: {e}")
                
                for i, (x, y, w, h, score) in enumerate(faces):
                    pres = (areas[i] / denom) if (denom > 0.0 and i < len(areas)) else None
                    
                    # #region agent log
                    _debug_log(
                        "face_scan.py:before_crop",
                        "Before creating crop thumbnail",
                        {
                            "file_path": p_disk,
                            "face_index": i,
                            "bbox_from_detector": {"x": x, "y": y, "w": w, "h": h},
                            "img_bgr_shape": list(img_bgr.shape),
                            "pil_size": list(pil.size),
                            "bbox_in_pil_bounds": x >= 0 and y >= 0 and (x + w) <= pil.size[0] and (y + h) <= pil.size[1],
                            "bbox_in_bgr_bounds": x >= 0 and y >= 0 and (x + w) <= img_bgr.shape[1] and (y + h) <= img_bgr.shape[0],
                        },
                        hypothesis_id="C",
                    )
                    # #endregion
                    
                    thumb = _crop_thumb_jpeg(img=pil, bbox=(x, y, w, h), thumb_size=int(args.thumb_size))
                    
                    # Извлекаем embedding для распознавания лиц (опционально)
                    embedding: bytes | None = None
                    if recognition_model is not None:
                        try:
                            import json
                            
                            # Кроп лица в формате BGR для распознавания
                            pad = int(round(max(w, h) * 0.18))
                            x0 = max(0, x - pad)
                            y0 = max(0, y - pad)
                            x1 = min(img_bgr.shape[1], x + w + pad)
                            y1 = min(img_bgr.shape[0], y + h + pad)
                            face_crop_bgr = img_bgr[y0:y1, x0:x1]
                            
                            if face_crop_bgr.size > 0:
                                # Извлекаем embedding
                                sess = recognition_model["session"]
                                input_name = recognition_model["input_name"]
                                output_name = recognition_model["output_name"]
                                input_shape = recognition_model["input_shape"]
                                
                                # ArcFace ожидает RGB и определённый размер (обычно 112x112)
                                face_img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                                
                                # Ресайзим до нужного размера
                                target_size = (input_shape[3], input_shape[2]) if len(input_shape) == 4 else (112, 112)
                                face_resized = cv2.resize(face_img_rgb, target_size)
                                
                                # Нормализуем: (pixel - 127.5) / 128.0
                                face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
                                
                                # Транспонируем в формат [1, 3, H, W]
                                face_transposed = np.transpose(face_normalized, (2, 0, 1))
                                face_batch = np.expand_dims(face_transposed, axis=0)
                                
                                # Запускаем инференс
                                outputs = sess.run([output_name], {input_name: face_batch})
                                emb = outputs[0][0]
                                
                                # Нормализуем embedding (L2 нормализация)
                                norm = np.linalg.norm(emb)
                                if norm > 0:
                                    emb = emb / norm
                                
                                # Сериализуем в JSON
                                embedding = json.dumps(emb.tolist()).encode("utf-8")
                        except Exception:
                            embedding = None
                    
                    # Получаем размеры исходного изображения (после EXIF transpose)
                    pil_width, pil_height = pil.size
                    
                    # #region agent log
                    _debug_log(
                        "face_scan.py:before_save",
                        "Before saving to database",
                        {
                            "file_path": p_disk,
                            "face_index": i,
                            "bbox_saved": {"x": x, "y": y, "w": w, "h": h},
                            "image_size_saved": {"width": pil_width, "height": pil_height},
                            "img_bgr_shape": list(img_bgr.shape),
                            "pil_size": list(pil.size),
                            "sizes_match": (img_bgr.shape[1], img_bgr.shape[0]) == (pil_width, pil_height),
                            "bbox_in_bounds": x >= 0 and y >= 0 and (x + w) <= pil_width and (y + h) <= pil_height,
                        },
                        hypothesis_id="D",
                    )
                    # #endregion
                    
                    # Для архивного режима используем archive_scope='archive' и опциональный run_id
                    # Для прогонов используем run_id и archive_scope=None
                    inserted = store.insert_detection(
                        run_id=run_id,
                        archive_scope='archive' if is_archive else None,
                        file_path=p_disk,
                        face_index=i,
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                        confidence=score,
                        presence_score=pres,
                        thumb_jpeg=thumb,
                        embedding=embedding,
                        image_width=pil_width,
                        image_height=pil_height,
                    )
                    # Учитываем только вставленные (не дубликаты)
                    if inserted:
                        stats.faces_found += 1

                stats.processed_files += 1
                if stats.processed_files % 20 == 0:
                    # Обновляем прогресс только для прогонов (для архива run_id может быть None)
                    if not is_archive and run_id is not None:
                        store.update_run_progress(
                            run_id=run_id,
                            processed_files=stats.processed_files,
                            faces_found=stats.faces_found,
                            last_path=p_disk,
                        )
                    print(f"progress: {stats.processed_files}/{len(files)} files, faces={stats.faces_found}")

            except Exception as e:  # noqa: BLE001
                # Обновляем прогресс только для прогонов (для архива run_id может быть None)
                if not is_archive and run_id is not None:
                    store.update_run_progress(run_id=run_id, last_path=p_disk, last_error=f"{type(e).__name__}: {e}")
                # продолжаем скан, но печатаем ошибку
                print(f"ERROR: {p_disk}: {type(e).__name__}: {e}")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        # Обновляем прогресс только для прогонов (для архива run_id может быть None)
        if not is_archive and run_id is not None:
            store.update_run_progress(
                run_id=run_id,
                processed_files=stats.processed_files,
                faces_found=stats.faces_found,
                last_path=_as_disk_path(root_path),
            )
        
        # Автоматическая кластеризация embeddings после завершения детекции
        try:
            from backend.logic.face_recognition import cluster_face_embeddings
            if is_archive:
                # Для архива используем archive_scope вместо run_id
                print(f"clustering: starting clustering for archive (root={root_path})")
                cluster_result = cluster_face_embeddings(
                    run_id=None,  # для архива run_id не используется
                    archive_scope='archive',
                    eps=0.4,
                    min_samples=2,
                    use_folder_context=True,
                )
            else:
                # Для прогонов используем run_id
                if run_id is None:
                    raise ValueError("run_id обязателен для неархивных записей")
                print(f"clustering: starting clustering for run_id={run_id}")
                cluster_result = cluster_face_embeddings(
                    run_id=run_id,
                    archive_scope=None,
                    eps=0.4,
                    min_samples=2,
                    use_folder_context=True,
                )
            clusters_count = cluster_result.get('clusters_count', 0)
            noise_count = cluster_result.get('noise_count', 0)
            total_faces = cluster_result.get('total_faces', 0)
            print(
                f"clustering: completed - {clusters_count} clusters, "
                f"{noise_count} noise, total {total_faces} faces"
            )
        except Exception as e:
            print(f"clustering: error - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        # Завершаем run только для прогонов (для архива run_id может быть None)
        if not is_archive and run_id is not None:
            store.finish_run(run_id=run_id, status="completed")
        print(f"DONE: files={stats.processed_files}, faces={stats.faces_found}")
        return 0
    except Exception as e:  # noqa: BLE001
        # Завершаем run только для прогонов (для архива run_id может быть None)
        if not is_archive and run_id is not None:
            store.finish_run(run_id=run_id, status="failed", last_error=f"{type(e).__name__}: {e}")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())


