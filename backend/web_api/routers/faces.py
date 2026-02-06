from __future__ import annotations

import base64
import hashlib
import json
import logging
import sqlite3
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

from common.db import DedupStore, FaceStore, PipelineStore, get_connection, get_outsider_person_id, list_folders, set_file_processed
from common.sort_rules import (
    determine_target_folder as _determine_target_folder,
    folder_rules_match as _folder_rules_match,
    get_all_person_names_for_file as _get_all_person_names_for_file,
    parse_content_rule as _parse_content_rule,
    resolve_target_folder_for_faces as _resolve_target_folder_for_faces,
)
from common.yadisk_client import get_disk
try:
    from yadisk import exceptions as yadisk_exceptions
except ImportError:
    yadisk_exceptions = None  # type: ignore[assignment]
from logic.gold.store import gold_expected_tab_by_path, gold_file_map, gold_read_lines, gold_write_lines, gold_read_ndjson_by_path, gold_write_ndjson_by_path, gold_faces_manual_rects_path, gold_faces_video_frames_path

router = APIRouter()

APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/debug/photo-card", response_class=HTMLResponse)
def debug_photo_card_page(
    request: Request,
    path: str | None = Query(default=None),
    file_id: str | None = Query(default=None),
    pipeline_run_id: str | None = Query(default=None),
    auto_open: bool = Query(default=True),
):
    """
    Отладочная страница для прямого открытия карточки фото по path/file_id.

    Примеры:
      /debug/photo-card?pipeline_run_id=123&path=local:C:\\tmp\\Photo\\_faces\\IMG.jpg
      /debug/photo-card?pipeline_run_id=123&file_id=456
      /debug/photo-card?path=local:C:\\tmp\\Photo\\_faces\\IMG.jpg (pipeline_run_id будет попытка вывести автоматически)
    """
    # Нормализация параметров
    path_s = (str(path).strip() if path is not None else "") or None
    file_id_s = (str(file_id).strip() if file_id is not None else "")
    pipeline_run_id_s = (str(pipeline_run_id).strip() if pipeline_run_id is not None else "")

    if file_id_s == "":
        file_id_i = None
    else:
        try:
            file_id_i = int(file_id_s)
        except Exception:
            raise HTTPException(status_code=400, detail="file_id must be a valid integer") from None

    if pipeline_run_id_s == "":
        pipeline_run_id_i = None
    else:
        try:
            pipeline_run_id_i = int(pipeline_run_id_s)
        except Exception:
            raise HTTPException(status_code=400, detail="pipeline_run_id must be a valid integer") from None

    if not path_s and file_id_i is None:
        raise HTTPException(status_code=400, detail="Either path or file_id must be provided")

    # Если path не передан — пробуем получить его по file_id.
    if not path_s and file_id_i is not None:
        fs = FaceStore()
        try:
            cur = fs.conn.cursor()
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (int(file_id_i),))
            row = cur.fetchone()
            if not row or not row["path"]:
                raise HTTPException(status_code=404, detail="file_id not found")
            path_s = str(row["path"])
        finally:
            fs.close()

    # Попытка вывести pipeline_run_id автоматически (удобно для sorting),
    # если его не передали явно.
    inferred: dict[str, Any] | None = None
    if pipeline_run_id_i is None:
        fs = FaceStore()
        try:
            cur = fs.conn.cursor()
            # По файлу берём "последний" face_run_id из photo_rectangles (если есть)
            if path_s:
                cur.execute("SELECT id FROM files WHERE path = ? LIMIT 1", (str(path_s),))
                fr = cur.fetchone()
                resolved_file_id = int(fr["id"]) if fr and fr["id"] is not None else None
            else:
                resolved_file_id = file_id_i

            if resolved_file_id is not None:
                cur.execute(
                    """
                    SELECT run_id
                    FROM photo_rectangles
                    WHERE file_id = ?
                      AND run_id IS NOT NULL
                      AND run_id != 0
                      AND COALESCE(ignore_flag, 0) = 0
                    ORDER BY run_id DESC
                    LIMIT 1
                    """,
                    (int(resolved_file_id),),
                )
                r = cur.fetchone()
                face_run_id = int(r["run_id"]) if r and r["run_id"] is not None else None
            else:
                face_run_id = None
        finally:
            fs.close()

        if face_run_id:
            ps = PipelineStore()
            try:
                cur2 = ps.conn.cursor()
                cur2.execute(
                    """
                    SELECT id
                    FROM pipeline_runs
                    WHERE face_run_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (int(face_run_id),),
                )
                pr = cur2.fetchone()
                if pr and pr["id"] is not None:
                    pipeline_run_id_i = int(pr["id"])
                    inferred = {"face_run_id": int(face_run_id), "pipeline_run_id": int(pipeline_run_id_i)}
            finally:
                ps.close()

    return templates.TemplateResponse(
        "debug_photo_card.html",
        {
            "request": request,
            "path": path_s,
            "file_id": file_id_i,
            "pipeline_run_id": pipeline_run_id_i,
            "auto_open": bool(auto_open),
            "inferred": inferred,
        },
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _agent_dbg(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any] | None = None, run_id: str = "pre-fix") -> None:
    """
    Tiny NDJSON logger for debug-mode evidence. Writes to .cursor/debug.log.
    Never log secrets/PII.
    """
    try:
        p = _repo_root() / ".cursor" / "debug.log"
        payload = {
            "sessionId": "debug-session",
            "runId": str(run_id),
            "hypothesisId": str(hypothesis_id),
            "location": str(location),
            "message": str(message),
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.open("a", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _strip_local_prefix(p: str) -> str:
    return p[len("local:") :] if (p or "").startswith("local:") else p


def _local_is_under_root(*, file_path: str, root_dir: str) -> bool:
    try:
        rp = os.path.abspath(str(root_dir))
        fp = os.path.abspath(str(file_path))
    except Exception:
        return False
    # Windows: case-insensitive + normalize separators
    rp_n = os.path.normcase(rp.rstrip("\\/") + os.sep)
    fp_n = os.path.normcase(fp)
    return fp_n.startswith(rp_n)


def _venv_face_python() -> Path:
    rr = _repo_root()
    return rr / ".venv-face" / "Scripts" / "python.exe"


def _video_keyframes_script() -> Path:
    rr = _repo_root()
    return rr / "backend" / "scripts" / "tools" / "video_keyframes.py"


_VIDEO_META_CACHE: dict[str, dict[str, Any]] = {}


def _video_keyframe_times_cached(*, abs_video_path: str, samples: int = 3) -> list[float]:
    """
    Returns list of times (seconds) for keyframes. Uses venv-face (cv2) via subprocess.
    """
    p = str(abs_video_path or "")
    if not p:
        return [0.0]
    try:
        st = os.stat(p)
        key = f"{os.path.normcase(p)}|{int(st.st_size)}|{int(st.st_mtime)}|{int(samples)}"
    except Exception:
        key = f"{os.path.normcase(p)}|?|?|{int(samples)}"

    cached = _VIDEO_META_CACHE.get(key)
    if cached and isinstance(cached.get("times_sec"), list):
        try:
            return [float(x) for x in (cached.get("times_sec") or []) if x is not None]
        except Exception:
            pass

    py = _venv_face_python()
    script = _video_keyframes_script()
    if not py.exists():
        raise HTTPException(status_code=500, detail=f"Missing .venv-face python: {py}")
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {script}")

    cmd = [
        str(py),
        str(script.relative_to(_repo_root())),
        "--mode",
        "meta",
        "--path",
        str(p),
        "--samples",
        str(max(1, min(3, int(samples or 3)))),
    ]
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_repo_root()), timeout=30)  # noqa: S603,S607
    except subprocess.TimeoutExpired:
        return [0.0]
    if pr.returncode != 0:
        return [0.0]
    try:
        obj = json.loads(pr.stdout or "{}")
        times = obj.get("times_sec") or []
        if not isinstance(times, list) or not times:
            return [0.0]
        out = [float(x) for x in times if x is not None]
        _VIDEO_META_CACHE[key] = {"times_sec": out}
        return out
    except Exception:
        return [0.0]


def _basename_from_disk_path(path: str) -> str:
    p = path or ""
    if "/" not in p:
        return p
    return p.rsplit("/", 1)[-1]


def _short_path_for_ui(path: str) -> str:
    # Как раньше: обрезаем disk:/Фото/ и оставляем хвост.
    p = path or ""
    prefix = "disk:/Фото/"
    if p.startswith(prefix):
        tail = p[len(prefix) :]
        return "…/" + tail
    return p


def _human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "—"
    try:
        x = float(n)
    except Exception:
        return str(n)
    if x < 1024:
        return f"{int(x)} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        x /= 1024.0
        if x < 1024.0:
            return f"{x:.1f} {unit}"
    return f"{x:.1f} PB"


@router.get("/faces", response_class=HTMLResponse)
def faces_results_page(request: Request, pipeline_run_id: int | None = None):
    """
    Результаты шага 3 (лица/нет лиц): две вкладки и ручная корректировка.
    pipeline_run_id нужен, чтобы брать актуальный root/run_id и безопасно отдавать local preview.
    """
    return templates.TemplateResponse("faces.html", {"request": request, "pipeline_run_id": pipeline_run_id})


@router.get("/api/faces/video-manual-frames")
def api_faces_video_manual_frames(pipeline_run_id: int, path: str) -> dict[str, Any]:
    if not isinstance(path, str) or not path.startswith("local:"):
        raise HTTPException(status_code=400, detail="path must start with local:")

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")

    root_path = str(pr.get("root_path") or "")
    abs_path = _strip_local_prefix(path)
    if not abs_path or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="file not found")
    if root_path and not _local_is_under_root(file_path=abs_path, root_dir=root_path):
        raise HTTPException(status_code=403, detail="Path is outside pipeline root")

    times = _video_keyframe_times_cached(abs_video_path=abs_path, samples=3)
    # 1..3
    times3: dict[int, float | None] = {}
    for i in range(1, 4):
        times3[i] = float(times[i - 1]) if i - 1 < len(times) else None

    fs = FaceStore()
    try:
        mf = fs.get_video_manual_frames(pipeline_run_id=int(pipeline_run_id), path=str(path))
    finally:
        fs.close()

    frames: list[dict[str, Any]] = []
    for i in (1, 2, 3):
        obj = mf.get(i) or {}
        frames.append(
            {
                "frame_idx": i,
                "t_sec": (obj.get("t_sec") if obj.get("t_sec") is not None else times3.get(i)),
                "rects": obj.get("rects") or [],
                "updated_at": obj.get("updated_at") or "",
            }
        )

    return {"ok": True, "pipeline_run_id": int(pipeline_run_id), "path": str(path), "frames": frames}


@router.post("/api/faces/video-manual-frame")
def api_faces_video_manual_frame(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    pipeline_run_id = payload.get("pipeline_run_id")
    path = payload.get("path")
    frame_idx = payload.get("frame_idx")
    rects = payload.get("rects")
    t_sec = payload.get("t_sec")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(path, str) or not path.startswith("local:"):
        raise HTTPException(status_code=400, detail="path must start with local:")
    try:
        idx = int(frame_idx)
    except Exception:
        raise HTTPException(status_code=400, detail="frame_idx must be int") from None
    if idx not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="frame_idx must be 1..3")
    if rects is None:
        rects = []
    if not isinstance(rects, list):
        raise HTTPException(status_code=400, detail="rects must be list")

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")

    root_path = str(pr.get("root_path") or "")
    abs_path = _strip_local_prefix(path)
    if not abs_path or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="file not found")
    if root_path and not _local_is_under_root(file_path=abs_path, root_dir=root_path):
        raise HTTPException(status_code=403, detail="Path is outside pipeline root")

    # t_sec: if not provided, compute using keyframe meta
    t_val: float | None = None
    try:
        if t_sec is not None:
            t_val = float(t_sec)
    except Exception:
        t_val = None
    if t_val is None:
        times = _video_keyframe_times_cached(abs_video_path=abs_path, samples=3)
        if idx - 1 < len(times):
            t_val = float(times[idx - 1])

    fs = FaceStore()
    try:
        fs.upsert_video_manual_frame(
            pipeline_run_id=int(pipeline_run_id),
            path=str(path),
            frame_idx=int(idx),
            t_sec=t_val,
            rects=rects,
        )
    finally:
        fs.close()

    # Считаем это разметкой "есть лица" для данного видео (в рамках прогона)
    ds = DedupStore()
    try:
        ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=str(path), label="faces")
    finally:
        ds.close()

    return {"ok": True}


@router.get("/api/faces/video-frame")
def api_faces_video_frame(pipeline_run_id: int, path: str, frame_idx: int = 1, max_dim: int = 960) -> FileResponse:
    if not isinstance(path, str) or not path.startswith("local:"):
        raise HTTPException(status_code=400, detail="path must start with local:")
    idx = int(frame_idx or 0)
    if idx not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="frame_idx must be 1..3")

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")

    root_path = str(pr.get("root_path") or "")
    abs_path = _strip_local_prefix(path)
    if not abs_path or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="file not found")
    if root_path and not _local_is_under_root(file_path=abs_path, root_dir=root_path):
        raise HTTPException(status_code=403, detail="Path is outside pipeline root")

    md = int(max_dim or 0)
    md = max(128, min(2048, md))

    # cache by (path, mtime, size, idx, md)
    try:
        st = os.stat(abs_path)
        h = hashlib.sha1(f"{os.path.normcase(abs_path)}|{int(st.st_size)}|{int(st.st_mtime)}|{idx}|{md}".encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        h = hashlib.sha1(f"{os.path.normcase(abs_path)}|{idx}|{md}".encode("utf-8", errors="ignore")).hexdigest()

    rr = _repo_root()
    cache_dir = rr / "data" / "cache" / "video_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{h}.jpg"
    if out_path.exists():
        return FileResponse(path=str(out_path), media_type="image/jpeg", filename=out_path.name, headers={"Cache-Control": "private, max-age=3600"})

    py = _venv_face_python()
    script = _video_keyframes_script()
    if not py.exists():
        raise HTTPException(status_code=500, detail=f"Missing .venv-face python: {py}")
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {script}")

    with tempfile.NamedTemporaryFile(prefix="ps_vf_", suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        str(py),
        str(script.relative_to(rr)),
        "--mode",
        "extract",
        "--path",
        str(abs_path),
        "--samples",
        "3",
        "--frame-idx",
        str(idx),
        "--max-dim",
        str(md),
        "--out",
        str(tmp_path),
    ]
    try:
        prc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(rr), timeout=45)  # noqa: S603,S607
    except subprocess.TimeoutExpired:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=504, detail="frame_extract_timeout") from None

    if prc.returncode != 0 or (not os.path.isfile(tmp_path)):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        msg = (prc.stderr or prc.stdout or "").strip()[:400]
        raise HTTPException(status_code=500, detail=f"frame_extract_failed: {msg or prc.returncode}")

    try:
        os.replace(tmp_path, str(out_path))
    except Exception:
        # fallback: keep tmp
        out_path = Path(tmp_path)

    return FileResponse(path=str(out_path), media_type="image/jpeg", filename=out_path.name, headers={"Cache-Control": "private, max-age=3600"})


def _faces_preview_meta(*, path: str, mime_type: str | None, media_type: str | None, pipeline_run_id: int | None) -> dict[str, Any]:
    preview_kind = "none"  # 'image'|'video'|'none'
    preview_url: Optional[str] = None
    open_url: Optional[str] = None
    mt = (media_type or "").lower()
    mime = (mime_type or "").lower()
    if path.startswith("disk:"):
        open_url = "/api/yadisk/open?path=" + urllib.parse.quote(path, safe="")
        if mt == "image" or mime.startswith("image/"):
            preview_kind = "image"
            preview_url = "/api/yadisk/preview-image?size=M&path=" + urllib.parse.quote(path, safe="")
        elif mt == "video" or mime.startswith("video/"):
            preview_kind = "video"
            preview_url = None
    elif path.startswith("local:"):
        if mt == "image" or mime.startswith("image/"):
            preview_kind = "image"
        elif mt == "video" or mime.startswith("video/"):
            preview_kind = "video"
        if preview_kind != "none":
            q = "path=" + urllib.parse.quote(path, safe="")
            if pipeline_run_id is not None:
                q += "&pipeline_run_id=" + urllib.parse.quote(str(int(pipeline_run_id)), safe="")
            preview_url = "/api/local/preview?" + q
    return {"preview_kind": preview_kind, "preview_url": preview_url, "open_url": open_url}


def _group_into_trips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Группирует файлы в поездки для вкладки "Нет людей" с сортировкой по месту и дате.
    
    Логика:
    - Страна обязательна для группировки по месту
    - Город только для России
    - Окно - максимальный разрыв между соседними датами (10 дней)
    - Файлы без GPS/геокодирования группируются только по дате (временные поездки)
    
    Возвращает список строк с добавленным полем trip_group для группировки.
    """
    TRIP_WINDOW_DAYS = 10
    
    # Разделяем на файлы с местом и без
    with_place: list[dict[str, Any]] = []
    without_place: list[dict[str, Any]] = []
    
    for r in rows:
        country = (str(r.get("place_country") or "")).strip()
        city = (str(r.get("place_city") or "")).strip()
        taken_at = (str(r.get("taken_at") or "")).strip()
        
        if not country:
            # Нет места - будем группировать по дате
            without_place.append(r)
        else:
            # Есть страна - группируем в поездки по месту
            with_place.append(r)
    
    # Группируем файлы с местом в поездки
    trips: list[list[dict[str, Any]]] = []
    current_trip: list[dict[str, Any]] | None = None
    
    # Сортируем по стране, городу (только для России), дате, пути (для стабильности)
    with_place_sorted = sorted(
        with_place,
        key=lambda x: (
            str(x.get("place_country") or ""),
            str(x.get("place_city") or "") if str(x.get("place_country") or "").lower() == "россия" else "",
            str(x.get("taken_at") or ""),
            str(x.get("path") or ""),
        ),
    )
    
    for r in with_place_sorted:
        country = (str(r.get("place_country") or "")).strip()
        city = (str(r.get("place_city") or "")).strip()
        taken_at = (str(r.get("taken_at") or "")).strip()
        
        # Формируем ключ места: страна + город (только для России)
        # Определяем год из даты
        year = ""
        if taken_at and len(taken_at) >= 4:
            year = taken_at[:4]
        
        if country.lower() == "россия" and city:
            place_key = f"{country}|{city}"
            # Формат: "Год Город" (например, "2023 Екатеринбург")
            trip_label = f"{year} {city}" if year else city
        else:
            place_key = country
            # Формат: "Год Страна" (например, "2023 Турция")
            trip_label = f"{year} {country}" if year else country
        
        # Парсим дату
        trip_date = None
        if taken_at and len(taken_at) >= 10:
            try:
                # Парсим только дату (YYYY-MM-DD), игнорируя время и timezone
                date_str = taken_at[:10]
                trip_date = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        
        # Определяем, начинать ли новую поездку
        if current_trip is None:
            # Первая поездка
            current_trip = [r]
            r["trip_group"] = place_key
            r["trip_label"] = trip_label
            r["trip_date"] = trip_date
        else:
            # Проверяем, относится ли файл к текущей поездке
            prev_place_key = current_trip[0].get("trip_group", "")
            # Берём дату последнего файла в поездке для проверки разрыва
            last_file_in_trip = current_trip[-1]
            prev_date = last_file_in_trip.get("trip_date")
            
            # Новая поездка, если:
            # 1. Другое место (страна или город для России)
            # 2. Разрыв в датах > 10 дней (от последнего файла в поездке)
            is_new_trip = False
            if place_key != prev_place_key:
                is_new_trip = True
            elif trip_date and prev_date:
                days_diff = abs((trip_date - prev_date).days)
                if days_diff > TRIP_WINDOW_DAYS:
                    is_new_trip = True
            
            if is_new_trip:
                # Сохраняем текущую поездку и начинаем новую
                trips.append(current_trip)
                current_trip = [r]
                r["trip_group"] = place_key
                r["trip_label"] = trip_label
                r["trip_date"] = trip_date
            else:
                # Добавляем к текущей поездке
                r["trip_group"] = place_key
                r["trip_label"] = trip_label
                r["trip_date"] = trip_date
                current_trip.append(r)
    
    # Добавляем последнюю поездку
    if current_trip:
        trips.append(current_trip)
    
    # Файлы без места не группируем в поездки - оставляем как есть, без trip_group и trip_label
    # Они будут отображаться с заголовком "Нет группы" на UI
    
    # Объединяем все поездки и сортируем по дате первой фотографии в поездке
    result: list[dict[str, Any]] = []
    
    # Собираем все поездки с их первой датой для сортировки
    all_trips_with_date: list[tuple[datetime | None, list[dict[str, Any]]]] = []
    
    # Поездки с местом
    for trip in trips:
        # Сортируем файлы внутри поездки по дате, затем по пути (для стабильности)
        trip_sorted = sorted(trip, key=lambda x: (str(x.get("taken_at") or ""), str(x.get("path") or "")))
        # Берём дату первого файла для сортировки
        first_date_str = trip_sorted[0].get("taken_at", "") if trip_sorted else ""
        first_date = None
        if first_date_str and len(first_date_str) >= 10:
            try:
                # Парсим только дату (YYYY-MM-DD), игнорируя время и timezone
                date_str = first_date_str[:10]
                first_date = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        all_trips_with_date.append((first_date, trip_sorted))
    
    # Сортируем поездки по дате (от старых к новым)
    all_trips_with_date.sort(key=lambda x: x[0] if x[0] else datetime.min)
    
    # Объединяем отсортированные поездки
    for _date, trip_sorted in all_trips_with_date:
        result.extend(trip_sorted)
    
    # Добавляем файлы без места в конец (без группировки, просто отсортированные по дате, затем по пути)
    without_place_sorted = sorted(
        without_place,
        key=lambda x: (str(x.get("taken_at") or ""), str(x.get("path") or "")),
    )
    result.extend(without_place_sorted)
    
    return result


@router.get("/api/faces/results")
def api_faces_results(
    pipeline_run_id: int,
    tab: str = "faces",
    subtab: str | None = None,
    sort: str | None = None,
    manual_filter: str | None = Query(None, description="Для закладки Посторонние лица: all|manual_only|no_manual"),
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    page: int = 1,
    page_size: int = 60,
) -> dict[str, Any]:
    start_time = time.time()
    msg = f"[API] api_faces_results: начало, pipeline_run_id={pipeline_run_id}, tab={tab}, subtab={subtab}, page={page}, page_size={page_size}"
    logger.info(msg)
    tab_n = (tab or "").strip().lower()
    if tab_n not in ("faces", "no_faces", "quarantine", "animals", "people_no_face"):
        raise HTTPException(status_code=400, detail="tab must be faces|no_faces|quarantine|animals|people_no_face")
    # Карантин теперь показывается в "Нет людей" -> "К разбору", но оставляем поддержку для обратной совместимости
    if tab_n == "quarantine":
        tab_n = "no_faces"

    subtab_n = (subtab or "").strip().lower() or "all"
    person_id_filter: int | None = None
    group_path_filter: str | None = None
    manual_filter_n: str | None = (manual_filter or "").strip().lower() or None
    if manual_filter_n and manual_filter_n not in ("all", "manual_only", "no_manual"):
        manual_filter_n = "all"
    if tab_n == "faces":
        if subtab_n.startswith("person_"):
            try:
                person_id_filter = int(subtab_n.replace("person_", ""))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid person_id in subtab")
        elif subtab_n not in ("all", "many_faces", "unsorted", "all_faces"):
            raise HTTPException(status_code=400, detail="subtab for faces must be all|many_faces|unsorted|all_faces|person_<id>")
    elif tab_n == "no_faces":
        if subtab_n.startswith("group_"):
            # Декодируем путь группы (может содержать "/" и другие спецсимволы)
            group_path_filter = urllib.parse.unquote(subtab_n.replace("group_", ""))
            # Убираем лишние пробелы (данные уже нормализованы в БД)
            group_path_filter = group_path_filter.strip()
        elif subtab_n not in ("all", "unsorted", "unsorted_photos", "unsorted_videos"):
            raise HTTPException(status_code=400, detail="subtab for no_faces must be all|unsorted|unsorted_photos|unsorted_videos|group_<path>")
        else:
            # Поддерживаем старый формат "all" и "unsorted" для совместимости
            if subtab_n in ("all", "unsorted"):
                subtab_n = "unsorted_photos"  # По умолчанию "Фото к разбору" для "Нет людей"
            elif subtab_n not in ("unsorted_photos", "unsorted_videos"):
                subtab_n = "unsorted_photos"
    else:
        subtab_n = "all"

    _agent_dbg(
        hypothesis_id="HUI_SUBTAB",
        location="web_api/routers/faces.py:api_faces_results",
        message="faces_results_request",
        data={"pipeline_run_id": int(pipeline_run_id), "tab": tab_n, "subtab": subtab_n, "page": int(page or 1), "page_size": int(page_size or 0)},
    )
    page_i = max(1, int(page or 1))
    size_i = max(10, min(200, int(page_size or 60)))
    offset = (page_i - 1) * size_i

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise HTTPException(status_code=400, detail="face_run_id is not set yet (step 3 not started)")
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")

    root_like = None
    if root_path.startswith("disk:"):
        rp = root_path.rstrip("/")
        root_like = rp + "/%"
    else:
        try:
            rp_abs = os.path.abspath(root_path)
            rp_abs = rp_abs.rstrip("\\/") + "\\"
            root_like = "local:" + rp_abs + "%"
        except Exception:
            root_like = None

    # Показываем файлы с faces_run_id = ? ИЛИ (видео без faces_run_id в корневой папке)
    # Это нужно, потому что видео могут быть не обработаны для детекции лиц
    where = ["f.status != 'deleted'"]
    params: list[Any] = []
    
    if root_like:
        # Показываем файлы с faces_run_id = ? ИЛИ видео без faces_run_id в корневой папке
        where.append("""
        (
          f.faces_run_id = ?
          OR (
            f.faces_run_id IS NULL 
            AND (COALESCE(f.media_type, '') = 'video' OR COALESCE(f.mime_type, '') LIKE 'video/%')
            AND f.path LIKE ?
          )
        )
        """)
        params.append(face_run_id_i)
        params.append(root_like)
    else:
        # Если нет root_like, используем только faces_run_id
        where.append("f.faces_run_id = ?")
        params.append(face_run_id_i)
    
    where_sql = " AND ".join(where)

    # ID персоны «Посторонний» — привязки к ней не переводят файл в «Люди»
    _conn = get_connection()
    ignored_person_id = -1
    try:
        from backend.common.db import get_outsider_person_id
        _val = get_outsider_person_id(_conn)
        if _val is not None:
            ignored_person_id = _val
    finally:
        try:
            _conn.close()
        except Exception:
            pass

    # Проверка привязок к персонам через прямоугольники и file_persons (используется в eff_sql)
    # Файл попадает в 'faces', если есть привязка к не-постороннему: rectangles (manual/кластеры) или file_persons.
    # Условие run_id/archive ИЛИ файл в текущем прогоне (faces_run_id) — чтобы учитывать привязки без run_id.
    has_person_binding_sql_results = """
    EXISTS (
        -- Ручные привязки (photo_rectangles.manual_person_id), исключая «Посторонний»
        SELECT 1 FROM photo_rectangles fr
        WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL AND fr.manual_person_id != ?
          AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
    ) OR EXISTS (
        -- Привязки через кластеры, исключая «Посторонний»
        SELECT 1 FROM photo_rectangles fr_cluster
        JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
        WHERE fr_cluster.file_id = f.id 
          AND (fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')
          AND COALESCE(fr_cluster.ignore_flag, 0) = 0
          AND fc.person_id IS NOT NULL AND fc.person_id != ?
          AND (fc.run_id = ? OR fc.archive_scope = 'archive')
    ) OR EXISTS (
        -- Прямая привязка файла к персоне (file_persons), исключая «Посторонний»
        SELECT 1 FROM file_persons fp
        WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL AND fp.person_id != ?
    )
    """

    # Привязка к персоне (file_persons или прямоугольники) — до проверки no_faces:
    # иначе файл с меткой «no_faces» попадает во вкладку «Нет людей», где «К разбору» не смотрит на file_persons.
    eff_sql = f"""
    CASE
      WHEN COALESCE(m.people_no_face_manual, 0) = 1 THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'faces' THEN 'faces'
      WHEN ({has_person_binding_sql_results}) THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'no_faces' THEN 'no_faces'
      WHEN COALESCE(m.quarantine_manual, 0) = 1 THEN 'no_faces'
      WHEN COALESCE(m.animals_manual, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.animals_auto, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.faces_auto_quarantine, 0) = 1
           AND COALESCE(f.faces_count, 0) > 0
           AND lower(trim(coalesce(f.faces_quarantine_reason, ''))) != 'many_small_faces'
        THEN 'no_faces'
      ELSE (CASE WHEN COALESCE(f.faces_count, 0) > 0 THEN 'faces' ELSE 'no_faces' END)
    END
    """

    sub_where = "1=1"
    sub_params: list[Any] = []
    person_filter_sql = "1=1"
    person_filter_params: list[Any] = []
    group_filter_sql = "1=1"
    group_filter_params: list[Any] = []
    media_filter_sql = "1=1"  # Фильтр по типу медиа (фото/видео)
    if tab_n == "faces" and subtab_n == "many_faces":
        # Файлы с >= 8 лицами И с хотя бы одним неназначенным прямоугольником
        sub_where = "COALESCE(faces_count, 0) >= 8"
        person_filter_sql = """
        EXISTS (
            SELECT 1 FROM photo_rectangles fr
            WHERE fr.file_id = f.id
              AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND fr.manual_person_id IS NULL
              AND (fr.cluster_id IS NULL OR NOT EXISTS (
                  SELECT 1 FROM face_clusters fc
                  WHERE fc.id = fr.cluster_id AND fc.person_id IS NOT NULL
              ))
        )
        """
        person_filter_params = [face_run_id_i]
    elif tab_n == "faces" and subtab_n == "all":
        sub_where = "COALESCE(faces_count, 0) < 8"
    elif tab_n == "faces" and subtab_n == "unsorted":
        # "К разбору": файлы без привязки к персонам ИЛИ люди без лиц (people_no_face_manual=1) без привязки.
        # Ветка people_no_face_manual: показывать только если нет привязки ни по прямоугольникам, ни по file_persons.
        sub_where = "COALESCE(faces_count, 0) < 8"
        # Условие «есть ручная привязка»: run_id/archive ИЛИ файл в текущем прогоне (faces_run_id).
        person_filter_sql = """
        (
          (
            NOT EXISTS (
                SELECT 1 FROM photo_rectangles fr
                WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL
                  AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
            )
            AND NOT EXISTS (
                SELECT 1 FROM photo_rectangles fr_cluster
                JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
                WHERE fr_cluster.file_id = f.id 
                  AND (fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')
                  AND COALESCE(fr_cluster.ignore_flag, 0) = 0
                  AND fc.person_id IS NOT NULL
                  AND (fc.run_id = ? OR fc.archive_scope = 'archive')
            )
            AND NOT EXISTS (
                SELECT 1 FROM file_persons fp
                WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
            )
          )
          OR (
            COALESCE(m.people_no_face_manual, 0) = 1
            AND NOT EXISTS (
                SELECT 1 FROM photo_rectangles fr
                WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL
                  AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
            )
            AND NOT EXISTS (
                SELECT 1 FROM file_persons fp
                WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
            )
          )
        )
        """
        person_filter_params = [face_run_id_i, face_run_id_i, face_run_id_i, face_run_id_i, int(pipeline_run_id), face_run_id_i, face_run_id_i, int(pipeline_run_id)]
    elif tab_n == "no_faces" and subtab_n in ("unsorted", "unsorted_photos", "unsorted_videos"):
        # "К разбору" для "Нет людей": файлы без группы
        group_filter_sql = """
        NOT EXISTS (
            SELECT 1 FROM file_groups fg
            WHERE fg.file_id = f.id AND fg.pipeline_run_id = ?
        )
        """
        group_filter_params = [int(pipeline_run_id)]
        
        # Фильтр по типу медиа для разделения фото и видео
        if subtab_n == "unsorted_photos":
            # Только фото: media_type = 'image' или пустой, и mime_type не начинается с 'video/'
            media_filter_sql = """
            (COALESCE(f.media_type, '') = 'image' OR COALESCE(f.media_type, '') = '')
            AND NOT (COALESCE(f.mime_type, '') LIKE 'video/%')
            """
        elif subtab_n == "unsorted_videos":
            # Только видео: media_type = 'video' или mime_type начинается с 'video/'
            media_filter_sql = """
            (COALESCE(f.media_type, '') = 'video' OR COALESCE(f.mime_type, '') LIKE 'video/%')
            """
        else:
            # Старый формат "unsorted" - показываем все
            media_filter_sql = "1=1"
    elif tab_n == "no_faces" and group_path_filter is not None:
        # Фильтр по конкретной группе
        # Ищем точно по названию (данные уже нормализованы в БД)
        group_filter_sql = """
        EXISTS (
            SELECT 1 FROM file_groups fg
            WHERE fg.file_id = f.id AND fg.pipeline_run_id = ? 
            AND fg.group_path = ?
        )
        """
        group_filter_params = [int(pipeline_run_id), str(group_path_filter)]
    elif tab_n == "faces" and person_id_filter is not None:
        # Для персоны «Посторонний» (person_6): только фото где ВСЕ прямоугольники либо посторонние, либо неназначенные + manual_filter
        if person_id_filter == ignored_person_id and ignored_person_id >= 0:
            # Подзакладка «Посторонние» в «Люди»: только фото где ВСЕ прямоугольники либо посторонние, либо неназначенные
            outsider_id = person_id_filter
            run_archive = "(fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')"
            no_other_sql = f"""
        NOT EXISTS (
            SELECT 1 FROM photo_rectangles fr
            WHERE fr.file_id = f.id AND {run_archive}
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND (
                  (fr.manual_person_id IS NOT NULL AND fr.manual_person_id != ?)
                  OR (fr.cluster_id IS NOT NULL AND EXISTS (
                      SELECT 1 FROM face_clusters fc WHERE fc.id = fr.cluster_id AND fc.person_id != ?
                  ))
              )
        )
        """
            has_outsider_or_unassigned_sql = f"""
        EXISTS (
            SELECT 1 FROM photo_rectangles fr
            LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
            WHERE fr.file_id = f.id AND {run_archive}
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND (
                  fr.manual_person_id = ?
                  OR (fc.person_id = ?)
                  OR (fr.manual_person_id IS NULL AND (fr.cluster_id IS NULL OR fc.person_id IS NULL))
              )
        )
        """
            person_filter_sql = "(" + no_other_sql + " AND " + has_outsider_or_unassigned_sql + ")"
            person_filter_params = [face_run_id_i, outsider_id, outsider_id, face_run_id_i, outsider_id, outsider_id]
            sub_where = "COALESCE(faces_count, 0) > 0"
            if manual_filter_n == "manual_only":
                person_filter_sql += """
            AND NOT EXISTS (
                SELECT 1 FROM photo_rectangles fr
                JOIN face_clusters fc ON fc.id = fr.cluster_id
                WHERE fr.file_id = f.id AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')
                  AND COALESCE(fr.ignore_flag, 0) = 0 AND fc.person_id = ?
            )
            """
                person_filter_params.extend([face_run_id_i, outsider_id])
            elif manual_filter_n == "no_manual":
                person_filter_sql += """
            AND NOT EXISTS (
                SELECT 1 FROM photo_rectangles fr
                WHERE fr.file_id = f.id AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')
                  AND COALESCE(fr.ignore_flag, 0) = 0 AND fr.manual_person_id = ?
            )
            """
                person_filter_params.extend([face_run_id_i, outsider_id])
        else:
            # Обычная персона: файл привязан к персоне через ручные привязки, кластеры или file_persons
            person_filter_sql = """
        EXISTS (
            SELECT 1 FROM photo_rectangles fr
            WHERE fr.file_id = f.id AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive') AND fr.manual_person_id = ?
        ) OR EXISTS (
            SELECT 1 FROM photo_rectangles fr_cluster
            JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
            WHERE fr_cluster.file_id = f.id 
              AND (fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')
              AND COALESCE(fr_cluster.ignore_flag, 0) = 0
              AND fc.person_id = ?
              AND (fc.run_id = ? OR fc.archive_scope = 'archive')
        ) OR EXISTS (
            SELECT 1 FROM file_persons fp
            WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id = ?
        )
        """
            person_filter_params = [face_run_id_i, person_id_filter, face_run_id_i, person_id_filter, face_run_id_i, int(pipeline_run_id), person_id_filter]

    def _norm_dt(s: str | None, *, is_to: bool) -> str | None:
        if not s:
            return None
        t = str(s).strip()
        if not t:
            return None
        if len(t) == 10 and t[4] == "-" and t[7] == "-":
            return (t + ("T23:59:59Z" if is_to else "T00:00:00Z"))
        return t

    dt_from = _norm_dt(from_ts, is_to=False)
    dt_to = _norm_dt(to_ts, is_to=True)
    if tab_n == "no_faces":
        if dt_from:
            where.append("COALESCE(f.taken_at,'') >= ?")
            params.append(dt_from)
        if dt_to:
            where.append("COALESCE(f.taken_at,'') <= ?")
            params.append(dt_to)
        where_sql = " AND ".join(where)

    # Для вкладки "Нет людей" добавляем сортировку по группам
    group_join = ""
    group_select = ""
    group_order = ""
    group_count_params = []
    if tab_n == "no_faces":
        # Добавляем LEFT JOIN с file_groups для получения group_path
        group_join = """
        LEFT JOIN (
            SELECT DISTINCT file_id, 
                   MIN(group_path) AS group_path
            FROM file_groups
            WHERE pipeline_run_id = ?
            GROUP BY file_id
        ) fg ON fg.file_id = f.id
        """
        group_select = ", COALESCE(fg.group_path, '') AS group_path"
        # Сортировка: сначала файлы с группами (по названию группы), потом без групп
        group_order = """
          (CASE WHEN COALESCE(fg.group_path, '') = '' THEN 1 ELSE 0 END) ASC,
          COALESCE(fg.group_path, '') ASC,
        """
        group_count_params = [int(pipeline_run_id)]

    # Параметры для проверки привязок к персонам в eff_sql:
    # manual: ignored, run_id, run_id (subquery); cluster: run_id, ignored, run_id; file_persons: pipeline_run_id, ignored
    person_binding_params_results = [
        ignored_person_id, face_run_id_i, face_run_id_i, face_run_id_i, ignored_person_id, face_run_id_i,
        int(pipeline_run_id), ignored_person_id,
    ]
    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        count_start = time.time()
        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id{group_join}
            WHERE {where_sql} AND ({eff_sql}) = ? AND ({sub_where}) AND ({person_filter_sql}) AND ({group_filter_sql}) AND ({media_filter_sql})
            """,
            group_count_params + [int(pipeline_run_id)] + params + person_binding_params_results + [tab_n] + sub_params + person_filter_params + group_filter_params,
        )
        total = int(cur.fetchone()[0] or 0)
        count_time = time.time() - count_start
        msg = f"[API] api_faces_results: COUNT запрос занял {count_time:.3f}с, total={total}"
        logger.info(msg)

        sort_n = str((sort or "").strip().lower() or "")
        select_start = time.time()
        cur.execute(
            f"""
            SELECT
              f.id AS file_id,
              f.path, f.name, f.parent_path, f.size, f.mime_type, f.media_type,
              COALESCE(f.taken_at, '') AS taken_at,
              COALESCE(f.place_country, '') AS place_country,
              COALESCE(f.place_city, '') AS place_city,
              COALESCE(f.faces_count, 0) AS faces_count,
              COALESCE(m.faces_manual_label, '') AS faces_manual_label,
              COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
              COALESCE(f.faces_auto_quarantine, 0) AS faces_auto_quarantine,
              COALESCE(f.faces_quarantine_reason, '') AS faces_quarantine_reason,
              COALESCE(f.animals_auto, 0) AS animals_auto,
              COALESCE(f.animals_kind, '') AS animals_kind,
              COALESCE(m.animals_manual, 0) AS animals_manual,
              COALESCE(m.animals_manual_kind, '') AS animals_manual_kind,
              COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual,
              COALESCE(m.people_no_face_person, '') AS people_no_face_person{group_select}
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id{group_join}
            WHERE {where_sql} AND ({eff_sql}) = ? AND ({sub_where}) AND ({person_filter_sql}) AND ({group_filter_sql}) AND ({media_filter_sql})
            ORDER BY
              {group_order}
              (CASE WHEN ? = 'place_date' AND (COALESCE(f.place_country,'') = '' AND COALESCE(f.place_city,'') = '') THEN 1 ELSE 0 END) ASC,
              (CASE WHEN ? = 'place_date' THEN COALESCE(f.place_country,'') ELSE '' END) ASC,
              (CASE WHEN ? = 'place_date' THEN COALESCE(f.place_city,'') ELSE '' END) ASC,
              (CASE WHEN ? = 'place_date' AND COALESCE(f.taken_at,'') = '' THEN 1 ELSE 0 END) ASC,
              (CASE WHEN ? = 'place_date' THEN COALESCE(f.taken_at,'') ELSE '' END) ASC,
              f.path ASC
            LIMIT ? OFFSET ?
            """,
            group_count_params
            + [int(pipeline_run_id)]
            + params
            + person_binding_params_results
            + [tab_n]
            + sub_params
            + person_filter_params
            + group_filter_params
            + [sort_n, sort_n, sort_n, sort_n, sort_n, size_i, offset],
        )
        rows = [dict(r) for r in cur.fetchall()]
        select_time = time.time() - select_start
        msg = f"[API] api_faces_results: SELECT запрос занял {select_time:.3f}с, строк: {len(rows)}"
        logger.info(msg)

        # Диагностика пустой вкладки персоны: при tab=faces, subtab=person_N и total=0
        # проверяем, сколько файлов в прогоне привязаны к персоне без учёта eff_sql
        if tab_n == "faces" and person_id_filter is not None and total == 0:
            try:
                diag_cur = ds.conn.cursor()
                diag_cur.execute(
                    f"""
                    SELECT COUNT(*) AS cnt
                    FROM files f
                    WHERE {where_sql} AND ({person_filter_sql})
                    """,
                    params + person_filter_params,
                )
                cnt_attached = int((diag_cur.fetchone() or [0])[0])
                _agent_dbg(
                    hypothesis_id="PERSON_TAB_EMPTY",
                    location="faces.py:api_faces_results",
                    message="person subtab empty diagnostic",
                    data={
                        "pipeline_run_id": int(pipeline_run_id),
                        "person_id": person_id_filter,
                        "total_from_full_query": total,
                        "cnt_files_in_run_attached_to_person_no_eff_sql": cnt_attached,
                        "interpretation": "cnt_attached>0 значит eff_sql отсекает; cnt_attached=0 значит where_sql или person_filter не находят файлов",
                    },
                )
            except Exception as e:
                _agent_dbg(
                    hypothesis_id="PERSON_TAB_EMPTY",
                    location="faces.py:api_faces_results",
                    message="person subtab diagnostic failed",
                    data={"person_id": person_id_filter, "error": str(e)},
                )
    finally:
        ds.close()

    gold_expected = gold_expected_tab_by_path(include_drawn_faces=False)
    
    # Группировка в поездки для tab=no_faces и sort=place_date
    if tab_n == "no_faces" and sort_n == "place_date":
        try:
            rows = _group_into_trips(rows)
        except Exception as e:
            logger.error(f"[API] api_faces_results: ошибка в _group_into_trips: {e}", exc_info=True)
            # В случае ошибки возвращаем строки без группировки
            rows = rows
    
    items: list[dict[str, Any]] = []
    for r in rows:
        path = str(r.get("path") or "")
        mime_type = str(r.get("mime_type") or "") or None
        media_type = str(r.get("media_type") or "") or None
        if not mime_type:
            guess_name = _strip_local_prefix(path) if path.startswith("local:") else _basename_from_disk_path(path)
            mt2, _enc = mimetypes.guess_type(guess_name)
            mime_type = mt2 or None
        if not media_type and mime_type:
            if mime_type.startswith("image/"):
                media_type = "image"
            elif mime_type.startswith("video/"):
                media_type = "video"

        items.append(
            {
                "path": path,
                "path_short": _short_path_for_ui(path) if path.startswith("disk:") else path,
                "size_human": _human_bytes(int(r["size"]) if r.get("size") is not None else None),
                "mime_type": mime_type,
                "media_type": media_type,
                "taken_at": (str(r.get("taken_at") or "") or "") or None,
                "place_country": (str(r.get("place_country") or "") or "") or None,
                "place_city": (str(r.get("place_city") or "") or "") or None,
                "faces_count": int(r.get("faces_count") or 0),
                "faces_manual_label": (str(r.get("faces_manual_label") or "") or ""),
                "quarantine_manual": int(r.get("quarantine_manual") or 0),
                "faces_auto_quarantine": int(r.get("faces_auto_quarantine") or 0),
                "faces_quarantine_reason": (str(r.get("faces_quarantine_reason") or "") or "") or None,
                "animals_auto": int(r.get("animals_auto") or 0),
                "animals_kind": (str(r.get("animals_kind") or "") or "") or None,
                "animals_manual": int(r.get("animals_manual") or 0),
                "animals_manual_kind": (str(r.get("animals_manual_kind") or "") or "") or None,
                "people_no_face_manual": int(r.get("people_no_face_manual") or 0),
                "people_no_face_person": (str(r.get("people_no_face_person") or "") or "") or None,
                "group_path": (str(r.get("group_path") or "") or "") or None if tab_n == "no_faces" else None,  # Группа для сортировки
                "trip_group": str(r.get("trip_group") or "") or None,  # Группа поездки для группировки на UI
                "trip_label": str(r.get("trip_label") or "") or None,  # Название поездки для отображения
                "sorted_past_gold": bool(gold_expected.get(path) is not None and gold_expected.get(path) != tab_n),
                "gold_expected_tab": gold_expected.get(path),
                **_faces_preview_meta(path=path, mime_type=mime_type, media_type=media_type, pipeline_run_id=int(pipeline_run_id)),
            }
        )

    elapsed = time.time() - start_time
    msg = f"[API] api_faces_results: завершено за {elapsed:.3f}с, элементов: {len(items)}, всего: {total}, tab={tab_n}, subtab={subtab_n}"
    logger.info(msg)

    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "face_run_id": face_run_id_i,
        "root_path": root_path,
        "tab": tab_n,
        "subtab": subtab_n,
        "page": page_i,
        "page_size": size_i,
        "total": total,
        "items": items,
    }


@router.get("/api/faces/tab-counts")
def api_faces_tab_counts(pipeline_run_id: int) -> dict[str, Any]:
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise HTTPException(status_code=400, detail="face_run_id is not set yet (step 3 not started)")
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")
    dedup_run_id = pr.get("dedup_run_id")

    where = ["f.status != 'deleted'"]
    params: list[Any] = []

    if dedup_run_id is not None:
        # Тот же набор, что в step4-report: inventory_scope='source' и last_run_id прогона
        where.append("COALESCE(f.inventory_scope, '') = 'source' AND f.last_run_id = ?")
        params.append(int(dedup_run_id))
    else:
        try:
            root_like = None
            if root_path.startswith("disk:"):
                rp = root_path.rstrip("/")
                root_like = rp + "/%"
            else:
                try:
                    rp_abs = os.path.abspath(root_path)
                    rp_abs = rp_abs.rstrip("\\/") + "\\"
                    root_like = "local:" + rp_abs + "%"
                except Exception:
                    root_like = None
            if root_like:
                where.append("""
        (
          f.faces_run_id = ?
          OR (
            f.faces_run_id IS NULL
            AND (COALESCE(f.media_type, '') = 'video' OR COALESCE(f.mime_type, '') LIKE 'video/%')
            AND f.path LIKE ?
          )
        )
        """)
                params.append(face_run_id_i)
                params.append(root_like)
            else:
                where.append("f.faces_run_id = ?")
                params.append(face_run_id_i)
        except Exception:
            where.append("f.faces_run_id = ?")
            params.append(face_run_id_i)

    where_sql = " AND ".join(where)

    try:
        # ID персоны «Посторонний» — привязки к ней не переводят файл в «Люди»
        _conn_tc = get_connection()
        ignored_person_id_tc = -1
        try:
            from backend.common.db import get_outsider_person_id
            _val_tc = get_outsider_person_id(_conn_tc)
            if _val_tc is not None:
                ignored_person_id_tc = _val_tc
        finally:
            try:
                _conn_tc.close()
            except Exception:
                pass

        # Проверка привязок к персонам (то же условие, что в api_faces_results: run_id/archive или файл в прогоне)
        has_person_binding_sql = """
    EXISTS (
        SELECT 1 FROM photo_rectangles fr
        WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL AND fr.manual_person_id != ?
          AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
    ) OR EXISTS (
        SELECT 1 FROM photo_rectangles fr_cluster
        JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
        WHERE fr_cluster.file_id = f.id 
          AND (fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')
          AND COALESCE(fr_cluster.ignore_flag, 0) = 0
          AND fc.person_id IS NOT NULL AND fc.person_id != ?
          AND (fc.run_id = ? OR fc.archive_scope = 'archive')
    ) OR EXISTS (
        SELECT 1 FROM file_persons fp
        WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL AND fp.person_id != ?
    )
    """

        # Порядок как в api_faces_results: привязка до no_faces
        eff_sql = f"""
    CASE
      WHEN COALESCE(m.people_no_face_manual, 0) = 1 THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'faces' THEN 'faces'
      WHEN ({has_person_binding_sql}) THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'no_faces' THEN 'no_faces'
      WHEN COALESCE(m.quarantine_manual, 0) = 1 THEN 'no_faces'
      WHEN COALESCE(m.animals_manual, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.animals_auto, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.faces_auto_quarantine, 0) = 1
           AND COALESCE(f.faces_count, 0) > 0
           AND lower(trim(coalesce(f.faces_quarantine_reason, ''))) != 'many_small_faces'
        THEN 'no_faces'
      ELSE (CASE WHEN COALESCE(f.faces_count, 0) > 0 THEN 'faces' ELSE 'no_faces' END)
    END
    """

        # Параметры для проверки привязок к персонам в eff_sql (8 плейсхолдеров)
        person_binding_params = [
            ignored_person_id_tc, face_run_id_i, face_run_id_i, face_run_id_i, ignored_person_id_tc, face_run_id_i,
            int(pipeline_run_id), ignored_person_id_tc,
        ]
        # Считаем вкладки 1-го уровня тремя отдельными COUNT (как many_faces/unsorted),
        # чтобы избежать проблем с порядком параметров при повторении eff_sql в GROUP BY.
        ds = DedupStore()
        try:
            cur = ds.conn.cursor()
            base_params = [int(pipeline_run_id)] + params + person_binding_params
            count_faces = 0
            count_no_faces = 0
            count_animals = 0
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM files f
                LEFT JOIN files_manual_labels m
                  ON m.pipeline_run_id = ? AND m.file_id = f.id
                WHERE {where_sql} AND ({eff_sql}) = 'faces'
                """,
                base_params,
            )
            count_faces = int(cur.fetchone()[0] or 0)
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM files f
                LEFT JOIN files_manual_labels m
                  ON m.pipeline_run_id = ? AND m.file_id = f.id
                WHERE {where_sql} AND ({eff_sql}) = 'no_faces'
                """,
                base_params,
            )
            count_no_faces = int(cur.fetchone()[0] or 0)
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM files f
                LEFT JOIN files_manual_labels m
                  ON m.pipeline_run_id = ? AND m.file_id = f.id
                WHERE {where_sql} AND ({eff_sql}) = 'animals'
                """,
                base_params,
            )
            count_animals = int(cur.fetchone()[0] or 0)
            counts = {
                "faces": count_faces,
                "no_faces": count_no_faces,
                "animals": count_animals,
                "quarantine": 0,
                "people_no_face": 0,
            }
            # «Много лиц»: только файлы с >= 8 лицами И с хотя бы одним неназначенным прямоугольником
            many_faces_filter_sql = """
                EXISTS (
                    SELECT 1 FROM photo_rectangles fr
                    WHERE fr.file_id = f.id
                      AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')
                      AND COALESCE(fr.ignore_flag, 0) = 0
                      AND fr.manual_person_id IS NULL
                      AND (fr.cluster_id IS NULL OR NOT EXISTS (
                          SELECT 1 FROM face_clusters fc
                          WHERE fc.id = fr.cluster_id AND fc.person_id IS NOT NULL
                      ))
                )
            """
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM files f
                LEFT JOIN files_manual_labels m
                  ON m.pipeline_run_id = ? AND m.file_id = f.id
                WHERE {where_sql}
                  AND ({eff_sql}) = 'faces'
                  AND COALESCE(f.faces_count, 0) >= 8
                  AND ({many_faces_filter_sql.strip()})
                """,
                [int(pipeline_run_id)] + params + person_binding_params + [face_run_id_i],
            )
            many_faces_cnt = int(cur.fetchone()[0] or 0)
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM files f
                LEFT JOIN files_manual_labels m
                  ON m.pipeline_run_id = ? AND m.file_id = f.id
                WHERE {where_sql}
                  AND ({eff_sql}) = 'faces'
                  AND COALESCE(f.faces_count, 0) < 8
                  AND (
                    -- Файл не привязан ни к одной персоне (run/archive или файл в прогоне)
                    (
                      NOT EXISTS (
                          SELECT 1 FROM photo_rectangles fr
                          WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL
                            AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM photo_rectangles fr
                          JOIN face_clusters fc ON fc.id = fr.cluster_id
                          WHERE fr.file_id = f.id AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive') AND fc.person_id IS NOT NULL
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM file_persons fp
                          WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
                      )
                    )
                    -- ИЛИ люди без лиц без привязки (ни по прямоугольникам, ни по file_persons)
                    OR (
                      COALESCE(m.people_no_face_manual, 0) = 1
                      AND NOT EXISTS (
                          SELECT 1 FROM photo_rectangles fr
                          WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL
                            AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM file_persons fp
                          WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
                      )
                    )
                  )
                """,
                [int(pipeline_run_id)] + params + person_binding_params + [face_run_id_i, face_run_id_i, face_run_id_i, int(pipeline_run_id), face_run_id_i, face_run_id_i, int(pipeline_run_id)],
            )
            unsorted_cnt = int(cur.fetchone()[0] or 0)
            # Счетчик для "Фото к разбору" в "Нет людей" (файлы без группы, только фото)
            query_params = [int(pipeline_run_id)] + params + person_binding_params + [int(pipeline_run_id)]
            try:
                cur.execute(
                    f"""
                    SELECT COUNT(*) AS cnt
                    FROM files f
                    LEFT JOIN files_manual_labels m
                      ON m.pipeline_run_id = ? AND m.file_id = f.id
                    WHERE {where_sql}
                      AND ({eff_sql}) = 'no_faces'
                      AND NOT EXISTS (
                          SELECT 1 FROM file_groups fg
                          WHERE fg.file_id = f.id AND fg.pipeline_run_id = ?
                      )
                      AND (COALESCE(f.media_type, '') = 'image' OR COALESCE(f.media_type, '') = '')
                      AND NOT (COALESCE(f.mime_type, '') LIKE 'video/%')
                    """,
                    query_params,
                )
            except Exception as e:
                logger.error(f"[API] api_faces_tab_counts: ошибка в запросе unsorted_photos: {e}, params count: {len(query_params)}, where_sql: {where_sql[:200]}")
                raise
            no_faces_unsorted_photos_cnt = int(cur.fetchone()[0] or 0)
            query_params_videos = [int(pipeline_run_id)] + params + person_binding_params + [int(pipeline_run_id)]
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM files f
                LEFT JOIN files_manual_labels m
                  ON m.pipeline_run_id = ? AND m.file_id = f.id
                WHERE {where_sql}
                  AND ({eff_sql}) = 'no_faces'
                  AND NOT EXISTS (
                      SELECT 1 FROM file_groups fg
                      WHERE fg.file_id = f.id AND fg.pipeline_run_id = ?
                  )
                  AND (COALESCE(f.media_type, '') = 'video' OR COALESCE(f.mime_type, '') LIKE 'video/%')
                """,
                query_params_videos,
            )
            no_faces_unsorted_videos_cnt = int(cur.fetchone()[0] or 0)
        finally:
            ds.close()
        return {
            "ok": True,
            "pipeline_run_id": int(pipeline_run_id),
            "counts": counts,
            "subcounts": {
                "faces": {"many_faces": many_faces_cnt, "unsorted": unsorted_cnt},
                "no_faces": {
                    "unsorted_photos": no_faces_unsorted_photos_cnt,
                    "unsorted_videos": no_faces_unsorted_videos_cnt
                }
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/faces/persons-with-files")
def api_faces_persons_with_files(pipeline_run_id: int) -> dict[str, Any]:
    """
    Возвращает список персон, у которых есть файлы в данном прогоне.
    Используется для отображения подзакладок по персонам в закладке "Люди".
    """
    start_time = time.time()
    msg = f"[API] api_faces_persons_with_files: начало, pipeline_run_id={pipeline_run_id}"
    logger.info(msg)
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise HTTPException(status_code=400, detail="face_run_id is not set yet (step 3 not started)")
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")

    root_like = None
    if root_path.startswith("disk:"):
        rp = root_path.rstrip("/")
        root_like = rp + "/%"
    else:
        try:
            rp_abs = os.path.abspath(root_path)
            rp_abs = rp_abs.rstrip("\\/") + "\\"
            root_like = "local:" + rp_abs + "%"
        except Exception:
            root_like = None

    # Условие «файл из текущего прогона и не удалён» — как во вкладке (счётчик совпадает с ней)
    file_scope_parts = ["f.status != 'deleted'"]
    if root_like:
        file_scope_parts.append(
            "(f.faces_run_id = ? OR (f.faces_run_id IS NULL AND (COALESCE(f.media_type, '') = 'video' OR COALESCE(f.mime_type, '') LIKE 'video/%') AND f.path LIKE ?))"
        )
        file_scope_params = [face_run_id_i, root_like]
    else:
        file_scope_parts.append("f.faces_run_id = ?")
        file_scope_params = [face_run_id_i]
    file_scope_sql = " AND ".join(file_scope_parts)

    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Персоны с файлами только из текущего прогона и не удалённые (как во вкладке)
        # 1. Через ручные привязки (photo_rectangles.manual_person_id)
        where_parts1 = [
            "(fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')",
            "fr.manual_person_id IS NOT NULL",
            file_scope_sql,
        ]
        params1 = [face_run_id_i] + file_scope_params
        
        query1_start = time.time()
        cur.execute(
            f"""
            SELECT DISTINCT fr.manual_person_id AS person_id, p.name AS person_name, COUNT(DISTINCT f.path) AS files_count
            FROM photo_rectangles fr
            JOIN files f ON f.id = fr.file_id
            LEFT JOIN persons p ON p.id = fr.manual_person_id
            WHERE {" AND ".join(where_parts1)}
            GROUP BY fr.manual_person_id, p.name
            """,
            params1,
        )
        persons_from_faces = {r["person_id"]: {"id": r["person_id"], "name": r["person_name"], "files_count": int(r["files_count"] or 0)} for r in cur.fetchall()}
        query1_time = time.time() - query1_start
        msg = f"[API] api_faces_persons_with_files: запрос 1 (manual_person_id) занял {query1_time:.3f}с, персон: {len(persons_from_faces)}"
        logger.info(msg)

        # 1b. Через кластеры — только файлы текущего прогона и не удалённые
        file_scope_sql_cluster = file_scope_sql.replace("f.", "f_cluster.")
        where_parts_cluster = [
            "(fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')",
            "COALESCE(fr_cluster.ignore_flag, 0) = 0",
            "fc.person_id IS NOT NULL",
            "(fc.run_id = ? OR fc.archive_scope = 'archive')",
            file_scope_sql_cluster,
        ]
        params_cluster = [face_run_id_i, face_run_id_i] + file_scope_params
        
        query2_start = time.time()
        cur.execute(
            f"""
            SELECT 
                fc.person_id,
                p.name AS person_name,
                COUNT(DISTINCT f_cluster.path) AS files_count
            FROM photo_rectangles fr_cluster
            JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
            JOIN files f_cluster ON f_cluster.id = fr_cluster.file_id
            LEFT JOIN persons p ON p.id = fc.person_id
            WHERE {" AND ".join(where_parts_cluster)}
              AND NOT EXISTS (
                  SELECT 1 FROM photo_rectangles fr_direct
                  WHERE fr_direct.id = fr_cluster.id
                    AND fr_direct.manual_person_id = fc.person_id
              )
            GROUP BY fc.person_id, p.name
            """,
            params_cluster,
        )
        persons_from_clusters = {r["person_id"]: {"id": r["person_id"], "name": r["person_name"], "files_count": int(r["files_count"] or 0)} for r in cur.fetchall()}
        query2_time = time.time() - query2_start
        msg = f"[API] api_faces_persons_with_files: запрос 2 (clusters) занял {query2_time:.3f}с, персон: {len(persons_from_clusters)}"
        logger.info(msg)

        # 2. person_rectangles удалена — персоны только через лица, кластеры и file_persons
        persons_from_rects: dict[int, dict[str, Any]] = {}

        # 3. Прямые привязки (file_persons) — только файлы текущего прогона и не удалённые
        file_scope_sql_fp = file_scope_sql.replace("f.", "f_fp.")
        where_parts3 = [
            "fp.pipeline_run_id = ?",
            "fp.person_id IS NOT NULL",
            file_scope_sql_fp,
        ]
        params3 = [int(pipeline_run_id)] + file_scope_params
        
        query4_start = time.time()
        cur.execute(
            f"""
            SELECT DISTINCT fp.person_id, p.name AS person_name, COUNT(DISTINCT f_fp.path) AS files_count
            FROM file_persons fp
            JOIN files f_fp ON f_fp.id = fp.file_id
            LEFT JOIN persons p ON p.id = fp.person_id
            WHERE {" AND ".join(where_parts3)}
            GROUP BY fp.person_id, p.name
            """,
            params3,
        )
        persons_from_direct = {r["person_id"]: {"id": r["person_id"], "name": r["person_name"], "files_count": int(r["files_count"] or 0)} for r in cur.fetchall()}
        query4_time = time.time() - query4_start
        msg = f"[API] api_faces_persons_with_files: запрос 4 (file_persons) занял {query4_time:.3f}с, персон: {len(persons_from_direct)}"
        logger.info(msg)

        # ID персоны «Посторонний» для флага is_ignored
        from backend.common.db import get_outsider_person_id
        _conn_apwf = get_connection()
        outsider_id_apwf = None
        try:
            outsider_id_apwf = get_outsider_person_id(_conn_apwf)
        finally:
            try:
                _conn_apwf.close()
            except Exception:
                pass

        # Объединяем все персоны и суммируем количество файлов
        all_persons: dict[int, dict[str, Any]] = {}
        for pid, pdata in persons_from_faces.items():
            all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"], "is_ignored": pid == outsider_id_apwf}
        for pid, pdata in persons_from_clusters.items():
            if pid in all_persons:
                all_persons[pid]["files_count"] += pdata["files_count"]
            else:
                all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"], "is_ignored": pid == outsider_id_apwf}
        for pid, pdata in persons_from_rects.items():
            if pid in all_persons:
                all_persons[pid]["files_count"] += pdata["files_count"]
            else:
                all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"], "is_ignored": pid == outsider_id_apwf}
        for pid, pdata in persons_from_direct.items():
            if pid in all_persons:
                all_persons[pid]["files_count"] += pdata["files_count"]
            else:
                all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"], "is_ignored": pid == outsider_id_apwf}

        # Для персоны «Посторонний» счётчик должен совпадать с содержимым вкладки:
        # только файлы, где ВСЕ прямоугольники — посторонние или неназначенные,
        # и (eff_sql) = 'faces' — как в api_faces_results
        if outsider_id_apwf is not None and outsider_id_apwf in all_persons:
            run_archive_apwf = "(fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')"
            no_other_apwf = f"""
                NOT EXISTS (
                    SELECT 1 FROM photo_rectangles fr
                    WHERE fr.file_id = f.id AND {run_archive_apwf}
                      AND COALESCE(fr.ignore_flag, 0) = 0
                      AND (
                          (fr.manual_person_id IS NOT NULL AND fr.manual_person_id != ?)
                          OR (fr.cluster_id IS NOT NULL AND EXISTS (
                              SELECT 1 FROM face_clusters fc WHERE fc.id = fr.cluster_id AND fc.person_id != ?
                          ))
                      )
                )
            """
            has_outsider_or_unassigned_apwf = f"""
                EXISTS (
                    SELECT 1 FROM photo_rectangles fr
                    LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
                    WHERE fr.file_id = f.id AND {run_archive_apwf}
                      AND COALESCE(fr.ignore_flag, 0) = 0
                      AND (
                          fr.manual_person_id = ?
                          OR (fc.person_id = ?)
                          OR (fr.manual_person_id IS NULL AND (fr.cluster_id IS NULL OR fc.person_id IS NULL))
                      )
                )
            """
            # eff_sql — тот же CASE, что в api_faces_results: вкладка «Люди» показывает только файлы с eff_sql = 'faces'
            has_person_binding_apwf = """
                EXISTS (
                    SELECT 1 FROM photo_rectangles fr
                    WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL AND fr.manual_person_id != ?
                      AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
                ) OR EXISTS (
                    SELECT 1 FROM photo_rectangles fr_cluster
                    JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
                    WHERE fr_cluster.file_id = f.id 
                      AND (fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')
                      AND COALESCE(fr_cluster.ignore_flag, 0) = 0
                      AND fc.person_id IS NOT NULL AND fc.person_id != ?
                      AND (fc.run_id = ? OR fc.archive_scope = 'archive')
                ) OR EXISTS (
                    SELECT 1 FROM file_persons fp
                    WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL AND fp.person_id != ?
                )
            """
            eff_sql_apwf = f"""
                CASE
                  WHEN COALESCE(m.people_no_face_manual, 0) = 1 THEN 'faces'
                  WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'faces' THEN 'faces'
                  WHEN ({has_person_binding_apwf}) THEN 'faces'
                  WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'no_faces' THEN 'no_faces'
                  WHEN COALESCE(m.quarantine_manual, 0) = 1 THEN 'no_faces'
                  WHEN COALESCE(m.animals_manual, 0) = 1 THEN 'animals'
                  WHEN COALESCE(f.animals_auto, 0) = 1 THEN 'animals'
                  WHEN COALESCE(f.faces_auto_quarantine, 0) = 1
                       AND COALESCE(f.faces_count, 0) > 0
                       AND lower(trim(coalesce(f.faces_quarantine_reason, ''))) != 'many_small_faces'
                    THEN 'no_faces'
                  ELSE (CASE WHEN COALESCE(f.faces_count, 0) > 0 THEN 'faces' ELSE 'no_faces' END)
                END
            """
            # Параметры: JOIN m (pipeline_run_id), file_scope, eff_sql (8), 'faces', no_other+has_outsider (6)
            person_binding_apwf = [
                outsider_id_apwf, face_run_id_i, face_run_id_i, face_run_id_i, outsider_id_apwf, face_run_id_i,
                int(pipeline_run_id), outsider_id_apwf,
            ]
            outsider_count_params = (
                [int(pipeline_run_id)]
                + file_scope_params
                + person_binding_apwf
                + ["faces"]
                + [
                    face_run_id_i, outsider_id_apwf, outsider_id_apwf,
                    face_run_id_i, outsider_id_apwf, outsider_id_apwf,
                ]
            )
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM files f
                LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.file_id = f.id
                WHERE {file_scope_sql}
                  AND ({eff_sql_apwf}) = ?
                  AND COALESCE(f.faces_count, 0) > 0
                  AND ({no_other_apwf} AND {has_outsider_or_unassigned_apwf})
                """,
                outsider_count_params,
            )
            outsider_count = int(cur.fetchone()[0] or 0)
            all_persons[outsider_id_apwf]["files_count"] = outsider_count

        # Сортируем по имени (Посторонние в конце — в loadPersonsSubtabs на фронте)
        persons_list = sorted(all_persons.values(), key=lambda x: (x["name"] or "").lower())
        
        elapsed = time.time() - start_time
        msg = f"[API] api_faces_persons_with_files: завершено за {elapsed:.3f}с, персон: {len(persons_list)}"
        logger.info(msg)

    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "persons": persons_list,
    }


@router.get("/api/faces/all-persons-faces")
def api_faces_all_persons_faces(pipeline_run_id: int) -> dict[str, Any]:
    """
    Возвращает данные для всех персон (кроме Посторонних) из прогона.
    Каждая персона содержит список лиц (кропов) из «Сортируется -> Через лица».
    Используется для закладки «Лица» — сетка кропов с кнопками по персонам.
    """
    start_time = time.time()
    from backend.common.db import get_outsider_person_id

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise HTTPException(status_code=400, detail="face_run_id is not set yet (step 3 not started)")
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")

    root_like = None
    if root_path.startswith("disk:"):
        rp = root_path.rstrip("/")
        root_like = rp + "/%"
    else:
        try:
            rp_abs = os.path.abspath(root_path)
            rp_abs = rp_abs.rstrip("\\/") + "\\"
            root_like = "local:" + rp_abs + "%"
        except Exception:
            root_like = None

    where = ["f.status != 'deleted'"]
    params: list[Any] = []
    if root_like:
        where.append(
            "(f.faces_run_id = ? OR (f.faces_run_id IS NULL AND (COALESCE(f.media_type, '') = 'video' OR COALESCE(f.mime_type, '') LIKE 'video/%') AND f.path LIKE ?))"
        )
        params.extend([face_run_id_i, root_like])
    else:
        where.append("f.faces_run_id = ?")
        params.append(face_run_id_i)
    where_sql = " AND ".join(where)

    _conn_apf = get_connection()
    ignored_person_id_apf = -1
    try:
        _val = get_outsider_person_id(_conn_apf)
        if _val is not None:
            ignored_person_id_apf = _val
    finally:
        try:
            _conn_apf.close()
        except Exception:
            pass

    has_person_binding_sql_apf = """
    EXISTS (
        SELECT 1 FROM photo_rectangles fr
        WHERE fr.file_id = f.id AND fr.manual_person_id IS NOT NULL AND fr.manual_person_id != ?
          AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive' OR (SELECT f2.faces_run_id FROM files f2 WHERE f2.id = fr.file_id) = ?)
    ) OR EXISTS (
        SELECT 1 FROM photo_rectangles fr_cluster
        JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
        WHERE fr_cluster.file_id = f.id 
          AND (fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')
          AND COALESCE(fr_cluster.ignore_flag, 0) = 0
          AND fc.person_id IS NOT NULL AND fc.person_id != ?
          AND (fc.run_id = ? OR fc.archive_scope = 'archive')
    ) OR EXISTS (
        SELECT 1 FROM file_persons fp
        WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL AND fp.person_id != ?
    )
    """
    eff_sql_apf = f"""
    CASE
      WHEN COALESCE(m.people_no_face_manual, 0) = 1 THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'faces' THEN 'faces'
      WHEN ({has_person_binding_sql_apf}) THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'no_faces' THEN 'no_faces'
      WHEN COALESCE(m.quarantine_manual, 0) = 1 THEN 'no_faces'
      WHEN COALESCE(m.animals_manual, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.animals_auto, 0) = 1 THEN 'animals'
      WHEN COALESCE(f.faces_auto_quarantine, 0) = 1
           AND COALESCE(f.faces_count, 0) > 0
           AND lower(trim(coalesce(f.faces_quarantine_reason, ''))) != 'many_small_faces'
        THEN 'no_faces'
      ELSE (CASE WHEN COALESCE(f.faces_count, 0) > 0 THEN 'faces' ELSE 'no_faces' END)
    END
    """
    person_binding_params_apf = [
        ignored_person_id_apf, face_run_id_i, face_run_id_i, face_run_id_i, ignored_person_id_apf, face_run_id_i,
        int(pipeline_run_id), ignored_person_id_apf,
    ]

    person_filter_sql_one = """
    EXISTS (
        SELECT 1 FROM photo_rectangles fr
        WHERE fr.file_id = f.id AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive') AND fr.manual_person_id = ?
    ) OR EXISTS (
        SELECT 1 FROM photo_rectangles fr_cluster
        JOIN face_clusters fc ON fc.id = fr_cluster.cluster_id
        WHERE fr_cluster.file_id = f.id 
          AND (fr_cluster.run_id = ? OR COALESCE(TRIM(fr_cluster.archive_scope), '') = 'archive')
          AND COALESCE(fr_cluster.ignore_flag, 0) = 0
          AND fc.person_id = ?
          AND (fc.run_id = ? OR fc.archive_scope = 'archive')
    ) OR EXISTS (
        SELECT 1 FROM file_persons fp
        WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id = ?
    )
    """

    data_persons = api_faces_persons_with_files(pipeline_run_id=int(pipeline_run_id))
    persons_list = data_persons.get("persons") or []
    persons_list = [p for p in persons_list if p.get("id") != ignored_person_id_apf]

    # Лица (кропы) прогона: photo_rectangles с run_id = face_run_id, привязка через кластер или manual_person_id
    conn_apf = get_connection()
    result_persons = []
    try:
        cur = conn_apf.cursor()
        cur.execute(
            """
            SELECT
              fr.id AS face_id,
              fr.file_id,
              fr.face_index,
              fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
              fr.thumb_jpeg,
              fr.manual_person_id,
              fr.cluster_id,
              f.path AS file_path,
              fc.person_id AS cluster_person_id
            FROM photo_rectangles fr
            LEFT JOIN files f ON fr.file_id = f.id
            LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
            WHERE fr.run_id = ? AND fr.is_face = 1 AND COALESCE(fr.ignore_flag, 0) = 0
              AND (f.status IS NULL OR f.status != 'deleted')
              AND (fr.manual_person_id IS NOT NULL OR fr.cluster_id IS NOT NULL)
            ORDER BY f.path ASC, fr.face_index ASC
            """,
            (face_run_id_i,),
        )
        rows = cur.fetchall()
        by_person: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            row = dict(r)
            pid = row.get("manual_person_id") or row.get("cluster_person_id")
            if pid is None or pid == ignored_person_id_apf:
                continue
            thumb_b64 = None
            if row.get("thumb_jpeg"):
                thumb_b64 = base64.b64encode(row["thumb_jpeg"]).decode("utf-8")
            bbox = None
            bx, by, bw, bh = row.get("bbox_x"), row.get("bbox_y"), row.get("bbox_w"), row.get("bbox_h")
            if bx is not None and by is not None and bw is not None and bh is not None:
                bbox = {"x": float(bx), "y": float(by), "w": float(bw), "h": float(bh)}
            assignment_type = "manual_face" if row.get("manual_person_id") else "cluster"
            face_item = {
                "face_id": row["face_id"],
                "file_id": row.get("file_id"),
                "file_path": row.get("file_path") or None,
                "thumb_jpeg_base64": thumb_b64,
                "bbox": bbox,
                "assignment_type": assignment_type,
            }
            if pid not in by_person:
                by_person[pid] = []
            by_person[pid].append(face_item)
        for p in persons_list:
            person_id = p.get("id")
            person_name = p.get("name") or f"Персона {person_id}"
            if person_id is None:
                continue
            result_persons.append({
                "person_id": person_id,
                "person_name": person_name,
                "faces": by_person.get(person_id, []),
            })
    finally:
        conn_apf.close()

    elapsed = time.time() - start_time
    logger.info(f"[API] api_faces_all_persons_faces: завершено за {elapsed:.3f}с, персон: {len(result_persons)}")
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "persons": result_persons,
    }


@router.get("/api/faces/groups-with-files")
def api_faces_groups_with_files(pipeline_run_id: int) -> dict[str, Any]:
    """
    Возвращает список групп с количеством файлов для прогона.
    Используется для отображения подзакладок в закладке "Нет людей".
    """
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    
    fs = FaceStore()
    try:
        groups = fs.list_file_groups_with_counts(pipeline_run_id=int(pipeline_run_id))
        # Возвращаем группы как есть, без нормализации "налету"
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "groups": groups,
    }


@router.post("/api/faces/assign-group")
def api_faces_assign_group(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Назначает файл в группу (file_groups).
    Параметры: pipeline_run_id, path, group_path
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    path = payload.get("path")
    group_path = payload.get("group_path")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(path, str) or not (path.startswith("local:") or path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="path must start with local: or disk:")
    if not isinstance(group_path, str) or not group_path.strip():
        raise HTTPException(status_code=400, detail="group_path is required")
    
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    
    fs = FaceStore()
    try:
        # Сохраняем group_path как есть (данные должны быть нормализованы в БД через миграцию)
        normalized_group_path = str(group_path).strip()
        
        # Проверяем, что группа не пустая
        if not normalized_group_path:
            raise HTTPException(status_code=400, detail="group_path cannot be empty")

        try:
            fs.insert_file_group(
                pipeline_run_id=int(pipeline_run_id),
                file_path=str(path),
                group_path=normalized_group_path,
            )
        except Exception as e:
            logger.exception("api_faces_assign_group: INSERT failed: %s", e)
            raise

        # Проверяем, что группа действительно сохранилась
        groups = fs.list_file_groups(
            pipeline_run_id=int(pipeline_run_id),
            file_path=str(path),
        )
        saved = any(g.get("group_path") == normalized_group_path for g in groups)
        if not saved:
            raise HTTPException(status_code=500, detail=f"Failed to save group assignment. Expected: '{normalized_group_path}', found: {[g.get('group_path') for g in groups]}")
    finally:
        fs.close()
    
    return {"ok": True, "group_path": normalized_group_path}


@router.post("/api/faces/remove-group")
def api_faces_remove_group(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Удаляет файл из группы.
    Параметры: pipeline_run_id, path, group_path
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    path = payload.get("path")
    group_path = payload.get("group_path")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(path, str) or not (path.startswith("local:") or path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="path must start with local: or disk:")
    if not isinstance(group_path, str):
        raise HTTPException(status_code=400, detail="group_path is required")
    
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    
    fs = FaceStore()
    try:
        fs.delete_file_group(
            pipeline_run_id=int(pipeline_run_id),
            file_path=str(path),
            group_path=str(group_path),
        )
    finally:
        fs.close()
    
    return {"ok": True}


@router.get("/api/faces/rectangles")
def api_faces_rectangles(pipeline_run_id: int | None = None, file_id: int | None = None, path: str | None = None) -> dict[str, Any]:
    """
    Возвращает список rectangles для файла.
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Если pipeline_run_id не передан, загружает все rectangles для файла (для архивных фотографий).
    """
    from backend.common.db import _get_file_id, get_connection
    
    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    
    face_run_id_i = None
    
    # Если pipeline_run_id передан, получаем face_run_id
    if pipeline_run_id is not None:
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        finally:
            ps.close()
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        face_run_id = pr.get("face_run_id")
        if not face_run_id:
            raise HTTPException(status_code=400, detail="face_run_id is not set yet")
        face_run_id_i = int(face_run_id)
    
    fs = FaceStore()
    try:
        # file_id передаётся явно при открытии карточки из списка — используем как есть
        if file_id is not None:
            resolved_file_id = int(file_id)
        else:
            resolved_file_id = _get_file_id(fs.conn, file_id=None, file_path=path)
            # Fallback: на Windows путь может прийти с backslash, в БД — с forward slash
            if resolved_file_id is None and path and str(path).strip().startswith("local:"):
                path_norm = str(path).replace("\\", "/")
                if path_norm != path:
                    resolved_file_id = _get_file_id(fs.conn, file_id=None, file_path=path_norm)
        if resolved_file_id is None:
            raise HTTPException(status_code=404, detail="File not found")

        cur = fs.conn.cursor()
        # Если pipeline_run_id не передан — загружаем все rectangles для файла (архив)
        # Если передан — загружаем rectangles прогона И архива (run_id = ? OR archive_scope = 'archive'),
        # чтобы в карточке были видны ручные привязки и из прогона, и из архива (как в eff_sql).
        if face_run_id_i is None:
            cur.execute(
                """
                SELECT
                  fr.id, fr.run_id, f.path AS file_path, fr.face_index,
                  fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                  fr.confidence, fr.presence_score,
                  fr.manual_person, fr.ignore_flag,
                  fr.created_at,
                  COALESCE(fr.is_manual, 0) AS is_manual,
                  fr.manual_created_at,
                  COALESCE(fr.is_face, 1) AS is_face
                FROM photo_rectangles fr
                JOIN files f ON f.id = fr.file_id
                WHERE fr.file_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
                ORDER BY COALESCE(fr.is_manual, 0) ASC, fr.face_index ASC, fr.id ASC
                """,
                (resolved_file_id,),
            )
            rects = [dict(r) for r in cur.fetchall()]
        else:
            cur.execute(
                """
                SELECT
                  fr.id, fr.run_id, f.path AS file_path, fr.face_index,
                  fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                  fr.confidence, fr.presence_score,
                  fr.manual_person, fr.ignore_flag,
                  fr.created_at,
                  COALESCE(fr.is_manual, 0) AS is_manual,
                  fr.manual_created_at,
                  COALESCE(fr.is_face, 1) AS is_face
                FROM photo_rectangles fr
                JOIN files f ON f.id = fr.file_id
                WHERE fr.file_id = ?
                  AND (fr.run_id = ? OR COALESCE(TRIM(fr.archive_scope), '') = 'archive')
                  AND COALESCE(fr.ignore_flag, 0) = 0
                ORDER BY COALESCE(fr.is_manual, 0) ASC, fr.face_index ASC, fr.id ASC
                """,
                (resolved_file_id, face_run_id_i),
            )
            rects = [dict(r) for r in cur.fetchall()]
        
        # Добавляем информацию о персонах для каждого прямоугольника
        # face_clusters, manual_person_id в photo_rectangles и persons находятся в FaceStore
        fs_conn = fs.conn
        fs_cur = fs_conn.cursor()
        
        # Получаем информацию о персонах для каждого прямоугольника
        for rect in rects:
                rect_id = rect.get("id")
                if not rect_id:
                    continue
                
                # Ищем персону через кластеры (photo_rectangles.cluster_id)
                fs_cur.execute("""
                    SELECT fc.person_id
                    FROM photo_rectangles fr
                    JOIN face_clusters fc ON fc.id = fr.cluster_id
                    WHERE fr.id = ? AND fc.person_id IS NOT NULL
                    LIMIT 1
                """, (rect_id,))
                cluster_row = fs_cur.fetchone()
                person_id = cluster_row["person_id"] if cluster_row else None
                assignment_type = "cluster" if person_id else None
                
                # Если не нашли через кластер, ищем через ручную привязку (photo_rectangles.manual_person_id)
                if not person_id:
                    fs_cur.execute("""
                        SELECT manual_person_id AS person_id
                        FROM photo_rectangles
                        WHERE id = ? AND manual_person_id IS NOT NULL
                        LIMIT 1
                    """, (rect_id,))
                    manual_row = fs_cur.fetchone()
                    person_id = manual_row["person_id"] if manual_row else None
                    assignment_type = "manual_face" if person_id else None
                
                # Если нашли person_id, получаем имя из FaceStore (persons находится там же)
                if person_id:
                    fs_cur.execute("SELECT name, is_me FROM persons WHERE id = ?", (person_id,))
                    person_row = fs_cur.fetchone()
                    rect["person_id"] = person_id
                    rect["person_name"] = person_row["name"] if person_row else None
                    rect["is_me"] = bool(person_row["is_me"]) if person_row and person_row["is_me"] else False
                else:
                    rect["person_id"] = None
                    rect["person_name"] = None
                    rect["is_me"] = False
                
                # Добавляем тип привязки
                rect["assignment_type"] = assignment_type
    finally:
        fs.close()
    
    # Получаем размеры изображения и EXIF orientation из таблицы files
    image_width = None
    image_height = None
    exif_orientation = None
    if file_id or path:
        conn = get_connection()
        try:
            resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
            if resolved_file_id:
                cur = conn.cursor()
                cur.execute(
                    "SELECT image_width, image_height, exif_orientation FROM files WHERE id = ?",
                    (resolved_file_id,)
                )
                file_row = cur.fetchone()
                if file_row:
                    image_width = file_row["image_width"]
                    image_height = file_row["image_height"]
                    exif_orientation = file_row["exif_orientation"]
        finally:
            conn.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id) if pipeline_run_id is not None else None,
        "run_id": face_run_id_i,
        "path": path,
        "rectangles": rects,
        "image_width": image_width,
        "image_height": image_height,
        "exif_orientation": exif_orientation
    }


@router.post("/api/faces/assign-face-person")
def api_faces_assign_face_person(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Назначает персону лицу (rectangle_id).
    
    Параметры:
    - pipeline_run_id: int (обязательно)
    - rectangle_id: int (обязательно)
    - person_id: int (обязательно)
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    rectangle_id = payload.get("rectangle_id")
    person_id = payload.get("person_id")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required and must be int")
    if not isinstance(rectangle_id, int):
        raise HTTPException(status_code=400, detail="rectangle_id is required and must be int")
    if not isinstance(person_id, int):
        raise HTTPException(status_code=400, detail="person_id is required and must be int")
    
    # Проверяем, что pipeline_run_id существует, и берём face_run_id для привязки к прогону
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        face_run_id = pr.get("face_run_id")
    finally:
        ps.close()

    face_run_id_i = int(face_run_id) if face_run_id else None

    # Проверяем, что лицо существует
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        cur.execute("SELECT id FROM photo_rectangles WHERE id = ?", (int(rectangle_id),))
        face_row = cur.fetchone()
        if not face_row:
            raise HTTPException(status_code=404, detail="rectangle_id not found")
        
        # Проверяем, что персона существует
        cur.execute("SELECT id, name FROM persons WHERE id = ?", (int(person_id),))
        person_row = cur.fetchone()
        if not person_row:
            raise HTTPException(status_code=404, detail="person_id not found")
        
        # Создаём или обновляем ручную привязку. run_id = face_run_id задаём, чтобы
        # фильтр «К разбору» (fr.run_id = ? OR archive) видел эту привязку и исключал файл из списка.
        if face_run_id_i is not None:
            cur.execute("""
                UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL, run_id = ? WHERE id = ?
            """, (int(person_id), face_run_id_i, int(rectangle_id)))
        else:
            cur.execute("""
                UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL WHERE id = ?
            """, (int(person_id), int(rectangle_id)))
        set_file_processed(conn, rectangle_id=int(rectangle_id))
        conn.commit()
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "rectangle_id": int(rectangle_id),
        "person_id": int(person_id),
        "person_name": person_row["name"],
    }


@router.get("/api/faces/file-persons")
def api_faces_file_persons(
    file_id: int | None = Query(None),
    path: str | None = Query(None),
    pipeline_run_id: int | None = Query(None),
) -> dict[str, Any]:
    """
    Возвращает список персон, привязанных к файлу (через любые способы), с информацией о лицах.
    Приоритет: file_id (если передан), иначе path (если передан).
    pipeline_run_id опционален: при отсутствии вычисляется по file_id (file_persons или photo_rectangles→pipeline_runs).
    """
    from backend.common.db import _get_file_id, get_connection

    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")

    conn = get_connection()
    try:
        if file_id is not None:
            resolved_file_id = int(file_id)
        else:
            resolved_file_id = _get_file_id(conn, file_id=None, file_path=path)
            if resolved_file_id is None and path and str(path).strip().startswith("local:"):
                path_norm = str(path).replace("\\", "/")
                if path_norm != path:
                    resolved_file_id = _get_file_id(conn, file_id=None, file_path=path_norm)
        if resolved_file_id is None:
            raise HTTPException(status_code=404, detail="File not found")
    finally:
        conn.close()

    # Вывод pipeline_run_id по file_id, если не передан (боевой режим — один прогон по папке)
    pipeline_run_id_i: int | None = int(pipeline_run_id) if pipeline_run_id is not None else None
    if pipeline_run_id_i is None and resolved_file_id is not None:
        ds_temp = DedupStore()
        try:
            cur = ds_temp.conn.cursor()
            cur.execute(
                "SELECT pipeline_run_id FROM file_persons WHERE file_id = ? ORDER BY pipeline_run_id DESC LIMIT 1",
                (resolved_file_id,),
            )
            r = cur.fetchone()
            if r and r["pipeline_run_id"] is not None:
                pipeline_run_id_i = int(r["pipeline_run_id"])
        finally:
            ds_temp.close()
        if pipeline_run_id_i is None:
            fs_temp = FaceStore()
            try:
                cur = fs_temp.conn.cursor()
                cur.execute(
                    """
                    SELECT run_id FROM photo_rectangles
                    WHERE file_id = ? AND run_id IS NOT NULL AND run_id != 0 AND COALESCE(ignore_flag, 0) = 0
                    ORDER BY run_id DESC LIMIT 1
                    """,
                    (resolved_file_id,),
                )
                r = cur.fetchone()
                if r and r["run_id"] is not None:
                    face_run_id_from_rect = int(r["run_id"])
                    ps_temp = PipelineStore()
                    try:
                        cur2 = ps_temp.conn.cursor()
                        cur2.execute(
                            "SELECT id FROM pipeline_runs WHERE face_run_id = ? ORDER BY id DESC LIMIT 1",
                            (face_run_id_from_rect,),
                        )
                        pr = cur2.fetchone()
                        if pr and pr["id"] is not None:
                            pipeline_run_id_i = int(pr["id"])
                    finally:
                        ps_temp.close()
            finally:
                fs_temp.close()

    # Без pipeline_run_id нельзя получить face_run_id и заполнить persons по прямоугольникам; direct_bindings — по file_persons по file_id
    if pipeline_run_id_i is not None:
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id_i))
        finally:
            ps.close()
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        face_run_id = pr.get("face_run_id")
        if not face_run_id:
            raise HTTPException(status_code=400, detail="face_run_id is not set yet")
        face_run_id_i = int(face_run_id)
    else:
        face_run_id_i = None

    fs = FaceStore()
    ds = DedupStore()
    try:
        fs_cur = fs.conn.cursor()
        ds_cur = ds.conn.cursor()
        persons_set = {}
        direct_bindings: list[dict[str, Any]] = []

        if face_run_id_i is not None:
            # 1. Ручные привязки (photo_rectangles.manual_person_id)
            fs_cur.execute("""
                SELECT
                    fr.manual_person_id AS person_id,
                    p.name AS person_name,
                    fr.id AS rectangle_id,
                    fr.bbox_x AS x, fr.bbox_y AS y, fr.bbox_w AS w, fr.bbox_h AS h
                FROM photo_rectangles fr
                LEFT JOIN persons p ON p.id = fr.manual_person_id
                WHERE fr.file_id = ? AND fr.run_id = ? AND fr.manual_person_id IS NOT NULL
                ORDER BY fr.id
            """, (resolved_file_id, face_run_id_i))
            for row in fs_cur.fetchall():
                pid = row["person_id"]
                if pid not in persons_set:
                    persons_set[pid] = {"id": pid, "name": row["person_name"], "faces": []}
                persons_set[pid]["faces"].append({
                    "id": row["rectangle_id"], "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"]
                })
            # 1b. Кластеры
            fs_cur.execute("""
                SELECT DISTINCT fc.person_id, p.name AS person_name,
                    fr_file.id AS rectangle_id, fr_file.bbox_x AS x, fr_file.bbox_y AS y, fr_file.bbox_w AS w, fr_file.bbox_h AS h
                FROM photo_rectangles fr_file
                JOIN face_clusters fc ON fc.id = fr_file.cluster_id
                LEFT JOIN persons p ON p.id = fc.person_id
                WHERE fr_file.file_id = ? AND fr_file.run_id = ?
                  AND COALESCE(fr_file.ignore_flag, 0) = 0 AND fc.person_id IS NOT NULL
                  AND (fc.run_id = ? OR fc.archive_scope = 'archive')
                  AND (fr_file.manual_person_id IS NULL OR fr_file.manual_person_id <> fc.person_id)
                ORDER BY fc.person_id, fr_file.id
            """, (resolved_file_id, face_run_id_i, face_run_id_i))
            for row in fs_cur.fetchall():
                pid = row["person_id"]
                if pid not in persons_set:
                    persons_set[pid] = {"id": pid, "name": row["person_name"], "faces": []}
                face_id = row["rectangle_id"]
                if not any(f["id"] == face_id for f in persons_set[pid]["faces"]):
                    persons_set[pid]["faces"].append({
                        "id": face_id, "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"]
                    })

        # 3. file_persons (прямая привязка): по run если есть, иначе по file_id без run — чтобы прямые привязки всегда были видны
        if pipeline_run_id_i is not None:
            ds_cur.execute("""
                SELECT DISTINCT fp.person_id, p.name AS person_name
                FROM file_persons fp
                LEFT JOIN persons p ON p.id = fp.person_id
                WHERE fp.file_id = ? AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
            """, (resolved_file_id, pipeline_run_id_i))
        else:
            ds_cur.execute("""
                SELECT DISTINCT fp.person_id, p.name AS person_name
                FROM file_persons fp
                LEFT JOIN persons p ON p.id = fp.person_id
                WHERE fp.file_id = ? AND fp.person_id IS NOT NULL
            """, (resolved_file_id,))
        for row in ds_cur.fetchall():
            direct_bindings.append({"person_id": row["person_id"], "person_name": row["person_name"] or ""})
            pid = row["person_id"]
            if pid not in persons_set:
                persons_set[pid] = {"id": pid, "name": row["person_name"], "faces": []}

        persons_list = list(persons_set.values())
    finally:
        fs.close()
        ds.close()

    return {
        "ok": True,
        "pipeline_run_id": pipeline_run_id_i,
        "file_id": resolved_file_id,
        "path": path,
        "persons": persons_list,
        "direct_bindings": direct_bindings,
    }


@router.post("/api/faces/manual-label")
def api_faces_manual_label(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    pipeline_run_id = payload.get("pipeline_run_id")
    path = payload.get("path")
    label = payload.get("label")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(path, str) or not (path.startswith("local:") or path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="path must start with local: or disk:")
    lab = (str(label) if label is not None else "").strip().lower()
    person = payload.get("person")
    if person is not None and not isinstance(person, str):
        raise HTTPException(status_code=400, detail="person must be string")
    if lab not in ("faces", "no_faces", "people_no_face", "quarantine", "cat", "not_animal_no_faces", ""):
        raise HTTPException(status_code=400, detail="label must be faces|no_faces|people_no_face|quarantine|cat|not_animal_no_faces|''")

    try:
        try:
            _agent_dbg(
                hypothesis_id="HNOFACES_CAT",
                location="web_api/routers/faces.py:api_faces_manual_label",
                message="faces_manual_label_request",
                data={"pipeline_run_id": int(pipeline_run_id), "label": lab, "path_sha1": hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:12]},
            )
        except Exception:
            pass

        ds = DedupStore()
        try:
            if lab == "":
                ds.delete_run_manual_labels(pipeline_run_id=int(pipeline_run_id), path=path)
            elif lab in ("faces", "no_faces"):
                ds.set_run_people_no_face_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_people_no_face=False, person=None)
                ds.set_run_quarantine_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_quarantine=False)
                ds.set_run_animals_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_animal=False, kind=None)
                ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=path, label=lab)
                ds.set_faces_auto_quarantine(path=path, is_quarantine=False, reason=None)
            elif lab == "not_animal_no_faces":
                ds.set_run_people_no_face_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_people_no_face=False, person=None)
                ds.set_run_quarantine_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_quarantine=False)
                ds.set_run_animals_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_animal=False, kind=None)
                ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=path, label="no_faces")
                ds.set_faces_auto_quarantine(path=path, is_quarantine=False, reason=None)
            elif lab == "people_no_face":
                ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=path, label=None)
                ds.set_run_quarantine_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_quarantine=False)
                ds.set_run_animals_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_animal=False, kind=None)
                ds.set_run_people_no_face_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_people_no_face=True, person=(person or None))
                ds.set_faces_auto_quarantine(path=path, is_quarantine=False, reason=None)
            elif lab == "quarantine":
                ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=path, label=None)
                ds.set_run_people_no_face_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_people_no_face=False, person=None)
                ds.set_run_animals_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_animal=False, kind=None)
                ds.set_run_quarantine_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_quarantine=True)
            elif lab == "cat":
                ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=path, label=None)
                ds.set_run_people_no_face_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_people_no_face=False, person=None)
                ds.set_run_quarantine_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_quarantine=False)
                ds.set_faces_auto_quarantine(path=path, is_quarantine=False, reason=None)
                ds.set_run_animals_manual(pipeline_run_id=int(pipeline_run_id), path=path, is_animal=True, kind="cat")
        finally:
            ds.close()

        if lab in ("no_faces", "people_no_face", "not_animal_no_faces"):
            ps = PipelineStore()
            try:
                pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
            finally:
                ps.close()
            face_run_id = int(pr.get("face_run_id") or 0) if pr else 0
            if face_run_id:
                from common.db import _get_file_id, get_connection
                conn = get_connection()
                try:
                    resolved_file_id = _get_file_id(conn, file_path=path)
                finally:
                    conn.close()
                if resolved_file_id:
                    fs = FaceStore()
                    try:
                        fs.replace_manual_rectangles(run_id=face_run_id, file_id=resolved_file_id, file_path=path, rects=[])
                    finally:
                        fs.close()

        return {"ok": True}
    except Exception as e:
        _agent_dbg(
            hypothesis_id="MANUAL_LABEL_ERR",
            location="web_api/routers/faces.py:api_faces_manual_label",
            message="api_faces_manual_label_exception",
            data={"error": str(e), "traceback": traceback.format_exc(), "lab": lab, "path_preview": (path or "")[:80]},
        )
        logger.exception("api_faces_manual_label failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/faces/manual-rectangles")
def api_faces_manual_rectangles(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Сохраняет ручные прямоугольники для файла.
    Приоритет: file_id (если передан), иначе path (если передан).
    """
    from backend.common.db import _get_file_id, get_connection
    
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    rects = payload.get("rects")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    if path is not None and not isinstance(path, str):
        raise HTTPException(status_code=400, detail="path must be str")
    if path is not None and not path.startswith("local:"):
        raise HTTPException(status_code=400, detail="path must start with local:")
    if rects is None:
        rects = []
    if not isinstance(rects, list):
        raise HTTPException(status_code=400, detail="rects must be list")

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise HTTPException(status_code=400, detail="face_run_id is not set yet")
    face_run_id_i = int(face_run_id)

    fs = FaceStore()
    try:
        fs.replace_manual_rectangles(run_id=face_run_id_i, file_id=file_id, file_path=path, rects=rects)
    finally:
        fs.close()

    ds = DedupStore()
    try:
        ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=str(path), label="faces")
    finally:
        ds.close()

    return {"ok": True}


@router.post("/api/faces/rotate-photo")
def api_faces_rotate_photo(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Поворачивает изображение на 90° влево или вправо и пересчитывает все прямоугольники (photo_rectangles).
    Только для локальных файлов (path должен начинаться с local:).
    Параметры: path (обязателен для local) или file_id; direction: "left" | "right".
    """
    from PIL import Image
    from PIL import ImageOps

    from backend.common.db import _get_file_id, get_connection

    path = payload.get("path")
    file_id = payload.get("file_id")
    direction = (payload.get("direction") or "").strip().lower()

    if direction not in ("left", "right"):
        raise HTTPException(status_code=400, detail="direction must be 'left' or 'right'")
    if not path and file_id is None:
        raise HTTPException(status_code=400, detail="path or file_id required")
    if path is not None and not isinstance(path, str):
        raise HTTPException(status_code=400, detail="path must be str")
    if path is not None and not path.startswith("local:"):
        raise HTTPException(status_code=400, detail="path must start with local: (only local files can be rotated)")

    conn = get_connection()
    try:
        resolved_file_id = _get_file_id(conn, file_id=int(file_id) if file_id is not None else None, file_path=path)
        if path is None and resolved_file_id:
            cur = conn.cursor()
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (resolved_file_id,))
            row = cur.fetchone()
            path = str(row["path"]) if row and row["path"] else None
        if not path or not path.startswith("local:"):
            raise HTTPException(status_code=404, detail="file not found or not local")
        abs_path = _strip_local_prefix(path)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="file not found on disk")

        with Image.open(abs_path) as img0:
            img = ImageOps.exif_transpose(img0)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            W, H = img.size
            # PIL: положительный угол = против часовой (влево), отрицательный = по часовой (вправо)
            angle = 90 if direction == "left" else -90
            img_rot = img.rotate(angle, expand=True)
            new_W, new_H = img_rot.size
            out_format = img0.format or "JPEG"

        save_kw = {"format": out_format}
        if out_format.upper() == "JPEG":
            save_kw["quality"] = 95
            save_kw["subsampling"] = 0
            # Пиксели уже в нужной ориентации — сбрасываем EXIF Orientation, иначе браузер повернёт ещё раз
            save_kw["exif"] = b""
        # Сохраняем во временный файл и заменяем оригинал — иначе на Windows файл может быть занят (превью)
        fd, temp_path = tempfile.mkstemp(suffix=Path(abs_path).suffix, dir=os.path.dirname(abs_path) or ".")
        try:
            os.close(fd)
            img_rot.save(temp_path, **save_kw)
            img_rot.close()
            os.replace(temp_path, abs_path)
        except Exception:
            if os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise

        cur = conn.cursor()
        cur.execute(
            "UPDATE files SET image_width = ?, image_height = ?, exif_orientation = 1 WHERE id = ?",
            (new_W, new_H, resolved_file_id),
        )

        cur.execute(
            "SELECT id, bbox_x, bbox_y, bbox_w, bbox_h FROM photo_rectangles WHERE file_id = ? AND (ignore_flag IS NULL OR ignore_flag = 0)",
            (resolved_file_id,),
        )
        rows = cur.fetchall()
        for r in rows:
            x, y, w, h = int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"])
            # Преобразование bbox при повороте: PIL rotate(-90)=CW (вправо), rotate(90)=CCW (влево).
            # Ранее прямоугольник после поворота «вправо» оказывался не на лице — пробуем поменять формулы местами.
            if direction == "right":
                x2, y2, w2, h2 = H - (y + h), x, h, w
            else:
                x2, y2, w2, h2 = y, W - (x + w), h, w
            x2 = max(0, min(x2, new_W - 1))
            y2 = max(0, min(y2, new_H - 1))
            w2 = max(1, min(w2, new_W - x2))
            h2 = max(1, min(h2, new_H - y2))
            cur.execute(
                "UPDATE photo_rectangles SET bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ? WHERE id = ?",
                (x2, y2, w2, h2, r["id"]),
            )
        # Инвалидируем сохранённые кропы — на странице персоны они пересчитаются на лету по новому файлу и bbox
        cur.execute("UPDATE photo_rectangles SET thumb_jpeg = NULL WHERE file_id = ?", (resolved_file_id,))
        conn.commit()
    finally:
        conn.close()

    return {"ok": True, "image_width": new_W, "image_height": new_H}


@router.post("/api/faces/rectangle/update")
def api_faces_rectangle_update(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Обновляет rectangle (координаты, тип, персона).
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Параметры:
    - pipeline_run_id: int (опционально, для сортируемых фото)
    - rectangle_id: int (обязательно)
    - file_id: int (опционально, для архивных фото)
    - path: str (опционально, для архивных фото)
    - bbox: dict (опционально) - {"x": int, "y": int, "w": int, "h": int}
    - person_id: int | None (опционально) - если None, удаляет привязку к персоне
    - assignment_type: str (опционально) - "cluster" | "manual_face" | None
    - is_face: int (опционально) - 1=лицо, 0=персона (для изменения типа прямоугольника)
    """
    from backend.common.db import _get_file_id, get_connection
    
    pipeline_run_id = payload.get("pipeline_run_id")
    rectangle_id = payload.get("rectangle_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    bbox = payload.get("bbox")
    person_id = payload.get("person_id")
    assignment_type = payload.get("assignment_type")
    is_face = payload.get("is_face")
    person_id_provided = "person_id" in payload
    is_face_provided = "is_face" in payload
    
    if not isinstance(rectangle_id, int):
        raise HTTPException(status_code=400, detail="rectangle_id is required and must be int")
    
    face_run_id_i = None
    
    # Если pipeline_run_id передан, получаем face_run_id (для сортируемых фото)
    if pipeline_run_id is not None:
        if not isinstance(pipeline_run_id, int):
            raise HTTPException(status_code=400, detail="pipeline_run_id must be int if provided")
        
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
            if not pr:
                raise HTTPException(status_code=404, detail="pipeline_run_id not found")
            face_run_id = pr.get("face_run_id")
            if not face_run_id:
                raise HTTPException(status_code=400, detail="face_run_id is not set yet")
            face_run_id_i = int(face_run_id)
        finally:
            ps.close()
    
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Проверяем, что rectangle существует
        if face_run_id_i is not None:
            # Для сортируемых фото: сначала по (id, run_id), затем по id (если run_id в БД NULL — чтобы обновление сработало)
            cur.execute("SELECT id, file_id, bbox_x, bbox_y, bbox_w, bbox_h FROM photo_rectangles WHERE id = ? AND (run_id = ? OR run_id IS NULL)", 
                       (int(rectangle_id), face_run_id_i))
        else:
            # Для архивных фото проверяем по file_id или path
            if file_id is None and path is None:
                raise HTTPException(status_code=400, detail="For archive photos, either file_id or path must be provided")
            
            resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
            if resolved_file_id is None:
                raise HTTPException(status_code=404, detail="File not found")
            
            cur.execute("SELECT id, file_id, bbox_x, bbox_y, bbox_w, bbox_h FROM photo_rectangles WHERE id = ? AND file_id = ?", 
                       (int(rectangle_id), resolved_file_id))
        
        rect_row = cur.fetchone()
        if not rect_row:
            raise HTTPException(status_code=404, detail="rectangle_id not found")
        
        file_id = rect_row["file_id"]
        
        # Обновляем координаты bbox, если переданы
        if bbox and isinstance(bbox, dict):
            bbox_x = int(bbox.get("x", 0))
            bbox_y = int(bbox.get("y", 0))
            bbox_w = int(bbox.get("w", 0))
            bbox_h = int(bbox.get("h", 0))
            
            if bbox_w > 0 and bbox_h > 0:
                cur.execute("""
                    UPDATE photo_rectangles 
                    SET bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ?
                    WHERE id = ?
                """, (bbox_x, bbox_y, bbox_w, bbox_h, int(rectangle_id)))
        
        # Обновляем is_face, если передан
        if is_face_provided:
            if not isinstance(is_face, int) or is_face not in (0, 1):
                raise HTTPException(status_code=400, detail="is_face must be 0 or 1")
            cur.execute("""
                UPDATE photo_rectangles 
                SET is_face = ?
                WHERE id = ?
            """, (int(is_face), int(rectangle_id)))
        
        # Обрабатываем привязку к персоне.
        # Важно: если ключ person_id передан и равен null, это означает "очистить привязку".
        # Раньше person_id=None игнорировался (из-за условия person_id is not None), что ломало UNDO.
        if person_id_provided:
            # Явная очистка (person_id: null / 0 / невалидное)
            if person_id is None or (isinstance(person_id, int) and person_id <= 0):
                cur.execute(
                    """
                    UPDATE photo_rectangles SET manual_person_id = NULL, cluster_id = NULL WHERE id = ?
                    """,
                    (int(rectangle_id),),
                )
            elif isinstance(person_id, int) and person_id > 0:
                # Проверяем, что персона существует
                cur.execute("SELECT id, name FROM persons WHERE id = ?", (int(person_id),))
                person_row = cur.fetchone()
                if not person_row:
                    raise HTTPException(status_code=404, detail="person_id not found")

                # Определяем тип привязки
                if assignment_type == "manual_face" or assignment_type is None:
                    # Создаём/обновляем ручную привязку в photo_rectangles.manual_person_id.
                    # Если передан face_run_id — выставляем run_id, чтобы файл учитывался в «К разбору» и ушёл из списка.
                    if face_run_id_i is not None:
                        cur.execute(
                            """
                            UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL, run_id = ? WHERE id = ?
                            """,
                            (int(person_id), face_run_id_i, int(rectangle_id)),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL WHERE id = ?
                            """,
                            (int(person_id), int(rectangle_id)),
                        )
                    set_file_processed(conn, rectangle_id=int(rectangle_id))
                elif assignment_type == "cluster":
                    # Добавляем в кластер (только для сортируемых фото с face_run_id)
                    if face_run_id_i is None:
                        # Для архивных фото кластеры не поддерживаются, используем manual_person_id
                        cur.execute(
                            """
                            UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL WHERE id = ?
                            """,
                            (int(person_id), int(rectangle_id)),
                        )
                        set_file_processed(conn, rectangle_id=int(rectangle_id))
                    else:
                        # Добавляем в кластер (нужно найти или создать кластер для персоны)
                        # Сначала снимаем ручную привязку (если была)
                        cur.execute(
                            """
                            UPDATE photo_rectangles SET manual_person_id = NULL WHERE id = ?
                            """,
                            (int(rectangle_id),),
                        )

                        # Ищем существующий кластер для персоны в этом run_id
                        cur.execute(
                            """
                            SELECT id FROM face_clusters
                            WHERE person_id = ? AND run_id = ?
                            LIMIT 1
                            """,
                            (int(person_id), face_run_id_i),
                        )
                        cluster_row = cur.fetchone()

                        if cluster_row:
                            cluster_id = cluster_row["id"]
                        else:
                            # Создаем новый кластер для персоны
                            from datetime import datetime, timezone

                            now = datetime.now(timezone.utc).isoformat()
                            cur.execute(
                                """
                                INSERT INTO face_clusters (run_id, person_id, created_at)
                                VALUES (?, ?, ?)
                                """,
                                (face_run_id_i, int(person_id), now),
                            )
                            cluster_id = cur.lastrowid

                        # Добавляем rectangle в кластер
                        cur.execute(
                            """
                            UPDATE photo_rectangles SET cluster_id = ? WHERE id = ?
                            """,
                            (cluster_id, int(rectangle_id)),
                        )
                        set_file_processed(conn, rectangle_id=int(rectangle_id))
            else:
                raise HTTPException(status_code=400, detail="person_id must be int or null")
        
        conn.commit()
        
        # Возвращаем обновлённый rectangle
        cur.execute("""
            SELECT 
                fr.id,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                COALESCE(fr.manual_person_id, fc.person_id) AS person_id,
                p.name AS person_name,
                p.is_me
            FROM photo_rectangles fr
            LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
            LEFT JOIN persons p ON p.id = COALESCE(fr.manual_person_id, fc.person_id)
            WHERE fr.id = ?
            LIMIT 1
        """, (int(rectangle_id),))
        updated_row = cur.fetchone()
        
        result = {
            "ok": True,
            "rectangle_id": int(rectangle_id),
            "bbox": {
                "x": updated_row["bbox_x"] if updated_row else None,
                "y": updated_row["bbox_y"] if updated_row else None,
                "w": updated_row["bbox_w"] if updated_row else None,
                "h": updated_row["bbox_h"] if updated_row else None,
            } if updated_row else None,
            "person_id": updated_row["person_id"] if updated_row else None,
            "person_name": updated_row["person_name"] if updated_row else None,
            "is_me": bool(updated_row["is_me"]) if updated_row and updated_row["is_me"] else False,
        }
        
    finally:
        fs.close()
    
    return result


@router.post("/api/faces/rectangle/create")
def api_faces_rectangle_create(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Создает новый rectangle (координаты, тип, персона).
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Параметры:
    - pipeline_run_id: int (опционально, для сортируемых фото)
    - file_id: int (опционально, для архивных фото)
    - path: str (опционально, для архивных фото)
    - bbox: dict (обязательно) - {"x": int, "y": int, "w": int, "h": int}
    - is_face: int (обязательно) - 1=лицо, 0=персона
    - person_id: int | None (опционально) - если передан, назначает персону
    - assignment_type: str (опционально) - "cluster" | "manual_face" | None
    """
    from backend.common.db import _get_file_id, get_connection
    from datetime import datetime, timezone
    
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    bbox = payload.get("bbox")
    is_face = payload.get("is_face", 1)  # По умолчанию 1 (лицо)
    person_id = payload.get("person_id")
    assignment_type = payload.get("assignment_type", "manual_face")
    
    # #region agent log
    import json
    log_data = {
        "location": "faces.py:2573",
        "message": "api_faces_rectangle_create payload received",
        "data": {
            "is_face": is_face,
            "is_face_type": str(type(is_face)),
            "person_id": person_id,
            "path": path,
            "file_id": file_id
        },
        "timestamp": int(__import__("time").time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D"
    }
    try:
        with open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except:
        pass
    # #endregion
    
    if not isinstance(bbox, dict):
        raise HTTPException(status_code=400, detail="bbox is required and must be dict")
    
    if not isinstance(is_face, int) or is_face not in (0, 1):
        raise HTTPException(status_code=400, detail="is_face must be 0 or 1")
    
    bbox_x = int(bbox.get("x", 0))
    bbox_y = int(bbox.get("y", 0))
    bbox_w = int(bbox.get("w", 0))
    bbox_h = int(bbox.get("h", 0))
    
    if bbox_w <= 0 or bbox_h <= 0:
        raise HTTPException(status_code=400, detail="bbox width and height must be positive")
    
    face_run_id_i = None
    
    # Если pipeline_run_id передан, получаем face_run_id (для сортируемых фото)
    if pipeline_run_id is not None:
        if not isinstance(pipeline_run_id, int):
            raise HTTPException(status_code=400, detail="pipeline_run_id must be int if provided")
        
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
            if not pr:
                raise HTTPException(status_code=404, detail="pipeline_run_id not found")
            face_run_id = pr.get("face_run_id")
            if not face_run_id:
                raise HTTPException(status_code=400, detail="face_run_id is not set yet")
            face_run_id_i = int(face_run_id)
        finally:
            ps.close()
    
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Получаем file_id
        if face_run_id_i is not None:
            # Для сортируемых фото file_id должен быть передан или получен из path
            if file_id is None and path is None:
                raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
            resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
            if resolved_file_id is None:
                raise HTTPException(status_code=404, detail="File not found")
        else:
            # Для архивных фото используем file_id или path
            if file_id is None and path is None:
                raise HTTPException(status_code=400, detail="For archive photos, either file_id or path must be provided")
            resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
            if resolved_file_id is None:
                raise HTTPException(status_code=404, detail="File not found")
        
        # Определяем face_index (максимальный + 1)
        if face_run_id_i is not None:
            cur.execute("""
                SELECT COALESCE(MAX(face_index), -1) + 1 as next_index
                FROM photo_rectangles
                WHERE run_id = ? AND file_id = ? AND is_face = 1
            """, (face_run_id_i, resolved_file_id))
        else:
            cur.execute("""
                SELECT COALESCE(MAX(face_index), -1) + 1 as next_index
                FROM photo_rectangles
                WHERE file_id = ? AND (run_id IS NULL OR run_id = 0) AND is_face = 1
            """, (resolved_file_id,))
        
        row = cur.fetchone()
        face_index = int(row["next_index"]) if row else 0
        
        # Создаем rectangle
        now = datetime.now(timezone.utc).isoformat()
        
        # #region agent log
        log_data = {
            "location": "faces.py:2663",
            "message": "inserting rectangle with is_face",
            "data": {
                "is_face": is_face,
                "is_face_type": str(type(is_face)),
                "resolved_file_id": resolved_file_id,
                "face_index": face_index,
                "person_id": person_id
            },
            "timestamp": int(__import__("time").time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "D"
        }
        try:
            with open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except:
            pass
        # #endregion
        
        if face_run_id_i is not None:
            cur.execute("""
                INSERT INTO photo_rectangles(
                    run_id, file_id, face_index,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    confidence, presence_score,
                    thumb_jpeg, manual_person, ignore_flag,
                    created_at,
                    is_manual, manual_created_at,
                    is_face
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0, ?, 1, ?, ?)
            """, (face_run_id_i, resolved_file_id, face_index, bbox_x, bbox_y, bbox_w, bbox_h, now, now, is_face))
        else:
            # Для архивных фото run_id = NULL
            cur.execute("""
                INSERT INTO photo_rectangles(
                    run_id, file_id, face_index,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    confidence, presence_score,
                    thumb_jpeg, manual_person, ignore_flag,
                    created_at,
                    is_manual, manual_created_at,
                    is_face
                )
                VALUES (NULL, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0, ?, 1, ?, ?)
            """, (resolved_file_id, face_index, bbox_x, bbox_y, bbox_w, bbox_h, now, now, is_face))
        
        rectangle_id = cur.lastrowid
        
        # Если персона назначена, создаем привязку
        if person_id is not None and isinstance(person_id, int) and person_id > 0:
            # Проверяем, что персона существует
            cur.execute("SELECT id, name FROM persons WHERE id = ?", (int(person_id),))
            person_row = cur.fetchone()
            if not person_row:
                raise HTTPException(status_code=404, detail="person_id not found")
            
            # Определяем тип привязки
            if assignment_type == "manual_face":
                # Создаём ручную привязку в photo_rectangles.manual_person_id
                cur.execute("""
                    UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL WHERE id = ?
                """, (int(person_id), rectangle_id))
            elif assignment_type == "cluster" and face_run_id_i is not None:
                # Добавляем в кластер (только для сортируемых фото)
                # Ищем существующий кластер для персоны в этом run_id
                cur.execute("""
                    SELECT id FROM face_clusters 
                    WHERE person_id = ? AND run_id = ? 
                    LIMIT 1
                """, (int(person_id), face_run_id_i))
                cluster_row = cur.fetchone()
                
                if cluster_row:
                    cluster_id = cluster_row["id"]
                else:
                    # Создаем новый кластер для персоны
                    cur.execute("""
                        INSERT INTO face_clusters (run_id, person_id, created_at)
                        VALUES (?, ?, ?)
                    """, (face_run_id_i, int(person_id), now))
                    cluster_id = cur.lastrowid
                
                # Добавляем rectangle в кластер
                cur.execute("""
                    UPDATE photo_rectangles SET cluster_id = ? WHERE id = ?
                """, (cluster_id, rectangle_id))
        
        conn.commit()
        
        # Возвращаем созданный rectangle
        cur.execute("""
            SELECT 
                fr.id,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                COALESCE(fr.manual_person_id, fc.person_id) AS person_id,
                p.name AS person_name,
                p.is_me
            FROM photo_rectangles fr
            LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
            LEFT JOIN persons p ON p.id = COALESCE(fr.manual_person_id, fc.person_id)
            WHERE fr.id = ?
            LIMIT 1
        """, (rectangle_id,))
        created_row = cur.fetchone()
        
        result = {
            "ok": True,
            "rectangle_id": int(rectangle_id),
            "bbox": {
                "x": created_row["bbox_x"] if created_row else None,
                "y": created_row["bbox_y"] if created_row else None,
                "w": created_row["bbox_w"] if created_row else None,
                "h": created_row["bbox_h"] if created_row else None,
            } if created_row else None,
            "person_id": created_row["person_id"] if created_row else None,
            "person_name": created_row["person_name"] if created_row else None,
            "is_me": bool(created_row["is_me"]) if created_row and created_row["is_me"] else False,
        }
        
    finally:
        fs.close()
    
    return result


@router.post("/api/faces/rectangle/delete")
def api_faces_rectangle_delete(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Удаляет rectangle (помечает как ignore_flag = 1).
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Параметры:
    - pipeline_run_id: int (опционально, для сортируемых фото)
    - rectangle_id: int (обязательно)
    - file_id: int (опционально, для архивных фото)
    - path: str (опционально, для архивных фото)
    """
    from backend.common.db import _get_file_id
    
    pipeline_run_id = payload.get("pipeline_run_id")
    rectangle_id = payload.get("rectangle_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    
    if not isinstance(rectangle_id, int):
        raise HTTPException(status_code=400, detail="rectangle_id is required and must be int")
    
    face_run_id_i = None
    
    # Если pipeline_run_id передан, получаем face_run_id (для сортируемых фото)
    if pipeline_run_id is not None:
        if not isinstance(pipeline_run_id, int):
            raise HTTPException(status_code=400, detail="pipeline_run_id must be int if provided")
        
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
            if not pr:
                raise HTTPException(status_code=404, detail="pipeline_run_id not found")
            face_run_id = pr.get("face_run_id")
            if not face_run_id:
                raise HTTPException(status_code=400, detail="face_run_id is not set yet")
            face_run_id_i = int(face_run_id)
        finally:
            ps.close()
    
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Проверяем, что rectangle существует
        if face_run_id_i is not None:
            # Для сортируемых фото проверяем по run_id
            cur.execute("SELECT id, file_id FROM photo_rectangles WHERE id = ? AND run_id = ?", 
                       (int(rectangle_id), face_run_id_i))
        else:
            # Для архивных фото проверяем по file_id или path
            if file_id is None and path is None:
                raise HTTPException(status_code=400, detail="For archive photos, either file_id or path must be provided")
            
            resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
            if resolved_file_id is None:
                raise HTTPException(status_code=404, detail="File not found")
            
            cur.execute("SELECT id, file_id FROM photo_rectangles WHERE id = ? AND file_id = ?", 
                       (int(rectangle_id), resolved_file_id))
        
        rect_row = cur.fetchone()
        if not rect_row:
            raise HTTPException(status_code=404, detail="rectangle_id not found")
        
        # Помечаем как игнорируемый и снимаем привязки
        cur.execute("""
            UPDATE photo_rectangles 
            SET ignore_flag = 1, manual_person_id = NULL, cluster_id = NULL
            WHERE id = ?
        """, (int(rectangle_id),))
        
        conn.commit()
    finally:
        fs.close()
    
    return {"ok": True, "rectangle_id": int(rectangle_id)}


@router.post("/api/faces/rectangles/assign-outsider")
def api_faces_rectangles_assign_outsider(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Назначает все неназначенные rectangles персоне "Посторонние".
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Параметры:
    - pipeline_run_id: int (опционально, для файлов из прогона сортировки)
    - file_id: int (опционально)
    - path: str (опционально)
    """
    from backend.common.db import _get_file_id, get_connection
    
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    
    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    
    # Определяем, архивный ли это файл
    is_archive = path and path.startswith("disk:")
    
    # Для файлов из прогона сортировки проверяем pipeline_run_id
    face_run_id_i = None
    if pipeline_run_id is not None:
        if not isinstance(pipeline_run_id, int):
            raise HTTPException(status_code=400, detail="pipeline_run_id must be int if provided")
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
            if not pr:
                raise HTTPException(status_code=404, detail="pipeline_run_id not found")
            face_run_id = pr.get("face_run_id")
            if not face_run_id:
                raise HTTPException(status_code=400, detail="face_run_id is not set yet")
            face_run_id_i = int(face_run_id)
        finally:
            ps.close()
    
    # Получаем file_id
    conn = get_connection()
    try:
        resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
        if resolved_file_id is None:
            raise HTTPException(status_code=404, detail="File not found")
    finally:
        conn.close()
    
    # Находим или создаем персону "Посторонний" (или "Посторонние")
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Используем функцию для получения ID персоны "Посторонний"
        from backend.common.db import get_outsider_person_id
        outsider_person_id = get_outsider_person_id(conn)
        if not outsider_person_id:
            raise HTTPException(status_code=500, detail="Персона 'Посторонний' не найдена. Требуется миграция.")
        
        # Находим все неназначенные rectangles для файла
        # Для архивных файлов используем archive_scope, для файлов из прогона - run_id
        if is_archive or face_run_id_i is None:
            cur.execute("""
                SELECT fr.id
                FROM photo_rectangles fr
                WHERE fr.file_id = ? 
                  AND fr.archive_scope = 'archive'
                  AND COALESCE(fr.ignore_flag, 0) = 0
                  AND fr.manual_person_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM face_clusters fc WHERE fc.id = fr.cluster_id AND fc.person_id IS NOT NULL
                  )
            """, (resolved_file_id,))
        else:
            cur.execute("""
                SELECT fr.id
                FROM photo_rectangles fr
                WHERE fr.file_id = ? 
                  AND fr.run_id = ?
                  AND COALESCE(fr.ignore_flag, 0) = 0
                  AND fr.manual_person_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM face_clusters fc WHERE fc.id = fr.cluster_id AND fc.person_id IS NOT NULL
                  )
            """, (resolved_file_id, face_run_id_i))
        
        unassigned_rects = cur.fetchall()
        assigned_count = 0
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        
        for rect_row in unassigned_rects:
            rect_id = rect_row["id"]
            # Удаляем из кластеров (если был)
            cur.execute("UPDATE photo_rectangles SET cluster_id = NULL WHERE id = ?", (rect_id,))
            # Создаем ручную привязку
            cur.execute("""
                UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL WHERE id = ?
            """, (outsider_person_id, rect_id))
            assigned_count += 1
        
        conn.commit()
    finally:
        fs.close()
    
    return {
        "ok": True,
        "assigned_count": assigned_count,
        "person_id": outsider_person_id,
        "person_name": "Посторонний"
    }


@router.post("/api/faces/file/mark-as-cat")
def api_faces_file_mark_as_cat(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Помечает файл как "кот" (удаляет все rectangles, устанавливает метку animals_manual).
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Параметры:
    - pipeline_run_id: int (обязательно)
    - file_id: int (опционально)
    - path: str (опционально)
    """
    from backend.common.db import _get_file_id, get_connection
    
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required and must be int")
    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    
    # Проверяем, что pipeline_run_id существует
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        face_run_id = pr.get("face_run_id")
        if not face_run_id:
            raise HTTPException(status_code=400, detail="face_run_id is not set yet")
        face_run_id_i = int(face_run_id)
    finally:
        ps.close()
    
    # Получаем file_id и path
    conn = get_connection()
    try:
        resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
        if resolved_file_id is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Получаем path, если не передан
        if path is None:
            cur = conn.cursor()
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (resolved_file_id,))
            row = cur.fetchone()
            if row:
                path = row["path"]
    finally:
        conn.close()
    
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    
    # Удаляем все rectangles для файла
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Помечаем все rectangles как игнорируемые и снимаем привязки
        cur.execute("""
            UPDATE photo_rectangles 
            SET ignore_flag = 1, manual_person_id = NULL, cluster_id = NULL
            WHERE file_id = ? AND run_id = ?
        """, (resolved_file_id, face_run_id_i))
        
        conn.commit()
    finally:
        fs.close()
    
    # Устанавливаем метку animals_manual
    ds = DedupStore()
    try:
        ds.set_run_animals_manual(pipeline_run_id=int(pipeline_run_id), path=str(path), is_animal=True, kind="cat")
    finally:
        ds.close()
    
    return {"ok": True, "file_id": resolved_file_id, "path": path}


@router.post("/api/faces/file/mark-as-no-people")
def api_faces_file_mark_as_no_people(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Помечает файл как "нет людей" (удаляет все rectangles, устанавливает метку no_faces).
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Параметры:
    - pipeline_run_id: int (обязательно)
    - file_id: int (опционально)
    - path: str (опционально)
    """
    from backend.common.db import _get_file_id, get_connection
    
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required and must be int")
    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    
    # Проверяем, что pipeline_run_id существует
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        face_run_id = pr.get("face_run_id")
        if not face_run_id:
            raise HTTPException(status_code=400, detail="face_run_id is not set yet")
        face_run_id_i = int(face_run_id)
    finally:
        ps.close()
    
    # Получаем file_id и path
    conn = get_connection()
    try:
        resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
        if resolved_file_id is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Получаем path, если не передан
        if path is None:
            cur = conn.cursor()
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (resolved_file_id,))
            row = cur.fetchone()
            if row:
                path = row["path"]
    finally:
        conn.close()
    
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    
    # Удаляем все rectangles для файла
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Помечаем все rectangles как игнорируемые и снимаем привязки
        cur.execute("""
            UPDATE photo_rectangles 
            SET ignore_flag = 1, manual_person_id = NULL, cluster_id = NULL
            WHERE file_id = ? AND run_id = ?
        """, (resolved_file_id, face_run_id_i))
        
        conn.commit()
    finally:
        fs.close()
    
    # Устанавливаем метку no_faces
    ds = DedupStore()
    try:
        ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=str(path), label="no_faces")
    finally:
        ds.close()
    
    return {"ok": True, "file_id": resolved_file_id, "path": path}


@router.get("/api/faces/rectangles/duplicates-check")
def api_faces_rectangles_duplicates_check(
    pipeline_run_id: int,
    file_id: int | None = None,
    path: str | None = None
) -> dict[str, Any]:
    """
    Проверяет дубликаты персоны на фото (один человек не должен быть дважды на одном фото).
    Приоритет: file_id (если передан), иначе path (если передан).
    
    Возвращает список rectangles с информацией о дубликатах.
    """
    from backend.common.db import _get_file_id, get_connection
    
    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    
    # Проверяем, что pipeline_run_id существует
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        face_run_id = pr.get("face_run_id")
        if not face_run_id:
            raise HTTPException(status_code=400, detail="face_run_id is not set yet")
        face_run_id_i = int(face_run_id)
    finally:
        ps.close()
    
    # Получаем file_id
    conn = get_connection()
    try:
        resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
        if resolved_file_id is None:
            raise HTTPException(status_code=404, detail="File not found")
    finally:
        conn.close()
    
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Получаем все rectangles с привязками к персонам (manual_person_id или cluster → person)
        cur.execute("""
            SELECT 
                fr.id AS rectangle_id,
                COALESCE(fr.manual_person_id, fc.person_id) AS person_id,
                p.name AS person_name
            FROM photo_rectangles fr
            LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
            LEFT JOIN persons p ON p.id = COALESCE(fr.manual_person_id, fc.person_id)
            WHERE fr.file_id = ? 
              AND fr.run_id = ?
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND (fr.manual_person_id IS NOT NULL OR fc.person_id IS NOT NULL)
        """, (resolved_file_id, face_run_id_i))
        
        rectangles = cur.fetchall()
        
        # Группируем по person_id для поиска дубликатов
        person_rects: dict[int, list[int]] = {}
        for rect in rectangles:
            person_id = rect["person_id"]
            if person_id:
                if person_id not in person_rects:
                    person_rects[person_id] = []
                person_rects[person_id].append(rect["rectangle_id"])
        
        # Находим дубликаты (персоны с более чем одним rectangle)
        # ИСКЛЮЧЕНИЕ: "Посторонний" - это специальная персона, для нее дубликаты разрешены
        # Получаем ID персоны "Посторонний" для исключения из проверки
        from backend.common.db import get_outsider_person_id
        outsider_person_id = get_outsider_person_id(conn)
        
        duplicates: dict[int, list[int]] = {}
        for person_id, rect_ids in person_rects.items():
            if len(rect_ids) > 1:
                # Исключаем "Посторонний" из проверки дубликатов по ID
                if person_id != outsider_person_id:
                    duplicates[person_id] = rect_ids
        
        # Формируем результат
        result_rectangles = []
        for rect in rectangles:
            person_id = rect["person_id"]
            is_duplicate = person_id is not None and person_id in duplicates
            
            result_rectangles.append({
                "rectangle_id": rect["rectangle_id"],
                "person_id": person_id,
                "person_name": rect["person_name"],
                "is_duplicate": is_duplicate,
                "duplicate_person_ids": [person_id] if is_duplicate else []
            })
        
    finally:
        fs.close()
    
    return {
        "ok": True,
        "file_id": resolved_file_id,
        "rectangles": result_rectangles,
        "has_duplicates": len(duplicates) > 0
    }


def _normalize_yadisk_path(path: str) -> str:
    """Нормализует путь YaDisk: убирает лишние пробелы, приводит к единому формату."""
    p = str(path or "").strip()
    if not p:
        return ""
    # Убираем двойные слеши, но сохраняем disk:/ в начале
    if p.startswith("disk:/"):
        p = "disk:/" + p[6:].lstrip("/")
    return p.replace("\\", "/")


def _ensure_yadisk_parent_dirs(disk: Any, remote_file_path: str) -> None:
    """
    Создаёт на Яндекс.Диске цепочку родительских папок для remote_file_path (полный путь к файлу).
    Игнорирует PathExistsError (папка уже есть).
    """
    p = (remote_file_path or "").replace("\\", "/").strip()
    if not p or p.count("/") < 2:
        return
    parent = p.rsplit("/", 1)[0]
    if not parent or parent in ("disk:", "disk:/"):
        return
    parts = parent.split("/")
    for i in range(2, len(parts) + 1):
        dir_path = "/".join(parts[:i])
        try:
            disk.mkdir(dir_path)
        except Exception as e:
            if yadisk_exceptions and isinstance(e, yadisk_exceptions.PathExistsError):
                pass
            elif "PathExistsError" in type(e).__name__ or "already exists" in str(e).lower():
                pass
            else:
                raise


def _parent_segment_starts_with_underscore(path: str) -> bool:
    """True, если родительская папка пути начинается с '_' (файл уже в отсортированной папке)."""
    p = (path or "").replace("\\", "/").strip("/")
    if not p:
        return True
    parts = p.split("/")
    if len(parts) < 2:
        return False
    parent_segment = parts[-2]
    return parent_segment.startswith("_")


@router.get("/api/faces/step4-report")
def api_faces_step4_report(pipeline_run_id: int) -> dict[str, Any]:
    """
    Отчёт шага 4 «разложить по правилам»:
    - by_folder: файлы прогона с заполненным target_folder в БД (только source — без local_done, чтобы «Всего в отчёте» совпадало с «Всего в прогоне» tab-counts).
    - unsorted: файлы прогона без target_folder в БД (только source).
    Возвращает: pipeline_run_id, root_path, by_folder, total, unsorted, unsorted_count.
    """
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    root_path = str(pr.get("root_path") or "")
    dedup_run_id = pr.get("dedup_run_id")
    face_run_id = pr.get("face_run_id")
    if not root_path or dedup_run_id is None:
        return {
            "pipeline_run_id": pipeline_run_id,
            "root_path": root_path or None,
            "by_folder": {},
            "total": 0,
            "unsorted": [],
            "unsorted_count": 0,
        }

    conn = get_connection()
    try:
        cur = conn.cursor()
        # Только source — без local_done, чтобы «Всего в отчёте» совпадало с «Всего в прогоне» (tab-counts считает только source).
        scope_run = (int(dedup_run_id),)
        cur.execute(
            """
            SELECT id, path, name, target_folder
            FROM files
            WHERE COALESCE(inventory_scope, '') = 'source' AND last_run_id = ?
              AND (status IS NULL OR status != 'deleted') AND target_folder IS NOT NULL AND trim(target_folder) != ''
            ORDER BY target_folder, path
            """,
            scope_run,
        )
        rows = cur.fetchall()

        unsorted: list[dict[str, Any]] = []
        cur.execute(
            """
            SELECT id, path, name
            FROM files
            WHERE COALESCE(inventory_scope, '') = 'source' AND last_run_id = ?
              AND (status IS NULL OR status != 'deleted')
              AND (target_folder IS NULL OR trim(target_folder) = '')
            ORDER BY path
            """,
            scope_run,
        )
        for r in cur.fetchall():
            unsorted.append({"id": r["id"], "path": r["path"], "name": r["name"]})
    finally:
        try:
            conn.close()
        except Exception:
            pass

    by_folder: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        folder = str(r["target_folder"] or "").strip()
        if not folder:
            continue
        entry = {"id": r["id"], "path": r["path"], "name": r["name"]}
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(entry)
    total = sum(len(v) for v in by_folder.values())
    unsorted_count_val = len(unsorted)
    # #region agent log
    try:
        _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
        _dl.write(json.dumps({"hypothesisId": "H2,H3", "location": "faces.py:step4-report", "message": "step4-report scope", "data": {"pipeline_run_id": pipeline_run_id, "dedup_run_id": dedup_run_id, "total_with_folder": total, "unsorted_count": unsorted_count_val, "scope_total": total + unsorted_count_val}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
        _dl.close()
    except Exception:
        pass
    # #endregion
    return {
        "pipeline_run_id": pipeline_run_id,
        "root_path": root_path,
        "by_folder": by_folder,
        "total": total,
        "unsorted": unsorted,
        "unsorted_count": unsorted_count_val,
    }


def clear_target_folders_for_run_impl(pipeline_run_id: int) -> dict[str, Any]:
    """
    Обнуляет target_folder у всех файлов прогона (inventory_scope='source' + last_run_id).
    Вызывается из API и из local_pipeline напрямую (без HTTP), чтобы избежать дедлока.
    Возвращает: {"cleared_count": N} или raises ValueError.
    """
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise ValueError("pipeline_run_id not found")
    dedup_run_id = pr.get("dedup_run_id")
    if dedup_run_id is None:
        return {"cleared_count": 0}
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT path FROM files
            WHERE COALESCE(inventory_scope, '') = 'source' AND last_run_id = ?
              AND (status IS NULL OR status != 'deleted')
            """,
            (int(dedup_run_id),),
        )
        paths = [str(r[0] or "") for r in cur.fetchall() if r[0]]
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not paths:
        return {"cleared_count": 0}
    ds = DedupStore()
    try:
        n = ds.clear_target_folder(paths=paths)
        return {"cleared_count": n}
    finally:
        ds.close()


@router.post("/api/faces/clear-target-folders-for-run")
def api_faces_clear_target_folders_for_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """HTTP-обёртка: обнуляет target_folder (см. clear_target_folders_for_run_impl)."""
    pipeline_run_id = payload.get("pipeline_run_id")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    try:
        return clear_target_folders_for_run_impl(pipeline_run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def fill_target_folders_impl(pipeline_run_id: int) -> dict[str, Any]:
    """
    Заполняет target_folder в БД по тому же правилу, что new_target_folder в скрипте.
    Вызывается из API и из local_pipeline напрямую (без HTTP), чтобы избежать дедлока при одном воркере.
    Возвращает: {"ok": True, "filled_count": N, "errors": [...]} или raises ValueError.
    """
    # #region agent log
    try:
        _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
        _dl.write(json.dumps({"hypothesisId": "fill-entry", "location": "faces.py:fill_target_folders_impl", "message": "fill_target_folders_impl entered", "data": {"pipeline_run_id": pipeline_run_id}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
        _dl.close()
    except Exception:
        pass
    # #endregion
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise ValueError("pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise ValueError("face_run_id is not set yet (step 3 not started)")
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")
    dedup_run_id = pr.get("dedup_run_id")
    if not root_path:
        raise ValueError("root_path is not set")

    if root_path.startswith("disk:"):
        root_like = root_path.rstrip("/") + "/%"
    else:
        try:
            rp_clean = root_path[6:] if root_path.startswith("local:") else root_path
            rp_abs = os.path.abspath(rp_clean).rstrip("\\/") + ("\\" if os.name == "nt" else "/")
            root_like = "local:" + rp_abs + "%"
        except Exception:
            root_like = root_path.rstrip("/") + "/%"

    ds = DedupStore()
    ps = PipelineStore()
    conn = get_connection()
    try:
        ignored_person_id_fill = get_outsider_person_id(conn) if conn else None
        if ignored_person_id_fill is None:
            ignored_person_id_fill = -1
        cur_preclean = ps.conn.cursor()
        cur_preclean.execute(
            "SELECT src_path, kind FROM preclean_moves WHERE pipeline_run_id = ?",
            (int(pipeline_run_id),),
        )
        preclean_map = {str(r[0] or ""): str(r[1] or "") for r in cur_preclean.fetchall() if r[0] and r[1]}

        # has_person_binding: file_persons ИЛИ привязки по лицам (manual/cluster) — как в tab-counts, чтобы файлы с людьми без лиц тоже шли в папки по правилам
        has_person_binding_sql = """
          (EXISTS (
            SELECT 1 FROM file_persons fp
            WHERE fp.file_id = f.id AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL AND fp.person_id != ?
          ) OR EXISTS (
            SELECT 1 FROM photo_rectangles fr
            LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
            WHERE fr.file_id = f.id
              AND (fr.run_id = ? OR COALESCE(TRIM(COALESCE(fr.archive_scope, '')), '') = 'archive')
              AND (fr.manual_person_id IS NOT NULL OR fc.person_id IS NOT NULL)
              AND COALESCE(fr.manual_person_id, fc.person_id) != ?
          ))
        """
        cur = ds.conn.cursor()
        # Тот же набор, что step4-report: только inventory_scope='source' + last_run_id (без OR по path, чтобы не зависеть от формата путей)
        # group_path для no_faces: как на вкладке «Нет людей» (file_groups)
        group_path_subquery = """
        LEFT JOIN (
            SELECT file_id, MIN(group_path) AS group_path
            FROM file_groups
            WHERE pipeline_run_id = ?
            GROUP BY file_id
        ) fg ON fg.file_id = f.id
        """
        group_path_select = ", COALESCE(fg.group_path, '') AS group_path"
        if dedup_run_id is not None:
            cur.execute(
                f"""
                SELECT
                  f.id, f.path, f.name, f.parent_path,
                  COALESCE(m.faces_manual_label, '') AS faces_manual_label,
                  COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
                  COALESCE(f.faces_auto_quarantine, 0) AS faces_auto_quarantine,
                  COALESCE(f.faces_count, 0) AS faces_count,
                  COALESCE(m.animals_manual, 0) AS animals_manual,
                  COALESCE(f.animals_auto, 0) AS animals_auto,
                  COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual,
                  {has_person_binding_sql} AS has_person_binding
                  {group_path_select}
                FROM files f
                LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.file_id = f.id
                {group_path_subquery}
                WHERE (f.status IS NULL OR f.status != 'deleted')
                  AND COALESCE(f.inventory_scope, '') = 'source' AND f.last_run_id = ?
                """,
                (int(pipeline_run_id), ignored_person_id_fill, face_run_id_i, ignored_person_id_fill, int(pipeline_run_id), int(pipeline_run_id), int(dedup_run_id)),
            )
        else:
            cur.execute(
                f"""
                SELECT
                  f.id, f.path, f.name, f.parent_path,
                  COALESCE(m.faces_manual_label, '') AS faces_manual_label,
                  COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
                  COALESCE(f.faces_auto_quarantine, 0) AS faces_auto_quarantine,
                  COALESCE(f.faces_count, 0) AS faces_count,
                  COALESCE(m.animals_manual, 0) AS animals_manual,
                  COALESCE(f.animals_auto, 0) AS animals_auto,
                  COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual,
                  {has_person_binding_sql} AS has_person_binding
                  {group_path_select}
                FROM files f
                LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.file_id = f.id
                {group_path_subquery}
                WHERE (f.status IS NULL OR f.status != 'deleted') AND f.path LIKE ? AND f.faces_run_id = ?
                """,
                (int(pipeline_run_id), ignored_person_id_fill, face_run_id_i, ignored_person_id_fill, int(pipeline_run_id), int(pipeline_run_id), root_like, face_run_id_i),
            )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        ds.close()
        ps.close()

    # #region agent log
    try:
        _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
        _dl.write(json.dumps({"hypothesisId": "H3,H4", "location": "faces.py:fill", "message": "fill query result", "data": {"len_rows": len(rows), "dedup_run_id": dedup_run_id, "pipeline_run_id": pipeline_run_id, "first_id": rows[0].get("id") if rows else None}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
        _dl.close()
    except Exception:
        pass
    # #endregion

    target_folders = list_folders(role="target")
    filled_count = 0
    errors: list[dict[str, str]] = []
    step4_total_rows = len(rows)
    _step4_progress_interval = 10  # обновлять прогресс в БД каждые N файлов (главная страница)
    ds = DedupStore()
    try:
        for r in rows:
            path = str(r.get("path") or "")
            if not path:
                continue
            file_id = r.get("id")
            if file_id is None:
                continue

            # То же правило, что в отладочном скрипте (new_target_folder)
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
            # Файлы с привязкой к персоне (file_persons или лица): раскладываем по правилам, не в _people_no_face
            if r.get("has_person_binding") and effective_tab in ("no_faces", "people_no_face"):
                effective_tab = "faces"

            person_name = None
            if effective_tab == "faces":
                try:
                    person_name = _resolve_target_folder_for_faces(
                        conn,
                        file_id=int(file_id),
                        pipeline_run_id=int(pipeline_run_id),
                        face_run_id=face_run_id_i,
                        target_folders=target_folders,
                    )
                except Exception as e:
                    errors.append({
                        "path": path,
                        "file_id": file_id,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    })
                    # Не подставляем «Другие люди» — файл остаётся без target_folder, причину смотрим в errors
                    continue

            # Вычисляем target_folder по тому же правилу, что new_target_folder в скрипте
            # Для no_faces передаём group_path: поездки → Путешествия/..., остальное → под корнем; без группы — несортировано
            group_path_val = (r.get("group_path") or "").strip() or None
            target_folder = _determine_target_folder(
                path=path,
                effective_tab=effective_tab or "no_faces",
                root_path=root_path,
                preclean_kind=preclean_kind,
                person_name=person_name,
                target_folders=target_folders,
                group_path=group_path_val,
            )
            if not target_folder:
                # #region agent log
                try:
                    _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
                    _dl.write(json.dumps({"hypothesisId": "H4", "location": "faces.py:fill skip", "message": "skip no target_folder", "data": {"file_id": file_id, "path": path[:80]}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
                    _dl.close()
                except Exception:
                    pass
                # #endregion
                continue
            try:
                ds.set_target_folder(file_id=int(file_id), target_folder=target_folder)
                filled_count += 1
                # Прогресс шага 4 на главной: обновляем run в БД периодически
                if filled_count == 1 or filled_count % _step4_progress_interval == 0 or filled_count == step4_total_rows:
                    # #region agent log
                    try:
                        _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
                        _dl.write(json.dumps({"hypothesisId": "H1,H4", "location": "faces.py:fill progress", "message": "writing step4 progress to DB", "data": {"filled_count": filled_count, "step4_total_rows": step4_total_rows}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
                        _dl.close()
                    except Exception:
                        pass
                    # #endregion
                    ps_prog = PipelineStore()
                    try:
                        ps_prog.update_run(
                            run_id=int(pipeline_run_id),
                            step4_processed=filled_count,
                            step4_total=step4_total_rows or None,
                        )
                    finally:
                        ps_prog.close()
            except sqlite3.OperationalError as e:
                # Не глушить «database is locked» — пробросить, чтобы пользователь видел ошибку
                if "locked" in str(e).lower():
                    raise
                errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
            except Exception as e:
                errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
    finally:
        ds.close()
    try:
        conn.close()
    except Exception:
        pass

    return {"ok": True, "filled_count": filled_count, "errors": errors}


@router.post("/api/faces/fill-target-folders")
def api_faces_fill_target_folders(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    HTTP-обёртка: заполняет target_folder в БД (см. fill_target_folders_impl).
    Параметры: pipeline_run_id (int). Возвращает: filled_count, errors.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    try:
        return fill_target_folders_impl(pipeline_run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _remove_empty_dirs_up_to(dir_path: str, stop_at: str) -> None:
    """
    Удаляет каталог dir_path и его пустых родителей вверх по дереву, пока не дойдём до stop_at.
    Не удаляет stop_at и ничего выше. Игнорирует ошибки (права, непустой каталог и т.д.).
    На Windows сравнение путей без учёта регистра (иначе path.startswith(stop) может дать False).
    """
    if not dir_path or not stop_at:
        return
    path = os.path.normpath(os.path.abspath(dir_path))
    stop = os.path.normpath(os.path.abspath(stop_at))
    # На Windows пути case-insensitive — иначе пустые папки могут не удаляться
    if os.name == "nt":
        path_lower = path.lower()
        stop_lower = stop.lower()
        if not path_lower.startswith(stop_lower):
            return
        while path and path.lower() != stop_lower and os.path.isdir(path):
            try:
                if os.listdir(path):
                    break
                os.rmdir(path)
                path = os.path.dirname(path)
            except OSError:
                break
    else:
        if not path.startswith(stop):
            return
        while path and path != stop and os.path.isdir(path):
            try:
                if os.listdir(path):
                    break
                os.rmdir(path)
                path = os.path.dirname(path)
            except OSError:
                break


def sort_into_folders_impl(
    pipeline_run_id: int,
    dry_run: bool = False,
    destination: str | None = None,
    limit_file_paths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Перемещает файлы с заполненным target_folder.
    limit_file_paths: при destination='archive' — переносить только файлы с path из списка (для теста по подмножеству).
    Возвращает: {"ok": True, "moved_count": N, "errors": [...]} или raises ValueError.
    """
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise ValueError("pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise ValueError("face_run_id is not set yet (step 3 not started)")
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")
    dedup_run_id = pr.get("dedup_run_id")
    if dedup_run_id is None:
        raise ValueError("dedup_run_id is not set (step 1 not started)")

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        cur.execute(
            """
            SELECT id, path, name, target_folder
            FROM files
            WHERE COALESCE(inventory_scope, '') = 'source' AND last_run_id = ?
              AND (status IS NULL OR status != 'deleted')
              AND target_folder IS NOT NULL AND trim(target_folder) != ''
            """,
            (int(dedup_run_id),),
        )
        files_to_move = [dict(r) for r in cur.fetchall()]
    finally:
        ds.close()

    if destination == "local":
        # Только файлы на локальном диске — переносим в целевые папки под корнем (target_folder всегда local:root/...)
        files_to_move = [r for r in files_to_move if (str(r.get("path") or "")).strip().startswith("local:")]
    elif destination == "archive":
        # В архив — только файлы на локальном диске (target_folder в БД всегда local:root/..., disk-путь для загрузки вычисляем ниже)
        files_to_move = [r for r in files_to_move if (str(r.get("path") or "")).strip().startswith("local:")]
        if limit_file_paths:
            path_set = set(limit_file_paths)
            files_to_move = [r for r in files_to_move if (r.get("path") or "") in path_set]
            if not files_to_move:
                raise ValueError("Нет файлов из списка limit_file_paths среди кандидатов на перенос в архив")

    if not files_to_move:
        msg = "Нет файлов с заполненным target_folder. Сначала нажмите «Заполнить целевые папки»."
        if destination == "local":
            msg = "Нет файлов для перемещения локально (целевая папка под корнем прогона)."
        elif destination == "archive":
            msg = "Нет файлов для перемещения в фотоархив (нет локальных файлов с целевой папкой)."
        raise ValueError(msg)

    # При dry_run + archive собираем список запланированных операций для отчёта
    planned: list[dict[str, Any]] = [] if (destination == "archive" and dry_run) else []

    # Прогресс для шагов «Переместить локально» / «Перенести в архив»: фаза и счётчики в БД (только при реальном переносе, не при dry_run)
    if destination in ("local", "archive") and not limit_file_paths and not dry_run:
        ps_phase = PipelineStore()
        try:
            ps_phase.update_run(
                run_id=pipeline_run_id,
                step4_phase=destination,
                step4_total=len(files_to_move),
                step4_processed=0,
            )
        finally:
            ps_phase.close()

    moved_count = 0
    errors: list[dict[str, str]] = []
    disk = None
    root_path_clean = root_path[6:] if root_path.startswith("local:") else root_path
    # Имена папок, которые идут в архив (лица, животные, поездки) — файлы в этих папках не помечаем local_done; остальные (Технологии, Чеки и т.д.) — local_done
    _folders_target = list_folders(role="target")
    archive_folder_names = {"Путешествия"}
    for f in _folders_target:
        name = (f.get("name") or "").strip()
        rule = (f.get("content_rule") or "").strip().lower()
        if not name:
            continue
        if rule in ("animals", "any_people") or rule.startswith("only_one") or rule.startswith("multiple_from") or rule.startswith("contains_group"):
            archive_folder_names.add(name)
    # Прогресс шага 4: 50–100% = move; step4_total в БД = 2*кол-во файлов, fill_total = step4_total//2
    step4_total_db = int(pr.get("step4_total") or 0)
    fill_total = (step4_total_db // 2) if step4_total_db >= 2 else 0
    _step4_move_progress_interval = 5  # обновлять прогресс в БД каждые N перемещённых файлов (чаще — счётчик на UI обновляется чаще)
    _step4_phase = destination in ("local", "archive") and not limit_file_paths and not dry_run  # прогресс для шагов 5/6 (не при dry_run)

    for row in files_to_move:
        path = str(row["path"] or "")
        file_name = str(row.get("name") or "") or os.path.basename(path)
        target_folder = str(row.get("target_folder") or "").strip()
        if not path or not target_folder:
            continue
        if path.startswith(target_folder + "/"):
            continue
        dst_path = target_folder + "/" + file_name

        if path.startswith("disk:"):
            if disk is None:
                disk = get_disk()
            try:
                if not dry_run:
                    src_norm = _normalize_yadisk_path(path)
                    dst_norm = _normalize_yadisk_path(dst_path)
                    disk.move(src_norm, dst_norm, overwrite=False)
                    ds = DedupStore()
                    try:
                        ds.update_path(
                            old_path=path,
                            new_path=dst_path,
                            new_name=file_name,
                            new_parent_path=target_folder,
                        )
                        ds.update_run_manual_labels_path(
                            pipeline_run_id=int(pipeline_run_id),
                            old_path=path,
                            new_path=dst_path,
                        )
                        # Не очищаем target_folder после перемещения — отчёт шага 4 показывает распределение по папкам
                    finally:
                        ds.close()
                    _update_gold_file_paths(old_path=path, new_path=dst_path)
                moved_count += 1
                if (fill_total > 0 or _step4_phase) and (moved_count % _step4_move_progress_interval == 0 or moved_count == len(files_to_move)):
                    ps_move = PipelineStore()
                    try:
                        val = moved_count if _step4_phase else (fill_total + moved_count)
                        ps_move.update_run(run_id=pipeline_run_id, step4_processed=val)
                    finally:
                        ps_move.close()
            except Exception as e:  # noqa: BLE001
                errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
        elif path.startswith("local:"):
            # Нормализуем путь под ОС (слэши, абсолютный путь) — иначе os.path.exists может не найти файл (напр. local:C:/tmp/Photo/Агата/... на Windows)
            local_path_raw = path[6:].strip()
            local_path = os.path.normpath(local_path_raw)
            if os.path.isabs(local_path_raw):
                local_path = os.path.abspath(local_path)
            # Перенос в архив: target_folder в БД всегда local:root/... — вычисляем disk-путь для загрузки: disk:/Фото/ + относительный путь после корня
            if destination == "archive":
                root_path_clean = os.path.normpath(root_path[6:] if root_path.startswith("local:") else root_path)
                if target_folder.strip().startswith("local:"):
                    tf_local = os.path.normpath(target_folder.strip()[6:].replace("/", os.sep))
                    rel = tf_local[len(root_path_clean):].lstrip(os.sep) if (tf_local == root_path_clean or tf_local.startswith(root_path_clean + os.sep)) else os.path.basename(tf_local)
                    if os.sep in rel:
                        rel = rel.replace(os.sep, "/")
                    disk_path = "disk:/Фото/" + rel
                else:
                    disk_path = target_folder.strip()
                archive_dst_path = (disk_path.rstrip("/") + "/" + file_name) if disk_path else dst_path
                root_path_clean_arch = root_path[6:] if root_path.startswith("local:") else root_path
                _sorted_dir = os.path.join(root_path_clean_arch, "_sorted") if root_path_clean_arch else None
                dest_local = os.path.join(_sorted_dir, file_name) if _sorted_dir else ""
                # Файл по исходному пути отсутствует — возможно уже перенесён в _sorted (предыдущий перенос обновил диск, но не БД)
                if not os.path.exists(local_path) and _sorted_dir and os.path.isdir(_sorted_dir):
                    found_in_sorted = os.path.join(_sorted_dir, file_name)
                    if not os.path.exists(found_in_sorted):
                        base, ext = os.path.splitext(file_name)
                        n = 1
                        while n < 100:
                            found_in_sorted = os.path.join(_sorted_dir, f"{base}_{n}{ext}")
                            if os.path.exists(found_in_sorted):
                                break
                            n += 1
                        else:
                            found_in_sorted = None
                    if found_in_sorted and not dry_run:
                        try:
                            ds = DedupStore()
                            try:
                                ds.update_path(
                                    old_path=path,
                                    new_path=archive_dst_path,
                                    new_name=file_name,
                                    new_parent_path=disk_path.rstrip("/"),
                                )
                                ds.set_inventory_scope_for_path(path=archive_dst_path, scope="archive")
                                ds.update_run_manual_labels_path(
                                    pipeline_run_id=int(pipeline_run_id),
                                    old_path=path,
                                    new_path=archive_dst_path,
                                )
                            finally:
                                ds.close()
                            fs = FaceStore()
                            try:
                                fs.update_file_path(old_file_path=path, new_file_path=archive_dst_path)
                            finally:
                                fs.close()
                            _update_gold_file_paths(old_path=path, new_path=archive_dst_path)
                            moved_count += 1
                            if (fill_total > 0 or _step4_phase) and (moved_count % _step4_move_progress_interval == 0 or moved_count == len(files_to_move)):
                                ps_move = PipelineStore()
                                try:
                                    val = moved_count if _step4_phase else (fill_total + moved_count)
                                    ps_move.update_run(run_id=pipeline_run_id, step4_processed=val)
                                finally:
                                    ps_move.close()
                        except Exception as e:  # noqa: BLE001
                            errors.append({"path": path, "error": f"Уже в _sorted, но обновление БД: {type(e).__name__}: {e}"})
                    elif found_in_sorted and dry_run:
                        moved_count += 1
                    elif not found_in_sorted:
                        errors.append({"path": path, "error": "File not found"})
                    continue
                elif not os.path.exists(local_path):
                    errors.append({"path": path, "error": "File not found"})
                    continue
                if dry_run:
                    planned.append({
                        "path": path,
                        "file_name": file_name,
                        "target_folder": target_folder.strip(),
                        "disk_path": archive_dst_path,
                        "local_sorted_path": dest_local,
                    })
                    continue
                if disk is None:
                    disk = get_disk()
                try:
                    if not dry_run:
                        root_path_clean = root_path[6:] if root_path.startswith("local:") else root_path
                        _sorted_dir = os.path.join(root_path_clean, "_sorted") if root_path_clean else None
                        try:
                            remote_norm = _normalize_yadisk_path(archive_dst_path)
                            _ensure_yadisk_parent_dirs(disk, remote_norm)
                            disk.upload(local_path, remote_norm)
                        except Exception as upload_err:  # noqa: BLE001
                            # Файл уже на ЯД (409 PathExistsError / DiskResourceAlreadyExistsError) — предыдущая загрузка не обновила БД; синхронизируем БД и локально переносим в _sorted
                            _is_already_exists = (
                                "PathExistsError" in type(upload_err).__name__
                                or "already exists" in str(upload_err).lower()
                                or "409" in str(upload_err)
                                or "DiskResourceAlreadyExistsError" in str(upload_err)
                            )
                            if not _is_already_exists:
                                raise
                            # Уже в архиве: обновляем БД и переносим локальный файл в _sorted
                            if _sorted_dir and os.path.exists(local_path):
                                os.makedirs(_sorted_dir, exist_ok=True)
                                dest_local = os.path.join(_sorted_dir, file_name)
                                if os.path.exists(dest_local):
                                    base, ext = os.path.splitext(file_name)
                                    n = 1
                                    while os.path.exists(dest_local):
                                        dest_local = os.path.join(_sorted_dir, f"{base}_{n}{ext}")
                                        n += 1
                                os.rename(local_path, dest_local)
                            ds = DedupStore()
                            try:
                                ds.update_path(
                                    old_path=path,
                                    new_path=archive_dst_path,
                                    new_name=file_name,
                                    new_parent_path=disk_path.rstrip("/"),
                                )
                                ds.set_inventory_scope_for_path(path=archive_dst_path, scope="archive")
                                ds.update_run_manual_labels_path(
                                    pipeline_run_id=int(pipeline_run_id),
                                    old_path=path,
                                    new_path=archive_dst_path,
                                )
                            finally:
                                ds.close()
                            fs = FaceStore()
                            try:
                                fs.update_file_path(old_file_path=path, new_file_path=archive_dst_path)
                            finally:
                                fs.close()
                            _update_gold_file_paths(old_path=path, new_path=archive_dst_path)
                            if root_path_clean:
                                try:
                                    _remove_empty_dirs_up_to(os.path.dirname(local_path), root_path_clean)
                                except Exception:  # noqa: BLE001
                                    pass
                            moved_count += 1
                            if (fill_total > 0 or _step4_phase) and (moved_count % _step4_move_progress_interval == 0 or moved_count == len(files_to_move)):
                                ps_move = PipelineStore()
                                try:
                                    val = moved_count if _step4_phase else (fill_total + moved_count)
                                    ps_move.update_run(run_id=pipeline_run_id, step4_processed=val)
                                finally:
                                    ps_move.close()
                            continue
                        if _sorted_dir:
                            os.makedirs(_sorted_dir, exist_ok=True)
                            dest_local = os.path.join(_sorted_dir, file_name)
                            if os.path.exists(dest_local):
                                base, ext = os.path.splitext(file_name)
                                n = 1
                                while os.path.exists(dest_local):
                                    dest_local = os.path.join(_sorted_dir, f"{base}_{n}{ext}")
                                    n += 1
                            os.rename(local_path, dest_local)
                        ds = DedupStore()
                        try:
                            ds.update_path(
                                old_path=path,
                                new_path=archive_dst_path,
                                new_name=file_name,
                                new_parent_path=disk_path.rstrip("/"),
                            )
                            ds.set_inventory_scope_for_path(path=archive_dst_path, scope="archive")
                            ds.update_run_manual_labels_path(
                                pipeline_run_id=int(pipeline_run_id),
                                old_path=path,
                                new_path=archive_dst_path,
                            )
                        finally:
                            ds.close()
                        fs = FaceStore()
                        try:
                            fs.update_file_path(old_file_path=path, new_file_path=archive_dst_path)
                        finally:
                            fs.close()
                        _update_gold_file_paths(old_path=path, new_path=archive_dst_path)
                        # После успешного переноса в архив удаляем пустые папки источника (как при «Переместить локально»)
                        if root_path_clean:
                            try:
                                _remove_empty_dirs_up_to(os.path.dirname(local_path), root_path_clean)
                            except Exception:  # noqa: BLE001
                                pass
                    moved_count += 1
                    if (fill_total > 0 or _step4_phase) and (moved_count % _step4_move_progress_interval == 0 or moved_count == len(files_to_move)):
                        ps_move = PipelineStore()
                        try:
                            val = moved_count if _step4_phase else (fill_total + moved_count)
                            ps_move.update_run(run_id=pipeline_run_id, step4_processed=val)
                        finally:
                            ps_move.close()
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
                continue
            # Перемещение локально в target-folder (destination=local или без archive). Для disk:/Фото/... строим локальный аналог под root_path (сохраняем структуру: Путешествия/2024 Турция)
            root_path_clean = root_path[6:] if root_path.startswith("local:") else root_path
            if target_folder.startswith("local:"):
                dst_local = os.path.join(target_folder[6:], file_name)
            elif target_folder.strip().startswith("disk:/Фото/"):
                rel = target_folder.strip()[len("disk:/Фото/"):].lstrip("/").replace("/", os.sep)
                base = root_path_clean if root_path_clean else os.path.dirname(local_path)
                dst_local = os.path.join(base, rel, file_name)
            elif target_folder.startswith("disk:"):
                folder_name = os.path.basename(target_folder)
                if root_path_clean and not root_path_clean.startswith("disk:"):
                    dst_local = os.path.join(root_path_clean, folder_name, file_name)
                else:
                    dst_local = os.path.join(os.path.dirname(local_path), folder_name, file_name)
            else:
                dst_local = os.path.join(target_folder, file_name)
            dst_dir = os.path.dirname(dst_local)
            if os.path.exists(dst_dir):
                if not os.path.isdir(dst_dir):
                    errors.append({"path": path, "error": f"Path {dst_dir} exists but is a file, not a directory"})
                    continue
            else:
                try:
                    os.makedirs(dst_dir, exist_ok=True)
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": path, "error": f"Cannot create directory {dst_dir}: {type(e).__name__}: {e}"})
                    continue
            if not dry_run:
                try:
                    os.rename(local_path, dst_local)
                    new_db_path = "local:" + dst_local
                    ds = DedupStore()
                    try:
                        ds.update_path(
                            old_path=path,
                            new_path=new_db_path,
                            new_name=file_name,
                            new_parent_path="local:" + os.path.dirname(dst_local),
                        )
                        ds.update_run_manual_labels_path(
                            pipeline_run_id=int(pipeline_run_id),
                            old_path=path,
                            new_path=new_db_path,
                        )
                        # local_done только для папок, которые остаются локально (Технологии, Чеки и т.д.). Путешествия и папки лиц/животных — в архив, для них local_done не выставляем.
                        tf = (str(row.get("target_folder") or "").strip())
                        if tf.startswith("local:") and root_path_clean:
                            rel = tf[6:][len(root_path_clean):].lstrip("/\\").replace("\\", "/")
                            rel_first = (rel.split("/")[0] or "").strip()
                            if rel_first and rel_first != "Путешествия" and rel_first not in archive_folder_names:
                                ds.set_inventory_scope_for_path(path=new_db_path, scope="local_done")
                    finally:
                        ds.close()
                    fs = FaceStore()
                    try:
                        fs.update_file_path(old_file_path=path, new_file_path=new_db_path)
                    finally:
                        fs.close()
                    _update_gold_file_paths(old_path=path, new_path=new_db_path)
                    # После успешного переноса удаляем пустые папки источника (только при «Переместить локально»)
                    if destination == "local" and root_path_clean:
                        try:
                            _remove_empty_dirs_up_to(os.path.dirname(local_path), root_path_clean)
                        except Exception:  # noqa: BLE001
                            pass
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
                    continue
            moved_count += 1
            if (fill_total > 0 or _step4_phase) and (moved_count % _step4_move_progress_interval == 0 or moved_count == len(files_to_move)):
                ps_move = PipelineStore()
                try:
                    val = moved_count if _step4_phase else (fill_total + moved_count)
                    ps_move.update_run(run_id=pipeline_run_id, step4_processed=val)
                finally:
                    ps_move.close()

    # Сброс фазы прогресса и отметка «выполнено» для шагов 5/6 только при реальном переносе (не при dry_run — иначе шаг показывался бы выполненным при 0 перенесённых файлах)
    if destination in ("local", "archive") and not limit_file_paths and not dry_run:
        ps_clear = PipelineStore()
        try:
            kwargs_clear: dict[str, Any] = {"run_id": pipeline_run_id, "step4_phase": ""}
            if destination == "local":
                kwargs_clear["step5_done"] = 1
            elif destination == "archive":
                kwargs_clear["step6_done"] = 1
            ps_clear.update_run(**kwargs_clear)
        finally:
            ps_clear.close()

    out: dict[str, Any] = {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "dry_run": bool(dry_run),
        "moved_count": moved_count,
        "errors": errors,
    }
    if destination == "archive":
        out["planned"] = planned
    return out


def _run_sort_into_folders_background(pipeline_run_id: int, destination: str) -> None:
    """Выполняет sort_into_folders_impl в фоне; при ошибке пишет только last_error (без step4_phase, чтобы не падать на старых БД без этой колонки)."""
    try:
        sort_into_folders_impl(pipeline_run_id=pipeline_run_id, dry_run=False, destination=destination)
    except Exception as e:
        logger.exception("sort_into_folders (background) failed: %s", e)
        try:
            ps = PipelineStore()
            try:
                ps.update_run(run_id=pipeline_run_id, last_error=str(e))
            finally:
                ps.close()
        except Exception:
            pass


@router.post("/api/faces/sort-into-folders-start")
def api_faces_sort_into_folders_start(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Запускает перенос файлов в фоне (локально или в архив).
    Параметры: pipeline_run_id (int), destination ("local" | "archive").
    Возвращает сразу; прогресс отдаётся через GET /api/local-pipeline/status (step5/step6).
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    destination = payload.get("destination")
    if destination not in ("local", "archive"):
        raise HTTPException(status_code=400, detail="destination must be 'local' or 'archive'")
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=pipeline_run_id)
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    th = threading.Thread(
        target=_run_sort_into_folders_background,
        args=(pipeline_run_id, destination),
        name=f"sort_into_folders_{destination}_{pipeline_run_id}",
        daemon=True,
    )
    th.start()
    return {"ok": True, "message": "started", "destination": destination}


@router.post("/api/faces/sort-into-folders-archive-limit")
def api_faces_sort_into_folders_archive_limit(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Перенос в архив только для файлов из списка paths (для теста по подмножеству).
    Параметры: pipeline_run_id (int), paths (list[str] — пути local:...).
    Выполняется синхронно, возвращает moved_count и errors.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths must be a non-empty list of file paths")
    paths = [str(p).strip() for p in paths if p]
    if not paths:
        raise HTTPException(status_code=400, detail="paths must contain at least one path")
    try:
        return sort_into_folders_impl(
            pipeline_run_id=pipeline_run_id,
            dry_run=False,
            destination="archive",
            limit_file_paths=paths,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/faces/sort-into-folders")
def api_faces_sort_into_folders(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """HTTP-обёртка: перемещает файлы по целевым папкам (см. sort_into_folders_impl)."""
    pipeline_run_id = payload.get("pipeline_run_id")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    dry_run = payload.get("dry_run", False)
    destination = payload.get("destination")
    try:
        return sort_into_folders_impl(pipeline_run_id=pipeline_run_id, dry_run=dry_run, destination=destination)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/faces/resort-file")
def api_faces_resort_file(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Пересортировка одного файла: заново вычисляет target_folder (с учётом персоны),
    записывает в БД, перемещает файл и обнуляет target_folder.
    Используется для исправления ошибочно размещённых файлов (например, в «Другие люди» вместо «Агата»).

    Параметры: pipeline_run_id (int), path (str).
    Возвращает: ok, new_path, error (при ошибке).
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    path = payload.get("path")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(path, str) or not (path.startswith("local:") or path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="path must start with local: or disk:")

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        raise HTTPException(status_code=400, detail="face_run_id is not set yet (step 3 not started)")
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")

    conn = get_connection()
    try:
        from common.db import _get_file_id

        file_id = _get_file_id(conn, file_path=path)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not file_id:
        raise HTTPException(status_code=404, detail="File not found in DB by path")

    ds = DedupStore()
    ps = PipelineStore()
    conn = get_connection()
    try:
        cur_preclean = ps.conn.cursor()
        cur_preclean.execute(
            "SELECT kind FROM preclean_moves WHERE pipeline_run_id = ? AND src_path = ?",
            (int(pipeline_run_id), path),
        )
        preclean_row = cur_preclean.fetchone()
        preclean_kind = str(preclean_row[0] or "").strip() if preclean_row else None

        cur = ds.conn.cursor()
        cur.execute(
            """
            SELECT
              f.id, f.path, f.name,
              COALESCE(m.faces_manual_label, '') AS faces_manual_label,
              COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
              COALESCE(f.faces_auto_quarantine, 0) AS faces_auto_quarantine,
              COALESCE(f.faces_count, 0) AS faces_count,
              COALESCE(m.animals_manual, 0) AS animals_manual,
              COALESCE(f.animals_auto, 0) AS animals_auto,
              COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual
            FROM files f
            LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE f.id = ?
            """,
            (int(pipeline_run_id), file_id),
        )
        r = cur.fetchone()
    finally:
        ds.close()
        ps.close()

    if not r:
        raise HTTPException(status_code=404, detail="File row not found")
    r = dict(r)
    path = str(r.get("path") or "")
    file_name = str(r.get("name") or "") or os.path.basename(path)

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

    target_folders = list_folders(role="target")
    person_name = None
    if effective_tab == "faces":
        try:
            person_name = _resolve_target_folder_for_faces(
                conn,
                file_id=int(file_id),
                pipeline_run_id=int(pipeline_run_id),
                face_run_id=face_run_id_i,
                target_folders=target_folders,
            )
        except Exception as e:
            return {"ok": False, "error": f"resolve_target_folder: {type(e).__name__}: {e}"}

    target_folder = _determine_target_folder(
        path=path,
        effective_tab=effective_tab or "no_faces",
        root_path=root_path,
        preclean_kind=preclean_kind,
        person_name=person_name,
        target_folders=target_folders,
    )
    if not target_folder:
        return {"ok": False, "error": "target_folder could not be determined"}

    ds = DedupStore()
    try:
        ds.set_target_folder(file_id=int(file_id), target_folder=target_folder)
    finally:
        ds.close()

    dst_path = target_folder + "/" + file_name
    if path.startswith(target_folder + "/"):
        try:
            conn.close()
        except Exception:
            pass
        return {"ok": True, "new_path": path, "message": "Already in target folder"}

    if path.startswith("disk:"):
        try:
            disk = get_disk()
            src_norm = _normalize_yadisk_path(path)
            dst_norm = _normalize_yadisk_path(dst_path)
            disk.move(src_norm, dst_norm, overwrite=False)
            ds = DedupStore()
            try:
                ds.update_path(old_path=path, new_path=dst_path, new_name=file_name, new_parent_path=target_folder)
                ds.update_run_manual_labels_path(pipeline_run_id=int(pipeline_run_id), old_path=path, new_path=dst_path)
                ds.clear_target_folder(paths=[path])
            finally:
                ds.close()
            _update_gold_file_paths(old_path=path, new_path=dst_path)
        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        local_path = path[6:]
        if not os.path.exists(local_path):
            try:
                conn.close()
            except Exception:
                pass
            return {"ok": False, "error": "File not found on disk"}
        if target_folder.startswith("local:"):
            dst_local = os.path.join(target_folder[6:], file_name)
        elif target_folder.startswith("disk:"):
            folder_name = os.path.basename(target_folder)
            root_path_clean = root_path[6:] if root_path.startswith("local:") else root_path
            dst_local = os.path.join(root_path_clean, folder_name, file_name) if root_path_clean and not root_path_clean.startswith("disk:") else os.path.join(os.path.dirname(local_path), folder_name, file_name)
        else:
            dst_local = os.path.join(target_folder, file_name)
        dst_dir = os.path.dirname(dst_local)
        if not os.path.exists(dst_dir):
            try:
                os.makedirs(dst_dir, exist_ok=True)
            except Exception as e:
                try:
                    conn.close()
                except Exception:
                    pass
                return {"ok": False, "error": f"Cannot create directory: {type(e).__name__}: {e}"}
        try:
            os.rename(local_path, dst_local)
            new_db_path = "local:" + dst_local
            ds = DedupStore()
            try:
                ds.update_path(old_path=path, new_path=new_db_path, new_name=file_name, new_parent_path="local:" + os.path.dirname(dst_local))
                ds.update_run_manual_labels_path(pipeline_run_id=int(pipeline_run_id), old_path=path, new_path=new_db_path)
                ds.clear_target_folder(paths=[path])
            finally:
                ds.close()
            fs = FaceStore()
            try:
                fs.update_file_path(old_file_path=path, new_file_path=new_db_path)
            finally:
                fs.close()
            _update_gold_file_paths(old_path=path, new_path=new_db_path)
        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    try:
        conn.close()
    except Exception:
        pass
    return {"ok": True, "new_path": dst_path}


def _update_gold_file_paths(old_path: str, new_path: str) -> int:
    """
    Обновляет пути в gold файлах при перемещении файла.
    Заменяет старый путь на новый во всех gold файлах (txt и ndjson).
    Возвращает количество обновлённых записей.
    """
    updated_count = 0
    old_path_normalized = old_path.strip()
    new_path_normalized = new_path.strip()
    
    # Варианты старого пути для поиска (с префиксом и без)
    old_path_variants = {old_path_normalized}
    if old_path_normalized.startswith("local:"):
        old_path_variants.add(old_path_normalized[6:])  # без "local:"
    elif not old_path_normalized.startswith("disk:"):
        old_path_variants.add("local:" + old_path_normalized)  # с "local:"
    
    # Обновляем во всех txt gold файлах
    gold_map = gold_file_map()
    for name, gold_path in gold_map.items():
        if not gold_path.exists():
            continue
        lines = gold_read_lines(gold_path)
        modified = False
        new_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped in old_path_variants:
                # Заменяем старый путь на новый
                new_lines.append(new_path_normalized)
                updated_count += 1
                modified = True
            else:
                new_lines.append(line)
        if modified:
            gold_write_lines(gold_path, new_lines)
    
    # Обновляем в NDJSON gold файлах (faces_manual_rects_gold.ndjson, faces_video_frames_gold.ndjson)
    for ndjson_path in [gold_faces_manual_rects_path(), gold_faces_video_frames_path()]:
        if not ndjson_path.exists():
            continue
        items = gold_read_ndjson_by_path(ndjson_path)
        modified = False
        # Обновляем все варианты старого пути
        for old_variant in old_path_variants:
            if old_variant in items:
                # Сохраняем данные, но с новым путём
                item_data = items[old_variant]
                del items[old_variant]
                items[new_path_normalized] = item_data
                updated_count += 1
                modified = True
        if modified:
            gold_write_ndjson_by_path(ndjson_path, items)
    
    return updated_count


def _delete_from_all_gold_files(path: str) -> int:
    """
    Удаляет путь из всех gold файлов (txt и ndjson).
    Возвращает количество удалённых записей.
    """
    removed_count = 0
    path_normalized = path.strip()
    # Варианты пути для поиска (с префиксом и без)
    path_variants = {path_normalized}
    if path_normalized.startswith("local:"):
        path_variants.add(path_normalized[6:])  # без "local:"
    elif not path_normalized.startswith("disk:"):
        path_variants.add("local:" + path_normalized)  # с "local:"

    # Удаляем из всех txt gold файлов
    gold_map = gold_file_map()
    for name, gold_path in gold_map.items():
        if not gold_path.exists():
            continue
        lines = gold_read_lines(gold_path)
        original_count = len(lines)
        # Удаляем все варианты пути
        new_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped in path_variants:
                removed_count += 1
                continue
            new_lines.append(line)
        if len(new_lines) != original_count:
            gold_write_lines(gold_path, new_lines)

    # Удаляем из NDJSON gold файлов (faces_manual_rects_gold.ndjson, faces_video_frames_gold.ndjson)
    for ndjson_path in [gold_faces_manual_rects_path(), gold_faces_video_frames_path()]:
        if not ndjson_path.exists():
            continue
        items = gold_read_ndjson_by_path(ndjson_path)
        modified = False
        # Удаляем все варианты пути
        for variant in path_variants:
            if variant in items:
                del items[variant]
                removed_count += 1
                modified = True
        if modified:
            gold_write_ndjson_by_path(ndjson_path, items)

    return removed_count


@router.post("/api/faces/delete")
def api_faces_delete(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Удаляет файл: перемещает в _delete, помечает как deleted в БД, удаляет из gold.

    Параметры:
    - pipeline_run_id: int (обязательно)
    - path: str (обязательно)

    Возвращает информацию об удалении и данные для undo.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    path = payload.get("path")

    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(path, str) or not (path.startswith("local:") or path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="path must start with local: or disk:")

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")

    root_path = str(pr.get("root_path") or "")

    # Получаем текущее состояние файла для undo
    ds = DedupStore()
    fs = FaceStore()
    undo_data: dict[str, Any] = {
        "path": path,
        "action": "delete",
    }
    try:
        # Получаем информацию о файле
        cur = ds.conn.cursor()
        cur.execute(
            """
            SELECT
              f.path, f.name, f.parent_path,
              COALESCE(m.faces_manual_label, '') AS faces_manual_label,
              COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
              COALESCE(m.animals_manual, 0) AS animals_manual,
              COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual,
              COALESCE(m.people_no_face_person, '') AS people_no_face_person,
              COALESCE(f.faces_count, 0) AS faces_count
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE f.path = ? AND f.status != 'deleted'
            LIMIT 1
            """,
            (int(pipeline_run_id), path),
        )
        row = cur.fetchone()
        if row:
            r = dict(row)
            # Определяем effective_tab для undo
            effective_tab = "no_faces"
            if r.get("people_no_face_manual"):
                effective_tab = "people_no_face"
            elif (r.get("faces_manual_label") or "").lower().strip() == "faces":
                effective_tab = "faces"
            elif (r.get("faces_manual_label") or "").lower().strip() == "no_faces":
                effective_tab = "no_faces"
            elif r.get("quarantine_manual") and r.get("faces_count", 0) > 0:
                effective_tab = "quarantine"
            elif r.get("animals_manual"):
                effective_tab = "animals"
            elif r.get("faces_auto_quarantine") and r.get("faces_count", 0) > 0:
                effective_tab = "quarantine"
            elif r.get("faces_count", 0) > 0:
                effective_tab = "faces"

            undo_data.update(
                {
                    "original_path": path,
                    "original_name": str(r.get("name") or ""),
                    "original_parent_path": str(r.get("parent_path") or ""),
                    "original_effective_tab": effective_tab,
                    "original_faces_manual_label": str(r.get("faces_manual_label") or ""),
                    "original_quarantine_manual": bool(r.get("quarantine_manual")),
                    "original_animals_manual": bool(r.get("animals_manual")),
                    "original_people_no_face_manual": bool(r.get("people_no_face_manual")),
                    "original_people_no_face_person": str(r.get("people_no_face_person") or ""),
                }
            )
    finally:
        ds.close()
        fs.close()

    # Определяем путь к папке _delete
    if root_path.startswith("disk:"):
        delete_folder = f"{root_path.rstrip('/')}/_delete"
    elif root_path.startswith("local:"):
        delete_folder = f"local:{os.path.join(root_path[6:], '_delete')}"
    else:
        delete_folder = f"local:{os.path.join(root_path, '_delete')}"

    # Формируем путь назначения
    file_name = undo_data.get("original_name") or os.path.basename(path)
    if not file_name:
        file_name = "file"

    # Перемещаем файл в _delete
    moved = False
    delete_path = None
    if path.startswith("disk:"):
        # YaDisk
        disk = get_disk()
        try:
            dst_path = f"{delete_folder}/{file_name}"
            src_norm = _normalize_yadisk_path(path)
            dst_norm = _normalize_yadisk_path(dst_path)
            disk.move(src_norm, dst_norm, overwrite=False)
            delete_path = dst_path
            moved = True
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Cannot move file to _delete: {type(e).__name__}: {e}") from e
    elif path.startswith("local:"):
        # Локальный файл
        local_path = path[6:]  # убираем "local:"
        actual_local_path = None
        
        # Проверяем, существует ли файл по указанному пути
        if os.path.exists(local_path) and os.path.isfile(local_path):
            actual_local_path = local_path
        else:
            # Файл не найден по указанному пути - возможно, он был перемещён
            # Пытаемся найти файл по имени в родительской директории или в корне прогона
            parent_dir = os.path.dirname(local_path)
            if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
                # Ищем файл с таким же именем в родительской директории
                potential_path = os.path.join(parent_dir, file_name)
                if os.path.exists(potential_path) and os.path.isfile(potential_path):
                    actual_local_path = potential_path
            # Если не нашли, пробуем поискать в корне прогона
            if not actual_local_path and root_path.startswith("local:"):
                root_local = root_path[6:]
                if os.path.exists(root_local) and os.path.isdir(root_local):
                    # Рекурсивно ищем файл по имени в корне
                    for root, dirs, files in os.walk(root_local):
                        if file_name in files:
                            actual_local_path = os.path.join(root, file_name)
                            break
        
        if not actual_local_path:
            # Файл физически не найден - просто помечаем как deleted в БД
            # Это может быть, если файл уже был удалён вручную или перемещён
            delete_path = None
            moved = False
        else:
            # Файл найден - перемещаем в _delete
            if delete_folder.startswith("local:"):
                dst_local = os.path.join(delete_folder[6:], file_name)
            else:
                dst_local = os.path.join(delete_folder, file_name)
            dst_dir = os.path.dirname(dst_local)
            try:
                os.makedirs(dst_dir, exist_ok=True)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Cannot create _delete directory: {type(e).__name__}: {e}") from e
            try:
                os.rename(actual_local_path, dst_local)
                delete_path = "local:" + dst_local
                moved = True
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Cannot move file to _delete: {type(e).__name__}: {e}") from e

    # Обновляем путь в БД и помечаем как deleted
    ds = DedupStore()
    try:
        if delete_path:
            # Файл был перемещён в _delete - обновляем путь в БД
            ds.update_path(
                old_path=path,
                new_path=delete_path,
                new_name=file_name,
                new_parent_path=delete_folder,
            )
            # Обновляем пути в manual labels
            ds.update_run_manual_labels_path(
                pipeline_run_id=int(pipeline_run_id),
                old_path=path,
                new_path=delete_path,
            )
            # Помечаем как deleted
            ds.mark_deleted(paths=[delete_path])
        else:
            # Файл физически не найден - просто помечаем как deleted по исходному пути
            ds.mark_deleted(paths=[path])
    finally:
        ds.close()

    # Обновляем file_id в photo_rectangles (после перемещения файла file_id не меняется, но нужно обновить для нового пути)
    # После рефакторинга photo_rectangles использует file_id, а не file_path
    # Поэтому нам не нужно обновлять photo_rectangles - file_id остается тем же, путь обновляется в таблице files
    # Но если есть rectangles с run_id, они уже связаны через file_id, так что обновление не требуется

    # Удаляем из всех gold файлов
    removed_from_gold = _delete_from_all_gold_files(path)

    undo_data["delete_path"] = delete_path

    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "path": path,
        "delete_path": delete_path,
        "removed_from_gold": removed_from_gold,
        "undo_data": undo_data,
    }


@router.post("/api/faces/restore-from-delete")
def api_faces_restore_from_delete(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Восстанавливает файл из _delete обратно в исходное место.

    Параметры:
    - pipeline_run_id: int (обязательно)
    - delete_path: str (путь в _delete)
    - original_path: str (исходный путь)
    - original_name: str (исходное имя)
    - original_parent_path: str (исходная родительская папка)
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    delete_path = payload.get("delete_path")
    original_path = payload.get("original_path")
    original_name = payload.get("original_name")
    original_parent_path = payload.get("original_parent_path")

    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(delete_path, str) or not (delete_path.startswith("local:") or delete_path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="delete_path must start with local: or disk:")
    if not isinstance(original_path, str) or not (original_path.startswith("local:") or original_path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="original_path must start with local: or disk:")

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")

    # Перемещаем файл обратно
    restored = False
    if delete_path.startswith("disk:"):
        # YaDisk
        disk = get_disk()
        try:
            src_norm = _normalize_yadisk_path(delete_path)
            dst_norm = _normalize_yadisk_path(original_path)
            disk.move(src_norm, dst_norm, overwrite=False)
            restored = True
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Cannot restore file from _delete: {type(e).__name__}: {e}") from e
    elif delete_path.startswith("local:"):
        # Локальный файл
        local_delete_path = delete_path[6:]  # убираем "local:"
        if not os.path.exists(local_delete_path):
            raise HTTPException(status_code=404, detail="File not found in _delete")
        local_original_path = original_path[6:] if original_path.startswith("local:") else original_path
        # Создаём родительскую директорию, если нужно
        dst_dir = os.path.dirname(local_original_path)
        try:
            os.makedirs(dst_dir, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Cannot create directory: {type(e).__name__}: {e}") from e
        try:
            os.rename(local_delete_path, local_original_path)
            restored = True
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Cannot restore file from _delete: {type(e).__name__}: {e}") from e

    if not restored:
        raise HTTPException(status_code=500, detail="File was not restored")

    # Обновляем путь в БД и снимаем статус deleted
    ds = DedupStore()
    try:
        # Обновляем путь в БД
        ds.update_path(
            old_path=delete_path,
            new_path=original_path,
            new_name=original_name,
            new_parent_path=original_parent_path,
        )
        # Обновляем пути в manual labels
        ds.update_run_manual_labels_path(
            pipeline_run_id=int(pipeline_run_id),
            old_path=delete_path,
            new_path=original_path,
        )
        # Снимаем статус deleted
        cur = ds.conn.cursor()
        cur.execute("UPDATE files SET status = 'new', error = NULL WHERE path = ?", (original_path,))
        ds.conn.commit()
    finally:
        ds.close()

    # После рефакторинга photo_rectangles использует file_id, а не file_path
    # При восстановлении файла file_id не меняется, путь обновляется в таблице files через ds.update_path()
    # Поэтому обновление photo_rectangles не требуется

    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "delete_path": delete_path,
        "restored_path": original_path,
    }


@router.post("/api/faces/fix-clipping")
def api_faces_fix_clipping(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Пересчитывает координаты bbox для одного файла с исправленным клиппингом.
    
    Параметры:
    - pipeline_run_id: int (опционально, приоритетнее)
    - face_run_id: int (опционально, используется если pipeline_run_id не указан)
    - rectangle_id: int (опционально, используется для получения face_run_id если face_run_id не указан)
    - path: str (обязательно, disk:/... или local:...)
    
    Возвращает количество обновлённых записей.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    face_run_id = payload.get("face_run_id")
    rectangle_id = payload.get("rectangle_id")
    path = payload.get("path")
    
    if not isinstance(path, str) or not (path.startswith("local:") or path.startswith("disk:")):
        raise HTTPException(status_code=400, detail="path must start with local: or disk:")
    
    # Нормализуем значения: если они не None и не int, пытаемся преобразовать
    if pipeline_run_id is not None and not isinstance(pipeline_run_id, int):
        try:
            pipeline_run_id = int(pipeline_run_id)
        except (ValueError, TypeError):
            pipeline_run_id = None
    
    if face_run_id is not None and not isinstance(face_run_id, int):
        try:
            face_run_id = int(face_run_id)
        except (ValueError, TypeError):
            face_run_id = None
    
    if rectangle_id is not None and not isinstance(rectangle_id, int):
        try:
            rectangle_id = int(rectangle_id)
        except (ValueError, TypeError):
            rectangle_id = None
    
    # pipeline_run_id и face_run_id не обязательны - скрипт recalc_face_bbox.py работает напрямую с БД по path
    # Если они указаны, пытаемся найти pipeline_run_id для логирования (опционально)
    if not isinstance(pipeline_run_id, int):
        # Если face_run_id не указан, но есть rectangle_id, получаем run_id из face_rectangle
        if not isinstance(face_run_id, int) and isinstance(rectangle_id, int):
            from common.db import FaceStore
            fs = FaceStore()
            try:
                cur = fs.conn.cursor()
                cur.execute(
                    """
                    SELECT run_id
                    FROM photo_rectangles
                    WHERE id = ?
                    """,
                    (int(rectangle_id),),
                )
                row = cur.fetchone()
                if row and row["run_id"] is not None:
                    face_run_id = int(row["run_id"])
            except Exception:  # noqa: BLE001
                pass
            finally:
                fs.close()
        
        # Если есть face_run_id, пытаемся найти pipeline_run_id (опционально, для логирования)
        if isinstance(face_run_id, int):
            ps = PipelineStore()
            try:
                cur = ps.conn.cursor()
                cur.execute(
                    """
                    SELECT id 
                    FROM pipeline_runs 
                    WHERE face_run_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (int(face_run_id),),
                )
                row = cur.fetchone()
                if row:
                    pipeline_run_id = int(row["id"])
            except Exception:  # noqa: BLE001
                pass
            finally:
                ps.close()
    
    # pipeline_run_id не обязателен - скрипт recalc_face_bbox.py работает напрямую с БД
    # Если pipeline_run_id есть, проверяем его существование (best-effort, не критично)
    if isinstance(pipeline_run_id, int):
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
            if not pr:
                _agent_dbg(
                    hypothesis_id="FIX_CLIPPING",
                    location="web_api/routers/faces.py:api_faces_fix_clipping",
                    message="Pipeline run not found, but continuing anyway",
                    data={"pipeline_run_id": pipeline_run_id},
                )
        except Exception:  # noqa: BLE001
            _agent_dbg(
                hypothesis_id="FIX_CLIPPING",
                location="web_api/routers/faces.py:api_faces_fix_clipping",
                message="Error checking pipeline_run, but continuing anyway",
                data={"pipeline_run_id": pipeline_run_id},
            )
        finally:
            ps.close()
    
    # Создаём временный файл с путём
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(path + "\n")
        paths_file = f.name
    
    try:
        # Запускаем скрипт пересчёта для одного файла
        script_path = _repo_root() / "backend" / "scripts" / "debug" / "recalc_face_bbox.py"
        if not script_path.exists():
            raise HTTPException(status_code=500, detail=f"Script not found: {script_path}")
        
        # Используем Python из .venv-face, где установлены cv2 и другие зависимости
        py_venv = _venv_face_python()
        if not py_venv.exists():
            raise HTTPException(status_code=500, detail=f"Missing .venv-face python: {py_venv}")
        python_exe = str(py_venv)
        
        _agent_dbg(
            hypothesis_id="FIX_CLIPPING",
            location="web_api/routers/faces.py:api_faces_fix_clipping",
            message="Starting script",
            data={"script_path": str(script_path), "python_exe": python_exe, "path": path, "paths_file": paths_file},
        )
        
        # Устанавливаем PYTHONPATH для импорта модулей из backend/
        env = os.environ.copy()
        backend_path = str(_repo_root() / "backend")
        env["PYTHONPATH"] = backend_path
        env["PYTHONUNBUFFERED"] = "1"  # Важно для UI: без буферизации stdout
        
        _agent_dbg(
            hypothesis_id="FIX_CLIPPING",
            location="web_api/routers/faces.py:api_faces_fix_clipping",
            message="About to run script",
            data={"script_path": str(script_path), "python_exe": python_exe, "paths_file": paths_file},
        )
        
        result = subprocess.run(
            [
                python_exe,
                str(script_path),
                "--paths-file", paths_file,
                "--apply",
            ],
            capture_output=True,
            text=True,
            timeout=120,  # 2 минуты максимум
            cwd=str(_repo_root()),
            env=env,  # Передаём окружение с PYTHONPATH
        )
        
        # Логируем stdout и stderr СРАЗУ после выполнения скрипта
        _agent_dbg(
            hypothesis_id="FIX_CLIPPING",
            location="web_api/routers/faces.py:api_faces_fix_clipping",
            message="Script output (immediate)",
            data={
                "returncode": result.returncode,
                "stdout": (result.stdout or "")[:2000],
                "stderr": (result.stderr or "")[:2000],
                "stdout_len": len(result.stdout or ""),
                "stderr_len": len(result.stderr or ""),
            },
        )
        
        _agent_dbg(
            hypothesis_id="FIX_CLIPPING",
            location="web_api/routers/faces.py:api_faces_fix_clipping",
            message="Script finished",
            data={"returncode": result.returncode, "stdout_len": len(result.stdout or ""), "stderr_len": len(result.stderr or "")},
        )
        
        if result.returncode != 0:
            error_detail = result.stderr[:2000] if result.stderr else "Unknown error"
            stdout_preview = result.stdout[:1000] if result.stdout else ""  # Увеличил до 1000 символов
            _agent_dbg(
                hypothesis_id="FIX_CLIPPING",
                location="web_api/routers/faces.py:api_faces_fix_clipping",
                message="Script error",
                data={"returncode": result.returncode, "stderr": error_detail, "stdout": stdout_preview, "stdout_full_len": len(result.stdout or "")},
            )
            # Пытаемся обработать stdout даже при ошибке, если там есть информация об обновлениях
            output = result.stdout or ""
            updated_count = 0
            for line in output.splitlines():
                if "updated" in line.lower() and "faces" in line.lower():
                    import re
                    match = re.search(r"updated\s+(\d+)", line.lower())
                    if match:
                        updated_count = int(match.group(1))
                        break
            # Если всё-таки обновили что-то, не бросаем ошибку
            if updated_count > 0:
                _agent_dbg(
                    hypothesis_id="FIX_CLIPPING",
                    location="web_api/routers/faces.py:api_faces_fix_clipping",
                    message="Script had errors but updated faces",
                    data={"updated_count": updated_count, "stderr_preview": error_detail[:200]},
                )
                return {
                    "ok": True,
                    "pipeline_run_id": int(pipeline_run_id) if isinstance(pipeline_run_id, int) else None,
                    "path": path,
                    "updated_count": updated_count,
                    "warning": error_detail[:200] if error_detail else None,
                }
            raise HTTPException(
                status_code=500,
                detail=f"Script error (code {result.returncode}): {error_detail[:500]}",
            )
        
        # Парсим вывод скрипта (ожидаем "updated N faces" или "would update N faces")
        output = result.stdout or ""
        _agent_dbg(
            hypothesis_id="FIX_CLIPPING",
            location="web_api/routers/faces.py:api_faces_fix_clipping",
            message="Script stdout",
            data={"stdout": output[:1000], "stdout_len": len(output)},
        )
        
        updated_count = 0
        for line in output.splitlines():
            if "updated" in line.lower() and "faces" in line.lower():
                # Ищем число после "updated"
                import re
                match = re.search(r"updated\s+(\d+)", line.lower())
                if match:
                    updated_count = int(match.group(1))
                    break
        
        _agent_dbg(
            hypothesis_id="FIX_CLIPPING",
            location="web_api/routers/faces.py:api_faces_fix_clipping",
            message="Parsed result",
            data={"updated_count": updated_count, "output_preview": output[:200]},
        )
        
        return {
            "ok": True,
            "pipeline_run_id": int(pipeline_run_id) if isinstance(pipeline_run_id, int) else None,
            "path": path,
            "updated_count": updated_count,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Script timeout")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        try:
            os.unlink(paths_file)
        except Exception:
            pass


@router.post("/api/debug/client-log")
def api_debug_client_log(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Записывает логи от клиента в файл для отладки.
    Параметры:
    - level: str (log, warn, error)
    - message: str
    - data: dict (опционально)
    """
    try:
        log_dir = _repo_root() / "backend" / "scripts" / "debug" / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "photo_card_client.log"
        
        level = payload.get("level", "log")
        message = payload.get("message", "")
        data = payload.get("data", {})
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        log_line = f"[{timestamp}] [{level.upper()}] {message}"
        if data:
            log_line += f" | {json.dumps(data, ensure_ascii=False)}"
        log_line += "\n"
        
        with log_file.open("a", encoding="utf-8") as f:
            f.write(log_line)
        
        return {"ok": True, "logged": True}
    except Exception as e:
        logger.error(f"Error writing client log: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}