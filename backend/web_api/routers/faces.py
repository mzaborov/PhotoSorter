from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import subprocess
import tempfile
import logging
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

from common.db import DedupStore, FaceStore, PipelineStore, list_folders
from common.yadisk_client import get_disk
from logic.gold.store import gold_expected_tab_by_path, gold_file_map, gold_read_lines, gold_write_lines, gold_read_ndjson_by_path, gold_write_ndjson_by_path, gold_faces_manual_rects_path, gold_faces_video_frames_path

router = APIRouter()

APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


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
                trip_date = datetime.fromisoformat(taken_at[:10].replace("Z", "+00:00"))
            except Exception:
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
                first_date = datetime.fromisoformat(first_date_str[:10].replace("Z", "+00:00"))
            except Exception:
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
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    page: int = 1,
    page_size: int = 60,
) -> dict[str, Any]:
    start_time = time.time()
    msg = f"[API] api_faces_results: начало, pipeline_run_id={pipeline_run_id}, tab={tab}, subtab={subtab}, page={page}, page_size={page_size}"
    logger.info(msg)
    print(msg)  # Дублируем в print для гарантированного вывода
    tab_n = (tab or "").strip().lower()
    if tab_n not in ("faces", "no_faces", "quarantine", "animals", "people_no_face"):
        raise HTTPException(status_code=400, detail="tab must be faces|no_faces|quarantine|animals|people_no_face")
    # Карантин теперь показывается в "Нет людей" -> "К разбору", но оставляем поддержку для обратной совместимости
    if tab_n == "quarantine":
        tab_n = "no_faces"

    subtab_n = (subtab or "").strip().lower() or "all"
    person_id_filter: int | None = None
    group_path_filter: str | None = None
    if tab_n == "faces":
        if subtab_n.startswith("person_"):
            try:
                person_id_filter = int(subtab_n.replace("person_", ""))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid person_id in subtab")
        elif subtab_n not in ("all", "many_faces", "unsorted"):
            raise HTTPException(status_code=400, detail="subtab for faces must be all|many_faces|unsorted|person_<id>")
    elif tab_n == "no_faces":
        if subtab_n.startswith("group_"):
            # Декодируем путь группы (может содержать "/" и другие спецсимволы)
            group_path_filter = urllib.parse.unquote(subtab_n.replace("group_", ""))
            # Убираем лишние пробелы (данные уже нормализованы в БД)
            group_path_filter = group_path_filter.strip()
            print(f"[DEBUG] Декодирован group_path_filter: '{group_path_filter}' из subtab_n='{subtab_n}'")
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

    eff_sql = """
    CASE
      WHEN COALESCE(m.people_no_face_manual, 0) = 1 THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'faces' THEN 'faces'
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
        sub_where = "COALESCE(faces_count, 0) >= 8"
    elif tab_n == "faces" and subtab_n == "all":
        sub_where = "COALESCE(faces_count, 0) < 8"
    elif tab_n == "faces" and subtab_n == "unsorted":
        # "К разбору": файлы без привязки к персонам ИЛИ люди без лиц (people_no_face_manual=1)
        sub_where = "COALESCE(faces_count, 0) < 8"
        # person_filter_sql будет исключать файлы с привязкой к персонам, но включать people_no_face_manual=1
        # Привязка может быть через: face_labels, person_rectangles, file_persons, или через кластеры
        person_filter_sql = """
        (
          -- Файл не привязан ни к одной персоне через лица (прямая привязка)
          NOT EXISTS (
              SELECT 1 FROM face_labels fl
              JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
              WHERE fr.file_path = f.path AND fr.run_id = ? AND fl.person_id IS NOT NULL
          )
          AND NOT EXISTS (
              -- Файл не привязан ни к одной персоне через кластеры (оптимизированный вариант)
              SELECT 1 FROM face_rectangles fr_cluster
              JOIN face_cluster_members fcm_all ON fcm_all.face_rectangle_id = fr_cluster.id
              JOIN (
                  -- Находим кластеры с персонами через подзапрос
                  SELECT DISTINCT fcm_labeled.cluster_id
                  FROM face_labels fl_cluster
                  JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl_cluster.face_rectangle_id
                  JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
                  WHERE fl_cluster.person_id IS NOT NULL
                    AND (fc.run_id = ? OR fc.archive_scope = 'archive')
              ) person_clusters ON person_clusters.cluster_id = fcm_all.cluster_id
              WHERE fr_cluster.file_path = f.path 
                AND fr_cluster.run_id = ? 
                AND COALESCE(fr_cluster.ignore_flag, 0) = 0
          )
          AND NOT EXISTS (
              -- Файл не привязан ни к одной персоне через прямоугольники
              SELECT 1 FROM person_rectangles pr
              WHERE pr.file_path = f.path AND pr.pipeline_run_id = ? AND pr.person_id IS NOT NULL
          )
          AND NOT EXISTS (
              -- Файл не привязан ни к одной персоне напрямую
              SELECT 1 FROM file_persons fp
              WHERE fp.file_path = f.path AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
          )
        ) OR COALESCE(m.people_no_face_manual, 0) = 1
        """
        # Параметры для оптимизированного запроса через кластеры: [face_run_id для подзапроса, face_run_id для fr_cluster]
        person_filter_params = [face_run_id_i, face_run_id_i, int(pipeline_run_id), int(pipeline_run_id)]
    elif tab_n == "no_faces" and subtab_n in ("unsorted", "unsorted_photos", "unsorted_videos"):
        # "К разбору" для "Нет людей": файлы без группы
        group_filter_sql = """
        NOT EXISTS (
            SELECT 1 FROM file_groups fg
            WHERE fg.file_path = f.path AND fg.pipeline_run_id = ?
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
        
        print(f"[DEBUG] Фильтр 'К разбору': group_filter_sql установлен, media_filter={subtab_n}, params={group_filter_params}")
    elif tab_n == "no_faces" and group_path_filter is not None:
        # Фильтр по конкретной группе
        # Ищем точно по названию (данные уже нормализованы в БД)
        group_filter_sql = """
        EXISTS (
            SELECT 1 FROM file_groups fg
            WHERE fg.file_path = f.path AND fg.pipeline_run_id = ? 
            AND fg.group_path = ?
        )
        """
        group_filter_params = [int(pipeline_run_id), str(group_path_filter)]
        print(f"[DEBUG] Фильтр по группе '{group_path_filter}': group_filter_sql установлен, params={group_filter_params}")
    elif tab_n == "faces" and person_id_filter is not None:
        # Фильтр по персоне: файл должен быть привязан к персоне через любой из 4 способов
        person_filter_sql = """
        EXISTS (
            -- Через лица (прямая привязка)
            SELECT 1 FROM face_labels fl
            JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
            WHERE fr.file_path = f.path AND fr.run_id = ? AND fl.person_id = ?
        ) OR EXISTS (
            -- Через кластеры (оптимизированный вариант с подзапросом)
            SELECT 1 FROM face_rectangles fr_cluster
            JOIN face_cluster_members fcm_all ON fcm_all.face_rectangle_id = fr_cluster.id
            JOIN (
                -- Находим кластеры с конкретной персоной через подзапрос
                SELECT DISTINCT fcm_labeled.cluster_id
                FROM face_labels fl_cluster
                JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl_cluster.face_rectangle_id
                JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
                WHERE fl_cluster.person_id = ?
                  AND (fc.run_id = ? OR fc.archive_scope = 'archive')
            ) person_clusters ON person_clusters.cluster_id = fcm_all.cluster_id
            WHERE fr_cluster.file_path = f.path 
              AND fr_cluster.run_id = ? 
              AND COALESCE(fr_cluster.ignore_flag, 0) = 0
        ) OR EXISTS (
            -- Через прямоугольники без лица
            SELECT 1 FROM person_rectangles pr
            WHERE pr.file_path = f.path AND pr.pipeline_run_id = ? AND pr.person_id = ?
        ) OR EXISTS (
            -- Прямая привязка
            SELECT 1 FROM file_persons fp
            WHERE fp.file_path = f.path AND fp.pipeline_run_id = ? AND fp.person_id = ?
        )
        """
        # Параметры для оптимизированного запроса:
        # 1. face_run_id_i - для EXISTS (face_labels) - fr.run_id
        # 2. person_id_filter - для EXISTS (face_labels) - fl.person_id
        # 3. person_id_filter - для подзапроса кластеров - fl_cluster.person_id
        # 4. face_run_id_i - для подзапроса кластеров - fc.run_id
        # 5. face_run_id_i - для основного запроса кластеров - fr_cluster.run_id
        # 6. int(pipeline_run_id) - для person_rectangles
        # 7. person_id_filter - для person_rectangles
        # 8. int(pipeline_run_id) - для file_persons
        # 9. person_id_filter - для file_persons
        person_filter_params = [face_run_id_i, person_id_filter, person_id_filter, face_run_id_i, face_run_id_i, int(pipeline_run_id), person_id_filter, int(pipeline_run_id), person_id_filter]

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
            SELECT DISTINCT file_path, 
                   MIN(group_path) AS group_path
            FROM file_groups
            WHERE pipeline_run_id = ?
            GROUP BY file_path
        ) fg ON fg.file_path = f.path
        """
        group_select = ", COALESCE(fg.group_path, '') AS group_path"
        # Сортировка: сначала файлы с группами (по названию группы), потом без групп
        group_order = """
          (CASE WHEN COALESCE(fg.group_path, '') = '' THEN 1 ELSE 0 END) ASC,
          COALESCE(fg.group_path, '') ASC,
        """
        group_count_params = [int(pipeline_run_id)]

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        count_start = time.time()
        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path{group_join}
            WHERE {where_sql} AND ({eff_sql}) = ? AND ({sub_where}) AND ({person_filter_sql}) AND ({group_filter_sql}) AND ({media_filter_sql})
            """,
            group_count_params + [int(pipeline_run_id)] + params + [tab_n] + sub_params + person_filter_params + group_filter_params,
        )
        total = int(cur.fetchone()[0] or 0)
        count_time = time.time() - count_start
        msg = f"[API] api_faces_results: COUNT запрос занял {count_time:.3f}с, total={total}"
        logger.info(msg)
        print(msg)

        sort_n = str((sort or "").strip().lower() or "")
        select_start = time.time()
        cur.execute(
            f"""
            SELECT
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
              ON m.pipeline_run_id = ? AND m.path = f.path{group_join}
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
        print(msg)
        
        # Отладочный вывод для проверки фильтрации
        if tab_n == "no_faces":
            print(f"[DEBUG] Загружено строк: {len(rows)} для tab_n={tab_n}, subtab_n={subtab_n}")
            if len(rows) == 0 and (subtab_n == "unsorted" or group_path_filter):
                # Проверяем, почему нет результатов
                print(f"[DEBUG] Проверяем почему нет результатов для subtab_n={subtab_n}, group_path_filter={group_path_filter}")
                # Делаем тестовый запрос
                test_cur = ds.conn.cursor()
                
                if group_path_filter:
                    # Проверяем файлы в группе БЕЗ фильтра по eff_sql
                    test_cur.execute(f"""
                        SELECT COUNT(*) as cnt
                        FROM files f
                        LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.path = f.path
                        WHERE f.faces_run_id = ? AND f.status != 'deleted'
                        AND EXISTS (
                            SELECT 1 FROM file_groups fg
                            WHERE fg.file_path = f.path AND fg.pipeline_run_id = ? 
                            AND fg.group_path = ?
                        )
                    """, [int(pipeline_run_id), face_run_id_i, int(pipeline_run_id), str(group_path_filter)])
                    group_count_all = test_cur.fetchone()[0]
                    print(f"[DEBUG] Файлов в группе '{group_path_filter}' (без фильтра по tab): {group_count_all}")
                    
                    # Проверяем файлы в группе С фильтром по eff_sql = 'no_faces'
                    test_cur.execute(f"""
                        SELECT COUNT(*) as cnt
                        FROM files f
                        LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.path = f.path
                        WHERE f.faces_run_id = ? AND f.status != 'deleted'
                        AND ({eff_sql}) = 'no_faces'
                        AND EXISTS (
                            SELECT 1 FROM file_groups fg
                            WHERE fg.file_path = f.path AND fg.pipeline_run_id = ? 
                            AND fg.group_path = ?
                        )
                    """, [int(pipeline_run_id), face_run_id_i, int(pipeline_run_id), str(group_path_filter)])
                    group_count_no_faces = test_cur.fetchone()[0]
                    print(f"[DEBUG] Файлов в группе '{group_path_filter}' с eff_sql='no_faces': {group_count_no_faces}")
                    
                    # Проверяем конкретные файлы в группе
                    test_cur.execute(f"""
                        SELECT f.path, ({eff_sql}) as eff_tab, f.faces_count, 
                               COALESCE(m.faces_manual_label, '') as manual_label
                        FROM files f
                        LEFT JOIN files_manual_labels m ON m.pipeline_run_id = ? AND m.path = f.path
                        WHERE EXISTS (
                            SELECT 1 FROM file_groups fg
                            WHERE fg.file_path = f.path AND fg.pipeline_run_id = ? 
                            AND fg.group_path = ?
                        )
                        LIMIT 5
                    """, [int(pipeline_run_id), int(pipeline_run_id), str(group_path_filter)])
                    group_files = test_cur.fetchall()
                    print(f"[DEBUG] Файлы в группе '{group_path_filter}':")
                    for row in group_files:
                        print(f"  {row[0]}: eff_tab={row[1]}, faces_count={row[2]}, manual_label={row[3]}")
    finally:
        ds.close()

    gold_expected = gold_expected_tab_by_path(include_drawn_faces=False)
    
    # Группировка в поездки для tab=no_faces и sort=place_date
    if tab_n == "no_faces" and sort_n == "place_date":
        rows = _group_into_trips(rows)
    
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
    print(msg)
    
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

    _agent_dbg(
        hypothesis_id="HTAB_COUNTS",
        location="web_api/routers/faces.py:api_faces_tab_counts",
        message="faces_tab_counts_request",
        data={"pipeline_run_id": int(pipeline_run_id), "root_path": root_path},
    )

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

    eff_sql = """
    CASE
      WHEN COALESCE(m.people_no_face_manual, 0) = 1 THEN 'faces'
      WHEN lower(trim(coalesce(m.faces_manual_label, ''))) = 'faces' THEN 'faces'
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

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        cur.execute(
            f"""
            SELECT ({eff_sql}) AS tab, COUNT(*) AS cnt
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE {where_sql}
            GROUP BY ({eff_sql})
            """,
            [int(pipeline_run_id)] + params,
        )
        rows = cur.fetchall()

        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE {where_sql}
              AND ({eff_sql}) = 'faces'
              AND COALESCE(faces_count, 0) >= 8
            """,
            [int(pipeline_run_id)] + params,
        )
        many_faces_cnt = int(cur.fetchone()[0] or 0)

        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE {where_sql}
              AND ({eff_sql}) = 'faces'
              AND COALESCE(faces_count, 0) < 8
              AND (
                -- Файл не привязан ни к одной персоне
                (
                  NOT EXISTS (
                      SELECT 1 FROM face_labels fl
                      JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
                      WHERE fr.file_path = f.path AND fr.run_id = ? AND fl.person_id IS NOT NULL
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM person_rectangles pr
                      WHERE pr.file_path = f.path AND pr.pipeline_run_id = ? AND pr.person_id IS NOT NULL
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM file_persons fp
                      WHERE fp.file_path = f.path AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
                  )
                )
                -- ИЛИ это люди без лиц
                OR COALESCE(m.people_no_face_manual, 0) = 1
              )
            """,
            [int(pipeline_run_id)] + params + [face_run_id_i, int(pipeline_run_id), int(pipeline_run_id)],
        )
        unsorted_cnt = int(cur.fetchone()[0] or 0)
        
        # Счетчик для "Фото к разбору" в "Нет людей" (файлы без группы, только фото)
        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE {where_sql}
              AND ({eff_sql}) = 'no_faces'
              AND NOT EXISTS (
                  SELECT 1 FROM file_groups fg
                  WHERE fg.file_path = f.path AND fg.pipeline_run_id = ?
              )
              AND (COALESCE(f.media_type, '') = 'image' OR COALESCE(f.media_type, '') = '')
              AND NOT (COALESCE(f.mime_type, '') LIKE 'video/%')
            """,
            [int(pipeline_run_id)] + params + [int(pipeline_run_id)],
        )
        no_faces_unsorted_photos_cnt = int(cur.fetchone()[0] or 0)
        
        # Счетчик для "Видео к разбору" в "Нет людей" (файлы без группы, только видео)
        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE {where_sql}
              AND ({eff_sql}) = 'no_faces'
              AND NOT EXISTS (
                  SELECT 1 FROM file_groups fg
                  WHERE fg.file_path = f.path AND fg.pipeline_run_id = ?
              )
              AND (COALESCE(f.media_type, '') = 'video' OR COALESCE(f.mime_type, '') LIKE 'video/%')
            """,
            [int(pipeline_run_id)] + params + [int(pipeline_run_id)],
        )
        no_faces_unsorted_videos_cnt = int(cur.fetchone()[0] or 0)
    finally:
        ds.close()

    counts = {"faces": 0, "no_faces": 0, "quarantine": 0, "animals": 0, "people_no_face": 0}
    for r in rows:
        t = str(r["tab"] or "")
        if t in counts:
            counts[t] = int(r["cnt"] or 0)
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


@router.get("/api/faces/persons-with-files")
def api_faces_persons_with_files(pipeline_run_id: int) -> dict[str, Any]:
    """
    Возвращает список персон, у которых есть файлы в данном прогоне.
    Используется для отображения подзакладок по персонам в закладке "Люди".
    """
    start_time = time.time()
    msg = f"[API] api_faces_persons_with_files: начало, pipeline_run_id={pipeline_run_id}"
    logger.info(msg)
    print(msg)
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

    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Получаем персон с файлами через все 4 способа привязки
        # 1. Через лица (face_labels) - прямая привязка
        where_parts = ["fr.run_id = ?"]
        params = [face_run_id_i]
        if root_like:
            where_parts.append("fr.file_path LIKE ?")
            params.append(root_like)
        where_sql = " AND ".join(where_parts)
        
        # Персоны через лица (прямая привязка)
        query1_start = time.time()
        cur.execute(
            f"""
            SELECT DISTINCT fl.person_id, p.name AS person_name, COUNT(DISTINCT fr.file_path) AS files_count
            FROM face_labels fl
            JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
            LEFT JOIN persons p ON p.id = fl.person_id
            WHERE {where_sql} AND fl.person_id IS NOT NULL
            GROUP BY fl.person_id, p.name
            """,
            params,
        )
        persons_from_faces = {r["person_id"]: {"id": r["person_id"], "name": r["person_name"], "files_count": int(r["files_count"] or 0)} for r in cur.fetchall()}
        query1_time = time.time() - query1_start
        msg = f"[API] api_faces_persons_with_files: запрос 1 (face_labels) занял {query1_time:.3f}с, персон: {len(persons_from_faces)}"
        logger.info(msg)
        print(msg)
        
        # 1b. Через кластеры (лицо в файле находится в кластере, где есть другие лица с face_labels для персоны)
        where_parts_cluster = ["fr_cluster.run_id = ?"]
        params_cluster = [face_run_id_i]
        if root_like:
            where_parts_cluster.append("fr_cluster.file_path LIKE ?")
            params_cluster.append(root_like)
        where_sql_cluster = " AND ".join(where_parts_cluster)
        
        query2_start = time.time()
        # Оптимизированный запрос: используем подзапрос для кластеров с персонами
        # Это должно быть быстрее, чем множественные JOIN'ы
        cur.execute(
            f"""
            SELECT 
                person_clusters.person_id,
                p.name AS person_name,
                COUNT(DISTINCT fr_cluster.file_path) AS files_count
            FROM (
                -- Находим все кластеры, где есть лица с привязанными персонами
                SELECT DISTINCT
                    fl_cluster.person_id,
                    fcm_labeled.cluster_id
                FROM face_labels fl_cluster
                JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl_cluster.face_rectangle_id
                JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
                WHERE fl_cluster.person_id IS NOT NULL
                  AND (fc.run_id = ? OR fc.archive_scope = 'archive')
            ) person_clusters
            -- Находим все лица в этих кластерах
            JOIN face_cluster_members fcm_all ON fcm_all.cluster_id = person_clusters.cluster_id
            JOIN face_rectangles fr_cluster ON fr_cluster.id = fcm_all.face_rectangle_id
            LEFT JOIN persons p ON p.id = person_clusters.person_id
            WHERE {where_sql_cluster}
              AND COALESCE(fr_cluster.ignore_flag, 0) = 0
              -- Исключаем лица, которые уже учтены в запросе 1 (прямая привязка)
              AND NOT EXISTS (
                  SELECT 1 FROM face_labels fl_direct
                  WHERE fl_direct.face_rectangle_id = fr_cluster.id
                    AND fl_direct.person_id = person_clusters.person_id
              )
            GROUP BY person_clusters.person_id, p.name
            """,
            [face_run_id_i] + params_cluster,
        )
        persons_from_clusters = {r["person_id"]: {"id": r["person_id"], "name": r["person_name"], "files_count": int(r["files_count"] or 0)} for r in cur.fetchall()}
        query2_time = time.time() - query2_start
        msg = f"[API] api_faces_persons_with_files: запрос 2 (clusters) занял {query2_time:.3f}с, персон: {len(persons_from_clusters)}"
        logger.info(msg)
        print(msg)
        
        # 2. Через прямоугольники без лица (person_rectangles)
        where_parts2 = ["pr.pipeline_run_id = ?"]
        params2 = [int(pipeline_run_id)]
        if root_like:
            where_parts2.append("pr.file_path LIKE ?")
            params2.append(root_like)
        where_sql2 = " AND ".join(where_parts2)
        
        query3_start = time.time()
        cur.execute(
            f"""
            SELECT DISTINCT pr.person_id, p.name AS person_name, COUNT(DISTINCT pr.file_path) AS files_count
            FROM person_rectangles pr
            LEFT JOIN persons p ON p.id = pr.person_id
            WHERE {where_sql2} AND pr.person_id IS NOT NULL
            GROUP BY pr.person_id, p.name
            """,
            params2,
        )
        persons_from_rects = {r["person_id"]: {"id": r["person_id"], "name": r["person_name"], "files_count": int(r["files_count"] or 0)} for r in cur.fetchall()}
        query3_time = time.time() - query3_start
        msg = f"[API] api_faces_persons_with_files: запрос 3 (person_rectangles) занял {query3_time:.3f}с, персон: {len(persons_from_rects)}"
        logger.info(msg)
        print(msg)
        
        # 3. Прямые привязки (file_persons)
        where_parts3 = ["fp.pipeline_run_id = ?"]
        params3 = [int(pipeline_run_id)]
        if root_like:
            where_parts3.append("fp.file_path LIKE ?")
            params3.append(root_like)
        where_sql3 = " AND ".join(where_parts3)
        
        query4_start = time.time()
        cur.execute(
            f"""
            SELECT DISTINCT fp.person_id, p.name AS person_name, COUNT(DISTINCT fp.file_path) AS files_count
            FROM file_persons fp
            LEFT JOIN persons p ON p.id = fp.person_id
            WHERE {where_sql3} AND fp.person_id IS NOT NULL
            GROUP BY fp.person_id, p.name
            """,
            params3,
        )
        persons_from_direct = {r["person_id"]: {"id": r["person_id"], "name": r["person_name"], "files_count": int(r["files_count"] or 0)} for r in cur.fetchall()}
        query4_time = time.time() - query4_start
        msg = f"[API] api_faces_persons_with_files: запрос 4 (file_persons) занял {query4_time:.3f}с, персон: {len(persons_from_direct)}"
        logger.info(msg)
        print(msg)
        
        # Объединяем все персоны и суммируем количество файлов
        all_persons: dict[int, dict[str, Any]] = {}
        for pid, pdata in persons_from_faces.items():
            all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"]}
        for pid, pdata in persons_from_clusters.items():
            if pid in all_persons:
                all_persons[pid]["files_count"] += pdata["files_count"]
            else:
                all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"]}
        for pid, pdata in persons_from_rects.items():
            if pid in all_persons:
                all_persons[pid]["files_count"] += pdata["files_count"]
            else:
                all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"]}
        for pid, pdata in persons_from_direct.items():
            if pid in all_persons:
                all_persons[pid]["files_count"] += pdata["files_count"]
            else:
                all_persons[pid] = {"id": pid, "name": pdata["name"], "files_count": pdata["files_count"]}
        
        # Сортируем по имени
        persons_list = sorted(all_persons.values(), key=lambda x: (x["name"] or "").lower())
        
        elapsed = time.time() - start_time
        msg = f"[API] api_faces_persons_with_files: завершено за {elapsed:.3f}с, персон: {len(persons_list)}"
        logger.info(msg)
        print(msg)
        
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "persons": persons_list,
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
        
        print(f"[DEBUG assign-group] Получено: pipeline_run_id={pipeline_run_id} (type: {type(pipeline_run_id)}), path={repr(path)} (type: {type(path)}), group_path={repr(group_path)} (type: {type(group_path)})")
        print(f"[DEBUG assign-group] Нормализовано: path={repr(str(path))}, group_path={repr(normalized_group_path)}")
        
        try:
            fs.insert_file_group(
                pipeline_run_id=int(pipeline_run_id),
                file_path=str(path),
                group_path=normalized_group_path,
            )
            print(f"[DEBUG assign-group] INSERT выполнен успешно")
        except Exception as e:
            print(f"[DEBUG assign-group] ОШИБКА при INSERT: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Проверяем, что группа действительно сохранилась
        groups = fs.list_file_groups(
            pipeline_run_id=int(pipeline_run_id),
            file_path=str(path),
        )
        print(f"[DEBUG assign-group] После сохранения найдено групп для файла: {len(groups)}")
        for g in groups:
            print(f"[DEBUG assign-group]   Группа: {repr(g.get('group_path'))}")
        
        saved = any(g.get("group_path") == normalized_group_path for g in groups)
        if not saved:
            print(f"[DEBUG assign-group] ОШИБКА: Группа не сохранилась! Искали: {repr(normalized_group_path)}, нашли: {[repr(g.get('group_path')) for g in groups]}")
            raise HTTPException(status_code=500, detail=f"Failed to save group assignment. Expected: '{normalized_group_path}', found: {[g.get('group_path') for g in groups]}")
        else:
            print(f"[DEBUG assign-group] ✅ Группа успешно сохранена и проверена")
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
def api_faces_rectangles(pipeline_run_id: int, path: str) -> dict[str, Any]:
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
        rects = fs.list_rectangles(run_id=face_run_id_i, file_path=str(path))
        
        # Добавляем информацию о персонах для каждого прямоугольника
        ds = DedupStore()
        try:
            conn = ds.conn
            cur = conn.cursor()
            
            # Получаем информацию о персонах для каждого прямоугольника
            for rect in rects:
                rect_id = rect.get("id")
                if not rect_id:
                    continue
                
                # Ищем персону через face_labels
                cur.execute("""
                    SELECT fl.person_id, p.name AS person_name
                    FROM face_labels fl
                    LEFT JOIN persons p ON p.id = fl.person_id
                    WHERE fl.face_rectangle_id = ? AND fl.person_id IS NOT NULL
                    LIMIT 1
                """, (rect_id,))
                person_row = cur.fetchone()
                if person_row:
                    rect["person_id"] = person_row["person_id"]
                    rect["person_name"] = person_row["person_name"]
                else:
                    rect["person_id"] = None
                    rect["person_name"] = None
        finally:
            ds.close()
    finally:
        fs.close()
    
    return {"ok": True, "pipeline_run_id": int(pipeline_run_id), "run_id": face_run_id_i, "path": path, "rectangles": rects}


@router.post("/api/faces/assign-face-person")
def api_faces_assign_face_person(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Назначает персону лицу (face_rectangle_id).
    
    Параметры:
    - pipeline_run_id: int (обязательно)
    - face_rectangle_id: int (обязательно)
    - person_id: int (обязательно)
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    face_rectangle_id = payload.get("face_rectangle_id")
    person_id = payload.get("person_id")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required and must be int")
    if not isinstance(face_rectangle_id, int):
        raise HTTPException(status_code=400, detail="face_rectangle_id is required and must be int")
    if not isinstance(person_id, int):
        raise HTTPException(status_code=400, detail="person_id is required and must be int")
    
    # Проверяем, что pipeline_run_id существует
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    finally:
        ps.close()
    
    # Проверяем, что лицо существует
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        cur.execute("SELECT id, file_path FROM face_rectangles WHERE id = ?", (int(face_rectangle_id),))
        face_row = cur.fetchone()
        if not face_row:
            raise HTTPException(status_code=404, detail="face_rectangle_id not found")
        
        # Проверяем, что персона существует
        cur.execute("SELECT id, name FROM persons WHERE id = ?", (int(person_id),))
        person_row = cur.fetchone()
        if not person_row:
            raise HTTPException(status_code=404, detail="person_id not found")
        
        # Создаем или обновляем face_labels
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        
        cur.execute("""
            INSERT OR REPLACE INTO face_labels (face_rectangle_id, person_id, source, created_at)
            VALUES (?, ?, 'manual', ?)
        """, (int(face_rectangle_id), int(person_id), now))
        conn.commit()
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "face_rectangle_id": int(face_rectangle_id),
        "person_id": int(person_id),
        "person_name": person_row["name"],
    }


@router.get("/api/faces/file-persons")
def api_faces_file_persons(pipeline_run_id: int, path: str) -> dict[str, Any]:
    """
    Возвращает список персон, привязанных к файлу (через любые способы), с информацией о лицах.
    """
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
    ds = DedupStore()
    try:
        # Используем подключение из FaceStore для работы с face_labels и face_rectangles
        fs_cur = fs.conn.cursor()
        # Используем подключение из DedupStore для работы с persons, person_rectangles и file_persons
        ds_cur = ds.conn.cursor()
        persons_set = {}
        
        # 1. Через face_labels (лица) - получаем информацию о лицах (прямая привязка)
        fs_cur.execute("""
            SELECT 
                fl.person_id, 
                p.name AS person_name,
                fr.id AS face_rectangle_id,
                fr.bbox_x AS x,
                fr.bbox_y AS y,
                fr.bbox_w AS w,
                fr.bbox_h AS h
            FROM face_labels fl
            JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
            LEFT JOIN persons p ON p.id = fl.person_id
            WHERE fr.file_path = ? AND fr.run_id = ? AND fl.person_id IS NOT NULL
            ORDER BY fr.id
        """, (str(path), face_run_id_i))
        for row in fs_cur.fetchall():
            pid = row["person_id"]
            if pid not in persons_set:
                persons_set[pid] = {
                    "id": pid, 
                    "name": row["person_name"],
                    "faces": []
                }
            persons_set[pid]["faces"].append({
                "id": row["face_rectangle_id"],
                "x": row["x"],
                "y": row["y"],
                "w": row["w"],
                "h": row["h"]
            })
        
        # 1b. Через кластеры (лицо в файле находится в кластере, где есть другие лица с face_labels для персоны)
        # Логика из analyze_person_distribution.py:
        # - Находим кластеры, где есть лица с face_labels для персоны
        # - Находим ВСЕ лица в этих кластерах (включая новые, без face_labels)
        # - Если лицо из файла находится в таком кластере, оно считается привязанным к персоне
        # Важно: показываем персону только если хотя бы одно лицо в файле находится в кластере с этой персоной
        fs_cur.execute("""
            SELECT DISTINCT
                fl_cluster.person_id,
                p.name AS person_name,
                fr_file.id AS face_rectangle_id,
                fr_file.bbox_x AS x,
                fr_file.bbox_y AS y,
                fr_file.bbox_w AS w,
                fr_file.bbox_h AS h
            FROM persons p
            JOIN face_labels fl_cluster ON fl_cluster.person_id = p.id
            -- Находим кластеры, где есть лица с face_labels для этой персоны
            JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl_cluster.face_rectangle_id
            JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
            -- Находим ВСЕ лица в этих кластерах (включая лица из нашего файла)
            JOIN face_cluster_members fcm_all ON fcm_all.cluster_id = fc.id
            JOIN face_rectangles fr_file ON fr_file.id = fcm_all.face_rectangle_id
            WHERE fr_file.file_path = ?
              AND fr_file.run_id = ?
              AND COALESCE(fr_file.ignore_flag, 0) = 0
              AND (fc.run_id = ? OR fc.archive_scope = 'archive')
              -- Исключаем лица, которые уже есть в прямой привязке (face_labels) для этой персоны
              AND NOT EXISTS (
                  SELECT 1 FROM face_labels fl_direct
                  WHERE fl_direct.face_rectangle_id = fr_file.id
                    AND fl_direct.person_id = fl_cluster.person_id
              )
              -- Исключаем случаи, когда лицо из файла само имеет face_label для другой персоны
              -- (т.е. если лицо уже привязано к другой персоне напрямую, не показываем его через кластеры)
              AND NOT EXISTS (
                  SELECT 1 FROM face_labels fl_other
                  WHERE fl_other.face_rectangle_id = fr_file.id
                    AND fl_other.person_id IS NOT NULL
              )
            ORDER BY fl_cluster.person_id, fr_file.id
        """, (str(path), face_run_id_i, face_run_id_i))
        for row in fs_cur.fetchall():
            pid = row["person_id"]
            if pid not in persons_set:
                persons_set[pid] = {
                    "id": pid, 
                    "name": row["person_name"],
                    "faces": []
                }
            # Проверяем, что это лицо еще не добавлено (может быть дубликат из-за нескольких кластеров)
            face_id = row["face_rectangle_id"]
            if not any(f["id"] == face_id for f in persons_set[pid]["faces"]):
                persons_set[pid]["faces"].append({
                    "id": face_id,
                    "x": row["x"],
                    "y": row["y"],
                    "w": row["w"],
                    "h": row["h"]
                })
        
        # 2. Через person_rectangles
        ds_cur.execute("""
            SELECT DISTINCT pr.person_id, p.name AS person_name
            FROM person_rectangles pr
            LEFT JOIN persons p ON p.id = pr.person_id
            WHERE pr.file_path = ? AND pr.pipeline_run_id = ? AND pr.person_id IS NOT NULL
        """, (str(path), int(pipeline_run_id)))
        for row in ds_cur.fetchall():
            pid = row["person_id"]
            if pid not in persons_set:
                persons_set[pid] = {"id": pid, "name": row["person_name"], "faces": []}
        
        # 3. Через file_persons (прямая привязка)
        ds_cur.execute("""
            SELECT DISTINCT fp.person_id, p.name AS person_name
            FROM file_persons fp
            LEFT JOIN persons p ON p.id = fp.person_id
            WHERE fp.file_path = ? AND fp.pipeline_run_id = ? AND fp.person_id IS NOT NULL
        """, (str(path), int(pipeline_run_id)))
        for row in ds_cur.fetchall():
            pid = row["person_id"]
            if pid not in persons_set:
                persons_set[pid] = {"id": pid, "name": row["person_name"], "faces": []}
        
        persons_list = list(persons_set.values())
    finally:
        fs.close()
        ds.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "path": path,
        "persons": persons_list,
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
            fs = FaceStore()
            try:
                fs.replace_manual_rectangles(run_id=face_run_id, file_path=path, rects=[])
            finally:
                fs.close()

    return {"ok": True}


@router.post("/api/faces/manual-rectangles")
def api_faces_manual_rectangles(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    pipeline_run_id = payload.get("pipeline_run_id")
    path = payload.get("path")
    rects = payload.get("rects")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    if not isinstance(path, str) or not path.startswith("local:"):
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
        fs.replace_manual_rectangles(run_id=face_run_id_i, file_path=str(path), rects=rects)
    finally:
        fs.close()

    ds = DedupStore()
    try:
        ds.set_run_faces_manual_label(pipeline_run_id=int(pipeline_run_id), path=str(path), label="faces")
    finally:
        ds.close()

    return {"ok": True}


def _normalize_yadisk_path(path: str) -> str:
    """Нормализует путь YaDisk: убирает лишние пробелы, приводит к единому формату."""
    p = str(path or "").strip()
    if not p:
        return ""
    # Убираем двойные слеши, но сохраняем disk:/ в начале
    if p.startswith("disk:/"):
        p = "disk:/" + p[6:].lstrip("/")
    return p.replace("\\", "/")


def _determine_target_folder(
    *,
    path: str,
    effective_tab: str,
    root_path: str,
    preclean_kind: str | None = None,
) -> str | None:
    """
    Определяет целевую папку для файла на основе результатов первых 3 шагов.

    Шаг 1 (предочистка):
    - non_media -> _non_media
    - broken_media -> _broken_media

    Шаг 3 (сортировка по лицам):
    - faces -> _faces
    - quarantine -> _quarantine
    - animals -> _animals
    - people_no_face -> _people_no_face
    - no_faces -> _no_faces

    Возвращает путь к целевой папке (например, "disk:/Фото/_faces" или "local:C:\\tmp\\Photo\\_faces") или None.
    """
    # Определяем базовый путь (root_path)
    if root_path.startswith("disk:"):
        base_path = root_path.rstrip("/")
    elif root_path.startswith("local:"):
        base_path = root_path[6:]  # убираем "local:"
    else:
        base_path = root_path

    # Шаг 1: предочистка (non_media, broken_media)
    if preclean_kind:
        if preclean_kind == "non_media":
            folder_name = "_non_media"
        elif preclean_kind == "broken_media":
            folder_name = "_broken_media"
        else:
            return None
    # Шаг 3: сортировка по лицам
    elif effective_tab == "faces":
        folder_name = "_faces"
    elif effective_tab == "quarantine":
        folder_name = "_quarantine"
    elif effective_tab == "animals":
        folder_name = "_animals"
    elif effective_tab == "people_no_face":
        folder_name = "_people_no_face"
    elif effective_tab == "no_faces":
        folder_name = "_no_faces"
    else:
        return None  # Неизвестная категория

    # Формируем полный путь
    if root_path.startswith("disk:"):
        result = f"{base_path}/{folder_name}"
    else:
        # Локальный путь
        result = f"local:{os.path.join(base_path, folder_name)}"
    return result


@router.post("/api/faces/sort-into-folders")
def api_faces_sort_into_folders(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Сортирует файлы по папкам на основе правил приоритета.

    Параметры:
    - pipeline_run_id: int (обязательно)
    - dry_run: bool (опционально, по умолчанию False)

    Возвращает статистику перемещённых файлов.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    dry_run = payload.get("dry_run", False)

    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")

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

    # Получаем все файлы для данного прогона
    ds = DedupStore()
    fs = FaceStore()
    ps = PipelineStore()
    try:
        # Получаем данные из preclean_moves для шага 1
        cur_preclean = ps.conn.cursor()
        cur_preclean.execute(
            """
            SELECT src_path, kind
            FROM preclean_moves
            WHERE pipeline_run_id = ?
            """,
            (int(pipeline_run_id),),
        )
        preclean_map: dict[str, str] = {}  # path -> kind (non_media или broken_media)
        for row in cur_preclean.fetchall():
            src_path = str(row[0] or "")
            kind = str(row[1] or "")
            if src_path and kind:
                preclean_map[src_path] = kind

        # Получаем список файлов с их категориями и метками
        cur = ds.conn.cursor()
        cur.execute(
            """
            SELECT
              f.path, f.name, f.parent_path,
              COALESCE(m.faces_manual_label, '') AS faces_manual_label,
              COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
              COALESCE(f.faces_auto_quarantine, 0) AS faces_auto_quarantine,
              COALESCE(f.faces_quarantine_reason, '') AS faces_quarantine_reason,
              COALESCE(f.animals_auto, 0) AS animals_auto,
              COALESCE(f.animals_kind, '') AS animals_kind,
              COALESCE(m.animals_manual, 0) AS animals_manual,
              COALESCE(m.animals_manual_kind, '') AS animals_manual_kind,
              COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual,
              COALESCE(m.people_no_face_person, '') AS people_no_face_person,
              COALESCE(f.faces_count, 0) AS faces_count
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE f.faces_run_id = ? AND f.status != 'deleted'
            """,
            (int(pipeline_run_id), face_run_id_i),
        )
        rows = [dict(r) for r in cur.fetchall()]

        # Формируем список файлов с их категориями
        files_data: list[dict[str, Any]] = []
        for r in rows:
            path = str(r.get("path") or "")
            if not path:
                continue

            # Проверяем, есть ли файл в preclean_moves (шаг 1)
            preclean_kind = preclean_map.get(path)

            # Определяем effective_tab (шаг 3)
            effective_tab = "no_faces"
            if preclean_kind:
                # Файл уже обработан на шаге 1, пропускаем шаг 3
                effective_tab = None
            elif r.get("people_no_face_manual"):
                effective_tab = "people_no_face"
            elif (r.get("faces_manual_label") or "").lower().strip() == "faces":
                effective_tab = "faces"
            elif (r.get("faces_manual_label") or "").lower().strip() == "no_faces":
                effective_tab = "no_faces"
            elif r.get("quarantine_manual") and r.get("faces_count", 0) > 0:
                effective_tab = "quarantine"
            elif r.get("animals_manual") or r.get("animals_auto"):
                effective_tab = "animals"
            elif r.get("faces_auto_quarantine") and r.get("faces_count", 0) > 0:
                effective_tab = "quarantine"
            elif r.get("faces_count", 0) > 0:
                effective_tab = "faces"

            files_data.append(
                {
                    "path": path,
                    "name": str(r.get("name") or ""),
                    "parent_path": str(r.get("parent_path") or ""),
                    "effective_tab": effective_tab,
                    "preclean_kind": preclean_kind,
                }
            )

    finally:
        ds.close()
        fs.close()
        ps.close()

    # Определяем целевую папку для каждого файла и перемещаем
    moved_count = 0
    errors: list[dict[str, str]] = []
    disk = None

    for file_data in files_data:
        path = file_data["path"]
        target_folder = _determine_target_folder(
            path=path,
            effective_tab=file_data["effective_tab"],
            root_path=root_path,
            preclean_kind=file_data.get("preclean_kind"),
        )

        if not target_folder:
            continue  # Пропускаем файлы, для которых не определена целевая папка

        # Формируем путь назначения
        file_name = file_data["name"]
        if not file_name:
            file_name = os.path.basename(path)

        # Если файл уже в целевой папке, пропускаем
        if path.startswith(target_folder + "/"):
            continue

        dst_path = target_folder + "/" + file_name

        # Перемещаем файл
        if path.startswith("disk:"):
            # YaDisk
            if disk is None:
                disk = get_disk()
            try:
                if not dry_run:
                    src_norm = _normalize_yadisk_path(path)
                    dst_norm = _normalize_yadisk_path(dst_path)
                    disk.move(src_norm, dst_norm, overwrite=False)
                    # Обновляем путь в БД
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
                    finally:
                        ds.close()
                    # Обновляем пути в gold файлах
                    _update_gold_file_paths(old_path=path, new_path=dst_path)
                moved_count += 1
            except Exception as e:  # noqa: BLE001
                errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
        elif path.startswith("local:"):
            # Локальный файл
            local_path = path[6:]  # убираем "local:"
            if not os.path.exists(local_path):
                errors.append({"path": path, "error": "File not found"})
                continue
            # target_folder может быть "local:C:\tmp\Photo\_faces" или "disk:/Фото/_faces"
            if target_folder.startswith("local:"):
                # target_folder уже локальный путь к папке, добавляем имя файла
                dst_local = os.path.join(target_folder[6:], file_name)  # убираем "local:" и добавляем file_name
            elif target_folder.startswith("disk:"):
                # target_folder это путь YaDisk, нужно преобразовать в локальный
                # target_folder = "disk:/Фото/_faces" -> "_faces"
                folder_name = os.path.basename(target_folder)
                # Используем root_path для формирования локального пути
                root_path_clean = root_path
                if root_path.startswith("local:"):
                    root_path_clean = root_path[6:]  # убираем "local:"
                if root_path_clean and not root_path_clean.startswith("disk:"):
                    # root_path локальный, например "C:\tmp\Photo"
                    dst_local = os.path.join(root_path_clean, folder_name, file_name)
                else:
                    # Fallback: используем имя папки из target_folder
                    dst_local = os.path.join(os.path.dirname(local_path), folder_name, file_name)
            else:
                # target_folder уже локальный путь без префикса
                dst_local = os.path.join(target_folder, file_name)
            dst_dir = os.path.dirname(dst_local)
            # Проверяем, существует ли путь и что это (файл или директория)
            if os.path.exists(dst_dir):
                if not os.path.isdir(dst_dir):
                    # Путь существует, но это файл, а не директория
                    errors.append({"path": path, "error": f"Path {dst_dir} exists but is a file, not a directory"})
                    continue
                # Директория уже существует, продолжаем
            else:
                # Директории нет, создаём
                try:
                    os.makedirs(dst_dir, exist_ok=True)
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": path, "error": f"Cannot create directory {dst_dir}: {type(e).__name__}: {e}"})
                    continue
            if not dry_run:
                try:
                    os.rename(local_path, dst_local)
                    # Обновляем путь в БД
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
                    finally:
                        ds.close()
                    fs = FaceStore()
                    try:
                        fs.update_file_path(old_file_path=path, new_file_path=new_db_path)
                    finally:
                        fs.close()
                    # Обновляем пути в gold файлах
                    _update_gold_file_paths(old_path=path, new_path=new_db_path)
                    # Обновляем пути в gold файлах
                    _update_gold_file_paths(old_path=path, new_path=new_db_path)
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
                    continue
            moved_count += 1

    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "dry_run": bool(dry_run),
        "moved_count": moved_count,
        "errors": errors,
    }


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
              ON m.pipeline_run_id = ? AND m.path = f.path
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

    # Обновляем пути в face_rectangles
    if delete_path:
        fs = FaceStore()
        try:
            face_run_id = pr.get("face_run_id")
            if face_run_id:
                # Обновляем пути в face_rectangles
                cur = fs.conn.cursor()
                cur.execute(
                    "UPDATE face_rectangles SET file_path = ? WHERE run_id = ? AND file_path = ?",
                    (delete_path, int(face_run_id), path),
                )
                fs.conn.commit()
        finally:
            fs.close()

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

    # Обновляем пути в face_rectangles
    fs = FaceStore()
    try:
        face_run_id = pr.get("face_run_id")
        if face_run_id:
            cur = fs.conn.cursor()
            cur.execute(
                "UPDATE face_rectangles SET file_path = ? WHERE run_id = ? AND file_path = ?",
                (original_path, int(face_run_id), delete_path),
            )
            fs.conn.commit()
    finally:
        fs.close()

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
    - face_rectangle_id: int (опционально, используется для получения face_run_id если face_run_id не указан)
    - path: str (обязательно, disk:/... или local:...)
    
    Возвращает количество обновлённых записей.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    face_run_id = payload.get("face_run_id")
    face_rectangle_id = payload.get("face_rectangle_id")
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
    
    if face_rectangle_id is not None and not isinstance(face_rectangle_id, int):
        try:
            face_rectangle_id = int(face_rectangle_id)
        except (ValueError, TypeError):
            face_rectangle_id = None
    
    # pipeline_run_id и face_run_id не обязательны - скрипт recalc_face_bbox.py работает напрямую с БД по path
    # Если они указаны, пытаемся найти pipeline_run_id для логирования (опционально)
    if not isinstance(pipeline_run_id, int):
        # Если face_run_id не указан, но есть face_rectangle_id, получаем run_id из face_rectangle
        if not isinstance(face_run_id, int) and isinstance(face_rectangle_id, int):
            from common.db import FaceStore
            fs = FaceStore()
            try:
                cur = fs.conn.cursor()
                cur.execute(
                    """
                    SELECT run_id
                    FROM face_rectangles
                    WHERE id = ?
                    """,
                    (int(face_rectangle_id),),
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