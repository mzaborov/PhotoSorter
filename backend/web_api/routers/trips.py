from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from common.db import (
    DedupStore,
    get_connection,
    get_file_taken_at_date,
    get_trip,
    get_trips_for_file,
    _get_file_id_from_path,
    list_trip_files_included,
    list_trip_files_excluded_in_range,
    list_trip_suggestions,
    list_trips,
    list_trips_suggest_by_date,
    trip_create,
    trip_dates_from_files,
    trip_file_attach,
    trip_file_exclude,
    trip_update,
)
from common.yadisk_client import get_disk

router = APIRouter()

APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


# Расширения, для которых YaDisk обычно не отдаёт превью (видео) — не запрашиваем preview-image (404).
_VIDEO_EXTENSIONS = frozenset((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"))


def _normalize_yadisk_path(path: str) -> str:
    p = path or ""
    if p.startswith("disk:"):
        p = p[len("disk:") :]
    if not p.startswith("/"):
        p = "/" + p
    return p


def _preview_url(path: str, pipeline_run_id: int | None = None, media_type: str | None = None) -> str:
    """URL для превью по path (disk: или local:). Для local: pipeline_run_id даёт доступ к файлам прогона.
    Для disk: видео не запрашиваем preview-image (YaDisk часто не даёт превью → 404)."""
    p = (path or "").strip()
    if not p:
        return ""
    # Нормализация: если путь локальный без префикса (C:\, D:\, /), добавляем local:
    if not p.startswith("disk:") and not p.startswith("local:") and p:
        if (len(p) >= 2 and p[1] == ":") or p.startswith("/"):
            p = "local:" + p
    if p.startswith("disk:"):
        mt = (media_type or "").lower()
        low = p.lower()
        is_video = mt == "video" or any(low.endswith(ext) for ext in _VIDEO_EXTENSIONS)
        if is_video:
            return "/api/yadisk/video-frame?path=" + urllib.parse.quote(p, safe="")
        return "/api/yadisk/preview-image?size=M&path=" + urllib.parse.quote(p, safe="")
    if p.startswith("local:"):
        mt = (media_type or "").lower()
        is_video = mt == "video" or any(p.lower().endswith(ext) for ext in _VIDEO_EXTENSIONS)
        if is_video:
            q = "path=" + urllib.parse.quote(p, safe="") + "&frame_idx=1"
            if pipeline_run_id is not None:
                q = "pipeline_run_id=" + str(int(pipeline_run_id)) + "&" + q
            return "/api/faces/video-frame?" + q
        url = "/api/local/preview?path=" + urllib.parse.quote(p, safe="")
        if pipeline_run_id is not None:
            url += "&pipeline_run_id=" + str(int(pipeline_run_id))
        return url
    return ""


@router.get("/trips", response_class=HTMLResponse)
def trips_list_page(request: Request):
    """Страница списка поездок."""
    trips_data = list_trips()
    cover_ids = [t["cover_file_id"] for t in trips_data if t.get("cover_file_id")]
    cover_meta: dict[int, dict] = {}
    if cover_ids:
        conn = get_connection()
        try:
            cur = conn.cursor()
            placeholders = ",".join("?" * len(cover_ids))
            cur.execute(
                f"""SELECT f.id, f.path, f.media_type,
                    (SELECT pipeline_run_id FROM file_groups WHERE file_id = f.id LIMIT 1) AS pipeline_run_id
                    FROM files f WHERE f.id IN ({placeholders}) AND (f.status IS NULL OR f.status != 'deleted')""",
                cover_ids,
            )
            cover_meta = {row["id"]: dict(row) for row in cur.fetchall()}
        finally:
            conn.close()
    for t in trips_data:
        cid = t.get("cover_file_id")
        meta = cover_meta.get(cid or 0) if cid else None
        path = (meta.get("path") or "") if meta else ""
        t["cover_preview_url"] = _preview_url(
            path,
            meta.get("pipeline_run_id") if meta else None,
            meta.get("media_type") if meta else None,
        ) or None
        if not (t.get("start_date") or "").strip():
            ds, de = trip_dates_from_files(int(t["id"]))
            t["derived_start_date"] = ds
            t["derived_end_date"] = de
    return templates.TemplateResponse(
        "trips_list.html",
        {"request": request, "trips": trips_data},
    )


def _group_by_date(items: list[dict], date_key: str = "taken_at") -> dict[str, list[dict]]:
    """Группирует список по дате (YYYY-MM-DD)."""
    out: dict[str, list[dict]] = {}
    for it in items:
        t = it.get(date_key)
        if not t or not str(t).strip():
            d = ""
        else:
            d = str(t).strip()[:10]
            if len(d) != 10 or d[4] != "-" or d[7] != "-":
                d = ""
        out.setdefault(d or "_no_date", []).append(it)
    return out


@router.get("/trips/{trip_id:int}", response_class=HTMLResponse)
def trip_detail_page(request: Request, trip_id: int):
    """Страница карточки поездки."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    files = list_trip_files_included(trip_id)
    suggestions = list_trip_suggestions(trip_id)
    excluded_in_range = list_trip_files_excluded_in_range(trip_id)
    for f in files:
        f["preview_url"] = _preview_url(
            f.get("path") or "",
            f.get("pipeline_run_id"),
            f.get("media_type"),
        ) or None
    for s in suggestions:
        s["preview_url"] = _preview_url(
            s.get("path") or "",
            s.get("pipeline_run_id"),
            s.get("media_type"),
        ) or None
    for e in excluded_in_range:
        e["preview_url"] = _preview_url(
            e.get("path") or "",
            e.get("pipeline_run_id"),
            e.get("media_type"),
        ) or None
    cover_preview_url = None
    if trip.get("cover_file_id"):
        for f in files:
            if f.get("file_id") == trip["cover_file_id"]:
                cover_preview_url = f.get("preview_url")
                break
    derived_start = None
    derived_end = None
    if not (trip.get("start_date") or "").strip():
        derived_start, derived_end = trip_dates_from_files(trip_id)
    files_by_date = _group_by_date(files)
    suggestions_by_date = _group_by_date(suggestions)
    excluded_by_date = _group_by_date(excluded_in_range)
    return templates.TemplateResponse(
        "trip_detail.html",
        {
            "request": request,
            "trip": trip,
            "files": files,
            "files_by_date": files_by_date,
            "suggestions": suggestions,
            "suggestions_by_date": suggestions_by_date,
            "excluded_in_range": excluded_in_range,
            "excluded_by_date": excluded_by_date,
            "cover_preview_url": cover_preview_url,
            "derived_start_date": derived_start,
            "derived_end_date": derived_end,
        },
    )


@router.get("/api/trips/list")
def api_trips_list() -> list[dict[str, Any]]:
    """API: список поездок."""
    return list_trips()


@router.get("/api/trips/suggest-by-date")
def api_trips_suggest_by_date(date: str, limit: int = 15) -> list[dict[str, Any]]:
    """API: поездки, близкие по дате к date (YYYY-MM-DD). Для выбора группы на faces/no_faces."""
    return list_trips_suggest_by_date(date_str=date, limit=max(1, min(limit, 50)))


@router.get("/api/trips/for-file")
def api_trips_for_file(file_id: int | None = None, path: str | None = None) -> dict[str, Any]:
    """API: поездки, к которым привязан файл (trip_files, status=included). Передать file_id или path (local:/disk:)."""
    fid = file_id
    if fid is None and path:
        conn = get_connection()
        try:
            fid = _get_file_id_from_path(conn, path.strip())
        finally:
            conn.close()
    if fid is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    trips = get_trips_for_file(fid)
    file_date = get_file_taken_at_date(fid)
    taken_at = (file_date or "").strip()[:10] if file_date else None
    return {"trips": trips, "file_taken_at": taken_at}


class CreateTripBody(BaseModel):
    name: str
    file_id: int | None = None
    path: str | None = None


@router.post("/api/trips")
def api_trip_create(body: CreateTripBody) -> dict[str, Any]:
    """API: создать поездку. Опционально file_id/path — задать start_date/end_date по дате файла (чтобы поездка не уходила в конец списка)."""
    name = (body.name or "").strip() or "Поездка"
    trip_id = trip_create(name=name, yd_folder_path=None)
    fid = body.file_id
    if fid is None and body.path:
        conn = get_connection()
        try:
            fid = _get_file_id_from_path(conn, body.path.strip())
        finally:
            conn.close()
    if fid is not None:
        file_date = get_file_taken_at_date(fid)
        if file_date:
            date_str = (file_date or "").strip()[:10]
            if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                trip_update(trip_id, start_date=date_str, end_date=date_str)
    trip = get_trip(trip_id)
    return dict(trip) if trip else {"id": trip_id, "name": name, "yd_folder_path": None}


@router.get("/api/trips/{trip_id:int}")
def api_trip_detail(trip_id: int) -> dict[str, Any]:
    """API: одна поездка."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


@router.get("/api/trips/{trip_id:int}/files")
def api_trip_files(trip_id: int) -> list[dict[str, Any]]:
    """API: файлы поездки (included)."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    files = list_trip_files_included(trip_id)
    for f in files:
        f["preview_url"] = _preview_url(f.get("path") or "", f.get("pipeline_run_id"), f.get("media_type")) or None
    return files


@router.get("/api/trips/{trip_id:int}/suggestions")
def api_trip_suggestions(trip_id: int) -> list[dict[str, Any]]:
    """API: предложения по датам (±1 день)."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    suggestions = list_trip_suggestions(trip_id)
    for s in suggestions:
        s["preview_url"] = _preview_url(s.get("path") or "", s.get("pipeline_run_id"), s.get("media_type")) or None
    return suggestions


class TripFileBody(BaseModel):
    file_id: int | None = None
    path: str | None = None


class TripUpdateBody(BaseModel):
    name: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    cover_file_id: int | None = None


@router.patch("/api/trips/{trip_id:int}")
def api_trip_update(trip_id: int, body: TripUpdateBody) -> dict[str, Any]:
    """API: обновить поездку (название, даты, обложка)."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    if body.cover_file_id is not None and body.cover_file_id != 0:
        files = list_trip_files_included(trip_id)
        file_ids = {f["file_id"] for f in files}
        if body.cover_file_id not in file_ids:
            raise HTTPException(
                status_code=400,
                detail="cover_file_id must be one of the trip's files",
            )
    trip_update(
        trip_id,
        name=body.name,
        start_date=body.start_date,
        end_date=body.end_date,
        cover_file_id=body.cover_file_id if body.cover_file_id is not None else None,
    )
    return {"ok": True, "trip_id": trip_id}


def _expand_trip_dates_if_needed(trip_id: int, file_date: str) -> None:
    """Если file_date вне дат поездки — расширяет start_date/end_date."""
    trip = get_trip(trip_id)
    if not trip:
        return
    start_s = (trip.get("start_date") or "").strip()
    end_s = (trip.get("end_date") or "").strip()
    if not start_s:
        trip_update(trip_id, start_date=file_date, end_date=file_date)
        return
    new_start = start_s
    new_end = end_s or start_s
    if file_date < new_start:
        new_start = file_date
    if file_date > new_end:
        new_end = file_date
    if new_start != start_s or new_end != end_s:
        trip_update(trip_id, start_date=new_start, end_date=new_end)


@router.post("/api/trips/{trip_id:int}/attach")
def api_trip_attach(trip_id: int, body: TripFileBody) -> dict[str, Any]:
    """API: привязать файл к поездке (file_id или path). Даты поездки расширяются, если фото вне текущего диапазона."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    fid = body.file_id
    if fid is None and body.path:
        conn = get_connection()
        try:
            fid = _get_file_id_from_path(conn, body.path.strip())
        finally:
            conn.close()
        if fid is None:
            raise HTTPException(status_code=404, detail="File not found by path (path must exist in files table)")
    if fid is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    trip_file_attach(trip_id, fid)
    file_date = get_file_taken_at_date(fid)
    if file_date:
        _expand_trip_dates_if_needed(trip_id, file_date)
    return {"ok": True, "trip_id": trip_id, "file_id": fid}


class TripDateBody(BaseModel):
    date: str  # YYYY-MM-DD


def _match_date(taken_at: str | None, date_str: str) -> bool:
    """Совпадение даты: date_str может быть '' или '_no_date' для файлов без даты."""
    if date_str in ("", "_no_date"):
        return not (taken_at or "").strip()
    d = (str(taken_at or "").strip())[:10]
    return d == date_str


@router.post("/api/trips/{trip_id:int}/attach-date")
def api_trip_attach_date(trip_id: int, body: TripDateBody) -> dict[str, Any]:
    """API: привязать к поездке все предложения за указанную дату."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    suggestions = list_trip_suggestions(trip_id)
    date_str = (body.date or "").strip().replace("_no_date", "")[:10]
    attached = 0
    for s in suggestions:
        if _match_date(s.get("taken_at"), date_str if date_str else "_no_date"):
            trip_file_attach(trip_id, s["file_id"])
            fd = get_file_taken_at_date(s["file_id"])
            if fd:
                _expand_trip_dates_if_needed(trip_id, fd)
            attached += 1
    return {"ok": True, "trip_id": trip_id, "date": date_str or "_no_date", "attached": attached}


@router.post("/api/trips/{trip_id:int}/exclude-date")
def api_trip_exclude_date(trip_id: int, body: TripDateBody) -> dict[str, Any]:
    """API: исключить из поездки все предложения за указанную дату."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    suggestions = list_trip_suggestions(trip_id)
    date_str = (body.date or "").strip().replace("_no_date", "")[:10]
    excluded = 0
    for s in suggestions:
        if _match_date(s.get("taken_at"), date_str if date_str else "_no_date"):
            trip_file_exclude(trip_id, s["file_id"])
            excluded += 1
    return {"ok": True, "trip_id": trip_id, "date": date_str or "_no_date", "excluded": excluded}


@router.post("/api/trips/{trip_id:int}/exclude")
def api_trip_exclude(trip_id: int, body: TripFileBody) -> dict[str, Any]:
    """API: исключить файл из поездки (не относится)."""
    trip = get_trip(trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    trip_file_exclude(trip_id, body.file_id)
    return {"ok": True, "trip_id": trip_id, "file_id": body.file_id}


class MoveFileToTripBody(BaseModel):
    file_id: int
    target_trip_id: int


@router.post("/api/trips/move-file-to-trip")
def api_trip_move_file_to_trip(body: MoveFileToTripBody) -> dict[str, Any]:
    """
    Добавить файл в поездку (или перенести из другой).
    Если у целевой поездки есть папка на ЯД и файл disk: — физически перемещает на YaDisk.
    Иначе только привязывает к поездке (attach). Пересчитывает даты целевой поездки.
    """
    source_trip_id: int | None = None
    target_trip = get_trip(body.target_trip_id)
    if not target_trip:
        raise HTTPException(status_code=404, detail="Target trip not found")
    yd_folder = (target_trip.get("yd_folder_path") or "").strip().rstrip("/")
    do_physical_move = bool(yd_folder and yd_folder.startswith("disk:"))

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT path, name, parent_path FROM files WHERE id = ?", (body.file_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        path = str(row["path"] or "")
        name = str(row["name"] or "") or os.path.basename(path)
        cur.execute(
            "SELECT trip_id FROM trip_files WHERE file_id = ? AND status = 'included'",
            (body.file_id,),
        )
        src_row = cur.fetchone()
        source_trip_id = int(src_row["trip_id"]) if src_row else None
    finally:
        conn.close()

    if do_physical_move and not path.startswith("disk:"):
        do_physical_move = False

    new_path: str | None = None
    if do_physical_move:
        new_path = yd_folder.rstrip("/") + "/" + name
        disk = get_disk()
        src_norm = _normalize_yadisk_path(path)
        dst_norm = _normalize_yadisk_path(new_path)
        try:
            disk.move(src_norm, dst_norm, overwrite=False)
        except TypeError:
            disk.move(src_norm, dst_norm)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"YaDisk move failed: {type(e).__name__}: {e}") from e
        ds = DedupStore()
        try:
            ds.update_path(old_path=path, new_path=new_path, new_name=name, new_parent_path=yd_folder.rstrip("/"))
        finally:
            ds.close()

    if source_trip_id is not None:
        trip_file_exclude(source_trip_id, body.file_id)
        src_trip = get_trip(source_trip_id)
        if src_trip and src_trip.get("cover_file_id") == body.file_id:
            trip_update(source_trip_id, cover_file_id=None)

    trip_file_attach(body.target_trip_id, body.file_id)
    file_date = get_file_taken_at_date(body.file_id)
    if file_date:
        _expand_trip_dates_if_needed(body.target_trip_id, file_date)
    else:
        start_date, end_date = trip_dates_from_files(body.target_trip_id) or (None, None)
        if start_date or end_date:
            trip_update(body.target_trip_id, start_date=start_date, end_date=end_date)

    return {
        "ok": True,
        "file_id": body.file_id,
        "source_trip_id": source_trip_id,
        "target_trip_id": body.target_trip_id,
        "new_path": new_path,
        "moved_on_disk": do_physical_move,
    }
