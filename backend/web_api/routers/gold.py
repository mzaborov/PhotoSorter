from __future__ import annotations

import mimetypes
import os
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from common.db import DedupStore, PipelineStore
from common.db import FaceStore
from logic.gold.store import (
    gold_faces_manual_rects_path,
    gold_faces_video_frames_path,
    gold_file_map,
    gold_merge_append_only,
    gold_normalize_path,
    gold_read_lines,
    gold_read_ndjson_by_path,
    gold_write_lines,
    gold_write_ndjson_by_path,
)

router = APIRouter()

APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _now_utc_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _strip_local_prefix(path: str) -> str:
    if (path or "").startswith("local:"):
        return path[len("local:") :]
    return path


def _basename_from_disk_path(path: str) -> str:
    p = path or ""
    if "/" not in p:
        return p
    return p.rsplit("/", 1)[-1]


def _short_path_for_ui(path: str) -> str:
    # Как в app/main.py: обрезаем disk:/Фото/ и оставляем хвост.
    p = path or ""
    prefix = "disk:/Фото/"
    if p.startswith(prefix):
        tail = p[len(prefix) :]
        return "…/" + tail
    return p


def _guess_mime_media_for_path(path: str) -> tuple[str | None, str | None]:
    """
    Для gold у нас обычно нет mime_type/media_type — угадываем по расширению.
    """
    guess_name = None
    if path.startswith("local:"):
        guess_name = _strip_local_prefix(path)
    elif path.startswith("disk:"):
        guess_name = _basename_from_disk_path(path)
    else:
        guess_name = path

    mt, _enc = mimetypes.guess_type(guess_name)
    mime_type = mt or None
    media_type = None
    if mime_type:
        if mime_type.startswith("image/"):
            media_type = "image"
        elif mime_type.startswith("video/"):
            media_type = "video"
    return mime_type, media_type


def _faces_preview_meta(*, path: str, mime_type: str | None, media_type: str | None, pipeline_run_id: int | None) -> dict[str, Any]:
    preview_kind = "none"  # 'image'|'video'|'none'
    preview_url: Optional[str] = None
    open_url: Optional[str] = None
    mt = (media_type or "").lower()
    mime = (mime_type or "").lower()
    if path.startswith("disk:"):
        open_url = "/api/yadisk/open?path=" + urllib.parse.quote(path, safe="")
        # Для disk: путей пробуем определить тип по расширению, если mime_type не задан
        if not mime_type:
            import mimetypes
            basename = _basename_from_disk_path(path)
            guessed_mime, _ = mimetypes.guess_type(basename)
            if guessed_mime:
                mime = guessed_mime.lower()
                if guessed_mime.startswith("image/"):
                    mt = "image"
                elif guessed_mime.startswith("video/"):
                    mt = "video"
        if mt == "image" or mime.startswith("image/"):
            preview_kind = "image"
            preview_url = "/api/yadisk/preview-image?size=M&path=" + urllib.parse.quote(path, safe="")
        elif mt == "video" or mime.startswith("video/"):
            preview_kind = "video"
            preview_url = None
        # Если тип не определен, но путь выглядит как изображение, пробуем показать preview
        elif preview_kind == "none" and path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")):
            preview_kind = "image"
            preview_url = "/api/yadisk/preview-image?size=M&path=" + urllib.parse.quote(path, safe="")
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


def _root_like_for_pipeline_run_id(pipeline_run_id: int) -> str | None:
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        return None
    root_path = str(pr.get("root_path") or "")
    if not root_path:
        return None
    if root_path.startswith("disk:"):
        rp = root_path.rstrip("/")
        return rp + "/%"
    try:
        rp_abs = os.path.abspath(root_path).rstrip("\\/") + "\\"
        return "local:" + rp_abs + "%"
    except Exception:
        return None


def _latest_pipeline_run_id(*, kind: str = "local_sort") -> int | None:
    ps = PipelineStore()
    try:
        pr = ps.get_latest_any(kind=str(kind))
        if not pr:
            # Fallback: если kind отличается (или появятся другие kind), берём самый новый прогон вообще.
            cur = ps.conn.cursor()
            cur.execute("SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                return int(row[0] or 0) or None
    finally:
        ps.close()
    if not pr:
        return None
    try:
        rid = int(pr.get("id") or 0)
        return rid or None
    except Exception:
        return None


def _effective_tab_sql() -> str:
    """
    Keep semantics in sync with /api/faces (run-scoped manual labels).
    Uses aliases: f (files), m (files_manual_labels).
    """
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
) -> dict[str, Any]:
    m = gold_file_map()
    fp = m.get(gold_name)
    if not fp:
        return {"gold": gold_name, "expected": expected_tab, "total": 0, "mism": 0}
    lines = gold_read_lines(fp)
    paths: list[str] = []
    for raw in lines:
        nm = gold_normalize_path(raw)
        pp = nm.get("path") or ""
        if pp and (pp.startswith("local:") or pp.startswith("disk:")):
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
        return {"gold": gold_name, "expected": expected_tab, "total": 0, "mism": 0}

    placeholders = ",".join(["?"] * len(uniq))
    eff_sql = _effective_tab_sql()
    sql = f"""
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN ({eff_sql}) != ? THEN 1 ELSE 0 END) AS mism
        FROM files f
        LEFT JOIN files_manual_labels m
          ON m.pipeline_run_id = ? AND m.file_id = f.id
        WHERE f.status != 'deleted'
          AND (? IS NULL OR COALESCE(f.faces_run_id, -1) = ?)
          AND f.path IN ({placeholders})
    """
    frid = int(face_run_id) if face_run_id else None
    row = ds.conn.execute(sql, [expected_tab, int(pipeline_run_id), frid, frid, *uniq]).fetchone()
    total = int(row["total"] or 0) if row else 0
    mism = int(row["mism"] or 0) if row else 0
    return {"gold": gold_name, "expected": expected_tab, "total": total, "mism": mism}


@router.get("/api/gold/runs-metrics")
def api_gold_runs_metrics(limit: int = 20, include_failed: bool = False) -> dict[str, Any]:
    """
    Table for /gold: last pipeline runs + a few gold-based metrics (effective).
    """
    lim = max(1, min(200, int(limit or 20)))
    ps = PipelineStore()
    try:
        cur = ps.conn.cursor()
        where = "WHERE kind = 'local_sort'"
        params: list[Any] = []
        if not bool(include_failed):
            where += " AND status IN ('running','completed')"
        params.append(lim)
        cur.execute(
            f"""
            SELECT id, kind, status, root_path, face_run_id, started_at, finished_at
            FROM pipeline_runs
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        )
        runs = [dict(r) for r in cur.fetchall()]
    finally:
        ps.close()

    fs = FaceStore()
    ds = DedupStore()
    psm = PipelineStore()
    try:
        out: list[dict[str, Any]] = []
        for r in runs:
            rid = int(r.get("id") or 0)
            root_like = _root_like_for_pipeline_run_id(rid) if rid else None
            face_run_id = int(r.get("face_run_id") or 0)

            sorted_total = None
            processed = None
            if face_run_id:
                try:
                    fr = fs.get_run_by_id(run_id=face_run_id) or {}
                    sorted_total = int(fr.get("total_files") or 0) if fr.get("total_files") is not None else None
                    processed = int(fr.get("processed_files") or 0) if fr.get("processed_files") is not None else None
                except Exception:
                    sorted_total = None
                    processed = None

            # Prefer persisted snapshot metrics (so values don't float after next run overwrites files.*).
            snap = None
            try:
                snap = psm.get_metrics_for_run(pipeline_run_id=rid)
            except Exception:
                snap = None

            if snap:
                cats = {"gold": "cats_gold", "expected": "animals", "total": snap.get("cats_total") or 0, "mism": snap.get("cats_mism") or 0}
                faces = {"gold": "faces_gold", "expected": "faces", "total": snap.get("faces_total") or 0, "mism": snap.get("faces_mism") or 0}
                no_faces = {"gold": "no_faces_gold", "expected": "no_faces", "total": snap.get("no_faces_total") or 0, "mism": snap.get("no_faces_mism") or 0}
                # step2 totals from snapshot if present
                sorted_total = snap.get("step2_total") if snap.get("step2_total") is not None else sorted_total
                processed = snap.get("step2_processed") if snap.get("step2_processed") is not None else processed
                metrics_source = "snapshot"
            else:
                # Строго "по прогону": без снапшота метрики не показываем.
                # Иначе для running (без face_run_id) получается "сюр": считаем по текущей базе, а не по этому прогону.
                cats = {"gold": "cats_gold", "expected": "animals", "total": None, "mism": None}
                faces = {"gold": "faces_gold", "expected": "faces", "total": None, "mism": None}
                no_faces = {"gold": "no_faces_gold", "expected": "no_faces", "total": None, "mism": None}
                metrics_source = "pending" if str(r.get("status") or "") == "running" else "unavailable"

            out.append(
                {
                    "pipeline_run_id": rid,
                    "status": r.get("status"),
                    "started_at": r.get("started_at"),
                    "finished_at": r.get("finished_at"),
                    "sorted_step2_total": sorted_total,
                    "sorted_step2_processed": processed,
                    "cats": cats,
                    "faces": faces,
                    "no_faces": no_faces,
                    "metrics_source": metrics_source,
                }
            )
    finally:
        fs.close()
        ds.close()
        psm.close()

    return {"ok": True, "limit": lim, "items": out}


@router.get("/gold", response_class=HTMLResponse)
def gold_debug_page(request: Request):
    """
    Debug UI для редактирования gold-списков (regression/cases/*_gold.txt) и обновления их из SQLite.
    """
    return templates.TemplateResponse("gold.html", {"request": request})


@router.get("/api/gold/names")
def api_gold_names() -> dict[str, Any]:
    m = gold_file_map()
    counts: dict[str, int] = {}
    for k, p in m.items():
        try:
            counts[k] = len(gold_read_lines(p))
        except Exception:
            counts[k] = 0
    # Virtual tab: duplicates across gold files
    try:
        dups = _gold_duplicates_index()
        counts["duplicates"] = len(dups)
    except Exception:
        counts["duplicates"] = 0
    # Virtual tab: faces_manual_rects (NDJSON)
    try:
        faces_manual_rects = gold_read_ndjson_by_path(gold_faces_manual_rects_path())
        counts["faces_manual_rects"] = len(faces_manual_rects)
    except Exception:
        counts["faces_manual_rects"] = 0
    return {"ok": True, "names": list(m.keys()) + ["duplicates", "faces_manual_rects"], "counts": counts}


def _gold_duplicates_index() -> dict[str, dict[str, Any]]:
    """
    Returns mapping: normalized_path -> info {names: [gold_name...], raw_by_name: {gold_name: raw_line}}
    Only includes paths present in 2+ distinct gold files.
    """
    m = gold_file_map()
    by_path: dict[str, dict[str, Any]] = {}
    for name, p in m.items():
        try:
            lines = gold_read_lines(p)
        except Exception:
            lines = []
        for raw in lines:
            nm = gold_normalize_path(raw)
            path = nm.get("path") or ""
            raw_path = nm.get("raw_path") or raw
            if not path:
                continue
            ent = by_path.setdefault(path, {"names": set(), "raw_by_name": {}})
            ent["names"].add(name)
            # keep first raw line per file name
            ent["raw_by_name"].setdefault(name, raw_path)
    out: dict[str, dict[str, Any]] = {}
    for path, ent in by_path.items():
        names = sorted(list(ent["names"]))
        if len(names) >= 2:
            out[path] = {"path": path, "names": names, "raw_by_name": dict(ent["raw_by_name"])}
    return out


@router.get("/api/gold/list")
def api_gold_list(
    name: str,
    q: str = "",
    cursor: str | None = None,
    limit: int = 80,
    pipeline_run_id: int | None = None,
) -> dict[str, Any]:
    m = gold_file_map()
    dup_idx: dict[str, dict[str, Any]] | None = None
    if name == "duplicates":
        idx = _gold_duplicates_index()
        dup_idx = idx
        # list of normalized paths
        items = sorted(idx.keys())
    elif name == "faces_manual_rects":
        # Специальная обработка для faces_manual_rects_gold.ndjson
        # Возвращаем список путей из NDJSON
        try:
            rects_by_path = gold_read_ndjson_by_path(gold_faces_manual_rects_path())
            items = sorted(rects_by_path.keys())
        except Exception:
            items = []
    else:
        if name not in m:
            raise HTTPException(status_code=400, detail="unknown gold name")
        items = gold_read_lines(m[name])
    qq = (q or "").strip().lower()
    if qq:
        items = [x for x in items if qq in x.lower()]

    start_idx = 0
    cur_s = (cursor or "").strip()
    if cur_s:
        try:
            start_idx = items.index(cur_s) + 1
        except ValueError:
            start_idx = 0
    limit_i = max(10, min(300, int(limit or 80)))
    chunk = items[start_idx : start_idx + limit_i]

    rects_by_path: dict[str, dict[str, Any]] | None = None
    video_frames_by_path: dict[str, dict[str, Any]] | None = None
    if name == "faces_gold" or name == "faces_manual_rects":
        try:
            rects_by_path = gold_read_ndjson_by_path(gold_faces_manual_rects_path())
        except Exception:
            rects_by_path = None
        try:
            video_frames_by_path = gold_read_ndjson_by_path(gold_faces_video_frames_path())
        except Exception:
            video_frames_by_path = None

    # Если pipeline_run_id не передали — берём последний, чтобы:
    # - можно было безопасно отдавать local preview через /api/local/preview (нужен pipeline_run_id для root guard)
    # - /gold мог открывать /faces по кнопке "Кадры" с корректным run_id
    eff_pipeline_run_id: int | None = None
    try:
        eff_pipeline_run_id = int(pipeline_run_id) if pipeline_run_id is not None else None
    except Exception:
        eff_pipeline_run_id = None
    if eff_pipeline_run_id is None:
        latest = _latest_pipeline_run_id()
        eff_pipeline_run_id = int(latest) if latest is not None else None

    out_items: list[dict[str, Any]] = []
    for raw in chunk:
        if name == "duplicates":
            # raw is already normalized path
            raw_path = str(raw)
            path = str(raw)
            nm = {"raw_path": raw_path, "path": path}
        elif name == "faces_manual_rects":
            # raw is already a normalized path from NDJSON keys
            raw_path = str(raw)
            path = str(raw)
            nm = {"raw_path": raw_path, "path": path}
        else:
            nm = gold_normalize_path(raw)
            raw_path = nm["raw_path"]
            path = nm["path"]
        mime_type, media_type = _guess_mime_media_for_path(path)
        meta = _faces_preview_meta(path=path, mime_type=mime_type, media_type=media_type, pipeline_run_id=eff_pipeline_run_id)

        manual_rects: list[dict[str, int]] | None = None
        manual_rects_run_id: int | None = None
        video_frames: list[dict[str, Any]] | None = None
        if rects_by_path is not None:
            obj = rects_by_path.get(path)
            if isinstance(obj, dict):
                rects = obj.get("rects")
                if isinstance(rects, list) and rects:
                    clean: list[dict[str, int]] = []
                    for r in rects:
                        if not isinstance(r, dict):
                            continue
                        try:
                            x = int(r.get("x") or 0)
                            y = int(r.get("y") or 0)
                            w = int(r.get("w") or 0)
                            h = int(r.get("h") or 0)
                        except Exception:
                            continue
                        if w <= 0 or h <= 0:
                            continue
                        clean.append({"x": x, "y": y, "w": w, "h": h})
                    if clean:
                        manual_rects = clean
                        try:
                            manual_rects_run_id = int(obj.get("run_id") or 0) or None
                        except Exception:
                            manual_rects_run_id = None

        if video_frames_by_path is not None:
            objv = video_frames_by_path.get(path)
            if isinstance(objv, dict):
                frames = objv.get("frames")
                if isinstance(frames, list) and frames:
                    clean_frames: list[dict[str, Any]] = []
                    for fr in frames:
                        if not isinstance(fr, dict):
                            continue
                        try:
                            idx = int(fr.get("idx") or 0)
                        except Exception:
                            idx = 0
                        if idx not in (1, 2, 3):
                            continue
                        t = fr.get("t_sec")
                        try:
                            t_sec = float(t) if t is not None else None
                        except Exception:
                            t_sec = None
                        rects = fr.get("rects")
                        rects_out: list[dict[str, int]] = []
                        if isinstance(rects, list):
                            for r in rects:
                                if not isinstance(r, dict):
                                    continue
                                try:
                                    x = int(r.get("x") or 0)
                                    y = int(r.get("y") or 0)
                                    w = int(r.get("w") or 0)
                                    h = int(r.get("h") or 0)
                                except Exception:
                                    continue
                                if w <= 0 or h <= 0:
                                    continue
                                rects_out.append({"x": x, "y": y, "w": w, "h": h})
                        clean_frames.append({"idx": idx, "t_sec": t_sec, "rects": rects_out})
                    if clean_frames:
                        video_frames = sorted(clean_frames, key=lambda z: int(z.get("idx") or 0))

        obj: dict[str, Any] = {
            "raw_path": raw_path,
            "path": path,
            "path_short": _short_path_for_ui(path) if path.startswith("disk:") else raw_path,
            "mime_type": mime_type,
            "media_type": media_type,
            "manual_rects": manual_rects,
            "manual_rects_run_id": manual_rects_run_id,
            "video_frames": video_frames,
            **meta,
        }
        if name == "duplicates":
            try:
                ent = (dup_idx or {}).get(path) or {}
                obj["dup_names"] = list(ent.get("names") or [])
            except Exception:
                obj["dup_names"] = []
        out_items.append(obj)

    next_cursor = out_items[-1]["raw_path"] if out_items else None
    has_more = bool(start_idx + limit_i < len(items))

    return {
        "ok": True,
        "name": name,
        "q": qq,
        "pipeline_run_id": eff_pipeline_run_id,
        "total": len(items),
        "count": len(out_items),
        "cursor": cur_s or None,
        "next_cursor": next_cursor if has_more else None,
        "has_more": bool(has_more),
        "limit": limit_i,
        "items": out_items,
    }


@router.get("/api/gold/faces-by-persons")
def api_gold_faces_by_persons() -> dict[str, Any]:
    """
    Возвращает лица из faces_manual_rects_gold.ndjson, сгруппированные по персонам.
    Каждая персона - это закладка с её лицами.
    """
    from common.db import get_connection
    
    # Читаем все записи из gold
    gold_path = gold_faces_manual_rects_path()
    gold_data = gold_read_ndjson_by_path(gold_path)
    
    if not gold_data:
        return {"ok": True, "persons": []}
    
    # Получаем соединение с БД для поиска персон по лицам
    conn = get_connection()
    cur = conn.cursor()
    
    # Группируем по персонам
    persons_dict: dict[int, dict[str, Any]] = {}  # person_id -> {name, faces: []}
    unassigned_faces: list[dict[str, Any]] = []  # Лица без персоны
    
    # Сначала получаем все персоны, у которых есть лица на файлах из gold
    # Это нужно, чтобы показать всех персон, даже если точное совпадение bbox не найдено
    gold_file_paths = list(gold_data.keys())
    if gold_file_paths:
        placeholders = ",".join(["?"] * len(gold_file_paths))
        cur.execute(
            f"""
            SELECT DISTINCT
                p.id as person_id,
                p.name as person_name
            FROM photo_rectangles pr
            JOIN files f ON f.id = pr.file_id
            LEFT JOIN face_clusters fc ON fc.id = pr.cluster_id
            LEFT JOIN persons p ON p.id = COALESCE(pr.manual_person_id, fc.person_id)
            WHERE f.path IN ({placeholders})
              AND pr.is_face = 1
              AND COALESCE(pr.ignore_flag, 0) = 0
              AND p.id IS NOT NULL
            ORDER BY 
              CASE WHEN p.name = ? THEN 1 ELSE 0 END,
              p.name
            """,
            gold_file_paths + ["Посторонние"],
        )
        for row in cur.fetchall():
            person_id = row["person_id"]
            person_name = row["person_name"] or f"Person {person_id}"
            if person_id not in persons_dict:
                persons_dict[person_id] = {
                    "person_id": person_id,
                    "person_name": person_name,
                    "faces": [],
                }
    
    for file_path, gold_entry in gold_data.items():
        rects = gold_entry.get("rects", [])
        if not isinstance(rects, list):
            continue
        
        # Для каждого прямоугольника ищем соответствующее лицо в БД
        for rect in rects:
            if not isinstance(rect, dict):
                continue
            try:
                x = int(rect.get("x", 0))
                y = int(rect.get("y", 0))
                w = int(rect.get("w", 0))
                h = int(rect.get("h", 0))
            except (ValueError, TypeError):
                continue
            
            if w <= 0 or h <= 0:
                continue
            
            # Ищем лицо в БД по file_path и bbox (с небольшой погрешностью)
            # Сначала пробуем точное совпадение, затем более широкое
            # Важно: берем только одно лицо (LIMIT 1) и сортируем по точности совпадения
            cur.execute(
                """
                SELECT 
                    pr.id as face_id,
                    f.path as file_path,
                    pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h,
                    COALESCE(pr.manual_person_id, fc.person_id) as person_id,
                    p.name as person_name
                FROM photo_rectangles pr
                JOIN files f ON f.id = pr.file_id
                LEFT JOIN face_clusters fc ON fc.id = pr.cluster_id
                LEFT JOIN persons p ON p.id = COALESCE(pr.manual_person_id, fc.person_id)
                WHERE f.path = ? 
                  AND pr.is_face = 1
                  AND ABS(pr.bbox_x - ?) <= 10
                  AND ABS(pr.bbox_y - ?) <= 10
                  AND ABS(pr.bbox_w - ?) <= 10
                  AND ABS(pr.bbox_h - ?) <= 10
                  AND COALESCE(pr.ignore_flag, 0) = 0
                ORDER BY 
                  CASE WHEN COALESCE(pr.manual_person_id, fc.person_id) IS NOT NULL THEN 0 ELSE 1 END,
                  (ABS(pr.bbox_x - ?) + ABS(pr.bbox_y - ?) + ABS(pr.bbox_w - ?) + ABS(pr.bbox_h - ?)) ASC,
                  pr.id ASC
                LIMIT 1
                """,
                (file_path, x, y, w, h, x, y, w, h),
            )
            face_row = cur.fetchone()
            
            # Если не нашли точное совпадение, пробуем найти любую персону на этом файле
            # face_row может быть Row объектом из SQLite, который поддерживает доступ через []
            if not face_row or "person_id" not in face_row.keys() or not face_row["person_id"]:
                cur.execute(
                    """
                    SELECT DISTINCT
                        COALESCE(pr.manual_person_id, fc.person_id) as person_id,
                        p.name as person_name
                    FROM photo_rectangles pr
                    JOIN files f ON f.id = pr.file_id
                    LEFT JOIN face_clusters fc ON fc.id = pr.cluster_id
                    LEFT JOIN persons p ON p.id = COALESCE(pr.manual_person_id, fc.person_id)
                    WHERE f.path = ?
                      AND pr.is_face = 1
                      AND COALESCE(pr.ignore_flag, 0) = 0
                      AND p.id IS NOT NULL
                    LIMIT 1
                    """,
                    (file_path,),
                )
                fallback_row = cur.fetchone()
                if fallback_row:
                    # Используем персону из fallback, но без face_id
                    face_row = {
                        "face_id": None,
                        "file_path": file_path,
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_w": w,
                        "bbox_h": h,
                        "person_id": fallback_row["person_id"],
                        "person_name": fallback_row["person_name"],
                    }
            
            # Получаем preview метаданные для файла
            mime_type, media_type = _guess_mime_media_for_path(file_path)
            # Если не определили тип, пробуем по расширению файла
            if not mime_type and file_path:
                import mimetypes
                # Для disk: путей используем basename
                if file_path.startswith("disk:"):
                    basename = _basename_from_disk_path(file_path)
                    guessed = mimetypes.guess_type(basename)[0]
                else:
                    ext = file_path.lower()
                    if "." in ext:
                        ext = ext.rsplit(".", 1)[-1]
                        guessed = mimetypes.guess_type(f"dummy.{ext}")[0]
                    else:
                        guessed = None
                if guessed:
                    mime_type = guessed
                    if guessed.startswith("image/"):
                        media_type = "image"
                    elif guessed.startswith("video/"):
                        media_type = "video"
            eff_pipeline_run_id = _latest_pipeline_run_id()
            meta = _faces_preview_meta(
                path=file_path,
                mime_type=mime_type,
                media_type=media_type,
                pipeline_run_id=eff_pipeline_run_id,
            )
            
            face_data = {
                "file_path": file_path,
                "path_short": _short_path_for_ui(file_path) if file_path.startswith("disk:") else file_path,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "face_id": face_row["face_id"] if face_row and "face_id" in face_row.keys() else None,
                "preview_kind": meta.get("preview_kind", "none"),
                "preview_url": meta.get("preview_url"),
                "mime_type": mime_type,
            }
            
            # Проверяем наличие person_id в face_row (может быть Row объект или dict)
            person_id_in_row = face_row["person_id"] if face_row and "person_id" in face_row.keys() else None
            if face_row and person_id_in_row:
                person_id = face_row["person_id"]
                person_name = face_row["person_name"] if "person_name" in face_row.keys() else f"Person {person_id}"
                
                if person_id not in persons_dict:
                    persons_dict[person_id] = {
                        "person_id": person_id,
                        "person_name": person_name,
                        "faces": [],
                    }
                
                # Дедупликация: проверяем, нет ли уже лица с таким же file_path и координатами (с погрешностью)
                # для этой персоны. Также проверяем face_id, если он есть (более надежная проверка).
                is_duplicate = False
                # face_row может быть dict или Row объектом из SQLite
                current_face_id = face_row["face_id"] if face_row and "face_id" in face_row.keys() else None
                
                for existing_face in persons_dict[person_id]["faces"]:
                    if existing_face["file_path"] == file_path:
                        # Если есть face_id, проверяем по нему (более надежно)
                        existing_face_id = existing_face.get("face_id")
                        if current_face_id and existing_face_id and current_face_id == existing_face_id:
                            is_duplicate = True
                            break
                        # Иначе проверяем по координатам (с погрешностью)
                        existing_bbox = existing_face["bbox"]
                        if (abs(existing_bbox.get("x", 0) - x) <= 10 and
                            abs(existing_bbox.get("y", 0) - y) <= 10 and
                            abs(existing_bbox.get("w", 0) - w) <= 10 and
                            abs(existing_bbox.get("h", 0) - h) <= 10):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    persons_dict[person_id]["faces"].append(face_data)
            else:
                # Дедупликация для unassigned_faces тоже
                is_duplicate = False
                for existing_face in unassigned_faces:
                    if existing_face["file_path"] == file_path:
                        existing_bbox = existing_face["bbox"]
                        if (abs(existing_bbox.get("x", 0) - x) <= 10 and
                            abs(existing_bbox.get("y", 0) - y) <= 10 and
                            abs(existing_bbox.get("w", 0) - w) <= 10 and
                            abs(existing_bbox.get("h", 0) - h) <= 10):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unassigned_faces.append(face_data)
    
    # Преобразуем в список и сортируем:
    # 1. Обычные персоны (по имени)
    # 2. "Посторонние" (если есть)
    # 3. "Не назначено" (если есть)
    IGNORED_PERSON_NAME = "Посторонние"
    regular_persons = []
    ignored_person = None
    for p in persons_dict.values():
        if p["person_name"] == IGNORED_PERSON_NAME:
            ignored_person = p
        else:
            regular_persons.append(p)
    
    persons_list = sorted(regular_persons, key=lambda p: p["person_name"])
    
    # Добавляем "Не назначено" перед "Посторонние"
    if unassigned_faces:
        persons_list.append({
            "person_id": None,
            "person_name": "Не назначено",
            "faces": unassigned_faces,
        })
    
    # Добавляем "Посторонние" в самый конец
    if ignored_person:
        persons_list.append(ignored_person)
    
    return {
        "ok": True,
        "persons": persons_list,
        "total_persons": len(persons_list),
        "total_faces": sum(len(p["faces"]) for p in persons_list),
    }


@router.post("/api/gold/resolve-duplicate")
def api_gold_resolve_duplicate(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Ensure a given path appears in exactly one gold list.
    Removes it from all other gold files; keeps (or adds) it to keep_name.
    """
    keep_name = str(payload.get("keep_name") or "").strip()
    path_s = str(payload.get("path") or "").strip()
    if not keep_name:
        raise HTTPException(status_code=400, detail="keep_name is required")
    if not path_s:
        raise HTTPException(status_code=400, detail="path is required")
    m = gold_file_map()
    if keep_name not in m:
        raise HTTPException(status_code=400, detail="unknown keep_name")

    # Normalize the requested path like gold does.
    norm = gold_normalize_path(path_s)
    norm_path = str(norm.get("path") or path_s).strip()
    if not norm_path:
        raise HTTPException(status_code=400, detail="invalid path")

    changed: dict[str, int] = {}
    kept_in = False

    for name, fp in m.items():
        lines = gold_read_lines(fp)
        new_lines: list[str] = []
        removed = 0
        kept_raw: str | None = None

        for raw in lines:
            nm = gold_normalize_path(raw)
            p2 = str(nm.get("path") or "").strip()
            if p2 != norm_path:
                new_lines.append(raw)
                continue
            # p2 matches target path
            if name == keep_name and kept_raw is None:
                kept_raw = raw
                new_lines.append(raw)
                kept_in = True
            else:
                removed += 1

        if name == keep_name and not kept_in:
            # Not present in keep file; add as normalized path line.
            new_lines.append(norm_path)
            kept_in = True
            changed[name] = changed.get(name, 0) + 1

        if removed > 0 or (name == keep_name and kept_raw is None):
            gold_write_lines(fp, new_lines)
        if removed > 0:
            changed[name] = changed.get(name, 0) + removed

    return {"ok": True, "path": norm_path, "keep_name": keep_name, "changed": changed}


@router.post("/api/gold/delete")
def api_gold_delete(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    path_s = str(payload.get("path") or "").strip()
    m = gold_file_map()
    if name not in m:
        raise HTTPException(status_code=400, detail="unknown gold name")
    if not path_s:
        raise HTTPException(status_code=400, detail="path is required")
    p = m[name]
    lines = gold_read_lines(p)
    new_lines = [x for x in lines if x != path_s]
    gold_write_lines(p, new_lines)
    return {"ok": True, "name": name, "removed": 1 if len(new_lines) != len(lines) else 0}


@router.post("/api/gold/update-from-db")
def api_gold_update_from_db(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Append-only обновление gold из SQLite по текущим (ручным) меткам/флагам.
    Если передан pipeline_run_id — ограничиваем выборку root этого прогона.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    root_like = None
    if pipeline_run_id is None:
        # По умолчанию — "текущий" прогон: последний pipeline_run_id (Variant A).
        latest = _latest_pipeline_run_id()
        if latest is not None:
            pipeline_run_id = latest
    if pipeline_run_id is not None:
        if not isinstance(pipeline_run_id, int):
            raise HTTPException(status_code=400, detail="pipeline_run_id must be int or null")
        root_like = _root_like_for_pipeline_run_id(pipeline_run_id)
        if root_like is None:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")

    base_where = ["status != 'deleted'"]
    base_params: list[Any] = []
    if root_like:
        base_where.append("path LIKE ?")
        base_params.append(root_like)
    base_where_sql = " AND ".join(base_where)

    # ВАЖНО: в запросах вида "FROM files f JOIN ..." нужно квалифицировать колонки,
    # иначе получим "ambiguous column name: path" (например, когда JOIN имеет свою колонку path).
    base_where_f = ["f.status != 'deleted'"]
    if root_like:
        base_where_f.append("f.path LIKE ?")
    base_where_sql_f = " AND ".join(base_where_f)

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()

        def q(sql: str, params: list[Any]) -> list[str]:
            cur.execute(sql, params)
            return [str(r[0]) for r in cur.fetchall()]

        # Run-scoped manual labels (legacy mode удален)
        cats = q(
            f"""
            SELECT f.path
            FROM files f
            JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE {base_where_sql_f}
              AND COALESCE(m.animals_manual,0)=1
            ORDER BY f.path ASC
            """,
            [int(pipeline_run_id)] + list(base_params),
        )
        faces_gold = q(
            f"""
            SELECT f.path
            FROM files f
            JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE {base_where_sql_f}
              AND lower(trim(coalesce(m.faces_manual_label,''))) = 'faces'
            ORDER BY f.path ASC
            """,
            [int(pipeline_run_id)] + list(base_params),
        )
        no_faces = q(
            f"""
            SELECT f.path
            FROM files f
            JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE {base_where_sql_f}
              AND lower(trim(coalesce(m.faces_manual_label,''))) = 'no_faces'
            ORDER BY f.path ASC
            """,
            [int(pipeline_run_id)] + list(base_params),
        )
        people_no_face = q(
            f"""
            SELECT f.path
            FROM files f
            JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE {base_where_sql_f}
              AND COALESCE(m.people_no_face_manual,0) = 1
            ORDER BY f.path ASC
            """,
            [int(pipeline_run_id)] + list(base_params),
        )
        quarantine_gold = q(
            f"""
            SELECT f.path
            FROM files f
            JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.file_id = f.id
            WHERE {base_where_sql_f}
              AND COALESCE(m.quarantine_manual,0) = 1
            ORDER BY f.path ASC
            """,
            [int(pipeline_run_id)] + list(base_params),
        )

        # --- manual rectangles export (NDJSON, overwrite by path) ---
        cur.execute(
            f"""
            SELECT
              f.path AS path,
              pr.run_id AS run_id,
              pr.bbox_x AS x,
              pr.bbox_y AS y,
              pr.bbox_w AS w,
              pr.bbox_h AS h
            FROM photo_rectangles pr
            JOIN files f ON f.id = pr.file_id
            WHERE {base_where_sql_f}
              AND pr.is_face = 1
              AND COALESCE(pr.is_manual, 0) = 1
              AND f.faces_run_id IS NOT NULL
              AND pr.run_id = f.faces_run_id
            ORDER BY f.path ASC, pr.face_index ASC, pr.id ASC
            """,
            list(base_params),
        )
        rect_rows = [dict(r) for r in cur.fetchall()]

        # --- video manual frames export (NDJSON, overwrite by path) — данные из photo_rectangles ---
        if isinstance(pipeline_run_id, int):
            cur.execute(
                f"""
                SELECT
                  f.path AS path,
                  fr.frame_idx AS frame_idx,
                  fr.frame_t_sec AS t_sec,
                  fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                  fr.manual_created_at AS updated_at
                FROM photo_rectangles fr
                JOIN files f ON f.id = fr.file_id
                JOIN pipeline_runs pr ON pr.face_run_id = fr.run_id
                WHERE {base_where_sql_f}
                  AND pr.id = ? AND fr.frame_idx IN (1, 2, 3)
                ORDER BY f.path ASC, fr.frame_idx ASC, fr.face_index ASC
                """,
                list(base_params) + [int(pipeline_run_id)],
            )
            raw_rows = [dict(r) for r in cur.fetchall()]
            # группируем по (path, frame_idx), собираем rects_json
            by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
            for r in raw_rows:
                p = str(r.get("path") or "")
                idx = int(r.get("frame_idx") or 0)
                if not p or idx not in (1, 2, 3):
                    continue
                k = (p, idx)
                if k not in by_key:
                    by_key[k] = []
                by_key[k].append(r)
            video_rows = []
            for (p, idx), grp in sorted(by_key.items()):
                t_sec = grp[0].get("t_sec") if grp else None
                updated_at = max((g.get("updated_at") or "") for g in grp) if grp else None
                rects = [
                    {"x": int(g.get("bbox_x") or 0), "y": int(g.get("bbox_y") or 0), "w": int(g.get("bbox_w") or 0), "h": int(g.get("bbox_h") or 0)}
                    for g in grp
                ]
                video_rows.append({"path": p, "frame_idx": idx, "t_sec": t_sec, "rects_json": json.dumps(rects, ensure_ascii=False), "updated_at": updated_at})
        else:
            video_rows = []
    finally:
        ds.close()

    m = gold_file_map()
    results: dict[str, Any] = {}
    results["cats_gold"] = {"added": gold_merge_append_only(m["cats_gold"], cats), "db_total": len(cats)}
    results["faces_gold"] = {"added": gold_merge_append_only(m["faces_gold"], faces_gold), "db_total": len(faces_gold)}
    results["no_faces_gold"] = {"added": gold_merge_append_only(m["no_faces_gold"], no_faces), "db_total": len(no_faces)}
    results["people_no_face_gold"] = {"added": gold_merge_append_only(m["people_no_face_gold"], people_no_face), "db_total": len(people_no_face)}
    results["quarantine_gold"] = {"added": gold_merge_append_only(m["quarantine_gold"], quarantine_gold), "db_total": len(quarantine_gold)}
    results["drawn_faces_gold"] = {"added": 0, "db_total": len(gold_read_lines(m["drawn_faces_gold"]))}

    rects_path = gold_faces_manual_rects_path()
    existing_rects = gold_read_ndjson_by_path(rects_path)
    by_path: dict[str, list[dict[str, int]]] = {}
    run_by_path: dict[str, int] = {}
    for r in rect_rows:
        p = str(r.get("path") or "")
        if not p:
            continue
        run_by_path[p] = int(r.get("run_id") or 0) or run_by_path.get(p, 0)
        by_path.setdefault(p, []).append({"x": int(r.get("x") or 0), "y": int(r.get("y") or 0), "w": int(r.get("w") or 0), "h": int(r.get("h") or 0)})

    for p, rects in by_path.items():
        existing_rects[p] = {"path": p, "run_id": int(run_by_path.get(p) or 0), "rects": rects}

    # delete those that no longer have manual rects in DB (within current scope)
    to_delete: list[str] = []
    for p in list(existing_rects.keys()):
        if p in by_path:
            continue
        if root_like and root_like.endswith("%"):
            pref = root_like[:-1]
            if not p.startswith(pref):
                continue
        to_delete.append(p)
    for p in to_delete:
        existing_rects.pop(p, None)

    gold_write_ndjson_by_path(rects_path, existing_rects)
    results["faces_manual_rects_gold"] = {"updated": len(by_path), "file": str(rects_path)}

    # --- video frames NDJSON (by path) ---
    vf_path = gold_faces_video_frames_path()
    existing_vf = gold_read_ndjson_by_path(vf_path)
    vf_by_path: dict[str, list[dict[str, Any]]] = {}
    for r in (video_rows or []):
        p = str(r.get("path") or "")
        if not p:
            continue
        try:
            idx = int(r.get("frame_idx") or 0)
        except Exception:
            idx = 0
        if idx not in (1, 2, 3):
            continue
        t = r.get("t_sec")
        try:
            t_sec = float(t) if t is not None else None
        except Exception:
            t_sec = None
        rects_out: list[dict[str, int]] = []
        try:
            raw = r.get("rects_json")
            if raw:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    for it in obj:
                        if not isinstance(it, dict):
                            continue
                        x = int(it.get("x") or 0)
                        y = int(it.get("y") or 0)
                        w = int(it.get("w") or 0)
                        h = int(it.get("h") or 0)
                        if w > 0 and h > 0:
                            rects_out.append({"x": x, "y": y, "w": w, "h": h})
        except Exception:
            rects_out = []
        vf_by_path.setdefault(p, []).append({"idx": idx, "t_sec": t_sec, "rects": rects_out})

    for p, frames in vf_by_path.items():
        existing_vf[p] = {"path": p, "frames": sorted(frames, key=lambda z: int(z.get("idx") or 0))}

    # delete those that no longer have video frames in DB (within current scope)
    to_delete_v: list[str] = []
    for p in list(existing_vf.keys()):
        if p in vf_by_path:
            continue
        if root_like and root_like.endswith("%"):
            pref = root_like[:-1]
            if not p.startswith(pref):
                continue
        to_delete_v.append(p)
    for p in to_delete_v:
        existing_vf.pop(p, None)

    gold_write_ndjson_by_path(vf_path, existing_vf)
    results["faces_video_frames_gold"] = {"updated": len(vf_by_path), "file": str(vf_path)}

    return {"ok": True, "pipeline_run_id": pipeline_run_id, "root_like": root_like, "results": results}


@router.post("/api/gold/apply-to-db")
def api_gold_apply_to_db(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Append-only заполнение ручной разметки в БД из gold-файлов.
    Пишем run-scoped manual labels в files_manual_labels (legacy mode удален).
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    root_like = None
    if pipeline_run_id is None:
        # По умолчанию — "текущий" прогон: последний pipeline_run_id (Variant A).
        latest = _latest_pipeline_run_id()
        if latest is not None:
            pipeline_run_id = latest
    
    # Требуем pipeline_run_id (legacy mode удален - метки должны быть run-scoped)
    if pipeline_run_id is None:
        raise HTTPException(status_code=400, detail="pipeline_run_id is required (legacy mode removed)")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id must be int")
    
    root_like = _root_like_for_pipeline_run_id(pipeline_run_id)
    if root_like is None:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")

    m = gold_file_map()

    ops: list[tuple[str, str]] = [
        ("faces_gold", "faces"),
        ("no_faces_gold", "no_faces"),
        ("people_no_face_gold", "people_no_face"),
        ("cats_gold", "cat"),
        ("quarantine_gold", "quarantine_manual"),
    ]

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        out: dict[str, Any] = {}
        total_applied = 0
        for gold_name, op in ops:
            p = m.get(gold_name)
            raw_lines = gold_read_lines(p) if p else []
            paths = []
            for raw in raw_lines:
                nm = gold_normalize_path(raw)
                pp = nm["path"]
                if pp and (pp.startswith("local:") or pp.startswith("disk:")):
                    paths.append(pp)
            seen = set()
            uniq = []
            for x in paths:
                if x in seen:
                    continue
                seen.add(x)
                uniq.append(x)

            existing: set[str] = set()
            if uniq:
                step = 400
                for i in range(0, len(uniq), step):
                    chunk = uniq[i : i + step]
                    qmarks = ",".join(["?"] * len(chunk))
                    if root_like:
                        cur.execute(
                            f"SELECT path FROM files WHERE status != 'deleted' AND path LIKE ? AND path IN ({qmarks})",
                            [root_like] + chunk,
                        )
                    else:
                        cur.execute(f"SELECT path FROM files WHERE status != 'deleted' AND path IN ({qmarks})", chunk)
                    existing.update([str(r[0]) for r in cur.fetchall()])

            applied = 0
            skipped = 0
            missing = 0

            for path in uniq:
                if path not in existing:
                    missing += 1
                    continue

                where_extra = ""
                params: list[Any] = []
                if root_like:
                    where_extra = " AND path LIKE ?"
                    params.append(root_like)

                # Run-scoped mode: write to files_manual_labels (legacy mode удален)
                # Получаем file_id
                cur.execute("SELECT id FROM files WHERE path = ? LIMIT 1", (path,))
                file_row = cur.fetchone()
                if not file_row:
                    missing += 1
                    continue
                file_id = file_row[0]
                
                cur.execute(
                    "INSERT OR IGNORE INTO files_manual_labels(pipeline_run_id, file_id) VALUES (?, ?)",
                    (int(pipeline_run_id), file_id),
                )
                if op in ("faces", "no_faces"):
                    cur.execute(
                        """
                        UPDATE files_manual_labels
                        SET faces_manual_label = ?, faces_manual_at = ?
                        WHERE pipeline_run_id = ?
                          AND file_id = ?
                          AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                          AND COALESCE(people_no_face_manual, 0) = 0
                          AND COALESCE(animals_manual, 0) = 0
                          AND COALESCE(quarantine_manual, 0) = 0
                        """,
                        [op, _now_utc_iso(), int(pipeline_run_id), file_id],
                    )
                elif op == "people_no_face":
                    cur.execute(
                        """
                        UPDATE files_manual_labels
                        SET
                          people_no_face_manual = 1,
                          people_no_face_person = NULL,
                          faces_manual_label = NULL,
                          faces_manual_at = NULL,
                          quarantine_manual = 0,
                          quarantine_manual_at = NULL,
                          animals_manual = 0,
                          animals_manual_kind = NULL,
                          animals_manual_at = NULL
                        WHERE pipeline_run_id = ?
                          AND file_id = ?
                          AND (
                            COALESCE(people_no_face_manual, 0) = 0
                            OR (faces_manual_label IS NOT NULL AND trim(coalesce(faces_manual_label,'')) != '')
                            OR COALESCE(animals_manual, 0) != 0
                            OR COALESCE(quarantine_manual, 0) != 0
                          )
                        """,
                        [int(pipeline_run_id), file_id],
                    )
                elif op == "cat":
                    cur.execute(
                        """
                        UPDATE files_manual_labels
                        SET animals_manual = 1, animals_manual_kind = 'cat', animals_manual_at = ?
                        WHERE pipeline_run_id = ?
                          AND file_id = ?
                          AND COALESCE(animals_manual, 0) = 0
                          AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                          AND COALESCE(people_no_face_manual, 0) = 0
                          AND COALESCE(quarantine_manual, 0) = 0
                        """,
                        [_now_utc_iso(), int(pipeline_run_id), file_id],
                    )
                elif op == "quarantine_manual":
                    cur.execute(
                        """
                        UPDATE files_manual_labels
                        SET quarantine_manual = 1, quarantine_manual_at = ?
                        WHERE pipeline_run_id = ?
                          AND file_id = ?
                          AND COALESCE(quarantine_manual, 0) = 0
                          AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                          AND COALESCE(people_no_face_manual, 0) = 0
                          AND COALESCE(animals_manual, 0) = 0
                        """,
                        [_now_utc_iso(), int(pipeline_run_id), file_id],
                    )
                # Legacy mode удален - всегда используем files_manual_labels

                rc = int(cur.rowcount or 0)
                if rc > 0:
                    applied += rc
                else:
                    skipped += 1

            ds.conn.commit()
            total_applied += applied
            out[gold_name] = {"op": op, "lines": len(raw_lines), "unique_paths": len(uniq), "applied": applied, "skipped": skipped, "missing": missing}

        # --- video manual frames import (NDJSON -> DB), run-scoped only ---
        # pipeline_run_id всегда есть (legacy mode удален)
            vf_path = gold_faces_video_frames_path()
            vf_by_path = gold_read_ndjson_by_path(vf_path)
            applied_v = 0
            skipped_v = 0
            missing_v = 0
            bad_v = 0
            paths_all = list(vf_by_path.keys())
            # ограничение по root_like
            if root_like and root_like.endswith("%"):
                pref = root_like[:-1]
                paths_all = [p for p in paths_all if str(p).startswith(pref)]

            # check existing in files (and within scope)
            existing: set[str] = set()
            if paths_all:
                step = 400
                for i in range(0, len(paths_all), step):
                    chunk = paths_all[i : i + step]
                    qmarks = ",".join(["?"] * len(chunk))
                    if root_like:
                        cur.execute(
                            f"SELECT path FROM files WHERE status != 'deleted' AND path LIKE ? AND path IN ({qmarks})",
                            [root_like] + list(chunk),
                        )
                    else:
                        cur.execute(f"SELECT path FROM files WHERE status != 'deleted' AND path IN ({qmarks})", list(chunk))
                    existing.update([str(r[0]) for r in cur.fetchall()])

            for p in paths_all:
                if p not in existing:
                    missing_v += 1
                    continue
                # Получаем file_id из path
                cur.execute("SELECT id FROM files WHERE path = ? LIMIT 1", (str(p),))
                file_row = cur.fetchone()
                if not file_row:
                    missing_v += 1
                    continue
                file_id = file_row[0]
                
                obj = vf_by_path.get(p)
                if not isinstance(obj, dict):
                    bad_v += 1
                    continue
                frames = obj.get("frames")
                if not isinstance(frames, list) or not frames:
                    skipped_v += 1
                    continue
                for fr in frames:
                    if not isinstance(fr, dict):
                        bad_v += 1
                        continue
                    try:
                        idx = int(fr.get("idx") or 0)
                    except Exception:
                        idx = 0
                    if idx not in (1, 2, 3):
                        bad_v += 1
                        continue
                    t = fr.get("t_sec")
                    try:
                        t_sec = float(t) if t is not None else None
                    except Exception:
                        t_sec = None
                    rects = fr.get("rects")
                    rects_clean: list[dict[str, Any]] = []
                    if isinstance(rects, list):
                        for r in rects:
                            if not isinstance(r, dict):
                                continue
                            try:
                                x = int(r.get("x") or 0)
                                y = int(r.get("y") or 0)
                                w = int(r.get("w") or 0)
                                h = int(r.get("h") or 0)
                            except Exception:
                                continue
                            if w > 0 and h > 0:
                                rects_clean.append({"x": x, "y": y, "w": w, "h": h})
                    try:
                        ds.upsert_video_manual_frame(
                            pipeline_run_id=int(pipeline_run_id),
                            path=str(p),
                            frame_idx=int(idx),
                            t_sec=float(t_sec) if t_sec is not None else None,
                            rects=rects_clean,
                        )
                        applied_v += 1
                    except Exception:
                        bad_v += 1

            ds.conn.commit()
            out["faces_video_frames_gold"] = {
                "file": str(vf_path),
                "paths": len(paths_all),
                "applied_frames": applied_v,
                "skipped_paths": skipped_v,
                "missing_paths": missing_v,
                "bad_records": bad_v,
            }
            total_applied += applied_v

        return {"ok": True, "pipeline_run_id": pipeline_run_id, "root_like": root_like, "total_applied": total_applied, "results": out}
    finally:
        ds.close()



