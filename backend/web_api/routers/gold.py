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
from logic.gold.store import (
    gold_faces_manual_rects_path,
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
    return {"ok": True, "names": list(m.keys()), "counts": counts}


@router.get("/api/gold/list")
def api_gold_list(
    name: str,
    q: str = "",
    cursor: str | None = None,
    limit: int = 80,
    pipeline_run_id: int | None = None,
) -> dict[str, Any]:
    m = gold_file_map()
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
    if name == "faces_gold":
        try:
            rects_by_path = gold_read_ndjson_by_path(gold_faces_manual_rects_path())
        except Exception:
            rects_by_path = None

    out_items: list[dict[str, Any]] = []
    for raw in chunk:
        nm = gold_normalize_path(raw)
        raw_path = nm["raw_path"]
        path = nm["path"]
        mime_type, media_type = _guess_mime_media_for_path(path)
        meta = _faces_preview_meta(path=path, mime_type=mime_type, media_type=media_type, pipeline_run_id=pipeline_run_id)

        manual_rects: list[dict[str, int]] | None = None
        manual_rects_run_id: int | None = None
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

        out_items.append(
            {
                "raw_path": raw_path,
                "path": path,
                "path_short": _short_path_for_ui(path) if path.startswith("disk:") else raw_path,
                "mime_type": mime_type,
                "media_type": media_type,
                "manual_rects": manual_rects,
                "manual_rects_run_id": manual_rects_run_id,
                **meta,
            }
        )

    next_cursor = out_items[-1]["raw_path"] if out_items else None
    has_more = bool(start_idx + limit_i < len(items))

    return {
        "ok": True,
        "name": name,
        "q": qq,
        "total": len(items),
        "count": len(out_items),
        "cursor": cur_s or None,
        "next_cursor": next_cursor if has_more else None,
        "has_more": bool(has_more),
        "limit": limit_i,
        "items": out_items,
    }


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

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()

        def q(sql: str, params: list[Any]) -> list[str]:
            cur.execute(sql, params)
            return [str(r[0]) for r in cur.fetchall()]

        cats = q(
            f"SELECT path FROM files WHERE {base_where_sql} AND COALESCE(animals_auto,0)=1 ORDER BY path ASC",
            list(base_params),
        )
        faces_gold = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND lower(trim(coalesce(faces_manual_label,''))) = 'faces'
            ORDER BY path ASC
            """,
            list(base_params),
        )
        no_faces = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND lower(trim(coalesce(faces_manual_label,''))) = 'no_faces'
            ORDER BY path ASC
            """,
            list(base_params),
        )
        people_no_face = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND COALESCE(people_no_face_manual,0) = 1
            ORDER BY path ASC
            """,
            list(base_params),
        )
        quarantine_gold = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND COALESCE(faces_auto_quarantine,0) = 1
              AND lower(trim(coalesce(faces_quarantine_reason,''))) = 'manual'
            ORDER BY path ASC
            """,
            list(base_params),
        )

        # --- manual rectangles export (NDJSON, overwrite by path) ---
        cur.execute(
            f"""
            SELECT
              fr.file_path AS path,
              fr.run_id AS run_id,
              fr.bbox_x AS x,
              fr.bbox_y AS y,
              fr.bbox_w AS w,
              fr.bbox_h AS h
            FROM face_rectangles fr
            JOIN files f ON f.path = fr.file_path
            WHERE {base_where_sql}
              AND COALESCE(fr.is_manual, 0) = 1
              AND f.faces_run_id IS NOT NULL
              AND fr.run_id = f.faces_run_id
            ORDER BY fr.file_path ASC, fr.face_index ASC, fr.id ASC
            """,
            list(base_params),
        )
        rect_rows = [dict(r) for r in cur.fetchall()]
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

    return {"ok": True, "pipeline_run_id": pipeline_run_id, "root_like": root_like, "results": results}


@router.post("/api/gold/apply-to-db")
def api_gold_apply_to_db(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Append-only заполнение ручной разметки в БД из gold-файлов.
    Принцип: НЕ перетирать уже проставленные ручные поля/решения, а только дополнять пустые.
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    root_like = None
    if pipeline_run_id is not None:
        if not isinstance(pipeline_run_id, int):
            raise HTTPException(status_code=400, detail="pipeline_run_id must be int or null")
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

                if op in ("faces", "no_faces"):
                    cur.execute(
                        f"""
                        UPDATE files
                        SET faces_manual_label = ?, faces_manual_at = ?
                        WHERE path = ?
                          AND status != 'deleted'
                          AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                          AND COALESCE(people_no_face_manual, 0) = 0
                          {where_extra}
                        """,
                        [op, _now_utc_iso(), path] + params,
                    )
                elif op == "people_no_face":
                    cur.execute(
                        f"""
                        UPDATE files
                        SET people_no_face_manual = 1
                        WHERE path = ?
                          AND status != 'deleted'
                          AND COALESCE(people_no_face_manual, 0) = 0
                          AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                          {where_extra}
                        """,
                        [path] + params,
                    )
                elif op == "cat":
                    cur.execute(
                        f"""
                        UPDATE files
                        SET animals_auto = 1, animals_kind = 'cat',
                            faces_auto_quarantine = 0, faces_quarantine_reason = NULL
                        WHERE path = ?
                          AND status != 'deleted'
                          AND COALESCE(animals_auto, 0) = 0
                          AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                          AND COALESCE(people_no_face_manual, 0) = 0
                          {where_extra}
                        """,
                        [path] + params,
                    )
                elif op == "quarantine_manual":
                    cur.execute(
                        f"""
                        UPDATE files
                        SET faces_auto_quarantine = 1, faces_quarantine_reason = 'manual'
                        WHERE path = ?
                          AND status != 'deleted'
                          AND COALESCE(faces_auto_quarantine, 0) = 0
                          AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                          AND COALESCE(people_no_face_manual, 0) = 0
                          AND COALESCE(animals_auto, 0) = 0
                          {where_extra}
                        """,
                        [path] + params,
                    )

                rc = int(cur.rowcount or 0)
                if rc > 0:
                    applied += rc
                else:
                    skipped += 1

            ds.conn.commit()
            total_applied += applied
            out[gold_name] = {"op": op, "lines": len(raw_lines), "unique_paths": len(uniq), "applied": applied, "skipped": skipped, "missing": missing}

        return {"ok": True, "pipeline_run_id": pipeline_run_id, "root_like": root_like, "total_applied": total_applied, "results": out}
    finally:
        ds.close()



