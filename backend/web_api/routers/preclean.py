from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from common.db import PipelineStore

router = APIRouter()

APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/preclean-results", response_class=HTMLResponse)
def preclean_results_page(request: Request, pipeline_run_id: int | None = None):
    """
    Результаты шага 1 (предочистка): что было/будет перемещено в _non_media и _broken_media.
    В DRY_RUN это "план перемещений" (список src->dst).
    """
    return templates.TemplateResponse("preclean_results.html", {"request": request, "pipeline_run_id": pipeline_run_id})


@router.get("/api/preclean-results/context")
def api_preclean_results_context(pipeline_run_id: int) -> dict[str, Any]:
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        cur = ps.conn.cursor()
        cur.execute(
            "SELECT root_path, dry_run, checked, non_media, broken_media, last_path, updated_at FROM preclean_state WHERE pipeline_run_id=? LIMIT 1",
            (int(pipeline_run_id),),
        )
        st = cur.fetchone()
    finally:
        ps.close()

    state = None
    if st:
        state = {
            "root_path": st[0],
            "dry_run": bool(int(st[1] or 0)),
            "checked": st[2],
            "non_media": st[3],
            "broken_media": st[4],
            "last_path": st[5],
            "updated_at": st[6],
        }
    return {"ok": True, "pipeline_run_id": int(pipeline_run_id), "run": pr, "state": state}


@router.get("/api/preclean-results/list")
def api_preclean_results_list(
    pipeline_run_id: int,
    kind: str = "non_media",
    q: str = "",
    page: int = 1,
    page_size: int = 80,
) -> dict[str, Any]:
    kind_n = (kind or "").strip().lower()
    if kind_n not in ("non_media", "broken_media"):
        raise HTTPException(status_code=400, detail="kind must be non_media|broken_media")
    qq = (q or "").strip().lower()
    p = max(1, int(page or 1))
    ps = max(10, min(400, int(page_size or 80)))
    off = (p - 1) * ps

    psr = PipelineStore()
    try:
        cur = psr.conn.cursor()
        where = "WHERE pipeline_run_id=? AND kind=?"
        params: list[Any] = [int(pipeline_run_id), str(kind_n)]
        if qq:
            where += " AND (lower(src_path) LIKE ? OR lower(dst_path) LIKE ?)"
            like = "%" + qq + "%"
            params.extend([like, like])
        cur.execute(f"SELECT COUNT(*) FROM preclean_moves {where}", tuple(params))
        total = int(cur.fetchone()[0] or 0)
        cur.execute(
            f"""
            SELECT src_path, dst_path, is_applied, created_at
            FROM preclean_moves
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params + [int(ps), int(off)]),
        )
        items = [
            {"src_path": r[0], "dst_path": r[1], "is_applied": bool(int(r[2] or 0)), "created_at": r[3]}
            for r in cur.fetchall()
        ]
    finally:
        psr.close()

    return {"ok": True, "pipeline_run_id": int(pipeline_run_id), "kind": kind_n, "page": p, "page_size": ps, "total": total, "items": items}

