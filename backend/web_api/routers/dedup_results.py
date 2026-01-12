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


@router.get("/dedup-results", response_class=HTMLResponse)
def dedup_results_page(request: Request):
    """
    Результаты шага 2 (дедупликация) в отдельных вкладках:
    2.1 дубли с архивом, 2.2 дубли внутри исходной папки.
    Источник данных: существующие API /api/sort/dup-in-archive и /api/sort/dup-in-source.
    """
    return templates.TemplateResponse("dedup_results.html", {"request": request})


@router.get("/api/dedup-results/context")
def api_dedup_results_context(pipeline_run_id: int) -> dict[str, Any]:
    """
    Контекст для страницы результатов шага 2:
    возвращает dedup_run_id и root_path для выбранного pipeline_run_id.
    """
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    drid = pr.get("dedup_run_id")
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "dedup_run_id": int(drid) if drid else None,
        "root_path": pr.get("root_path"),
    }

