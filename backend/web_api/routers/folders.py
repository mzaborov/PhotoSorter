from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from common.db import list_folders

router = APIRouter()

APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/api/folders")
def api_folders() -> list[dict[str, Any]]:
    return list_folders(location="yadisk", role="target")


@router.get("/folders", response_class=HTMLResponse)
def folders_page(request: Request):
    folders = list_folders(location="yadisk", role="target")
    return templates.TemplateResponse(
        "folders.html",
        {
            "request": request,
            "folders": folders,
        },
    )

