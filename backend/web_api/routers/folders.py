from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from common.db import list_folders, update_folder_content_rule

router = APIRouter()


class FolderPatchBody(BaseModel):
    content_rule: str | None = None

APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/api/folders")
def api_folders() -> list[dict[str, Any]]:
    return list_folders(location="yadisk", role="target")


@router.patch("/api/folders/{folder_id}")
def api_folders_patch(folder_id: int, body: FolderPatchBody) -> dict[str, Any]:
    """
    Обновляет content_rule папки. Тело: {"content_rule": "только:[Агата]" или null}.
    """
    content_rule = (body.content_rule or "").strip() or None
    updated = update_folder_content_rule(folder_id, content_rule)
    if not updated:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"ok": True, "folder_id": folder_id}


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

