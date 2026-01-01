from __future__ import annotations

import time
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from DB.db import list_folders
from yadisk_client import get_disk

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="PhotoSorter")

_T = TypeVar("_T")

# Общий пул для вызовов YaDisk (нужно, чтобы иметь timeout на уровне приложения).
# Важно: timeout не убивает запрос внутри библиотеки, но предотвращает "вечные" зависания в обработчике.
_YD_POOL = ThreadPoolExecutor(max_workers=16)

# Таймауты/ретраи по умолчанию (можно будет вынести в настройки).
YD_CALL_TIMEOUT_SEC = 15
YD_RETRIES = 2
YD_RETRY_DELAY_SEC = 0.5


def _get(item: Any, key: str) -> Optional[Any]:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _normalize_yadisk_path(path: str) -> str:
    # В БД у нас может храниться `disk:/...`, но в yadisk-python удобно использовать `/...`.
    p = (path or "").strip()
    if p.startswith("disk:"):
        p = p[len("disk:") :]
    if not p.startswith("/"):
        p = "/" + p
    return p


def _as_disk_path(path: str) -> str:
    p = (path or "").strip()
    if p.startswith("disk:"):
        return p
    p2 = p if p.startswith("/") else ("/" + p)
    return "disk:" + p2


def _yd_call(fn: Callable[[], _T], *, timeout_sec: float = YD_CALL_TIMEOUT_SEC) -> _T:
    fut = _YD_POOL.submit(fn)
    try:
        return fut.result(timeout=timeout_sec)
    except FuturesTimeoutError as e:
        raise TimeoutError(f"YaDisk call timeout after {timeout_sec}s") from e


def _yd_call_retry(fn: Callable[[], _T]) -> _T:
    last: Optional[Exception] = None
    for attempt in range(YD_RETRIES + 1):
        try:
            return _yd_call(fn)
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < YD_RETRIES:
                time.sleep(YD_RETRY_DELAY_SEC * (attempt + 1))
                continue
            raise last


def _count_files_recursive(disk, root_path: str) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Рекурсивно считает количество файлов внутри папки (включая подпапки).
    Возвращает (count, error, error_path). Если ошибка — count=None.
    """
    root = _normalize_yadisk_path(root_path)
    count = 0
    stack = [root]
    visited: set[str] = set()

    try:
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            items = _yd_call_retry(lambda: list(disk.listdir(current)))
            for item in items:
                t = _get(item, "type")
                if t == "dir":
                    child_path = _get(item, "path")
                    if child_path:
                        stack.append(_normalize_yadisk_path(str(child_path)))
                elif t == "file":
                    count += 1
        return count, None, None
    except Exception as e:  # noqa: BLE001 — возвращаем в UI как текст
        # `current` помогает понять, на какой подпапке сломалось.
        err_path = locals().get("current")
        return None, f"{type(e).__name__}: {e}", (_as_disk_path(str(err_path)) if err_path else None)


def _looks_like_not_found(err: Optional[str]) -> bool:
    if not err:
        return False
    markers = [
        "DiskNotFoundError",
        "PathNotFoundError",
        "Resource not found",
        "Не удалось найти запрошенный ресурс",
        "Status code: 404",
    ]
    return any(m in err for m in markers)


def _resolve_first_level_folder_path(disk, *, base_dir: str, folder_name: str) -> Optional[str]:
    """
    Пытается найти папку `folder_name` в первом уровне `base_dir` (например, /Фото).
    Возвращает path (в формате yadisk, обычно 'disk:/...') или None.
    """
    base = _normalize_yadisk_path(base_dir)
    items = _yd_call_retry(lambda: list(disk.listdir(base)))
    for item in items:
        if _get(item, "type") != "dir":
            continue
        if str(_get(item, "name") or "") != folder_name:
            continue
        p = _get(item, "path")
        return str(p) if p else None
    return None


@app.get("/api/folders")
def api_folders() -> list[dict]:
    return list_folders(location="yadisk", role="target")


@app.get("/api/debug/module-path")
def api_debug_module_path() -> dict[str, Any]:
    """
    Диагностика: откуда реально импортирован модуль `app.main` и какой cwd/sys.path.
    Полезно, когда кажется, что uvicorn поднял "не тот" код.
    """
    return {
        "module": __name__,
        "module_file": __file__,
        "cwd": os.getcwd(),
        "sys_path_head": sys.path[:10],
    }


@app.get("/api/folder-counts")
def api_folder_counts() -> dict[str, dict[str, Any]]:
    """
    Возвращает словарь по code:
      { "<code>": {"count": int|null, "error": str|null} }

    Подсчёт файлов делаем рекурсивно через YaDisk API.
    """
    folders = list_folders(location="yadisk", role="target")
    disk = get_disk()

    result: dict[str, dict[str, Any]] = {}
    for f in folders:
        code = str(f.get("code") or "")
        path = str(f.get("path") or "")
        if not code or not path:
            continue
        cnt, err, err_path = _count_files_recursive(disk, path)
        result[code] = {"count": cnt, "error": err, "error_path": err_path}

    return result


@app.get("/api/folder-count/{code}")
def api_folder_count(code: str) -> dict[str, Any]:
    """
    Считает файлы рекурсивно для ОДНОЙ папки по её `code`.
    Возвращает:
      {"code": str, "count": int|null, "seconds": float|null, "error": str|null}
    """
    folders = list_folders(location="yadisk", role="target")
    folder = next((f for f in folders if f.get("code") == code), None)
    if not folder:
        return {"code": code, "count": None, "seconds": None, "error": "Folder not found"}

    disk = get_disk()

    t0 = time.perf_counter()
    path = str(folder.get("path") or "")
    name = str(folder.get("name") or "")

    cnt, err, err_path = _count_files_recursive(disk, path)

    # Если путь в БД устарел (папку переименовали/переместили/восстановили), попробуем
    # найти её заново по имени в /Фото и пересчитать.
    if cnt is None and _looks_like_not_found(err) and name:
        resolved = _resolve_first_level_folder_path(disk, base_dir="/Фото", folder_name=name)
        if resolved:
            cnt, err, err_path = _count_files_recursive(disk, str(resolved))

    dt = time.perf_counter() - t0
    return {"code": code, "count": cnt, "seconds": round(dt, 2), "error": err, "error_path": err_path}


@app.get("/api/folder-meta/{code}")
def api_folder_meta(code: str) -> dict[str, Any]:
    """
    Быстрая диагностика: метаданные папки через YaDisk API (get_meta).
    Пытаемся получить `embedded.total` (обычно это количество элементов 1-го уровня, НЕ рекурсивно).
    """
    folders = list_folders(location="yadisk", role="target")
    folder = next((f for f in folders if f.get("code") == code), None)
    if not folder:
        return {"code": code, "meta": None, "seconds": None, "error": "Folder not found"}

    disk = get_disk()
    path = str(folder.get("path") or "")
    p = _normalize_yadisk_path(path)

    t0 = time.perf_counter()
    try:
        meta = _yd_call_retry(lambda: disk.get_meta(p, limit=0))
        md = meta.to_json() if hasattr(meta, "to_json") else (meta if isinstance(meta, dict) else {})
        embedded = md.get("embedded") if isinstance(md, dict) else None
        embedded_total = embedded.get("total") if isinstance(embedded, dict) else None
        dt = time.perf_counter() - t0
        return {
            "code": code,
            "path": md.get("path") if isinstance(md, dict) else None,
            "name": md.get("name") if isinstance(md, dict) else None,
            "type": md.get("type") if isinstance(md, dict) else None,
            "size": md.get("size") if isinstance(md, dict) else None,
            "modified": md.get("modified") if isinstance(md, dict) else None,
            "embedded_total": embedded_total,
            "seconds": round(dt, 2),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return {"code": code, "meta": None, "seconds": round(dt, 2), "error": f"{type(e).__name__}: {e}"}


@app.get("/api/path-listing")
def api_path_listing(path: str) -> dict[str, Any]:
    """
    Листинг 1-го уровня для `path` (папки) + количество файлов, лежащих прямо в этой папке
    (без подпапок).
    """
    disk = get_disk()
    p = _normalize_yadisk_path(path)

    t0 = time.perf_counter()
    items = _yd_call_retry(lambda: list(disk.listdir(p)))
    seconds = round(time.perf_counter() - t0, 2)

    direct_files = 0
    dirs: list[dict[str, str]] = []
    for it in items:
        t = _get(it, "type")
        if t == "file":
            direct_files += 1
        elif t == "dir":
            name = str(_get(it, "name") or "")
            ipath = str(_get(it, "path") or "")
            if name and ipath:
                dirs.append({"name": name, "path": ipath})

    return {
        "path": _as_disk_path(p),
        "direct_files": direct_files,
        "dirs": sorted(dirs, key=lambda x: x["name"].lower()),
        "seconds": seconds,
        "error": None,
    }


@app.get("/api/path-count")
def api_path_count(path: str) -> dict[str, Any]:
    """
    Рекурсивный подсчёт файлов для произвольного пути (disk:/...).
    Возвращает:
      {"path": "...", "count": int|null, "seconds": float, "error": str|null, "error_path": str|null}
    """
    disk = get_disk()
    t0 = time.perf_counter()
    cnt, err, err_path = _count_files_recursive(disk, path)
    dt = time.perf_counter() - t0
    return {"path": path, "count": cnt, "seconds": round(dt, 2), "error": err, "error_path": err_path}


@app.get("/folders", response_class=HTMLResponse)
def folders_page(request: Request):
    folders = list_folders(location="yadisk", role="target")
    return templates.TemplateResponse(
        "folders.html",
        {
            "request": request,
            "folders": folders,
        },
    )


@app.get("/browse", response_class=HTMLResponse)
def browse_page(request: Request, path: str):
    return templates.TemplateResponse(
        "browse.html",
        {
            "request": request,
            "path": path,
        },
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )