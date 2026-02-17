from __future__ import annotations

import hashlib
import tempfile
import threading
import time
import os
import sys
import mimetypes
import urllib.parse
import subprocess
import json
from json import JSONDecodeError
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from fastapi import Body, FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from common.db import DedupStore, FaceStore, PipelineStore, get_connection, list_folders, init_db
from common.yadisk_client import get_disk
from web_api.routers.gold import router as gold_router
from web_api.routers.preclean import router as preclean_router
from web_api.routers.dedup_results import router as dedup_results_router
from web_api.routers.faces import _delete_from_all_gold_files, router as faces_router
from web_api.routers.local_pipeline import router as local_pipeline_router
from web_api.routers.folders import router as folders_router
from web_api.routers.face_clusters import router as face_clusters_router
from web_api.routers.trips import router as trips_router
from web_api.routers import gold as gold_api
from logic.gold.store import gold_expected_tab_by_path

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# --- dotenv (secrets.env/.env) ---
# ВАЖНО: uvicorn worker наследует окружение текущего процесса.
# Если не загрузить secrets.env здесь — LOCAL_PIPELINE_VIDEO_SAMPLES и прочие env не попадут в worker.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    rr0 = Path(__file__).resolve().parents[2]
    # приоритет: secrets.env, затем .env (без override)
    load_dotenv(dotenv_path=str(rr0 / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(rr0 / ".env"), override=False)
except Exception:
    pass

app = FastAPI(title="PhotoSorter")


@app.on_event("startup")
def _startup_init_db() -> None:
    """
    Инициализация БД при старте сервера: создаёт таблицы, если их нет (подъём системы с нуля).
    Для существующей БД только проверяет наличие files и выходит — новые колонки только через миграции.
    """
    init_db()
    # Сброс «зависшего» статуса шагов 5/6: после перезапуска uvicorn фоновый перенос убит,
    # но в БД мог остаться step4_phase=local|archive — сбрасываем, чтобы UI показывал idle и можно было нажать «Перенести в архив» снова (resume).
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("UPDATE pipeline_runs SET step4_phase = '' WHERE step4_phase IN ('local', 'archive')")
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # колонка step4_phase может отсутствовать в старых БД


# Favicon должен быть ДО mount static, чтобы не конфликтовать
@app.get("/favicon.ico")
def favicon():
    """Возвращает favicon для PhotoSorter (ICO, PNG или SVG)."""
    # Сначала пробуем ICO (стандартный формат)
    favicon_ico = APP_DIR / "static" / "favicon.ico"
    if favicon_ico.exists():
        return FileResponse(
            path=str(favicon_ico),
            media_type="image/x-icon",
            headers={"Cache-Control": "public, max-age=86400"}  # Кеш на 1 день
        )
    # Fallback на PNG
    favicon_png = APP_DIR / "static" / "favicon.png"
    if favicon_png.exists():
        return FileResponse(
            path=str(favicon_png),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"}  # Кеш на 1 день
        )
    # Fallback на SVG
    favicon_svg = APP_DIR / "static" / "favicon.svg"
    if favicon_svg.exists():
        return FileResponse(
            path=str(favicon_svg),
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"}  # Кеш на 1 день
        )
    return Response(status_code=204)  # No Content если файл не найден

# Подключаем статические файлы
static_dir = APP_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(gold_router)
app.include_router(preclean_router)
app.include_router(dedup_results_router)
app.include_router(faces_router)
app.include_router(local_pipeline_router)
app.include_router(folders_router)
app.include_router(face_clusters_router)
app.include_router(trips_router)

_T = TypeVar("_T")

# Build/runtime идентификатор процесса — чтобы быстро проверять, что запущена "правильная" версия.
# Генерируется один раз при импорте модуля и остаётся неизменным до перезапуска uvicorn.
_STARTED_AT = time.time()
STARTED_AT_UTC_ISO = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_STARTED_AT))
BUILD_ID = time.strftime("%Y%m%d-%H%M%S", time.gmtime(_STARTED_AT)) + f"-pid{os.getpid()}"

def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@app.middleware("http")
async def _add_build_id_header(request: Request, call_next):  # type: ignore[no-untyped-def]
    resp = await call_next(request)
    resp.headers["X-PhotoSorter-Build"] = BUILD_ID
    return resp

# Общий пул для вызовов YaDisk (нужно, чтобы иметь timeout на уровне приложения).
# Важно: timeout не убивает запрос внутри библиотеки, но предотвращает "вечные" зависания в обработчике.
_YD_POOL = ThreadPoolExecutor(max_workers=16)

# Отдельные одиночные исполнители для дедуп-сканов (чтобы не запускать несколько сканов одного типа параллельно).
_DEDUP_ARCHIVE_EXEC = ThreadPoolExecutor(max_workers=1)
_DEDUP_SOURCE_EXEC = ThreadPoolExecutor(max_workers=1)
_DEDUP_LOCK = threading.Lock()
_DEDUP_FUTURES: dict[str, Any] = {"archive": None, "source": None}
_DEDUP_RUN_IDS: dict[str, Optional[int]] = {"archive": None, "source": None}

# Сверка архива (актуализация списка файлов после редких ручных изменений в Я.Диске).
_RECONCILE_EXEC = ThreadPoolExecutor(max_workers=1)
_RECONCILE_LOCK = threading.Lock()
_RECONCILE_FUTURE: Any = None
_RECONCILE_RUN_ID: Optional[int] = None

# Таймауты/ретраи по умолчанию (можно будет вынести в настройки).
YD_CALL_TIMEOUT_SEC = 60
YD_RETRIES = 2
YD_RETRY_DELAY_SEC = 0.5

# Превью (proxy): защита от "забивания" сервера пачкой запросов со страницы /duplicates.
_PREVIEW_MAX_CONCURRENT = 4
_PREVIEW_SEM = threading.BoundedSemaphore(_PREVIEW_MAX_CONCURRENT)
# Важно: YaDisk preview URL обычно "временный". Держим небольшой TTL, чтобы ускорить повторы,
# но не полагаться на вечную валидность ссылки.
_PREVIEW_CACHE_TTL_SEC = 10 * 60  # 10 minutes
_PREVIEW_CACHE_MAX_ITEMS = 256
_PREVIEW_CACHE_LOCK = threading.Lock()
# key -> (ts, preview_url)
_PREVIEW_CACHE: "OrderedDict[tuple[str, str], tuple[float, str]]" = OrderedDict()

# Длительность видео (ffprobe): отдельный пул + защита от штормов.
_VIDEO_EXEC = ThreadPoolExecutor(max_workers=2)
_VIDEO_LOCK = threading.Lock()
_VIDEO_FUTURES: dict[str, Any] = {}
_VIDEO_ERRORS: dict[str, str] = {}

# (local pipeline endpoints moved to web_api/routers/local_pipeline.py)
def _preview_cache_get(key: tuple[str, str]) -> Optional[str]:
    now = time.time()
    with _PREVIEW_CACHE_LOCK:
        item = _PREVIEW_CACHE.get(key)
        if not item:
            return None
        ts, preview_url = item
        if now - ts > _PREVIEW_CACHE_TTL_SEC:
            # expired
            try:
                del _PREVIEW_CACHE[key]
            except KeyError:
                pass
            return None
        # LRU bump
        _PREVIEW_CACHE.move_to_end(key, last=True)
        return preview_url


def _preview_cache_put(key: tuple[str, str], preview_url: str) -> None:
    now = time.time()
    with _PREVIEW_CACHE_LOCK:
        _PREVIEW_CACHE[key] = (now, preview_url)
        _PREVIEW_CACHE.move_to_end(key, last=True)
        while len(_PREVIEW_CACHE) > _PREVIEW_CACHE_MAX_ITEMS:
            _PREVIEW_CACHE.popitem(last=False)


def _get(item: Any, key: str) -> Optional[Any]:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _normalize_yadisk_path(path: str) -> str:
    # В БД у нас может храниться `disk:/...`, но в yadisk-python удобно использовать `/...`.
    # Важно: НЕ используем .strip() — в именах папок могут быть значимые пробелы/символы,
    # и их "подчистка" ломает путь (приводит к 404 при listdir/get_meta).
    p = path or ""
    if p.startswith("disk:"):
        p = p[len("disk:") :]
    if not p.startswith("/"):
        p = "/" + p
    return p


def _as_disk_path(path: str) -> str:
    p = path or ""
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


def _yd_call_retry_timeout(fn: Callable[[], _T], *, timeout_sec: float) -> _T:
    """
    То же самое, что _yd_call_retry, но с настраиваемым timeout.
    Нужно, например, для скачивания файлов (оно может занимать минуты).
    """
    last: Optional[Exception] = None
    for attempt in range(YD_RETRIES + 1):
        try:
            return _yd_call(fn, timeout_sec=timeout_sec)
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


def _sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedup_scan_archive(
    *,
    run_id: int,
    root_paths: list[str],
    limit_files: int | None,
    max_download_bytes: int | None,
    inventory_scope: str,
) -> None:
    """
    Заполняет БД дублей: рекурсивно проходит папку на YaDisk,
    сохраняет метаданные файлов и хэш (sha256/md5). Если хэша нет — докачивает файл и считает sha256.
    """
    disk = get_disk()
    store = DedupStore()

    processed = 0
    hashed = 0
    meta_hashed = 0
    downloaded_hashed = 0
    skipped_large = 0
    errors = 0

    roots = [_normalize_yadisk_path(r) for r in (root_paths or []) if str(r or "").strip()]
    stack = list(roots)
    visited: set[str] = set()

    try:
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            try:
                items = _yd_call_retry(lambda: list(disk.listdir(current)))
            except Exception as e:  # noqa: BLE001
                errors += 1
                store.update_run_progress(
                    run_id=run_id,
                    errors_count=errors,
                    last_path=_as_disk_path(current),
                    last_error=f"{type(e).__name__}: {e}",
                )
                continue

            for item in items:
                t = _get(item, "type")
                if t == "dir":
                    child_path = _get(item, "path")
                    if child_path:
                        stack.append(_normalize_yadisk_path(str(child_path)))
                    continue

                if t != "file":
                    continue

                path = str(_get(item, "path") or "")
                if not path:
                    continue
                path = _as_disk_path(path)

                processed += 1
                resource_id = str(_get(item, "resource_id") or _get(item, "resourceId") or "") or None
                name = str(_get(item, "name") or "")
                size_val = _get(item, "size")
                size = int(size_val) if isinstance(size_val, (int, float)) else None
                created = str(_get(item, "created") or "") or None
                modified = str(_get(item, "modified") or "") or None
                mime_type = str(_get(item, "mime_type") or "") or None
                media_type = str(_get(item, "media_type") or "") or None

                # parent_path: disk:/Фото/..../<file> -> disk:/Фото/....
                parent_path: Optional[str] = None
                if path.startswith("disk:"):
                    p = path[len("disk:") :]
                    if "/" in p:
                        parent_path = "disk:" + p.rsplit("/", 1)[0]

                # Если уже есть хэш в БД — не перехешируем.
                existing_alg, existing_hash = store.get_existing_hash(path=path)
                if existing_alg and existing_hash:
                        store.upsert_file(
                        run_id=run_id,
                        path=path,
                        resource_id=resource_id,
                            inventory_scope=inventory_scope,
                        name=name or None,
                        parent_path=parent_path,
                        size=size,
                        created=created,
                        modified=modified,
                        mime_type=mime_type,
                        media_type=media_type,
                        hash_alg=None,
                        hash_value=None,
                        hash_source=None,
                        status="hashed",
                        error=None,
                        scanned_at=None,
                        hashed_at=None,
                    )
                else:
                    sha256 = str(_get(item, "sha256") or "") or None
                    md5 = str(_get(item, "md5") or "") or None

                    if sha256:
                        hashed += 1
                        meta_hashed += 1
                        store.upsert_file(
                            run_id=run_id,
                            path=path,
                            resource_id=resource_id,
                            inventory_scope=inventory_scope,
                            name=name or None,
                            parent_path=parent_path,
                            size=size,
                            created=created,
                            modified=modified,
                            mime_type=mime_type,
                            media_type=media_type,
                            hash_alg="sha256",
                            hash_value=sha256,
                            hash_source="meta",
                            status="hashed",
                            error=None,
                            scanned_at=None,
                            hashed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                    elif md5:
                        hashed += 1
                        meta_hashed += 1
                        store.upsert_file(
                            run_id=run_id,
                            path=path,
                            resource_id=resource_id,
                            inventory_scope=inventory_scope,
                            name=name or None,
                            parent_path=parent_path,
                            size=size,
                            created=created,
                            modified=modified,
                            mime_type=mime_type,
                            media_type=media_type,
                            hash_alg="md5",
                            hash_value=md5,
                            hash_source="meta",
                            status="hashed",
                            error=None,
                            scanned_at=None,
                            hashed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                    else:
                        # Нету хэша в метаданных -> докачиваем файл и считаем sha256.
                        if max_download_bytes is not None and size is not None and size > max_download_bytes:
                            skipped_large += 1
                            store.upsert_file(
                                run_id=run_id,
                                path=path,
                                resource_id=resource_id,
                                inventory_scope=inventory_scope,
                                name=name or None,
                                parent_path=parent_path,
                                size=size,
                                created=created,
                                modified=modified,
                                mime_type=mime_type,
                                media_type=media_type,
                                hash_alg=None,
                                hash_value=None,
                                hash_source=None,
                                status="skipped_large",
                                error=f"too_large: size={size} > max_download_bytes={max_download_bytes}",
                                scanned_at=None,
                                hashed_at=None,
                            )
                        else:
                            remote = _normalize_yadisk_path(path)
                            tmp_path = None
                            try:
                                with tempfile.NamedTemporaryFile(prefix="photosorter_dedup_", suffix=".bin", delete=False) as tmp:
                                    tmp_path = tmp.name

                                # На скачивание даём увеличенный таймаут (по умолчанию 10 минут).
                                _yd_call_retry_timeout(lambda: disk.download(remote, tmp_path), timeout_sec=600)

                                sha = _sha256_file(tmp_path)
                                hashed += 1
                                downloaded_hashed += 1
                                store.upsert_file(
                                    run_id=run_id,
                                    path=path,
                                    resource_id=resource_id,
                                    inventory_scope=inventory_scope,
                                    name=name or None,
                                    parent_path=parent_path,
                                    size=size,
                                    created=created,
                                    modified=modified,
                                    mime_type=mime_type,
                                    media_type=media_type,
                                    hash_alg="sha256",
                                    hash_value=sha,
                                    hash_source="download",
                                    status="hashed",
                                    error=None,
                                    scanned_at=None,
                                    hashed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                )
                            except Exception as e:  # noqa: BLE001
                                errors += 1
                                store.upsert_file(
                                    run_id=run_id,
                                    path=path,
                                    resource_id=resource_id,
                                    inventory_scope=inventory_scope,
                                    name=name or None,
                                    parent_path=parent_path,
                                    size=size,
                                    created=created,
                                    modified=modified,
                                    mime_type=mime_type,
                                    media_type=media_type,
                                    hash_alg=None,
                                    hash_value=None,
                                    hash_source=None,
                                    status="error",
                                    error=f"{type(e).__name__}: {e}",
                                    scanned_at=None,
                                    hashed_at=None,
                                )
                            finally:
                                if tmp_path:
                                    try:
                                        os.unlink(tmp_path)
                                    except OSError:
                                        pass

                store.update_run_progress(
                    run_id=run_id,
                    processed_files=processed,
                    hashed_files=hashed,
                    meta_hashed_files=meta_hashed,
                    downloaded_hashed_files=downloaded_hashed,
                    skipped_large_files=skipped_large,
                    errors_count=errors,
                    last_path=path,
                )

                if limit_files is not None and processed >= limit_files:
                    return
    finally:
        store.close()


def _dedup_archive_roots() -> list[str]:
    """
    Корни "архива" для сверки дублей (YaDisk). По умолчанию: Фото + Гитара.
    Настраивается через env `PHOTOSORTER_DEDUP_ARCHIVE_ROOTS`, разделители: ; , или перенос строки.
    """
    # По умолчанию оставляем прежнее поведение (только disk:/Фото).
    # Если нужно добавить другие "архивные" корни (например disk:/Гитара) — задайте env.
    raw = (os.getenv("PHOTOSORTER_DEDUP_ARCHIVE_ROOTS") or "disk:/Фото").strip()
    parts: list[str] = []
    for chunk in raw.replace("\r", "\n").replace(",", ";").split(";"):
        s = (chunk or "").strip()
        if not s:
            continue
        # Нормализуем trailing slash
        parts.append(s.rstrip("/"))
    # unique, preserve order
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out or ["disk:/Фото"]


def _as_local_path(p: str) -> str:
    # Храним локальные пути в общей таблице с префиксом, чтобы не конфликтовать с disk:/...
    return "local:" + str(p)

def _strip_local_prefix(p: str) -> str:
    return p[len("local:") :] if (p or "").startswith("local:") else p

def _local_is_under_root(*, file_path: str, root_dir: str) -> bool:
    try:
        file_abs = os.path.abspath(file_path)
        root_abs = os.path.abspath(root_dir)
        return os.path.commonpath([file_abs, root_abs]) == root_abs
    except Exception:
        return False


def _dedup_scan_local_source(*, run_id: int, root_dir: str) -> None:
    """
    Заполняет БД дублей для локальной папки-источника (например C:\\tmp\\Photo).
    Всегда считаем sha256 локально.
    """
    store = DedupStore()
    processed = 0
    hashed = 0
    errors = 0

    root = os.path.abspath(root_dir)

    # Считаем total_files для процентов.
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        total += len(filenames)
    store.update_run_progress(run_id=run_id, total_files=total)

    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                abspath = os.path.join(dirpath, fn)
                processed += 1

                db_path = _as_local_path(abspath)
                parent_path = _as_local_path(dirpath)
                mime_type, _enc = mimetypes.guess_type(abspath)
                media_type: Optional[str] = None
                if mime_type:
                    if mime_type.startswith("image/"):
                        media_type = "image"
                    elif mime_type.startswith("video/"):
                        media_type = "video"

                size: Optional[int] = None
                modified: Optional[str] = None
                try:
                    st = os.stat(abspath)
                    size = int(st.st_size)
                    modified = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
                except OSError:
                    pass

                existing_alg, existing_hash = store.get_existing_hash(path=db_path)
                if existing_alg and existing_hash:
                    hashed += 1
                    store.upsert_file(
                        run_id=run_id,
                        path=db_path,
                        inventory_scope="source",
                        name=fn,
                        parent_path=parent_path,
                        size=size,
                        created=None,
                        modified=modified,
                        mime_type=mime_type,
                        media_type=media_type,
                        hash_alg=None,
                        hash_value=None,
                        hash_source=None,
                        status="hashed",
                        error=None,
                        scanned_at=None,
                        hashed_at=None,
                    )
                else:
                    try:
                        sha = _sha256_file(abspath)
                        hashed += 1
                        store.upsert_file(
                            run_id=run_id,
                            path=db_path,
                            inventory_scope="source",
                            name=fn,
                            parent_path=parent_path,
                            size=size,
                            created=None,
                            modified=modified,
                            mime_type=mime_type,
                            media_type=media_type,
                            hash_alg="sha256",
                            hash_value=sha,
                            hash_source="local",
                            status="hashed",
                            error=None,
                            scanned_at=None,
                            hashed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                    except Exception as e:  # noqa: BLE001
                        errors += 1
                        store.upsert_file(
                            run_id=run_id,
                            path=db_path,
                            inventory_scope="source",
                            name=fn,
                            parent_path=parent_path,
                            size=size,
                            created=None,
                            modified=modified,
                            mime_type=mime_type,
                            media_type=media_type,
                            hash_alg=None,
                            hash_value=None,
                            hash_source=None,
                            status="error",
                            error=f"{type(e).__name__}: {e}",
                            scanned_at=None,
                            hashed_at=None,
                        )

                store.update_run_progress(
                    run_id=run_id,
                    processed_files=processed,
                    hashed_files=hashed,
                    errors_count=errors,
                    last_path=db_path,
                )
    finally:
        store.close()


def _human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "—"
    x = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    if i == 0:
        return f"{int(x)} {units[i]}"
    return f"{x:.1f} {units[i]}"


def _short_folder_from_disk_path(path: str) -> str:
    # disk:/Фото/<Top>/<...>/<file>
    p = path or ""
    if not p.startswith("disk:/"):
        return "—"
    parts = p.split("/")
    # parts: ["disk:", "Фото", "<Top>", ...]
    if len(parts) >= 3:
        return parts[2] or "—"
    return "—"


def _short_path_for_ui(path: str) -> str:
    # Показываем путь короче: обрезаем disk:/Фото/ и оставляем хвост.
    p = path or ""
    prefix = "disk:/Фото/"
    if p.startswith(prefix):
        tail = p[len(prefix) :]
        return "…/" + tail
    return p


def _basename_from_disk_path(path: str) -> str:
    p = path or ""
    if "/" not in p:
        return p
    return p.rsplit("/", 1)[-1]


def _parent_from_disk_path(path: str) -> Optional[str]:
    p = path or ""
    if not p.startswith("disk:"):
        return None
    tail = p[len("disk:") :]
    if "/" not in tail:
        return None
    return "disk:" + tail.rsplit("/", 1)[0]


def _disk_join(dir_path: str, name: str) -> str:
    d = (dir_path or "").rstrip("/")
    return d + "/" + (name or "").lstrip("/")


def _yadisk_web_url(path: str) -> str:
    """
    Открывает файл/папку в веб-интерфейсе Я.Диска (в браузере).
    """
    # disk:/Фото/... -> /Фото/...
    p = _normalize_yadisk_path(path)
    encoded = urllib.parse.quote(p, safe="/")
    return "https://disk.yandex.ru/client/disk" + encoded


def _yadisk_slider_url(path: str) -> str:
    """
    Открывает КОНКРЕТНЫЙ файл в веб-интерфейсе Я.Диска через "slider" (просмотр),
    чтобы открывался именно файл, а не только папка.

    Пример формата:
      https://disk.yandex.ru/client/disk/<DIR>?idApp=client&dialog=slider&idDialog=%2Fdisk%2F<DIR>%2F<FILE>
    """
    # disk:/Фото/... -> /Фото/... (для /disk/... в idDialog)
    p = _normalize_yadisk_path(path)
    # Базовый URL берём на папку (без имени файла)
    dir_path = p.rsplit("/", 1)[0] if "/" in p else p
    base = "https://disk.yandex.ru/client/disk" + urllib.parse.quote(dir_path, safe="/")
    # idDialog ожидает абсолютный путь вида /disk/Фото/.../file.ext, с экранированием '/' как %2F
    id_dialog = urllib.parse.quote("/disk" + p, safe="")
    return f"{base}?idApp=client&dialog=slider&idDialog={id_dialog}"


def _resolve_target_folder_path_kids_together() -> str:
    """
    Возвращает путь до папки 'Дети вместе' в формате disk:/...
    Пытаемся взять из таблицы folders (target yadisk), иначе fallback.
    """
    folders = list_folders(location="yadisk", role="target")
    for f in folders:
        code = str(f.get("code") or "").lower()
        name = str(f.get("name") or "").lower()
        if "deti_vmeste" in code or name == "дети вместе":
            p = str(f.get("path") or "")
            if p:
                return p
    return "disk:/Фото/Дети вместе"


def _unique_dest_name(*, store: DedupStore, disk, dest_dir: str, src_name: str) -> str:
    """
    Подбирает имя в dest_dir так, чтобы не конфликтовать ни с YaDisk, ни с локальной БД.
    """
    base = src_name or "file"
    if "." in base:
        stem, ext = base.rsplit(".", 1)
        ext = "." + ext
    else:
        stem, ext = base, ""

    for n in range(0, 200):
        candidate = base if n == 0 else f"{stem} ({n}){ext}"
        dest_disk = _disk_join(dest_dir, candidate)
        # Проверяем и на стороне Я.Диска, и в БД (чтобы избежать UNIQUE конфликтов).
        exists_remote = False
        try:
            exists_remote = bool(_yd_call_retry(lambda: disk.exists(_normalize_yadisk_path(dest_disk))))
        except Exception:
            exists_remote = False
        if exists_remote:
            continue
        if store.path_exists(path=dest_disk):
            continue
        return candidate
    raise RuntimeError("Не удалось подобрать уникальное имя (слишком много конфликтов)")


def _human_duration(seconds: Optional[int]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _extract_duration_seconds(md: Any) -> Optional[int]:
    """
    Best-effort: пытаемся достать длительность видео из метаданных YaDisk.
    Форматы/поля отличаются между версиями/типами ресурсов.
    """
    # Частые варианты: md.media_info.duration, md.duration, md.video_duration, dict['duration']
    candidates: list[Any] = []
    for attr in ("duration", "video_duration", "duration_sec", "duration_seconds", "duration_ms"):
        candidates.append(getattr(md, attr, None))
        candidates.append(_get(md, attr))

    mi = getattr(md, "media_info", None)
    if mi is None:
        mi = _get(md, "media_info")
    if mi is not None:
        for attr in ("duration", "video_duration", "duration_sec", "duration_seconds", "duration_ms"):
            candidates.append(getattr(mi, attr, None))
            candidates.append(_get(mi, attr))

    for v in candidates:
        if v is None or v == "":
            continue
        try:
            if isinstance(v, str):
                # иногда приходит строка числа
                v2 = float(v)
            else:
                v2 = float(v)
        except Exception:
            continue
        # если похоже на миллисекунды
        if v2 > 10_000:  # 10k seconds ~ 2.7h, типично видео короче; ms будут >> 10k
            # эвристика: ms
            sec = int(round(v2 / 1000.0))
        else:
            sec = int(round(v2))
        if sec >= 0:
            return sec
    return None


def _ffprobe_path() -> str:
    """
    ffprobe должен быть либо в PATH, либо задан в env `FFPROBE_PATH` (secrets.env/.env).
    """
    return os.getenv("FFPROBE_PATH") or "ffprobe"


def _ffprobe_duration_seconds_from_url(url: str, *, timeout_sec: float = 30) -> Optional[int]:
    """
    Запускает ffprobe на URL и пытается извлечь длительность (секунды).
    """
    # ffprobe -v error -of json -show_entries format=duration <url>
    cmd = [
        _ffprobe_path(),
        "-v",
        "error",
        "-of",
        "json",
        "-show_entries",
        "format=duration",
        url,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)  # noqa: S603,S607
    except FileNotFoundError as e:
        raise RuntimeError("ffprobe не найден. Установите ffmpeg/ffprobe или задайте FFPROBE_PATH в secrets.env/.env") from e
    except subprocess.TimeoutExpired:
        return None

    if p.returncode != 0:
        # иногда ffprobe пишет в stderr; не считаем это фатальным, просто нет длительности
        return None
    try:
        obj = json.loads(p.stdout or "{}")
    except Exception:
        return None
    dur = (((obj or {}).get("format") or {}).get("duration"))
    try:
        if dur is None:
            return None
        sec = int(round(float(dur)))
        return sec if sec >= 0 else None
    except Exception:
        return None


def _get_download_url(disk, *, path: str) -> str:
    """
    Получает прямую ссылку скачивания для ресурса YaDisk.
    """
    p = _normalize_yadisk_path(path)
    if hasattr(disk, "get_download_link"):
        return str(_yd_call_retry(lambda: disk.get_download_link(p)))
    # fallback: некоторые обёртки могут отдавать link иначе
    raise RuntimeError("YaDisk client не поддерживает get_download_link()")


def _reconcile_archive_inventory(*, run_id: int, root_path: str) -> None:
    """
    Актуализация инвентаря архива (disk:/Фото) после редких ручных изменений на Я.Диске.

    Делает:
    - рекурсивный обход файлов под root_path
    - upsert/обновление метаданных по resource_id (если есть) или path
    - помечает отсутствующие в скане записи как status='deleted'
    - если size/modified изменились — сбрасывает hash_* (нужно пересчитать)
    """
    disk = get_disk()
    store = DedupStore()

    processed = 0
    errors = 0

    root = _normalize_yadisk_path(root_path)
    stack = [root]
    visited: set[str] = set()
    seen_resource_ids: set[str] = set()
    seen_paths: set[str] = set()

    try:
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            try:
                items = _yd_call_retry(lambda: list(disk.listdir(current)))
            except Exception as e:  # noqa: BLE001
                errors += 1
                store.update_run_progress(
                    run_id=run_id,
                    errors_count=errors,
                    last_path=_as_disk_path(current),
                    last_error=f"{type(e).__name__}: {e}",
                )
                continue

            for item in items:
                t = _get(item, "type")
                if t == "dir":
                    child_path = _get(item, "path")
                    if child_path:
                        stack.append(_normalize_yadisk_path(str(child_path)))
                    continue
                if t != "file":
                    continue

                path = str(_get(item, "path") or "")
                if not path:
                    continue

                processed += 1
                resource_id = str(_get(item, "resource_id") or _get(item, "resourceId") or "") or None
                if resource_id:
                    seen_resource_ids.add(resource_id)
                seen_paths.add(path)

                name = str(_get(item, "name") or "") or None
                size_val = _get(item, "size")
                size = int(size_val) if isinstance(size_val, (int, float)) else None
                created = str(_get(item, "created") or "") or None
                modified = str(_get(item, "modified") or "") or None
                mime_type = str(_get(item, "mime_type") or "") or None
                media_type = str(_get(item, "media_type") or "") or None

                parent_path: Optional[str] = None
                if path.startswith("disk:"):
                    p = path[len("disk:") :]
                    if "/" in p:
                        parent_path = "disk:" + p.rsplit("/", 1)[0]

                try:
                    store.reconcile_upsert_present_file(
                        run_id=run_id,
                        path=path,
                        resource_id=resource_id,
                        inventory_scope="archive",
                        name=name,
                        parent_path=parent_path,
                        size=size,
                        created=created,
                        modified=modified,
                        mime_type=mime_type,
                        media_type=media_type,
                    )
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    store.update_run_progress(
                        run_id=run_id,
                        errors_count=errors,
                        last_path=path,
                        last_error=f"upsert_failed: {type(e).__name__}: {e}",
                    )

                # прогресс
                if processed % 200 == 0:
                    store.update_run_progress(
                        run_id=run_id,
                        processed_files=processed,
                        errors_count=errors,
                        last_path=path,
                    )

        # Финальный прогресс
        store.update_run_progress(
            run_id=run_id,
            processed_files=processed,
            errors_count=errors,
            last_path=_as_disk_path(root),
        )

        # Помечаем отсутствующие (под root_path) как deleted.
        prefix = (root_path.rstrip("/") + "/%") if root_path else "disk:/Фото/%"
        cur = store.conn.cursor()  # type: ignore[attr-defined]
        cur.execute(
            """
            SELECT path, resource_id
            FROM files
            WHERE path LIKE ? AND status != 'deleted'
            """,
            (prefix,),
        )
        rows = cur.fetchall()
        missing: list[str] = []
        for r in rows:
            p = str(r["path"] or "")
            rid = str(r["resource_id"] or "") or None
            if rid:
                if rid not in seen_resource_ids:
                    missing.append(p)
            else:
                if p not in seen_paths:
                    missing.append(p)

        # UPDATE батчами (чтобы не делать IN на десятки тысяч).
        for i in range(0, len(missing), 500):
            store.mark_deleted(paths=missing[i : i + 500])

    finally:
        store.close()


def _pick_keep_indexes(items: list[dict[str, Any]], sort_order_by_folder: dict[str, int]) -> int:
    """
    Выбираем индекс элемента, который "оставляем" по умолчанию.
    1) минимальный sort_order по top-level папке (Темка/Нюся/...), если известен
    2) fallback: самый короткий path
    """
    best_idx = 0
    best_order: Optional[int] = None
    best_len: Optional[int] = None
    for i, it in enumerate(items):
        p = str(it.get("path") or "")
        folder = _short_folder_from_disk_path(p)
        order = sort_order_by_folder.get(folder.lower())
        plen = len(p)
        if order is not None:
            if best_order is None or order < best_order or (order == best_order and plen < (best_len or plen + 1)):
                best_idx = i
                best_order = order
                best_len = plen
        else:
            if best_order is None:  # пока нет кандидата с sort_order
                if best_len is None or plen < best_len:
                    best_idx = i
                    best_len = plen
    return best_idx


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


@app.get("/api/debug/build-info")
def api_debug_build_info() -> dict[str, Any]:
    """
    Диагностика: build/runtime информация о запущенном процессе.
    Полезно, чтобы быстро понять, что запущена именно текущая версия кода.
    """
    return {
        "ok": True,
        "build_id": BUILD_ID,
        "started_at_utc": STARTED_AT_UTC_ISO,
        "pid": os.getpid(),
        "module_file": __file__,
        "cwd": os.getcwd(),
    }


@app.get("/api/yadisk/open")
def api_yadisk_open(path: str) -> Response:
    """
    Redirect на веб-интерфейс Яндекс.Диска для указанного disk:/... пути.
    """
    if not path.startswith("disk:"):
        raise HTTPException(status_code=400, detail="Only disk: paths are supported")
    # Открываем файл в "slider" просмотре (как в веб-интерфейсе Я.Диска).
    return RedirectResponse(url=_yadisk_slider_url(path), status_code=307)


@app.post("/api/dedup/archive/start")
def api_dedup_archive_start(max_download_gb: int = 5) -> dict[str, Any]:
    """
    Запускает фоновый скан архива (disk:/Фото) для заполнения БД дублей.
    Упрощение: всегда "полный" прогон, без лимита; можно запускать повторно (resume по данным).
    """
    global _DEDUP_FUTURES, _DEDUP_RUN_IDS  # noqa: PLW0603

    roots = _dedup_archive_roots()
    # root_path сохраняем совместимым (первый корень), чтобы UI/старые куски кода не ломались.
    root_path = str(roots[0] if roots else "disk:/Фото")
    max_download_bytes = None if max_download_gb <= 0 else (max_download_gb * 1024 * 1024 * 1024)

    with _DEDUP_LOCK:
        fut = _DEDUP_FUTURES.get("archive")
        if fut is not None and not fut.done():
            return {"ok": False, "message": "dedup scan already running", "run_id": _DEDUP_RUN_IDS.get("archive")}

        store = DedupStore()
        try:
            run_id = store.create_run(
                scope="archive",
                root_path=root_path,
                max_download_bytes=max_download_bytes,
            )
        finally:
            store.close()

        _DEDUP_RUN_IDS["archive"] = run_id

        def _runner() -> None:
            store2 = DedupStore()
            try:
                # Всегда считаем total_files, чтобы UI мог рисовать процент.
                try:
                    disk = get_disk()
                    total_sum = 0
                    any_total = False
                    for r in roots:
                        total, err, err_path = _count_files_recursive(disk, r)
                        if total is not None:
                            total_sum += int(total)
                            any_total = True
                        else:
                            store2.update_run_progress(
                                run_id=run_id,
                                last_path=err_path,
                                last_error=f"total_files_count_failed({r}): {err}",
                            )
                    if any_total:
                        store2.update_run_progress(run_id=run_id, total_files=int(total_sum))
                except Exception as e:  # noqa: BLE001
                    store2.update_run_progress(
                        run_id=run_id,
                        last_error=f"total_files_count_failed: {type(e).__name__}: {e}",
                    )

                _dedup_scan_archive(
                    run_id=run_id,
                    root_paths=roots,
                    limit_files=None,
                    max_download_bytes=max_download_bytes,
                    inventory_scope="archive",
                )
                store2.finish_run(run_id=run_id, status="completed")
            except Exception as e:  # noqa: BLE001
                store2.finish_run(run_id=run_id, status="failed", last_error=f"{type(e).__name__}: {e}")
            finally:
                store2.close()

        _DEDUP_FUTURES["archive"] = _DEDUP_ARCHIVE_EXEC.submit(_runner)

    return {"ok": True, "message": "started", "run_id": run_id, "root_path": root_path}


@app.post("/api/archive/reconcile/start")
def api_archive_reconcile_start() -> dict[str, Any]:
    """
    Запускает фоновую сверку архива (актуализация списка файлов в files),
    чтобы "догонять" редкие ручные изменения в веб-интерфейсе Я.Диска.
    """
    global _RECONCILE_FUTURE, _RECONCILE_RUN_ID  # noqa: PLW0603

    root_path = "disk:/Фото"

    with _RECONCILE_LOCK:
        if _RECONCILE_FUTURE is not None and not _RECONCILE_FUTURE.done():
            return {"ok": False, "message": "reconcile already running", "run_id": _RECONCILE_RUN_ID}

        store = DedupStore()
        try:
            run_id = store.create_run(scope="archive_reconcile", root_path=root_path, max_download_bytes=None)
        finally:
            store.close()

        _RECONCILE_RUN_ID = run_id

        def _runner() -> None:
            store2 = DedupStore()
            try:
                # total_files считаем ПАРАЛЛЕЛЬНО (иначе старт сверки может долго "висеть" на 0%).
                def _count_total() -> None:
                    s = DedupStore()
                    try:
                        disk = get_disk()
                        total, err, err_path = _count_files_recursive(disk, root_path)
                        if total is not None:
                            s.update_run_progress(run_id=run_id, total_files=int(total))
                        else:
                            s.update_run_progress(
                                run_id=run_id,
                                last_path=err_path,
                                last_error=f"total_files_count_failed: {err}",
                            )
                    except Exception as e:  # noqa: BLE001
                        s.update_run_progress(run_id=run_id, last_error=f"total_files_count_failed: {type(e).__name__}: {e}")
                    finally:
                        s.close()

                threading.Thread(target=_count_total, daemon=True).start()

                _reconcile_archive_inventory(run_id=run_id, root_path=root_path)
                store2.finish_run(run_id=run_id, status="completed")
                
                # После завершения reconcile запускаем досчёт лиц для архива (в фоне)
                def _scan_faces_runner() -> None:
                    try:
                        repo_root = Path(__file__).resolve().parents[2]
                        script_path = repo_root / "backend" / "scripts" / "tools" / "face_scan_archive.py"
                        venv_python = repo_root / ".venv-face" / "Scripts" / "python.exe"
                        
                        if not script_path.exists():
                            return  # Скрипт не найден - пропускаем
                        
                        if venv_python.exists():
                            subprocess.run(
                                [str(venv_python), str(script_path)],
                                cwd=str(repo_root),
                                timeout=None,  # Может быть долго
                            )
                        else:
                            # Fallback на системный Python
                            subprocess.run(
                                [sys.executable, str(script_path)],
                                cwd=str(repo_root),
                                timeout=None,
                            )
                    except Exception as e:  # noqa: BLE001
                        # Логируем, но не блокируем reconcile
                        print(f"face_scan_archive failed: {type(e).__name__}: {e}", file=sys.stderr)
                
                # Запускаем в отдельном потоке, чтобы не блокировать
                threading.Thread(target=_scan_faces_runner, daemon=True).start()
            except Exception as e:  # noqa: BLE001
                store2.finish_run(run_id=run_id, status="failed", last_error=f"{type(e).__name__}: {e}")
            finally:
                store2.close()

        _RECONCILE_FUTURE = _RECONCILE_EXEC.submit(_runner)

    return {"ok": True, "message": "started", "run_id": run_id, "root_path": root_path}


@app.get("/api/archive/reconcile/status")
def api_archive_reconcile_status() -> dict[str, Any]:
    global _RECONCILE_FUTURE, _RECONCILE_RUN_ID  # noqa: PLW0603

    store = DedupStore()
    try:
        latest = store.get_latest_run(scope="archive_reconcile")
    finally:
        store.close()

    running = False
    with _RECONCILE_LOCK:
        running = _RECONCILE_FUTURE is not None and not _RECONCILE_FUTURE.done()

    return {"running": running, "active_run_id": _RECONCILE_RUN_ID, "latest": latest}


@app.post("/api/archive/backfill-manual-embeddings")
def api_archive_backfill_manual_embeddings() -> dict[str, Any]:
    """
    Запускает backfill embeddings и кластеров для ручных привязок в архиве.
    Обрабатывает photo_rectangles с manual_person_id и без embedding/cluster_id
    (например, нарисованные вручную после предыдущего backfill).
    Запускает скрипт в .venv-face (ML-окружение). Блокирует до 5 минут.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "backend" / "scripts" / "tools" / "backfill_archive_manual_embeddings.py"
    venv_python = repo_root / ".venv-face" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    if not script_path.exists():
        return {"ok": False, "message": "Скрипт backfill не найден"}

    try:
        result = subprocess.run(
            [python_exe, str(script_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        last_lines = "\n".join(out.splitlines()[-5:]) if out else ""

        if result.returncode == 0:
            return {"ok": True, "message": last_lines or "Готово"}
        return {"ok": False, "message": err or last_lines or f"Код выхода {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Превышено время ожидания (5 мин)"}
    except Exception as e:
        return {"ok": False, "message": f"{type(e).__name__}: {e}"}


@app.get("/api/dedup/archive/status")
def api_dedup_archive_status() -> dict[str, Any]:
    global _DEDUP_FUTURES, _DEDUP_RUN_IDS  # noqa: PLW0603

    store = DedupStore()
    try:
        latest = store.get_latest_run(scope="archive")
    finally:
        store.close()

    running = False
    with _DEDUP_LOCK:
        fut = _DEDUP_FUTURES.get("archive")
        running = fut is not None and not fut.done()

    return {"running": running, "active_run_id": _DEDUP_RUN_IDS.get("archive"), "latest": latest}


@app.post("/api/dedup/source/start")
def api_dedup_source_start(path: str = r"C:\tmp\Photo", max_download_gb: int = 5) -> dict[str, Any]:
    """
    Запускает/продолжает дедупликацию папки-источника (локальная или YaDisk).
    - локальная: считаем sha256 локально
    - YaDisk: используем sha256/md5 из метаданных, иначе (опционально) докачиваем и считаем sha256

    Важно: для YaDisk source запрещаем выбирать папку внутри корней архива, чтобы не смешивать "archive" и "source"
    (в таблице files путь уникален).
    """
    global _DEDUP_FUTURES, _DEDUP_RUN_IDS  # noqa: PLW0603

    root_path = path
    is_disk = isinstance(root_path, str) and root_path.startswith("disk:")
    if is_disk:
        # защитимся от пересечения с архивом
        p = root_path.rstrip("/")
        arc_roots = _dedup_archive_roots()
        def _is_under_any_archive_root(pp: str) -> bool:
            s = (pp or "").rstrip("/")
            for r in arc_roots:
                rr = (r or "").rstrip("/")
                if not rr:
                    continue
                if s == rr or s.startswith(rr + "/"):
                    return True
            return False
        if _is_under_any_archive_root(p):
            raise HTTPException(
                status_code=400,
                detail="Нельзя выбирать source внутри архива (например disk:/Фото или disk:/Гитара). Выберите папку вне архива, например disk:/Загрузки или disk:/Фотокамера.",
            )

    max_download_bytes = None if max_download_gb <= 0 else (max_download_gb * 1024 * 1024 * 1024)
    with _DEDUP_LOCK:
        fut = _DEDUP_FUTURES.get("source")
        if fut is not None and not fut.done():
            return {"ok": False, "message": "dedup scan already running", "run_id": _DEDUP_RUN_IDS.get("source")}

        store = DedupStore()
        try:
            run_id = store.create_run(scope="source", root_path=root_path, max_download_bytes=(max_download_bytes if is_disk else None))
        finally:
            store.close()

        _DEDUP_RUN_IDS["source"] = run_id

        def _runner() -> None:
            store2 = DedupStore()
            try:
                if is_disk:
                    # total_files считаем, чтобы UI мог показывать прогресс (как в archive)
                    try:
                        disk = get_disk()
                        total, err, err_path = _count_files_recursive(disk, root_path)
                        if total is not None:
                            store2.update_run_progress(run_id=run_id, total_files=int(total))
                        else:
                            store2.update_run_progress(
                                run_id=run_id,
                                last_path=err_path,
                                last_error=f"total_files_count_failed: {err}",
                            )
                    except Exception as e:  # noqa: BLE001
                        store2.update_run_progress(
                            run_id=run_id,
                            last_error=f"total_files_count_failed: {type(e).__name__}: {e}",
                        )

                    _dedup_scan_archive(
                        run_id=run_id,
                        root_paths=[root_path],
                        limit_files=None,
                        max_download_bytes=max_download_bytes,
                        inventory_scope="source",
                    )
                else:
                    _dedup_scan_local_source(run_id=run_id, root_dir=root_path)
                store2.finish_run(run_id=run_id, status="completed")
            except Exception as e:  # noqa: BLE001
                store2.finish_run(run_id=run_id, status="failed", last_error=f"{type(e).__name__}: {e}")
            finally:
                store2.close()

        _DEDUP_FUTURES["source"] = _DEDUP_SOURCE_EXEC.submit(_runner)

    return {"ok": True, "message": "started", "run_id": run_id, "root_path": root_path}


@app.get("/api/dedup/source/status")
def api_dedup_source_status() -> dict[str, Any]:
    global _DEDUP_FUTURES, _DEDUP_RUN_IDS  # noqa: PLW0603

    store = DedupStore()
    try:
        latest = store.get_latest_run(scope="source")
    finally:
        store.close()

    running = False
    with _DEDUP_LOCK:
        fut = _DEDUP_FUTURES.get("source")
        running = fut is not None and not fut.done()

    return {"running": running, "active_run_id": _DEDUP_RUN_IDS.get("source"), "latest": latest}


@app.post("/api/sort/start")
def api_sort_start(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Единый старт для "рабочей формы сортировки" (главная страница):
    - запускаем дедуп-скан source (YaDisk или local)
    - при первом запуске (если archive ещё не сканировался) — стартуем archive scan
    """
    location = str(payload.get("location") or "").lower()
    path = str(payload.get("path") or "")
    max_download_gb = payload.get("max_download_gb")
    try:
        max_download_gb_i = int(max_download_gb) if max_download_gb is not None else 5
    except Exception:
        max_download_gb_i = 5

    if location not in ("yadisk", "local"):
        raise HTTPException(status_code=400, detail="location must be 'yadisk' or 'local'")
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    if location == "yadisk" and not path.startswith("disk:"):
        raise HTTPException(status_code=400, detail="Для YaDisk путь должен начинаться с 'disk:/'")
    if location == "local" and path.startswith("disk:"):
        raise HTTPException(status_code=400, detail="Для локальной папки укажите путь вида C:\\... (без disk:)")

    # 1) source
    src = api_dedup_source_start(path=path, max_download_gb=max_download_gb_i)

    # 2) archive (если ещё не запускали ни разу)
    store = DedupStore()
    try:
        archive_latest = store.get_latest_run(scope="archive")
    finally:
        store.close()
    archive_started = False
    archive_start_resp: dict[str, Any] | None = None
    if archive_latest is None:
        archive_start_resp = api_dedup_archive_start(max_download_gb=5)
        archive_started = bool(archive_start_resp.get("ok"))

    return {
        "ok": True,
        "source": src,
        "archive_started": archive_started,
        "archive_start": archive_start_resp,
    }


@app.post("/api/_legacy/local-pipeline/start")
def api_local_pipeline_start(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Запуск локального конвейера (ML в отдельном процессе через .venv-face).
    Запускает scripts/tools/local_sort_by_faces.py напрямую через python из .venv-face
    (без .ps1), чтобы не упираться в ExecutionPolicy/подписи PowerShell-скриптов.
    """
    raise HTTPException(status_code=410, detail="Moved to web_api/routers/local_pipeline.py")
    root_path = str(payload.get("root_path") or "").strip()
    apply = bool(payload.get("apply") or False)
    skip_dedup = bool(payload.get("skip_dedup") or False)
    start_mode = str(payload.get("start_mode") or "").strip().lower()  # 'new'|'continue'|''
    resume_run_id = payload.get("resume_run_id")
    # Решение по UX: дубликаты всегда складываем в _duplicates; удаление делаем осознанно из формы дублей.
    no_dedup_move = False

    if not root_path:
        raise HTTPException(status_code=400, detail="root_path is required")
    if root_path.startswith("disk:"):
        raise HTTPException(status_code=400, detail="Only local folders are supported here (use C:\\... path)")
    if not os.path.isdir(root_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {root_path}")

    rr = _repo_root()
    py = rr / ".venv-face" / "Scripts" / "python.exe"
    script = rr / "backend" / "scripts" / "tools" / "local_sort_by_faces.py"
    if not py.exists():
        raise HTTPException(status_code=500, detail="Missing .venv-face (Python 3.12) — create it before running ML pipeline")
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"Missing: {script}")

    # Создаём/переиспользуем pipeline_run_id, чтобы поддержать resume после рестарта.
    ps = PipelineStore()
    try:
        latest = ps.get_latest_run(kind="local_sort", root_path=root_path)
        resumed = False
        prev_run_id: int | None = None
        prev_face_run_id: int | None = None
        prev_root_like: str | None = None
        prev_status: str | None = None
        # candidate previous run (only for "new" starts)
        if latest:
            try:
                prev_run_id = int(latest.get("id") or 0) or None
            except Exception:
                prev_run_id = None
            try:
                prev_face_run_id = int(latest.get("face_run_id") or 0) or None
            except Exception:
                prev_face_run_id = None
            prev_status = str(latest.get("status") or "")
            try:
                prev_root_like = gold_api._root_like_for_pipeline_run_id(int(prev_run_id)) if prev_run_id else None
            except Exception:
                prev_root_like = None
        # 0) Явный resume конкретного run_id (UI кнопка "Продолжить")
        if resume_run_id is not None:
            try:
                rid = int(resume_run_id)
            except Exception:
                raise HTTPException(status_code=400, detail="resume_run_id must be int") from None
            pr0 = ps.get_run_by_id(run_id=rid)
            if not pr0:
                raise HTTPException(status_code=404, detail="resume_run_id not found")
            if str(pr0.get("kind") or "") != "local_sort":
                raise HTTPException(status_code=400, detail="resume_run_id kind mismatch")
            if str(pr0.get("root_path") or "") != root_path:
                raise HTTPException(status_code=400, detail="resume_run_id root_path mismatch")
            st0 = str(pr0.get("status") or "")
            if st0 not in ("failed", "running"):
                raise HTTPException(status_code=400, detail=f"resume_run_id not resumable (status={st0})")
            # безопасность: опции должны совпадать с тем, что было у прогона
            same_apply = bool(int(pr0.get("apply") or 0)) == bool(apply)
            same_skip = bool(int(pr0.get("skip_dedup") or 0)) == bool(skip_dedup)
            if not (same_apply and same_skip):
                raise HTTPException(status_code=400, detail="resume_run_id options mismatch (apply/skip_dedup)")
            pipeline_run_id = rid
            resumed = True
            ps.update_run(run_id=pipeline_run_id, status="running", last_error="", finished_at="")
        # 1) Принудительно "новый прогон"
        elif start_mode == "new":
            pipeline_run_id = ps.create_run(kind="local_sort", root_path=root_path, apply=apply, skip_dedup=skip_dedup)
        elif latest:
            same_opts = bool(int(latest.get("apply") or 0)) == bool(apply) and bool(int(latest.get("skip_dedup") or 0)) == bool(skip_dedup)
            st = str(latest.get("status") or "")
            # Если прошлый прогон "упал" — продолжаем его же (resume).
            if st in ("failed",) and same_opts:
                pipeline_run_id = int(latest["id"])
                resumed = True
                ps.update_run(run_id=pipeline_run_id, status="running", last_error="", finished_at="")
            # Если он ещё "running", но давно не обновлялся — считаем его залипшим и продолжаем (resume).
            elif st == "running" and same_opts:
                # updated_at хранится в UTC ISO; если пусто — считаем залипшим.
                updated_at = str(latest.get("updated_at") or "")
                is_stale = True
                try:
                    from datetime import datetime, timezone

                    dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    is_stale = (datetime.now(timezone.utc) - dt).total_seconds() > 180
                except Exception:
                    is_stale = True
                if not is_stale:
                    return {"ok": False, "message": "local pipeline already running", "run_id": int(latest["id"])}
                pipeline_run_id = int(latest["id"])
                resumed = True
                ps.update_run(run_id=pipeline_run_id, status="running", last_error="stale: restarting worker", finished_at="")
            else:
                pipeline_run_id = ps.create_run(kind="local_sort", root_path=root_path, apply=apply, skip_dedup=skip_dedup)
        else:
            pipeline_run_id = ps.create_run(kind="local_sort", root_path=root_path, apply=apply, skip_dedup=skip_dedup)
    finally:
        ps.close()

    # Auto-snapshot metrics for previous run (best-effort), before the new run overwrites files.* with a new face_run_id.
    # No button: this runs automatically on every "new" start (not on resume).
    try:
        if not bool(resumed):
            # Ensure we snapshot only a real previous completed/failed run, not "running".
            if prev_run_id and int(prev_run_id) != int(pipeline_run_id) and (prev_status in ("completed", "failed")) and prev_face_run_id:
                psm = PipelineStore()
                try:
                    already = psm.get_metrics_for_run(pipeline_run_id=int(prev_run_id))
                    if not already:
                        ds = DedupStore()
                        fs = FaceStore()
                        try:
                            cats = gold_api._count_misses_for_gold(
                                ds=ds,
                                pipeline_run_id=int(prev_run_id),
                                face_run_id=int(prev_face_run_id),
                                root_like=prev_root_like,
                                gold_name="cats_gold",
                                expected_tab="animals",
                            )
                            faces = gold_api._count_misses_for_gold(
                                ds=ds,
                                pipeline_run_id=int(prev_run_id),
                                face_run_id=int(prev_face_run_id),
                                root_like=prev_root_like,
                                gold_name="faces_gold",
                                expected_tab="faces",
                            )
                            no_faces = gold_api._count_misses_for_gold(
                                ds=ds,
                                pipeline_run_id=int(prev_run_id),
                                face_run_id=int(prev_face_run_id),
                                root_like=prev_root_like,
                                gold_name="no_faces_gold",
                                expected_tab="no_faces",
                            )
                        finally:
                            ds.close()
                        step2_total = step2_processed = None
                        try:
                            fr = fs.get_run_by_id(run_id=int(prev_face_run_id)) or {}
                            step2_total = fr.get("total_files")
                            step2_processed = fr.get("processed_files")
                        finally:
                            fs.close()
                        psm.upsert_metrics(
                            pipeline_run_id=int(prev_run_id),
                            metrics={
                                "computed_at": _now_utc_iso(),
                                "face_run_id": int(prev_face_run_id),
                                "step2_total": step2_total,
                                "step2_processed": step2_processed,
                                "cats_total": cats.get("total"),
                                "cats_mism": cats.get("mism"),
                                "faces_total": faces.get("total"),
                                "faces_mism": faces.get("mism"),
                                "no_faces_total": no_faces.get("total"),
                                "no_faces_mism": no_faces.get("mism"),
                            },
                        )
                finally:
                    psm.close()
    except Exception:
        # best-effort: do not block start due to snapshot failures
        pass

    # Если resume происходит после старого прогона, где dedup_run уже completed,
    # но processed_files остались 0/total — "долечим" это в БД (данные, не UI).
    try:
        ps_fix = PipelineStore()
        try:
            pr = ps_fix.get_run_by_id(run_id=int(pipeline_run_id))
        finally:
            ps_fix.close()
        dedup_run_id = pr.get("dedup_run_id") if pr else None
        if dedup_run_id:
            ds_fix = DedupStore()
            try:
                dr = ds_fix.get_run_by_id(run_id=int(dedup_run_id))
                if dr and str(dr.get("status") or "") == "completed":
                    total = dr.get("total_files")
                    proc = dr.get("processed_files")
                    if total is not None:
                        try:
                            ti = int(total)
                            pi = int(proc or 0)
                            if ti > 0 and pi < ti:
                                ds_fix.update_run_progress(run_id=int(dedup_run_id), processed_files=ti)
                        except Exception:
                            pass
            finally:
                ds_fix.close()
    except Exception:
        # best-effort: не ломаем старт конвейера из-за "лечения" статистики
        pass

    global _LOCAL_PIPELINE_FUTURE  # noqa: PLW0603
    with _LOCAL_PIPELINE_LOCK:
        fut = _LOCAL_PIPELINE_FUTURE
        if fut is not None and not fut.done():
            return {"ok": False, "message": "local pipeline already running"}
        global _LOCAL_PIPELINE_RUN_ID  # noqa: PLW0603
        _LOCAL_PIPELINE_RUN_ID = int(pipeline_run_id)
        _LOCAL_PIPELINE_FUTURE = _LOCAL_PIPELINE_EXEC.submit(
            _run_local_pipeline,
            root_path=root_path,
            apply=apply,
            skip_dedup=skip_dedup,
            no_dedup_move=no_dedup_move,
            pipeline_run_id=int(pipeline_run_id),
        )

    return {
        "ok": True,
        "message": "started",
        "run_id": int(pipeline_run_id),
        "resumed": bool(resumed),
        "start_mode": start_mode or ("continue" if resumed else "new"),
        "root_path": root_path,
        "apply": apply,
        "skip_dedup": skip_dedup,
        "no_dedup_move": no_dedup_move,
    }


@app.get("/api/_legacy/local-pipeline/latest")
def api_local_pipeline_latest(root_path: str) -> dict[str, Any]:
    """
    Возвращает последний pipeline_run для указанной папки (root_path) — чтобы UI мог показать кнопку "Продолжить".
    """
    raise HTTPException(status_code=410, detail="Moved to web_api/routers/local_pipeline.py")
    rp = str(root_path or "").strip()
    if not rp:
        raise HTTPException(status_code=400, detail="root_path is required")
    ps = PipelineStore()
    try:
        pr = ps.get_latest_run(kind="local_sort", root_path=rp)
    finally:
        ps.close()
    return {"ok": True, "latest": pr}


@app.get("/api/_legacy/local-pipeline/status")
def api_local_pipeline_status() -> dict[str, Any]:
    raise HTTPException(status_code=410, detail="Moved to web_api/routers/local_pipeline.py")
    ps = PipelineStore()
    try:
        pr = ps.get_latest_any(kind="local_sort")
    finally:
        ps.close()

    if not pr:
        return {
            "running": False,
            "exit_code": None,
            "error": None,
            "run_id": None,
            "root_path": None,
            "apply": False,
            "skip_dedup": False,
            "step": {"num": 0, "total": 0, "title": ""},
            "step1": {"status": "idle", "pct": 0, "processed": None, "total": None},
            "step2": {"status": "idle", "pct": 0, "images": None, "total": None, "faces": None},
            "log_tail": "",
        }

    status = str(pr.get("status") or "")
    running = status == "running"
    # Стараемся вернуть реальный exit code (например, для нативных крэшей OpenCV).
    # В БД last_error часто хранится как "exit_code=3221225786".
    exit_code: int | None
    if status == "completed":
        exit_code = 0
    elif status == "failed":
        le = str(pr.get("last_error") or "")
        if le.startswith("exit_code="):
            try:
                exit_code = int(le.split("=", 1)[1].strip())
            except Exception:
                exit_code = 1
        else:
            exit_code = 1
    else:
        exit_code = None

    step_num = int(pr.get("step_num") or 0)
    step_total = int(pr.get("step_total") or 0)
    step_title = str(pr.get("step_title") or "")

    # log tail
    log = str(pr.get("log_tail") or "").replace("\r\n", "\n")
    log_tail = "\n".join(log.splitlines()[-160:])

    # --- debug (video env / cmd flags) ---
    video_samples_env: int | None
    try:
        video_samples_env = int(os.getenv("LOCAL_PIPELINE_VIDEO_SAMPLES") or "0")
    except Exception:
        video_samples_env = None

    def _last_run_cmd_from_log(log_text: str) -> str | None:
        for line in reversed((log_text or "").splitlines()):
            if line.startswith("RUN: "):
                return line[len("RUN: ") :].strip()
        return None

    run_cmd = _last_run_cmd_from_log(log_tail)
    cmd_has_video_samples = bool(run_cmd and "--video-samples" in run_cmd)
    cmd_video_samples: int | None = None
    if run_cmd and "--video-samples" in run_cmd:
        try:
            parts = run_cmd.split()
            i = parts.index("--video-samples")
            if i + 1 < len(parts):
                cmd_video_samples = int(parts[i + 1])
        except Exception:
            cmd_video_samples = None

    # Шаг 1 (предочистка): прогресс best-effort из log_tail (и/или preclean_state, если есть).
    step0_checked = step0_non_media = step0_broken_media = None
    try:
        import re

        # ВАЖНО: парсим по смысловому тегу, а не по номерам шагов (они меняются в UI).
        m0 = re.search(r"preclean:\s*checked=(\d+)\s+moved_non_media=(\d+)\s+moved_broken_media=(\d+)", log_tail)
        if m0:
            step0_checked = int(m0.group(1))
            step0_non_media = int(m0.group(2))
            step0_broken_media = int(m0.group(3))
    except Exception:
        pass

    # статус шага 0 определяем по step_num/title
    title_l = (step_title or "").lower()
    if status == "completed":
        step0_status = "done"
        step0_pct = 100
    elif status == "failed":
        step0_status = "error" if ("предочист" in title_l or step_num <= 1) else "done"
        step0_pct = 100 if step0_status == "done" else (50 if step0_checked is not None else 0)
    elif running:
        if "предочист" in title_l:
            step0_status = "running"
            step0_pct = 50 if step0_checked is not None else 10
        elif step_num >= 2:
            step0_status = "done"
            step0_pct = 100
        else:
            step0_status = "pending"
            step0_pct = 0
    else:
        step0_status = "idle"
        step0_pct = 0

    # Шаг 2 (дедупликация): прогресс берём из dedup_runs
    dedup_proc = dedup_total = None
    dedup_run_id = pr.get("dedup_run_id")
    if dedup_run_id:
        ds = DedupStore()
        try:
            dr = ds.get_run_by_id(run_id=int(dedup_run_id))
        finally:
            ds.close()
        if dr:
            dedup_proc = dr.get("processed_files")
            dedup_total = dr.get("total_files")

    if dedup_total and int(dedup_total) > 0 and dedup_proc is not None:
        step1_pct = int(round((int(dedup_proc) / int(dedup_total)) * 100))
        step1_pct = max(0, min(100, step1_pct))
    else:
        step1_pct = 100 if step_num >= 2 or status == "completed" else (0 if status in ("", "failed") else 0)
    # Статус шага 1 определяем по номеру шага, а не по общему статусу:
    # если упал шаг 2, шаг 1 всё равно "done".
    if step_num >= 2 or status == "completed":
        step1_status = "done"
    elif running and step_num == 1:
        step1_status = "running"
    elif status == "failed" and step_num <= 1:
        step1_status = "error"
    else:
        step1_status = "idle"

    # Шаг 3 (лица/нет лиц): прогресс из face_runs, фазы 85/95 как раньше
    faces_img = faces_total = None
    faces_found = None
    face_run_id = pr.get("face_run_id")
    if face_run_id:
        fs = FaceStore()
        try:
            fr = fs.get_run_by_id(run_id=int(face_run_id))
        finally:
            fs.close()
        if fr:
            faces_img = fr.get("processed_files")
            faces_total = fr.get("total_files")
            faces_found = fr.get("faces_found")

    # Сейчас у конвейера 2 шага (внутренне), но UI планируем 1..6.
    def _scan_pct() -> int:
        if faces_total and int(faces_total) > 0 and faces_img is not None:
            pct = int(round((int(faces_img) / int(faces_total)) * 90))
            return max(0, min(90, pct))
        return 0

    scan_pct = _scan_pct()
    is_split_phase = "разлож" in step_title.lower()

    if status == "completed":
        step2_pct = 100
        step2_status = "done"
    elif running:
        if step_num < 2:
            step2_pct = 0
            step2_status = "pending"
        else:
            step2_status = "running"
            step2_pct = scan_pct
            if is_split_phase:
                step2_pct = max(step2_pct, 95)
    elif status == "failed":
        # Важно: даже при ошибке показываем реальный прогресс скана по face_runs,
        # иначе UI выглядит как "0%", хотя обработано N/total.
        step2_status = "error" if step_num >= 2 else "idle"
        step2_pct = scan_pct
        if is_split_phase:
            step2_pct = max(step2_pct, 95)
    else:
        step2_pct = 0
        step2_status = "idle"

    # UI plan step (1..6) — best-effort mapping.
    plan_total = 6
    if "предочист" in title_l:
        plan_num = 1
        plan_title = "предочистка"
    elif "дедуп" in title_l:
        plan_num = 2
        plan_title = "дедупликация"
    elif "лица" in title_l:
        plan_num = 3
        plan_title = "лица/животные/нет людей"
    else:
        plan_num = max(0, int(step_num or 0))
        plan_title = step_title

    return {
        "running": running,
        "exit_code": exit_code,
        "error": pr.get("last_error"),
        "run_id": pr.get("id"),
        "root_path": pr.get("root_path"),
        "apply": bool(int(pr.get("apply") or 0)),
        "skip_dedup": bool(int(pr.get("skip_dedup") or 0)),
        "started_at": pr.get("started_at"),
        "updated_at": pr.get("updated_at"),
        "finished_at": pr.get("finished_at"),
        "step": {"num": step_num, "total": step_total, "title": step_title},
        "plan_step": {"num": int(plan_num), "total": int(plan_total), "title": str(plan_title or "")},
        "step0": {"status": step0_status, "pct": step0_pct, "checked": step0_checked, "non_media": step0_non_media, "broken_media": step0_broken_media},
        "step1": {"status": step1_status, "pct": step1_pct, "processed": dedup_proc, "total": dedup_total},
        "step2": {"status": step2_status, "pct": step2_pct, "images": faces_img, "total": faces_total, "faces": faces_found},
        "debug": {"video_samples_env": video_samples_env, "cmd_has_video_samples": cmd_has_video_samples, "cmd_video_samples": cmd_video_samples},
        "log_tail": log_tail,
    }


@app.get("/api/sort/context")
def api_sort_context() -> dict[str, Any]:
    """
    Контекст для UI: последние прогоны archive/source + флаги running.
    """
    archive = api_dedup_archive_status()
    source = api_dedup_source_status()
    return {"archive": archive, "source": source}


@app.get("/api/sort/dup-in-archive")
def api_sort_dup_in_archive(source_run_id: int | None = None) -> dict[str, Any]:
    """
    Шаг 1: source-файлы, которые уже есть в архиве (совпадает хэш с inventory_scope=archive).
    """
    store = DedupStore()
    try:
        src_latest = store.get_latest_run(scope="source")
        run_id = int(source_run_id) if source_run_id is not None else (int(src_latest["id"]) if src_latest else None)
        if not run_id:
            raise HTTPException(status_code=400, detail="Нет активного source run. Сначала запустите сортировку на главной странице.")

        rows = store.list_source_dups_in_archive(source_run_id=run_id, archive_prefixes=tuple(_dedup_archive_roots()))
        # root_path должен соответствовать выбранному run_id (а не обязательно последнему scope=source)
        dr = store.get_run_by_id(run_id=int(run_id))
        root_path = str(dr.get("root_path") or "") if dr else None

        archive_latest = store.get_latest_run(scope="archive")
        archive_scanned = bool(archive_latest and str(archive_latest.get("status") or "") == "completed")
    finally:
        store.close()

    # Группируем по source_path
    grouped: dict[str, dict[str, Any]] = {}
    for r in rows:
        sp = str(r.get("source_path") or "")
        if not sp:
            continue
        g = grouped.get(sp)
        if not g:
            g = {
                "source_path": sp,
                "source_name": r.get("source_name"),
                "source_size": r.get("source_size"),
                "source_mime_type": r.get("source_mime_type"),
                "source_media_type": r.get("source_media_type"),
                "hash_alg": r.get("hash_alg"),
                "hash_value": r.get("hash_value"),
                "matches": [],
            }
            grouped[sp] = g
        g["matches"].append(
            {
                "path": r.get("archive_path"),
                "name": r.get("archive_name"),
                "size": r.get("archive_size"),
                "mime_type": r.get("archive_mime_type"),
                "media_type": r.get("archive_media_type"),
            }
        )

    items = list(grouped.values())
    return {
        "ok": True,
        "source_run_id": run_id,
        "source_root_path": root_path,
        "items": items,
        "total": len(items),
        "archive_scanned": bool(locals().get("archive_scanned", False)),
    }


@app.post("/api/sort/source/ignore-archive-dup")
def api_sort_source_ignore_archive_dup(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Шаг 1: "Оставить (не дубль)" — скрыть совпадение до следующего пересканирования source.
    """
    paths = payload.get("paths")
    run_id = payload.get("source_run_id")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths[] is required")
    try:
        run_id_i = int(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="source_run_id is required") from None

    clean: list[str] = [p for p in paths if isinstance(p, str) and p]
    if not clean:
        raise HTTPException(status_code=400, detail="no valid paths")

    store = DedupStore()
    try:
        # минимальная валидация: пути должны принадлежать текущему run_id
        bad: list[str] = []
        for p in clean:
            row = store.get_row_by_path(path=p)
            if not row:
                bad.append(p)
                continue
            if str(row.get("inventory_scope") or "") != "source":
                bad.append(p)
                continue
            if int(row.get("last_run_id") or 0) != run_id_i:
                bad.append(p)
        if bad:
            raise HTTPException(status_code=400, detail=f"paths do not belong to source run {run_id_i}: {bad[:5]}")

        n = store.set_ignore_archive_dup(paths=clean, run_id=run_id_i)
    finally:
        store.close()
    return {"ok": True, "updated": n}


@app.post("/api/sort/source/delete")
def api_sort_source_delete(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Удаляет файлы ИЗ ИСТОЧНИКА:
    - YaDisk source: в корзину
    - local source: навсегда (os.remove)
    """
    paths = payload.get("paths")
    run_id = payload.get("source_run_id")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths[] is required")
    try:
        run_id_i = int(run_id) if run_id is not None else None
    except Exception:
        run_id_i = None

    clean: list[str] = [p for p in paths if isinstance(p, str) and p]
    if not clean:
        raise HTTPException(status_code=400, detail="no valid paths")

    store = DedupStore()
    try:
        # Если source_run_id не передали — работаем по последнему scope=source (legacy).
        if run_id_i is None:
            src_latest = store.get_latest_run(scope="source")
            if not src_latest:
                raise HTTPException(status_code=400, detail="Нет активного source run.")
            run_id_i = int(src_latest["id"])
        dr = store.get_run_by_id(run_id=int(run_id_i))
        if not dr:
            raise HTTPException(status_code=400, detail=f"source_run_id not found: {run_id_i}")
        root_path = str(dr.get("root_path") or "")
    finally:
        store.close()

    disk = None
    ok_paths: list[str] = []
    errors: list[dict[str, str]] = []

    for p in clean:
        # v1 безопасности: удаляем только файлы, которые реально были в текущем source run
        st = DedupStore()
        try:
            row = st.get_row_by_path(path=p)
        finally:
            st.close()
        if not row or str(row.get("inventory_scope") or "") != "source" or int(row.get("last_run_id") or 0) != int(run_id_i or 0):
            errors.append({"path": p, "error": "not_in_active_source_run"})
            continue

        try:
            if p.startswith("disk:"):
                if disk is None:
                    disk = get_disk()
                rp = root_path.rstrip("/")
                if rp and not (p == rp or p.startswith(rp + "/")):
                    raise RuntimeError("path_outside_source_root")
                _yadisk_remove_to_trash(disk, path=p)
                ok_paths.append(p)
            elif p.startswith("local:"):
                file_path = _strip_local_prefix(p)
                if root_path and not _local_is_under_root(file_path=file_path, root_dir=root_path):
                    raise RuntimeError("path_outside_source_root")
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    # считаем как удалённый
                    pass
                ok_paths.append(p)
            else:
                raise RuntimeError("unknown_path_scheme")
        except Exception as e:  # noqa: BLE001
            errors.append({"path": p, "error": f"{type(e).__name__}: {e}"})

    st2 = DedupStore()
    try:
        st2.mark_deleted(paths=ok_paths)
    finally:
        st2.close()

    return {"ok": True, "deleted": len(ok_paths), "errors": errors}


@app.get("/api/sort/dup-in-source")
def api_sort_dup_in_source(source_run_id: int | None = None) -> dict[str, Any]:
    """
    Шаг 2: дубли ВНУТРИ выбранной папки source (по текущему source run).
    """
    store = DedupStore()
    try:
        src_latest = store.get_latest_run(scope="source")
        run_id = int(source_run_id) if source_run_id is not None else (int(src_latest["id"]) if src_latest else None)
        if not run_id:
            raise HTTPException(status_code=400, detail="Нет активного source run. Сначала запустите сортировку на главной странице.")

        groups_raw = store.list_dup_groups_for_run(inventory_scope="source", run_id=run_id)
        dr = store.get_run_by_id(run_id=int(run_id))
        root_path = str(dr.get("root_path") or "") if dr else None
    finally:
        store.close()

    groups: list[dict[str, Any]] = []
    max_group_size = 0
    for g in groups_raw:
        hash_alg = str(g.get("hash_alg") or "")
        hash_value = str(g.get("hash_value") or "")
        cnt = int(g.get("cnt") or 0)
        max_group_size = max(max_group_size, cnt)

        store2 = DedupStore()
        try:
            items = store2.list_group_items(hash_alg=hash_alg, hash_value=hash_value, inventory_scope="source", last_run_id=run_id)
        finally:
            store2.close()

        keep_idx = _pick_keep_indexes(items, {})

        ui_items: list[dict[str, Any]] = []
        for i, it in enumerate(items):
            path = str(it.get("path") or "")
            mime_type = str(it.get("mime_type") or "") or None
            media_type = str(it.get("media_type") or "") or None
            # Fallback: для старых записей mime/media могут быть пустыми -> попробуем угадать по расширению.
            if not mime_type:
                guess_name = _strip_local_prefix(path) if path.startswith("local:") else _basename_from_disk_path(path)
                mt2, _enc = mimetypes.guess_type(guess_name)
                mime_type = mt2 or None
            if not media_type and mime_type:
                if mime_type.startswith("image/"):
                    media_type = "image"
                elif mime_type.startswith("video/"):
                    media_type = "video"
            size = it.get("size")
            size_i = int(size) if isinstance(size, (int, float)) else None

            preview_kind = "none"  # 'image'|'video'|'none'
            preview_url: Optional[str] = None
            open_url: Optional[str] = None

            if path.startswith("disk:"):
                open_url = "/api/yadisk/open?path=" + urllib.parse.quote(path, safe="")
                mt = (media_type or "").lower()
                mime = (mime_type or "").lower()
                if mt == "image" or mime.startswith("image/"):
                    preview_kind = "image"
                    preview_url = "/api/yadisk/preview-image?size=M&path=" + urllib.parse.quote(path, safe="")
                elif mt == "video" or mime.startswith("video/"):
                    preview_kind = "video"
                    preview_url = None
            elif path.startswith("local:"):
                local_p = path
                mt = (media_type or "").lower()
                mime = (mime_type or "").lower()
                if mt == "image" or mime.startswith("image/"):
                    preview_kind = "image"
                    preview_url = "/api/local/preview?path=" + urllib.parse.quote(local_p, safe="")
                elif mt == "video" or mime.startswith("video/"):
                    preview_kind = "video"
                    preview_url = "/api/local/preview?path=" + urllib.parse.quote(local_p, safe="")

            ui_items.append(
                {
                    "path": path,
                    "path_short": _short_path_for_ui(path) if path.startswith("disk:") else path,
                    "size_human": _human_bytes(size_i),
                    "keep": i == keep_idx,
                    "mime_type": mime_type,
                    "media_type": media_type,
                    "preview_kind": preview_kind,
                    "preview_url": preview_url,
                    "open_url": open_url,
                }
            )

        groups.append(
            {
                "hash_alg": hash_alg,
                "hash_short": (hash_value[:12] + "…") if len(hash_value) > 12 else hash_value,
                "cnt": cnt,
                "files": ui_items,
            }
        )

    summary = {"groups": len(groups), "max_group_size": max_group_size}
    return {"ok": True, "source_run_id": run_id, "source_root_path": root_path, "summary": summary, "groups": groups}


@app.get("/sort/dup-in-archive", response_class=HTMLResponse)
def sort_dup_in_archive_page(request: Request):
    # One-page mode: оставляем только главную рабочую форму.
    return RedirectResponse(url="/", status_code=307)


@app.get("/sort/dup-in-source", response_class=HTMLResponse)
def sort_dup_in_source_page(request: Request):
    # One-page mode: оставляем только главную рабочую форму.
    return RedirectResponse(url="/", status_code=307)


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
    try:
        items = _yd_call_retry(lambda: list(disk.listdir(p)))
        seconds = round(time.perf_counter() - t0, 2)
    except Exception as e:  # noqa: BLE001
        seconds = round(time.perf_counter() - t0, 2)
        return {
            "path": _as_disk_path(p),
            "direct_files": None,
            "dirs": [],
            "seconds": seconds,
            "error": f"{type(e).__name__}: {e}",
        }

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


@app.get("/duplicates", response_class=HTMLResponse)
def duplicates_page(request: Request):
    """
    Просмотр дублей в архиве (disk:/Фото). Пока read-only.
    Превью и тип подтягиваем "лениво" через YaDisk get_meta.
    """
    store = DedupStore()
    try:
        groups_raw = store.list_dup_groups_archive()
        folders = list_folders(location="yadisk", role="target")
    finally:
        store.close()

    sort_order_by_folder: dict[str, int] = {}
    for f in folders:
        name = str(f.get("name") or "")
        so = f.get("sort_order")
        if name and isinstance(so, int):
            sort_order_by_folder[name.lower()] = so

    groups: list[dict[str, Any]] = []
    max_group_size = 0
    for g in groups_raw:
        hash_alg = str(g.get("hash_alg") or "")
        hash_value = str(g.get("hash_value") or "")
        cnt = int(g.get("cnt") or 0)
        max_group_size = max(max_group_size, cnt)

        store2 = DedupStore()
        try:
            items = store2.list_group_items(hash_alg=hash_alg, hash_value=hash_value, inventory_scope="archive")
        finally:
            store2.close()

        keep_idx = _pick_keep_indexes(items, sort_order_by_folder)

        ui_items: list[dict[str, Any]] = []
        for i, it in enumerate(items):
            path = str(it.get("path") or "")
            mime_type = str(it.get("mime_type") or "") or None
            media_type = str(it.get("media_type") or "") or None
            size = it.get("size")
            size_i = int(size) if isinstance(size, (int, float)) else None

            ui_items.append(
                {
                    "path": path,
                    "path_short": _short_path_for_ui(path),
                    "folder_short": _short_folder_from_disk_path(path),
                    "size_human": _human_bytes(size_i),
                    "mime_type": mime_type,
                    "media_type": media_type,
                    "keep": i == keep_idx,
                }
            )

        groups.append(
            {
                "hash_alg": hash_alg,
                "hash_short": (hash_value[:12] + "…") if len(hash_value) > 12 else hash_value,
                "cnt": cnt,
                "files": ui_items,
                "keep_invalid": False,
                "note": None,
            }
        )

    summary = {"groups": len(groups), "max_group_size": max_group_size}
    return templates.TemplateResponse("duplicates.html", {"request": request, "groups": groups, "summary": summary})


def _yadisk_remove_to_trash(disk, *, path: str) -> None:
    p = _normalize_yadisk_path(path)
    try:
        _yd_call_retry(lambda: disk.remove(p, permanently=False))
    except TypeError:
        # старые версии/обёртки могли не поддерживать permanently
        _yd_call_retry(lambda: disk.remove(p))


def _yadisk_move(disk, *, src_path: str, dst_path: str) -> None:
    src = _normalize_yadisk_path(src_path)
    dst = _normalize_yadisk_path(dst_path)
    try:
        _yd_call_retry(lambda: disk.move(src, dst, overwrite=False))
    except TypeError:
        _yd_call_retry(lambda: disk.move(src, dst))


@app.post("/api/duplicates/delete")
def api_duplicates_delete(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Удаляет (в корзину) набор файлов Я.Диска и помечает их как deleted в БД,
    чтобы они исчезали из /duplicates без перескана.
    """
    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths[] is required")
    if len(paths) > 500:
        raise HTTPException(status_code=400, detail="too many paths (max 500)")

    clean: list[str] = []
    for p in paths:
        if not isinstance(p, str) or not p.startswith("disk:"):
            continue
        clean.append(p)
    if not clean:
        raise HTTPException(status_code=400, detail="no valid disk: paths")

    disk = get_disk()
    ok_paths: list[str] = []
    errors: list[dict[str, str]] = []
    for p in clean:
        try:
            _yadisk_remove_to_trash(disk, path=p)
            ok_paths.append(p)
        except Exception as e:  # noqa: BLE001
            errors.append({"path": p, "error": f"{type(e).__name__}: {e}"})

    store = DedupStore()
    try:
        store.mark_deleted(paths=ok_paths)
    finally:
        store.close()

    return {"ok": True, "deleted": len(ok_paths), "errors": errors}


@app.post("/api/archive/delete")
def api_archive_delete(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Физически удаляет файл из архива (YaDisk): permanently=True, помечает deleted в БД.
    Требуется подтверждение на клиенте.
    """
    path = payload.get("path")
    if not isinstance(path, str) or not path.startswith("disk:"):
        raise HTTPException(status_code=400, detail="path (disk:) is required for archive delete")

    disk = get_disk()
    p_norm = _normalize_yadisk_path(path)
    try:
        _yd_call_retry(lambda: disk.remove(p_norm, permanently=True))
    except TypeError:
        _yd_call_retry(lambda: disk.remove(p_norm))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"YaDisk delete failed: {type(e).__name__}: {e}") from e

    store = DedupStore()
    try:
        store.mark_deleted(paths=[path])
    finally:
        store.close()

    removed_from_gold = _delete_from_all_gold_files(path)

    return {"ok": True, "path": path, "removed_from_gold": removed_from_gold}


@app.post("/api/duplicates/move-to-kids")
def api_duplicates_move_to_kids(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Перемещает один файл в папку 'Дети вместе' и удаляет (в корзину) остальные
    файлы из группы.
    """
    paths = payload.get("paths")
    prefer = payload.get("prefer_path")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths[] is required")

    all_paths: list[str] = []
    for p in paths:
        if isinstance(p, str) and p.startswith("disk:"):
            all_paths.append(p)
    if len(all_paths) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 disk: paths")

    # Приоритет: prefer_path (если валиден), затем остальные.
    candidates: list[str] = []
    if isinstance(prefer, str) and prefer in all_paths:
        candidates.append(prefer)
    for p in all_paths:
        if p not in candidates:
            candidates.append(p)

    dest_dir = _resolve_target_folder_path_kids_together()
    dest_dir_norm = _normalize_yadisk_path(dest_dir)

    disk = get_disk()
    store = DedupStore()
    try:
        # Сначала пытаемся выбрать файл, который не УЖЕ в целевой папке (чтобы move не был no-op).
        dest_prefix = dest_dir_norm.rstrip("/") + "/"
        candidates_sorted = sorted(
            candidates,
            key=lambda p: 0 if not _normalize_yadisk_path(p).startswith(dest_prefix) else 1,
        )

        def _is_free_name(name: str) -> bool:
            dp = _disk_join(dest_dir, name)
            try:
                if bool(_yd_call_retry(lambda: disk.exists(_normalize_yadisk_path(dp)))):
                    return False
            except Exception:
                # если exists не работает, полагаемся на БД + fallback-rename
                pass
            return not store.path_exists(path=dp)

        chosen_src: str | None = None
        chosen_name: str | None = None
        # Попробуем сначала "взять файл с другим именем" (без переименования).
        for p in candidates_sorted:
            base = _basename_from_disk_path(p)
            if base and _is_free_name(base):
                chosen_src = p
                chosen_name = base
                break

        # Если все имена конфликтуют — переименуем.
        if not chosen_src:
            chosen_src = candidates_sorted[0]
            chosen_name = _unique_dest_name(store=store, disk=disk, dest_dir=dest_dir, src_name=_basename_from_disk_path(chosen_src))

        dest_path = _disk_join(dest_dir, str(chosen_name))

        _yadisk_move(disk, src_path=chosen_src, dst_path=dest_path)

        # Удаляем остальные (в корзину).
        to_delete = [p for p in all_paths if p != chosen_src]
        deleted_ok: list[str] = []
        errors: list[dict[str, str]] = []
        for p in to_delete:
            try:
                _yadisk_remove_to_trash(disk, path=p)
                deleted_ok.append(p)
            except Exception as e:  # noqa: BLE001
                errors.append({"path": p, "error": f"{type(e).__name__}: {e}"})

        # Обновляем БД: moved path + помечаем удалённые.
        store.update_path(
            old_path=chosen_src,
            new_path=dest_path,
            new_name=_basename_from_disk_path(dest_path),
            new_parent_path=_parent_from_disk_path(dest_path),
        )
        store.mark_deleted(paths=deleted_ok)

        return {
            "ok": True,
            "moved_from": chosen_src,
            "moved_to": dest_path,
            "deleted": len(deleted_ok),
            "errors": errors,
        }
    finally:
        store.close()


@app.get("/api/yadisk/preview")
def api_yadisk_preview(path: str) -> dict[str, Any]:
    """
    Ленивая подгрузка превью/типа через YaDisk get_meta для конкретного файла.
    Используется страницей /duplicates, чтобы она открывалась быстро.
    """
    disk = get_disk()
    try:
        # `preview` часто приходит только при явном запросе размера превью.
        # Если библиотека/эндпойнт не поддерживают preview_size — fallback на обычный get_meta.
        try:
            md = _yd_call_retry(lambda: disk.get_meta(_normalize_yadisk_path(path), limit=0, preview_size="M"))
        except TypeError:
            md = _yd_call_retry(lambda: disk.get_meta(_normalize_yadisk_path(path), limit=0))
        # В yadisk 3.4.x meta-объект обычно отдаёт значения как атрибуты (meta.preview),
        # а `to_json()` может отсутствовать.
        preview = getattr(md, "preview", None)
        mime_type = getattr(md, "mime_type", None)
        media_type = getattr(md, "media_type", None)
        dur_sec: Optional[int] = None
        mt = (media_type or "").lower()
        mime = (mime_type or "").lower()
        if mt == "video" or mime.startswith("video/"):
            dur_sec = _extract_duration_seconds(md)
        return {
            "ok": True,
            "path": path,
            "preview": preview or None,
            "mime_type": mime_type or None,
            "media_type": media_type or None,
            "duration_sec": dur_sec,
            "duration_human": _human_duration(dur_sec),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "path": path,
            "preview": None,
            "mime_type": None,
            "media_type": None,
            "duration_sec": None,
            "duration_human": "—",
            "error": f"{type(e).__name__}: {e}",
        }


@app.get("/api/yadisk/video-duration")
def api_yadisk_video_duration(path: str) -> Response:
    """
    Асинхронно вычисляет длительность видео через ffprobe (вариант B).
    Возвращает:
      - 200 {status:'ready', duration_sec, duration_human}
      - 202 {status:'pending'}
      - 400/500 {status:'error', error}
    """
    if not path.startswith("disk:"):
        return JSONResponse(status_code=400, content={"ok": False, "status": "error", "error": "Only disk: paths are supported"})

    store = DedupStore()
    try:
        cached = store.get_duration(path=path)
    finally:
        store.close()

    if isinstance(cached, int) and cached >= 0:
        return JSONResponse(
            status_code=200,
            content={"ok": True, "status": "ready", "duration_sec": cached, "duration_human": _human_duration(cached)},
            headers={"Cache-Control": "private, max-age=3600"},
        )

    with _VIDEO_LOCK:
        # Если уже есть ошибка — отдадим её (чтобы UI мог остановиться).
        err = _VIDEO_ERRORS.get(path)
        if err:
            return JSONResponse(status_code=500, content={"ok": False, "status": "error", "error": err}, headers={"Cache-Control": "no-store"})

        fut = _VIDEO_FUTURES.get(path)
        if fut is None or fut.done():
            # Планируем вычисление.
            def _runner() -> None:
                try:
                    disk = get_disk()
                    url = _get_download_url(disk, path=path)
                    sec = _ffprobe_duration_seconds_from_url(url, timeout_sec=35)
                    store2 = DedupStore()
                    try:
                        store2.set_duration(path=path, duration_sec=sec, source="ffprobe")
                    finally:
                        store2.close()
                except Exception as e:  # noqa: BLE001
                    with _VIDEO_LOCK:
                        _VIDEO_ERRORS[path] = f"{type(e).__name__}: {e}"
                finally:
                    # очищаем future
                    with _VIDEO_LOCK:
                        _VIDEO_FUTURES.pop(path, None)

            _VIDEO_FUTURES[path] = _VIDEO_EXEC.submit(_runner)

    return JSONResponse(status_code=202, content={"ok": True, "status": "pending"}, headers={"Retry-After": "1", "Cache-Control": "no-store"})


@app.get("/api/local/video-duration")
def api_local_video_duration(path: str) -> Response:
    """
    Асинхронно вычисляет длительность ЛОКАЛЬНОГО видео через ffprobe.
    Возвращает:
      - 200 {status:'ready', duration_sec, duration_human}
      - 202 {status:'pending'}
      - 400/403/404/500 {status:'error', error}
    """
    if not isinstance(path, str) or not path.startswith("local:"):
        return JSONResponse(status_code=400, content={"ok": False, "status": "error", "error": "Only local: paths are supported"})

    store = DedupStore()
    try:
        src_latest = store.get_latest_run(scope="source")
        if not src_latest:
            return JSONResponse(status_code=400, content={"ok": False, "status": "error", "error": "No active source run"})
        root_path = str(src_latest.get("root_path") or "")
        cached = store.get_duration(path=path)
    finally:
        store.close()

    if isinstance(cached, int) and cached >= 0:
        return JSONResponse(
            status_code=200,
            content={"ok": True, "status": "ready", "duration_sec": cached, "duration_human": _human_duration(cached)},
            headers={"Cache-Control": "private, max-age=3600"},
        )

    file_path = _strip_local_prefix(path)
    if root_path and not _local_is_under_root(file_path=file_path, root_dir=root_path):
        return JSONResponse(status_code=403, content={"ok": False, "status": "error", "error": "Path is outside active source root"})
    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        return JSONResponse(status_code=404, content={"ok": False, "status": "error", "error": "File not found"})

    with _VIDEO_LOCK:
        err = _VIDEO_ERRORS.get(path)
        if err:
            return JSONResponse(status_code=500, content={"ok": False, "status": "error", "error": err}, headers={"Cache-Control": "no-store"})

        fut = _VIDEO_FUTURES.get(path)
        if fut is None or fut.done():
            def _runner() -> None:
                try:
                    sec = _ffprobe_duration_seconds_from_url(abs_path, timeout_sec=35)
                    store2 = DedupStore()
                    try:
                        store2.set_duration(path=path, duration_sec=sec, source="ffprobe")
                    finally:
                        store2.close()
                except Exception as e:  # noqa: BLE001
                    with _VIDEO_LOCK:
                        _VIDEO_ERRORS[path] = f"{type(e).__name__}: {e}"
                finally:
                    with _VIDEO_LOCK:
                        _VIDEO_FUTURES.pop(path, None)

            _VIDEO_FUTURES[path] = _VIDEO_EXEC.submit(_runner)

    return JSONResponse(status_code=202, content={"ok": True, "status": "pending"}, headers={"Retry-After": "1", "Cache-Control": "no-store"})


@app.get("/api/yadisk/preview-image")
def api_yadisk_preview_image(path: str, size: str = "M") -> Response:
    """
    Проксирует превью-картинку через наш сервер (localhost), чтобы браузеру не нужно было
    грузить изображения напрямую с downloader.disk.yandex.ru (иногда это блокируется).

    size: размер превью (S/M/L/XL, зависит от API; по умолчанию M).
    """
    # Кэшируем preview_url, чтобы повторные заходы на /duplicates были быстрее и дешевле по YaDisk API.
    cache_key = (path, size)
    cached_url = _preview_cache_get(cache_key)
    if cached_url:
        return RedirectResponse(url=cached_url, status_code=307, headers={"Cache-Control": "private, max-age=300"})

    # Ограничиваем параллелизм; при перегрузке ждём слот до 2 сек, чтобы реже отдавать 429.
    acquired = _PREVIEW_SEM.acquire(timeout=2.0)
    if not acquired:
        return Response(
            status_code=429,
            content=b"Too many requests; retry later",
            media_type="text/plain",
            headers={"Retry-After": "1", "Cache-Control": "no-store"},
        )

    try:
        disk = get_disk()
        p = _normalize_yadisk_path(path)

        try:
            md = _yd_call_retry(lambda: disk.get_meta(p, limit=0, preview_size=size))
        except TypeError:
            md = _yd_call_retry(lambda: disk.get_meta(p, limit=0))

        preview_url = getattr(md, "preview", None)
        if not preview_url:
            return Response(
                status_code=404,
                content=b"No preview for this file",
                media_type="text/plain",
                headers={"Cache-Control": "no-store", "X-Preview-Reason": "no-preview-url"},
            )

        # Важно: Яндекс часто блокирует "серверное" скачивание превью (403 Forbidden),
        # но браузер может открыть этот URL напрямую. Поэтому вместо проксирования байтов
        # делаем redirect на preview_url.
        preview_url_str = str(preview_url)
        _preview_cache_put(cache_key, preview_url_str)
        return RedirectResponse(url=preview_url_str, status_code=307, headers={"Cache-Control": "private, max-age=300"})
    except Exception as e:  # noqa: BLE001
        # Важно: на /duplicates нам нужно понять причину (timeout/403/SSL/etc).
        # Отдаём текст ошибки — его можно посмотреть в DevTools → Network → Response.
        msg = f"{type(e).__name__}: {e}"
        # Не раздуваем заголовки
        hdr_msg = (msg[:200]).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        return Response(
            status_code=502,
            content=msg.encode("utf-8", errors="ignore"),
            media_type="text/plain",
            headers={"Cache-Control": "no-store", "X-Preview-Error": hdr_msg},
        )
    finally:
        try:
            _PREVIEW_SEM.release()
        except ValueError:
            pass


# Максимальный размер видео (байт) для превью-кадра из YaDisk — не скачивать огромные файлы
_YD_VIDEO_FRAME_MAX_BYTES = int(os.getenv("PHOTOSORTER_YD_VIDEO_FRAME_MAX_BYTES") or str(150 * 1024 * 1024))


@app.get("/api/yadisk/video-frame")
def api_yadisk_video_frame(path: str, frame_idx: int = 1, max_dim: int = 960) -> FileResponse:
    """
    Извлекает кадр из видео на YaDisk: скачивает во временный файл, вызывает video_keyframes.py.
    Для больших видео (>150MB) возвращает 413.
    """
    if not isinstance(path, str) or not path.startswith("disk:"):
        raise HTTPException(status_code=400, detail="path must start with disk:")

    idx = int(frame_idx or 0)
    if idx not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="frame_idx must be 1..3")

    md = int(max_dim or 0)
    md = max(128, min(2048, md))

    rr = APP_DIR.parent.parent
    py = rr / ".venv-face" / "Scripts" / "python.exe"
    script = rr / "backend" / "scripts" / "tools" / "video_keyframes.py"
    if not py.exists():
        raise HTTPException(status_code=500, detail=f"Missing .venv-face python: {py}")
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {script}")

    disk = get_disk()
    p_norm = _normalize_yadisk_path(path)
    try:
        md_meta = _yd_call_retry(lambda: disk.get_meta(p_norm, limit=0))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"File not found on YaDisk: {e}") from e

    size_val = getattr(md_meta, "size", None)
    size_i = int(size_val) if size_val is not None else None
    if size_i is not None and size_i > _YD_VIDEO_FRAME_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large for preview (>{_YD_VIDEO_FRAME_MAX_BYTES // (1024 * 1024)}MB)",
        )

    cache_dir = rr / "data" / "cache" / "video_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(f"yd|{path}|{idx}|{md}".encode("utf-8", errors="ignore")).hexdigest()
    out_path = cache_dir / f"yd_{cache_key}.jpg"
    if out_path.exists():
        return FileResponse(path=str(out_path), media_type="image/jpeg", filename=out_path.name, headers={"Cache-Control": "private, max-age=3600"})

    with tempfile.NamedTemporaryFile(prefix="ps_ydvf_", suffix=".mp4", delete=False) as tmp_video:
        tmp_video_path = tmp_video.name
    with tempfile.NamedTemporaryFile(prefix="ps_ydvf_", suffix=".jpg", delete=False) as tmp_jpg:
        tmp_jpg_path = tmp_jpg.name

    try:
        _yd_call_retry_timeout(lambda: disk.download(p_norm, tmp_video_path), timeout_sec=120)
    except Exception as e:  # noqa: BLE001
        try:
            os.remove(tmp_video_path)
        except Exception:
            pass
        try:
            os.remove(tmp_jpg_path)
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"YaDisk download failed: {type(e).__name__}: {e}") from e

    try:
        prc = subprocess.run(
            [
                str(py),
                str(script.relative_to(rr)),
                "--mode", "extract",
                "--path", tmp_video_path,
                "--samples", "3",
                "--frame-idx", str(idx),
                "--max-dim", str(md),
                "--out", tmp_jpg_path,
            ],
            capture_output=True,
            text=True,
            cwd=str(rr),
            timeout=60,  # noqa: S603
        )
    except subprocess.TimeoutExpired:
        try:
            os.remove(tmp_video_path)
            os.remove(tmp_jpg_path)
        except Exception:
            pass
        raise HTTPException(status_code=504, detail="frame_extract_timeout") from None
    finally:
        try:
            os.remove(tmp_video_path)
        except Exception:
            pass

    if prc.returncode != 0 or not os.path.isfile(tmp_jpg_path):
        try:
            os.remove(tmp_jpg_path)
        except Exception:
            pass
        msg = (prc.stderr or prc.stdout or "").strip()[:400]
        raise HTTPException(status_code=500, detail=f"frame_extract_failed: {msg or prc.returncode}")

    try:
        os.replace(tmp_jpg_path, str(out_path))
    except Exception:
        out_path = Path(tmp_jpg_path)

    return FileResponse(path=str(out_path), media_type="image/jpeg", filename=out_path.name, headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/local/preview")
def api_local_preview(path: str, pipeline_run_id: int | None = None) -> Response:
    """
    Preview для локальных файлов.
    Принимает path в формате `local:C:\\...\\file.ext`.
    ВАЖНО: отдаём файл только если он лежит внутри текущего source root (последний scope=source).
    """
    if not isinstance(path, str) or not path.startswith("local:"):
        raise HTTPException(status_code=400, detail="Only local: paths are supported")

    root_path = ""
    # 1) Если явно указан pipeline_run_id — доверяем root_path этого прогона конвейера сортировки исходной папки.
    if pipeline_run_id is not None:
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        finally:
            ps.close()
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        root_path = str(pr.get("root_path") or "")
    # 2) Fallback: legacy-логика (последний scope=source)
    if not root_path:
        store = DedupStore()
        try:
            src_latest = store.get_latest_run(scope="source")
            if not src_latest:
                raise HTTPException(status_code=400, detail="Нет активного source run.")
            root_path = str(src_latest.get("root_path") or "")
        finally:
            store.close()

    file_path = _strip_local_prefix(path)
    if not file_path:
        raise HTTPException(status_code=400, detail="Empty local path")
    if root_path and not _local_is_under_root(file_path=file_path, root_dir=root_path):
        raise HTTPException(status_code=403, detail="Path is outside active source root")

    # минимальные защиты
    try:
        abs_path = os.path.abspath(file_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Bad path") from None

    # Если файл не найден по указанному пути, пытаемся найти его по имени в других папках внутри root
    if not os.path.isfile(abs_path) and root_path:
        root_abs = os.path.abspath(root_path)
        file_name = os.path.basename(abs_path)
        # Ищем файл по имени в корне и всех подпапках
        if os.path.exists(root_abs) and os.path.isdir(root_abs):
            for root_dir, dirs, files in os.walk(root_abs):
                if file_name in files:
                    potential_path = os.path.join(root_dir, file_name)
                    if os.path.isfile(potential_path):
                        abs_path = potential_path
                        break

    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found")

    mime_type, _enc = mimetypes.guess_type(abs_path)
    media_type = mime_type or "application/octet-stream"
    # FileResponse поддерживает Range-запросы (полезно для <video>)
    return FileResponse(path=abs_path, media_type=media_type, filename=os.path.basename(abs_path))


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