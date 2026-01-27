"""
API для работы с кластерами лиц и справочником персон.
"""

from typing import Any
from pathlib import Path
import urllib.parse
import subprocess
import json
import os
import logging
from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.common.db import get_connection, FaceStore, PipelineStore
from backend.common.yadisk_client import get_disk
from backend.logic.face_recognition import (
    get_cluster_info,
    assign_cluster_to_person,
    cluster_face_embeddings,
    remove_face_from_cluster,
    find_closest_cluster_for_face,
    find_closest_cluster_with_person_for_face,
    find_closest_cluster_with_person_for_face_by_min_distance,
    find_similar_single_face_clusters,
    find_small_clusters_to_merge_in_person,
    find_optimal_clusters_to_merge_in_person,
    merge_clusters,
)

# Локальные копии функций для работы с YaDisk (чтобы избежать циклического импорта)
def _normalize_yadisk_path(path: str) -> str:
    """Нормализует путь YaDisk: disk:/... -> /..."""
    p = path or ""
    if p.startswith("disk:"):
        p = p[len("disk:"):]
    if not p.startswith("/"):
        p = "/" + p
    return p

def _yd_call_retry(fn):
    """Простая обёртка для вызова YaDisk API (без retry для упрощения)"""
    return fn()

def _repo_root() -> Path:
    """Возвращает корень репозитория."""
    return APP_DIR.parent.parent

def _agent_dbg(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any] | None = None, run_id: str = "pre-fix") -> None:
    """
    Tiny NDJSON logger for debug-mode evidence. Writes to .cursor/debug.log.
    Never log secrets/PII.
    """
    import time
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
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _generate_face_thumbnail_on_fly(file_path: str, bbox: tuple[int, int, int, int], thumb_size: int = 200) -> str | None:
    """
    Генерирует превью лица на лету из оригинального файла, если thumb_jpeg отсутствует в БД.
    
    Args:
        file_path: путь к файлу (disk:/... или local:...)
        bbox: координаты лица (x, y, w, h)
        thumb_size: размер превью (по умолчанию 200)
    
    Returns:
        base64-encoded JPEG thumbnail или None в случае ошибки
    """
    from PIL import Image
    from PIL import ImageOps
    from io import BytesIO
    import base64
    import requests
    
    try:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        
        # Загружаем изображение
        img = None
        if file_path.startswith("disk:"):
            disk = get_disk()
            p = _normalize_yadisk_path(file_path)
            md = _yd_call_retry(lambda: disk.get_meta(p, limit=0))
            sizes = getattr(md, "sizes", None)
            if sizes and "ORIGINAL" in sizes:
                original_url = sizes["ORIGINAL"]
                resp = requests.get(original_url, timeout=10, stream=True)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
            else:
                return None
        elif file_path.startswith("local:"):
            local_path = file_path.replace("local:", "")
            if not os.path.isfile(local_path):
                return None
            img = Image.open(local_path)
        else:
            return None
        
        if img is None:
            return None
        
        # Применяем EXIF transpose для правильной ориентации
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        
        # Генерируем кроп лица с padding
        iw, ih = img.size
        pad_ratio = 0.18
        pad = int(round(max(w, h) * pad_ratio))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(iw, x + w + pad)
        y1 = min(ih, y + h + pad)
        
        crop = img.crop((x0, y0, x1, y1)).convert("RGB")
        crop.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        
        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=78, optimize=True)
        thumb_bytes = buf.getvalue()
        
        img.close()
        
        return base64.b64encode(thumb_bytes).decode("utf-8")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error generating thumbnail on fly for {file_path}: {e}")
        return None

def _venv_face_python() -> Path:
    """Возвращает путь к Python из .venv-face."""
    rr = _repo_root()
    return rr / ".venv-face" / "Scripts" / "python.exe"

router = APIRouter()
APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Константа для специальной персоны "Посторонний" (или "Посторонние")
IGNORED_PERSON_NAME = "Посторонние"  # Для обратной совместимости, но используем ID

def get_outsider_person_id(conn) -> int | None:
    """
    Получает ID персоны "Посторонний" (ID = 6).
    Всегда использует фиксированный ID 6.
    Если персона с ID 6 не существует - создает её с именем "Посторонний".
    НЕ ищет по имени и НЕ создает дубликатов - всегда использует только ID 6.
    """
    cur = conn.cursor()
    # Проверяем существование персоны с ID 6
    cur.execute("SELECT id FROM persons WHERE id = 6")
    person_row = cur.fetchone()
    if person_row:
        return 6
    
    # Персона с ID 6 не найдена - создаем её
    # ВАЖНО: SQLite не позволяет явно указать ID в INSERT,
    # но если ID 6 свободен (был удален), автоинкремент может его использовать
    # Если нет - создастся персона с другим ID, что нежелательно
    # В этом случае нужно будет запустить миграцию для исправления
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO persons (name, mode, is_me, created_at, updated_at)
        VALUES ('Посторонний', 'active', 0, ?, ?)
    """, (now, now))
    conn.commit()
    created_id = cur.lastrowid
    
    # Если созданная персона получила ID 6 - отлично
    if created_id == 6:
        return 6
    
    # Если получила другой ID - это проблема, но возвращаем его для работоспособности
    # В логе можно будет увидеть, что нужна миграция
    import logging
    logging.warning(f"Персона 'Посторонний' создана с ID {created_id} вместо ожидаемого ID 6. Требуется миграция.")
    return created_id

# Группы персон с порядком сортировки
PERSON_GROUPS = {
    "Я и Супруга": {"order": 1},
    "Дети": {"order": 2},
    "Родственники": {"order": 3},
    "Синяя диагональ": {"order": 4},
    "Работа": {"order": 5},
}

def get_group_order(group_name: str | None) -> int | None:
    """Возвращает порядок группы по её названию. Если группы нет в PERSON_GROUPS, возвращает None."""
    if not group_name:
        return None
    group_info = PERSON_GROUPS.get(group_name)
    return group_info["order"] if group_info else None


@router.get("/face-clusters", response_class=HTMLResponse)
async def page_face_clusters(request: Request) -> Any:
    """Страница для просмотра и назначения кластеров лиц."""
    return templates.TemplateResponse("face_clusters.html", {"request": request})


@router.get("/persons", response_class=HTMLResponse)
async def page_persons_list(request: Request) -> Any:
    """Страница списка всех персон со статистикой."""
    return templates.TemplateResponse("persons_list.html", {"request": request})


@router.get("/persons/{person_id}", response_class=HTMLResponse)
async def page_person_detail(request: Request, person_id: int) -> Any:
    """Страница для просмотра и управления персоной."""
    return templates.TemplateResponse("person_detail.html", {"request": request, "person_id": person_id})


@router.get("/persons/{person_id}/clusters", response_class=HTMLResponse)
async def page_person_clusters(request: Request, person_id: int) -> Any:
    """Страница для просмотра кластеров конкретной персоны."""
    return templates.TemplateResponse("person_clusters.html", {"request": request, "person_id": person_id})


@router.get("/api/face-clusters/list")
async def api_face_clusters_list(*, run_id: int | None = None, archive_scope: str | None = None, person_id: int | None = None, unassigned_only: bool = False, show_run_only: bool = False, page: int = 1, page_size: int = 50) -> dict[str, Any]:
    """
    Получает список кластеров лиц, сгруппированных по персоне.
    
    Args:
        run_id: опционально, фильтр по run_id (для прогонов)
        archive_scope: опционально, фильтр по archive_scope (для архива, обычно 'archive')
        person_id: опционально, фильтр по person_id (для кластеров конкретной персоны)
        unassigned_only: если True, возвращает только неназначенные кластеры (person_id IS NULL)
        show_run_only: если True, возвращает только сортируемые кластеры (исключает архивные)
        page: номер страницы (начинается с 1, по умолчанию 1)
        page_size: размер страницы (по умолчанию 50)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие в зависимости от режима
    where_parts = []
    where_params = []
    
    if archive_scope == 'archive':
        where_parts.append("fc.archive_scope = 'archive'")
    elif show_run_only:
        # Только сортируемые (исключаем архивные)
        where_parts.append("(fc.archive_scope IS NULL OR fc.archive_scope != 'archive')")
        if run_id is not None:
            where_parts.append("fc.run_id = ?")
            where_params.append(run_id)
    elif run_id is not None:
        where_parts.append("fc.run_id = ?")
        where_params.append(run_id)
    
    # Фильтр по person_id (для кластеров конкретной персоны)
    if person_id is not None:
        where_parts.append("fc.person_id = ?")
        where_params.append(person_id)
    elif unassigned_only:
        # Только неназначенные кластеры (person_id IS NULL)
        where_parts.append("fc.person_id IS NULL")
    
    # Формируем финальное WHERE условие
    if where_parts:
        where_clause = " AND ".join(where_parts)
    else:
        where_clause = "1=1"
    
    # Получаем ID персоны "Посторонний" для ORDER BY
    outsider_person_id = get_outsider_person_id(conn)
    
    # Параметры для ORDER BY (используем ID вместо имени)
    params = tuple(where_params) + (outsider_person_id,)
    
    # Подсчет общего количества кластеров для пагинации
    # Для COUNT нужны только параметры WHERE (без outsider_person_id, который используется только в ORDER BY)
    count_params = tuple(where_params)
    
    cur_count = conn.cursor()
    cur_count.execute(
        f"""
        SELECT COUNT(DISTINCT fc.id) as total
        FROM face_clusters fc
        WHERE {where_clause}
        AND (SELECT COUNT(DISTINCT fr2.id)
             FROM photo_rectangles fr2
             WHERE fr2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) > 0
        """,
        count_params,
    )
    total_row = cur_count.fetchone()
    total_count = total_row["total"] if total_row else 0
    
    # Пагинация
    page = max(1, page)
    page_size = max(1, min(500, page_size))  # Ограничиваем размер страницы до 500
    offset = (page - 1) * page_size
    
    # Единый SQL-запрос для всех режимов с пагинацией
    cur.execute(
        f"""
        SELECT 
            fc.id, fc.run_id, fc.archive_scope, fc.method, fc.params_json, fc.created_at,
            (SELECT COUNT(DISTINCT fr2.id)
             FROM photo_rectangles fr2
             WHERE fr2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) as faces_count,
            COALESCE((SELECT COUNT(DISTINCT fr2.id)
             FROM photo_rectangles fr2
             WHERE fr2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0 
               AND (fr2.archive_scope = 'archive' 
                    OR (fr2.archive_scope IS NULL AND fc.archive_scope = 'archive'))), 0) as faces_count_archive,
            COALESCE((SELECT COUNT(DISTINCT fr2.id)
             FROM photo_rectangles fr2
             WHERE fr2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0 
               AND (fr2.archive_scope IS NULL OR fr2.archive_scope != 'archive') 
               AND fr2.run_id IS NOT NULL), 0) as faces_count_run,
            fc.person_id as person_id,
            p.name as person_name,
            p.avatar_face_id as avatar_face_id
        FROM face_clusters fc
        LEFT JOIN persons p ON fc.person_id = p.id
        WHERE {where_clause}
        AND (SELECT COUNT(DISTINCT fr2.id)
             FROM photo_rectangles fr2
             WHERE fr2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) > 0
        ORDER BY 
            CASE WHEN p.id = ? THEN 1 ELSE 0 END,
            person_name ASC, 
            faces_count DESC, 
            fc.created_at DESC
        LIMIT ? OFFSET ?
        """,
        params + (page_size, offset),
    )
    
    clusters = []
    for row in cur.fetchall():
        # person_ids теперь всегда один элемент (или пустой список) из face_clusters.person_id
        person_ids = [row["person_id"]] if row["person_id"] else []
        
        # Получаем preview-лицо (самое крупное) для кластера - только из архива
        preview_face_id = None
        cur_preview = conn.cursor()
        cur_preview.execute(
            """
            SELECT fr.id
            FROM photo_rectangles fr
            WHERE fr.cluster_id = ? 
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND fr.archive_scope = 'archive'
              AND fr.is_face = 1
            ORDER BY (fr.bbox_w * fr.bbox_h) DESC, fr.confidence DESC
            LIMIT 1
            """,
            (row["id"],),
        )
        preview_row = cur_preview.fetchone()
        if preview_row:
            preview_face_id = preview_row["id"]
        
        # Определяем тип кластера: архив или текущий прогон
        archive_scope = row["archive_scope"]
        run_id = row["run_id"]
        is_archive = archive_scope == 'archive' if archive_scope else False
        is_run = not is_archive and run_id is not None
        
        # Явно получаем значения счетчиков из результата запроса
        faces_count_val = row["faces_count"] if row["faces_count"] is not None else 0
        
        # Получаем значения для архив/прогон
        # Используем прямое обращение, так как sqlite3.Row не поддерживает .get()
        faces_count_archive_val = 0
        faces_count_run_val = 0
        
        try:
            if "faces_count_archive" in row.keys():
                faces_count_archive_val = row["faces_count_archive"]
                if faces_count_archive_val is None:
                    faces_count_archive_val = 0
                else:
                    faces_count_archive_val = int(faces_count_archive_val)
        except (KeyError, TypeError, ValueError):
            faces_count_archive_val = 0
        
        try:
            if "faces_count_run" in row.keys():
                faces_count_run_val = row["faces_count_run"]
                if faces_count_run_val is None:
                    faces_count_run_val = 0
                else:
                    faces_count_run_val = int(faces_count_run_val)
        except (KeyError, TypeError, ValueError):
            faces_count_run_val = 0
        
        # Создаем словарь с ВСЕМИ полями сразу
        clusters.append({
            "id": row["id"],
            "run_id": run_id,
            "archive_scope": archive_scope,
            "is_archive": is_archive,
            "is_run": is_run,
            "method": row["method"],
            "params_json": row["params_json"],
            "created_at": row["created_at"],
            "faces_count": faces_count_val,
            "faces_count_archive": int(faces_count_archive_val) if faces_count_archive_val is not None else 0,
            "faces_count_run": int(faces_count_run_val) if faces_count_run_val is not None else 0,
            "person_ids": person_ids,
            "person_id": row["person_id"],
            "person_name": row["person_name"],
            "avatar_face_id": row["avatar_face_id"],
            "preview_face_id": preview_face_id,  # ID самого крупного лица для preview
        })
    
    return {"clusters": clusters, "total": total_count, "page": page, "page_size": page_size}


@router.get("/api/face-clusters/suggestions-for-single")
async def api_face_clusters_suggestions_for_single(*, max_distance: float = 0.45, run_id: int | None = None, archive_scope: str | None = None) -> dict[str, Any]:
    """
    Находит предложения для кластеров с 1 лицом без персоны.
    
    Для каждого такого кластера находит ближайший кластер с персоной (исключая "Посторонние").
    
    Args:
        max_distance: максимальное косинусное расстояние для предложения (по умолчанию 0.35)
        run_id: опционально, фильтр по run_id
        archive_scope: опционально, фильтр по archive_scope (например, 'archive')
    
    Returns:
        dict с предложениями: {
            'suggestions': [
                {
                    'cluster_id': int,
                    'face_id': int,
                    'suggested_cluster_id': int,
                    'suggested_person_id': int,
                    'suggested_person_name': str,
                    'distance': float
                },
                ...
            ],
            'total': int
        }
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие для кластеров с 1 лицом без персоны
    where_parts = []
    where_params = []
    
    if archive_scope == 'archive':
        where_parts.append("fc.archive_scope = 'archive'")
    elif run_id is not None:
        where_parts.append("fc.run_id = ?")
        where_params.append(run_id)
    
    # Дополнительное условие для фильтра кластеров с 1 лицом и без персоны
    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    
    # Находим все кластеры с 1 лицом без персоны
    cur.execute(
        f"""
        SELECT 
            fc.id as cluster_id,
            fr.id as face_id
        FROM face_clusters fc
        JOIN photo_rectangles fr ON fr.cluster_id = fc.id
        WHERE {where_clause}
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND fr.embedding IS NOT NULL
          AND (SELECT COUNT(DISTINCT fr2.id)
               FROM photo_rectangles fr2
               WHERE fr2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) = 1
          AND fc.person_id IS NULL
        """,
        tuple(where_params),
    )
    
    single_face_clusters = cur.fetchall()
    
    suggestions = []
    
    # Для каждого кластера с 1 лицом ищем ближайший кластер с персоной
    for row in single_face_clusters:
        cluster_id = row["cluster_id"]
        face_id = row["face_id"]
        
        # Ищем ближайший кластер с персоной (исключая "Посторонние")
        # Используем версию с минимальным расстоянием (лучше для детей)
        suggestion = find_closest_cluster_with_person_for_face_by_min_distance(
            rectangle_id=face_id,
            exclude_cluster_id=cluster_id,
            max_distance=max_distance,
            ignored_person_name=IGNORED_PERSON_NAME,
        )
        
        if suggestion:
            suggestions.append({
                "cluster_id": cluster_id,
                "face_id": face_id,
                "suggested_cluster_id": suggestion["cluster_id"],
                "suggested_person_id": suggestion["person_id"],
                "suggested_person_name": suggestion["person_name"],
                "distance": suggestion["distance"],
            })
    
    # Сортируем по расстоянию (от меньшего к большему)
    suggestions.sort(key=lambda x: x["distance"])
    
    return {
        "suggestions": suggestions,
        "total": len(suggestions),
    }


@router.get("/api/face-clusters/similar-single-clusters")
async def api_face_clusters_similar_single(*, max_distance: float = 0.6, run_id: int | None = None, archive_scope: str | None = None) -> dict[str, Any]:
    """
    Находит пары похожих одиночных кластеров (без персоны) для объединения.
    
    Args:
        max_distance: максимальное косинусное расстояние для попадания в пару (по умолчанию 0.45)
        run_id: опционально, фильтр по run_id
        archive_scope: опционально, фильтр по archive_scope (например, 'archive')
    
    Returns:
        dict с парами похожих кластеров:
        {
            'pairs': [
                {
                    'cluster1_id': int,
                    'cluster1_face_id': int,
                    'cluster2_id': int,
                    'cluster2_face_id': int,
                    'distance': float
                },
                ...
            ],
            'total': int
        }
    """
    pairs = find_similar_single_face_clusters(
        max_distance=max_distance,
        run_id=run_id,
        archive_scope=archive_scope,
    )
    
    return {
        "pairs": pairs,
        "total": len(pairs),
    }


@router.get("/api/face-clusters/suggest-merge-small-clusters")
async def api_face_clusters_suggest_merge_small(
    *, max_size: int = 2, max_distance: float = 0.3, person_id: int | None = None,
    run_id: int | None = None, archive_scope: str | None = None
) -> dict[str, Any]:
    """
    Находит маленькие кластеры (1-2 фото) внутри персоны для возможного объединения.
    
    Для каждой персоны находит все маленькие кластеры и предлагает объединение,
    если минимальное расстояние между кластерами не превышает порог.
    
    Args:
        max_size: максимальный размер кластера для рассмотрения (по умолчанию 2)
        max_distance: максимальное косинусное расстояние для предложения объединения (по умолчанию 0.3)
        person_id: опционально, фильтр по person_id (только для одной персоны)
        run_id: опционально, фильтр по run_id
        archive_scope: опционально, фильтр по archive_scope (например, 'archive')
    
    Returns:
        dict с предложениями для объединения:
        {
            'suggestions': [
                {
                    'person_id': int,
                    'person_name': str,
                    'source_cluster_id': int,
                    'source_cluster_size': int,
                    'target_cluster_id': int,
                    'target_cluster_size': int,
                    'distance': float
                },
                ...
            ],
            'total': int
        }
    """
    suggestions = find_small_clusters_to_merge_in_person(
        max_size=max_size,
        max_distance=max_distance,
        person_id=person_id,
        run_id=run_id,
        archive_scope=archive_scope,
    )
    
    return {
        "suggestions": suggestions,
        "total": len(suggestions),
    }


@router.get("/api/face-clusters/suggest-optimal-merge")
async def api_face_clusters_suggest_optimal_merge(
    *, max_source_size: int = 4, max_distance: float = 0.3, person_id: int | None = None,
    run_id: int | None = None, archive_scope: str | None = None
) -> dict[str, Any]:
    """
    Находит оптимальные объединения кластеров для минимизации их количества.
    
    Для каждого маленького кластера находит ближайший кластер любого размера и предлагает
    объединение, если расстояние не превышает порог. Несколько маленьких кластеров могут
    объединяться в один большой.
    
    Args:
        max_source_size: максимальный размер кластера-источника (по умолчанию 4)
        max_distance: максимальное косинусное расстояние для предложения объединения (по умолчанию 0.3)
        person_id: опционально, фильтр по person_id (только для одной персоны)
        run_id: опционально, фильтр по run_id
        archive_scope: опционально, фильтр по archive_scope (например, 'archive')
    
    Returns:
        dict с предложениями для объединения:
        {
            'suggestions': [
                {
                    'person_id': int,
                    'person_name': str,
                    'source_cluster_id': int,
                    'source_cluster_size': int,
                    'target_cluster_id': int,
                    'target_cluster_size': int,
                    'distance': float
                },
                ...
            ],
            'total': int
        }
    """
    import logging
    from backend.logic.face_recognition import ML_AVAILABLE
    
    logger = logging.getLogger(__name__)
    logger.info(f"[api_face_clusters_suggest_optimal_merge] person_id={person_id}, max_source_size={max_source_size}, max_distance={max_distance}, ML_AVAILABLE={ML_AVAILABLE}")
    
    try:
        suggestions = find_optimal_clusters_to_merge_in_person(
            max_source_size=max_source_size,
            max_distance=max_distance,
            person_id=person_id,
            run_id=run_id,
            archive_scope=archive_scope,
        )
        
        logger.info(f"[api_face_clusters_suggest_optimal_merge] Найдено предложений: {len(suggestions)}")
        if len(suggestions) == 0:
            logger.warning(f"[api_face_clusters_suggest_optimal_merge] Функция вернула пустой список для person_id={person_id}")
    except Exception as e:
        logger.error(f"[api_face_clusters_suggest_optimal_merge] Ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to find suggestions: {str(e)}")
    
    return {
        "suggestions": suggestions,
        "total": len(suggestions),
    }


@router.post("/api/face-clusters/merge")
async def api_face_clusters_merge(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Объединяет два кластера: перемещает все лица из source_cluster в target_cluster.
    
    Args:
        source_cluster_id: ID кластера-источника
        target_cluster_id: ID целевого кластера
    """
    source_cluster_id = payload.get("source_cluster_id")
    target_cluster_id = payload.get("target_cluster_id")
    
    if source_cluster_id is None or target_cluster_id is None:
        raise HTTPException(status_code=400, detail="source_cluster_id and target_cluster_id are required")
    
    try:
        merge_clusters(
            source_cluster_id=int(source_cluster_id),
            target_cluster_id=int(target_cluster_id),
        )
        
        return {
            "status": "ok",
            "source_cluster_id": source_cluster_id,
            "target_cluster_id": target_cluster_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge clusters: {str(e)}")


@router.post("/api/persons/{person_id}/find-pending-merges")
async def api_persons_find_pending_merges(
    person_id: int,
    max_source_size: int = 4,
    max_distance: float = 0.3,
) -> dict[str, Any]:
    """
    Запускает скрипт для поиска кандидатов на объединение кластеров и сохраняет их в JSON.
    
    Args:
        person_id: ID персоны
        max_source_size: Максимальный размер маленького кластера (по умолчанию: 4)
        max_distance: Максимальное расстояние между кластерами (по умолчанию: 0.3)
    """
    py = _venv_face_python()
    if not py.exists():
        raise HTTPException(status_code=500, detail=f"Missing .venv-face python: {py}")
    
    script_path = _repo_root() / "backend" / "scripts" / "tools" / "find_pending_merges.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {script_path}")
    
    # Директория для сохранения JSON
    data_dir = _repo_root() / "backend" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Запускаем скрипт через subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root() / "backend")
    
    try:
        result = subprocess.run(
            [
                str(py),
                str(script_path.relative_to(_repo_root())),
                "--person-id", str(person_id),
                "--max-source-size", str(max_source_size),
                "--max-distance", str(max_distance),
                "--output-dir", str(data_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 минут максимум
            cwd=str(_repo_root()),
            env=env,
        )
        
        if result.returncode != 0:
            error_detail = result.stderr[:2000] if result.stderr else "Unknown error"
            raise HTTPException(
                status_code=500,
                detail=f"Script failed: {error_detail}",
            )
        
        # Читаем результат из JSON файла
        output_file = data_dir / f"pending_merges_person_{person_id}.json"
        if not output_file.exists():
            raise HTTPException(status_code=500, detail=f"Output file not found: {output_file}")
        
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return {
            "status": "ok",
            "person_id": person_id,
            "total": data.get("total", 0),
            "suggestions": data.get("suggestions", []),
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Script timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/api/persons/{person_id}/pending-merges")
async def api_persons_pending_merges(person_id: int) -> dict[str, Any]:
    """
    Получает список кандидатов на объединение из JSON файла.
    
    Args:
        person_id: ID персоны
    """
    data_dir = _repo_root() / "backend" / "data"
    output_file = data_dir / f"pending_merges_person_{person_id}.json"
    
    if not output_file.exists():
        return {
            "status": "not_found",
            "person_id": person_id,
            "total": 0,
            "suggestions": [],
        }
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return {
            "status": "ok",
            "person_id": person_id,
            "total": data.get("total", 0),
            "suggestions": data.get("suggestions", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading JSON: {str(e)}")


@router.post("/api/persons/{person_id}/apply-all-pending-merges")
async def api_persons_apply_all_pending_merges(person_id: int) -> dict[str, Any]:
    """
    Объединяет все найденные кандидаты на объединение кластеров.
    
    Args:
        person_id: ID персоны
    """
    # Читаем кандидатов из JSON
    data_dir = _repo_root() / "backend" / "data"
    output_file = data_dir / f"pending_merges_person_{person_id}.json"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail=f"No pending merges found for person {person_id}")
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        suggestions = data.get("suggestions", [])
        if not suggestions:
            return {
                "status": "ok",
                "person_id": person_id,
                "merged_count": 0,
                "message": "No suggestions to merge",
            }
        
        # Объединяем все кандидаты
        merged_count = 0
        errors = []
        
        for suggestion in suggestions:
            source_cluster_id = suggestion.get("source_cluster_id")
            target_cluster_id = suggestion.get("target_cluster_id")
            
            if source_cluster_id is None or target_cluster_id is None:
                continue
            
            try:
                merge_clusters(
                    source_cluster_id=int(source_cluster_id),
                    target_cluster_id=int(target_cluster_id),
                )
                merged_count += 1
            except Exception as e:
                errors.append({
                    "source_cluster_id": source_cluster_id,
                    "target_cluster_id": target_cluster_id,
                    "error": str(e),
                })
        
        # Удаляем JSON файл после успешного объединения
        if merged_count > 0:
            try:
                output_file.unlink()
            except Exception:
                pass  # Игнорируем ошибки удаления
        
        return {
            "status": "ok",
            "person_id": person_id,
            "merged_count": merged_count,
            "total": len(suggestions),
            "errors": errors,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/face-clusters/{cluster_id}", response_class=HTMLResponse)
async def page_face_cluster_detail(request: Request, cluster_id: int) -> Any:
    """Страница для детального просмотра кластера."""
    # #region agent log
    _agent_dbg(hypothesis_id="A", location="face_clusters.py:843", message="page_face_cluster_detail called", data={"cluster_id": cluster_id})
    # #endregion
    return templates.TemplateResponse("face_cluster_detail.html", {
        "request": request,
        "cluster_id": cluster_id,
    })


@router.get("/api/face-rectangles/{rectangle_id}/thumbnail")
async def api_face_rectangle_thumbnail(*, rectangle_id: int) -> dict[str, Any]:
    """Получает thumbnail лица для отображения аватара."""
    import base64
    import logging
    logger = logging.getLogger(__name__)
    
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT fr.thumb_jpeg, f.path as file_path, fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM photo_rectangles fr
        LEFT JOIN files f ON fr.file_id = f.id
        WHERE fr.id = ?
        """,
        (rectangle_id,),
    )
    
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    thumb_base64 = None
    if row["thumb_jpeg"]:
        thumb_base64 = base64.b64encode(row["thumb_jpeg"]).decode("utf-8")
    elif row["file_path"] and row["bbox_x"] is not None and row["bbox_y"] is not None and row["bbox_w"] is not None and row["bbox_h"] is not None:
        # Генерируем превью на лету, если thumb_jpeg отсутствует
        try:
            thumb_base64 = _generate_face_thumbnail_on_fly(
                file_path=row["file_path"],
                bbox=(row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
            )
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail on fly for face_id={rectangle_id}: {e}")
    
    if not thumb_base64:
        raise HTTPException(status_code=404, detail="Face rectangle has no thumbnail and cannot generate one")
    
    return {"thumb_jpeg_base64": thumb_base64}


@router.post("/api/debug/log")
async def api_debug_log(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Endpoint для приёма логов с клиента (JavaScript)."""
    # #region agent log
    _agent_dbg(
        hypothesis_id=payload.get("hypothesisId", "D"),
        location=payload.get("location", "unknown"),
        message=payload.get("message", "client log"),
        data=payload.get("data", {}),
        run_id=payload.get("runId", "pre-fix")
    )
    # #endregion
    return {"ok": True}

@router.get("/api/face-clusters/{cluster_id}")
async def api_face_cluster_info(*, cluster_id: int, limit: int | None = None) -> dict[str, Any]:
    """
    Получает детальную информацию о кластере.
    
    Args:
        cluster_id: ID кластера
        limit: максимальное количество лиц (None = все лица)
    """
    # #region agent log
    _agent_dbg(hypothesis_id="B", location="face_clusters.py:895", message="api_face_cluster_info called", data={"cluster_id": cluster_id, "limit": limit})
    # #endregion
    import base64
    try:
        info = get_cluster_info(cluster_id=cluster_id, limit=limit)
        # #region agent log
        _agent_dbg(
            hypothesis_id="B",
            location="face_clusters.py:907",
            message="get_cluster_info returned",
            data={
                "cluster_id": info.get("cluster_id"),
                "total_faces": info.get("total_faces"),
                "faces_count": len(info.get("faces", [])),
                "persons_count": len(info.get("persons", []))
            }
        )
        # #endregion
        # thumb_jpeg уже конвертирован в base64 в get_cluster_info, дополнительная обработка не нужна
        return info
    except Exception as e:
        # #region agent log
        _agent_dbg(hypothesis_id="B", location="face_clusters.py:909", message="api_face_cluster_info exception", data={"error": str(e), "error_type": type(e).__name__})
        # #endregion
        raise HTTPException(status_code=404, detail=f"Cluster not found: {e}")


@router.get("/api/persons/list")
async def api_persons_list() -> dict[str, Any]:
    """Получает список персон из справочника. Автоматически создаёт персону "Посторонний" если её нет."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем или создаем персону "Посторонний" один раз
    outsider_person_id = get_outsider_person_id(conn)
    if outsider_person_id:
        print(f"Персона 'Посторонний' найдена (ID: {outsider_person_id})")
    
    cur.execute(
        """
        SELECT id, name, mode, is_me, kinship, avatar_face_id, created_at, updated_at, "group", group_order
        FROM persons
        ORDER BY COALESCE(group_order, 999) ASC, name ASC
        """
    )
    
    persons = []
    for row in cur.fetchall():
        persons.append({
            "id": row["id"],
            "name": row["name"],
            "mode": row["mode"],
            "is_me": row["is_me"],
            "kinship": row["kinship"],
            "avatar_face_id": row["avatar_face_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "group": row["group"],
            "group_order": row["group_order"],
            "is_ignored": row["id"] == outsider_person_id,  # Флаг для персоны "Посторонний"
        })
    
    return {"persons": persons}


@router.get("/api/persons/stats")
async def api_persons_stats() -> dict[str, Any]:
    """
    Возвращает список персон со статистикой: количество кластеров и лиц.
    """
    import time
    import traceback
    metrics = {}
    total_start = time.time()
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Получаем ID персоны "Посторонний" один раз
        outsider_person_id = get_outsider_person_id(conn)
        
        # Оптимизированный запрос: используем JOIN и агрегацию вместо множественных подзапросов
        # Сначала получаем базовую информацию о персонах
        step_start = time.time()
        cur.execute(
        """
        SELECT 
            p.id,
            p.name,
            p.avatar_face_id,
            p."group",
            p.group_order
        FROM persons p
        ORDER BY 
            CASE WHEN p.id = ? THEN 1 ELSE 0 END,
            COALESCE(p.group_order, 999) ASC,
            p.name ASC
        """,
        (outsider_person_id,),
    )
    
        person_rows = cur.fetchall()
        metrics["step1_get_persons"] = time.time() - step_start
        
        # Теперь для каждой персоны получаем статистику через отдельные оптимизированные запросы
        # Это быстрее, чем множественные подзапросы в одном SELECT
        persons_data = {}
        for row in person_rows:
            person_id = row["id"]
            persons_data[person_id] = {
                "id": person_id,
                "name": row["name"],
                "avatar_face_id": row["avatar_face_id"],
                "group": row["group"],
                "group_order": row["group_order"],
                "clusters_count": 0,
                "clusters_count_archive": 0,
                "clusters_count_run": 0,
                "faces_count": 0,
                "faces_count_archive": 0,
                "faces_count_run": 0,
                # Разбивка по способам привязки
                "faces_via_clusters": 0,
                "faces_via_clusters_archive": 0,
                "faces_via_clusters_run": 0,
                "faces_via_manual": 0,
                "faces_via_manual_archive": 0,
                "faces_via_manual_run": 0,
                "person_rectangles_files_count": 0,
                "person_rectangles_files_count_archive": 0,
                "person_rectangles_files_count_run": 0,
                "file_persons_files_count": 0,
                "file_persons_files_count_archive": 0,
                "file_persons_files_count_run": 0,
            }
    
        # Получаем статистику кластеров одним запросом
        step_start = time.time()
        cur.execute("""
        SELECT 
            fc.person_id,
            COUNT(DISTINCT CASE WHEN fc.archive_scope = 'archive' THEN fc.id END) as clusters_archive,
            COUNT(DISTINCT CASE WHEN (fc.archive_scope IS NULL OR fc.archive_scope = '') AND fc.run_id IS NOT NULL THEN fc.id END) as clusters_run,
            COUNT(DISTINCT fc.id) as clusters_total
        FROM face_clusters fc
        WHERE fc.person_id IS NOT NULL
        GROUP BY fc.person_id
    """)
        for row in cur.fetchall():
            if row["person_id"] in persons_data:
                persons_data[row["person_id"]]["clusters_count"] = row["clusters_total"] or 0
                persons_data[row["person_id"]]["clusters_count_archive"] = row["clusters_archive"] or 0
                persons_data[row["person_id"]]["clusters_count_run"] = row["clusters_run"] or 0
        metrics["step2_get_clusters_stats"] = time.time() - step_start
        
        # Получаем статистику лиц через кластеры одним запросом
        # Оптимизация: начинаем с face_clusters (есть индекс по person_id), затем JOIN
        step_start = time.time()
        cur.execute("""
        SELECT 
            fc.person_id,
            COUNT(DISTINCT CASE WHEN fr.archive_scope = 'archive' THEN fr.id END) as faces_archive,
            COUNT(DISTINCT CASE WHEN (fr.archive_scope IS NULL OR fr.archive_scope = '') AND fr.run_id IS NOT NULL THEN fr.id END) as faces_run,
            COUNT(DISTINCT fr.id) as faces_total
        FROM face_clusters fc
        JOIN photo_rectangles fr ON fr.cluster_id = fc.id
        WHERE fc.person_id IS NOT NULL 
          AND fr.is_face = 1
          AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY fc.person_id
    """)
        for row in cur.fetchall():
            if row["person_id"] in persons_data:
                clusters_total = row["faces_total"] or 0
                clusters_archive = row["faces_archive"] or 0
                clusters_run = row["faces_run"] or 0
                persons_data[row["person_id"]]["faces_via_clusters"] = clusters_total
                persons_data[row["person_id"]]["faces_via_clusters_archive"] = clusters_archive
                persons_data[row["person_id"]]["faces_via_clusters_run"] = clusters_run
                persons_data[row["person_id"]]["faces_count"] = (persons_data[row["person_id"]]["faces_count"] or 0) + clusters_total
                persons_data[row["person_id"]]["faces_count_archive"] = (persons_data[row["person_id"]]["faces_count_archive"] or 0) + clusters_archive
                persons_data[row["person_id"]]["faces_count_run"] = (persons_data[row["person_id"]]["faces_count_run"] or 0) + clusters_run
        metrics["step3_get_faces_via_clusters"] = time.time() - step_start
        
        # Получаем статистику лиц через ручные привязки (исключая те, что уже в кластерах этой персоны)
        # Используем LEFT JOIN вместо NOT EXISTS для лучшей производительности
        # ВАЖНО: учитываем только лица (is_face=1), не персоны (is_face=0)
        step_start = time.time()
        cur.execute("""
        SELECT 
            fr.manual_person_id AS person_id,
            COUNT(DISTINCT CASE WHEN fr.archive_scope = 'archive' AND fc_check.person_id IS NULL THEN fr.id END) as faces_archive,
            COUNT(DISTINCT CASE WHEN (fr.archive_scope IS NULL OR fr.archive_scope = '') AND fr.run_id IS NOT NULL AND fc_check.person_id IS NULL THEN fr.id END) as faces_run,
            COUNT(DISTINCT CASE WHEN fc_check.person_id IS NULL THEN fr.id END) as faces_total
        FROM photo_rectangles fr
        LEFT JOIN face_clusters fc_check ON fc_check.id = fr.cluster_id AND fc_check.person_id = fr.manual_person_id
        WHERE fr.manual_person_id IS NOT NULL AND fr.is_face = 1
          AND COALESCE(fr.ignore_flag, 0) = 0
        GROUP BY fr.manual_person_id
    """)
        for row in cur.fetchall():
            if row["person_id"] in persons_data:
                manual_total = row["faces_total"] or 0
                manual_archive = row["faces_archive"] or 0
                manual_run = row["faces_run"] or 0
                persons_data[row["person_id"]]["faces_via_manual"] = manual_total
                persons_data[row["person_id"]]["faces_via_manual_archive"] = manual_archive
                persons_data[row["person_id"]]["faces_via_manual_run"] = manual_run
                persons_data[row["person_id"]]["faces_count"] = (persons_data[row["person_id"]]["faces_count"] or 0) + manual_total
                persons_data[row["person_id"]]["faces_count_archive"] = (persons_data[row["person_id"]]["faces_count_archive"] or 0) + manual_archive
                persons_data[row["person_id"]]["faces_count_run"] = (persons_data[row["person_id"]]["faces_count_run"] or 0) + manual_run
        metrics["step4_get_faces_via_manual"] = time.time() - step_start
    
        # Получаем статистику через прямоугольники без лица (ручные) и прямую привязку (person_rectangles удалена)
        step_start = time.time()
        # 1. Через photo_rectangles.manual_person_id с is_face=0 (прямоугольники «без лица» для архивных файлов)
        cur.execute("""
            SELECT 
                fr.manual_person_id AS person_id,
                COUNT(DISTINCT CASE WHEN fr.archive_scope = 'archive' THEN fr.file_id END) as files_archive,
                COUNT(DISTINCT CASE WHEN (fr.archive_scope IS NULL OR fr.archive_scope = '') AND fr.run_id IS NOT NULL THEN fr.file_id END) as files_run,
                COUNT(DISTINCT fr.file_id) as files_total
            FROM photo_rectangles fr
            WHERE fr.manual_person_id IS NOT NULL AND fr.is_face = 0
              AND COALESCE(fr.ignore_flag, 0) = 0
            GROUP BY fr.manual_person_id
        """)
        for row in cur.fetchall():
            person_id = row["person_id"]
            if person_id in persons_data:
                persons_data[person_id]["person_rectangles_files_count"] = (persons_data[person_id].get("person_rectangles_files_count", 0) or 0) + (row["files_total"] or 0)
                persons_data[person_id]["person_rectangles_files_count_archive"] = (persons_data[person_id].get("person_rectangles_files_count_archive", 0) or 0) + (row["files_archive"] or 0)
                persons_data[person_id]["person_rectangles_files_count_run"] = (persons_data[person_id].get("person_rectangles_files_count_run", 0) or 0) + (row["files_run"] or 0)
        
        metrics["step5_get_person_rectangles"] = time.time() - step_start
    
        step_start = time.time()
        # Проверяем существование таблицы file_persons
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_persons'")
        if cur.fetchone():
            # Определяем архив/прогон через JOIN с files по file_id
            cur.execute("""
            SELECT 
                fp.person_id,
                COUNT(DISTINCT CASE WHEN f.path LIKE 'disk:/Фото%' THEN fp.file_id END) as files_archive,
                COUNT(DISTINCT CASE WHEN f.path NOT LIKE 'disk:/Фото%' AND fp.pipeline_run_id IS NOT NULL THEN fp.file_id END) as files_run,
                COUNT(DISTINCT fp.file_id) as files_total
            FROM file_persons fp
            LEFT JOIN files f ON fp.file_id = f.id
            GROUP BY fp.person_id
        """)
            for row in cur.fetchall():
                if row["person_id"] in persons_data:
                    persons_data[row["person_id"]]["file_persons_files_count"] = row["files_total"] or 0
                    persons_data[row["person_id"]]["file_persons_files_count_archive"] = row["files_archive"] or 0
                    persons_data[row["person_id"]]["file_persons_files_count_run"] = row["files_run"] or 0
        metrics["step6_get_file_persons"] = time.time() - step_start
    
        # Формируем финальный список персон с аватарами
        step_start = time.time()
        persons = []
        avatar_queries_count = 0
        for person_id, person_data in persons_data.items():
            # Получаем preview для аватара, если есть
            avatar_face_id = person_data["avatar_face_id"]
            avatar_preview_url = None
            if avatar_face_id:
                avatar_queries_count += 1
                cur.execute(
                    """
                    SELECT f.path as file_path, fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
                    FROM photo_rectangles fr
                    LEFT JOIN files f ON fr.file_id = f.id
                    WHERE fr.id = ?
                    """,
                    (avatar_face_id,),
                )
                avatar_row = cur.fetchone()
                if avatar_row:
                    file_path = avatar_row["file_path"]
                    if file_path and file_path.startswith("disk:"):
                        avatar_preview_url = f"/api/yadisk/preview-image?size=M&path={urllib.parse.quote(file_path)}"
                    elif file_path and file_path.startswith("local:"):
                        avatar_preview_url = f"/api/local/preview?path={urllib.parse.quote(file_path)}"
            
            persons.append({
            "id": person_data["id"],
            "name": person_data["name"],
            "avatar_face_id": avatar_face_id,
            "avatar_preview_url": avatar_preview_url,
            "clusters_count": person_data["clusters_count"],
            "clusters_count_archive": person_data["clusters_count_archive"],
            "clusters_count_run": person_data["clusters_count_run"],
            "faces_count": person_data["faces_count"],
            "faces_count_archive": person_data["faces_count_archive"],
            "faces_count_run": person_data["faces_count_run"],
            # Разбивка по способам привязки
            "faces_via_clusters": person_data["faces_via_clusters"],
            "faces_via_clusters_archive": person_data["faces_via_clusters_archive"],
            "faces_via_clusters_run": person_data["faces_via_clusters_run"],
            "faces_via_manual": person_data["faces_via_manual"],
            "faces_via_manual_archive": person_data["faces_via_manual_archive"],
            "faces_via_manual_run": person_data["faces_via_manual_run"],
            "person_rectangles_files_count": person_data["person_rectangles_files_count"],
            "person_rectangles_files_count_archive": person_data["person_rectangles_files_count_archive"],
            "person_rectangles_files_count_run": person_data["person_rectangles_files_count_run"],
            "file_persons_files_count": person_data["file_persons_files_count"],
            "file_persons_files_count_archive": person_data["file_persons_files_count_archive"],
            "file_persons_files_count_run": person_data["file_persons_files_count_run"],
            "group": person_data["group"],
            "group_order": person_data["group_order"],
                "is_ignored": person_data["id"] == outsider_person_id,
            })
        metrics["step7_build_final_list"] = time.time() - step_start
        metrics["avatar_queries_count"] = avatar_queries_count
    
        total_time = time.time() - total_start
        metrics["total_time"] = total_time
        
        # Логируем метрики для отладки
        logger = logging.getLogger(__name__)
        logger.info(f"[api_persons_stats] Metrics: total={total_time:.3f}s, steps={metrics}, persons_count={len(persons)}")
        
        result = {"persons": persons}
        
        # Добавляем метрики производительности под ключом debug (только если включен debug режим)
        # Для простоты всегда добавляем, но можно проверять через env переменную
        result["debug"] = {"metrics": metrics}
        
        return result
    except Exception as e:
        logger = logging.getLogger(__name__)
        error_msg = f"Error in api_persons_stats: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/persons/{person_id}")
async def api_person_detail(*, person_id: int) -> dict[str, Any]:
    # #region agent log
    log_path = r"c:\Projects\PhotoSorter\.cursor\debug.log"
    # #endregion
    """Получает детальную информацию о персоне и все её лица."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Получаем информацию о персоне
        cur.execute(
            """
            SELECT id, name, mode, is_me, kinship, avatar_face_id, created_at, updated_at, "group", group_order
            FROM persons
            WHERE id = ?
            """,
            (person_id,),
        )
        
        person_row = cur.fetchone()
        if not person_row:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Собираем все типы привязки персоны
        all_items = []
        
        # 1. Через кластеры: photo_rectangles.cluster_id -> face_clusters -> person_id
        cur.execute(
            """
            SELECT DISTINCT
                fr.id as face_id,
                fr.run_id,
                fr.archive_scope,
                f.path as file_path,
                f.id as file_id,
                fr.face_index,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                fr.confidence, fr.presence_score,
                fr.thumb_jpeg,
                fr.embedding IS NOT NULL as has_embedding,
                fr.cluster_id,
                fc.run_id as cluster_run_id,
                fc.archive_scope as cluster_archive_scope,
                pr.id as pipeline_run_id
            FROM photo_rectangles fr
            JOIN face_clusters fc ON fc.id = fr.cluster_id
            LEFT JOIN files f ON fr.file_id = f.id
            LEFT JOIN pipeline_runs pr ON pr.face_run_id = fr.run_id
            WHERE fc.person_id = ? 
              AND fr.is_face = 1
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND (f.status IS NULL OR f.status != 'deleted')
            """,
            (person_id,),
        )
        
        for row in cur.fetchall():
            all_items.append({
                "row": dict(row),
                "assignment_type": "cluster",
            })
        
        # 2. Через ручные привязки (photo_rectangles.manual_person_id) — только лица (is_face=1)
        cur.execute(
            """
            SELECT DISTINCT
                fr.id as face_id,
                fr.run_id,
                fr.archive_scope,
                f.path as file_path,
                f.id as file_id,
                fr.face_index,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                fr.confidence, fr.presence_score,
                fr.thumb_jpeg,
                fr.embedding IS NOT NULL as has_embedding,
                NULL as cluster_id,
                NULL as cluster_run_id,
                NULL as cluster_archive_scope,
                pr.id as pipeline_run_id
            FROM photo_rectangles fr
            LEFT JOIN files f ON fr.file_id = f.id
            LEFT JOIN pipeline_runs pr ON pr.face_run_id = fr.run_id
            WHERE fr.manual_person_id = ?
              AND fr.is_face = 1
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND (f.status IS NULL OR f.status != 'deleted')
              AND NOT EXISTS (
                  SELECT 1 FROM face_clusters fc2 WHERE fc2.id = fr.cluster_id AND fc2.person_id = ?
              )
            """,
            (person_id, person_id),
        )
        
        for row in cur.fetchall():
            all_items.append({
                "row": dict(row),
                "assignment_type": "manual_face",
            })
        
        # 3. Через photo_rectangles.manual_person_id с is_face=0 (прямоугольники без лица)
        fs = FaceStore()
        try:
            fs_cur = fs.conn.cursor()
            fs_cur.execute(
                """
                SELECT DISTINCT
                    NULL as person_rectangle_id,
                    fr.id as face_id,
                    fr.run_id,
                    fr.archive_scope,
                    f.path as file_path,
                    f.id as file_id,
                    fr.face_index,
                    fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                    fr.confidence, fr.presence_score,
                    fr.thumb_jpeg,
                    fr.embedding IS NOT NULL as has_embedding,
                    NULL as cluster_id,
                    NULL as cluster_run_id,
                    NULL as cluster_archive_scope,
                    pr.id as pipeline_run_id
                FROM photo_rectangles fr
                LEFT JOIN files f ON fr.file_id = f.id
                LEFT JOIN pipeline_runs pr ON pr.face_run_id = fr.run_id
                WHERE fr.manual_person_id = ?
                  AND fr.is_face = 0
                  AND COALESCE(fr.ignore_flag, 0) = 0
                  AND (f.status IS NULL OR f.status != 'deleted')
                """,
                (person_id,),
            )
            rows = fs_cur.fetchall()
        finally:
            fs.close()
        
        processed_count = 0
        for row in rows:
            # #region agent log
            try:
                row_dict = dict(row)
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps({"location":"face_clusters.py:1470","message":"Processing person_rectangle row","data":{"person_id":person_id,"person_rectangle_id":row_dict.get("person_rectangle_id"),"file_path":row_dict.get("file_path"),"pipeline_run_id":row_dict.get("pipeline_run_id"),"file_id":row_dict.get("file_id")},"timestamp":__import__("time").time()*1000,"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + "\n")
            except: pass
            # #endregion
            all_items.append({
                "row": dict(row),
                "assignment_type": "person_rectangle",
            })
            processed_count += 1
        # #region agent log
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps({"location":"face_clusters.py:1475","message":"After processing person_rectangles","data":{"person_id":person_id,"processed_count":processed_count,"all_items_count":len(all_items)},"timestamp":__import__("time").time()*1000,"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + "\n")
        except: pass
        # #endregion
        
        # 4. Через file_persons (прямая привязка файла к персоне)
        cur.execute(
            """
            SELECT DISTINCT
                NULL as face_id,
                NULL as run_id,
                CASE WHEN f.path LIKE 'disk:/Фото%' THEN 'archive' ELSE NULL END as archive_scope,
                f.path as file_path,
                fp.file_id,
                0 as face_index,
                NULL as bbox_x,
                NULL as bbox_y,
                NULL as bbox_w,
                NULL as bbox_h,
                NULL as confidence,
                NULL as presence_score,
                NULL as thumb_jpeg,
                0 as has_embedding,
                NULL as cluster_id,
                NULL as cluster_run_id,
                NULL as cluster_archive_scope,
                fp.pipeline_run_id
            FROM file_persons fp
            LEFT JOIN files f ON fp.file_id = f.id
            WHERE fp.person_id = ?
              AND (f.status IS NULL OR f.status != 'deleted')
            """,
            (person_id,),
        )
        
        for row in cur.fetchall():
            all_items.append({
                "row": dict(row),
                "assignment_type": "file_person",
            })
        
        # Сортируем все элементы
        all_items.sort(key=lambda x: (
            # Сначала архивные, потом прогоны
            0 if (x["row"].get("archive_scope") == 'archive' or 
                  (x["row"].get("file_path") and x["row"]["file_path"].startswith("disk:/Фото"))) else 1,
            # Затем по пути
            x["row"].get("file_path") or "",
            # Затем по индексу лица
            x["row"].get("face_index") or 0,
        ))
        faces = []
        row_count = 0
        for item in all_items:
            row = item["row"]
            assignment_type = item["assignment_type"]
            row_count += 1
            try:
                thumb_base64 = None
                try:
                    thumb_jpeg_val = row["thumb_jpeg"]
                    if thumb_jpeg_val:
                        import base64
                        thumb_base64 = base64.b64encode(thumb_jpeg_val).decode("utf-8")
                except (KeyError, TypeError):
                    thumb_jpeg_val = None
                
                # Если thumb_jpeg отсутствует, пытаемся сгенерировать превью на лету
                if not thumb_base64:
                    try:
                        file_path_val = row["file_path"]
                        bbox_x_val = row["bbox_x"]
                        bbox_y_val = row["bbox_y"]
                        bbox_w_val = row["bbox_w"]
                        bbox_h_val = row["bbox_h"]
                        if file_path_val and bbox_x_val is not None and bbox_y_val is not None and bbox_w_val is not None and bbox_h_val is not None:
                            try:
                                thumb_base64 = _generate_face_thumbnail_on_fly(
                                    file_path=file_path_val,
                                    bbox=(bbox_x_val, bbox_y_val, bbox_w_val, bbox_h_val)
                                )
                            except Exception as e:
                                try:
                                    face_id_val = row["face_id"]
                                except (KeyError, TypeError):
                                    face_id_val = "unknown"
                                logger.warning(f"Failed to generate thumbnail on fly for face_id={face_id_val}: {e}")
                                thumb_base64 = None
                    except (KeyError, TypeError):
                        thumb_base64 = None
                
                # Определяем, относится ли элемент к архиву или текущему прогону
                try:
                    archive_scope_val = row["archive_scope"]
                except (KeyError, TypeError):
                    archive_scope_val = None
                
                try:
                    file_path_val = row["file_path"]
                except (KeyError, TypeError):
                    file_path_val = None
                
                try:
                    run_id_val = row["run_id"]
                except (KeyError, TypeError):
                    run_id_val = None
                
                try:
                    pipeline_run_id_val = row["pipeline_run_id"]
                except (KeyError, TypeError):
                    pipeline_run_id_val = None
                
                # Извлекаем cluster_run_id и cluster_archive_scope для элементов из кластеров
                try:
                    cluster_run_id_val = row["cluster_run_id"]
                except (KeyError, TypeError):
                    cluster_run_id_val = None
                
                try:
                    cluster_archive_scope_val = row["cluster_archive_scope"]
                except (KeyError, TypeError):
                    cluster_archive_scope_val = None
                
                # Определяем is_archive и is_run в зависимости от типа привязки
                if assignment_type == "cluster":
                    # Для кластеров используем данные из face_clusters, а не из photo_rectangles
                    is_archive = (cluster_archive_scope_val == 'archive') if cluster_archive_scope_val else False
                    is_run = not is_archive and (cluster_run_id_val is not None)
                elif assignment_type == "manual_face":
                    # Для ручных привязок лиц используем данные из photo_rectangles
                    is_archive = (archive_scope_val == 'archive') if archive_scope_val else False
                    is_run = not is_archive and (run_id_val is not None)
                else:
                    # Для file_persons (и бывших person_rectangle) определяем через file_path
                    is_archive = (file_path_val and file_path_val.startswith("disk:/Фото")) if file_path_val else False
                    is_run = not is_archive and (pipeline_run_id_val is not None)
            except Exception as e:
                # Логируем ошибку для отладки
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Error processing face row {row_count}: {e}\nRow keys: {list(row.keys()) if hasattr(row, 'keys') else 'N/A'}\n{error_trace}")
                continue
            
            try:
                bbox_x = row["bbox_x"] if row["bbox_x"] is not None else 0
                bbox_y = row["bbox_y"] if row["bbox_y"] is not None else 0
                bbox_w = row["bbox_w"] if row["bbox_w"] is not None else 0
                bbox_h = row["bbox_h"] if row["bbox_h"] is not None else 0
                confidence = row["confidence"] if row["confidence"] is not None else 0.0
                
                # Обрабатываем presence_score: может быть None, inf, -inf
                presence_score_val = row["presence_score"]
                if presence_score_val is not None:
                    import math
                    if math.isinf(presence_score_val) or math.isnan(presence_score_val):
                        presence_score = None
                    else:
                        presence_score = presence_score_val
                else:
                    presence_score = None
                    
                cluster_id = row["cluster_id"] if row["cluster_id"] is not None else None
                cluster_run_id = row["cluster_run_id"] if row["cluster_run_id"] is not None else None
            except (KeyError, TypeError) as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Error extracting row values for row {row_count}: {e}\nRow keys: {list(row.keys()) if hasattr(row, 'keys') else 'N/A'}\n{error_trace}")
                continue
            
            # Безопасное извлечение значений из row
            try:
                face_id_val = row.get("face_id")
                person_rectangle_id_val = row.get("person_rectangle_id")
            except (KeyError, TypeError):
                face_id_val = None
                person_rectangle_id_val = None
            
            try:
                file_id_val = row["file_id"] if row["file_id"] is not None else None
            except (KeyError, TypeError):
                file_id_val = None
            
            try:
                face_index_val = row["face_index"]
            except (KeyError, TypeError):
                face_index_val = 0
            
            try:
                has_embedding_val = bool(row["has_embedding"])
            except (KeyError, TypeError):
                has_embedding_val = False
            
            try:
                pipeline_run_id_val = row["pipeline_run_id"] if row["pipeline_run_id"] is not None else None
            except (KeyError, TypeError):
                pipeline_run_id_val = None
            
            # Определяем source (archive или run)
            if assignment_type in ("cluster", "manual_face"):
                # Для лиц определяем через archive_scope
                source = "archive" if is_archive else ("run" if is_run else "unknown")
            else:
                # Для file_persons определяем через file_path и pipeline_run_id (person_rectangles удалена)
                # Используем уже вычисленные is_archive и is_run для консистентности
                if is_archive:
                    source = "archive"
                elif is_run:
                    source = "run"
                else:
                    # Fallback: проверяем напрямую, если is_archive/is_run не установлены
                    if file_path_val and file_path_val.startswith("disk:/Фото"):
                        source = "archive"
                    elif pipeline_run_id_val is not None:
                        source = "run"
                    else:
                        source = "unknown"
            # #region agent log
            if assignment_type == "person_rectangle":
                try:
                    with open(log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps({"location":"face_clusters.py:1665","message":"Determined source for person_rectangle","data":{"assignment_type":assignment_type,"source":source,"is_archive":is_archive,"is_run":is_run,"file_path":file_path_val,"pipeline_run_id":pipeline_run_id_val},"timestamp":__import__("time").time()*1000,"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + "\n")
                except: pass
            # #endregion
            
            faces.append({
                "face_id": face_id_val,
                "person_rectangle_id": person_rectangle_id_val,
                "assignment_type": assignment_type,
                "source": source,
                "run_id": run_id_val,
                "archive_scope": archive_scope_val,
                "is_archive": is_archive,
                "is_run": is_run,
                "file_path": file_path_val,
                "file_id": file_id_val,
                "face_index": face_index_val,
                "bbox": {
                    "x": bbox_x,
                    "y": bbox_y,
                    "w": bbox_w,
                    "h": bbox_h,
                } if bbox_x is not None and bbox_y is not None and bbox_w is not None and bbox_h is not None else None,
                "confidence": confidence,
                "presence_score": presence_score,
                "thumb_jpeg_base64": thumb_base64,
                "has_embedding": has_embedding_val,
                "cluster_id": cluster_id,
                "cluster_run_id": cluster_run_id,
                "pipeline_run_id": pipeline_run_id_val,
            })
        
        return {
        "person": {
            "id": person_row["id"],
            "name": person_row["name"],
            "mode": person_row["mode"],
            "is_me": bool(person_row["is_me"]),
            "kinship": person_row["kinship"],
            "avatar_face_id": person_row["avatar_face_id"],
            "group": person_row["group"],
            "group_order": person_row["group_order"],
            "created_at": person_row["created_at"],
            "updated_at": person_row["updated_at"],
        },
            "faces": faces,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in api_person_detail for person_id={person_id}: {e}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/api/persons/{person_id}")
async def api_person_update(*, person_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Обновляет имя и группу персоны."""
    from datetime import datetime, timezone
    
    name = payload.get("name")
    group = payload.get("group")  # Может быть None для удаления группы
    
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, существует ли персона
    cur.execute("SELECT id FROM persons WHERE id = ?", (person_id,))
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Вычисляем group_order на основе группы
    group_order = get_group_order(group)
    
    # Обновляем имя и группу
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        UPDATE persons
        SET name = ?, "group" = ?, group_order = ?, updated_at = ?
        WHERE id = ?
        """,
        (name, group, group_order, now, person_id),
    )
    
    conn.commit()
    
    return {"status": "ok", "person_id": person_id, "name": name, "group": group, "group_order": group_order}


@router.post("/api/persons/{person_id}/set-avatar")
async def api_person_set_avatar(*, person_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Устанавливает аватар персоны (avatar_face_id)."""
    from datetime import datetime, timezone
    
    face_id = payload.get("face_id")
    if face_id is None:
        raise HTTPException(status_code=400, detail="Field 'face_id' is required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что персона существует
    cur.execute("SELECT id FROM persons WHERE id = ?", (person_id,))
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Проверяем, что лицо принадлежит персоне (через кластеры или ручные привязки)
    cur.execute(
        """
        SELECT 1
        FROM photo_rectangles fr
        JOIN face_clusters fc ON fc.id = fr.cluster_id
        WHERE fc.person_id = ? AND fr.id = ?
        
        UNION
        
        SELECT 1
        FROM photo_rectangles fr2
        WHERE fr2.id = ? AND fr2.manual_person_id = ?
        LIMIT 1
        """,
        (person_id, face_id, face_id, person_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=400, detail="Face does not belong to this person")
    
    # Обновляем аватар
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        UPDATE persons
        SET avatar_face_id = ?, updated_at = ?
        WHERE id = ?
        """,
        (face_id, now, person_id),
    )
    
    conn.commit()
    
    return {"status": "ok", "person_id": person_id, "avatar_face_id": face_id}


@router.post("/api/persons/{person_id}/faces/{face_id}/ignore")
async def api_person_face_ignore(*, person_id: int, face_id: int) -> dict[str, Any]:
    """Помечает лицо персоны как 'это не лицо' (устанавливает ignore_flag=1)."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что лицо принадлежит персоне (через кластеры или ручные привязки)
    cur.execute(
        """
        SELECT 1
        FROM photo_rectangles fr
        JOIN face_clusters fc ON fc.id = fr.cluster_id
        WHERE fc.person_id = ? AND fr.id = ?
        
        UNION
        
        SELECT 1
        FROM photo_rectangles fr2
        WHERE fr2.id = ? AND fr2.manual_person_id = ?
        LIMIT 1
        """,
        (person_id, face_id, face_id, person_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=400, detail="Face does not belong to this person")
    
    # Устанавливаем ignore_flag
    cur.execute(
        """
        UPDATE photo_rectangles
        SET ignore_flag = 1
        WHERE id = ?
        """,
        (face_id,),
    )
    
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    conn.commit()
    
    return {"status": "ok", "face_id": face_id}


@router.get("/api/persons/{person_id}/faces/{face_id}/similar")
async def api_person_face_similar(*, person_id: int, face_id: int, limit: int = 3) -> dict[str, Any]:
    """Находит похожие лица по embedding (3 ближайших по умолчанию)."""
    # Проверяем доступность ML
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_distances
        ml_available = True
    except ImportError:
        ml_available = False
    
    if not ml_available:
        return {"similar_faces": []}
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_distances
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем embedding текущего лица
    cur.execute(
        """
        SELECT embedding
        FROM photo_rectangles
        WHERE id = ? AND embedding IS NOT NULL
        """,
        (face_id,),
    )
    
    face_row = cur.fetchone()
    if not face_row or not face_row["embedding"]:
        return {"similar_faces": []}
    
    try:
        emb_json = face_row["embedding"]
        emb_list = json.loads(emb_json.decode("utf-8"))
        face_emb = np.array(emb_list, dtype=np.float32)
        
        # Нормализуем
        norm = np.linalg.norm(face_emb)
        if norm == 0:
            return {"similar_faces": []}
        face_emb_normalized = face_emb / norm
    except Exception:
        return {"similar_faces": []}
    
    # Получаем все лица с embeddings (исключаем текущее и игнорированные)
    cur.execute(
        """
        SELECT 
            fr.id,
            f.path as file_path,
            fr.face_index,
            fr.embedding,
            COALESCE(fr.manual_person_id, fc.person_id) as person_id,
            p.name as person_name
        FROM photo_rectangles fr
        LEFT JOIN files f ON fr.file_id = f.id
        LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
        LEFT JOIN persons p ON p.id = COALESCE(fr.manual_person_id, fc.person_id)
        WHERE fr.embedding IS NOT NULL
          AND fr.id != ?
          AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (face_id,),
    )
    
    similar_faces = []
    
    for row in cur.fetchall():
        try:
            emb_json = row["embedding"]
            emb_list = json.loads(emb_json.decode("utf-8"))
            emb_array = np.array(emb_list, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm == 0:
                continue
            emb_normalized = emb_array / norm
            
            # Вычисляем косинусное расстояние
            distance = cosine_distances([face_emb_normalized], [emb_normalized])[0][0]
            
            similar_faces.append({
                "face_id": row["id"],
                "file_path": row["file_path"],
                "face_index": row["face_index"],
                "person_id": row["person_id"],
                "person_name": row["person_name"],
                "distance": float(distance),
            })
        except Exception:
            continue
    
    # Сортируем по расстоянию и берём топ-N
    similar_faces.sort(key=lambda x: x["distance"])
    similar_faces = similar_faces[:limit]
    
    return {"similar_faces": similar_faces}


@router.post("/api/persons/{person_id}/faces/{face_id}/reassign")
async def api_person_face_reassign(*, person_id: int, face_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Переназначает лицо на другую персону или удаляет из текущей."""
    conn = get_connection()
    cur = conn.cursor()
    
    target_person_id = payload.get("target_person_id")  # Может быть None для удаления
    
    # Проверяем, что лицо принадлежит текущей персоне (через кластеры или ручные привязки)
    cur.execute(
        """
        SELECT 'cluster' as source, fr.cluster_id
        FROM photo_rectangles fr
        JOIN face_clusters fc ON fc.id = fr.cluster_id
        WHERE fc.person_id = ? AND fr.id = ?
        
        UNION
        
        SELECT 'manual' as source, NULL as cluster_id
        FROM photo_rectangles fr2
        WHERE fr2.id = ? AND fr2.manual_person_id = ?
        LIMIT 1
        """,
        (person_id, face_id, face_id, person_id),
    )
    label_row = cur.fetchone()
    if not label_row:
        raise HTTPException(status_code=400, detail="Face does not belong to this person")
    
    source_type = label_row["source"]
    cluster_id = label_row["cluster_id"]
    
    # Если лицо привязано через кластер — изменяем person_id кластера
    if source_type == "cluster" and cluster_id:
        if target_person_id is None:
            cur.execute("UPDATE face_clusters SET person_id = NULL WHERE id = ?", (cluster_id,))
        else:
            cur.execute("UPDATE face_clusters SET person_id = ? WHERE id = ?", (target_person_id, cluster_id))
    else:
        # Если лицо привязано вручную — работаем с photo_rectangles.manual_person_id
        if target_person_id is None:
            cur.execute(
                "UPDATE photo_rectangles SET manual_person_id = NULL, cluster_id = NULL WHERE id = ?",
                (face_id,),
            )
        else:
            cur.execute(
                "UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL WHERE id = ?",
                (target_person_id, face_id),
            )
    
    conn.commit()
    
    return {"status": "ok", "face_id": face_id, "target_person_id": target_person_id}


@router.post("/api/persons/{person_id}/faces/{face_id}/clear")
async def api_person_face_clear(*, person_id: int, face_id: int) -> dict[str, Any]:
    """Удаляет лицо из персоны и ищет ближайший кластер."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что лицо принадлежит персоне
    # Проверяем, принадлежит ли лицо персоне (через кластер или ручную привязку)
    cur.execute(
        """
        SELECT fc.id as cluster_id
        FROM photo_rectangles fr
        JOIN face_clusters fc ON fc.id = fr.cluster_id
        WHERE fr.id = ? AND fc.person_id = ?
        LIMIT 1
        """,
        (face_id, person_id),
    )
    cluster_row = cur.fetchone()
    
    # Проверяем ручную привязку (photo_rectangles.manual_person_id)
    cur.execute(
        "SELECT 1 FROM photo_rectangles WHERE id = ? AND manual_person_id = ? LIMIT 1",
        (face_id, person_id),
    )
    manual_row = cur.fetchone()
    
    if not cluster_row and not manual_row:
        raise HTTPException(status_code=400, detail="Face does not belong to this person")
    
    old_cluster_id = cluster_row["cluster_id"] if cluster_row else None
    
    if manual_row:
        cur.execute(
            "UPDATE photo_rectangles SET manual_person_id = NULL WHERE id = ?",
            (face_id,),
        )
    
    # Ищем ближайший кластер (исключая текущий)
    target_cluster_id = find_closest_cluster_for_face(
        rectangle_id=face_id,
        exclude_cluster_id=old_cluster_id,
        max_distance=0.3,
    )
    
    # Если найден кластер, добавляем лицо в него (если ещё не там)
    if target_cluster_id:
        cur.execute(
            """
            SELECT 1 FROM photo_rectangles WHERE id = ? AND cluster_id = ?
            """,
            (face_id, target_cluster_id),
        )
        if not cur.fetchone():
            cur.execute(
                """
                UPDATE photo_rectangles SET cluster_id = ? WHERE id = ?
                """,
                (target_cluster_id, face_id),
            )
    
    conn.commit()
    
    return {"status": "ok", "face_id": face_id, "target_cluster_id": target_cluster_id}


@router.post("/api/persons/{person_id}/refresh-gold")
async def api_person_refresh_gold(*, person_id: int) -> dict[str, Any]:
    """Обновляет gold-файл, перенося все лица персоны из всех её кластеров."""
    from backend.logic.gold.store import gold_faces_manual_rects_path, gold_read_ndjson_by_path, gold_write_ndjson_by_path
    
    person_name = payload.get("person_name")
    if not person_name:
        raise HTTPException(status_code=422, detail="person_name is required in request body")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем имя персоны
    cur.execute("SELECT name FROM persons WHERE id = ?", (person_id,))
    person_row = cur.fetchone()
    if not person_row:
        raise HTTPException(status_code=404, detail="Person not found")
    
    person_name = person_row["name"]
    
    # Получаем все лица персоны (ручные привязки + через кластеры)
    cur.execute(
        """
        SELECT DISTINCT
            fr.id, fr.run_id, f.path as file_path, fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM (
            -- Ручные привязки (photo_rectangles.manual_person_id)
            SELECT fr.id, fr.run_id, f.path as file_path, fr.face_index,
                   fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
            FROM photo_rectangles fr
            LEFT JOIN files f ON fr.file_id = f.id
            WHERE fr.manual_person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
            
            UNION
            
            -- Привязки через кластеры
            SELECT fr.id, fr.run_id, f.path as file_path, fr.face_index,
                   fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
            FROM photo_rectangles fr
            JOIN face_clusters fc ON fc.id = fr.cluster_id
            LEFT JOIN files f ON fr.file_id = f.id
            WHERE fc.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        ) fr
        """,
        (person_id, person_id),
    )
    
    faces = cur.fetchall()
    if not faces:
        raise HTTPException(status_code=400, detail="No faces for this person")
    
    # Группируем по file_path
    rects_by_path: dict[str, list[dict[str, int]]] = {}
    run_ids_by_path: dict[str, int] = {}
    
    for face in faces:
        path = face["file_path"]
        if path not in rects_by_path:
            rects_by_path[path] = []
            run_ids_by_path[path] = face["run_id"]
        
        rects_by_path[path].append({
            "x": face["bbox_x"],
            "y": face["bbox_y"],
            "w": face["bbox_w"],
            "h": face["bbox_h"],
        })
    
    # Читаем существующие manual rects
    gold_path = gold_faces_manual_rects_path()
    existing_dict = gold_read_ndjson_by_path(gold_path)  # dict[path, record], где path - это значение из поля "path" в JSON
    
    # Перезаписываем записи для файлов с лицами персоны (rewrite, не merge)
    updated_count = 0
    for file_path, rects in rects_by_path.items():
        run_id = run_ids_by_path[file_path]
        
        # Формируем запись для gold (полностью перезаписываем, не объединяем)
        # gold_write_ndjson_by_path сама добавит поле "path" из ключа словаря
        entry_data = {
            "run_id": run_id,
            "rects": rects,  # Только rects этой персоны, без объединения со старыми
        }
        
        # Ключ в словаре - это путь (будет использован как "path" в JSON)
        existing_dict[file_path] = entry_data
        updated_count += 1
    
    # Записываем обратно (gold_write_ndjson_by_path ожидает dict[path, record])
    # Функция сама добавит поле "path" из ключа словаря
    try:
        gold_write_ndjson_by_path(gold_path, existing_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write gold file: {str(e)}")
    
    return {
        "status": "ok",
        "person_id": person_id,
        "person_name": person_name,
        "files_updated": updated_count,
    }


@router.post("/api/persons/create")
async def api_persons_create(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Создаёт новую персону в справочнике."""
    import sqlite3
    import asyncio
    from datetime import datetime, timezone
    
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")
    
    mode = payload.get("mode", "active")
    is_me = payload.get("is_me", 0)
    kinship = payload.get("kinship")
    group = payload.get("group")  # Может быть None
    
    # Вычисляем group_order на основе группы
    group_order = get_group_order(group)
    
    now = datetime.now(timezone.utc).isoformat()
    
    # Retry логика для обработки временных блокировок БД
    max_retries = 5
    retry_delay = 0.5  # секунды
    
    for attempt in range(max_retries):
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            try:
                cur.execute(
                    """
                    INSERT INTO persons (name, mode, is_me, kinship, "group", group_order, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, mode, is_me, kinship, group, group_order, now, now),
                )
                person_id = cur.lastrowid
                conn.commit()
                
                return {"id": person_id, "name": name, "status": "created"}
            except sqlite3.OperationalError as e:
                conn.rollback()
                error_msg = str(e).lower()
                if "database is locked" in error_msg or "locked" in error_msg:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))  # Экспоненциальная задержка
                        continue
                    raise HTTPException(
                        status_code=503,
                        detail="Database is temporarily locked. Please try again in a few seconds."
                    )
                raise HTTPException(status_code=400, detail=f"Failed to create person: {e}")
            except Exception as e:
                conn.rollback()
                raise HTTPException(status_code=400, detail=f"Failed to create person: {e}")
            finally:
                conn.close()
        except HTTPException:
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            raise HTTPException(status_code=500, detail=f"Failed to create person after {max_retries} attempts: {e}")
    
    raise HTTPException(status_code=500, detail="Failed to create person: maximum retries exceeded")


@router.post("/api/face-clusters/{cluster_id}/assign-person")
async def api_assign_cluster_to_person(*, cluster_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Назначает кластер персоне или снимает назначение."""
    person_id = payload.get("person_id")
    # person_id может быть None для снятия назначения
    if person_id is not None:
        person_id = int(person_id)
    
    try:
        assign_cluster_to_person(cluster_id=cluster_id, person_id=person_id)
        return {"status": "ok", "cluster_id": cluster_id, "person_id": person_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to assign cluster: {e}")


@router.post("/api/face-clusters/clusterize")
async def api_clusterize(*, run_id: int, eps: float = 0.4, min_samples: int = 2) -> dict[str, Any]:
    """Запускает кластеризацию для указанного run_id."""
    try:
        result = cluster_face_embeddings(
            run_id=run_id,
            eps=eps,
            min_samples=min_samples,
            use_folder_context=True,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to clusterize: {e}")


@router.post("/api/face-clusters/{cluster_id}/remove-face")
async def api_remove_face_from_cluster(*, cluster_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Исключает лицо из кластера."""
    rectangle_id = payload.get("rectangle_id")
    if not rectangle_id:
        raise HTTPException(status_code=400, detail="rectangle_id is required")
    
    try:
        remove_face_from_cluster(cluster_id=cluster_id, rectangle_id=int(rectangle_id))
        return {"status": "ok", "cluster_id": cluster_id, "rectangle_id": int(rectangle_id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to remove face: {e}")


@router.post("/api/face-clusters/{cluster_id}/add-face")
async def api_add_face_to_cluster(*, cluster_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Добавляет лицо обратно в кластер (для undo)."""
    rectangle_id = payload.get("rectangle_id")
    if not rectangle_id:
        raise HTTPException(status_code=400, detail="rectangle_id is required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что лицо не уже в кластере
    cur.execute(
        """
        SELECT 1 FROM photo_rectangles WHERE id = ? AND cluster_id = ?
        """,
        (int(rectangle_id), cluster_id),
    )
    
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="Face already in cluster")
    
    # Добавляем обратно
    cur.execute(
        """
        UPDATE photo_rectangles SET cluster_id = ? WHERE id = ?
        """,
        (cluster_id, int(rectangle_id)),
    )
    
    conn.commit()
    
    return {"status": "ok", "cluster_id": cluster_id, "rectangle_id": int(rectangle_id)}


@router.post("/api/face-rectangles/{rectangle_id}/ignore")
async def api_ignore_face_rectangle(*, rectangle_id: int) -> dict[str, Any]:
    """Помечает лицо как 'это не лицо' (устанавливает ignore_flag=1)."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        UPDATE photo_rectangles
        SET ignore_flag = 1
        WHERE id = ?
        """,
        (rectangle_id,),
    )
    
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    conn.commit()
    
    return {"status": "ok", "rectangle_id": rectangle_id}


@router.post("/api/face-rectangles/{rectangle_id}/unignore")
async def api_unignore_face_rectangle(*, rectangle_id: int) -> dict[str, Any]:
    """Убирает пометку 'не лицо' (устанавливает ignore_flag=0, для undo)."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        UPDATE photo_rectangles
        SET ignore_flag = 0
        WHERE id = ?
        """,
        (rectangle_id,),
    )
    
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    conn.commit()
    
    return {"status": "ok", "rectangle_id": rectangle_id}


@router.post("/api/face-clusters/{cluster_id}/move-face-to-cluster")
async def api_move_face_to_cluster(*, cluster_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Переносит лицо в другой кластер.
    Если target_cluster_id не указан, автоматически находит ближайший кластер по embedding.
    Если ближайший кластер не найден, создаётся новый.
    """
    rectangle_id = payload.get("rectangle_id")
    if not rectangle_id:
        raise HTTPException(status_code=400, detail="rectangle_id is required")
    rectangle_id = int(rectangle_id)
    
    target_cluster_id = payload.get("target_cluster_id")
    if target_cluster_id is not None:
        target_cluster_id = int(target_cluster_id)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Удаляем из текущего кластера
    cur.execute(
        """
        UPDATE photo_rectangles SET cluster_id = NULL WHERE id = ? AND cluster_id = ?
        """,
        (rectangle_id, cluster_id),
    )
    
    # Если target_cluster_id не указан, ищем ближайший кластер
    if target_cluster_id is None:
        target_cluster_id = find_closest_cluster_for_face(
            rectangle_id=rectangle_id,
            exclude_cluster_id=cluster_id,
            max_distance=0.3,
        )
    
    # Если ближайший кластер не найден, создаём новый
    if target_cluster_id is None:
        # Получаем информацию о текущем кластере для создания нового с теми же параметрами
        cur.execute(
            """
            SELECT run_id, method, params_json
            FROM face_clusters
            WHERE id = ?
            """,
            (cluster_id,),
        )
        cluster_row = cur.fetchone()
        if not cluster_row:
            raise HTTPException(status_code=404, detail="Source cluster not found")
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        
        cur.execute(
            """
            INSERT INTO face_clusters (run_id, method, params_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (cluster_row["run_id"], cluster_row["method"], cluster_row["params_json"], now),
        )
        target_cluster_id = cur.lastrowid
    
    # Добавляем в целевой кластер (photo_rectangles.cluster_id)
    cur.execute(
        """
        UPDATE photo_rectangles SET cluster_id = ? WHERE id = ?
        """,
        (target_cluster_id, rectangle_id),
    )
    
    conn.commit()
    
    return {"status": "ok", "rectangle_id": rectangle_id, "target_cluster_id": target_cluster_id}


@router.post("/api/face-clusters/{cluster_id}/confirm-to-gold")
async def api_confirm_cluster_to_gold(*, cluster_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Записывает все лица кластера в gold с указанным именем персоны.
    Формат: {person_name}_gold (например, Agatha_gold).
    """
    from backend.logic.gold.store import gold_faces_manual_rects_path, gold_read_ndjson_by_path, gold_write_ndjson_by_path
    
    person_name = payload.get("person_name")
    if not person_name:
        raise HTTPException(status_code=422, detail="person_name is required in request body")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем все лица из кластера
    cur.execute(
        """
        SELECT 
            fr.id, fr.run_id, f.path as file_path, fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM photo_rectangles fr
        LEFT JOIN files f ON fr.file_id = f.id
        WHERE fr.cluster_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (cluster_id,),
    )
    
    faces = cur.fetchall()
    if not faces:
        raise HTTPException(status_code=400, detail="No faces in cluster")
    
    # Группируем по file_path
    rects_by_path: dict[str, list[dict[str, int]]] = {}
    run_ids_by_path: dict[str, int] = {}
    
    for face in faces:
        path = face["file_path"]
        if path not in rects_by_path:
            rects_by_path[path] = []
            run_ids_by_path[path] = face["run_id"]
        
        rects_by_path[path].append({
            "x": face["bbox_x"],
            "y": face["bbox_y"],
            "w": face["bbox_w"],
            "h": face["bbox_h"],
        })
    
    # Читаем существующие manual rects
    gold_path = gold_faces_manual_rects_path()
    existing = gold_read_ndjson_by_path(gold_path)
    
    # Перезаписываем записи для файлов кластера (rewrite, не merge)
    for path, rects in rects_by_path.items():
        run_id = run_ids_by_path[path]
        
        # Полностью перезаписываем rects для этого файла (только лица из этого кластера)
        existing[path] = {
            "path": path,
            "run_id": run_id,
            "rects": rects,  # В gold используется "rects"
        }
    
    # Записываем обратно
    gold_write_ndjson_by_path(gold_path, existing)
    
    return {
        "status": "ok",
        "cluster_id": cluster_id,
        "person_name": person_name,
        "paths_updated": len(rects_by_path),
        "faces_count": len(faces),
    }


@router.post("/api/persons/save-all-to-gold")
async def api_save_all_persons_to_gold() -> dict[str, Any]:
    """
    Сохраняет все лица всех назначенных персон в gold файл.
    Объединяет rects для одного файла от разных персон.
    """
    from backend.logic.gold.store import gold_faces_manual_rects_path, gold_read_ndjson_by_path, gold_write_ndjson_by_path
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем все персоны (кроме "Посторонний", если нужно)
    cur.execute(
        """
        SELECT id, name FROM persons
        ORDER BY name
        """
    )
    persons = cur.fetchall()
    
    if not persons:
        raise HTTPException(status_code=400, detail="No persons found")
    
    # Собираем все лица всех персон
    all_faces_by_path: dict[str, dict[str, Any]] = {}  # path -> {run_id, rects: []}
    
    for person in persons:
        person_id = person["id"]
        
        # Получаем все лица персоны (ручные привязки + через кластеры)
        cur.execute(
            """
            SELECT DISTINCT
                fr.id, fr.run_id, fr.file_path, fr.face_index,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
            FROM (
                -- Ручные привязки (photo_rectangles.manual_person_id)
                SELECT fr.id, fr.run_id, f.path as file_path, fr.face_index,
                       fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
                FROM photo_rectangles fr
                LEFT JOIN files f ON fr.file_id = f.id
                WHERE fr.manual_person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
                
                UNION
                
                -- Привязки через кластеры (photo_rectangles.cluster_id)
                SELECT fr.id, fr.run_id, f.path as file_path, fr.face_index,
                       fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
                FROM photo_rectangles fr
                JOIN face_clusters fc ON fc.id = fr.cluster_id
                LEFT JOIN files f ON fr.file_id = f.id
                WHERE fc.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
            ) fr
            """,
            (person_id, person_id),
        )
        
        faces = cur.fetchall()
        
        for face in faces:
            path = face["file_path"]
            if path not in all_faces_by_path:
                all_faces_by_path[path] = {
                    "run_id": face["run_id"],
                    "rects": [],
                }
            
            # Добавляем rect (объединяем, не перезаписываем)
            rect = {
                "x": face["bbox_x"],
                "y": face["bbox_y"],
                "w": face["bbox_w"],
                "h": face["bbox_h"],
            }
            # Проверяем, нет ли уже такого rect (избегаем дублей)
            rects = all_faces_by_path[path]["rects"]
            if rect not in rects:
                rects.append(rect)
    
    # Читаем существующие manual rects
    gold_path = gold_faces_manual_rects_path()
    existing_dict = gold_read_ndjson_by_path(gold_path)
    
    # Перезаписываем записи для всех файлов с назначенными лицами
    updated_count = 0
    total_faces = 0
    for file_path, data in all_faces_by_path.items():
        run_id = data["run_id"]
        rects = data["rects"]
        
        if not rects:
            continue
        
        # Формируем запись для gold (полностью перезаписываем для этого файла)
        entry_data = {
            "run_id": run_id,
            "rects": rects,  # Все rects от всех персон для этого файла
        }
        
        existing_dict[file_path] = entry_data
        updated_count += 1
        total_faces += len(rects)
    
    # Записываем обратно
    try:
        gold_write_ndjson_by_path(gold_path, existing_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write gold file: {str(e)}")
    
    return {
        "status": "ok",
        "persons_count": len(persons),
        "faces_count": total_faces,
        "files_updated": updated_count,
    }


@router.post("/api/persons/{person_id}/save-to-gold")
async def api_save_person_to_gold(*, person_id: int) -> dict[str, Any]:
    """
    Сохраняет все лица конкретной персоны в gold файл.
    Объединяет rects для одного файла от разных кластеров этой персоны.
    """
    from backend.logic.gold.store import gold_faces_manual_rects_path, gold_read_ndjson_by_path, gold_write_ndjson_by_path
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что персона существует
    cur.execute("SELECT id, name FROM persons WHERE id = ?", (person_id,))
    person = cur.fetchone()
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    
    # Получаем все лица персоны (ручные привязки + через кластеры)
    cur.execute(
        """
        SELECT DISTINCT
            fr.id, fr.run_id, f.path as file_path, fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM (
            -- Ручные привязки (photo_rectangles.manual_person_id)
            SELECT fr.id, fr.run_id, f.path as file_path, fr.face_index,
                   fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
            FROM photo_rectangles fr
            LEFT JOIN files f ON fr.file_id = f.id
            WHERE fr.manual_person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
            
            UNION
            
            -- Привязки через кластеры
            SELECT fr.id, fr.run_id, f.path as file_path, fr.face_index,
                   fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
            FROM photo_rectangles fr
            JOIN face_clusters fc ON fc.id = fr.cluster_id
            LEFT JOIN files f ON fr.file_id = f.id
            WHERE fc.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        ) fr
        """,
        (person_id, person_id),
    )
    
    faces = cur.fetchall()
    
    # Собираем все лица по файлам
    all_faces_by_path: dict[str, dict[str, Any]] = {}  # path -> {run_id, rects: []}
    
    for face in faces:
        path = face["file_path"]
        if path not in all_faces_by_path:
            all_faces_by_path[path] = {
                "run_id": face["run_id"],
                "rects": [],
            }
        
        # Добавляем rect (объединяем, не перезаписываем)
        rect = {
            "x": face["bbox_x"],
            "y": face["bbox_y"],
            "w": face["bbox_w"],
            "h": face["bbox_h"],
        }
        # Проверяем, нет ли уже такого rect (избегаем дублей)
        rects = all_faces_by_path[path]["rects"]
        if rect not in rects:
            rects.append(rect)
    
    # Читаем существующие manual rects
    gold_path = gold_faces_manual_rects_path()
    existing_dict = gold_read_ndjson_by_path(gold_path)
    
    # Перезаписываем записи для всех файлов с лицами этой персоны
    updated_count = 0
    total_faces = 0
    for file_path, data in all_faces_by_path.items():
        run_id = data["run_id"]
        rects = data["rects"]
        
        if not rects:
            continue
        
        # Формируем запись для gold (полностью перезаписываем для этого файла)
        entry_data = {
            "run_id": run_id,
            "rects": rects,  # Все rects этой персоны для этого файла
        }
        
        existing_dict[file_path] = entry_data
        updated_count += 1
        total_faces += len(rects)
    
    # Записываем обратно
    try:
        gold_write_ndjson_by_path(gold_path, existing_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write gold file: {str(e)}")
    
    return {
        "status": "ok",
        "person_id": person_id,
        "person_name": person["name"],
        "faces_count": total_faces,
        "files_updated": updated_count,
    }


@router.get("/api/face-runs/list")
async def api_face_runs_list() -> dict[str, Any]:
    """Получает список прогонов детекции лиц."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT id, scope, root_path, status, total_files, processed_files, faces_found, started_at, finished_at
        FROM face_runs
        ORDER BY started_at DESC
        LIMIT 50
        """
    )
    
    runs = []
    for row in cur.fetchall():
        runs.append({
            "id": row["id"],
            "scope": row["scope"],
            "root_path": row["root_path"],
            "status": row["status"],
            "total_files": row["total_files"],
            "processed_files": row["processed_files"],
            "faces_found": row["faces_found"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        })
    
    return {"runs": runs}


@router.get("/api/file-faces")
async def api_file_faces(file_id: int | None = None, file_path: str | None = None) -> dict[str, Any]:
    """Получает все лица на указанном файле с информацией о назначенных персонах.
    
    Приоритет: file_id (если передан), иначе file_path (если передан).
    """
    import logging
    from backend.common.db import _get_file_id
    
    logger = logging.getLogger(__name__)
    
    if file_id is None and file_path is None:
        raise HTTPException(status_code=400, detail="Either file_id or file_path must be provided")
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Получаем file_id
        resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Получаем все лица на этом файле
        # Фильтруем дубликаты: для одинаковых позиций (face_index + bbox) выбираем:
        # 1. Кластеризованное лицо (если есть)
        # 2. Иначе - лицо с максимальным run_id (последний прогон)
        # Персона определяется:
        # 1. Через ручные привязки: photo_rectangles.manual_person_id (приоритет)
        # 2. Через кластеры: face_clusters.person_id
        cur.execute(
        """
        WITH ranked_faces AS (
            SELECT 
                fr.id as face_id,
                fr.face_index,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                fr.run_id,
                fr.archive_scope,
                COALESCE(fr.manual_person_id, fc.person_id) as person_id,
                COALESCE(p_manual.name, p_cluster.name) as person_name,
                COALESCE(p_manual.is_me, p_cluster.is_me, 0) as is_me,
                fr.cluster_id,
                ROW_NUMBER() OVER (
                    PARTITION BY fr.file_id, fr.face_index, fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
                    ORDER BY 
                        CASE WHEN fr.cluster_id IS NOT NULL THEN 0 ELSE 1 END,
                        CASE WHEN fr.archive_scope = 'archive' THEN 0 ELSE 1 END,
                        CASE WHEN fr.run_id IS NULL THEN 1 ELSE 0 END,
                        fr.run_id DESC
                ) as rn
            FROM photo_rectangles fr
            LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
            LEFT JOIN persons p_cluster ON fc.person_id = p_cluster.id
            LEFT JOIN persons p_manual ON fr.manual_person_id = p_manual.id
            WHERE fr.file_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        )
        SELECT 
            face_id,
            face_index,
            bbox_x, bbox_y, bbox_w, bbox_h,
            person_id,
            person_name,
            is_me,
            cluster_id
        FROM ranked_faces
        WHERE rn = 1
        ORDER BY face_index
        """,
        (resolved_file_id,),
    )
    
        faces = []
        for row in cur.fetchall():
            # Проверяем, что bbox данные есть
            if row["bbox_x"] is None or row["bbox_y"] is None or row["bbox_w"] is None or row["bbox_h"] is None:
                logger.warning(f"Face {row['face_id']} has NULL bbox data, skipping")
                continue
                
            faces.append({
                "face_id": row["face_id"],
                "face_index": row["face_index"],
                "bbox": {
                    "x": row["bbox_x"],
                    "y": row["bbox_y"],
                    "w": row["bbox_w"],
                    "h": row["bbox_h"],
                },
                "person_id": row["person_id"],
                "person_name": row["person_name"],
                "is_me": bool(row["is_me"]) if row["is_me"] else False,
                "cluster_id": row["cluster_id"],
            })
        
        logger.info(f"api_file_faces: file_id={resolved_file_id}, file_path={file_path}, faces_count={len(faces)}")
        return {"faces": faces}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in api_file_faces for file_id={file_id}, file_path={file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()


@router.post("/api/face-rectangles/{rectangle_id}/find-cluster")
async def api_find_cluster_for_face(*, rectangle_id: int) -> dict[str, Any]:
    """Находит ближайший кластер для указанного лица."""
    from backend.logic.face_recognition import find_closest_cluster_for_face
    
    target_cluster_id = find_closest_cluster_for_face(
        rectangle_id=rectangle_id,
        exclude_cluster_id=None,
        max_distance=0.3,
    )
    
    if target_cluster_id is None:
        return {"cluster_id": None, "message": "Ближайший кластер не найден"}
    
    return {"cluster_id": target_cluster_id}


@router.post("/api/face-rectangles/{rectangle_id}/assign-person")
async def api_assign_person_to_face(*, rectangle_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Назначает персону напрямую на face_rectangle (без кластера)."""
    person_id = payload.get("person_id")
    if person_id is None:
        raise HTTPException(status_code=400, detail="person_id is required")
    person_id = int(person_id)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что лицо существует
    cur.execute(
        """
        SELECT id FROM photo_rectangles WHERE id = ?
        """,
        (rectangle_id,),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    # Получаем cluster_id для этого лица (если есть)
    cur.execute(
        """
        SELECT cluster_id FROM photo_rectangles WHERE id = ? LIMIT 1
        """,
        (rectangle_id,),
    )
    cluster_row = cur.fetchone()
    cluster_id = cluster_row["cluster_id"] if cluster_row else None
    
    # Назначаем ручную привязку в photo_rectangles.manual_person_id (и снимаем кластер)
    cur.execute(
        """
        UPDATE photo_rectangles SET manual_person_id = ?, cluster_id = NULL WHERE id = ?
        """,
        (person_id, rectangle_id),
    )
    
    conn.commit()
    
    # Получаем информацию о персоне для ответа
    cur.execute(
        """
        SELECT name FROM persons WHERE id = ?
        """,
        (person_id,),
    )
    person_row = cur.fetchone()
    person_name = person_row["name"] if person_row else None
    
    return {
        "status": "ok",
        "rectangle_id": rectangle_id,
        "person_id": person_id,
        "person_name": person_name,
        "cluster_id": cluster_id,
        "message": f"Персона '{person_name}' назначена на лицо. " + 
                   (f"Лицо в кластере #{cluster_id}" if cluster_id else "Лицо не в кластере (cluster_id = None)")
    }


@router.get("/api/image-dimensions")
async def api_image_dimensions(path: str, save: bool = True) -> dict[str, Any]:
    """
    Получает размеры исходного изображения для масштабирования bbox координат.
    Если save=True, сохраняет размеры в БД (таблица files).
    
    Args:
        path: путь к изображению (disk:/... или local:...)
        save: сохранять ли размеры в БД (по умолчанию True)
    """
    from PIL import Image
    import requests
    from io import BytesIO
    
    try:
        width = None
        height = None
        exif_orientation = None
        
        # Сначала проверяем БД на наличие размеров и EXIF orientation
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT image_width, image_height, exif_orientation
            FROM files
            WHERE path = ?
            """,
            (path,),
        )
        db_row = cur.fetchone()
        if db_row and db_row[0] and db_row[1]:
            width = db_row[0]
            height = db_row[1]
            exif_orientation = db_row[2]  # Может быть None, если не сохранено
        
        # Если размеры или EXIF orientation не найдены в БД, получаем из изображения
        if width is None or height is None or (exif_orientation is None and save):
            if path.startswith("disk:"):
                # Для YaDisk получаем размеры через ORIGINAL URL
                disk = get_disk()
                p = _normalize_yadisk_path(path)
                md = _yd_call_retry(lambda: disk.get_meta(p, limit=0))
                sizes = getattr(md, "sizes", None)
                if sizes and "ORIGINAL" in sizes:
                    # Загружаем оригинальное изображение и получаем его размеры
                    original_url = sizes["ORIGINAL"]
                    resp = requests.get(original_url, timeout=10, stream=True)
                    resp.raise_for_status()
                    # Читаем только заголовки для получения размеров
                    img = Image.open(BytesIO(resp.content))
                    # Получаем EXIF orientation до применения transpose (если еще не получен из БД)
                    if exif_orientation is None:
                        try:
                            from PIL.ExifTags import ORIENTATION
                            exif = img.getexif()
                            if exif is not None:
                                exif_orientation = exif.get(ORIENTATION)
                        except Exception:
                            pass
                    # Применяем EXIF transpose для получения правильных размеров
                    # (размеры должны соответствовать повернутому изображению)
                    try:
                        from PIL import ImageOps
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass
                    width, height = img.size
                else:
                    return {
                        "ok": False,
                        "error": "ORIGINAL size not available",
                        "path": path,
                    }
            elif path.startswith("local:"):
                # Для локальных файлов открываем напрямую
                local_path = path.replace("local:", "")
                img = Image.open(local_path)
                # Получаем EXIF orientation до применения transpose (если еще не получен из БД)
                if exif_orientation is None:
                    try:
                        from PIL.ExifTags import ORIENTATION
                        exif = img.getexif()
                        if exif is not None:
                            exif_orientation = exif.get(ORIENTATION)
                    except Exception:
                        pass
                # Применяем EXIF transpose для получения правильных размеров
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                width, height = img.size
                img.close()
            else:
                return {
                    "ok": False,
                    "error": f"Unsupported path format: {path}",
                    "path": path,
                }
        
        # Сохраняем размеры и EXIF orientation в БД, если они получены и save=True
        if width is not None and height is not None and save:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE files 
                SET image_width = ?, image_height = ?, exif_orientation = ?
                WHERE path = ? AND (image_width IS NULL OR image_height IS NULL OR exif_orientation IS NULL)
                """,
                (int(width), int(height), exif_orientation, path),
            )
            conn.commit()
        
        return {
            "ok": True,
            "width": width,
            "height": height,
            "exif_orientation": exif_orientation,  # EXIF orientation tag (1-8) или None
            "path": path,
            "saved": save and width is not None and height is not None,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "path": path,
        }


@router.post("/api/persons/assign-rectangle")
async def api_persons_assign_rectangle(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Устаревший эндпойнт: таблица person_rectangles удалена.
    Для привязки персоны к прямоугольнику без лица используйте photo_rectangles.manual_person_id (face UI).
    (назначение лица персоне в UI лиц).
    """
    raise HTTPException(
        status_code=410,
        detail="person_rectangles table removed. Use face assignment UI to assign person to a non-face rectangle (photo_rectangles.manual_person_id).",
    )


@router.post("/api/persons/assign-file")
async def api_persons_assign_file(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Назначает персону файлу напрямую (без прямоугольника) - file_persons.
    
    Параметры:
    - pipeline_run_id: int (обязательно)
    - file_id: int (опционально, приоритет над file_path)
    - file_path: str (опционально, fallback если file_id не передан)
    - person_id: int (обязательно)
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    file_path = payload.get("file_path")
    person_id = payload.get("person_id")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required and must be int")
    if file_id is None and file_path is None:
        raise HTTPException(status_code=400, detail="Either file_id or file_path must be provided")
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
    
    # Проверяем, что персона существует
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM persons WHERE id = ?", (int(person_id),))
    person_row = cur.fetchone()
    if not person_row:
        raise HTTPException(status_code=404, detail="person_id not found")
    conn.close()
    
    # Вставляем привязку
    # Получаем file_id если передан file_path
    file_id = payload.get("file_id")
    if file_id is None and file_path is None:
        raise HTTPException(status_code=400, detail="Either file_id or file_path must be provided")
    
    fs = FaceStore()
    try:
        fs.insert_file_person(
            pipeline_run_id=int(pipeline_run_id),
            file_id=file_id,
            file_path=file_path,
            person_id=int(person_id),
        )
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "file_id": file_id,
        "file_path": str(file_path) if file_path else None,
        "person_id": int(person_id),
        "person_name": person_row["name"],
    }


@router.post("/api/faces/file/assign-person")
async def api_faces_file_assign_person(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Назначает персону файлу напрямую (без прямоугольника) - file_persons.
    Поддерживает архивные файлы (без pipeline_run_id).
    
    Параметры:
    - pipeline_run_id: int (опционально, для сортируемых фото)
    - file_id: int (опционально, для архивных фото)
    - path: str (опционально, для архивных фото)
    - person_id: int (обязательно)
    """
    from backend.common.db import _get_file_id, get_connection
    
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    path = payload.get("path")
    person_id = payload.get("person_id")
    
    if not isinstance(person_id, int):
        raise HTTPException(status_code=400, detail="person_id is required and must be int")
    if file_id is None and path is None:
        raise HTTPException(status_code=400, detail="Either file_id or path must be provided")
    
    # Для сортируемых фото требуется pipeline_run_id
    if pipeline_run_id is not None:
        if not isinstance(pipeline_run_id, int):
            raise HTTPException(status_code=400, detail="pipeline_run_id must be int if provided")
        
        # Проверяем, что pipeline_run_id существует
        ps = PipelineStore()
        try:
            pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
            if not pr:
                raise HTTPException(status_code=404, detail="pipeline_run_id not found")
        finally:
            ps.close()
    
    # Проверяем, что персона существует
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM persons WHERE id = ?", (int(person_id),))
    person_row = cur.fetchone()
    if not person_row:
        conn.close()
        raise HTTPException(status_code=404, detail="person_id not found")
    
    # Получаем file_id
    resolved_file_id = _get_file_id(conn, file_id=file_id, file_path=path)
    if resolved_file_id is None:
        conn.close()
        raise HTTPException(status_code=404, detail="File not found")
    
    # Вставляем привязку
    fs = FaceStore()
    try:
        if pipeline_run_id is not None:
            # Для сортируемых фото
            fs.insert_file_person(
                pipeline_run_id=int(pipeline_run_id),
                file_id=resolved_file_id,
                file_path=path,
                person_id=int(person_id),
            )
        else:
            # Для архивных фото - вставляем напрямую в БД
            from datetime import datetime, timezone
            cur.execute("""
                INSERT OR REPLACE INTO file_persons (file_id, person_id, created_at)
                VALUES (?, ?, ?)
            """, (resolved_file_id, int(person_id), datetime.now(timezone.utc).isoformat()))
            conn.commit()
    finally:
        fs.close()
        conn.close()
    
    return {
        "ok": True,
        "pipeline_run_id": pipeline_run_id,
        "file_id": resolved_file_id,
        "path": path,
        "person_id": int(person_id),
        "person_name": person_row["name"],
    }


@router.post("/api/persons/remove-assignment")
async def api_persons_remove_assignment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Удаляет привязку к персоне (любого типа).
    
    Параметры:
    - assignment_type: str (обязательно) - "face", "person_rectangle", "file"
    - pipeline_run_id: int (обязательно для person_rectangle и file)
    - file_id: int (опционально, приоритет над file_path)
    - file_path: str (опционально, fallback если file_id не передан)
    - person_id: int (обязательно)
    - rectangle_id: int (обязательно для type="face")
    - rectangle_id: int (обязательно для type="person_rectangle")
    """
    from backend.common.db import _get_file_id
    
    assignment_type = payload.get("assignment_type")
    pipeline_run_id = payload.get("pipeline_run_id")
    file_id = payload.get("file_id")
    file_path = payload.get("file_path")
    person_id = payload.get("person_id")
    rectangle_id = payload.get("rectangle_id")
    rectangle_id = payload.get("rectangle_id")
    
    if not isinstance(assignment_type, str):
        raise HTTPException(status_code=400, detail="assignment_type is required and must be str")
    if assignment_type not in ("face", "person_rectangle", "file"):
        raise HTTPException(status_code=400, detail="assignment_type must be 'face', 'person_rectangle', or 'file'")
    if not isinstance(person_id, int):
        raise HTTPException(status_code=400, detail="person_id is required and must be int")
    
    fs = FaceStore()
    conn = get_connection()
    try:
        if assignment_type == "face":
            # Снимаем привязку через лицо (photo_rectangles.manual_person_id)
            if not isinstance(rectangle_id, int):
                raise HTTPException(status_code=400, detail="rectangle_id is required for type='face'")
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE photo_rectangles SET manual_person_id = NULL WHERE id = ? AND manual_person_id = ?
                """,
                (int(rectangle_id), int(person_id)),
            )
            conn.commit()
            
        elif assignment_type == "person_rectangle":
            # person_rectangles удалена; привязки «прямоугольник без лица» через photo_rectangles.manual_person_id (face UI)
            raise HTTPException(
                status_code=410,
                detail="person_rectangles removed. Remove assignment via face UI (photo_rectangles.manual_person_id).",
            )
            
        elif assignment_type == "file":
            # Удаляем прямую привязку файла
            if not isinstance(pipeline_run_id, int):
                raise HTTPException(status_code=400, detail="pipeline_run_id is required for type='file'")
            if file_id is None and file_path is None:
                raise HTTPException(status_code=400, detail="Either file_id or file_path is required for type='file'")
            fs.delete_file_person(
                pipeline_run_id=int(pipeline_run_id),
                file_id=file_id,
                file_path=file_path,
                person_id=int(person_id),
            )
    finally:
        fs.close()
        conn.close()
    
    return {
        "ok": True,
        "assignment_type": assignment_type,
        "person_id": int(person_id),
    }


@router.get("/api/persons/file-assignments")
async def api_persons_file_assignments(
    pipeline_run_id: int,
    file_id: int | None = None,
    file_path: str | None = None,
) -> dict[str, Any]:
    """
    Возвращает все привязки файла к персонам:
    - через лица (photo_rectangles.manual_person_id и photo_rectangles.cluster_id → face_clusters.person_id)
    - прямая привязка (file_persons).
    
    Приоритет: file_id (если передан), иначе file_path (если передан).
    """
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id must be int")
    if file_id is None and file_path is None:
        raise HTTPException(status_code=400, detail="Either file_id or file_path must be provided")
    
    # Проверяем, что pipeline_run_id существует
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
        if not pr:
            raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    finally:
        ps.close()
    
    fs = FaceStore()
    try:
        assignments = fs.get_file_all_assignments(
            pipeline_run_id=int(pipeline_run_id),
            file_id=file_id,
            file_path=file_path,
        )
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "file_id": file_id,
        "file_path": file_path,
        "assignments": assignments,
    }
