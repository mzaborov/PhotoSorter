"""
API для работы с кластерами лиц и справочником персон.
"""

from typing import Any
from pathlib import Path
import urllib.parse
import subprocess
import json
import os
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

def _venv_face_python() -> Path:
    """Возвращает путь к Python из .venv-face."""
    rr = _repo_root()
    return rr / ".venv-face" / "Scripts" / "python.exe"

router = APIRouter()
APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Константа для специальной персоны "Посторонние"
IGNORED_PERSON_NAME = "Посторонние"

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
async def api_face_clusters_list(*, run_id: int | None = None, archive_scope: str | None = None, person_id: int | None = None, unassigned_only: bool = False, page: int = 1, page_size: int = 50) -> dict[str, Any]:
    """
    Получает список кластеров лиц, сгруппированных по персоне.
    
    Args:
        run_id: опционально, фильтр по run_id (для прогонов)
        archive_scope: опционально, фильтр по archive_scope (для архива, обычно 'archive')
        person_id: опционально, фильтр по person_id (для кластеров конкретной персоны)
        unassigned_only: если True, возвращает только неназначенные кластеры (person_id IS NULL)
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
    elif run_id is not None:
        where_parts.append("fc.run_id = ?")
        where_params.append(run_id)
    
    # Фильтр по person_id (для кластеров конкретной персоны)
    if person_id is not None:
        where_parts.append("(SELECT p2.id FROM face_labels fl2 JOIN persons p2 ON fl2.person_id = p2.id JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id WHERE fcm_fl2.cluster_id = fc.id LIMIT 1) = ?")
        where_params.append(person_id)
    elif unassigned_only:
        # Только неназначенные кластеры (person_id IS NULL)
        where_parts.append("(SELECT p2.id FROM face_labels fl2 JOIN persons p2 ON fl2.person_id = p2.id JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id WHERE fcm_fl2.cluster_id = fc.id LIMIT 1) IS NULL")
    
    # Формируем финальное WHERE условие
    if where_parts:
        where_clause = " AND ".join(where_parts)
    else:
        where_clause = "1=1"
    
    # Параметры для ORDER BY (IGNORED_PERSON_NAME)
    params = tuple(where_params) + (IGNORED_PERSON_NAME,)
    
    # Подсчет общего количества кластеров для пагинации
    # Для COUNT нужны только параметры WHERE (без IGNORED_PERSON_NAME, который используется только в ORDER BY)
    count_params = tuple(where_params)
    
    cur_count = conn.cursor()
    cur_count.execute(
        f"""
        SELECT COUNT(DISTINCT fc.id) as total
        FROM face_clusters fc
        WHERE {where_clause}
        AND (SELECT COUNT(DISTINCT fcm2.face_rectangle_id)
             FROM face_cluster_members fcm2
             JOIN face_rectangles fr2 ON fcm2.face_rectangle_id = fr2.id
             WHERE fcm2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) > 0
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
            fc.id, fc.run_id, fc.method, fc.params_json, fc.created_at,
            (SELECT COUNT(DISTINCT fcm2.face_rectangle_id)
             FROM face_cluster_members fcm2
             JOIN face_rectangles fr2 ON fcm2.face_rectangle_id = fr2.id
             WHERE fcm2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) as faces_count,
            (SELECT GROUP_CONCAT(DISTINCT fl2.person_id)
             FROM face_labels fl2
             JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id
             WHERE fcm_fl2.cluster_id = fc.id) as person_ids,
            (SELECT p2.id
             FROM face_labels fl2
             JOIN persons p2 ON fl2.person_id = p2.id
             JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id
             WHERE fcm_fl2.cluster_id = fc.id
             LIMIT 1) as person_id,
            (SELECT p2.name
             FROM face_labels fl2
             JOIN persons p2 ON fl2.person_id = p2.id
             JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id
             WHERE fcm_fl2.cluster_id = fc.id
             LIMIT 1) as person_name,
            (SELECT p2.avatar_face_id
             FROM face_labels fl2
             JOIN persons p2 ON fl2.person_id = p2.id
             JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id
             WHERE fcm_fl2.cluster_id = fc.id
             LIMIT 1) as avatar_face_id
        FROM face_clusters fc
        WHERE {where_clause}
        AND (SELECT COUNT(DISTINCT fcm2.face_rectangle_id)
             FROM face_cluster_members fcm2
             JOIN face_rectangles fr2 ON fcm2.face_rectangle_id = fr2.id
             WHERE fcm2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) > 0
        ORDER BY 
            CASE WHEN (SELECT p2.name FROM face_labels fl2 JOIN persons p2 ON fl2.person_id = p2.id JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id WHERE fcm_fl2.cluster_id = fc.id LIMIT 1) = ? THEN 1 ELSE 0 END,
            person_name ASC, 
            faces_count DESC, 
            fc.created_at DESC
        LIMIT ? OFFSET ?
        """,
        params + (page_size, offset),
    )
    
    clusters = []
    for row in cur.fetchall():
        person_ids_str = row["person_ids"] or ""
        person_ids = [int(p) for p in person_ids_str.split(",") if p.strip()] if person_ids_str else []
        
        # Получаем preview-лицо (самое крупное) для кластера
        preview_face_id = None
        cur_preview = conn.cursor()
        cur_preview.execute(
            """
            SELECT fr.id
            FROM face_cluster_members fcm
            JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
            WHERE fcm.cluster_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
            ORDER BY (fr.bbox_w * fr.bbox_h) DESC, fr.confidence DESC
            LIMIT 1
            """,
            (row["id"],),
        )
        preview_row = cur_preview.fetchone()
        if preview_row:
            preview_face_id = preview_row["id"]
        
        clusters.append({
            "id": row["id"],
            "run_id": row["run_id"],
            "method": row["method"],
            "params_json": row["params_json"],
            "created_at": row["created_at"],
            "faces_count": row["faces_count"],
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
            fcm.face_rectangle_id as face_id
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE {where_clause}
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND fr.embedding IS NOT NULL
          AND (SELECT COUNT(DISTINCT fcm2.face_rectangle_id)
               FROM face_cluster_members fcm2
               JOIN face_rectangles fr2 ON fcm2.face_rectangle_id = fr2.id
               WHERE fcm2.cluster_id = fc.id AND COALESCE(fr2.ignore_flag, 0) = 0) = 1
          AND (SELECT p2.id 
               FROM face_labels fl2 
               JOIN persons p2 ON fl2.person_id = p2.id 
               JOIN face_cluster_members fcm_fl2 ON fl2.face_rectangle_id = fcm_fl2.face_rectangle_id
               WHERE fcm_fl2.cluster_id = fc.id 
               LIMIT 1) IS NULL
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
            face_rectangle_id=face_id,
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
    return templates.TemplateResponse("face_cluster_detail.html", {
        "request": request,
        "cluster_id": cluster_id,
    })


@router.get("/api/face-rectangles/{face_rectangle_id}/thumbnail")
async def api_face_rectangle_thumbnail(*, face_rectangle_id: int) -> dict[str, Any]:
    """Получает thumbnail лица для отображения аватара."""
    import base64
    
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT thumb_jpeg
        FROM face_rectangles
        WHERE id = ?
        """,
        (face_rectangle_id,),
    )
    
    row = cur.fetchone()
    if not row or not row["thumb_jpeg"]:
        raise HTTPException(status_code=404, detail="Face rectangle not found or no thumbnail")
    
    thumb_base64 = base64.b64encode(row["thumb_jpeg"]).decode("utf-8")
    
    return {"thumb_jpeg_base64": thumb_base64}


@router.get("/api/face-clusters/{cluster_id}")
async def api_face_cluster_info(*, cluster_id: int, limit: int | None = None) -> dict[str, Any]:
    """
    Получает детальную информацию о кластере.
    
    Args:
        cluster_id: ID кластера
        limit: максимальное количество лиц (None = все лица)
    """
    import base64
    try:
        info = get_cluster_info(cluster_id=cluster_id, limit=limit)
        # thumb_jpeg уже конвертирован в base64 в get_cluster_info, дополнительная обработка не нужна
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Cluster not found: {e}")


@router.get("/api/persons/list")
async def api_persons_list() -> dict[str, Any]:
    """Получает список персон из справочника. Автоматически создаёт персону "Посторонние" если её нет."""
    from datetime import datetime, timezone
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, есть ли персона "Посторонние"
    cur.execute(
        """
        SELECT id FROM persons WHERE name = ?
        """,
        (IGNORED_PERSON_NAME,),
    )
    ignored_person_row = cur.fetchone()
    
    # Если нет - создаём автоматически
    if not ignored_person_row:
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT INTO persons (name, mode, is_me, kinship, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (IGNORED_PERSON_NAME, "active", 0, None, now, now),
        )
        conn.commit()
        print(f"Автоматически создана персона '{IGNORED_PERSON_NAME}'")
    
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
            "is_ignored": row["name"] == IGNORED_PERSON_NAME,  # Флаг для персоны "Посторонние"
        })
    
    return {"persons": persons}


@router.get("/api/persons/stats")
async def api_persons_stats() -> dict[str, Any]:
    """
    Возвращает список персон со статистикой: количество кластеров и лиц.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем всех персон со статистикой по 3 способам привязки
    cur.execute(
        """
        SELECT 
            p.id,
            p.name,
            p.avatar_face_id,
            p."group",
            p.group_order,
            COUNT(DISTINCT CASE WHEN COALESCE(fr.ignore_flag, 0) = 0 THEN fcm.cluster_id ELSE NULL END) as clusters_count,
            COUNT(DISTINCT CASE WHEN COALESCE(fr.ignore_flag, 0) = 0 THEN fl.face_rectangle_id ELSE NULL END) as faces_count,
            -- Статистика через прямоугольники без лица (person_rectangles)
            (SELECT COUNT(DISTINCT pr.file_path) 
             FROM person_rectangles pr 
             WHERE pr.person_id = p.id) as person_rectangles_files_count,
            -- Статистика прямой привязки (file_persons)
            (SELECT COUNT(DISTINCT fp.file_path) 
             FROM file_persons fp 
             WHERE fp.person_id = p.id) as file_persons_files_count
        FROM persons p
        LEFT JOIN face_labels fl ON fl.person_id = p.id
        LEFT JOIN face_cluster_members fcm ON fl.face_rectangle_id = fcm.face_rectangle_id
        LEFT JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        GROUP BY p.id, p.name, p.avatar_face_id, p."group", p.group_order
        ORDER BY 
            CASE WHEN p.name = ? THEN 1 ELSE 0 END,
            COALESCE(p.group_order, 999) ASC,
            p.name ASC
        """,
        (IGNORED_PERSON_NAME,),
    )
    
    persons = []
    for row in cur.fetchall():
        # Получаем preview для аватара, если есть
        avatar_face_id = row["avatar_face_id"]
        avatar_preview_url = None
        if avatar_face_id:
            cur.execute(
                """
                SELECT fr.file_path, fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
                FROM face_rectangles fr
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
            "id": row["id"],
            "name": row["name"],
            "avatar_face_id": avatar_face_id,
            "avatar_preview_url": avatar_preview_url,
            "clusters_count": row["clusters_count"] or 0,
            "faces_count": row["faces_count"] or 0,
            "person_rectangles_files_count": row["person_rectangles_files_count"] or 0,
            "file_persons_files_count": row["file_persons_files_count"] or 0,
            "group": row["group"],
            "group_order": row["group_order"],
            "is_ignored": row["name"] == IGNORED_PERSON_NAME,
        })
    
    return {"persons": persons}


@router.get("/api/persons/{person_id}")
async def api_person_detail(*, person_id: int) -> dict[str, Any]:
    """Получает детальную информацию о персоне и все её лица."""
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
    
    # Получаем все лица персоны через face_labels
    # Используем DISTINCT, чтобы исключить дубликаты, если в face_labels есть несколько записей для одного face_rectangle_id
    cur.execute(
        """
        SELECT DISTINCT
            fr.id as face_id,
            fr.run_id,
            fr.file_path,
            fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
            fr.confidence, fr.presence_score,
            fr.thumb_jpeg,
            fr.embedding IS NOT NULL as has_embedding,
            fcm.cluster_id,
            fc.run_id as cluster_run_id
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        LEFT JOIN face_cluster_members fcm ON fr.id = fcm.face_rectangle_id
        LEFT JOIN face_clusters fc ON fcm.cluster_id = fc.id
        WHERE fl.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        ORDER BY fr.file_path, fr.face_index
        """,
        (person_id,),
    )
    
    faces = []
    for row in cur.fetchall():
        thumb_base64 = None
        if row["thumb_jpeg"]:
            import base64
            thumb_base64 = base64.b64encode(row["thumb_jpeg"]).decode("utf-8")
        
        faces.append({
            "face_id": row["face_id"],
            "run_id": row["run_id"],
            "file_path": row["file_path"],
            "face_index": row["face_index"],
            "bbox": {
                "x": row["bbox_x"],
                "y": row["bbox_y"],
                "w": row["bbox_w"],
                "h": row["bbox_h"],
            },
            "confidence": row["confidence"],
            "presence_score": row["presence_score"],
            "thumb_jpeg_base64": thumb_base64,
            "has_embedding": bool(row["has_embedding"]),
            "cluster_id": row["cluster_id"],
            "cluster_run_id": row["cluster_run_id"],
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
    
    # Проверяем, что лицо принадлежит персоне
    cur.execute(
        """
        SELECT fl.id
        FROM face_labels fl
        WHERE fl.person_id = ? AND fl.face_rectangle_id = ?
        """,
        (person_id, face_id),
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
    
    # Проверяем, что лицо принадлежит персоне
    cur.execute(
        """
        SELECT fl.id
        FROM face_labels fl
        WHERE fl.person_id = ? AND fl.face_rectangle_id = ?
        """,
        (person_id, face_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=400, detail="Face does not belong to this person")
    
    # Устанавливаем ignore_flag
    cur.execute(
        """
        UPDATE face_rectangles
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
        FROM face_rectangles
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
            fr.file_path,
            fr.face_index,
            fr.embedding,
            fl.person_id,
            p.name as person_name
        FROM face_rectangles fr
        LEFT JOIN face_labels fl ON fr.id = fl.face_rectangle_id
        LEFT JOIN persons p ON fl.person_id = p.id
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
    
    # Проверяем, что лицо принадлежит текущей персоне
    cur.execute(
        """
        SELECT fl.id
        FROM face_labels fl
        WHERE fl.person_id = ? AND fl.face_rectangle_id = ?
        """,
        (person_id, face_id),
    )
    label_row = cur.fetchone()
    if not label_row:
        raise HTTPException(status_code=400, detail="Face does not belong to this person")
    
    # Получаем cluster_id через face_cluster_members (если нужно для логирования)
    # Но НЕ сохраняем его в face_labels
    cur.execute(
        """
        SELECT cluster_id
        FROM face_cluster_members
        WHERE face_rectangle_id = ?
        LIMIT 1
        """,
        (face_id,),
    )
    cluster_row = cur.fetchone()
    cluster_id = cluster_row["cluster_id"] if cluster_row else None
    
    # Удаляем из текущей персоны
    cur.execute(
        """
        DELETE FROM face_labels
        WHERE person_id = ? AND face_rectangle_id = ?
        """,
        (person_id, face_id),
    )
    
    # Если указана целевая персона - добавляем к ней
    if target_person_id is not None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        
        # ВАЖНО: cluster_id больше не храним в face_labels
        # Используем INSERT OR REPLACE для предотвращения дубликатов (UNIQUE индекс на face_rectangle_id, person_id)
        cur.execute(
            """
            INSERT OR REPLACE INTO face_labels (face_rectangle_id, person_id, source, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (face_id, target_person_id, "manual", 1.0, now),
        )
    
    conn.commit()
    
    return {"status": "ok", "face_id": face_id, "target_person_id": target_person_id}


@router.post("/api/persons/{person_id}/faces/{face_id}/clear")
async def api_person_face_clear(*, person_id: int, face_id: int) -> dict[str, Any]:
    """Удаляет лицо из персоны и ищет ближайший кластер."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что лицо принадлежит персоне
    cur.execute(
        """
        SELECT fl.id
        FROM face_labels fl
        WHERE fl.person_id = ? AND fl.face_rectangle_id = ?
        """,
        (person_id, face_id),
    )
    label_row = cur.fetchone()
    if not label_row:
        raise HTTPException(status_code=400, detail="Face does not belong to this person")
    
    old_cluster_id = label_row["cluster_id"]
    
    # Удаляем из текущей персоны
    cur.execute(
        """
        DELETE FROM face_labels
        WHERE person_id = ? AND face_rectangle_id = ?
        """,
        (person_id, face_id),
    )
    
    # Ищем ближайший кластер (исключая текущий)
    target_cluster_id = find_closest_cluster_for_face(
        face_rectangle_id=face_id,
        exclude_cluster_id=old_cluster_id,
        max_distance=0.3,
    )
    
    # Если найден кластер, добавляем лицо в него (если ещё не там)
    if target_cluster_id:
        cur.execute(
            """
            SELECT 1 FROM face_cluster_members
            WHERE cluster_id = ? AND face_rectangle_id = ?
            """,
            (target_cluster_id, face_id),
        )
        if not cur.fetchone():
            cur.execute(
                """
                INSERT INTO face_cluster_members (cluster_id, face_rectangle_id)
                VALUES (?, ?)
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
    
    # Получаем все лица персоны через face_labels
    cur.execute(
        """
        SELECT 
            fr.id, fr.run_id, fr.file_path, fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
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
    face_rectangle_id = payload.get("face_rectangle_id")
    if not face_rectangle_id:
        raise HTTPException(status_code=400, detail="face_rectangle_id is required")
    
    try:
        remove_face_from_cluster(cluster_id=cluster_id, face_rectangle_id=int(face_rectangle_id))
        return {"status": "ok", "cluster_id": cluster_id, "face_rectangle_id": int(face_rectangle_id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to remove face: {e}")


@router.post("/api/face-clusters/{cluster_id}/add-face")
async def api_add_face_to_cluster(*, cluster_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Добавляет лицо обратно в кластер (для undo)."""
    face_rectangle_id = payload.get("face_rectangle_id")
    if not face_rectangle_id:
        raise HTTPException(status_code=400, detail="face_rectangle_id is required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что лицо не уже в кластере
    cur.execute(
        """
        SELECT 1 FROM face_cluster_members
        WHERE cluster_id = ? AND face_rectangle_id = ?
        """,
        (cluster_id, int(face_rectangle_id)),
    )
    
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="Face already in cluster")
    
    # Добавляем обратно
    cur.execute(
        """
        INSERT INTO face_cluster_members (cluster_id, face_rectangle_id)
        VALUES (?, ?)
        """,
        (cluster_id, int(face_rectangle_id)),
    )
    
    conn.commit()
    
    return {"status": "ok", "cluster_id": cluster_id, "face_rectangle_id": int(face_rectangle_id)}


@router.post("/api/face-rectangles/{face_rectangle_id}/ignore")
async def api_ignore_face_rectangle(*, face_rectangle_id: int) -> dict[str, Any]:
    """Помечает лицо как 'это не лицо' (устанавливает ignore_flag=1)."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        UPDATE face_rectangles
        SET ignore_flag = 1
        WHERE id = ?
        """,
        (face_rectangle_id,),
    )
    
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    conn.commit()
    
    return {"status": "ok", "face_rectangle_id": face_rectangle_id}


@router.post("/api/face-rectangles/{face_rectangle_id}/unignore")
async def api_unignore_face_rectangle(*, face_rectangle_id: int) -> dict[str, Any]:
    """Убирает пометку 'не лицо' (устанавливает ignore_flag=0, для undo)."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        UPDATE face_rectangles
        SET ignore_flag = 0
        WHERE id = ?
        """,
        (face_rectangle_id,),
    )
    
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    conn.commit()
    
    return {"status": "ok", "face_rectangle_id": face_rectangle_id}


@router.post("/api/face-clusters/{cluster_id}/move-face-to-cluster")
async def api_move_face_to_cluster(*, cluster_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Переносит лицо в другой кластер.
    Если target_cluster_id не указан, автоматически находит ближайший кластер по embedding.
    Если ближайший кластер не найден, создаётся новый.
    """
    face_rectangle_id = payload.get("face_rectangle_id")
    if not face_rectangle_id:
        raise HTTPException(status_code=400, detail="face_rectangle_id is required")
    face_rectangle_id = int(face_rectangle_id)
    
    target_cluster_id = payload.get("target_cluster_id")
    if target_cluster_id is not None:
        target_cluster_id = int(target_cluster_id)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Удаляем из текущего кластера
    cur.execute(
        """
        DELETE FROM face_cluster_members
        WHERE cluster_id = ? AND face_rectangle_id = ?
        """,
        (cluster_id, face_rectangle_id),
    )
    
    # Если target_cluster_id не указан, ищем ближайший кластер
    if target_cluster_id is None:
        target_cluster_id = find_closest_cluster_for_face(
            face_rectangle_id=face_rectangle_id,
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
    
    # Добавляем в целевой кластер
    cur.execute(
        """
        INSERT INTO face_cluster_members (cluster_id, face_rectangle_id)
        VALUES (?, ?)
        """,
        (target_cluster_id, face_rectangle_id),
    )
    
    conn.commit()
    
    return {"status": "ok", "face_rectangle_id": face_rectangle_id, "target_cluster_id": target_cluster_id}


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
            fr.id, fr.run_id, fr.file_path, fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM face_cluster_members fcm
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fcm.cluster_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
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
    
    # Получаем все персоны (кроме "Посторонние", если нужно)
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
        
        # Получаем все лица персоны через face_labels
        cur.execute(
            """
            SELECT 
                fr.id, fr.run_id, fr.file_path, fr.face_index,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
            FROM face_labels fl
            JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
            WHERE fl.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
            """,
            (person_id,),
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
    
    # Получаем все лица персоны через face_labels
    cur.execute(
        """
        SELECT 
            fr.id, fr.run_id, fr.file_path, fr.face_index,
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
        FROM face_labels fl
        JOIN face_rectangles fr ON fl.face_rectangle_id = fr.id
        WHERE fl.person_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (person_id,),
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
async def api_file_faces(file_path: str) -> dict[str, Any]:
    """Получает все лица на указанном файле с информацией о назначенных персонах."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем все лица на этом файле
    # Фильтруем дубликаты: для одинаковых позиций (face_index + bbox) выбираем:
    # 1. Кластеризованное лицо (если есть)
    # 2. Иначе - лицо с максимальным run_id (последний прогон)
    cur.execute(
        """
        WITH ranked_faces AS (
            SELECT 
                fr.id as face_id,
                fr.face_index,
                fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                fr.run_id,
                fr.archive_scope,
                fl.person_id,
                p.name as person_name,
                p.is_me,
                fcm.cluster_id,
                ROW_NUMBER() OVER (
                    PARTITION BY fr.file_path, fr.face_index, fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
                    ORDER BY 
                        CASE WHEN fcm.cluster_id IS NOT NULL THEN 0 ELSE 1 END,  -- Сначала кластеризованные
                        CASE WHEN fr.archive_scope = 'archive' THEN 0 ELSE 1 END,  -- Приоритет архивным лицам
                        CASE WHEN fr.run_id IS NULL THEN 1 ELSE 0 END,  -- NULL в конце
                        fr.run_id DESC  -- Потом по run_id (последний прогон)
                ) as rn
            FROM face_rectangles fr
            LEFT JOIN face_labels fl ON fr.id = fl.face_rectangle_id
            LEFT JOIN persons p ON fl.person_id = p.id
            LEFT JOIN face_cluster_members fcm ON fr.id = fcm.face_rectangle_id
            WHERE fr.file_path = ? AND COALESCE(fr.ignore_flag, 0) = 0
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
        (file_path,),
    )
    
    faces = []
    for row in cur.fetchall():
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
    
    return {"faces": faces}


@router.post("/api/face-rectangles/{face_rectangle_id}/find-cluster")
async def api_find_cluster_for_face(*, face_rectangle_id: int) -> dict[str, Any]:
    """Находит ближайший кластер для указанного лица."""
    from backend.logic.face_recognition import find_closest_cluster_for_face
    
    target_cluster_id = find_closest_cluster_for_face(
        face_rectangle_id=face_rectangle_id,
        exclude_cluster_id=None,
        max_distance=0.3,
    )
    
    if target_cluster_id is None:
        return {"cluster_id": None, "message": "Ближайший кластер не найден"}
    
    return {"cluster_id": target_cluster_id}


@router.post("/api/face-rectangles/{face_rectangle_id}/assign-person")
async def api_assign_person_to_face(*, face_rectangle_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
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
        SELECT id FROM face_rectangles WHERE id = ?
        """,
        (face_rectangle_id,),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Face rectangle not found")
    
    # Получаем cluster_id для этого лица (если есть)
    cur.execute(
        """
        SELECT cluster_id FROM face_cluster_members WHERE face_rectangle_id = ? LIMIT 1
        """,
        (face_rectangle_id,),
    )
    cluster_row = cur.fetchone()
    cluster_id = cluster_row["cluster_id"] if cluster_row else None
    
    # Удаляем старые назначения для этого лица
    cur.execute(
        """
        DELETE FROM face_labels WHERE face_rectangle_id = ?
        """,
        (face_rectangle_id,),
    )
    
    # Создаем новое назначение
    # ВАЖНО: cluster_id больше не храним в face_labels, кластер определяется через face_cluster_members
    # Используем INSERT OR REPLACE для предотвращения дубликатов (UNIQUE индекс на face_rectangle_id, person_id)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    
    cur.execute(
        """
        INSERT OR REPLACE INTO face_labels (face_rectangle_id, person_id, source, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (face_rectangle_id, person_id, "manual", 1.0, now),
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
        "face_rectangle_id": face_rectangle_id,
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
    Назначает персону через прямоугольник без лица (person_rectangles).
    
    Параметры:
    - pipeline_run_id: int (обязательно)
    - file_path: str (обязательно)
    - frame_idx: int | None (для видео: 1..3, для фото: None)
    - bbox_x, bbox_y, bbox_w, bbox_h: int (координаты прямоугольника)
    - person_id: int (обязательно)
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    file_path = payload.get("file_path")
    frame_idx = payload.get("frame_idx")
    bbox_x = payload.get("bbox_x")
    bbox_y = payload.get("bbox_y")
    bbox_w = payload.get("bbox_w")
    bbox_h = payload.get("bbox_h")
    person_id = payload.get("person_id")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required and must be int")
    if not isinstance(file_path, str):
        raise HTTPException(status_code=400, detail="file_path is required and must be str")
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
    
    # Валидация координат
    try:
        bbox_x = int(bbox_x) if bbox_x is not None else 0
        bbox_y = int(bbox_y) if bbox_y is not None else 0
        bbox_w = int(bbox_w) if bbox_w is not None else 0
        bbox_h = int(bbox_h) if bbox_h is not None else 0
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="bbox coordinates must be integers")
    
    if bbox_w <= 0 or bbox_h <= 0:
        raise HTTPException(status_code=400, detail="bbox width and height must be positive")
    
    # Валидация frame_idx для видео
    if frame_idx is not None:
        try:
            frame_idx = int(frame_idx)
            if frame_idx not in (1, 2, 3):
                raise HTTPException(status_code=400, detail="frame_idx must be 1, 2, or 3 for video")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="frame_idx must be integer")
    
    # Вставляем прямоугольник
    fs = FaceStore()
    try:
        rectangle_id = fs.insert_person_rectangle(
            pipeline_run_id=int(pipeline_run_id),
            file_path=str(file_path),
            frame_idx=frame_idx,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            person_id=int(person_id),
        )
    finally:
        fs.close()
    
    return {
        "ok": True,
        "rectangle_id": rectangle_id,
        "pipeline_run_id": int(pipeline_run_id),
        "file_path": str(file_path),
        "person_id": int(person_id),
        "person_name": person_row["name"],
    }


@router.post("/api/persons/assign-file")
async def api_persons_assign_file(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Назначает персону файлу напрямую (без прямоугольника) - file_persons.
    
    Параметры:
    - pipeline_run_id: int (обязательно)
    - file_path: str (обязательно)
    - person_id: int (обязательно)
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    file_path = payload.get("file_path")
    person_id = payload.get("person_id")
    
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required and must be int")
    if not isinstance(file_path, str):
        raise HTTPException(status_code=400, detail="file_path is required and must be str")
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
    fs = FaceStore()
    try:
        fs.insert_file_person(
            pipeline_run_id=int(pipeline_run_id),
            file_path=str(file_path),
            person_id=int(person_id),
        )
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "file_path": str(file_path),
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
    - file_path: str (обязательно для person_rectangle и file)
    - person_id: int (обязательно)
    - face_rectangle_id: int (обязательно для type="face")
    - rectangle_id: int (обязательно для type="person_rectangle")
    """
    assignment_type = payload.get("assignment_type")
    pipeline_run_id = payload.get("pipeline_run_id")
    file_path = payload.get("file_path")
    person_id = payload.get("person_id")
    face_rectangle_id = payload.get("face_rectangle_id")
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
            # Удаляем привязку через лицо (face_labels)
            if not isinstance(face_rectangle_id, int):
                raise HTTPException(status_code=400, detail="face_rectangle_id is required for type='face'")
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM face_labels
                WHERE face_rectangle_id = ? AND person_id = ?
                """,
                (int(face_rectangle_id), int(person_id)),
            )
            conn.commit()
            
        elif assignment_type == "person_rectangle":
            # Удаляем привязку через прямоугольник без лица
            if not isinstance(rectangle_id, int):
                raise HTTPException(status_code=400, detail="rectangle_id is required for type='person_rectangle'")
            fs.delete_person_rectangle(rectangle_id=int(rectangle_id))
            
        elif assignment_type == "file":
            # Удаляем прямую привязку файла
            if not isinstance(pipeline_run_id, int):
                raise HTTPException(status_code=400, detail="pipeline_run_id is required for type='file'")
            if not isinstance(file_path, str):
                raise HTTPException(status_code=400, detail="file_path is required for type='file'")
            fs.delete_file_person(
                pipeline_run_id=int(pipeline_run_id),
                file_path=str(file_path),
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
    file_path: str,
) -> dict[str, Any]:
    """
    Возвращает все привязки файла к персонам (через все 3 способа):
    - через лица (face_labels через face_rectangles)
    - через прямоугольники без лица (person_rectangles)
    - прямая привязка (file_persons)
    """
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id must be int")
    if not isinstance(file_path, str):
        raise HTTPException(status_code=400, detail="file_path must be str")
    
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
            file_path=str(file_path),
        )
    finally:
        fs.close()
    
    return {
        "ok": True,
        "pipeline_run_id": int(pipeline_run_id),
        "file_path": str(file_path),
        "assignments": assignments,
    }
