"""
Логика распознавания лиц: извлечение embeddings, кластеризация, использование контекста папок.
"""

import json
from typing import Any
from pathlib import Path

# Опциональные импорты для ML (могут быть недоступны в окружении веб-сервера)
try:
    import numpy as np
    from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]
    from sklearn.metrics.pairwise import cosine_distances  # type: ignore[import-untyped]
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Заглушки для type checking
    np = None  # type: ignore[assignment, misc]
    DBSCAN = None  # type: ignore[assignment, misc]
    cosine_distances = None  # type: ignore[assignment, misc]

from backend.common.db import get_connection


def extract_folder_name_from_path(file_path: str) -> str | None:
    """
    Извлекает имя папки из пути файла для использования как контекст.
    
    Примеры:
        'disk:/Фото/Агата/photo.jpg' -> 'Агата'
        'disk:/Фото/Дети/photo.jpg' -> 'Дети'
        'local:/tmp/photo.jpg' -> None (если нет структуры папок)
    """
    try:
        # Убираем префикс 'disk:/' или 'local:/'
        path = file_path.replace("disk:/", "").replace("local:/", "")
        parts = path.split("/")
        
        # Ищем папку после 'Фото' (обычно это имя персоны)
        if len(parts) >= 2:
            # Если путь начинается с 'Фото', берём следующую папку
            if parts[0] == "Фото" and len(parts) >= 2:
                return parts[1]
            # Иначе берём предпоследнюю папку (перед именем файла)
            if len(parts) >= 2:
                return parts[-2]
    except Exception:
        pass
    return None


def get_folder_context_weights(*, conn: Any, file_paths: list[str]) -> dict[str, float]:
    """
    Получает веса контекста папок для файлов.
    
    Если файл находится в папке с именем персоны (например, 'disk:/Фото/Агата/...'),
    то лица из этого файла с высокой вероятностью принадлежат этой персоне.
    
    Returns:
        dict: {file_path: weight} где weight от 0.0 до 1.0 (высокая вероятность)
    """
    weights: dict[str, float] = {}
    
    try:
        # Получаем список папок из БД
        cur = conn.cursor()
        cur.execute("SELECT code, path, name FROM folders WHERE name IS NOT NULL AND TRIM(name) != ''")
        folders = {row["name"]: row for row in cur.fetchall()}
        
        # Для каждого файла проверяем, находится ли он в папке с именем персоны
        for file_path in file_paths:
            folder_name = extract_folder_name_from_path(file_path)
            if folder_name and folder_name in folders:
                # Высокая вероятность (0.8) что лица в этой папке принадлежат персоне с таким именем
                weights[file_path] = 0.8
            else:
                weights[file_path] = 0.0
    except Exception:
        pass
    
    return weights


def cluster_face_embeddings(
    *,
    run_id: int | None = None,
    archive_scope: str | None = None,
    eps: float = 0.4,
    min_samples: int = 2,
    use_folder_context: bool = True,
) -> dict[str, Any]:
    """
    Кластеризует face embeddings из БД используя DBSCAN.
    
    Args:
        run_id: ID прогона детекции лиц (для сортируемых папок, опционально)
        archive_scope: 'archive' для архивного режима (для фотоархива, опционально)
        eps: максимальное расстояние между точками в одном кластере (для косинусного расстояния)
        min_samples: минимальное количество точек для формирования кластера
        use_folder_context: использовать ли контекст папок для улучшения кластеризации
    
    Returns:
        dict с результатами: {
            'clusters': {cluster_id: [face_rectangle_id, ...]},
            'noise': [face_rectangle_id, ...],  # точки, не попавшие в кластеры
            'cluster_id': int,  # ID созданного кластера в БД
        }
    """
    if not ML_AVAILABLE:
        raise RuntimeError(
            "ML dependencies (numpy, scikit-learn) are not available. "
            "Please use the face recognition environment (.venv-face) to run clustering."
        )
    
    # Проверяем валидность параметров
    if run_id is None and archive_scope != 'archive':
        raise ValueError("Необходимо указать либо run_id (для прогонов), либо archive_scope='archive' (для архива)")
    
    if run_id is not None and archive_scope == 'archive':
        raise ValueError("Нельзя одновременно использовать run_id и archive_scope='archive'")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем SQL-запрос в зависимости от режима
    if archive_scope == 'archive':
        # Для архива фильтруем по archive_scope
        cur.execute(
            """
            SELECT 
                id, file_path, embedding
            FROM face_rectangles
            WHERE archive_scope = 'archive'
              AND embedding IS NOT NULL 
              AND COALESCE(ignore_flag, 0) = 0
            ORDER BY id
            """
        )
    else:
        # Для прогонов фильтруем по run_id
        if run_id is None:
            raise ValueError("run_id обязателен для неархивных записей")
        cur.execute(
            """
            SELECT 
                id, file_path, embedding
            FROM face_rectangles
            WHERE run_id = ? 
              AND embedding IS NOT NULL 
              AND COALESCE(ignore_flag, 0) = 0
            ORDER BY id
            """,
            (run_id,),
        )
    
    rows = cur.fetchall()
    
    if len(rows) == 0:
        return {
            "clusters": {},
            "noise": [],
            "cluster_id": None,
            "total_faces": 0,
        }
    
    # Для архивного режима исключаем лица, которые уже находятся в кластерах
    # (append без дублирования - не кластеризуем лица, которые уже были обработаны)
    if archive_scope == 'archive':
        # Получаем список ID лиц, которые уже находятся в каких-то кластерах
        cur.execute("""
            SELECT DISTINCT fcm.face_rectangle_id
            FROM face_cluster_members fcm
            INNER JOIN face_clusters fc ON fcm.cluster_id = fc.id
            WHERE fc.archive_scope = 'archive'
        """)
        clustered_face_ids = {row['face_rectangle_id'] for row in cur.fetchall()}
        
        # Фильтруем rows - оставляем только лица, которых ещё нет в кластерах
        rows = [row for row in rows if row['id'] not in clustered_face_ids]
        
        if len(rows) == 0:
            # Все лица уже в кластерах - нечего кластеризовать
            return {
                "clusters": {},
                "noise": [],
                "cluster_id": None,
                "total_faces": 0,
                "skipped_already_clustered": len(clustered_face_ids),
            }
    
    # Подготавливаем данные для кластеризации
    face_ids: list[int] = []
    file_paths: list[str] = []
    embeddings: list[np.ndarray] = []
    
    for row in rows:
        face_id = row["id"]
        file_path = row["file_path"]
        embedding_json = row["embedding"]
        
        if embedding_json is None:
            continue
        
        try:
            # Десериализуем embedding
            emb_list = json.loads(embedding_json.decode("utf-8"))
            emb_array = np.array(emb_list, dtype=np.float32)
            
            # Проверяем, что embedding валидный
            if emb_array.size == 0 or np.isnan(emb_array).any():
                continue
            
            face_ids.append(face_id)
            file_paths.append(file_path)
            embeddings.append(emb_array)
        except Exception:
            continue
    
    if len(embeddings) == 0:
        return {
            "clusters": {},
            "noise": [],
            "cluster_id": None,
            "total_faces": 0,
        }
    
    # Преобразуем в numpy array
    X = np.array(embeddings)
    
    # Нормализуем embeddings (L2 норма) для корректной работы косинусного расстояния
    # Это критично: без нормализации косинусное расстояние работает некорректно
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Избегаем деления на ноль
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    
    # Кластеризация DBSCAN с косинусным расстоянием
    # DBSCAN использует метрику 'cosine' для косинусного расстояния
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(X_normalized)
    
    # Группируем результаты
    clusters: dict[int, list[int]] = {}
    noise: list[int] = []
    
    for idx, label in enumerate(labels):
        face_id = face_ids[idx]
        if label == -1:
            # Шум (не попал в кластер)
            noise.append(face_id)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(face_id)
    
    # Используем контекст папок для улучшения кластеризации
    if use_folder_context:
        folder_weights = get_folder_context_weights(conn=conn, file_paths=file_paths)
        # Можно использовать веса для перераспределения или приоритизации кластеров
        # Пока просто сохраняем информацию о весах в метаданных
    
    # Сохраняем кластеры в БД (только если есть хотя бы один кластер с лицами)
    cluster_id = None
    if len(clusters) > 0:
        cluster_id = _save_clusters_to_db(
            conn=conn,
            run_id=run_id,
            archive_scope=archive_scope,
            clusters=clusters,
            noise=noise,
            method="DBSCAN",
            params={"eps": eps, "min_samples": min_samples},
        )
    
    return {
        "clusters": clusters,
        "noise": noise,
        "cluster_id": cluster_id,
        "total_faces": len(face_ids),
        "clusters_count": len(clusters),
        "noise_count": len(noise),
    }


def _save_clusters_to_db(
    *,
    conn: Any,
    run_id: int | None,
    archive_scope: str | None,
    clusters: dict[int, list[int]],
    noise: list[int],
    method: str,
    params: dict[str, Any],
) -> int | None:
    """
    Сохраняет результаты кластеризации в БД.
    Каждый кластер из DBSCAN сохраняется как отдельная запись в face_clusters.
    
    Returns:
        ID первого созданного кластера в БД (или None, если кластеров нет)
    """
    from datetime import datetime, timezone
    
    cur = conn.cursor()
    
    if len(clusters) == 0:
        return None
    
    now = datetime.now(timezone.utc).isoformat()
    params_json = json.dumps(params)
    
    first_cluster_id = None
    
    # Создаём отдельную запись для каждого кластера
    for cluster_label, face_ids in clusters.items():
        if len(face_ids) == 0:
            continue  # Пропускаем пустые кластеры
        
        # Создаём запись о кластере (для архива run_id может быть NULL)
        if archive_scope == 'archive':
            cur.execute(
                """
                INSERT INTO face_clusters (run_id, archive_scope, method, params_json, created_at)
                VALUES (NULL, ?, ?, ?, ?)
                """,
                (archive_scope, method, params_json, now),
            )
        else:
            if run_id is None:
                raise ValueError("run_id обязателен для неархивных записей")
            cur.execute(
                """
                INSERT INTO face_clusters (run_id, archive_scope, method, params_json, created_at)
                VALUES (?, NULL, ?, ?, ?)
                """,
                (run_id, method, params_json, now),
            )
        
        cluster_id = cur.lastrowid
        if first_cluster_id is None:
            first_cluster_id = cluster_id
        
        # Сохраняем связи лиц с этим кластером
        for face_id in face_ids:
            cur.execute(
                """
                INSERT INTO face_cluster_members (cluster_id, face_rectangle_id)
                VALUES (?, ?)
                """,
                (cluster_id, face_id),
            )
    
    # Шум (noise) не сохраняем в кластеры, они остаются без кластера
    
    conn.commit()
    
    return first_cluster_id


def assign_cluster_to_person(*, cluster_id: int, person_id: int | None) -> None:
    """
    Назначает кластер персоне (создаёт face_labels для всех лиц в кластере).
    При назначении новой персоны удаляет все старые назначения для этого кластера.
    Если person_id=None, только удаляет назначения (снимает назначение).
    
    Args:
        cluster_id: ID кластера
        person_id: ID персоны из справочника, или None для снятия назначения
    """
    from datetime import datetime, timezone
    
    conn = get_connection()
    cur = conn.cursor()
    
    # УДАЛЯЕМ все старые назначения для этого кластера
    cur.execute(
        """
        DELETE FROM face_labels
        WHERE cluster_id = ?
        """,
        (cluster_id,),
    )
    
    # Если person_id=None, только удаляем назначения и выходим
    if person_id is None:
        conn.commit()
        return
    
    # Получаем все лица из кластера
    cur.execute(
        """
        SELECT face_rectangle_id
        FROM face_cluster_members
        WHERE cluster_id = ?
        """,
        (cluster_id,),
    )
    
    face_ids = [row["face_rectangle_id"] for row in cur.fetchall()]
    
    if len(face_ids) == 0:
        conn.commit()
        return
    
    # Создаём новые face_labels для всех лиц в кластере с новой персоной
    now = datetime.now(timezone.utc).isoformat()
    
    for face_id in face_ids:
        # Создаём новую метку (старые уже удалены)
        cur.execute(
            """
            INSERT INTO face_labels (face_rectangle_id, person_id, cluster_id, source, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (face_id, person_id, cluster_id, "cluster", 0.8, now),
        )
    
    conn.commit()


def get_cluster_info(*, cluster_id: int, limit: int | None = None) -> dict[str, Any]:
    """
    Получает информацию о кластере: количество лиц, примеры, связанные персоны.
    
    Args:
        cluster_id: ID кластера
        limit: максимальное количество лиц для возврата (None = все лица)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем информацию о кластере
    # Исключаем помеченные как "не лицо" и сортируем по размеру bbox (самое крупное первым)
    query = """
        SELECT 
            fc.id, fc.run_id, fc.method, fc.params_json, fc.created_at,
            fr.id as face_rectangle_id, fr.file_path, fr.face_index, 
            fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
            f.image_width, f.image_height, f.exif_orientation,
            fr.thumb_jpeg, fr.confidence,
            (fr.bbox_w * fr.bbox_h) as bbox_area
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        LEFT JOIN files f ON fr.file_path = f.path
        WHERE fc.id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        ORDER BY bbox_area DESC, fr.confidence DESC
    """
    params = [cluster_id]
    
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    
    cur.execute(query, tuple(params))
    
    faces = []
    for row in cur.fetchall():
        # Конвертируем thumb_jpeg в base64 сразу, чтобы избежать проблем с сериализацией bytes
        thumb_base64 = None
        if row["thumb_jpeg"]:
            import base64
            thumb_base64 = base64.b64encode(row["thumb_jpeg"]).decode("utf-8")
        
        faces.append({
            "face_id": row["face_rectangle_id"],
            "file_path": row["file_path"],
            "face_index": row["face_index"],
            "bbox": {
                "x": row["bbox_x"],
                "y": row["bbox_y"],
                "w": row["bbox_w"],
                "h": row["bbox_h"],
            },
            "image_size": {
                "width": row["image_width"],
                "height": row["image_height"],
                "exif_orientation": row["exif_orientation"],
            } if row["image_width"] is not None and row["image_height"] is not None else None,
            "thumb_jpeg_base64": thumb_base64,  # Возвращаем base64, а не bytes
            "confidence": row["confidence"],
        })
    
    # Получаем количество лиц в кластере (исключая помеченные как "не лицо")
    cur.execute(
        """
        SELECT COUNT(*) as count
        FROM face_cluster_members fcm
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fcm.cluster_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
        """,
        (cluster_id,),
    )
    
    count_row = cur.fetchone()
    total_faces = count_row["count"] if count_row else 0
    
    # Получаем связанные персоны (если есть)
    cur.execute(
        """
        SELECT DISTINCT fl.person_id, p.name
        FROM face_labels fl
        JOIN persons p ON fl.person_id = p.id
        WHERE fl.cluster_id = ?
        """,
        (cluster_id,),
    )
    
    persons = [{"id": row["person_id"], "name": row["name"]} for row in cur.fetchall()]
    
    # Получаем информацию о кластере (run_id, method, created_at)
    cur.execute(
        """
        SELECT run_id, method, params_json, created_at
        FROM face_clusters
        WHERE id = ?
        """,
        (cluster_id,),
    )
    cluster_row = cur.fetchone()
    
    return {
        "cluster_id": cluster_id,
        "run_id": cluster_row["run_id"] if cluster_row else None,
        "method": cluster_row["method"] if cluster_row else None,
        "created_at": cluster_row["created_at"] if cluster_row else None,
        "total_faces": total_faces,
        "faces": faces,  # Все лица (или ограниченное количество, если указан limit)
        "persons": persons,
    }


def find_closest_cluster_for_face(*, face_rectangle_id: int, exclude_cluster_id: int | None = None, max_distance: float = 0.3) -> int | None:
    """
    Находит ближайший кластер для лица по embedding.
    
    Args:
        face_rectangle_id: ID лица
        exclude_cluster_id: ID кластера, который нужно исключить из поиска
        max_distance: максимальное косинусное расстояние для попадания в кластер
    
    Returns:
        ID ближайшего кластера или None, если не найден подходящий
    """
    if not ML_AVAILABLE:
        return None
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем embedding лица
    cur.execute(
        """
        SELECT embedding
        FROM face_rectangles
        WHERE id = ? AND embedding IS NOT NULL
        """,
        (face_rectangle_id,),
    )
    
    face_row = cur.fetchone()
    if not face_row or not face_row["embedding"]:
        return None
    
    try:
        emb_json = face_row["embedding"]
        emb_list = json.loads(emb_json.decode("utf-8"))
        face_emb = np.array(emb_list, dtype=np.float32)
        
        # Нормализуем
        norm = np.linalg.norm(face_emb)
        if norm == 0:
            return None
        face_emb_normalized = face_emb / norm
    except Exception:
        return None
    
    # Получаем все кластеры с их средними embeddings
    # Исключаем текущий кластер и кластеры без лиц
    exclude_clause = ""
    exclude_params = []
    if exclude_cluster_id is not None:
        exclude_clause = "AND fc.id != ?"
        exclude_params.append(exclude_cluster_id)
    
    cur.execute(
        f"""
        SELECT 
            fc.id as cluster_id,
            fr.embedding
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE fr.embedding IS NOT NULL
          AND COALESCE(fr.ignore_flag, 0) = 0
          {exclude_clause}
        GROUP BY fc.id
        """,
        tuple(exclude_params),
    )
    
    clusters_embeddings: dict[int, list[np.ndarray]] = {}
    
    for row in cur.fetchall():
        cluster_id = row["cluster_id"]
        emb_json = row["embedding"]
        
        try:
            emb_list = json.loads(emb_json.decode("utf-8"))
            emb_array = np.array(emb_list, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_normalized = emb_array / norm
                if cluster_id not in clusters_embeddings:
                    clusters_embeddings[cluster_id] = []
                clusters_embeddings[cluster_id].append(emb_normalized)
        except Exception:
            continue
    
    if not clusters_embeddings:
        return None
    
    # Вычисляем средний embedding для каждого кластера и находим ближайший
    best_cluster_id = None
    best_distance = float('inf')
    
    for cluster_id, embs in clusters_embeddings.items():
        if not embs:
            continue
        
        # Средний embedding кластера
        cluster_center = np.mean(embs, axis=0)
        cluster_center_norm = np.linalg.norm(cluster_center)
        if cluster_center_norm > 0:
            cluster_center = cluster_center / cluster_center_norm
        
        # Косинусное расстояние
        distance = 1.0 - np.dot(face_emb_normalized, cluster_center)
        
        if distance < best_distance and distance <= max_distance:
            best_distance = distance
            best_cluster_id = cluster_id
    
    return best_cluster_id


def remove_face_from_cluster(*, cluster_id: int, face_rectangle_id: int) -> None:
    """
    Исключает лицо из кластера (удаляет из face_cluster_members).
    
    Args:
        cluster_id: ID кластера
        face_rectangle_id: ID лица для исключения
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что лицо действительно в этом кластере
    cur.execute(
        """
        SELECT 1 FROM face_cluster_members
        WHERE cluster_id = ? AND face_rectangle_id = ?
        """,
        (cluster_id, face_rectangle_id),
    )
    
    if not cur.fetchone():
        raise ValueError(f"Face {face_rectangle_id} is not in cluster {cluster_id}")
    
    # Удаляем связь
    cur.execute(
        """
        DELETE FROM face_cluster_members
        WHERE cluster_id = ? AND face_rectangle_id = ?
        """,
        (cluster_id, face_rectangle_id),
    )
    
    # Если у кластера больше нет лиц, можно удалить и сам кластер (опционально)
    # Пока оставляем кластер, даже если он пустой
    
    conn.commit()
