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
    
    # Получаем список ID лиц, которые уже находятся в каких-то кластерах
    # (для исключения из повторной кластеризации)
    if archive_scope == 'archive':
        cur.execute("""
            SELECT DISTINCT fcm.face_rectangle_id
            FROM face_cluster_members fcm
            INNER JOIN face_clusters fc ON fcm.cluster_id = fc.id
            WHERE fc.archive_scope = 'archive'
        """)
    else:
        # Для run_id проверяем ВСЕ существующие кластеры (не только для этого run_id)
        # чтобы можно было добавлять новые лица в существующие кластеры
        cur.execute("""
            SELECT DISTINCT fcm.face_rectangle_id
            FROM face_cluster_members fcm
            INNER JOIN face_clusters fc ON fcm.cluster_id = fc.id
        """)
    
    clustered_face_ids = {row['face_rectangle_id'] for row in cur.fetchall()}
    
    # Фильтруем rows - оставляем только лица, которых ещё нет в кластерах
    new_face_rows = [row for row in rows if row['id'] not in clustered_face_ids]
    
    if len(new_face_rows) == 0:
        # Все лица уже в кластерах - нечего кластеризовать
        return {
            "clusters": {},
            "noise": [],
            "cluster_id": None,
            "total_faces": 0,
            "skipped_already_clustered": len(clustered_face_ids),
        }
    
    # Для run_id: пытаемся добавить новые лица в существующие кластеры
    # (приоритет поиска существующих кластеров перед созданием новых)
    faces_added_to_existing = 0
    if run_id is not None and archive_scope != 'archive':
        faces_added_to_existing, new_face_rows = _try_add_to_existing_clusters(
            conn=conn,
            new_face_rows=new_face_rows,
            eps=eps,
        )
    
    # Используем отфильтрованный список для кластеризации
    rows = new_face_rows
    
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
    
    # Для run_id удаляем только старые кластеры, созданные для этого run_id
    # (чтобы избежать дубликатов при повторном запуске, но сохранить кластеры из других run_id)
    if run_id is not None and archive_scope != 'archive':
        # Удаляем старые кластеры для этого run_id
        # Сначала удаляем связи (face_cluster_members), затем сами кластеры
        cur.execute("""
            DELETE FROM face_cluster_members
            WHERE cluster_id IN (
                SELECT id FROM face_clusters WHERE run_id = ?
            )
        """, (run_id,))
        
        cur.execute("""
            DELETE FROM face_clusters WHERE run_id = ?
        """, (run_id,))
        
        conn.commit()
    
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
    
    result = {
        "clusters": clusters,
        "noise": noise,
        "cluster_id": cluster_id,
        "total_faces": len(face_ids),
        "clusters_count": len(clusters),
        "noise_count": len(noise),
    }
    
    # Добавляем информацию о лицах, добавленных в существующие кластеры
    if run_id is not None and archive_scope != 'archive' and faces_added_to_existing > 0:
        result["faces_added_to_existing"] = faces_added_to_existing
    
    return result


def _try_add_to_existing_clusters(
    *,
    conn: Any,
    new_face_rows: list[dict[str, Any]],
    eps: float,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Пытается добавить новые лица в существующие кластеры по схожести embeddings.
    
    Args:
        conn: соединение с БД
        new_face_rows: список новых лиц (словари с 'id', 'file_path', 'embedding')
        eps: порог схожести (косинусное расстояние)
    
    Returns:
        tuple: (количество добавленных лиц, список оставшихся лиц для кластеризации)
    """
    if not ML_AVAILABLE or len(new_face_rows) == 0:
        return 0, new_face_rows
    
    cur = conn.cursor()
    
    # Получаем все существующие кластеры с их лицами и embeddings
    cur.execute("""
        SELECT 
            fc.id AS cluster_id,
            fr.id AS face_rectangle_id,
            fr.embedding
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
        JOIN face_rectangles fr ON fr.id = fcm.face_rectangle_id
        WHERE fr.embedding IS NOT NULL
          AND COALESCE(fr.ignore_flag, 0) = 0
        ORDER BY fc.id
    """)
    
    # Группируем по кластерам
    clusters_data: dict[int, list[tuple[int, bytes]]] = {}  # cluster_id -> [(face_id, embedding), ...]
    for row in cur.fetchall():
        cluster_id = row['cluster_id']
        face_id = row['face_rectangle_id']
        embedding = row['embedding']
        if cluster_id not in clusters_data:
            clusters_data[cluster_id] = []
        clusters_data[cluster_id].append((face_id, embedding))
    
    if len(clusters_data) == 0:
        return 0, new_face_rows
    
    # Вычисляем центроиды для каждого кластера (средний embedding)
    cluster_centroids: dict[int, np.ndarray] = {}
    for cluster_id, faces_data in clusters_data.items():
        embeddings_list = []
        for face_id, emb_json in faces_data:
            try:
                emb_array = np.array(json.loads(emb_json.decode("utf-8")), dtype=np.float32)
                if emb_array.size > 0 and not np.isnan(emb_array).any():
                    # Нормализуем
                    norm = np.linalg.norm(emb_array)
                    if norm > 0:
                        embeddings_list.append(emb_array / norm)
            except Exception:
                continue
        
        if len(embeddings_list) > 0:
            # Вычисляем средний embedding (центроид)
            centroid = np.mean(embeddings_list, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                cluster_centroids[cluster_id] = centroid / norm
    
    if len(cluster_centroids) == 0:
        return 0, new_face_rows
    
    # Получаем привязку кластеров к персонам (если есть)
    # Если у кластера несколько персон, берем самую частую
    if len(cluster_centroids) > 0:
        cur.execute("""
            SELECT 
                fc.id AS cluster_id,
                fl.person_id,
                COUNT(*) AS person_count
            FROM face_clusters fc
            JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id
            JOIN face_labels fl ON fl.face_rectangle_id = fcm.face_rectangle_id
            WHERE fc.id IN ({})
            GROUP BY fc.id, fl.person_id
            ORDER BY fc.id, person_count DESC
        """.format(",".join("?" * len(cluster_centroids))), list(cluster_centroids.keys()))
        
        cluster_to_person: dict[int, int] = {}  # cluster_id -> person_id
        for row in cur.fetchall():
            cluster_id = row['cluster_id']
            person_id = row['person_id']
            # Берем первую (самую частую) персону для кластера
            if cluster_id not in cluster_to_person:
                cluster_to_person[cluster_id] = person_id
    else:
        cluster_to_person = {}
    
    # Пытаемся добавить новые лица в существующие кластеры
    faces_to_add: list[tuple[int, int]] = []  # [(face_id, cluster_id), ...]
    remaining_faces: list[dict[str, Any]] = []
    
    for face_row in new_face_rows:
        face_id = face_row['id']
        embedding_json = face_row['embedding']
        
        if not embedding_json:
            remaining_faces.append(face_row)
            continue
        
        try:
            # Десериализуем и нормализуем embedding нового лица
            emb_array = np.array(json.loads(embedding_json.decode("utf-8")), dtype=np.float32)
            if emb_array.size == 0 or np.isnan(emb_array).any():
                remaining_faces.append(face_row)
                continue
            
            norm = np.linalg.norm(emb_array)
            if norm == 0:
                remaining_faces.append(face_row)
                continue
            emb_normalized = emb_array / norm
            
            # Ищем наиболее подходящий кластер
            best_cluster_id = None
            best_similarity = -1.0
            
            for cluster_id, centroid in cluster_centroids.items():
                # Косинусное расстояние (1 - cosine similarity)
                similarity = float(np.dot(emb_normalized, centroid))
                # Для косинусного расстояния: чем больше similarity, тем ближе
                # eps - это максимальное расстояние, поэтому similarity >= (1 - eps)
                if similarity >= (1.0 - eps) and similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_id = cluster_id
            
            if best_cluster_id is not None:
                faces_to_add.append((face_id, best_cluster_id))
            else:
                remaining_faces.append(face_row)
        except Exception:
            remaining_faces.append(face_row)
    
    # Добавляем найденные лица в существующие кластеры
    # НЕ создаем face_labels - если кластер привязан к персоне, 
    # то все лица в кластере автоматически относятся к персоне через JOIN
    # (согласно архитектуре: face_labels только для ручных привязок,
    # кластер определяется через face_cluster_members)
    added_count = 0
    
    for face_id, cluster_id in faces_to_add:
        try:
            # Добавляем лицо в кластер
            cur.execute("""
                INSERT OR IGNORE INTO face_cluster_members (cluster_id, face_rectangle_id)
                VALUES (?, ?)
            """, (cluster_id, face_id))
            
            if cur.rowcount > 0:
                added_count += 1
        except Exception:
            continue
    
    if added_count > 0:
        conn.commit()
    
    return added_count, remaining_faces


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
    total_clusters = len(clusters)
    processed_clusters = 0
    
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
        # Для больших кластеров коммитим периодически
        batch_size = 100  # Коммитим после каждых 100 лиц в кластере
        for i, face_id in enumerate(face_ids):
            cur.execute(
                """
                INSERT INTO face_cluster_members (cluster_id, face_rectangle_id)
                VALUES (?, ?)
                """,
                (cluster_id, face_id),
            )
            # Периодический коммит для больших кластеров
            if (i + 1) % batch_size == 0:
                conn.commit()
        
        # Коммитим после каждого кластера, чтобы не блокировать БД
        conn.commit()
        processed_clusters += 1
        
        # Периодический вывод прогресса для больших наборов кластеров
        if total_clusters > 100 and processed_clusters % 100 == 0:
            print(f"Обработано кластеров: {processed_clusters}/{total_clusters}")
    
    # Шум (noise) не сохраняем в кластеры, они остаются без кластера
    
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
    # Получаем все лица из кластера и удаляем их face_labels
    cur.execute(
        """
        DELETE FROM face_labels
        WHERE face_rectangle_id IN (
            SELECT face_rectangle_id
            FROM face_cluster_members
            WHERE cluster_id = ?
        )
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
        # ВАЖНО: cluster_id больше не храним в face_labels, кластер определяется через face_cluster_members
        # Используем INSERT OR REPLACE для предотвращения дубликатов (UNIQUE индекс на face_rectangle_id, person_id)
        cur.execute(
            """
            INSERT OR REPLACE INTO face_labels (face_rectangle_id, person_id, source, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (face_id, person_id, "cluster", 0.8, now),
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
        JOIN face_cluster_members fcm ON fl.face_rectangle_id = fcm.face_rectangle_id
        WHERE fcm.cluster_id = ?
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


def find_closest_cluster_with_person_for_face(
    *, face_rectangle_id: int, exclude_cluster_id: int | None = None, max_distance: float = 0.3, ignored_person_name: str = "Посторонние"
) -> dict[str, Any] | None:
    """
    Находит ближайший кластер с персоной (исключая "Посторонние") для лица по embedding.
    
    Args:
        face_rectangle_id: ID лица
        exclude_cluster_id: ID кластера, который нужно исключить из поиска
        max_distance: максимальное косинусное расстояние для попадания в кластер
        ignored_person_name: имя персоны, которую нужно исключить (по умолчанию "Посторонние")
    
    Returns:
        dict с информацией о ближайшем кластере:
        {
            'cluster_id': int,
            'person_id': int,
            'person_name': str,
            'distance': float
        }
        или None, если не найден подходящий
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
    
    # Получаем все кластеры с персоной (исключая "Посторонние") с их средними embeddings
    exclude_clause = ""
    exclude_params = [ignored_person_name]
    if exclude_cluster_id is not None:
        exclude_clause = "AND fc.id != ?"
        exclude_params.append(exclude_cluster_id)
    
    cur.execute(
        f"""
        SELECT 
            fc.id as cluster_id,
            fl.person_id,
            p.name as person_name,
            fr.embedding
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        JOIN face_labels fl ON fcm.face_rectangle_id = fl.face_rectangle_id
        JOIN persons p ON fl.person_id = p.id
        WHERE fr.embedding IS NOT NULL
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND p.name != ?
          {exclude_clause}
        GROUP BY fc.id, fl.person_id, p.name
        """,
        tuple(exclude_params),
    )
    
    clusters_embeddings: dict[int, dict[str, Any]] = {}
    
    for row in cur.fetchall():
        cluster_id = row["cluster_id"]
        emb_json = row["embedding"]
        
        # Сохраняем информацию о персоне для кластера
        if cluster_id not in clusters_embeddings:
            clusters_embeddings[cluster_id] = {
                "person_id": row["person_id"],
                "person_name": row["person_name"],
                "embeddings": [],
            }
        
        try:
            emb_list = json.loads(emb_json.decode("utf-8"))
            emb_array = np.array(emb_list, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_normalized = emb_array / norm
                clusters_embeddings[cluster_id]["embeddings"].append(emb_normalized)
        except Exception:
            continue
    
    if not clusters_embeddings:
        return None
    
    # Вычисляем средний embedding для каждого кластера и находим ближайший
    best_result = None
    best_distance = float('inf')
    
    for cluster_id, cluster_data in clusters_embeddings.items():
        embs = cluster_data["embeddings"]
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
            best_result = {
                "cluster_id": cluster_id,
                "person_id": cluster_data["person_id"],
                "person_name": cluster_data["person_name"],
                "distance": float(distance),
            }
    
    return best_result


def find_closest_cluster_with_person_for_face_by_min_distance(
    *, face_rectangle_id: int, exclude_cluster_id: int | None = None, max_distance: float = 0.3, ignored_person_name: str = "Посторонние"
) -> dict[str, Any] | None:
    """
    Находит ближайший кластер с персоной (исключая "Посторонние") для лица по embedding.
    Использует минимальное расстояние до любого лица в кластере (не среднее).
    
    Это лучше работает для детей, где лица могут сильно отличаться в разных фотографиях.
    Используется специально для поиска предложений для одиночных кластеров.
    
    Args:
        face_rectangle_id: ID лица
        exclude_cluster_id: ID кластера, который нужно исключить из поиска
        max_distance: максимальное косинусное расстояние для попадания в кластер
        ignored_person_name: имя персоны, которую нужно исключить (по умолчанию "Посторонние")
    
    Returns:
        dict с информацией о ближайшем кластере:
        {
            'cluster_id': int,
            'person_id': int,
            'person_name': str,
            'distance': float
        }
        или None, если не найден подходящий
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
    
    # Получаем все кластеры с персоной (исключая "Посторонние") с их embeddings
    exclude_clause = ""
    exclude_params = [ignored_person_name]
    if exclude_cluster_id is not None:
        exclude_clause = "AND fc.id != ?"
        exclude_params.append(exclude_cluster_id)
    
    cur.execute(
        f"""
        SELECT 
            fc.id as cluster_id,
            fl.person_id,
            p.name as person_name,
            fr.embedding
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        JOIN face_labels fl ON fcm.face_rectangle_id = fl.face_rectangle_id
        JOIN persons p ON fl.person_id = p.id
        WHERE fr.embedding IS NOT NULL
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND p.name != ?
          {exclude_clause}
        GROUP BY fc.id, fl.person_id, p.name
        """,
        tuple(exclude_params),
    )
    
    clusters_embeddings: dict[int, dict[str, Any]] = {}
    
    for row in cur.fetchall():
        cluster_id = row["cluster_id"]
        emb_json = row["embedding"]
        
        # Сохраняем информацию о персоне для кластера
        if cluster_id not in clusters_embeddings:
            clusters_embeddings[cluster_id] = {
                "person_id": row["person_id"],
                "person_name": row["person_name"],
                "embeddings": [],
            }
        
        try:
            emb_list = json.loads(emb_json.decode("utf-8"))
            emb_array = np.array(emb_list, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_normalized = emb_array / norm
                clusters_embeddings[cluster_id]["embeddings"].append(emb_normalized)
        except Exception:
            continue
    
    if not clusters_embeddings:
        return None
    
    # Вычисляем минимальное расстояние до любого лица в кластере (не среднее!)
    # Это лучше работает для детей, где лица могут сильно отличаться в разных фотографиях
    best_result = None
    best_distance = float('inf')
    
    for cluster_id, cluster_data in clusters_embeddings.items():
        embs = cluster_data["embeddings"]
        if not embs:
            continue
        
        # Для каждого лица в кластере вычисляем расстояние
        # Берем минимальное расстояние (ближайшее лицо)
        min_cluster_distance = float('inf')
        for emb_normalized in embs:
            # Косинусное расстояние
            distance = 1.0 - np.dot(face_emb_normalized, emb_normalized)
            if distance < min_cluster_distance:
                min_cluster_distance = distance
        
        # Используем минимальное расстояние из кластера
        if min_cluster_distance < best_distance and min_cluster_distance <= max_distance:
            best_distance = min_cluster_distance
            best_result = {
                "cluster_id": cluster_id,
                "person_id": cluster_data["person_id"],
                "person_name": cluster_data["person_name"],
                "distance": float(min_cluster_distance),
            }
    
    return best_result


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


def find_similar_single_face_clusters(
    *, max_distance: float = 0.6, run_id: int | None = None, archive_scope: str | None = None
) -> list[dict[str, Any]]:
    """
    Находит пары похожих одиночных кластеров (без персоны).
    
    Для каждого одиночного кластера ищет другой одиночный кластер с похожим лицом.
    Используется для предложения объединения кластеров.
    
    Args:
        max_distance: максимальное косинусное расстояние для попадания в пару
        run_id: опционально, фильтр по run_id
        archive_scope: опционально, фильтр по archive_scope (например, 'archive')
    
    Returns:
        list[dict] с парами похожих кластеров:
        [
            {
                'cluster1_id': int,
                'cluster1_face_id': int,
                'cluster2_id': int,
                'cluster2_face_id': int,
                'distance': float
            },
            ...
        ]
    """
    if not ML_AVAILABLE:
        return []
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие
    where_parts = []
    where_params = []
    
    if archive_scope == 'archive':
        where_parts.append("fc.archive_scope = 'archive'")
    elif run_id is not None:
        where_parts.append("fc.run_id = ?")
        where_params.append(run_id)
    
    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    
    # Находим все одиночные кластеры без персоны
    cur.execute(
        f"""
        SELECT 
            fc.id as cluster_id,
            fcm.face_rectangle_id as face_id,
            fr.embedding
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
               JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
               LIMIT 1) IS NULL
        ORDER BY fc.id
        """,
        tuple(where_params),
    )
    
    single_clusters = cur.fetchall()
    
    if len(single_clusters) < 2:
        return []  # Нужно минимум 2 кластера для сравнения
    
    # Загружаем embeddings
    cluster_embeddings: dict[int, dict[str, Any]] = {}
    
    for row in single_clusters:
        cluster_id = row["cluster_id"]
        face_id = row["face_id"]
        emb_json = row["embedding"]
        
        try:
            emb_list = json.loads(emb_json.decode("utf-8"))
            emb_array = np.array(emb_list, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_normalized = emb_array / norm
                cluster_embeddings[cluster_id] = {
                    "face_id": face_id,
                    "embedding": emb_normalized,
                }
        except Exception:
            continue
    
    if len(cluster_embeddings) < 2:
        return []
    
    # Находим пары похожих кластеров
    similar_pairs = []
    cluster_ids = list(cluster_embeddings.keys())
    
    # Сравниваем каждый кластер с каждым (но избегаем дублирования пар)
    for i in range(len(cluster_ids)):
        cluster1_id = cluster_ids[i]
        emb1 = cluster_embeddings[cluster1_id]["embedding"]
        face1_id = cluster_embeddings[cluster1_id]["face_id"]
        
        for j in range(i + 1, len(cluster_ids)):
            cluster2_id = cluster_ids[j]
            emb2 = cluster_embeddings[cluster2_id]["embedding"]
            face2_id = cluster_embeddings[cluster2_id]["face_id"]
            
            # Косинусное расстояние
            distance = 1.0 - np.dot(emb1, emb2)
            
            if distance <= max_distance:
                similar_pairs.append({
                    "cluster1_id": cluster1_id,
                    "cluster1_face_id": face1_id,
                    "cluster2_id": cluster2_id,
                    "cluster2_face_id": face2_id,
                    "distance": float(distance),
                })
    
    # Сортируем по расстоянию (от меньшего к большему)
    similar_pairs.sort(key=lambda x: x["distance"])
    
    return similar_pairs


def find_small_clusters_to_merge_in_person(
    *, max_size: int = 2, max_distance: float = 0.3, person_id: int | None = None, 
    run_id: int | None = None, archive_scope: str | None = None
) -> list[dict[str, Any]]:
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
        list[dict] с предложениями для объединения:
        [
            {
                'person_id': int,
                'person_name': str,
                'source_cluster_id': int,
                'source_cluster_size': int,
                'target_cluster_id': int,
                'target_cluster_size': int,
                'distance': float  # минимальное расстояние между кластерами
            },
            ...
        ]
    """
    if not ML_AVAILABLE:
        return []
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие
    where_parts = []
    where_params = []
    
    if archive_scope == 'archive':
        where_parts.append("fc.archive_scope = 'archive'")
    elif run_id is not None:
        where_parts.append("fc.run_id = ?")
        where_params.append(run_id)
    
    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    
    # Фильтр по person_id используем через подзапрос
    person_id_condition = ""
    if person_id is not None:
        person_id_condition = "AND (SELECT fl2.person_id FROM face_labels fl2 JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id WHERE fcm2.cluster_id = fc.id LIMIT 1) = ?"
        where_params.append(person_id)
    
    # Отладочное логирование
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"[find_small_clusters_to_merge_in_person] person_id={person_id}, max_size={max_size}, max_distance={max_distance}")
    logger.debug(f"[find_small_clusters_to_merge_in_person] WHERE clause: {where_clause}, person_id_condition: {person_id_condition}, params: {where_params}")
    
    # Находим все маленькие кластеры с персоной
    # Используем подзапрос для получения person_id, чтобы избежать дубликатов от JOIN с face_labels
    sql_query = f"""
        SELECT 
            (SELECT fl2.person_id 
             FROM face_labels fl2 
             JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
             LIMIT 1) as person_id,
            (SELECT p2.name 
             FROM face_labels fl2 
             JOIN persons p2 ON fl2.person_id = p2.id 
             JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
             LIMIT 1) as person_name,
            fc.id as cluster_id,
            COUNT(DISTINCT fcm.face_rectangle_id) as cluster_size
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE {where_clause}
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND fr.embedding IS NOT NULL
          AND (SELECT fl2.person_id 
               FROM face_labels fl2 
               JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
               LIMIT 1) IS NOT NULL
          {person_id_condition}
        GROUP BY fc.id
        HAVING cluster_size <= ?
        ORDER BY person_id, fc.id
    """
    
    logger.debug(f"[find_small_clusters_to_merge_in_person] SQL query: {sql_query}")
    logger.debug(f"[find_small_clusters_to_merge_in_person] SQL params: {tuple(where_params) + (max_size,)}")
    
    cur.execute(sql_query, tuple(where_params) + (max_size,))
    
    small_clusters_by_person: dict[int, list[dict[str, Any]]] = {}
    rows = cur.fetchall()
    logger.debug(f"[find_small_clusters_to_merge_in_person] Найдено маленьких кластеров: {len(rows)}")
    
    for row in rows:
        person_id_val = row["person_id"]
        logger.debug(f"[find_small_clusters_to_merge_in_person] Персона {person_id_val} ({row['person_name']}): кластер {row['cluster_id']}, размер {row['cluster_size']}")
        if person_id_val not in small_clusters_by_person:
            small_clusters_by_person[person_id_val] = {
                "person_name": row["person_name"],
                "clusters": [],
            }
        
        small_clusters_by_person[person_id_val]["clusters"].append({
            "cluster_id": row["cluster_id"],
            "size": row["cluster_size"],
        })
    
    logger.debug(f"[find_small_clusters_to_merge_in_person] Групп персон с маленькими кластерами: {len(small_clusters_by_person)}")
    
    if not small_clusters_by_person:
        return []
    
    # Для каждой персоны загружаем embeddings кластеров и находим похожие пары
    suggestions = []
    
    for person_id_val, person_data in small_clusters_by_person.items():
        person_name = person_data["person_name"]
        clusters = person_data["clusters"]
        
        # Если у персоны меньше 2 маленьких кластеров, пропускаем
        logger.debug(f"[find_small_clusters_to_merge_in_person] Персона {person_id_val} ({person_name}): {len(clusters)} маленьких кластеров")
        if len(clusters) < 2:
            logger.debug(f"[find_small_clusters_to_merge_in_person] Пропускаем персону {person_id_val}: меньше 2 кластеров")
            continue
        
        # Загружаем embeddings для всех кластеров этой персоны
        cluster_ids = [c["cluster_id"] for c in clusters]
        placeholders = ",".join("?" * len(cluster_ids))
        
        cur.execute(
            f"""
            SELECT 
                fc.id as cluster_id,
                fr.id as face_id,
                fr.embedding
            FROM face_clusters fc
            JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
            JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
            WHERE fc.id IN ({placeholders})
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND fr.embedding IS NOT NULL
            ORDER BY fc.id, fr.id
            """,
            tuple(cluster_ids),
        )
        
        # Группируем embeddings по кластерам
        cluster_embeddings: dict[int, list[np.ndarray]] = {}
        cluster_sizes: dict[int, int] = {}
        
        for row in cur.fetchall():
            cluster_id = row["cluster_id"]
            emb_json = row["embedding"]
            
            if cluster_id not in cluster_embeddings:
                cluster_embeddings[cluster_id] = []
                cluster_sizes[cluster_id] = 0
            
            try:
                emb_list = json.loads(emb_json.decode("utf-8"))
                emb_array = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_normalized = emb_array / norm
                    cluster_embeddings[cluster_id].append(emb_normalized)
                    cluster_sizes[cluster_id] += 1
            except Exception:
                continue
        
        # Находим пары похожих кластеров
        # Используем минимальное расстояние между любыми лицами из разных кластеров
        cluster_ids_list = list(cluster_embeddings.keys())
        
        for i in range(len(cluster_ids_list)):
            cluster1_id = cluster_ids_list[i]
            embs1 = cluster_embeddings[cluster1_id]
            size1 = cluster_sizes[cluster1_id]
            
            if not embs1:
                continue
            
            for j in range(i + 1, len(cluster_ids_list)):
                cluster2_id = cluster_ids_list[j]
                embs2 = cluster_embeddings[cluster2_id]
                size2 = cluster_sizes[cluster2_id]
                
                if not embs2:
                    continue
                
                # Вычисляем минимальное расстояние между любыми лицами из двух кластеров
                min_distance = float('inf')
                for emb1 in embs1:
                    for emb2 in embs2:
                        distance = 1.0 - np.dot(emb1, emb2)
                        if distance < min_distance:
                            min_distance = distance
                
                # Если минимальное расстояние не превышает порог, предлагаем объединение
                logger.debug(f"[find_small_clusters_to_merge_in_person] Кластеры {cluster1_id} и {cluster2_id}: расстояние {min_distance:.4f}, порог {max_distance}")
                if min_distance <= max_distance:
                    # Выбираем больший кластер как target (целевой)
                    if size2 > size1:
                        source_cluster_id = cluster1_id
                        target_cluster_id = cluster2_id
                        source_size = size1
                        target_size = size2
                    else:
                        source_cluster_id = cluster2_id
                        target_cluster_id = cluster1_id
                        source_size = size2
                        target_size = size1
                    
                    suggestions.append({
                        "person_id": person_id_val,
                        "person_name": person_name,
                        "source_cluster_id": source_cluster_id,
                        "source_cluster_size": source_size,
                        "target_cluster_id": target_cluster_id,
                        "target_cluster_size": target_size,
                        "distance": float(min_distance),
                    })
    
    # Сортируем по расстоянию (от меньшего к большему), затем по person_id
    suggestions.sort(key=lambda x: (x["distance"], x["person_id"]))
    
    return suggestions


def find_optimal_clusters_to_merge_in_person(
    *, max_source_size: int = 2, max_distance: float = 0.3, person_id: int | None = None,
    run_id: int | None = None, archive_scope: str | None = None
) -> list[dict[str, Any]]:
    """
    Находит оптимальные объединения кластеров в персоне для минимизации их количества.
    
    Для каждого маленького кластера находит ближайший кластер любого размера и предлагает
    объединение, если расстояние не превышает порог. Несколько маленьких кластеров могут
    объединяться в один большой.
    
    Args:
        max_source_size: максимальный размер кластера-источника для рассмотрения (по умолчанию 2)
        max_distance: максимальное косинусное расстояние для предложения объединения (по умолчанию 0.3)
        person_id: опционально, фильтр по person_id (только для одной персоны)
        run_id: опционально, фильтр по run_id
        archive_scope: опционально, фильтр по archive_scope (например, 'archive')
    
    Returns:
        list[dict] с предложениями для объединения:
        [
            {
                'person_id': int,
                'person_name': str,
                'source_cluster_id': int,
                'source_cluster_size': int,
                'target_cluster_id': int,
                'target_cluster_size': int,
                'distance': float  # минимальное расстояние между кластерами
            },
            ...
        ]
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[find_optimal_clusters_to_merge_in_person] ML_AVAILABLE = {ML_AVAILABLE}")
    
    if not ML_AVAILABLE:
        logger.warning("[find_optimal_clusters_to_merge_in_person] ML_AVAILABLE = False, возвращаем пустой список")
        logger.warning(f"[find_optimal_clusters_to_merge_in_person] numpy доступен: {np is not None}")
        try:
            import numpy as np_test
            logger.warning(f"[find_optimal_clusters_to_merge_in_person] numpy импортируется: {np_test is not None}")
        except ImportError as e:
            logger.warning(f"[find_optimal_clusters_to_merge_in_person] numpy НЕ импортируется: {e}")
        return []
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Формируем WHERE условие
    where_parts = []
    where_params = []
    
    if archive_scope == 'archive':
        where_parts.append("fc.archive_scope = 'archive'")
    elif run_id is not None:
        where_parts.append("fc.run_id = ?")
        where_params.append(run_id)
    
    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    
    # Фильтр по person_id используем через подзапрос
    # ВАЖНО: person_id НЕ добавляем в where_params, т.к. он передается отдельно в параметрах запросов
    person_id_condition = ""
    if person_id is not None:
        person_id_condition = "AND (SELECT fl2.person_id FROM face_labels fl2 JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id WHERE fcm2.cluster_id = fc.id LIMIT 1) = ?"
    logger.info(f"[find_optimal_clusters_to_merge_in_person] person_id={person_id}, max_source_size={max_source_size}, max_distance={max_distance}")
    
    # Шаг 1: Находим маленькие кластеры (источники)
    sql_query_sources = f"""
        SELECT 
            (SELECT fl2.person_id 
             FROM face_labels fl2 
             JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
             LIMIT 1) as person_id,
            (SELECT p2.name 
             FROM face_labels fl2 
             JOIN persons p2 ON fl2.person_id = p2.id 
             JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
             LIMIT 1) as person_name,
            fc.id as cluster_id,
            COUNT(DISTINCT fcm.face_rectangle_id) as cluster_size
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
        WHERE {where_clause}
          AND COALESCE(fr.ignore_flag, 0) = 0
          AND fr.embedding IS NOT NULL
          AND (SELECT fl2.person_id 
               FROM face_labels fl2 
               JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
               LIMIT 1) IS NOT NULL
          {person_id_condition}
        GROUP BY fc.id
        HAVING cluster_size <= ?
        ORDER BY person_id, fc.id
    """
    
    # Формируем параметры для первого запроса
    # Порядок: where_params (для where_clause) + person_id (для person_id_condition, если указан) + max_source_size (для HAVING)
    if person_id is not None:
        query_params = tuple(where_params) + (person_id, max_source_size)
    else:
        query_params = tuple(where_params) + (max_source_size,)
    
    logger.info(f"[find_optimal_clusters_to_merge_in_person] SQL params: where_params={where_params}, query_params={query_params}, person_id={person_id}")
    logger.debug(f"[find_optimal_clusters_to_merge_in_person] SQL query: {sql_query_sources}")
    
    cur.execute(sql_query_sources, query_params)
    small_clusters_rows = cur.fetchall()
    
    logger.info(f"[find_optimal_clusters_to_merge_in_person] Найдено маленьких кластеров: {len(small_clusters_rows)}")
    
    if len(small_clusters_rows) == 0:
        logger.warning(f"[find_optimal_clusters_to_merge_in_person] Не найдено маленьких кластеров (<= {max_source_size} фото)")
        return []
    
    # Группируем по персоне
    small_clusters_by_person: dict[int, list[dict[str, Any]]] = {}
    for row in small_clusters_rows:
        person_id_val = row["person_id"]
        if person_id_val not in small_clusters_by_person:
            small_clusters_by_person[person_id_val] = {
                "person_name": row["person_name"],
                "clusters": [],
            }
        small_clusters_by_person[person_id_val]["clusters"].append({
            "cluster_id": row["cluster_id"],
            "size": row["cluster_size"],
        })
    
    # Логируем, какие персоны найдены
    found_person_ids = list(small_clusters_by_person.keys())
    logger.info(f"[find_optimal_clusters_to_merge_in_person] Найдены персоны в результатах: {found_person_ids}, ожидается person_id={person_id}")
    
    # Шаг 2: Для каждой персоны загружаем embeddings всех кластеров (не только маленьких)
    suggestions = []
    
    for person_id_val, person_data in small_clusters_by_person.items():
        # Если указан person_id, обрабатываем только эту персону
        if person_id is not None and person_id_val != person_id:
            continue
            
        person_name = person_data["person_name"]
        source_clusters = person_data["clusters"]
        
        if len(source_clusters) == 0:
            continue
        
        # Получаем ВСЕ кластеры персоны (включая большие) для поиска целевых
        sql_all_clusters = f"""
            SELECT 
                fc.id as cluster_id,
                COUNT(DISTINCT fcm.face_rectangle_id) as cluster_size
            FROM face_clusters fc
            JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
            JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
            WHERE {where_clause}
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND fr.embedding IS NOT NULL
              AND (SELECT fl2.person_id 
                   FROM face_labels fl2 
                   JOIN face_cluster_members fcm2 ON fl2.face_rectangle_id = fcm2.face_rectangle_id
WHERE fcm2.cluster_id = fc.id 
                   LIMIT 1) = ?
            GROUP BY fc.id
            ORDER BY fc.id
        """
        
        # Для sql_all_clusters используем только where_params (без person_id) + person_id_val
        # person_id_val уже учтен через условие в SQL-запросе
        sql_all_params = tuple(where_params) + (person_id_val,)
        cur.execute(sql_all_clusters, sql_all_params)
        all_clusters_rows = cur.fetchall()
        all_cluster_ids = [row["cluster_id"] for row in all_clusters_rows]
        all_cluster_sizes = {row["cluster_id"]: row["cluster_size"] for row in all_clusters_rows}
        
        logger.info(f"[find_optimal_clusters_to_merge_in_person] Персона {person_id_val} ({person_name}): {len(source_clusters)} источников, {len(all_cluster_ids)} всего кластеров")
        
        if len(all_cluster_ids) == 0:
            continue
        
        # Загружаем embeddings для всех кластеров
        placeholders = ",".join("?" * len(all_cluster_ids))
        cur.execute(
            f"""
            SELECT 
                fc.id as cluster_id,
                fr.id as face_id,
                fr.embedding
            FROM face_clusters fc
            JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
            JOIN face_rectangles fr ON fcm.face_rectangle_id = fr.id
            WHERE fc.id IN ({placeholders})
              AND COALESCE(fr.ignore_flag, 0) = 0
              AND fr.embedding IS NOT NULL
            ORDER BY fc.id, fr.id
            """,
            tuple(all_cluster_ids),
        )
        
        # Группируем embeddings по кластерам
        cluster_embeddings: dict[int, list[np.ndarray]] = {}
        
        for row in cur.fetchall():
            cluster_id = row["cluster_id"]
            emb_json = row["embedding"]
            
            if cluster_id not in cluster_embeddings:
                cluster_embeddings[cluster_id] = []
            
            try:
                emb_list = json.loads(emb_json.decode("utf-8"))
                emb_array = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_normalized = emb_array / norm
                    cluster_embeddings[cluster_id].append(emb_normalized)
            except Exception:
                continue
        
        # Шаг 3: Для каждого маленького кластера находим ближайший любой кластер
        source_cluster_ids = [c["cluster_id"] for c in source_clusters]
        
        for source_cluster_info in source_clusters:
            source_cluster_id = source_cluster_info["cluster_id"]
            source_size = source_cluster_info["size"]
            source_embs = cluster_embeddings.get(source_cluster_id, [])
            
            if not source_embs:
                continue
            
            # Исключаем сам кластер из поиска
            candidate_target_ids = [cid for cid in all_cluster_ids if cid != source_cluster_id]
            
            best_target_id = None
            best_distance = float('inf')
            
            for target_cluster_id in candidate_target_ids:
                target_embs = cluster_embeddings.get(target_cluster_id, [])
                if not target_embs:
                    continue
                
                # Вычисляем минимальное расстояние между любыми лицами из двух кластеров
                min_distance = float('inf')
                for emb1 in source_embs:
                    for emb2 in target_embs:
                        distance = 1.0 - np.dot(emb1, emb2)
                        if distance < min_distance:
                            min_distance = distance
                
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_target_id = target_cluster_id
            
            # Если найдено подходящее объединение
            if best_target_id is not None:
                logger.debug(f"[find_optimal_clusters_to_merge_in_person] Кластер {source_cluster_id} -> {best_target_id}, расстояние: {best_distance:.4f}, порог: {max_distance}")
            if best_target_id is not None and best_distance <= max_distance:
                target_size = all_cluster_sizes.get(best_target_id, 0)
                
                suggestions.append({
                    "person_id": person_id_val,
                    "person_name": person_name,
                    "source_cluster_id": source_cluster_id,
                    "source_cluster_size": source_size,
                    "target_cluster_id": best_target_id,
                    "target_cluster_size": target_size,
                    "distance": float(best_distance),
                })
    
    # Сортируем по расстоянию (от меньшего к большему), затем по person_id
    suggestions.sort(key=lambda x: (x["distance"], x["person_id"]))
    
    logger.info(f"[find_optimal_clusters_to_merge_in_person] Итого найдено предложений: {len(suggestions)}")
    
    return suggestions


def merge_clusters(*, source_cluster_id: int, target_cluster_id: int) -> None:
    """
    Объединяет два кластера: перемещает все лица из source_cluster в target_cluster.
    
    Args:
        source_cluster_id: ID кластера-источника (будет пустым после объединения)
        target_cluster_id: ID целевого кластера (куда перемещаются лица)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, что оба кластера существуют
    cur.execute("SELECT id FROM face_clusters WHERE id IN (?, ?)", (source_cluster_id, target_cluster_id))
    found_clusters = {row["id"] for row in cur.fetchall()}
    
    if source_cluster_id not in found_clusters:
        raise ValueError(f"Source cluster {source_cluster_id} not found")
    if target_cluster_id not in found_clusters:
        raise ValueError(f"Target cluster {target_cluster_id} not found")
    
    if source_cluster_id == target_cluster_id:
        raise ValueError("Cannot merge cluster with itself")
    
    # Получаем все лица из source_cluster
    cur.execute(
        """
        SELECT face_rectangle_id
        FROM face_cluster_members
        WHERE cluster_id = ?
        """,
        (source_cluster_id,),
    )
    
    face_ids = [row["face_rectangle_id"] for row in cur.fetchall()]
    
    if len(face_ids) == 0:
        # Источник пустой, нечего перемещать
        return
    
    # Проверяем, нет ли уже этих лиц в target_cluster (избегаем дублирования)
    cur.execute(
        """
        SELECT face_rectangle_id
        FROM face_cluster_members
        WHERE cluster_id = ? AND face_rectangle_id IN ({})
        """.format(",".join("?" * len(face_ids))),
        [target_cluster_id] + face_ids,
    )
    
    existing_face_ids = {row["face_rectangle_id"] for row in cur.fetchall()}
    
    # Перемещаем только те лица, которых еще нет в target_cluster
    faces_to_move = [fid for fid in face_ids if fid not in existing_face_ids]
    
    # ВАЖНО: Получаем все face_labels для source_cluster ДО перемещения лиц
    # чтобы потом удалить их для всех перемещенных лиц
    # Кластер определяется через JOIN с face_cluster_members
    cur.execute(
        """
        SELECT fl.face_rectangle_id, fl.person_id
        FROM face_labels fl
        JOIN face_cluster_members fcm ON fl.face_rectangle_id = fcm.face_rectangle_id
        WHERE fcm.cluster_id = ?
        """,
        (source_cluster_id,),
    )
    source_labels_to_cleanup = cur.fetchall()
    source_labels_by_face = {(row["face_rectangle_id"], row["person_id"]) for row in source_labels_to_cleanup}
    
    # Если все лица уже в target_cluster, просто удаляем из source
    if len(faces_to_move) == 0:
        # Удаляем все связи из source_cluster (даже если они дублируются)
        cur.execute(
            """
            DELETE FROM face_cluster_members
            WHERE cluster_id = ?
            """,
            (source_cluster_id,),
        )
    else:
        # Перемещаем лица в target_cluster
        for face_id in faces_to_move:
            # Сначала удаляем из source
            cur.execute(
                """
                DELETE FROM face_cluster_members
                WHERE cluster_id = ? AND face_rectangle_id = ?
                """,
                (source_cluster_id, face_id),
            )
            
            # Затем добавляем в target
            cur.execute(
                """
                INSERT OR IGNORE INTO face_cluster_members (cluster_id, face_rectangle_id)
                VALUES (?, ?)
                """,
                (target_cluster_id, face_id),
            )
    
    # ВАЖНО: Удаляем face_labels для source_cluster для ВСЕХ лиц, которые были в source_cluster
    # независимо от того, были ли они уже в target_cluster или нет
    # Это предотвращает дубликаты записей face_labels
    for face_id in face_ids:  # Все лица, которые были в source_cluster
        # Находим person_id для этого лица из source_labels
        person_ids_for_face = [pid for fid, pid in source_labels_by_face if fid == face_id]
        
        for person_id in person_ids_for_face:
            # Удаляем face_label для этого лица и персоны
            # ВАЖНО: cluster_id больше не храним, удаляем только по face_rectangle_id + person_id
            cur.execute(
                """
                DELETE FROM face_labels
                WHERE face_rectangle_id = ? AND person_id = ?
                """,
                (face_id, person_id),
            )
            
            # Проверяем, есть ли уже face_label для этого лица и персоны
            # (независимо от кластера, т.к. cluster_id больше не храним)
            cur.execute(
                """
                SELECT id FROM face_labels
                WHERE face_rectangle_id = ? AND person_id = ?
                """,
                (face_id, person_id),
            )
            
            existing_label = cur.fetchone()
            
            if not existing_label:
                # Получаем данные из старой записи для создания новой
                # Ищем по face_rectangle_id + person_id (без cluster_id)
                cur.execute(
                    """
                    SELECT source, confidence, created_at
                    FROM face_labels
                    WHERE face_rectangle_id = ? AND person_id = ?
                    LIMIT 1
                    """,
                    (face_id, person_id),
                )
                old_label = cur.fetchone()
                
                if not old_label:
                    # Если не нашли старую запись (уже удалили), используем значения по умолчанию
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc).isoformat()
                    # ВАЖНО: cluster_id больше не храним в face_labels
                    # Используем INSERT OR REPLACE для предотвращения дубликатов
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO face_labels (face_rectangle_id, person_id, source, confidence, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (face_id, person_id, "cluster", 0.8, now),
                    )
                else:
                    # Создаем новую запись с данными из старой
                    # ВАЖНО: cluster_id больше не храним в face_labels
                    # Используем INSERT OR REPLACE для предотвращения дубликатов
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO face_labels (face_rectangle_id, person_id, source, confidence, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (face_id, person_id, old_label["source"], old_label["confidence"], old_label["created_at"]),
                    )
    
    # Проверяем, что source_cluster стал пустым, и удаляем его
    cur.execute(
        """
        SELECT COUNT(*) as count
        FROM face_cluster_members
        WHERE cluster_id = ?
        """,
        (source_cluster_id,),
    )
    remaining_faces = cur.fetchone()["count"]
    
    if remaining_faces == 0:
        # Удаляем оставшиеся face_labels для source_cluster (если они еще есть)
        # Основная очистка уже была сделана выше при перемещении лиц
        # Кластер определяется через JOIN с face_cluster_members
        cur.execute(
            """
            DELETE FROM face_labels
            WHERE face_rectangle_id IN (
                SELECT face_rectangle_id
                FROM face_cluster_members
                WHERE cluster_id = ?
            )
            """,
            (source_cluster_id,),
        )
        
        # Удаляем сам пустой кластер
        cur.execute("DELETE FROM face_clusters WHERE id = ?", (source_cluster_id,))
    
    conn.commit()
