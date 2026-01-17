import os
import sqlite3
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

# data/photosorter.db рядом с репозиторием (НЕ внутри backend/)
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "photosorter.db"


def get_connection():
    # ВАЖНО (Windows/SQLite): при параллельной работе web-сервера и worker-процесса
    # возможны кратковременные write-lock. Даем коннекту шанс "подождать", а не падать.
    try:
        timeout_sec = float(os.getenv("PHOTOSORTER_SQLITE_TIMEOUT_SEC") or "5")
    except Exception:
        timeout_sec = 5.0
    timeout_sec = max(0.1, min(60.0, timeout_sec))
    conn = sqlite3.connect(DB_PATH, timeout=timeout_sec)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
    except Exception:
        pass
    return conn


def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _ensure_columns(conn: sqlite3.Connection, table: str, ddl_by_column: dict[str, str]) -> None:
    """
    Безопасно добавляет колонки в существующую таблицу (если их нет).
    SQLite не поддерживает `ADD COLUMN IF NOT EXISTS`, поэтому проверяем через PRAGMA.
    """
    existing = _get_columns(conn, table)
    cur = conn.cursor()
    for col, ddl in ddl_by_column.items():
        if col in existing:
            continue
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        except sqlite3.OperationalError as e:
            # Возможна гонка: два потока одновременно делают миграцию.
            # В этом случае один успевает добавить колонку, второй падает на duplicate column name.
            msg = str(e).lower()
            if "duplicate column name" in msg:
                continue
            raise


def _now_utc_iso() -> str:
    # Храним времена в UTC в ISO-формате, чтобы не путать часовые пояса.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_db():
    # на всякий случай создаём папку data
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    cur = conn.cursor()

    # --- Миграция имени таблицы: yd_files -> files ---
    # Исторически таблица называлась yd_files, но давно хранит и local: пути.
    # Переименовываем безопасно при старте, чтобы старые базы не ломались после рефакторинга.
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('yd_files','files')")
        existing_tables = {str(r[0]) for r in cur.fetchall()}
        if "files" not in existing_tables and "yd_files" in existing_tables:
            cur.execute("ALTER TABLE yd_files RENAME TO files;")
            conn.commit()
        elif "files" in existing_tables and "yd_files" in existing_tables:
            # Возможна промежуточная ситуация, если код уже успел создать пустую `files`,
            # а реальные данные остались в `yd_files`.
            try:
                cur.execute("SELECT COUNT(*) FROM files")
                files_cnt = int(cur.fetchone()[0] or 0)
            except Exception:
                files_cnt = -1
            try:
                cur.execute("SELECT COUNT(*) FROM yd_files")
                yd_cnt = int(cur.fetchone()[0] or 0)
            except Exception:
                yd_cnt = -1

            if files_cnt == 0 and yd_cnt > 0:
                # Считаем `files` мусорной/пустой, восстанавливаем данные.
                cur.execute("DROP TABLE files;")
                cur.execute("ALTER TABLE yd_files RENAME TO files;")
                conn.commit()
            elif yd_cnt == 0 and files_cnt > 0:
                # Старую таблицу можно убрать (best-effort).
                cur.execute("DROP TABLE yd_files;")
                conn.commit()
    except Exception:
        # best-effort: не валим приложение из-за миграции имени;
        # если rename не прошёл, дальнейшие запросы покажут проблему явно.
        pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS folders (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            code                TEXT UNIQUE NOT NULL,  -- 'inbox', 'children_agata' ...
            path                TEXT NOT NULL,         -- 'disk:/...'
            name                TEXT,                  -- имя папки (последний сегмент)
            location            TEXT NOT NULL,         -- 'yadisk' / 'local'
            role                TEXT NOT NULL,         -- 'source' / 'target'
            priority_after_code TEXT,                  -- код папки-предшественника или NULL
            sort_order          INTEGER,               -- порядок этапа сортировки (меньше = раньше)
            content_rule        TEXT                   -- строка с правилом
        );
    """)

    _ensure_columns(
        conn,
        "folders",
        {
            "name": "name TEXT",
            "sort_order": "sort_order INTEGER",
        },
    )

    # --- Dedup: инвентарь файлов и прогоны скана архива (disk:/Фото) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dedup_runs (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            scope                   TEXT,            -- 'archive'|'source' (храним только последний на scope)
            root_path               TEXT NOT NULL,   -- например 'disk:/Фото'
            status                  TEXT NOT NULL,   -- 'running'|'completed'|'failed'
            limit_files             INTEGER,
            max_download_bytes      INTEGER,
            total_files             INTEGER,
            started_at              TEXT NOT NULL,
            finished_at             TEXT,
            processed_files         INTEGER NOT NULL DEFAULT 0,
            hashed_files            INTEGER NOT NULL DEFAULT 0,
            meta_hashed_files       INTEGER NOT NULL DEFAULT 0,
            downloaded_hashed_files INTEGER NOT NULL DEFAULT 0,
            skipped_large_files     INTEGER NOT NULL DEFAULT 0,
            errors_count            INTEGER NOT NULL DEFAULT 0,
            last_path               TEXT,
            last_error              TEXT
        );
    """)

    _ensure_columns(
        conn,
        "dedup_runs",
        {
            "scope": "scope TEXT",
            "total_files": "total_files INTEGER",
        },
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            path            TEXT UNIQUE NOT NULL,    -- 'disk:/...'
            resource_id     TEXT,                   -- устойчивый ID ресурса Я.Диска (не зависит от пути)
            inventory_scope TEXT,                   -- 'archive'|'source' (чтобы не смешивать разные инвентари)
            name            TEXT,
            parent_path     TEXT,
            size            INTEGER,
            created         TEXT,
            modified        TEXT,
            mime_type       TEXT,
            media_type      TEXT,
            hash_alg        TEXT,                   -- 'sha256'|'md5'
            hash_value      TEXT,
            hash_source     TEXT,                   -- 'meta'|'download'
            status          TEXT NOT NULL DEFAULT 'new', -- 'new'|'hashed'|'skipped_large'|'error'
            error           TEXT,
            scanned_at      TEXT,
            hashed_at       TEXT,
            last_run_id     INTEGER,
            ignore_archive_dup_run_id INTEGER,      -- если = source run_id, то не показывать "уже есть в архиве" до нового перескана
            duration_sec    INTEGER,                -- длительность видео (сек), если известна
            duration_source TEXT,                   -- 'ffprobe'|'meta'
            duration_at     TEXT                    -- когда обновляли длительность (UTC ISO)
        );
    """)

    _ensure_columns(
        conn,
        "files",
        {
            "resource_id": "resource_id TEXT",
            "inventory_scope": "inventory_scope TEXT",
            "ignore_archive_dup_run_id": "ignore_archive_dup_run_id INTEGER",
            "duration_sec": "duration_sec INTEGER",
            "duration_source": "duration_source TEXT",
            "duration_at": "duration_at TEXT",
            # Face-метаданные (quick win: лица/не лица для локальной сортировки)
            "faces_count": "faces_count INTEGER",
            "faces_run_id": "faces_run_id INTEGER",
            "faces_scanned_at": "faces_scanned_at TEXT",
            # Ручные правки (UI шага 2: лица/нет лиц)
            # faces_manual_label: 'faces'|'no_faces'|NULL
            "faces_manual_label": "faces_manual_label TEXT",
            "faces_manual_at": "faces_manual_at TEXT",

            # Локальная сортировка: дополнительные категории
            # Авто-карантин (экраны/технические фото/сомнительные кейсы)
            "faces_auto_quarantine": "faces_auto_quarantine INTEGER NOT NULL DEFAULT 0",
            "faces_quarantine_reason": "faces_quarantine_reason TEXT",
            # Авто-группы (2-й уровень вкладок /faces), НЕ связанные с карантином.
            # Пример: 'many_faces' (>=8 лиц).
            "faces_auto_group": "faces_auto_group TEXT",
            # Авто-детект животных (MVP: кошки)
            "animals_auto": "animals_auto INTEGER NOT NULL DEFAULT 0",
            "animals_kind": "animals_kind TEXT",
            # Ручная разметка животных (ground truth для метрик; не смешивать с animals_auto)
            "animals_manual": "animals_manual INTEGER NOT NULL DEFAULT 0",
            "animals_manual_kind": "animals_manual_kind TEXT",
            "animals_manual_at": "animals_manual_at TEXT",
            # Люди, но лица не найдены (пока в основном manual)
            "people_no_face_manual": "people_no_face_manual INTEGER NOT NULL DEFAULT 0",
            "people_no_face_person": "people_no_face_person TEXT",

            # --- Photo geo/time metadata for UI sorting (No Faces) ---
            # taken_at: ISO string (best-effort, from EXIF DateTimeOriginal; fallback: mtime in UTC)
            "taken_at": "taken_at TEXT",
            "gps_lat": "gps_lat REAL",
            "gps_lon": "gps_lon REAL",
            # normalized place (from geocoder)
            "place_country": "place_country TEXT",
            "place_city": "place_city TEXT",
            "place_source": "place_source TEXT",   # 'yandex'|'manual'|...
            "place_at": "place_at TEXT",           # when updated (UTC ISO)
            # Размеры исходного изображения (для правильного масштабирования bbox координат)
            "image_width": "image_width INTEGER",  # ширина исходного изображения (после EXIF transpose)
            "image_height": "image_height INTEGER",  # высота исходного изображения (после EXIF transpose)
        },
    )

    # --- Geocode cache (lat/lon -> country/city) ---
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS geocode_cache (
            key          TEXT PRIMARY KEY,   -- e.g. "55.7558,37.6173" (rounded)
            lat          REAL,
            lon          REAL,
            country      TEXT,
            city         TEXT,
            source       TEXT,               -- 'yandex'
            updated_at   TEXT NOT NULL,
            raw_json     TEXT                -- optional (debug)
        );
        """
    )

    # --- Pipeline: единый конвейер сортировки локальной папки с resume ---
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            kind            TEXT NOT NULL,        -- 'local_sort'
            root_path        TEXT NOT NULL,        -- 'C:\\Photos'
            status           TEXT NOT NULL,        -- 'running'|'completed'|'failed'
            step_num         INTEGER NOT NULL DEFAULT 0,
            step_total       INTEGER NOT NULL DEFAULT 0,
            step_title       TEXT,
            apply            INTEGER NOT NULL DEFAULT 0,
            skip_dedup       INTEGER NOT NULL DEFAULT 0,
            dedup_run_id     INTEGER,
            face_run_id      INTEGER,
            pid              INTEGER,
            last_src_path    TEXT,
            last_dst_path    TEXT,
            last_error       TEXT,
            log_tail         TEXT,
            started_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            finished_at      TEXT
        );
        """
    )

    _ensure_columns(
        conn,
        "pipeline_runs",
        {
            "kind": "kind TEXT NOT NULL DEFAULT 'local_sort'",
            "pid": "pid INTEGER",
            "last_src_path": "last_src_path TEXT",
            "last_dst_path": "last_dst_path TEXT",
            "log_tail": "log_tail TEXT",
            "updated_at": "updated_at TEXT",
        },
    )

    # --- Pipeline run metrics snapshots (gold-based) ---
    # Храним агрегаты "мимо gold" по каждому pipeline_run_id, чтобы они не "плыли" после следующего прогона
    # (так как авто-результаты в files.* перезаписываются).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_run_metrics (
            pipeline_run_id        INTEGER PRIMARY KEY,
            computed_at            TEXT NOT NULL,
            face_run_id            INTEGER,
            step0_checked          INTEGER,
            step0_non_media        INTEGER,
            step0_broken_media     INTEGER,
            step2_total            INTEGER,
            step2_processed        INTEGER,
            cats_total             INTEGER,
            cats_mism              INTEGER,
            faces_total            INTEGER,
            faces_mism             INTEGER,
            no_faces_total         INTEGER,
            no_faces_mism          INTEGER
        );
        """
    )

    # --- Preclean (шаг 1: предочистка): results + resume state ---
    # ВАЖНО: в DRY_RUN ничего не перемещаем, поэтому "результат" шага 1 — это список планируемых перемещений.
    # Для resume шага 1 сохраняем last_path и счётчики, чтобы после перезапуска продолжить.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS preclean_state (
            pipeline_run_id      INTEGER PRIMARY KEY,
            root_path            TEXT NOT NULL,
            dry_run              INTEGER NOT NULL,
            checked              INTEGER NOT NULL DEFAULT 0,
            non_media            INTEGER NOT NULL DEFAULT 0,
            broken_media         INTEGER NOT NULL DEFAULT 0,
            last_path            TEXT,
            updated_at           TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_preclean_state_root ON preclean_state(root_path);")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS preclean_moves (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_run_id  INTEGER NOT NULL,
            kind             TEXT NOT NULL, -- 'non_media'|'broken_media'
            src_path         TEXT NOT NULL,
            dst_path         TEXT NOT NULL,
            is_applied       INTEGER NOT NULL DEFAULT 0,
            created_at       TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_preclean_moves_run_kind ON preclean_moves(pipeline_run_id, kind);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_preclean_moves_run_src ON preclean_moves(pipeline_run_id, src_path);")

    # --- Run-scoped manual labels (pipeline_run_id + path) ---
    #
    # ВАЖНО: ручные метки должны быть привязаны к прогону, иначе они "утекают" между разными папками/прогонами
    # и портят отладку. Перенос ручных решений между прогонами делаем только через gold.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files_manual_labels (
            pipeline_run_id       INTEGER NOT NULL,
            path                  TEXT NOT NULL,
            faces_manual_label    TEXT,                      -- 'faces'|'no_faces'|NULL
            faces_manual_at       TEXT,
            people_no_face_manual INTEGER NOT NULL DEFAULT 0,
            people_no_face_person TEXT,
            animals_manual        INTEGER NOT NULL DEFAULT 0,
            animals_manual_kind   TEXT,
            animals_manual_at     TEXT,
            quarantine_manual     INTEGER NOT NULL DEFAULT 0,
            quarantine_manual_at  TEXT,
            PRIMARY KEY (pipeline_run_id, path)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_manual_labels_run ON files_manual_labels(pipeline_run_id);")

    # --- Run-scoped manual rectangles for VIDEO frames (3 frames per video) ---
    # ВАЖНО: для видео нам нужны прямоугольники с привязкой к кадру/таймкоду.
    # Храним отдельной таблицей, чтобы не смешивать с face_rectangles (она про фото/одно изображение).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS video_manual_frames (
            pipeline_run_id   INTEGER NOT NULL,
            path              TEXT NOT NULL,
            frame_idx         INTEGER NOT NULL, -- 1..3
            t_sec             REAL,             -- таймкод кадра (секунды), best-effort
            rects_json        TEXT,             -- JSON: [{"x":..,"y":..,"w":..,"h":..}, ...]
            updated_at        TEXT NOT NULL,
            PRIMARY KEY (pipeline_run_id, path, frame_idx)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_video_manual_frames_path ON video_manual_frames(path);")

    # Индексы для быстрых группировок дублей.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash_alg, hash_value);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent_path);")
    # Устойчивый идентификатор: уникален, если известен (partial unique index).
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_files_resource_id ON files(resource_id) WHERE resource_id IS NOT NULL;"
    )

    conn.commit()
    conn.close()


def _as_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


class FaceStore:
    """
    Хранилище результатов распознавания лиц (детект лиц на фото/видео).

    Важно: ML-часть запускается из отдельного окружения (.venv-face, Python 3.12),
    но данные пишем в общую SQLite БД проекта (data/photosorter.db).
    """

    def __init__(self) -> None:
        init_db()
        self.conn = get_connection()
        self._ensure_face_schema()

    def close(self) -> None:
        self.conn.close()

    def _ensure_face_schema(self) -> None:
        cur = self.conn.cursor()

        # Миграция: раньше таблица называлась face_detections, теперь face_rectangles.
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('face_detections','face_rectangles')")
        existing = {r[0] for r in cur.fetchall()}
        if "face_detections" in existing and "face_rectangles" not in existing:
            cur.execute("ALTER TABLE face_detections RENAME TO face_rectangles;")
            # Старые индексы могли называться idx_face_det_*. Их можно оставить,
            # но создаём новые с актуальными именами для читаемости.
            try:
                cur.execute("DROP INDEX IF EXISTS idx_face_det_run;")
                cur.execute("DROP INDEX IF EXISTS idx_face_det_file;")
            except Exception:
                pass

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS face_runs (
              id              INTEGER PRIMARY KEY AUTOINCREMENT,
              scope           TEXT NOT NULL,    -- 'yadisk'|'local'|'...'
              root_path       TEXT NOT NULL,    -- например 'disk:/Фото/Агата'
              status          TEXT NOT NULL,    -- 'running'|'completed'|'failed'
              total_files     INTEGER,
              processed_files INTEGER NOT NULL DEFAULT 0,
              faces_found     INTEGER NOT NULL DEFAULT 0,
              started_at      TEXT NOT NULL,
              finished_at     TEXT,
              last_path       TEXT,
              last_error      TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS face_rectangles (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id         INTEGER NOT NULL,
              file_path      TEXT NOT NULL,    -- 'disk:/...'
              face_index     INTEGER NOT NULL, -- индекс лица внутри файла в рамках текущего прогона
              bbox_x         INTEGER NOT NULL,
              bbox_y         INTEGER NOT NULL,
              bbox_w         INTEGER NOT NULL,
              bbox_h         INTEGER NOT NULL,
              confidence     REAL,
              presence_score REAL,             -- доля площади лица среди всех лиц в кадре
              thumb_jpeg     BLOB,             -- маленький кроп лица для UI/Inbox
              manual_person  TEXT,             -- имя/код персоны (пока строка; схему персон сделаем позже)
              ignore_flag    INTEGER NOT NULL DEFAULT 0,
              created_at     TEXT NOT NULL
            );
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_rect_run ON face_rectangles(run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_rect_file ON face_rectangles(file_path);")

        # Ручная разметка (UI шага 2: корректировка лиц/нет лиц)
        _ensure_columns(
            self.conn,
            "face_rectangles",
            {
                "is_manual": "is_manual INTEGER NOT NULL DEFAULT 0",
                "manual_created_at": "manual_created_at TEXT",
            },
        )
        
        # Face recognition embeddings (векторное представление лица для распознавания)
        _ensure_columns(
            self.conn,
            "face_rectangles",
            {
                "embedding": "embedding BLOB",  # JSON массив float32 (обычно 512 или 1024 элементов)
            },
        )
        
        # Archive scope для поддержки архива без привязки к прогонам
        # NULL или '' = для прогонов (сортируемые папки), 'archive' = для архива (текущий статус)
        _ensure_columns(
            self.conn,
            "face_rectangles",
            {
                "archive_scope": "archive_scope TEXT",  # NULL|'' для прогонов, 'archive' для архива
            },
        )

        # Справочник персон (людей)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT UNIQUE NOT NULL,
                mode            TEXT NOT NULL DEFAULT 'active',  -- 'active'|'deferred'|'never'
                is_me           INTEGER NOT NULL DEFAULT 0,
                kinship         TEXT,                          -- степень родства/близости
                avatar_face_id  INTEGER,                       -- FK к face_rectangles.id
                created_at       TEXT NOT NULL,
                updated_at       TEXT
            );
        """)

        # Кластеры лиц
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_clusters (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL,
                method          TEXT NOT NULL,                  -- 'DBSCAN'|'HDBSCAN'|...
                params_json     TEXT,                          -- JSON с параметрами кластеризации
                created_at       TEXT NOT NULL
            );
        """)

        # Связь лиц с кластерами
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_cluster_members (
                cluster_id          INTEGER NOT NULL,
                face_rectangle_id   INTEGER NOT NULL,
                PRIMARY KEY (cluster_id, face_rectangle_id)
            );
        """)

        # Метки лиц (назначение персоны лицу)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_labels (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                face_rectangle_id   INTEGER NOT NULL,
                person_id           INTEGER NOT NULL,
                cluster_id          INTEGER,                   -- optional: если source=cluster
                source              TEXT NOT NULL,             -- 'manual'|'cluster'|'ai'
                confidence          REAL,
                created_at          TEXT NOT NULL
            );
        """)

        # Archive scope для кластеров (аналогично face_rectangles)
        _ensure_columns(
            self.conn,
            "face_clusters",
            {
                "archive_scope": "archive_scope TEXT",  # NULL|'' для прогонов, 'archive' для архива
            },
        )
        
        # Индексы
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_run ON face_clusters(run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_rect_archive_scope ON face_rectangles(archive_scope);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_archive_scope ON face_clusters(archive_scope);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_cluster_members_cluster ON face_cluster_members(cluster_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_cluster_members_face ON face_cluster_members(face_rectangle_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_labels_face ON face_labels(face_rectangle_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_labels_person ON face_labels(person_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_labels_cluster ON face_labels(cluster_id);")

        self.conn.commit()

    def list_rectangles(self, *, run_id: int, file_path: str) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
              id, run_id, file_path, face_index,
              bbox_x, bbox_y, bbox_w, bbox_h,
              confidence, presence_score,
              manual_person, ignore_flag,
              created_at,
              COALESCE(is_manual, 0) AS is_manual,
              manual_created_at
            FROM face_rectangles
            WHERE run_id = ? AND file_path = ?
            ORDER BY COALESCE(is_manual, 0) ASC, face_index ASC, id ASC
            """,
            (int(run_id), str(file_path)),
        )
        return [dict(r) for r in cur.fetchall()]

    def replace_manual_rectangles(self, *, run_id: int, file_path: str, rects: list[dict[str, int]]) -> None:
        """
        Заменяет ручные прямоугольники для файла (run_id + file_path).
        rects: [{"x":int,"y":int,"w":int,"h":int}, ...]
        """
        now = _now_utc_iso()
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM face_rectangles WHERE run_id = ? AND file_path = ? AND COALESCE(is_manual, 0) = 1",
            (int(run_id), str(file_path)),
        )
        for i, r in enumerate(rects or []):
            x = int(r.get("x") or 0)
            y = int(r.get("y") or 0)
            w = int(r.get("w") or 0)
            h = int(r.get("h") or 0)
            if w <= 0 or h <= 0:
                continue
            cur.execute(
                """
                INSERT INTO face_rectangles(
                  run_id, file_path, face_index,
                  bbox_x, bbox_y, bbox_w, bbox_h,
                  confidence, presence_score,
                  thumb_jpeg, manual_person, ignore_flag,
                  created_at,
                  is_manual, manual_created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0, ?, 1, ?)
                """,
                (int(run_id), str(file_path), int(i), x, y, w, h, now, now),
            )
        self.conn.commit()

    def create_run(self, *, scope: str, root_path: str, total_files: int | None) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO face_runs(scope, root_path, status, total_files, started_at)
            VALUES (?, ?, 'running', ?, ?)
            """,
            (scope, root_path, _as_int(total_files), _now_utc_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_run_by_id(self, *, run_id: int) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM face_runs WHERE id = ? LIMIT 1", (int(run_id),))
        row = cur.fetchone()
        return dict(row) if row else None

    def finish_run(self, *, run_id: int, status: str, last_error: str | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE face_runs
            SET status = ?, finished_at = ?, last_error = COALESCE(?, last_error)
            WHERE id = ?
            """,
            (status, _now_utc_iso(), last_error, run_id),
        )
        self.conn.commit()

    def update_run_progress(
        self,
        *,
        run_id: int,
        processed_files: int | None = None,
        faces_found: int | None = None,
        last_path: str | None = None,
        last_error: str | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []
        for key, val in [
            ("processed_files", processed_files),
            ("faces_found", faces_found),
            ("last_path", last_path),
            ("last_error", last_error),
        ]:
            if val is None:
                continue
            fields.append(f"{key} = ?")
            params.append(val)
        if not fields:
            return
        params.append(run_id)
        cur = self.conn.cursor()
        cur.execute(f"UPDATE face_runs SET {', '.join(fields)} WHERE id = ?", params)
        self.conn.commit()

    def clear_run_detections_for_file(self, *, run_id: int, file_path: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM face_rectangles WHERE run_id = ? AND file_path = ?", (run_id, file_path))
        self.conn.commit()

    def clear_run_auto_rectangles_for_file(self, *, run_id: int, file_path: str) -> None:
        """
        Удаляет только авто-прямоугольники (is_manual=0) для файла в рамках прогона.
        Ручные прямоугольники (is_manual=1) сохраняем.
        """
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM face_rectangles WHERE run_id = ? AND file_path = ? AND COALESCE(is_manual, 0) = 0",
            (int(run_id), str(file_path)),
        )
        self.conn.commit()

    def insert_detection(
        self,
        *,
        run_id: int | None = None,
        archive_scope: str | None = None,
        file_path: str,
        face_index: int,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        confidence: float | None,
        presence_score: float | None,
        thumb_jpeg: bytes | None,
        embedding: bytes | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> bool:
        """
        Вставляет детекцию лица. Для архивного режима (archive_scope='archive') 
        проверяет дубликаты перед вставкой (append без дублирования).
        
        Returns:
            True если запись была вставлена, False если была пропущена (дубликат в архиве)
        """
        cur = self.conn.cursor()
        
        # Для архивного режима проверяем дубликаты (append без дублирования)
        if archive_scope == 'archive':
            # Проверяем существование по file_path + bbox
            cur.execute(
                """
                SELECT id FROM face_rectangles
                WHERE archive_scope = 'archive'
                  AND file_path = ?
                  AND bbox_x = ?
                  AND bbox_y = ?
                  AND bbox_w = ?
                  AND bbox_h = ?
                LIMIT 1
                """,
                (file_path, int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)),
            )
            existing = cur.fetchone()
            if existing is not None:
                # Дубликат найден - пропускаем вставку
                return False
        
        # Определяем колонки для INSERT в зависимости от наличия run_id и archive_scope
        if archive_scope == 'archive':
            # Для архива run_id может быть NULL
            cur.execute(
                """
                INSERT INTO face_rectangles(
                  run_id, archive_scope, file_path, face_index,
                  bbox_x, bbox_y, bbox_w, bbox_h,
                  confidence, presence_score, thumb_jpeg,
                  embedding, manual_person, ignore_flag, created_at,
                  is_manual, manual_created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, 0, NULL)
                """,
                (
                    run_id,  # может быть NULL для архива
                    archive_scope,
                    file_path,
                    int(face_index),
                    int(bbox_x),
                    int(bbox_y),
                    int(bbox_w),
                    int(bbox_h),
                    float(confidence) if confidence is not None else None,
                    float(presence_score) if presence_score is not None else None,
                    thumb_jpeg,
                    embedding,  # может быть NULL
                    _now_utc_iso(),
                ),
            )
        else:
            # Для прогонов run_id обязателен, archive_scope NULL
            if run_id is None:
                raise ValueError("run_id обязателен для неархивных записей")
            cur.execute(
                """
                INSERT INTO face_rectangles(
                  run_id, archive_scope, file_path, face_index,
                  bbox_x, bbox_y, bbox_w, bbox_h,
                  confidence, presence_score, thumb_jpeg,
                  embedding, manual_person, ignore_flag, created_at,
                  is_manual, manual_created_at
                )
                VALUES(?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, 0, NULL)
                """,
                (
                    run_id,
                    file_path,
                    int(face_index),
                    int(bbox_x),
                    int(bbox_y),
                    int(bbox_w),
                    int(bbox_h),
                    float(confidence) if confidence is not None else None,
                    float(presence_score) if presence_score is not None else None,
                    thumb_jpeg,
                    embedding,  # может быть NULL
                    _now_utc_iso(),
                ),
            )
        
        # Обновляем размеры изображения в таблице files (если они указаны)
        if image_width is not None and image_height is not None:
            cur.execute(
                """
                UPDATE files 
                SET image_width = ?, image_height = ?
                WHERE path = ? AND (image_width IS NULL OR image_height IS NULL)
                """,
                (int(image_width), int(image_height), file_path),
            )
        self.conn.commit()
        return True

    def find_similar_faces(
        self,
        *,
        embedding_json: bytes,
        run_id: int | None = None,
        similarity_threshold: float = 0.6,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Находит похожие лица по embedding (косинусное расстояние).
        
        Args:
            embedding_json: JSON-сериализованный embedding (bytes)
            run_id: ограничить поиск определённым run_id (опционально)
            similarity_threshold: минимальный порог схожести (0.0-1.0, косинусное расстояние)
            limit: максимальное количество результатов
        
        Returns:
            список словарей с информацией о похожих лицах
        """
        try:
            import numpy as np
            
            # Десериализуем query embedding
            query_emb = np.array(json.loads(embedding_json.decode("utf-8")), dtype=np.float32)
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                return []
            query_emb = query_emb / query_norm  # нормализуем
            
            cur = self.conn.cursor()
            
            # Получаем все embeddings из БД
            if run_id is not None:
                cur.execute(
                    """
                    SELECT id, run_id, file_path, face_index, bbox_x, bbox_y, bbox_w, bbox_h,
                           confidence, embedding
                    FROM face_rectangles
                    WHERE run_id = ? AND embedding IS NOT NULL AND COALESCE(ignore_flag, 0) = 0
                    """,
                    (int(run_id),),
                )
            else:
                cur.execute(
                    """
                    SELECT id, run_id, file_path, face_index, bbox_x, bbox_y, bbox_w, bbox_h,
                           confidence, embedding
                    FROM face_rectangles
                    WHERE embedding IS NOT NULL AND COALESCE(ignore_flag, 0) = 0
                    """,
                )
            
            results: list[tuple[float, dict[str, Any]]] = []
            
            for row in cur.fetchall():
                try:
                    db_emb_json = row[9]  # embedding column
                    if not db_emb_json:
                        continue
                    
                    # Десериализуем embedding из БД
                    db_emb = np.array(json.loads(db_emb_json.decode("utf-8")), dtype=np.float32)
                    db_norm = np.linalg.norm(db_emb)
                    if db_norm == 0:
                        continue
                    db_emb = db_emb / db_norm  # нормализуем
                    
                    # Вычисляем косинусное расстояние (cosine similarity)
                    similarity = float(np.dot(query_emb, db_emb))
                    
                    if similarity >= similarity_threshold:
                        results.append((
                            similarity,
                            {
                                "id": row[0],
                                "run_id": row[1],
                                "file_path": row[2],
                                "face_index": row[3],
                                "bbox_x": row[4],
                                "bbox_y": row[5],
                                "bbox_w": row[6],
                                "bbox_h": row[7],
                                "confidence": row[8],
                                "similarity": similarity,
                            },
                        ))
                except Exception:
                    continue
            
            # Сортируем по similarity (от большего к меньшему)
            results.sort(key=lambda x: x[0], reverse=True)
            
            # Возвращаем top-K
            return [r[1] for r in results[:limit]]
        except Exception:
            return []

    def update_file_path(self, *, old_file_path: str, new_file_path: str) -> None:
        """
        Обновляет file_path для face_rectangles (когда файл физически перенесли на диске/в YaDisk).

        Важно: это чисто "техническая" миграция ссылок. Семантика детекта не меняется.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE face_rectangles
            SET file_path = ?
            WHERE file_path = ?
            """,
            (new_file_path, old_file_path),
        )
        self.conn.commit()


def list_folders(*, location: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
    """
    Возвращает папки из таблицы `folders` в порядке:
    1) с заданным sort_order (по возрастанию)
    2) остальные (по name)

    Фильтры `location` и `role` опциональны.
    """
    init_db()
    conn = get_connection()
    try:
        where: list[str] = []
        params: list[Any] = []
        if location is not None:
            where.append("location = ?")
            params.append(location)
        if role is not None:
            where.append("role = ?")
            params.append(role)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, code, name, path, location, role, sort_order, priority_after_code, content_rule
            FROM folders
            {where_sql}
            ORDER BY
              (sort_order IS NULL) ASC,
              sort_order ASC,
              COALESCE(name, '') ASC,
              code ASC
            """,
            params,
        )

        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


class DedupStore:
    """
    Хранилище дедуп-данных поверх текущей SQLite БД (data/photosorter.db).
    Используется для заполнения БД дублей (инвентарь файлов + хэши + статус прогона).
    """

    def __init__(self) -> None:
        init_db()
        self.conn = get_connection()

    def close(self) -> None:
        self.conn.close()

    def create_run(
        self,
        *,
        scope: str,
        root_path: str,
        max_download_bytes: int | None,
    ) -> int:
        cur = self.conn.cursor()
        # Упрощение: истории не ведём — держим только последний прогон на scope.
        cur.execute("DELETE FROM dedup_runs WHERE scope = ?", (scope,))
        cur.execute(
            """
            INSERT INTO dedup_runs(scope, root_path, status, limit_files, max_download_bytes, started_at)
            VALUES (?, ?, 'running', NULL, ?, ?)
            """,
            (scope, root_path, max_download_bytes, _now_utc_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, *, run_id: int, status: str, last_error: str | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE dedup_runs
            SET status = ?, finished_at = ?, last_error = COALESCE(?, last_error)
            WHERE id = ?
            """,
            (status, _now_utc_iso(), last_error, run_id),
        )
        # Если прогон завершён, но processed_files не успели "догнать" total_files,
        # доводим до консистентного состояния (иначе UI показывает 0% при completed).
        if str(status) == "completed":
            cur.execute(
                """
                UPDATE dedup_runs
                SET processed_files = CASE
                    WHEN total_files IS NOT NULL AND processed_files < total_files THEN total_files
                    ELSE processed_files
                END
                WHERE id = ?
                """,
                (int(run_id),),
            )
        self.conn.commit()

    def update_run_progress(
        self,
        *,
        run_id: int,
        total_files: int | None = None,
        processed_files: int | None = None,
        hashed_files: int | None = None,
        meta_hashed_files: int | None = None,
        downloaded_hashed_files: int | None = None,
        skipped_large_files: int | None = None,
        errors_count: int | None = None,
        last_path: str | None = None,
        last_error: str | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []
        for key, val in [
            ("total_files", total_files),
            ("processed_files", processed_files),
            ("hashed_files", hashed_files),
            ("meta_hashed_files", meta_hashed_files),
            ("downloaded_hashed_files", downloaded_hashed_files),
            ("skipped_large_files", skipped_large_files),
            ("errors_count", errors_count),
            ("last_path", last_path),
            ("last_error", last_error),
        ]:
            if val is None:
                continue
            fields.append(f"{key} = ?")
            params.append(val)

        if not fields:
            return
        params.append(run_id)
        sql = f"UPDATE dedup_runs SET {', '.join(fields)} WHERE id = ?"
        cur = self.conn.cursor()
        cur.execute(sql, params)
        self.conn.commit()

    def get_latest_run(self, *, scope: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM dedup_runs
            WHERE scope = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (scope,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_run_by_id(self, *, run_id: int) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM dedup_runs WHERE id = ? LIMIT 1", (int(run_id),))
        row = cur.fetchone()
        return dict(row) if row else None

    # --- files helpers (инвентарь/хэши/дубли) ---
    #
    # Эти методы нужны и Web UI (дедуп архива/источника), и локальному конвейеру
    # (dedup локальной папки + idempotent update_path). Ранее они оказались
    # внутри PipelineStore, из-за чего DedupStore падал с AttributeError.

    def upsert_file(
        self,
        *,
        run_id: int,
        path: str,
        resource_id: str | None = None,
        inventory_scope: str | None = None,
        name: str | None,
        parent_path: str | None,
        size: int | None,
        created: str | None,
        modified: str | None,
        mime_type: str | None,
        media_type: str | None,
        hash_alg: str | None,
        hash_value: str | None,
        hash_source: str | None,
        status: str,
        error: str | None,
        scanned_at: str | None = None,
        hashed_at: str | None = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO files(
              path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type,
              hash_alg, hash_value, hash_source, status, error, scanned_at, hashed_at, last_run_id
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              resource_id = COALESCE(excluded.resource_id, files.resource_id),
              inventory_scope = COALESCE(excluded.inventory_scope, files.inventory_scope),
              name = excluded.name,
              parent_path = excluded.parent_path,
              size = excluded.size,
              created = excluded.created,
              modified = excluded.modified,
              mime_type = excluded.mime_type,
              media_type = excluded.media_type,
              hash_alg = COALESCE(excluded.hash_alg, files.hash_alg),
              hash_value = COALESCE(excluded.hash_value, files.hash_value),
              hash_source = COALESCE(excluded.hash_source, files.hash_source),
              status = excluded.status,
              error = excluded.error,
              scanned_at = excluded.scanned_at,
              hashed_at = COALESCE(excluded.hashed_at, files.hashed_at),
              last_run_id = excluded.last_run_id
            """,
            (
                path,
                resource_id,
                inventory_scope,
                name,
                parent_path,
                size,
                created,
                modified,
                mime_type,
                media_type,
                hash_alg,
                hash_value,
                hash_source,
                status,
                error,
                scanned_at or _now_utc_iso(),
                hashed_at,
                run_id,
            ),
        )
        self.conn.commit()

    def get_existing_hash(self, *, path: str) -> tuple[str | None, str | None]:
        cur = self.conn.cursor()
        cur.execute("SELECT hash_alg, hash_value FROM files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if not row:
            return None, None
        return (row["hash_alg"], row["hash_value"])

    def list_dup_groups_archive(self) -> list[dict[str, Any]]:
        """
        Возвращает группы дублей для архива YaDisk (inventory_scope='archive').
        Группа = одинаковый (hash_alg, hash_value), где количество файлов > 1.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
              hash_alg,
              hash_value,
              COUNT(*) AS cnt
            FROM files
            WHERE
              hash_value IS NOT NULL
              AND COALESCE(inventory_scope, 'archive') = 'archive'
              AND status != 'deleted'
            GROUP BY hash_alg, hash_value
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC, hash_alg ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def list_group_items(self, *, hash_alg: str, hash_value: str, inventory_scope: str | None = None, last_run_id: int | None = None) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        where: list[str] = ["hash_alg = ? AND hash_value = ?", "status != 'deleted'"]
        params: list[Any] = [hash_alg, hash_value]
        if inventory_scope is not None:
            where.append("inventory_scope = ?")
            params.append(inventory_scope)
        if last_run_id is not None:
            where.append("last_run_id = ?")
            params.append(last_run_id)
        cur.execute(
            f"""
            SELECT path, name, parent_path, size, modified, mime_type, media_type, hash_source
            FROM files
            WHERE {' AND '.join(where)}
            ORDER BY path ASC
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]

    def set_ignore_archive_dup(self, *, paths: list[str], run_id: int) -> int:
        """
        Помечает source-файлы как "не дубль архива" ДЛЯ ТЕКУЩЕГО source run_id.
        При новом пересканировании source run_id изменится — пометка автоматически перестанет действовать.
        """
        if not paths:
            return 0
        cur = self.conn.cursor()
        q = ",".join(["?"] * len(paths))
        cur.execute(
            f"""
            UPDATE files
            SET ignore_archive_dup_run_id = ?
            WHERE path IN ({q})
            """,
            [run_id, *paths],
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def list_source_dups_in_archive(
        self,
        *,
        source_run_id: int,
        archive_prefixes: tuple[str, ...] = ("disk:/Фото",),
    ) -> list[dict[str, Any]]:
        """
        Возвращает пары (source file -> archive matches) по совпадающему хэшу.
        Источник ограничиваем last_run_id=source_run_id, чтобы не смешивать разные выборы папки.
        """
        cur = self.conn.cursor()
        prefixes = [str(p or "").rstrip("/") + "/%" for p in (archive_prefixes or ()) if str(p or "").strip()]
        if not prefixes:
            return []
        like_sql = " OR ".join(["a.path LIKE ?"] * len(prefixes))
        cur.execute(
            f"""
            SELECT
              s.path AS source_path,
              s.name AS source_name,
              s.size AS source_size,
              s.mime_type AS source_mime_type,
              s.media_type AS source_media_type,
              s.hash_alg AS hash_alg,
              s.hash_value AS hash_value,
              a.path AS archive_path,
              a.name AS archive_name,
              a.size AS archive_size,
              a.mime_type AS archive_mime_type,
              a.media_type AS archive_media_type
            FROM files s
            JOIN files a
              ON a.hash_alg = s.hash_alg AND a.hash_value = s.hash_value
            WHERE
              s.inventory_scope = 'source'
              AND s.last_run_id = ?
              AND s.status != 'deleted'
              AND s.hash_value IS NOT NULL
              AND COALESCE(s.ignore_archive_dup_run_id, -1) != ?
              AND COALESCE(a.inventory_scope, 'archive') = 'archive'
              AND a.status != 'deleted'
              AND a.hash_value IS NOT NULL
              AND ({like_sql})
            ORDER BY s.path ASC, a.path ASC
            """,
            (source_run_id, source_run_id, *prefixes),
        )
        return [dict(r) for r in cur.fetchall()]

    def list_dup_groups_for_run(self, *, inventory_scope: str, run_id: int) -> list[dict[str, Any]]:
        """
        Группы дублей ВНУТРИ конкретного прогона (run_id) выбранного scope.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
              hash_alg,
              hash_value,
              COUNT(*) AS cnt
            FROM files
            WHERE
              hash_value IS NOT NULL
              AND inventory_scope = ?
              AND last_run_id = ?
              AND status != 'deleted'
            GROUP BY hash_alg, hash_value
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC, hash_alg ASC
            """,
            (inventory_scope, run_id),
        )
        return [dict(r) for r in cur.fetchall()]

    def path_exists(self, *, path: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM files WHERE path = ? LIMIT 1", (path,))
        return cur.fetchone() is not None

    def mark_deleted(self, *, paths: list[str]) -> int:
        """
        Помечает файлы как удалённые (в корзину/удалены на стороне YaDisk), чтобы они
        исчезали из /duplicates без перескана.
        """
        if not paths:
            return 0
        cur = self.conn.cursor()
        q = ",".join(["?"] * len(paths))
        cur.execute(f"UPDATE files SET status = 'deleted', error = NULL WHERE path IN ({q})", paths)
        self.conn.commit()
        return int(cur.rowcount or 0)

    def update_path(self, *, old_path: str, new_path: str, new_name: str | None, new_parent_path: str | None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET path = ?, name = COALESCE(?, name), parent_path = COALESCE(?, parent_path)
            WHERE path = ?
            """,
            (new_path, new_name, new_parent_path, old_path),
        )
        self.conn.commit()

    def set_faces_summary(self, *, path: str, faces_run_id: int, faces_count: int, faces_scanned_at: str | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET faces_count = ?, faces_run_id = ?, faces_scanned_at = ?
            WHERE path = ?
            """,
            (int(faces_count), int(faces_run_id), faces_scanned_at or _now_utc_iso(), path),
        )
        self.conn.commit()

    def set_faces_manual_label(self, *, path: str, label: str | None) -> None:
        """
        Ручная правка результата "лица/нет лиц" для файла.
        label: 'faces' | 'no_faces' | None (сброс)
        """
        lab = (label or "").strip().lower()
        if lab == "":
            lab = ""
        if lab not in ("", "faces", "no_faces"):
            raise ValueError("label must be one of: faces, no_faces, (empty)")
        cur = self.conn.cursor()
        if lab == "":
            cur.execute(
                """
                UPDATE files
                SET faces_manual_label = NULL, faces_manual_at = NULL
                WHERE path = ?
                """,
                (path,),
            )
        else:
            cur.execute(
                """
                UPDATE files
                SET faces_manual_label = ?, faces_manual_at = ?
                WHERE path = ?
                """,
                (lab, _now_utc_iso(), path),
            )
        self.conn.commit()

    # --- run-scoped manual labels (pipeline_run_id + path) ---
    def _ensure_run_manual_row(self, *, pipeline_run_id: int, path: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO files_manual_labels(pipeline_run_id, path) VALUES (?, ?)",
            (int(pipeline_run_id), str(path)),
        )

    def delete_run_manual_labels(self, *, pipeline_run_id: int, path: str) -> None:
        """
        Полный сброс ручных меток для конкретного прогона и пути.
        """
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM files_manual_labels WHERE pipeline_run_id = ? AND path = ?",
            (int(pipeline_run_id), str(path)),
        )
        self.conn.commit()

    def set_run_faces_manual_label(self, *, pipeline_run_id: int, path: str, label: str | None) -> None:
        """
        Ручная правка результата "лица/нет лиц" для файла В РАМКАХ ПРОГОНА.
        label: 'faces' | 'no_faces' | None (сброс)
        """
        lab = (label or "").strip().lower()
        if lab == "":
            lab = ""
        if lab not in ("", "faces", "no_faces"):
            raise ValueError("label must be one of: faces, no_faces, (empty)")
        self._ensure_run_manual_row(pipeline_run_id=int(pipeline_run_id), path=str(path))
        cur = self.conn.cursor()
        if lab == "":
            cur.execute(
                """
                UPDATE files_manual_labels
                SET faces_manual_label = NULL, faces_manual_at = NULL
                WHERE pipeline_run_id = ? AND path = ?
                """,
                (int(pipeline_run_id), str(path)),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET faces_manual_label = ?, faces_manual_at = ?
                WHERE pipeline_run_id = ? AND path = ?
                """,
                (lab, _now_utc_iso(), int(pipeline_run_id), str(path)),
            )
        self.conn.commit()

    def update_run_manual_labels_path(self, *, pipeline_run_id: int, old_path: str, new_path: str) -> None:
        """
        При перемещении файла (и обновлении files.path) нужно переносить run-scoped manual метки,
        иначе они "теряются" после move (labels привязаны к path).

        Поведение:
        - если old_path отсутствует в files_manual_labels для этого прогона — ничего не делаем
        - если new_path уже есть — сливаем значения (бережно) и удаляем old_path
        - иначе просто обновляем path -> new_path
        """
        oldp = str(old_path or "")
        newp = str(new_path or "")
        if not oldp or not newp or oldp == newp:
            return
        rid = int(pipeline_run_id)
        cur = self.conn.cursor()

        cur.execute(
            "SELECT * FROM files_manual_labels WHERE pipeline_run_id = ? AND path = ? LIMIT 1",
            (rid, oldp),
        )
        old_row = cur.fetchone()
        if not old_row:
            return

        cur.execute(
            "SELECT * FROM files_manual_labels WHERE pipeline_run_id = ? AND path = ? LIMIT 1",
            (rid, newp),
        )
        new_row = cur.fetchone()

        if not new_row:
            # Simple rename
            cur.execute(
                "UPDATE files_manual_labels SET path = ? WHERE pipeline_run_id = ? AND path = ?",
                (newp, rid, oldp),
            )
            self.conn.commit()
            return

        # Merge (new wins if it has a value; otherwise take old)
        o = dict(old_row)
        n = dict(new_row)

        def _nz_str(v: Any) -> str | None:
            s = (str(v) if v is not None else "").strip()
            return s if s else None

        merged: dict[str, Any] = {}

        # faces_manual_label + at
        n_fml = _nz_str(n.get("faces_manual_label"))
        o_fml = _nz_str(o.get("faces_manual_label"))
        if n_fml is not None:
            merged["faces_manual_label"] = n_fml
            merged["faces_manual_at"] = n.get("faces_manual_at")
        else:
            merged["faces_manual_label"] = o_fml
            merged["faces_manual_at"] = o.get("faces_manual_at")

        # people_no_face
        n_pnf = int(n.get("people_no_face_manual") or 0)
        o_pnf = int(o.get("people_no_face_manual") or 0)
        merged["people_no_face_manual"] = 1 if (n_pnf or o_pnf) else 0
        merged["people_no_face_person"] = _nz_str(n.get("people_no_face_person")) or _nz_str(o.get("people_no_face_person"))

        # animals_manual
        n_am = int(n.get("animals_manual") or 0)
        o_am = int(o.get("animals_manual") or 0)
        merged["animals_manual"] = 1 if (n_am or o_am) else 0
        merged["animals_manual_kind"] = _nz_str(n.get("animals_manual_kind")) or _nz_str(o.get("animals_manual_kind"))
        merged["animals_manual_at"] = n.get("animals_manual_at") or o.get("animals_manual_at")

        # quarantine_manual
        n_qm = int(n.get("quarantine_manual") or 0)
        o_qm = int(o.get("quarantine_manual") or 0)
        merged["quarantine_manual"] = 1 if (n_qm or o_qm) else 0
        merged["quarantine_manual_at"] = n.get("quarantine_manual_at") or o.get("quarantine_manual_at")

        cur.execute(
            """
            UPDATE files_manual_labels
            SET
              faces_manual_label = ?,
              faces_manual_at = ?,
              people_no_face_manual = ?,
              people_no_face_person = ?,
              animals_manual = ?,
              animals_manual_kind = ?,
              animals_manual_at = ?,
              quarantine_manual = ?,
              quarantine_manual_at = ?
            WHERE pipeline_run_id = ? AND path = ?
            """,
            (
                merged.get("faces_manual_label"),
                merged.get("faces_manual_at"),
                int(merged.get("people_no_face_manual") or 0),
                merged.get("people_no_face_person"),
                int(merged.get("animals_manual") or 0),
                merged.get("animals_manual_kind"),
                merged.get("animals_manual_at"),
                int(merged.get("quarantine_manual") or 0),
                merged.get("quarantine_manual_at"),
                rid,
                newp,
            ),
        )
        # Drop old row (we merged it)
        cur.execute("DELETE FROM files_manual_labels WHERE pipeline_run_id = ? AND path = ?", (rid, oldp))
        self.conn.commit()

    # --- video manual frames (run-scoped) ---
    def get_video_manual_frames(self, *, pipeline_run_id: int, path: str) -> dict[int, dict[str, Any]]:
        """
        Returns mapping: frame_idx -> {frame_idx, t_sec, rects:[...], updated_at}
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT frame_idx, t_sec, rects_json, updated_at
            FROM video_manual_frames
            WHERE pipeline_run_id = ? AND path = ?
            ORDER BY frame_idx ASC
            """,
            (int(pipeline_run_id), str(path)),
        )
        out: dict[int, dict[str, Any]] = {}
        for r in cur.fetchall():
            idx = int(r["frame_idx"] or 0)
            if idx <= 0:
                continue
            rects: list[dict[str, int]] = []
            try:
                raw = r["rects_json"]
                if raw:
                    obj = json.loads(raw)
                    if isinstance(obj, list):
                        for it in obj:
                            if not isinstance(it, dict):
                                continue
                            x = int(it.get("x") or 0)
                            y = int(it.get("y") or 0)
                            w = int(it.get("w") or 0)
                            h = int(it.get("h") or 0)
                            if w > 0 and h > 0:
                                rects.append({"x": x, "y": y, "w": w, "h": h})
            except Exception:
                rects = []
            out[idx] = {
                "frame_idx": idx,
                "t_sec": (float(r["t_sec"]) if r["t_sec"] is not None else None),
                "rects": rects,
                "updated_at": str(r["updated_at"] or ""),
            }
        return out

    def upsert_video_manual_frame(
        self,
        *,
        pipeline_run_id: int,
        path: str,
        frame_idx: int,
        t_sec: float | None,
        rects: list[dict[str, int]],
    ) -> None:
        """
        Upsert one frame's manual rectangles for a video.
        """
        idx = int(frame_idx)
        if idx not in (1, 2, 3):
            raise ValueError("frame_idx must be 1..3")
        clean: list[dict[str, int]] = []
        for r in rects or []:
            if not isinstance(r, dict):
                continue
            x = int(r.get("x") or 0)
            y = int(r.get("y") or 0)
            w = int(r.get("w") or 0)
            h = int(r.get("h") or 0)
            if w > 0 and h > 0:
                clean.append({"x": x, "y": y, "w": w, "h": h})
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO video_manual_frames(pipeline_run_id, path, frame_idx, t_sec, rects_json, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(pipeline_run_id, path, frame_idx) DO UPDATE SET
              t_sec = excluded.t_sec,
              rects_json = excluded.rects_json,
              updated_at = excluded.updated_at
            """,
            (
                int(pipeline_run_id),
                str(path),
                idx,
                float(t_sec) if t_sec is not None else None,
                json.dumps(clean, ensure_ascii=False),
                _now_utc_iso(),
            ),
        )
        self.conn.commit()

    def set_run_people_no_face_manual(self, *, pipeline_run_id: int, path: str, is_people_no_face: bool, person: str | None = None) -> None:
        """
        Ручная пометка: "есть люди, но лица не найдены" В РАМКАХ ПРОГОНА.
        """
        self._ensure_run_manual_row(pipeline_run_id=int(pipeline_run_id), path=str(path))
        cur = self.conn.cursor()
        if bool(is_people_no_face):
            cur.execute(
                """
                UPDATE files_manual_labels
                SET people_no_face_manual = 1, people_no_face_person = ?
                WHERE pipeline_run_id = ? AND path = ?
                """,
                ((person or "").strip() or None, int(pipeline_run_id), str(path)),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET people_no_face_manual = 0, people_no_face_person = NULL
                WHERE pipeline_run_id = ? AND path = ?
                """,
                (int(pipeline_run_id), str(path)),
            )
        self.conn.commit()

    def set_run_animals_manual(self, *, pipeline_run_id: int, path: str, is_animal: bool, kind: str | None = None) -> None:
        """
        Ручная разметка животных (ground truth) В РАМКАХ ПРОГОНА.
        """
        self._ensure_run_manual_row(pipeline_run_id=int(pipeline_run_id), path=str(path))
        cur = self.conn.cursor()
        if bool(is_animal):
            cur.execute(
                """
                UPDATE files_manual_labels
                SET animals_manual = 1, animals_manual_kind = ?, animals_manual_at = ?
                WHERE pipeline_run_id = ? AND path = ?
                """,
                ((kind or "").strip() or None, _now_utc_iso(), int(pipeline_run_id), str(path)),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET animals_manual = 0, animals_manual_kind = NULL, animals_manual_at = NULL
                WHERE pipeline_run_id = ? AND path = ?
                """,
                (int(pipeline_run_id), str(path)),
            )
        self.conn.commit()

    def set_run_quarantine_manual(self, *, pipeline_run_id: int, path: str, is_quarantine: bool) -> None:
        """
        Ручная пометка "карантин" В РАМКАХ ПРОГОНА.
        """
        self._ensure_run_manual_row(pipeline_run_id=int(pipeline_run_id), path=str(path))
        cur = self.conn.cursor()
        if bool(is_quarantine):
            cur.execute(
                """
                UPDATE files_manual_labels
                SET quarantine_manual = 1, quarantine_manual_at = ?
                WHERE pipeline_run_id = ? AND path = ?
                """,
                (_now_utc_iso(), int(pipeline_run_id), str(path)),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET quarantine_manual = 0, quarantine_manual_at = NULL
                WHERE pipeline_run_id = ? AND path = ?
                """,
                (int(pipeline_run_id), str(path)),
            )
        self.conn.commit()

    def set_faces_auto_quarantine(self, *, path: str, is_quarantine: bool, reason: str | None = None) -> None:
        """
        Авто-карантин для локальной сортировки (экраны/технические кейсы).
        Manual labels (faces/no_faces/people_no_face) в UI имеют приоритет над этим флагом.
        """
        cur = self.conn.cursor()
        if bool(is_quarantine):
            cur.execute(
                """
                UPDATE files
                SET faces_auto_quarantine = 1, faces_quarantine_reason = ?
                WHERE path = ?
                """,
                ((reason or "").strip() or None, path),
            )
        else:
            cur.execute(
                """
                UPDATE files
                SET faces_auto_quarantine = 0, faces_quarantine_reason = NULL
                WHERE path = ?
                """,
                (path,),
            )
        self.conn.commit()

    def set_faces_auto_group(self, *, path: str, group: str | None) -> None:
        """
        Авто-группа для 2-го уровня вкладок /faces (НЕ карантин).
        group: None | '' -> сброс, иначе строка (например 'many_faces').
        """
        g = (group or "").strip().lower()
        if g == "":
            g = ""
        cur = self.conn.cursor()
        if g == "":
            cur.execute(
                """
                UPDATE files
                SET faces_auto_group = NULL
                WHERE path = ?
                """,
                (path,),
            )
        else:
            cur.execute(
                """
                UPDATE files
                SET faces_auto_group = ?
                WHERE path = ?
                """,
                (g, path),
            )
        self.conn.commit()

    def set_animals_auto(self, *, path: str, is_animal: bool, kind: str | None = None) -> None:
        """
        Авто-детект животных (MVP: кошки).
        """
        cur = self.conn.cursor()
        if bool(is_animal):
            cur.execute(
                """
                UPDATE files
                SET animals_auto = 1, animals_kind = ?
                WHERE path = ?
                """,
                ((kind or "").strip() or None, path),
            )
        else:
            cur.execute(
                """
                UPDATE files
                SET animals_auto = 0, animals_kind = NULL
                WHERE path = ?
                """,
                (path,),
            )
        self.conn.commit()

    # --- geo/time metadata + place (for UI sorting) ---
    def set_taken_at_and_gps(self, *, path: str, taken_at: str | None, gps_lat: float | None, gps_lon: float | None) -> None:
        """
        Stores best-effort taken_at (ISO string) and GPS coordinates for a file.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET taken_at = COALESCE(?, taken_at),
                gps_lat = COALESCE(?, gps_lat),
                gps_lon = COALESCE(?, gps_lon)
            WHERE path = ?
            """,
            ((taken_at or None), gps_lat, gps_lon, str(path)),
        )
        self.conn.commit()

    def set_place(self, *, path: str, country: str | None, city: str | None, source: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET place_country = ?, place_city = ?, place_source = ?, place_at = ?
            WHERE path = ?
            """,
            ((country or "").strip() or None, (city or "").strip() or None, (source or "").strip() or None, _now_utc_iso(), str(path)),
        )
        self.conn.commit()

    def get_place(self, *, path: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT place_country, place_city, place_source, place_at, gps_lat, gps_lon, taken_at
            FROM files
            WHERE path = ?
            LIMIT 1
            """,
            (str(path),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def geocode_cache_get(self, *, key: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM geocode_cache WHERE key = ? LIMIT 1", (str(key),))
        row = cur.fetchone()
        return dict(row) if row else None

    def geocode_cache_upsert(
        self,
        *,
        key: str,
        lat: float | None,
        lon: float | None,
        country: str | None,
        city: str | None,
        source: str,
        raw_json: str | None = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO geocode_cache(key, lat, lon, country, city, source, updated_at, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              lat = excluded.lat,
              lon = excluded.lon,
              country = excluded.country,
              city = excluded.city,
              source = excluded.source,
              updated_at = excluded.updated_at,
              raw_json = COALESCE(excluded.raw_json, geocode_cache.raw_json)
            """,
            (
                str(key),
                float(lat) if lat is not None else None,
                float(lon) if lon is not None else None,
                (country or "").strip() or None,
                (city or "").strip() or None,
                (source or "").strip() or None,
                _now_utc_iso(),
                raw_json,
            ),
        )
        self.conn.commit()

    def set_animals_manual(self, *, path: str, is_animal: bool, kind: str | None = None) -> None:
        """
        Ручная разметка животных (ground truth для метрик).
        ВАЖНО: не смешивать с animals_auto (который пишет автоматика).
        """
        cur = self.conn.cursor()
        if bool(is_animal):
            cur.execute(
                """
                UPDATE files
                SET animals_manual = 1, animals_manual_kind = ?, animals_manual_at = ?
                WHERE path = ?
                """,
                ((kind or "").strip() or None, _now_utc_iso(), path),
            )
        else:
            cur.execute(
                """
                UPDATE files
                SET animals_manual = 0, animals_manual_kind = NULL, animals_manual_at = NULL
                WHERE path = ?
                """,
                (path,),
            )
        self.conn.commit()

    def set_people_no_face_manual(self, *, path: str, is_people_no_face: bool, person: str | None = None) -> None:
        """
        Ручная пометка: "есть люди, но лица не найдены".
        """
        cur = self.conn.cursor()
        if bool(is_people_no_face):
            cur.execute(
                """
                UPDATE files
                SET people_no_face_manual = 1, people_no_face_person = ?
                WHERE path = ?
                """,
                ((person or "").strip() or None, path),
            )
        else:
            cur.execute(
                """
                UPDATE files
                SET people_no_face_manual = 0, people_no_face_person = NULL
                WHERE path = ?
                """,
                (path,),
            )
        self.conn.commit()

    def get_row_by_resource_id(self, *, resource_id: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM files WHERE resource_id = ? LIMIT 1", (resource_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_row_by_path(self, *, path: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        return dict(row) if row else None

    def reconcile_upsert_present_file(
        self,
        *,
        run_id: int,
        path: str,
        resource_id: str | None,
        inventory_scope: str,
        name: str | None,
        parent_path: str | None,
        size: int | None,
        created: str | None,
        modified: str | None,
        mime_type: str | None,
        media_type: str | None,
    ) -> None:
        """
        Сверка архива: файл "присутствует" в текущем скане YaDisk.

        - Если запись уже есть (по resource_id, иначе по path): обновляем метаданные.
        - Если size/modified изменились: сбрасываем hash_* и переводим status в 'new' (нужно пересчитать).
        - Если запись была 'deleted', но файл снова найден: "воскрешаем" (status='hashed' если hash есть, иначе 'new').
        - Если это новый файл: создаём запись со status='new'.
        """
        cur = self.conn.cursor()

        existing: dict[str, Any] | None = None
        if resource_id:
            existing = self.get_row_by_resource_id(resource_id=resource_id)
        if not existing:
            existing = self.get_row_by_path(path=path)

        now = _now_utc_iso()

        if not existing:
            cur.execute(
                """
                INSERT INTO files(
                  path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type,
                  hash_alg, hash_value, hash_source, status, error, scanned_at, hashed_at, last_run_id
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 'new', NULL, ?, NULL, ?)
                """,
                (path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type, now, run_id),
            )
            return

        existing_id = int(existing["id"])
        existing_path = str(existing.get("path") or "")
        existing_size = existing.get("size")
        existing_modified = str(existing.get("modified") or "") or None
        existing_hash_value = str(existing.get("hash_value") or "") or None
        existing_status = str(existing.get("status") or "") or "new"

        # Если путь изменился, но новый путь уже занят другой строкой — пометим конфликтную строку deleted.
        if existing_path != path:
            cur.execute("SELECT id, resource_id, status FROM files WHERE path = ? LIMIT 1", (path,))
            other = cur.fetchone()
            if other and int(other["id"]) != existing_id:
                cur.execute(
                    "UPDATE files SET status='deleted', error=NULL WHERE id = ?",
                    (int(other["id"]),),
                )

        size_changed = (existing_size is None) != (size is None) or (existing_size is not None and size is not None and int(existing_size) != int(size))
        mod_changed = (existing_modified is None) != (modified is None) or (existing_modified is not None and modified is not None and existing_modified != modified)
        content_changed = bool(size_changed or mod_changed)

        if content_changed:
            # Содержимое могло поменяться -> сбрасываем хэш и (на всякий случай) длительность.
            cur.execute(
                """
                UPDATE files
                SET
                  path = ?,
                  resource_id = COALESCE(?, resource_id),
                  inventory_scope = COALESCE(?, inventory_scope),
                  name = ?,
                  parent_path = ?,
                  size = ?,
                  created = ?,
                  modified = ?,
                  mime_type = ?,
                  media_type = ?,
                  status = 'new',
                  error = NULL,
                  scanned_at = ?,
                  last_run_id = ?,
                  hash_alg = NULL,
                  hash_value = NULL,
                  hash_source = NULL,
                  hashed_at = NULL,
                  duration_sec = NULL,
                  duration_source = NULL,
                  duration_at = NULL
                WHERE id = ?
                """,
                (
                    path,
                    resource_id,
                    inventory_scope,
                    name,
                    parent_path,
                    size,
                    created,
                    modified,
                    mime_type,
                    media_type,
                    now,
                    run_id,
                    existing_id,
                ),
            )
        else:
            new_status = existing_status
            if new_status == "deleted":
                new_status = "hashed" if existing_hash_value else "new"
            cur.execute(
                """
                UPDATE files
                SET
                  path = ?,
                  resource_id = COALESCE(?, resource_id),
                  inventory_scope = COALESCE(?, inventory_scope),
                  name = ?,
                  parent_path = ?,
                  size = ?,
                  created = ?,
                  modified = ?,
                  mime_type = ?,
                  media_type = ?,
                  status = ?,
                  error = NULL,
                  scanned_at = ?,
                  last_run_id = ?
                WHERE id = ?
                """,
                (
                    path,
                    resource_id,
                    inventory_scope,
                    name,
                    parent_path,
                    size,
                    created,
                    modified,
                    mime_type,
                    media_type,
                    new_status,
                    now,
                    run_id,
                    existing_id,
                ),
            )
        # Коммитим батчами снаружи (через update_run_progress/finish_run), чтобы сверка работала быстрее.

    def get_duration(self, *, path: str) -> int | None:
        cur = self.conn.cursor()
        cur.execute("SELECT duration_sec FROM files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if not row:
            return None
        val = row[0]
        return int(val) if isinstance(val, (int, float)) else None

    def set_duration(self, *, path: str, duration_sec: int | None, source: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET duration_sec = ?, duration_source = ?, duration_at = ?
            WHERE path = ?
            """,
            (duration_sec, source, _now_utc_iso(), path),
        )
        self.conn.commit()


class PipelineStore:
    """
    Хранилище прогонов единого конвейера сортировки (resume после рестарта).

    Важно: запись в pipeline_runs — это "текущая правда" для UI, поэтому держим log_tail
    и updated_at в БД, а не только в памяти процесса.
    """

    def __init__(self) -> None:
        init_db()
        self.conn = get_connection()

    def close(self) -> None:
        self.conn.close()

    def create_run(self, *, kind: str, root_path: str, apply: bool, skip_dedup: bool) -> int:
        cur = self.conn.cursor()
        now = _now_utc_iso()
        cur.execute(
            """
            INSERT INTO pipeline_runs(kind, root_path, status, step_num, step_total, step_title,
                                      apply, skip_dedup, started_at, updated_at, log_tail)
            VALUES(?, ?, 'running', 0, 0, NULL, ?, ?, ?, ?, '')
            """,
            (str(kind), str(root_path), 1 if apply else 0, 1 if skip_dedup else 0, now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_run_by_id(self, *, run_id: int) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM pipeline_runs WHERE id = ? LIMIT 1", (int(run_id),))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_run(self, *, kind: str, root_path: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM pipeline_runs
            WHERE kind = ? AND root_path = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (str(kind), str(root_path)),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_any(self, *, kind: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM pipeline_runs
            WHERE kind = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (str(kind),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_run(
        self,
        *,
        run_id: int,
        status: str | None = None,
        step_num: int | None = None,
        step_total: int | None = None,
        step_title: str | None = None,
        dedup_run_id: int | None = None,
        face_run_id: int | None = None,
        pid: int | None = None,
        last_src_path: str | None = None,
        last_dst_path: str | None = None,
        last_error: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []

        def _set(key: str, val: Any) -> None:
            fields.append(f"{key} = ?")
            params.append(val)

        if status is not None:
            _set("status", str(status))
        if step_num is not None:
            _set("step_num", int(step_num))
        if step_total is not None:
            _set("step_total", int(step_total))
        if step_title is not None:
            _set("step_title", str(step_title))
        if dedup_run_id is not None:
            _set("dedup_run_id", int(dedup_run_id))
        if face_run_id is not None:
            _set("face_run_id", int(face_run_id))
        if pid is not None:
            _set("pid", int(pid))
        if last_src_path is not None:
            _set("last_src_path", str(last_src_path))
        if last_dst_path is not None:
            _set("last_dst_path", str(last_dst_path))
        if last_error is not None:
            _set("last_error", str(last_error))
        if finished_at is not None:
            _set("finished_at", str(finished_at))

        # updated_at всегда двигаем при любом update
        _set("updated_at", _now_utc_iso())

        if not fields:
            return
        params.append(int(run_id))
        cur = self.conn.cursor()
        cur.execute(f"UPDATE pipeline_runs SET {', '.join(fields)} WHERE id = ?", params)
        self.conn.commit()

    def upsert_metrics(self, *, pipeline_run_id: int, metrics: dict[str, Any]) -> None:
        """
        Upsert aggregated metrics snapshot for a pipeline run.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO pipeline_run_metrics(
              pipeline_run_id, computed_at, face_run_id,
              step0_checked, step0_non_media, step0_broken_media,
              step2_total, step2_processed,
              cats_total, cats_mism,
              faces_total, faces_mism,
              no_faces_total, no_faces_mism
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pipeline_run_id) DO UPDATE SET
              computed_at = excluded.computed_at,
              face_run_id = excluded.face_run_id,
              step0_checked = excluded.step0_checked,
              step0_non_media = excluded.step0_non_media,
              step0_broken_media = excluded.step0_broken_media,
              step2_total = excluded.step2_total,
              step2_processed = excluded.step2_processed,
              cats_total = excluded.cats_total,
              cats_mism = excluded.cats_mism,
              faces_total = excluded.faces_total,
              faces_mism = excluded.faces_mism,
              no_faces_total = excluded.no_faces_total,
              no_faces_mism = excluded.no_faces_mism
            """,
            (
                int(pipeline_run_id),
                str(metrics.get("computed_at") or _now_utc_iso()),
                metrics.get("face_run_id"),
                metrics.get("step0_checked"),
                metrics.get("step0_non_media"),
                metrics.get("step0_broken_media"),
                metrics.get("step2_total"),
                metrics.get("step2_processed"),
                metrics.get("cats_total"),
                metrics.get("cats_mism"),
                metrics.get("faces_total"),
                metrics.get("faces_mism"),
                metrics.get("no_faces_total"),
                metrics.get("no_faces_mism"),
            ),
        )
        self.conn.commit()

    def get_metrics_for_run(self, *, pipeline_run_id: int) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM pipeline_run_metrics WHERE pipeline_run_id = ? LIMIT 1", (int(pipeline_run_id),))
        row = cur.fetchone()
        return dict(row) if row else None

    def append_log(self, *, run_id: int, line: str, max_chars: int = 120_000) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT log_tail FROM pipeline_runs WHERE id = ? LIMIT 1", (int(run_id),))
        row = cur.fetchone()
        s = (row["log_tail"] if row and row["log_tail"] is not None else "") + (line or "")
        s = s.replace("\r\n", "\n")
        if len(s) > max_chars:
            s = s[-max_chars:]
        cur.execute("UPDATE pipeline_runs SET log_tail = ?, updated_at = ? WHERE id = ?", (s, _now_utc_iso(), int(run_id)))
        self.conn.commit()

    def upsert_file(
        self,
        *,
        run_id: int,
        path: str,
        resource_id: str | None = None,
        inventory_scope: str | None = None,
        name: str | None,
        parent_path: str | None,
        size: int | None,
        created: str | None,
        modified: str | None,
        mime_type: str | None,
        media_type: str | None,
        hash_alg: str | None,
        hash_value: str | None,
        hash_source: str | None,
        status: str,
        error: str | None,
        scanned_at: str | None = None,
        hashed_at: str | None = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO files(
              path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type,
              hash_alg, hash_value, hash_source, status, error, scanned_at, hashed_at, last_run_id
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              resource_id = COALESCE(excluded.resource_id, files.resource_id),
              inventory_scope = COALESCE(excluded.inventory_scope, files.inventory_scope),
              name = excluded.name,
              parent_path = excluded.parent_path,
              size = excluded.size,
              created = excluded.created,
              modified = excluded.modified,
              mime_type = excluded.mime_type,
              media_type = excluded.media_type,
              hash_alg = COALESCE(excluded.hash_alg, files.hash_alg),
              hash_value = COALESCE(excluded.hash_value, files.hash_value),
              hash_source = COALESCE(excluded.hash_source, files.hash_source),
              status = excluded.status,
              error = excluded.error,
              scanned_at = excluded.scanned_at,
              hashed_at = COALESCE(excluded.hashed_at, files.hashed_at),
              last_run_id = excluded.last_run_id
            """,
            (
                path,
                resource_id,
                inventory_scope,
                name,
                parent_path,
                size,
                created,
                modified,
                mime_type,
                media_type,
                hash_alg,
                hash_value,
                hash_source,
                status,
                error,
                scanned_at or _now_utc_iso(),
                hashed_at,
                run_id,
            ),
        )
        self.conn.commit()

    def get_existing_hash(self, *, path: str) -> tuple[str | None, str | None]:
        cur = self.conn.cursor()
        cur.execute("SELECT hash_alg, hash_value FROM files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if not row:
            return None, None
        return (row["hash_alg"], row["hash_value"])

    def list_dup_groups_archive(self) -> list[dict[str, Any]]:
        """
        Возвращает группы дублей для архива YaDisk (inventory_scope='archive').
        Группа = одинаковый (hash_alg, hash_value), где количество файлов > 1.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
              hash_alg,
              hash_value,
              COUNT(*) AS cnt
            FROM files
            WHERE
              hash_value IS NOT NULL
              AND COALESCE(inventory_scope, 'archive') = 'archive'
              AND status != 'deleted'
            GROUP BY hash_alg, hash_value
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC, hash_alg ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def list_group_items(self, *, hash_alg: str, hash_value: str, inventory_scope: str | None = None, last_run_id: int | None = None) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        where: list[str] = ["hash_alg = ? AND hash_value = ?", "status != 'deleted'"]
        params: list[Any] = [hash_alg, hash_value]
        if inventory_scope is not None:
            where.append("inventory_scope = ?")
            params.append(inventory_scope)
        if last_run_id is not None:
            where.append("last_run_id = ?")
            params.append(last_run_id)
        cur.execute(
            f"""
            SELECT path, name, parent_path, size, modified, mime_type, media_type, hash_source
            FROM files
            WHERE {' AND '.join(where)}
            ORDER BY path ASC
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]

    def set_ignore_archive_dup(self, *, paths: list[str], run_id: int) -> int:
        """
        Помечает source-файлы как "не дубль архива" ДЛЯ ТЕКУЩЕГО source run_id.
        При новом пересканировании source run_id изменится — пометка автоматически перестанет действовать.
        """
        if not paths:
            return 0
        cur = self.conn.cursor()
        q = ",".join(["?"] * len(paths))
        cur.execute(
            f"""
            UPDATE files
            SET ignore_archive_dup_run_id = ?
            WHERE path IN ({q})
            """,
            [run_id, *paths],
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def list_source_dups_in_archive(
        self,
        *,
        source_run_id: int,
        archive_prefixes: tuple[str, ...] = ("disk:/Фото",),
    ) -> list[dict[str, Any]]:
        """
        Возвращает пары (source file -> archive matches) по совпадающему хэшу.
        Источник ограничиваем last_run_id=source_run_id, чтобы не смешивать разные выборы папки.
        """
        cur = self.conn.cursor()
        prefixes = [str(p or "").rstrip("/") + "/%" for p in (archive_prefixes or ()) if str(p or "").strip()]
        if not prefixes:
            return []
        like_sql = " OR ".join(["a.path LIKE ?"] * len(prefixes))
        cur.execute(
            f"""
            SELECT
              s.path AS source_path,
              s.name AS source_name,
              s.size AS source_size,
              s.mime_type AS source_mime_type,
              s.media_type AS source_media_type,
              s.hash_alg AS hash_alg,
              s.hash_value AS hash_value,
              a.path AS archive_path,
              a.name AS archive_name,
              a.size AS archive_size,
              a.mime_type AS archive_mime_type,
              a.media_type AS archive_media_type
            FROM files s
            JOIN files a
              ON a.hash_alg = s.hash_alg AND a.hash_value = s.hash_value
            WHERE
              s.inventory_scope = 'source'
              AND s.last_run_id = ?
              AND s.status != 'deleted'
              AND s.hash_value IS NOT NULL
              AND COALESCE(s.ignore_archive_dup_run_id, -1) != ?
              AND COALESCE(a.inventory_scope, 'archive') = 'archive'
              AND a.status != 'deleted'
              AND a.hash_value IS NOT NULL
              AND ({like_sql})
            ORDER BY s.path ASC, a.path ASC
            """,
            (source_run_id, source_run_id, *prefixes),
        )
        return [dict(r) for r in cur.fetchall()]

    def list_dup_groups_for_run(self, *, inventory_scope: str, run_id: int) -> list[dict[str, Any]]:
        """
        Группы дублей ВНУТРИ конкретного прогона (run_id) выбранного scope.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
              hash_alg,
              hash_value,
              COUNT(*) AS cnt
            FROM files
            WHERE
              hash_value IS NOT NULL
              AND inventory_scope = ?
              AND last_run_id = ?
              AND status != 'deleted'
            GROUP BY hash_alg, hash_value
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC, hash_alg ASC
            """,
            (inventory_scope, run_id),
        )
        return [dict(r) for r in cur.fetchall()]

    def path_exists(self, *, path: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM files WHERE path = ? LIMIT 1", (path,))
        return cur.fetchone() is not None

    def mark_deleted(self, *, paths: list[str]) -> int:
        """
        Помечает файлы как удалённые (в корзину/удалены на стороне YaDisk), чтобы они
        исчезали из /duplicates без перескана.
        """
        if not paths:
            return 0
        cur = self.conn.cursor()
        q = ",".join(["?"] * len(paths))
        cur.execute(f"UPDATE files SET status = 'deleted', error = NULL WHERE path IN ({q})", paths)
        self.conn.commit()
        return int(cur.rowcount or 0)

    def update_path(self, *, old_path: str, new_path: str, new_name: str | None, new_parent_path: str | None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET path = ?, name = COALESCE(?, name), parent_path = COALESCE(?, parent_path)
            WHERE path = ?
            """,
            (new_path, new_name, new_parent_path, old_path),
        )
        self.conn.commit()

    def set_faces_summary(self, *, path: str, faces_run_id: int, faces_count: int, faces_scanned_at: str | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET faces_count = ?, faces_run_id = ?, faces_scanned_at = ?
            WHERE path = ?
            """,
            (int(faces_count), int(faces_run_id), faces_scanned_at or _now_utc_iso(), path),
        )
        self.conn.commit()

    def get_row_by_resource_id(self, *, resource_id: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM files WHERE resource_id = ? LIMIT 1", (resource_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_row_by_path(self, *, path: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        return dict(row) if row else None

    def reconcile_upsert_present_file(
        self,
        *,
        run_id: int,
        path: str,
        resource_id: str | None,
        inventory_scope: str,
        name: str | None,
        parent_path: str | None,
        size: int | None,
        created: str | None,
        modified: str | None,
        mime_type: str | None,
        media_type: str | None,
    ) -> None:
        """
        Сверка архива: файл "присутствует" в текущем скане YaDisk.

        - Если запись уже есть (по resource_id, иначе по path): обновляем метаданные.
        - Если size/modified изменились: сбрасываем hash_* и переводим status в 'new' (нужно пересчитать).
        - Если запись была 'deleted', но файл снова найден: "воскрешаем" (status='hashed' если hash есть, иначе 'new').
        - Если это новый файл: создаём запись со status='new'.
        """
        cur = self.conn.cursor()

        existing: dict[str, Any] | None = None
        if resource_id:
            existing = self.get_row_by_resource_id(resource_id=resource_id)
        if not existing:
            existing = self.get_row_by_path(path=path)

        now = _now_utc_iso()

        if not existing:
            cur.execute(
                """
                INSERT INTO files(
                  path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type,
                  hash_alg, hash_value, hash_source, status, error, scanned_at, hashed_at, last_run_id
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 'new', NULL, ?, NULL, ?)
                """,
                (path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type, now, run_id),
            )
            return

        existing_id = int(existing["id"])
        existing_path = str(existing.get("path") or "")
        existing_size = existing.get("size")
        existing_modified = str(existing.get("modified") or "") or None
        existing_hash_value = str(existing.get("hash_value") or "") or None
        existing_status = str(existing.get("status") or "") or "new"

        # Если путь изменился, но новый путь уже занят другой строкой — пометим конфликтную строку deleted.
        if existing_path != path:
            cur.execute("SELECT id, resource_id, status FROM files WHERE path = ? LIMIT 1", (path,))
            other = cur.fetchone()
            if other and int(other["id"]) != existing_id:
                cur.execute(
                    "UPDATE files SET status='deleted', error=NULL WHERE id = ?",
                    (int(other["id"]),),
                )

        size_changed = (existing_size is None) != (size is None) or (existing_size is not None and size is not None and int(existing_size) != int(size))
        mod_changed = (existing_modified is None) != (modified is None) or (existing_modified is not None and modified is not None and existing_modified != modified)
        content_changed = bool(size_changed or mod_changed)

        if content_changed:
            # Содержимое могло поменяться -> сбрасываем хэш и (на всякий случай) длительность.
            cur.execute(
                """
                UPDATE files
                SET
                  path = ?,
                  resource_id = COALESCE(?, resource_id),
                  inventory_scope = COALESCE(?, inventory_scope),
                  name = ?,
                  parent_path = ?,
                  size = ?,
                  created = ?,
                  modified = ?,
                  mime_type = ?,
                  media_type = ?,
                  status = 'new',
                  error = NULL,
                  scanned_at = ?,
                  last_run_id = ?,
                  hash_alg = NULL,
                  hash_value = NULL,
                  hash_source = NULL,
                  hashed_at = NULL,
                  duration_sec = NULL,
                  duration_source = NULL,
                  duration_at = NULL
                WHERE id = ?
                """,
                (
                    path,
                    resource_id,
                    inventory_scope,
                    name,
                    parent_path,
                    size,
                    created,
                    modified,
                    mime_type,
                    media_type,
                    now,
                    run_id,
                    existing_id,
                ),
            )
        else:
            new_status = existing_status
            if new_status == "deleted":
                new_status = "hashed" if existing_hash_value else "new"
            cur.execute(
                """
                UPDATE files
                SET
                  path = ?,
                  resource_id = COALESCE(?, resource_id),
                  inventory_scope = COALESCE(?, inventory_scope),
                  name = ?,
                  parent_path = ?,
                  size = ?,
                  created = ?,
                  modified = ?,
                  mime_type = ?,
                  media_type = ?,
                  status = ?,
                  error = NULL,
                  scanned_at = ?,
                  last_run_id = ?
                WHERE id = ?
                """,
                (
                    path,
                    resource_id,
                    inventory_scope,
                    name,
                    parent_path,
                    size,
                    created,
                    modified,
                    mime_type,
                    media_type,
                    new_status,
                    now,
                    run_id,
                    existing_id,
                ),
            )
        # Коммитим батчами снаружи (через update_run_progress/finish_run), чтобы сверка работала быстрее.

    def get_duration(self, *, path: str) -> int | None:
        cur = self.conn.cursor()
        cur.execute("SELECT duration_sec FROM files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if not row:
            return None
        val = row[0]
        return int(val) if isinstance(val, (int, float)) else None

    def set_duration(self, *, path: str, duration_sec: int | None, source: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE files
            SET duration_sec = ?, duration_source = ?, duration_at = ?
            WHERE path = ?
            """,
            (duration_sec, source, _now_utc_iso(), path),
        )
        self.conn.commit()


if __name__ == "__main__":
    init_db()
    print("DB initialized:", DB_PATH)
