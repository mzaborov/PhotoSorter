import os
import sqlite3
import json
import threading
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
        # Включаем поддержку FOREIGN KEY constraints
        conn.execute("PRAGMA foreign_keys = ON")
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


def _get_file_id_from_path(conn: sqlite3.Connection, file_path: str) -> int | None:
    """
    Получает file_id из таблицы files по file_path.
    Возвращает None, если файл не найден.
    """
    cur = conn.cursor()
    cur.execute("SELECT id FROM files WHERE path = ? LIMIT 1", (file_path,))
    row = cur.fetchone()
    return row[0] if row else None


def _get_file_id(conn: sqlite3.Connection, *, file_id: int | None = None, file_path: str | None = None) -> int | None:
    """
    Универсальная функция для получения file_id.
    Приоритет: file_id (если передан), иначе file_path (если передан).
    Возвращает None, если ни file_id, ни file_path не передан, или файл не найден.
    """
    if file_id is not None:
        return int(file_id)
    if file_path is not None:
        return _get_file_id_from_path(conn, file_path)
    return None


# Глобальный флаг для отслеживания инициализации БД
_db_initialized = False
_db_init_lock = threading.Lock()

def init_db():
    """
    Инициализирует БД (создает таблицы, если их нет).
    Идемпотентная функция - безопасно вызывать многократно.
    """
    global _db_initialized
    
    # Быстрая проверка без блокировки (double-checked locking pattern)
    if _db_initialized:
        return
    
    with _db_init_lock:
        # Повторная проверка после получения блокировки
        if _db_initialized:
            return
        
        # на всякий случай создаём папку data
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = get_connection()
        cur = conn.cursor()
        
        # Быстрая проверка: существует ли уже таблица files (основная таблица)?
        # Если да, значит БД уже инициализирована, пропускаем создание таблиц
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
            if cur.fetchone():
                # БД уже инициализирована, только проверяем миграции (если нужно)
                # Миграции выполняются ниже, но без создания всех таблиц
                conn.close()
                _db_initialized = True
                return
        except Exception:
            # Если ошибка при проверке, продолжаем инициализацию
            pass

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

        # Создаем таблицу folders с обработкой возможных блокировок при параллельных запросах
        try:
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
        except sqlite3.OperationalError as e:
            # Игнорируем ошибки "table already exists" и блокировки при параллельных запросах
            # CREATE TABLE IF NOT EXISTS должен быть безопасным, но на всякий случай обрабатываем
            if "already exists" not in str(e).lower() and "locked" not in str(e).lower():
                raise

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
            # Локальная сортировка: дополнительные категории
            # ПРИМЕЧАНИЕ: Ручные метки (*_manual_*) удалены из files (приведение к 3NF, ЭТАП 1.9).
            # Метки хранятся только в files_manual_labels (run-scoped).
            # Авто-карантин (экраны/технические фото/сомнительные кейсы)
            "faces_auto_quarantine": "faces_auto_quarantine INTEGER NOT NULL DEFAULT 0",
            "faces_quarantine_reason": "faces_quarantine_reason TEXT",
            # Авто-группы (2-й уровень вкладок /faces), НЕ связанные с карантином.
            # Пример: 'many_faces' (>=8 лиц).
            "faces_auto_group": "faces_auto_group TEXT",
            # Авто-детект животных (MVP: кошки)
            "animals_auto": "animals_auto INTEGER NOT NULL DEFAULT 0",
            "animals_kind": "animals_kind TEXT",
            # Люди, но лица не найдены (пока в основном manual)
            # ПРИМЕЧАНИЕ: people_no_face_manual удален из files (приведение к 3NF, ЭТАП 1.9).
            # Метки хранятся только в files_manual_labels (run-scoped).
            # 
            # ВАЖНО: people_no_face_person в files - это ДЛЯ АРХИВНЫХ ФАЙЛОВ.
            # Это глобальная привязка персоны к файлу, которая сохраняется после переезда в архив.
            # При переезде файла в архив нужно копировать значение из files_manual_labels.people_no_face_person
            # в files.people_no_face_person, чтобы привязка сохранилась.
            # 
            # TODO: Реализовать копирование people_no_face_person при переезде в архив
            # (см. migrate_archive_faces.py или API для перемещения в архив).
            "people_no_face_person": "people_no_face_person TEXT",  # ТОЛЬКО для архивных файлов

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
            "exif_orientation": "exif_orientation INTEGER",  # EXIF Orientation (1-8), 1 = normal
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
            file_id               INTEGER NOT NULL,
            faces_manual_label    TEXT,                      -- 'faces'|'no_faces'|NULL
            faces_manual_at       TEXT,
            people_no_face_manual INTEGER NOT NULL DEFAULT 0,
            people_no_face_person TEXT,
            animals_manual        INTEGER NOT NULL DEFAULT 0,
            animals_manual_kind   TEXT,
            animals_manual_at     TEXT,
            quarantine_manual     INTEGER NOT NULL DEFAULT 0,
            quarantine_manual_at  TEXT,
            PRIMARY KEY (pipeline_run_id, file_id),
            FOREIGN KEY (file_id) REFERENCES files(id)
        );
        """
    )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_manual_labels_run ON files_manual_labels(pipeline_run_id);")
        
        # Миграция file_path → file_id: добавляем колонку file_id (пока NULL)
        _ensure_columns(
            conn,
            "files_manual_labels",
        {
            "file_id": "file_id INTEGER",  # FOREIGN KEY на files.id, пока NULL (заполним миграцией)
        },
    )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_manual_labels_file_id ON files_manual_labels(file_id);")

        # --- Run-scoped manual rectangles for VIDEO frames (3 frames per video) ---
        # ВАЖНО: для видео нам нужны прямоугольники с привязкой к кадру/таймкоду.
        # Храним отдельной таблицей, чтобы не смешивать с photo_rectangles (она про фото/одно изображение).
        cur.execute(
        """
        CREATE TABLE IF NOT EXISTS video_manual_frames (
            pipeline_run_id   INTEGER NOT NULL,
            file_id           INTEGER NOT NULL,
            frame_idx         INTEGER NOT NULL, -- 1..3
            t_sec             REAL,             -- таймкод кадра (секунды), best-effort
            rects_json        TEXT,             -- JSON: [{"x":..,"y":..,"w":..,"h":..}, ...]
            updated_at        TEXT NOT NULL,
            PRIMARY KEY (pipeline_run_id, file_id, frame_idx),
            FOREIGN KEY (file_id) REFERENCES files(id)
        );
        """
    )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_video_manual_frames_file_id ON video_manual_frames(file_id);")
        
        # Миграция file_path → file_id: добавляем колонку file_id (пока NULL)
        _ensure_columns(
            conn,
            "video_manual_frames",
        {
            "file_id": "file_id INTEGER",  # FOREIGN KEY на files.id, пока NULL (заполним миграцией)
        },
    )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_video_manual_frames_file_id ON video_manual_frames(file_id);")

        # Индексы для быстрых группировок дублей.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash_alg, hash_value);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent_path);")
        # Устойчивый идентификатор: уникален, если известен (partial unique index).
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_files_resource_id ON files(resource_id) WHERE resource_id IS NOT NULL;"
        )

        conn.commit()
        conn.close()
        
        # Помечаем БД как инициализированную
        _db_initialized = True


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
        self.conn = get_connection()

    def close(self) -> None:
        self.conn.close()

    def _ensure_face_schema(self) -> None:
        cur = self.conn.cursor()

        # Миграция: раньше таблица называлась face_detections, теперь photo_rectangles.
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('face_detections','face_rectangles','photo_rectangles')")
        existing = {r[0] for r in cur.fetchall()}
        if "face_detections" in existing and "photo_rectangles" not in existing and "face_rectangles" not in existing:
            cur.execute("ALTER TABLE face_detections RENAME TO photo_rectangles;")
            # Старые индексы могли называться idx_face_det_*. Их можно оставить,
            # но создаём новые с актуальными именами для читаемости.
            try:
                cur.execute("DROP INDEX IF EXISTS idx_face_det_run;")
                cur.execute("DROP INDEX IF EXISTS idx_face_det_file;")
            except Exception:
                pass
        elif "face_rectangles" in existing and "photo_rectangles" not in existing:
            # Миграция: face_rectangles → photo_rectangles
            # Добавляем is_face если его нет
            cur.execute("PRAGMA table_info(face_rectangles)")
            columns = {row[1] for row in cur.fetchall()}
            if "is_face" not in columns:
                cur.execute("ALTER TABLE face_rectangles ADD COLUMN is_face INTEGER DEFAULT 1")
                cur.execute("UPDATE face_rectangles SET is_face = 1 WHERE is_face IS NULL")
            # Переименовываем таблицу
            cur.execute("ALTER TABLE face_rectangles RENAME TO photo_rectangles;")

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
            CREATE TABLE IF NOT EXISTS photo_rectangles (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id         INTEGER,
              file_id        INTEGER NOT NULL,    -- FOREIGN KEY на files.id
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
              created_at     TEXT NOT NULL,
              is_face        INTEGER NOT NULL DEFAULT 1,  -- 1=лицо, 0=персона
              FOREIGN KEY (file_id) REFERENCES files(id)
            );
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_photo_rect_run ON photo_rectangles(run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photo_rect_file ON photo_rectangles(file_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photo_rect_file_id ON photo_rectangles(file_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photo_rect_is_face ON photo_rectangles(is_face);")

        # Ручная разметка (UI шага 2: корректировка лиц/нет лиц)
        _ensure_columns(
            self.conn,
            "photo_rectangles",
            {
                "is_manual": "is_manual INTEGER NOT NULL DEFAULT 0",
                "manual_created_at": "manual_created_at TEXT",
            },
        )
        
        # Face recognition embeddings (векторное представление лица для распознавания)
        _ensure_columns(
            self.conn,
            "photo_rectangles",
            {
                "embedding": "embedding BLOB",  # JSON массив float32 (обычно 512 или 1024 элементов)
            },
        )
        
        # Archive scope для поддержки архива без привязки к прогонам
        # NULL или '' = для прогонов (сортируемые папки), 'archive' = для архива (текущий статус)
        _ensure_columns(
            self.conn,
            "photo_rectangles",
            {
                "archive_scope": "archive_scope TEXT",  # NULL|'' для прогонов, 'archive' для архива
            },
        )
        
        # Кластер: принадлежность прямоугольника кластеру (вместо таблицы face_cluster_members)
        _ensure_columns(
            self.conn,
            "photo_rectangles",
            {
                "cluster_id": "cluster_id INTEGER REFERENCES face_clusters(id)",  # NULL = не в кластере
            },
        )
        # Ручная привязка к персоне (вместо person_rectangle_manual_assignments; CHECK: только один из cluster_id/manual_person_id)
        _ensure_columns(
            self.conn,
            "photo_rectangles",
            {
                "manual_person_id": "manual_person_id INTEGER REFERENCES persons(id)",  # NULL = не назначено вручную
            },
        )
        
        # Убеждаемся, что is_face имеет значение для всех записей
        cur.execute("UPDATE photo_rectangles SET is_face = 1 WHERE is_face IS NULL")

        # Справочник персон (людей)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT UNIQUE NOT NULL,
                mode            TEXT NOT NULL DEFAULT 'active',  -- 'active'|'deferred'|'never'
                is_me           INTEGER NOT NULL DEFAULT 0,
                kinship         TEXT,                          -- степень родства/близости
                avatar_face_id  INTEGER,                       -- FK к photo_rectangles.id
                created_at       TEXT NOT NULL,
                updated_at       TEXT
            );
        """)
        
        # Миграция: добавление полей для групп персон
        # ВАЖНО: "group" - зарезервированное слово в SQL, используем кавычки
        _ensure_columns(
            self.conn,
            "persons",
            {
                "group": '"group" TEXT',  # Название группы персоны
                "group_order": "group_order INTEGER",  # Порядок группы для сортировки
            },
        )

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

        # Миграция: переименование face_person_manual_assignments → person_rectangle_manual_assignments
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('face_person_manual_assignments','person_rectangle_manual_assignments')")
        existing_assignments = {r[0] for r in cur.fetchall()}
        if "face_person_manual_assignments" in existing_assignments and "person_rectangle_manual_assignments" not in existing_assignments:
            # Проверяем, есть ли колонка rectangle_id
            cur.execute("PRAGMA table_info(face_person_manual_assignments)")
            columns = {row[1] for row in cur.fetchall()}
            if "rectangle_id" not in columns:
                # Пересоздаем таблицу с новым именем колонки
                cur.execute("""
                    CREATE TABLE person_rectangle_manual_assignments_new (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        rectangle_id        INTEGER NOT NULL,
                        person_id           INTEGER NOT NULL,
                        source              TEXT NOT NULL,
                        confidence          REAL,
                        created_at          TEXT NOT NULL
                    )
                """)
                cur.execute("""
                    INSERT INTO person_rectangle_manual_assignments_new (
                        id, rectangle_id, person_id, source, confidence, created_at
                    )
                    SELECT 
                        id, face_rectangle_id, person_id, source, confidence, created_at
                    FROM face_person_manual_assignments
                """)
                cur.execute("DROP TABLE face_person_manual_assignments")
                cur.execute("ALTER TABLE person_rectangle_manual_assignments_new RENAME TO person_rectangle_manual_assignments")
            else:
                # Просто переименовываем таблицу
                cur.execute("ALTER TABLE face_person_manual_assignments RENAME TO person_rectangle_manual_assignments")

        # Связь лиц с кластерами: хранится в photo_rectangles.cluster_id (таблица face_cluster_members удалена миграцией migrate_face_cluster_members_to_photo_rectangles.py).

        # Ручные привязки перенесены в photo_rectangles.manual_person_id (миграция migrate_person_rectangle_manual_to_photo_rectangles.py).
        # Таблица person_rectangle_manual_assignments больше не создаётся.

        # Archive scope для кластеров (аналогично photo_rectangles)
        _ensure_columns(
            self.conn,
            "face_clusters",
            {
                "archive_scope": "archive_scope TEXT",  # NULL|'' для прогонов, 'archive' для архива
            },
        )
        
        # Индексы
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_run ON face_clusters(run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_person ON face_clusters(person_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photo_rect_archive_scope ON photo_rectangles(archive_scope);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_clusters_archive_scope ON face_clusters(archive_scope);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photo_rect_cluster_id ON photo_rectangles(cluster_id);")
        # Старые индексы для face_labels (для обратной совместимости, будут удалены после миграции)
        # ВАЖНО: таблица face_labels больше не используется в рабочем коде, индексы оставлены только для совместимости
        # После полного тестирования можно удалить таблицу через migrate_drop_face_labels_table.py
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_labels_face ON face_labels(face_rectangle_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_labels_person ON face_labels(person_id);")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_face_labels_unique ON face_labels(face_rectangle_id, person_id);")

        # Таблица person_rectangles удалена миграцией (migrate_drop_person_rectangles.py).

        # Простая привязка файла к персоне (без прямоугольника)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_persons (
                pipeline_run_id     INTEGER NOT NULL,
                file_id             INTEGER NOT NULL,
                person_id           INTEGER NOT NULL,
                created_at          TEXT NOT NULL,
                PRIMARY KEY (pipeline_run_id, file_id, person_id),
                FOREIGN KEY (file_id) REFERENCES files(id),
                FOREIGN KEY (person_id) REFERENCES persons(id)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_persons_run ON file_persons(pipeline_run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_persons_file ON file_persons(file_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_persons_person ON file_persons(person_id);")
        
        # Миграция file_path → file_id: добавляем колонку file_id (пока NULL)
        _ensure_columns(
            self.conn,
            "file_persons",
            {
                "file_id": "file_id INTEGER",  # FOREIGN KEY на files.id, пока NULL (заполним миграцией)
            },
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_persons_file_id ON file_persons(file_id);")

        # Группы для файлов "Нет людей" (иерархические: Поездки/2023 Турция, Мемы и т.д.)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_groups (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_run_id     INTEGER NOT NULL,
                file_id             INTEGER NOT NULL,
                group_path          TEXT NOT NULL,              -- например, "Поездки/2023 Турция"
                created_at          TEXT NOT NULL,
                UNIQUE(pipeline_run_id, file_id, group_path),
                FOREIGN KEY (file_id) REFERENCES files(id)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_groups_run ON file_groups(pipeline_run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_groups_file ON file_groups(file_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_groups_path ON file_groups(group_path);")
        
        # Миграция file_path → file_id: добавляем колонку file_id (пока NULL)
        _ensure_columns(
            self.conn,
            "file_groups",
            {
                "file_id": "file_id INTEGER",  # FOREIGN KEY на files.id, пока NULL (заполним миграцией)
            },
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_groups_file_id ON file_groups(file_id);")

        # Привязка персон к файлам в группах "Артефакты людей"
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_group_persons (
                pipeline_run_id     INTEGER NOT NULL,
                file_id             INTEGER NOT NULL,
                group_path          TEXT NOT NULL,
                person_id           INTEGER NOT NULL,
                created_at          TEXT NOT NULL,
                PRIMARY KEY (pipeline_run_id, file_id, group_path, person_id),
                FOREIGN KEY (file_id) REFERENCES files(id),
                FOREIGN KEY (person_id) REFERENCES persons(id)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_group_persons_run ON file_group_persons(pipeline_run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_group_persons_file ON file_group_persons(file_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_group_persons_person ON file_group_persons(person_id);")
        
        # Миграция file_path → file_id: добавляем колонку file_id (пока NULL)
        _ensure_columns(
            self.conn,
            "file_group_persons",
            {
                "file_id": "file_id INTEGER",  # FOREIGN KEY на files.id, пока NULL (заполним миграцией)
            },
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_group_persons_file_id ON file_group_persons(file_id);")

        self.conn.commit()

    def list_rectangles(self, *, run_id: int, file_id: int | None = None, file_path: str | None = None) -> list[dict[str, Any]]:
        """
        Возвращает список rectangles для файла.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        # Получаем file_id
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        
        cur = self.conn.cursor()
        # Используем file_id для запроса (приоритет над file_path)
        # Фильтруем rectangles с ignore_flag = 1 (помеченные как "нет людей" или "игнорировать")
        cur.execute(
            """
            SELECT
              fr.id, fr.run_id, f.path AS file_path, fr.face_index,
              fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
              fr.confidence, fr.presence_score,
              fr.manual_person, fr.ignore_flag,
              fr.created_at,
              COALESCE(fr.is_manual, 0) AS is_manual,
              fr.manual_created_at,
              COALESCE(fr.is_face, 1) AS is_face
            FROM photo_rectangles fr
            JOIN files f ON f.id = fr.file_id
            WHERE fr.run_id = ? AND fr.file_id = ? AND COALESCE(fr.ignore_flag, 0) = 0
            ORDER BY COALESCE(fr.is_manual, 0) ASC, fr.face_index ASC, fr.id ASC
            """,
            (int(run_id), resolved_file_id),
        )
        return [dict(r) for r in cur.fetchall()]

    def replace_manual_rectangles(self, *, run_id: int, file_id: int | None = None, file_path: str | None = None, rects: list[dict[str, int]]) -> None:
        """
        Заменяет ручные прямоугольники для файла (run_id + file_id/file_path).
        Приоритет: file_id (если передан), иначе file_path (если передан).
        rects: [{"x":int,"y":int,"w":int,"h":int}, ...]
        """
        # Получаем file_id
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        
        # Получаем file_path для обратной совместимости (если не передан, получаем из files)
        if file_path is None:
            cur = self.conn.cursor()
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (resolved_file_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"File with id={resolved_file_id} not found in files table")
            file_path = row[0]
        
        now = _now_utc_iso()
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM photo_rectangles WHERE run_id = ? AND file_id = ? AND COALESCE(is_manual, 0) = 1",
            (int(run_id), resolved_file_id),
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
                INSERT INTO photo_rectangles(
                  run_id, file_id, face_index,
                  bbox_x, bbox_y, bbox_w, bbox_h,
                  confidence, presence_score,
                  is_face,
                  thumb_jpeg, manual_person, ignore_flag,
                  created_at,
                  is_manual, manual_created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0, ?, 1, ?)
                """,
                (int(run_id), resolved_file_id, int(i), x, y, w, h, now, now),
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

    def clear_run_detections_for_file(self, *, run_id: int, file_id: int | None = None, file_path: str | None = None) -> None:
        """
        Удаляет все детекции для файла в рамках прогона.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        cur = self.conn.cursor()
        cur.execute("DELETE FROM photo_rectangles WHERE run_id = ? AND file_id = ?", (run_id, resolved_file_id))
        self.conn.commit()

    def clear_run_auto_rectangles_for_file(self, *, run_id: int, file_id: int | None = None, file_path: str | None = None) -> None:
        """
        Удаляет только авто-прямоугольники (is_manual=0) для файла в рамках прогона.
        Ручные прямоугольники (is_manual=1) сохраняем.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM photo_rectangles WHERE run_id = ? AND file_id = ? AND COALESCE(is_manual, 0) = 0",
            (int(run_id), resolved_file_id),
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
        
        # Получаем file_id из file_path
        resolved_file_id = _get_file_id_from_path(self.conn, file_path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {file_path}")
        
        # Для архивного режима проверяем дубликаты (append без дублирования)
        if archive_scope == 'archive':
            # Проверяем существование по file_id + bbox
            cur.execute(
                """
                SELECT id FROM photo_rectangles
                WHERE archive_scope = 'archive'
                  AND file_id = ?
                  AND bbox_x = ?
                  AND bbox_y = ?
                  AND bbox_w = ?
                  AND bbox_h = ?
                LIMIT 1
                """,
                (resolved_file_id, int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)),
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
                INSERT INTO photo_rectangles(
                  run_id, archive_scope, file_id, face_index,
                  bbox_x, bbox_y, bbox_w, bbox_h,
                  confidence, presence_score, thumb_jpeg,
                  embedding, manual_person, ignore_flag, created_at,
                  is_manual, manual_created_at,
                  is_face
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, 0, NULL, 1)
                """,
                (
                    run_id,  # может быть NULL для архива
                    archive_scope,
                    resolved_file_id,
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
                INSERT INTO photo_rectangles(
                  run_id, archive_scope, file_id, face_index,
                  bbox_x, bbox_y, bbox_w, bbox_h,
                  confidence, presence_score, thumb_jpeg,
                  embedding, manual_person, ignore_flag, created_at,
                  is_manual, manual_created_at,
                  is_face
                )
                VALUES(?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, 0, NULL, 1)
                """,
                (
                    run_id,
                    resolved_file_id,
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
                    SELECT fr.id, fr.run_id, f.path AS file_path, fr.face_index, fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                           fr.confidence, fr.embedding
                    FROM photo_rectangles fr
                    JOIN files f ON f.id = fr.file_id
                    WHERE fr.run_id = ? AND fr.embedding IS NOT NULL AND COALESCE(fr.ignore_flag, 0) = 0 AND fr.is_face = 1
                    """,
                    (int(run_id),),
                )
            else:
                cur.execute(
                    """
                    SELECT fr.id, fr.run_id, f.path AS file_path, fr.face_index, fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                           fr.confidence, fr.embedding
                    FROM photo_rectangles fr
                    JOIN files f ON f.id = fr.file_id
                    WHERE fr.embedding IS NOT NULL AND COALESCE(fr.ignore_flag, 0) = 0 AND fr.is_face = 1
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
        Обновляет file_path для photo_rectangles (когда файл физически перенесли на диске/в YaDisk).

        Важно: это чисто "техническая" миграция ссылок. Семантика детекта не меняется.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE photo_rectangles
            SET file_path = ?
            WHERE file_path = ?
            """,
            (new_file_path, old_file_path),
        )
        self.conn.commit()

    def insert_file_person(
        self,
        *,
        pipeline_run_id: int,
        file_id: int | None = None,
        file_path: str | None = None,
        person_id: int,
    ) -> None:
        """
        Вставляет простую привязку файла к персоне (file_persons).
        Использует INSERT OR REPLACE для предотвращения дубликатов.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        
        # Получаем file_path для обратной совместимости
        if file_path is None:
            cur = self.conn.cursor()
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (resolved_file_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"File with id={resolved_file_id} not found in files table")
            file_path = row[0]
        
        now = _now_utc_iso()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO file_persons (
                pipeline_run_id, file_id, person_id, created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                int(pipeline_run_id),
                resolved_file_id,
                int(person_id),
                now,
            ),
        )
        self.conn.commit()

    def delete_file_person(
        self,
        *,
        pipeline_run_id: int,
        file_id: int | None = None,
        file_path: str | None = None,
        person_id: int,
    ) -> None:
        """
        Удаляет простую привязку файла к персоне.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        cur = self.conn.cursor()
        cur.execute(
            """
            DELETE FROM file_persons
            WHERE pipeline_run_id = ? AND file_id = ? AND person_id = ?
            """,
            (
                int(pipeline_run_id),
                resolved_file_id,
                int(person_id),
            ),
        )
        self.conn.commit()

    def list_file_persons(
        self,
        *,
        pipeline_run_id: int | None = None,
        file_id: int | None = None,
        file_path: str | None = None,
        person_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Возвращает список простых привязок файлов к персонам (file_persons).
        Фильтры опциональны.
        Приоритет для file_id/file_path: file_id (если передан), иначе file_path (если передан).
        """
        cur = self.conn.cursor()
        where = []
        params = []
        if pipeline_run_id is not None:
            where.append("pipeline_run_id = ?")
            params.append(int(pipeline_run_id))
        # Обрабатываем file_id/file_path
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path) if (file_id is not None or file_path is not None) else None
        if resolved_file_id is not None:
            where.append("file_id = ?")
            params.append(resolved_file_id)
        elif file_path is not None:
            # Fallback на file_path для обратной совместимости
            where.append("file_path = ?")
            params.append(str(file_path))
        if person_id is not None:
            where.append("person_id = ?")
            params.append(int(person_id))
        where_sql = " AND ".join(where) if where else "1=1"
        cur.execute(
            f"""
            SELECT fp.pipeline_run_id, f.path AS file_path, fp.person_id, fp.created_at
            FROM file_persons fp
            JOIN files f ON f.id = fp.file_id
            WHERE {where_sql}
            ORDER BY fp.created_at DESC
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]

    def insert_file_group(
        self,
        *,
        pipeline_run_id: int,
        file_id: int | None = None,
        file_path: str | None = None,
        group_path: str,
    ) -> None:
        """
        Вставляет файл в группу (file_groups).
        Использует INSERT OR REPLACE для предотвращения дубликатов.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        
        # Получаем file_path для обратной совместимости
        if file_path is None:
            cur = self.conn.cursor()
            cur.execute("SELECT path FROM files WHERE id = ? LIMIT 1", (resolved_file_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"File with id={resolved_file_id} not found in files table")
            file_path = row[0]
        
        now = _now_utc_iso()
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT OR REPLACE INTO file_groups (
                    pipeline_run_id, file_id, group_path, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(pipeline_run_id),
                    resolved_file_id,
                    str(group_path),
                    now,
                ),
            )
            self.conn.commit()
            # Проверяем, что запись действительно вставлена
            cur.execute(
                """
                SELECT fg.id, fg.pipeline_run_id, f.path AS file_path, fg.group_path
                FROM file_groups fg
                JOIN files f ON f.id = fg.file_id
                WHERE fg.pipeline_run_id = ? AND fg.file_id = ? AND fg.group_path = ?
                LIMIT 1
                """,
                (int(pipeline_run_id), resolved_file_id, str(group_path)),
            )
            check = cur.fetchone()
            if not check:
                raise RuntimeError(f"Failed to verify insert: pipeline_run_id={pipeline_run_id}, file_id={resolved_file_id}, group_path={repr(group_path)}")
        except Exception as e:
            self.conn.rollback()
            raise

    def delete_file_group(
        self,
        *,
        pipeline_run_id: int,
        file_id: int | None = None,
        file_path: str | None = None,
        group_path: str,
    ) -> None:
        """
        Удаляет файл из группы.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        cur = self.conn.cursor()
        cur.execute(
            """
            DELETE FROM file_groups
            WHERE pipeline_run_id = ? AND file_id = ? AND group_path = ?
            """,
            (
                int(pipeline_run_id),
                resolved_file_id,
                str(group_path),
            ),
        )
        self.conn.commit()

    def list_file_groups(
        self,
        *,
        pipeline_run_id: int | None = None,
        file_id: int | None = None,
        file_path: str | None = None,
        group_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Возвращает список файлов в группах (file_groups).
        Фильтры опциональны.
        Приоритет для file_id/file_path: file_id (если передан), иначе file_path (если передан).
        """
        cur = self.conn.cursor()
        where = []
        params = []
        if pipeline_run_id is not None:
            where.append("pipeline_run_id = ?")
            params.append(int(pipeline_run_id))
        # Обрабатываем file_id/file_path
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path) if (file_id is not None or file_path is not None) else None
        if resolved_file_id is not None:
            where.append("file_id = ?")
            params.append(resolved_file_id)
        elif file_path is not None:
            # Fallback на file_path для обратной совместимости
            where.append("file_path = ?")
            params.append(str(file_path))
        if group_path is not None:
            where.append("group_path = ?")
            params.append(str(group_path))
        where_sql = " AND ".join(where) if where else "1=1"
        cur.execute(
            f"""
            SELECT fg.id, fg.pipeline_run_id, f.path AS file_path, fg.group_path, fg.created_at
            FROM file_groups fg
            JOIN files f ON f.id = fg.file_id
            WHERE {where_sql}
            ORDER BY fg.group_path ASC, fg.created_at DESC
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]

    def list_file_groups_with_counts(
        self,
        *,
        pipeline_run_id: int,
    ) -> list[dict[str, Any]]:
        """
        Возвращает список групп с количеством файлов в каждой для прогона.
        Используется для отображения подзакладок в UI.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT 
                group_path,
                COUNT(DISTINCT file_id) AS files_count,
                MAX(created_at) AS last_created_at
            FROM file_groups
            WHERE pipeline_run_id = ?
            GROUP BY group_path
            ORDER BY group_path ASC
            """,
            (int(pipeline_run_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_file_all_assignments(
        self,
        *,
        pipeline_run_id: int | None = None,
        file_id: int | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Возвращает все привязки файла к персонам:
        - через лица (photo_rectangles.manual_person_id и photo_rectangles.cluster_id → face_clusters.person_id)
        - прямая привязка (file_persons)
        
        Если указан pipeline_run_id, получает face_run_id из pipeline_runs для фильтрации photo_rectangles.
        Приоритет: file_id (если передан), иначе file_path (если передан).
        """
        # Получаем file_id
        resolved_file_id = _get_file_id(self.conn, file_id=file_id, file_path=file_path)
        if resolved_file_id is None:
            raise ValueError("Either file_id or file_path must be provided")
        
        cur = self.conn.cursor()
        
        # Если указан pipeline_run_id, получаем face_run_id для фильтрации
        face_run_id = None
        if pipeline_run_id is not None:
            try:
                from backend.common.db import PipelineStore
                ps = PipelineStore()
                try:
                    pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
                    if pr:
                        face_run_id = pr.get("face_run_id")
                finally:
                    ps.close()
            except Exception:
                pass
        
        # Привязки через лица (ручные привязки + через кластеры через photo_rectangles.cluster_id)
        if face_run_id is not None:
            cur.execute(
                """
                SELECT DISTINCT person_id, person_name
                FROM (
                    -- Ручные привязки (photo_rectangles.manual_person_id)
                    SELECT DISTINCT fr.manual_person_id AS person_id, p.name AS person_name
                    FROM photo_rectangles fr
                    LEFT JOIN persons p ON p.id = fr.manual_person_id
                    WHERE fr.file_id = ? AND fr.run_id = ? AND fr.manual_person_id IS NOT NULL
                    
                    UNION
                    
                    -- Привязки через кластеры (photo_rectangles.cluster_id)
                    SELECT DISTINCT fc.person_id, p.name AS person_name
                    FROM photo_rectangles fr
                    JOIN face_clusters fc ON fc.id = fr.cluster_id
                    LEFT JOIN persons p ON p.id = fc.person_id
                    WHERE fr.file_id = ? AND fr.run_id = ? AND fr.is_face = 1 AND fc.person_id IS NOT NULL
                )
                """,
                (resolved_file_id, int(face_run_id), resolved_file_id, int(face_run_id)),
            )
        else:
            cur.execute(
                """
                SELECT DISTINCT person_id, person_name
                FROM (
                    -- Ручные привязки (photo_rectangles.manual_person_id)
                    SELECT DISTINCT fr.manual_person_id AS person_id, p.name AS person_name
                    FROM photo_rectangles fr
                    LEFT JOIN persons p ON p.id = fr.manual_person_id
                    WHERE fr.file_id = ? AND fr.manual_person_id IS NOT NULL
                    
                    UNION
                    
                    -- Привязки через кластеры (photo_rectangles.cluster_id)
                    SELECT DISTINCT fc.person_id, p.name AS person_name
                    FROM photo_rectangles fr
                    JOIN face_clusters fc ON fc.id = fr.cluster_id
                    LEFT JOIN persons p ON p.id = fc.person_id
                    WHERE fr.file_id = ? AND fr.is_face = 1 AND fc.person_id IS NOT NULL
                )
                """,
                (resolved_file_id, resolved_file_id),
            )
        face_assignments = [dict(r) for r in cur.fetchall()]
        
        # Прямые привязки
        file_persons_list = self.list_file_persons(
            pipeline_run_id=pipeline_run_id,
            file_id=resolved_file_id,
        )
        direct_assignments = []
        for fp in file_persons_list:
            cur.execute("SELECT id, name FROM persons WHERE id = ?", (fp["person_id"],))
            person_row = cur.fetchone()
            if person_row:
                direct_assignments.append({
                    "person_id": fp["person_id"],
                    "person_name": person_row["name"],
                })
        
        return {
            "face_assignments": face_assignments,
            "person_rectangle_assignments": [],  # таблица person_rectangles удалена
            "direct_assignments": direct_assignments,
        }


def list_folders(*, location: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
    """
    Возвращает папки из таблицы `folders` в порядке:
    1) с заданным sort_order (по возрастанию)
    2) остальные (по name)

    Фильтры `location` и `role` опциональны.
    """
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
        DEPRECATED: Метод удален - метки должны быть run-scoped.
        Используйте set_run_faces_manual_label() с pipeline_run_id.
        """
        raise DeprecationWarning(
            "set_faces_manual_label() is deprecated. Use set_run_faces_manual_label() with pipeline_run_id instead."
        )

    # --- run-scoped manual labels (pipeline_run_id + file_id) ---
    def _ensure_run_manual_row(self, *, pipeline_run_id: int, path: str) -> None:
        # Получаем file_id из path
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO files_manual_labels(pipeline_run_id, file_id) VALUES (?, ?)",
            (int(pipeline_run_id), resolved_file_id),
        )

    def delete_run_manual_labels(self, *, pipeline_run_id: int, path: str) -> None:
        """
        Полный сброс ручных меток для конкретного прогона и пути.
        """
        # Получаем file_id из path
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM files_manual_labels WHERE pipeline_run_id = ? AND file_id = ?",
            (int(pipeline_run_id), resolved_file_id),
        )
        self.conn.commit()

    def set_run_faces_manual_label(self, *, pipeline_run_id: int, path: str, label: str | None) -> None:
        """
        Ручная правка результата "лица/нет лиц" для файла В РАМКАХ ПРОГОНА.
        label: 'faces' | 'no_faces' | None (сброс)
        """
        # Получаем file_id из path
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        
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
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                (int(pipeline_run_id), resolved_file_id),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET faces_manual_label = ?, faces_manual_at = ?
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                (lab, _now_utc_iso(), int(pipeline_run_id), resolved_file_id),
            )
        self.conn.commit()

    def update_run_manual_labels_path(self, *, pipeline_run_id: int, old_path: str, new_path: str) -> None:
        """
        При перемещении файла (и обновлении files.path) нужно переносить run-scoped manual метки,
        иначе они "теряются" после move (labels привязаны к file_id, который не меняется при move).

        Поведение:
        - если old_path отсутствует в files_manual_labels для этого прогона — ничего не делаем
        - если new_path уже есть — сливаем значения (бережно) и удаляем old_path
        - иначе просто обновляем file_id (если файл переместился, file_id должен остаться тем же)
        
        ПРИМЕЧАНИЕ: После миграции на file_id этот метод может быть упрощен или удален,
        так как file_id не меняется при перемещении файла.
        """
        oldp = str(old_path or "")
        newp = str(new_path or "")
        if not oldp or not newp or oldp == newp:
            return
        
        # Получаем file_id из old_path и new_path
        old_file_id = _get_file_id_from_path(self.conn, oldp)
        new_file_id = _get_file_id_from_path(self.conn, newp)
        
        if old_file_id is None:
            return  # Старый файл не найден
        
        if new_file_id is None:
            return  # Новый файл не найден
        
        # Если file_id одинаковый - файл не переместился, только путь изменился
        # В этом случае ничего делать не нужно, так как метки привязаны к file_id
        if old_file_id == new_file_id:
            return
        
        rid = int(pipeline_run_id)
        cur = self.conn.cursor()

        cur.execute(
            "SELECT * FROM files_manual_labels WHERE pipeline_run_id = ? AND file_id = ? LIMIT 1",
            (rid, old_file_id),
        )
        old_row = cur.fetchone()
        if not old_row:
            return

        cur.execute(
            "SELECT * FROM files_manual_labels WHERE pipeline_run_id = ? AND file_id = ? LIMIT 1",
            (rid, new_file_id),
        )
        new_row = cur.fetchone()

        if not new_row:
            # Simple rename - обновляем file_id
            cur.execute(
                "UPDATE files_manual_labels SET file_id = ? WHERE pipeline_run_id = ? AND file_id = ?",
                (new_file_id, rid, old_file_id),
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
            WHERE pipeline_run_id = ? AND file_id = ?
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
                new_file_id,
            ),
        )
        # Drop old row (we merged it)
        cur.execute("DELETE FROM files_manual_labels WHERE pipeline_run_id = ? AND file_id = ?", (rid, old_file_id))
        self.conn.commit()

    # --- video manual frames (run-scoped) ---
    def get_video_manual_frames(self, *, pipeline_run_id: int, path: str) -> dict[int, dict[str, Any]]:
        """
        Returns mapping: frame_idx -> {frame_idx, t_sec, rects:[...], updated_at}
        """
        # Получаем file_id из path
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT frame_idx, t_sec, rects_json, updated_at
            FROM video_manual_frames
            WHERE pipeline_run_id = ? AND file_id = ?
            ORDER BY frame_idx ASC
            """,
            (int(pipeline_run_id), resolved_file_id),
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
        # Получаем file_id из path
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO video_manual_frames(pipeline_run_id, file_id, frame_idx, t_sec, rects_json, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(pipeline_run_id, file_id, frame_idx) DO UPDATE SET
              t_sec = excluded.t_sec,
              rects_json = excluded.rects_json,
              updated_at = excluded.updated_at
            """,
            (
                int(pipeline_run_id),
                resolved_file_id,
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
        # Получаем file_id из path
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        
        self._ensure_run_manual_row(pipeline_run_id=int(pipeline_run_id), path=str(path))
        cur = self.conn.cursor()
        if bool(is_people_no_face):
            cur.execute(
                """
                UPDATE files_manual_labels
                SET people_no_face_manual = 1, people_no_face_person = ?
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                ((person or "").strip() or None, int(pipeline_run_id), resolved_file_id),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET people_no_face_manual = 0, people_no_face_person = NULL
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                (int(pipeline_run_id), resolved_file_id),
            )
        self.conn.commit()

    def set_run_animals_manual(self, *, pipeline_run_id: int, path: str, is_animal: bool, kind: str | None = None) -> None:
        """
        Ручная разметка животных (ground truth) В РАМКАХ ПРОГОНА.
        """
        # Получаем file_id из path
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        
        self._ensure_run_manual_row(pipeline_run_id=int(pipeline_run_id), path=str(path))
        cur = self.conn.cursor()
        if bool(is_animal):
            cur.execute(
                """
                UPDATE files_manual_labels
                SET animals_manual = 1, animals_manual_kind = ?, animals_manual_at = ?
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                ((kind or "").strip() or None, _now_utc_iso(), int(pipeline_run_id), resolved_file_id),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET animals_manual = 0, animals_manual_kind = NULL, animals_manual_at = NULL
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                (int(pipeline_run_id), resolved_file_id),
            )
        self.conn.commit()

    def set_run_quarantine_manual(self, *, pipeline_run_id: int, path: str, is_quarantine: bool) -> None:
        """
        Ручная пометка "карантин" В РАМКАХ ПРОГОНА.
        """
        resolved_file_id = _get_file_id_from_path(self.conn, path)
        if resolved_file_id is None:
            raise ValueError(f"File not found in files table: {path}")
        self._ensure_run_manual_row(pipeline_run_id=int(pipeline_run_id), path=str(path))
        cur = self.conn.cursor()
        if bool(is_quarantine):
            cur.execute(
                """
                UPDATE files_manual_labels
                SET quarantine_manual = 1, quarantine_manual_at = ?
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                (_now_utc_iso(), int(pipeline_run_id), resolved_file_id),
            )
        else:
            cur.execute(
                """
                UPDATE files_manual_labels
                SET quarantine_manual = 0, quarantine_manual_at = NULL
                WHERE pipeline_run_id = ? AND file_id = ?
                """,
                (int(pipeline_run_id), resolved_file_id),
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
        DEPRECATED: Метод удален - метки должны быть run-scoped.
        Используйте set_run_animals_manual() с pipeline_run_id.
        """
        raise DeprecationWarning(
            "set_animals_manual() is deprecated. Use set_run_animals_manual() with pipeline_run_id instead."
        )

    def set_people_no_face_manual(self, *, path: str, is_people_no_face: bool, person: str | None = None) -> None:
        """
        DEPRECATED: Метод удален - метки должны быть run-scoped.
        Используйте set_run_people_no_face_manual() с pipeline_run_id.
        """
        raise DeprecationWarning(
            "set_people_no_face_manual() is deprecated. Use set_run_people_no_face_manual() with pipeline_run_id instead."
        )

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
