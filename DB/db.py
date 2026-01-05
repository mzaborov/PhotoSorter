import sqlite3
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

# data/photosorter.db рядом с проектом
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "photosorter.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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
        CREATE TABLE IF NOT EXISTS yd_files (
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
        "yd_files",
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
        },
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

    # Индексы для быстрых группировок дублей.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_yd_files_hash ON yd_files(hash_alg, hash_value);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_yd_files_parent ON yd_files(parent_path);")
    # Устойчивый идентификатор: уникален, если известен (partial unique index).
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_yd_files_resource_id ON yd_files(resource_id) WHERE resource_id IS NOT NULL;"
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

    def insert_detection(
        self,
        *,
        run_id: int,
        file_path: str,
        face_index: int,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        confidence: float | None,
        presence_score: float | None,
        thumb_jpeg: bytes | None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO face_rectangles(
              run_id, file_path, face_index,
              bbox_x, bbox_y, bbox_w, bbox_h,
              confidence, presence_score, thumb_jpeg,
              manual_person, ignore_flag, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?)
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
                _now_utc_iso(),
            ),
        )
        self.conn.commit()

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

    # --- yd_files helpers (инвентарь/хэши/дубли) ---
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
            INSERT INTO yd_files(
              path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type,
              hash_alg, hash_value, hash_source, status, error, scanned_at, hashed_at, last_run_id
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              resource_id = COALESCE(excluded.resource_id, yd_files.resource_id),
              inventory_scope = COALESCE(excluded.inventory_scope, yd_files.inventory_scope),
              name = excluded.name,
              parent_path = excluded.parent_path,
              size = excluded.size,
              created = excluded.created,
              modified = excluded.modified,
              mime_type = excluded.mime_type,
              media_type = excluded.media_type,
              hash_alg = COALESCE(excluded.hash_alg, yd_files.hash_alg),
              hash_value = COALESCE(excluded.hash_value, yd_files.hash_value),
              hash_source = COALESCE(excluded.hash_source, yd_files.hash_source),
              status = excluded.status,
              error = excluded.error,
              scanned_at = excluded.scanned_at,
              hashed_at = COALESCE(excluded.hashed_at, yd_files.hashed_at),
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
        cur.execute("SELECT hash_alg, hash_value FROM yd_files WHERE path = ? LIMIT 1", (path,))
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
            FROM yd_files
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
            FROM yd_files
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
            UPDATE yd_files
            SET ignore_archive_dup_run_id = ?
            WHERE path IN ({q})
            """,
            [run_id, *paths],
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def list_source_dups_in_archive(self, *, source_run_id: int, archive_prefix: str = "disk:/Фото") -> list[dict[str, Any]]:
        """
        Возвращает пары (source file -> archive matches) по совпадающему хэшу.
        Источник ограничиваем last_run_id=source_run_id, чтобы не смешивать разные выборы папки.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
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
            FROM yd_files s
            JOIN yd_files a
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
              AND a.path LIKE ?
            ORDER BY s.path ASC, a.path ASC
            """,
            (source_run_id, source_run_id, f"{archive_prefix.rstrip('/')}/%"),
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
            FROM yd_files
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
        cur.execute("SELECT 1 FROM yd_files WHERE path = ? LIMIT 1", (path,))
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
        cur.execute(f"UPDATE yd_files SET status = 'deleted', error = NULL WHERE path IN ({q})", paths)
        self.conn.commit()
        return int(cur.rowcount or 0)

    def update_path(self, *, old_path: str, new_path: str, new_name: str | None, new_parent_path: str | None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE yd_files
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
            UPDATE yd_files
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
                UPDATE yd_files
                SET faces_manual_label = NULL, faces_manual_at = NULL
                WHERE path = ?
                """,
                (path,),
            )
        else:
            cur.execute(
                """
                UPDATE yd_files
                SET faces_manual_label = ?, faces_manual_at = ?
                WHERE path = ?
                """,
                (lab, _now_utc_iso(), path),
            )
        self.conn.commit()

    def get_row_by_resource_id(self, *, resource_id: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM yd_files WHERE resource_id = ? LIMIT 1", (resource_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_row_by_path(self, *, path: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM yd_files WHERE path = ? LIMIT 1", (path,))
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
                INSERT INTO yd_files(
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
            cur.execute("SELECT id, resource_id, status FROM yd_files WHERE path = ? LIMIT 1", (path,))
            other = cur.fetchone()
            if other and int(other["id"]) != existing_id:
                cur.execute(
                    "UPDATE yd_files SET status='deleted', error=NULL WHERE id = ?",
                    (int(other["id"]),),
                )

        size_changed = (existing_size is None) != (size is None) or (existing_size is not None and size is not None and int(existing_size) != int(size))
        mod_changed = (existing_modified is None) != (modified is None) or (existing_modified is not None and modified is not None and existing_modified != modified)
        content_changed = bool(size_changed or mod_changed)

        if content_changed:
            # Содержимое могло поменяться -> сбрасываем хэш и (на всякий случай) длительность.
            cur.execute(
                """
                UPDATE yd_files
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
                UPDATE yd_files
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
        cur.execute("SELECT duration_sec FROM yd_files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if not row:
            return None
        val = row[0]
        return int(val) if isinstance(val, (int, float)) else None

    def set_duration(self, *, path: str, duration_sec: int | None, source: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE yd_files
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
            INSERT INTO yd_files(
              path, resource_id, inventory_scope, name, parent_path, size, created, modified, mime_type, media_type,
              hash_alg, hash_value, hash_source, status, error, scanned_at, hashed_at, last_run_id
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              resource_id = COALESCE(excluded.resource_id, yd_files.resource_id),
              inventory_scope = COALESCE(excluded.inventory_scope, yd_files.inventory_scope),
              name = excluded.name,
              parent_path = excluded.parent_path,
              size = excluded.size,
              created = excluded.created,
              modified = excluded.modified,
              mime_type = excluded.mime_type,
              media_type = excluded.media_type,
              hash_alg = COALESCE(excluded.hash_alg, yd_files.hash_alg),
              hash_value = COALESCE(excluded.hash_value, yd_files.hash_value),
              hash_source = COALESCE(excluded.hash_source, yd_files.hash_source),
              status = excluded.status,
              error = excluded.error,
              scanned_at = excluded.scanned_at,
              hashed_at = COALESCE(excluded.hashed_at, yd_files.hashed_at),
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
        cur.execute("SELECT hash_alg, hash_value FROM yd_files WHERE path = ? LIMIT 1", (path,))
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
            FROM yd_files
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
            FROM yd_files
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
            UPDATE yd_files
            SET ignore_archive_dup_run_id = ?
            WHERE path IN ({q})
            """,
            [run_id, *paths],
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def list_source_dups_in_archive(self, *, source_run_id: int, archive_prefix: str = "disk:/Фото") -> list[dict[str, Any]]:
        """
        Возвращает пары (source file -> archive matches) по совпадающему хэшу.
        Источник ограничиваем last_run_id=source_run_id, чтобы не смешивать разные выборы папки.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
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
            FROM yd_files s
            JOIN yd_files a
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
              AND a.path LIKE ?
            ORDER BY s.path ASC, a.path ASC
            """,
            (source_run_id, source_run_id, f"{archive_prefix.rstrip('/')}/%"),
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
            FROM yd_files
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
        cur.execute("SELECT 1 FROM yd_files WHERE path = ? LIMIT 1", (path,))
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
        cur.execute(f"UPDATE yd_files SET status = 'deleted', error = NULL WHERE path IN ({q})", paths)
        self.conn.commit()
        return int(cur.rowcount or 0)

    def update_path(self, *, old_path: str, new_path: str, new_name: str | None, new_parent_path: str | None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE yd_files
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
            UPDATE yd_files
            SET faces_count = ?, faces_run_id = ?, faces_scanned_at = ?
            WHERE path = ?
            """,
            (int(faces_count), int(faces_run_id), faces_scanned_at or _now_utc_iso(), path),
        )
        self.conn.commit()

    def get_row_by_resource_id(self, *, resource_id: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM yd_files WHERE resource_id = ? LIMIT 1", (resource_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_row_by_path(self, *, path: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM yd_files WHERE path = ? LIMIT 1", (path,))
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
                INSERT INTO yd_files(
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
            cur.execute("SELECT id, resource_id, status FROM yd_files WHERE path = ? LIMIT 1", (path,))
            other = cur.fetchone()
            if other and int(other["id"]) != existing_id:
                cur.execute(
                    "UPDATE yd_files SET status='deleted', error=NULL WHERE id = ?",
                    (int(other["id"]),),
                )

        size_changed = (existing_size is None) != (size is None) or (existing_size is not None and size is not None and int(existing_size) != int(size))
        mod_changed = (existing_modified is None) != (modified is None) or (existing_modified is not None and modified is not None and existing_modified != modified)
        content_changed = bool(size_changed or mod_changed)

        if content_changed:
            # Содержимое могло поменяться -> сбрасываем хэш и (на всякий случай) длительность.
            cur.execute(
                """
                UPDATE yd_files
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
                UPDATE yd_files
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
        cur.execute("SELECT duration_sec FROM yd_files WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if not row:
            return None
        val = row[0]
        return int(val) if isinstance(val, (int, float)) else None

    def set_duration(self, *, path: str, duration_sec: int | None, source: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE yd_files
            SET duration_sec = ?, duration_source = ?, duration_at = ?
            WHERE path = ?
            """,
            (duration_sec, source, _now_utc_iso(), path),
        )
        self.conn.commit()


if __name__ == "__main__":
    init_db()
    print("DB initialized:", DB_PATH)
