import sqlite3
from pathlib import Path
from typing import Any

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
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


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

    conn.commit()
    conn.close()


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


if __name__ == "__main__":
    init_db()
    print("DB initialized:", DB_PATH)
