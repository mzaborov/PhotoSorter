"""
Отдельная БД для экспериментов (тюнинг кластеризации, альтернативные эмбеддинги).
Не трогает основную data/photosorter.db.
"""

import sqlite3
from pathlib import Path

# backend/scripts/debug/data/experiments.db (относительно корня репозитория)
EXPERIMENTS_DB_PATH = Path(__file__).resolve().parents[2] / "backend" / "scripts" / "debug" / "data" / "experiments.db"


def get_experiments_connection():
    """Подключение к БД экспериментов (отдельная от основной)."""
    EXPERIMENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(EXPERIMENTS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_experiments_tables(conn: sqlite3.Connection) -> None:
    """Создаёт таблицы в БД экспериментов, если их ещё нет."""
    cur = conn.cursor()
    # Снапшот ground truth + эмбеддинги по умолчанию (копия из основной БД для тюнинга)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tune_snapshot (
            rectangle_id  INTEGER PRIMARY KEY,
            person_id     INTEGER NOT NULL,
            embedding     BLOB NOT NULL
        );
    """)
    # Эмбеддинги альтернативных моделей (считаются скриптом backfill, ключ — model_key)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings_alt (
            rectangle_id  INTEGER NOT NULL,
            model_key     TEXT NOT NULL,
            embedding     BLOB NOT NULL,
            PRIMARY KEY (rectangle_id, model_key)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_alt_model ON face_embeddings_alt(model_key);")
    # Результаты прогонов тюнинга (eps, min_samples, ARI и т.д.)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tune_face_results (
            run_ts    TEXT NOT NULL,
            n_faces   INTEGER NOT NULL,
            n_persons INTEGER NOT NULL,
            eps       REAL NOT NULL,
            min_samples INTEGER NOT NULL,
            ari       REAL NOT NULL,
            n_clusters INTEGER NOT NULL,
            n_noise   INTEGER NOT NULL,
            model_key TEXT,
            algorithm TEXT DEFAULT 'dbscan'
        );
    """)
    # Миграция: добавить algorithm если таблица создана раньше без неё
    cur.execute("PRAGMA table_info(tune_face_results)")
    columns = [row[1] for row in cur.fetchall()]
    if "algorithm" not in columns:
        cur.execute("ALTER TABLE tune_face_results ADD COLUMN algorithm TEXT DEFAULT 'dbscan'")
    # Аутлайеры экспериментов (только в experiments.db; основная БД не трогаем)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment_outliers (
            rectangle_id INTEGER NOT NULL PRIMARY KEY,
            run_ts       TEXT NOT NULL,
            cluster_id   INTEGER,
            distance     REAL
        );
    """)
    # Результаты прогонов кластеризации в experiments (rectangle_id -> cluster_label), без записи в основную БД
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment_cluster_labels (
            run_ts        TEXT NOT NULL,
            rectangle_id  INTEGER NOT NULL,
            cluster_label INTEGER NOT NULL,
            PRIMARY KEY (run_ts, rectangle_id)
        );
    """)
    conn.commit()
