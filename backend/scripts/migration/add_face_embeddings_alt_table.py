#!/usr/bin/env python3
"""
Создаёт БД экспериментов (experiments.db) и таблицы tune_snapshot, face_embeddings_alt, tune_face_results.
Все эксперименты по тюнингу и альтернативным эмбеддингам — только в этой БД, основная не трогается.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.experiments_db import get_experiments_connection, ensure_experiments_tables


def main() -> int:
    conn = get_experiments_connection()
    ensure_experiments_tables(conn)
    conn.close()
    print("OK: experiments.db и таблицы (tune_snapshot, face_embeddings_alt, tune_face_results) созданы")
    return 0


if __name__ == "__main__":
    sys.exit(main())
