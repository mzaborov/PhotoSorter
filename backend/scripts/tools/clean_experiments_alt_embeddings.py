#!/usr/bin/env python3
"""
Удаляет эмбеддинги альтернативной модели из experiments.db (face_embeddings_alt).
Использование после того как модель не дала улучшения (например antelopev2):

  python backend/scripts/tools/clean_experiments_alt_embeddings.py --model-key antelopev2
  python backend/scripts/tools/clean_experiments_alt_embeddings.py --model-key antelopev2 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backend.common.experiments_db import get_experiments_connection


def main() -> int:
    parser = argparse.ArgumentParser(description="Удалить записи face_embeddings_alt по model_key")
    parser.add_argument("--model-key", type=str, required=True, help="Ключ модели (например antelopev2)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, сколько записей будет удалено")
    args = parser.parse_args()

    conn = get_experiments_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM face_embeddings_alt WHERE model_key = ?", (args.model_key,))
    n = cur.fetchone()[0]
    if n == 0:
        print(f"Записей с model_key={args.model_key!r} нет.")
        conn.close()
        return 0
    if args.dry_run:
        print(f"Будет удалено записей: {n} (model_key={args.model_key!r}). Запустите без --dry-run для удаления.")
        conn.close()
        return 0
    cur.execute("DELETE FROM face_embeddings_alt WHERE model_key = ?", (args.model_key,))
    conn.commit()
    print(f"Удалено записей: {n} (model_key={args.model_key!r})")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
