#!/usr/bin/env python3
"""
Создаёт бекап базы данных photosorter.db.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import DB_PATH


def main() -> int:
    if not DB_PATH.exists():
        print(f"БД не найдена: {DB_PATH}")
        return 1
    
    # Создаём имя бекапа с меткой времени
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DB_PATH.parent / f"photosorter_backup_{timestamp}.db"
    
    # Копируем файл
    shutil.copy2(DB_PATH, backup_path)
    
    print(f"Бекап создан: {backup_path}")
    print(f"Размер: {backup_path.stat().st_size / 1024 / 1024:.2f} МБ")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
