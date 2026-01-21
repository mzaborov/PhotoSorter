#!/usr/bin/env python3
"""
Скрипт для создания резервной копии базы данных PhotoSorter.
Создает timestamped backup файл в папке data/backups/
"""
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os

# Добавляем корень проекта в путь
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.common.db import DB_PATH


def create_backup(backup_dir: Path | None = None) -> Path:
    """
    Создает резервную копию базы данных.
    
    Args:
        backup_dir: Папка для сохранения бекапа (по умолчанию data/backups/)
    
    Returns:
        Path к созданному файлу бекапа
    """
    db_path = DB_PATH
    
    if not db_path.exists():
        raise FileNotFoundError(f"База данных не найдена: {db_path}")
    
    if backup_dir is None:
        backup_dir = project_root / "data" / "backups"
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"photosorter_backup_{timestamp}.db"
    backup_path = backup_dir / backup_filename
    
    print(f"Создание резервной копии базы данных...")
    print(f"  Источник: {db_path}")
    print(f"  Назначение: {backup_path}")
    
    # Используем SQLite backup API для безопасного копирования
    source_conn = sqlite3.connect(str(db_path))
    backup_conn = sqlite3.connect(str(backup_path))
    
    try:
        source_conn.backup(backup_conn)
        backup_conn.commit()
        
        # Проверяем размер файла
        backup_size = backup_path.stat().st_size
        print(f"  Размер бекапа: {backup_size / 1024 / 1024:.2f} MB")
        print(f"  [OK] Резервная копия создана успешно")
    finally:
        source_conn.close()
        backup_conn.close()
    
    return backup_path


def main() -> int:
    """Главная функция."""
    try:
        backup_path = create_backup()
        print(f"\n[OK] Бекап сохранен: {backup_path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Ошибка при создании бекапа: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
