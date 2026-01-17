"""Выполняет VACUUM для освобождения места в БД после удаления данных."""

import sqlite3
import argparse
from pathlib import Path
import shutil
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Выполняет VACUUM для освобождения места в БД")
    parser.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    parser.add_argument("--backup", action="store_true", help="Создать резервную копию перед VACUUM")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"БД не найдена: {db_path}")
        return
    
    # Размер до VACUUM
    size_before = db_path.stat().st_size
    print(f"Размер БД до VACUUM: {size_before / (1024**3):.2f} GB ({size_before:,} байт)")
    print()
    
    # Создаём резервную копию, если нужно
    if args.backup:
        backup_path = db_path.parent / f"{db_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{db_path.suffix}"
        print(f"Создание резервной копии: {backup_path}")
        shutil.copy2(db_path, backup_path)
        print(f"Резервная копия создана: {backup_path.stat().st_size / (1024**3):.2f} GB")
        print()
    
    # Выполняем VACUUM
    print("Выполнение VACUUM...")
    print("(это может занять несколько минут для большой БД)")
    
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("VACUUM")
        conn.commit()
        print("VACUUM выполнен успешно")
    except Exception as e:
        print(f"Ошибка при выполнении VACUUM: {e}")
        conn.close()
        return
    finally:
        conn.close()
    
    # Размер после VACUUM
    size_after = db_path.stat().st_size
    size_freed = size_before - size_after
    
    print()
    print(f"Размер БД после VACUUM: {size_after / (1024**3):.2f} GB ({size_after:,} байт)")
    print(f"Освобождено места: {size_freed / (1024**3):.2f} GB ({size_freed:,} байт)")
    print(f"Процент освобождения: {size_freed * 100 / max(size_before, 1):.1f}%")


if __name__ == "__main__":
    main()
