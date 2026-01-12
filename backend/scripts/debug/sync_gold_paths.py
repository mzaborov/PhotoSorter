#!/usr/bin/env python3
"""
Синхронизирует пути в gold файлах с реальными путями в БД.

Для каждого пути в gold файлах:
1. Проверяет, существует ли файл с таким путём в БД
2. Если нет, пытается найти файл по имени (basename)
3. Если найден новый путь, обновляет его в gold файлах
"""

import os
import sys
from pathlib import Path

# Добавляем backend в путь для импортов
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from common.db import DedupStore
from logic.gold.store import (
    gold_file_map,
    gold_read_lines,
    gold_write_lines,
    gold_read_ndjson_by_path,
    gold_write_ndjson_by_path,
    gold_faces_manual_rects_path,
    gold_faces_video_frames_path,
    gold_normalize_path,
)


def _find_file_by_name_in_db(ds: DedupStore, file_name: str) -> str | None:
    """Ищет файл в БД по имени (basename). Возвращает путь или None."""
    cur = ds.conn.cursor()
    cur.execute(
        """
        SELECT path, name
        FROM files
        WHERE status != 'deleted'
          AND (name = ? OR path LIKE ?)
        LIMIT 1
        """,
        (file_name, f"%/{file_name}"),
    )
    row = cur.fetchone()
    if row:
        return str(row[0] or "")
    return None


def _normalize_path_for_comparison(path: str) -> str:
    """Нормализует путь для сравнения (убирает префиксы, приводит к единому формату)."""
    p = path.strip()
    if p.startswith("local:"):
        return p[6:]  # убираем "local:"
    return p


def sync_gold_paths(dry_run: bool = True) -> dict[str, int]:
    """
    Синхронизирует пути в gold файлах с БД.
    
    Возвращает статистику: {"updated": количество обновлённых путей, "not_found": количество не найденных}.
    """
    stats = {"updated": 0, "not_found": 0}
    
    ds = DedupStore()
    try:
        # Получаем все пути из БД для быстрого поиска
        cur = ds.conn.cursor()
        cur.execute("SELECT path FROM files WHERE status != 'deleted'")
        db_paths = {_normalize_path_for_comparison(str(row[0] or "")): str(row[0] or "") for row in cur.fetchall()}
        
        # Обрабатываем txt gold файлы
        gold_map = gold_file_map()
        for name, gold_path in gold_map.items():
            if not gold_path.exists():
                continue
            print(f"\nОбработка {name}...")
            lines = gold_read_lines(gold_path)
            updated_lines = []
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    updated_lines.append(line)
                    continue
                
                # Нормализуем путь из gold
                nm = gold_normalize_path(line_stripped)
                gold_path_normalized = _normalize_path_for_comparison(nm["path"])
                
                # Проверяем, есть ли такой путь в БД
                if gold_path_normalized in db_paths:
                    # Путь найден, оставляем как есть
                    updated_lines.append(line)
                else:
                    # Путь не найден, пытаемся найти по имени
                    file_name = os.path.basename(gold_path_normalized)
                    if file_name:
                        new_path = _find_file_by_name_in_db(ds, file_name)
                        if new_path:
                            # Найден новый путь, обновляем
                            new_path_normalized = _normalize_path_for_comparison(new_path)
                            if new_path_normalized != gold_path_normalized:
                                print(f"  Обновление: {line_stripped[:60]}... -> {new_path[:60]}...")
                                updated_lines.append(new_path)
                                stats["updated"] += 1
                            else:
                                updated_lines.append(line)
                        else:
                            # Файл не найден в БД
                            print(f"  Не найден: {line_stripped[:60]}...")
                            updated_lines.append(line)
                            stats["not_found"] += 1
                    else:
                        updated_lines.append(line)
            
            # Записываем обновлённые строки
            if not dry_run:
                gold_write_lines(gold_path, updated_lines)
            else:
                # Подсчитываем количество изменённых строк
                changed_count = sum(1 for i, line in enumerate(updated_lines) if i < len(lines) and line != lines[i])
                print(f"  (DRY-RUN: было бы обновлено {changed_count} строк)")
        
        # Обрабатываем NDJSON gold файлы
        for ndjson_path in [gold_faces_manual_rects_path(), gold_faces_video_frames_path()]:
            if not ndjson_path.exists():
                continue
            print(f"\nОбработка {ndjson_path.name}...")
            items = gold_read_ndjson_by_path(ndjson_path)
            updated_items = {}
            for old_path, item_data in items.items():
                old_path_normalized = _normalize_path_for_comparison(old_path)
                
                # Проверяем, есть ли такой путь в БД
                if old_path_normalized in db_paths:
                    # Путь найден, оставляем как есть
                    updated_items[old_path] = item_data
                else:
                    # Путь не найден, пытаемся найти по имени
                    file_name = os.path.basename(old_path_normalized)
                    if file_name:
                        new_path = _find_file_by_name_in_db(ds, file_name)
                        if new_path:
                            # Найден новый путь, обновляем
                            new_path_normalized = _normalize_path_for_comparison(new_path)
                            if new_path_normalized != old_path_normalized:
                                print(f"  Обновление: {old_path[:60]}... -> {new_path[:60]}...")
                                updated_items[new_path] = item_data
                                stats["updated"] += 1
                            else:
                                updated_items[old_path] = item_data
                        else:
                            # Файл не найден в БД
                            print(f"  Не найден: {old_path[:60]}...")
                            updated_items[old_path] = item_data
                            stats["not_found"] += 1
                    else:
                        updated_items[old_path] = item_data
            
            # Записываем обновлённые данные
            if not dry_run:
                gold_write_ndjson_by_path(ndjson_path, updated_items)
            else:
                print(f"  (DRY-RUN: было бы обновлено {len([k for k in updated_items.keys() if k not in items])} записей)")
    
    finally:
        ds.close()
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Синхронизирует пути в gold файлах с БД")
    parser.add_argument("--apply", action="store_true", help="Применить изменения (по умолчанию dry-run)")
    args = parser.parse_args()
    
    dry_run = not args.apply
    
    if dry_run:
        print("=== DRY-RUN режим (изменения не будут сохранены) ===")
    else:
        print("=== ПРИМЕНЕНИЕ ИЗМЕНЕНИЙ ===")
    
    stats = sync_gold_paths(dry_run=dry_run)
    
    print(f"\n=== Результаты ===")
    print(f"Обновлено путей: {stats['updated']}")
    print(f"Не найдено файлов: {stats['not_found']}")
    
    if dry_run and stats['updated'] > 0:
        print("\nДля применения изменений запустите с флагом --apply")
