#!/usr/bin/env python3
"""
Скрипт для исправления is_face=0 для прямоугольников "без лица" для указанных файлов и персон.
"""

import sys
import os
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.common.db import FaceStore, get_connection, _get_file_id


def fix_is_face_for_files_and_persons(file_paths: list[str], person_names: list[str], dry_run: bool = True):
    """
    Исправляет is_face=0 для прямоугольников указанных файлов и персон.
    
    Args:
        file_paths: Список путей к файлам (с префиксом local: или без)
        person_names: Список имен персон
        dry_run: Если True, только показывает что будет исправлено, не изменяет БД
    """
    fs = FaceStore()
    conn = get_connection()
    
    try:
        fs_cur = fs.conn.cursor()
        cur = conn.cursor()
        
        # Получаем ID персон из FaceStore (персоны хранятся там)
        person_ids = {}
        for person_name in person_names:
            # Сначала пробуем точное совпадение
            fs_cur.execute("SELECT id, name FROM persons WHERE name = ?", (person_name,))
            row = fs_cur.fetchone()
            if not row:
                # Если не найдено, пробуем без учета регистра
                fs_cur.execute("SELECT id, name FROM persons WHERE LOWER(name) = LOWER(?)", (person_name,))
                row = fs_cur.fetchone()
            if row:
                person_ids[row["id"]] = row["name"]
                print(f"✅ Персона найдена: {row['name']} (id={row['id']})")
            else:
                print(f"⚠️  Персона '{person_name}' не найдена в БД")
                # Показываем похожие имена
                fs_cur.execute("SELECT id, name FROM persons WHERE LOWER(name) LIKE LOWER(?) LIMIT 10", (f"%{person_name}%",))
                similar = fs_cur.fetchall()
                if similar:
                    print(f"   Похожие имена:")
                    for p in similar:
                        print(f"     - {p['name']} (id={p['id']})")
        
        if not person_ids:
            print("❌ Не найдено ни одной персоны. Выход.")
            return
        
        print(f"✅ Найдено персон: {list(person_ids.values())}")
        
        # Обрабатываем каждый файл
        total_fixed = 0
        for file_path in file_paths:
            # Сохраняем оригинальный путь для вывода
            original_path = file_path
            
            # Убираем префикс local: если есть для поиска
            clean_path = file_path[6:] if file_path.startswith("local:") else file_path
            
            # В БД файлы могут храниться как с префиксом local:, так и без него
            # Пробуем найти с префиксом и без
            db_path_with_prefix = f"local:{clean_path}" if not clean_path.startswith("local:") else clean_path
            db_path_without_prefix = clean_path
            
            print(f"\n📁 Файл: {original_path}")
            
            # Получаем file_id - пробуем оба варианта пути
            cur.execute("SELECT id FROM files WHERE path = ? OR path = ? LIMIT 1", (db_path_with_prefix, db_path_without_prefix))
            file_row = cur.fetchone()
            if not file_row:
                print(f"  ⚠️  Файл не найден в БД (пробовали: '{db_path_with_prefix}' и '{db_path_without_prefix}')")
                continue
            
            resolved_file_id = file_row["id"]
            
            # Получаем реальный путь из БД для информации
            cur.execute("SELECT path FROM files WHERE id = ?", (resolved_file_id,))
            file_info = cur.fetchone()
            print(f"  ✅ file_id: {resolved_file_id}, path в БД: {file_info['path']}")
            
            print(f"  ✅ file_id: {resolved_file_id}")
            
            # Находим все прямоугольники для этого файла с привязками к указанным персонам
            for person_id, person_name in person_ids.items():
                # Ищем прямоугольники через person_rectangle_manual_assignments
                fs_cur.execute("""
                    SELECT 
                        fr.id AS rectangle_id,
                        fr.is_face,
                        fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
                        fpma.person_id,
                        p.name AS person_name
                    FROM photo_rectangles fr
                    JOIN person_rectangle_manual_assignments fpma ON fr.id = fpma.rectangle_id
                    LEFT JOIN persons p ON p.id = fpma.person_id
                    WHERE fr.file_id = ? 
                      AND fpma.person_id = ?
                      AND fr.is_face = 1
                    ORDER BY fr.id
                """, (resolved_file_id, person_id))
                
                rows = fs_cur.fetchall()
                
                if not rows:
                    print(f"  ℹ️  Для персоны '{person_name}': прямоугольников с is_face=1 не найдено")
                    continue
                
                print(f"  👤 Персона '{person_name}': найдено {len(rows)} прямоугольников с is_face=1")
                
                for row in rows:
                    rect_id = row["rectangle_id"]
                    current_is_face = row["is_face"]
                    
                    print(f"    - rectangle_id={rect_id}, текущий is_face={current_is_face}, bbox=({row['bbox_x']},{row['bbox_y']},{row['bbox_w']},{row['bbox_h']})")
                    
                    if not dry_run:
                        # Обновляем is_face=0
                        fs_cur.execute("""
                            UPDATE photo_rectangles
                            SET is_face = 0
                            WHERE id = ?
                        """, (rect_id,))
                        print(f"      ✅ Обновлено: is_face=0")
                        total_fixed += 1
                    else:
                        print(f"      🔍 [DRY RUN] Будет обновлено: is_face=0")
                        total_fixed += 1
        
        if not dry_run:
            fs.conn.commit()
            print(f"\n✅ Всего исправлено: {total_fixed} прямоугольников")
        else:
            print(f"\n🔍 [DRY RUN] Всего будет исправлено: {total_fixed} прямоугольников")
            print("   Запустите с --apply для применения изменений")
    
    finally:
        fs.close()
        conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Исправляет is_face=0 для прямоугольников 'без лица'")
    parser.add_argument("--apply", action="store_true", help="Применить изменения (по умолчанию только dry-run)")
    parser.add_argument("--person", action="append", help="Имя персоны (можно указать несколько раз)")
    parser.add_argument("--person-id", action="append", type=int, help="ID персоны (можно указать несколько раз)")
    parser.add_argument("--file", action="append", required=True, help="Путь к файлу (можно указать несколько раз)")
    
    args = parser.parse_args()
    
    # Собираем имена и ID персон
    person_names = args.person or []
    person_ids = args.person_id or []
    
    if not person_names and not person_ids:
        parser.error("Необходимо указать хотя бы --person или --person-id")
    
    dry_run = not args.apply
    
    if dry_run:
        print("🔍 РЕЖИМ DRY-RUN (проверка без изменений)")
        print("   Используйте --apply для применения изменений\n")
    else:
        print("⚠️  РЕЖИМ ПРИМЕНЕНИЯ ИЗМЕНЕНИЙ\n")
    
    # Если указаны ID, используем их напрямую
    if person_ids:
        fs = FaceStore()
        try:
            fs_cur = fs.conn.cursor()
            person_names_from_ids = []
            for pid in person_ids:
                fs_cur.execute("SELECT id, name FROM persons WHERE id = ?", (pid,))
                row = fs_cur.fetchone()
                if row:
                    person_names_from_ids.append(row["name"])
                    print(f"✅ Персона по ID {pid}: {row['name']}")
                else:
                    print(f"⚠️  Персона с ID {pid} не найдена")
            person_names.extend(person_names_from_ids)
        finally:
            fs.close()
    
    fix_is_face_for_files_and_persons(
        file_paths=args.file,
        person_names=person_names,
        dry_run=dry_run
    )


if __name__ == "__main__":
    main()
