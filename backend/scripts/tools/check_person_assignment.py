#!/usr/bin/env python3
"""
Проверяет привязки персоны к файлу в БД.
"""
import sys
import argparse
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.common.db import FaceStore, get_connection


def check_person_assignment(file_path: str, person_name: str | None = None):
    """
    Проверяет привязки персоны к файлу.
    
    Args:
        file_path: Путь к файлу (может быть с префиксом local:)
        person_name: Имя персоны (опционально)
    """
    # Сохраняем оригинальный путь для вывода
    original_path = file_path
    
    # Убираем префикс local: если есть для поиска
    clean_path = file_path[6:] if file_path.startswith("local:") else file_path
    
    # В БД файлы могут храниться как с префиксом local:, так и без него
    # Пробуем найти с префиксом и без
    db_path_with_prefix = f"local:{clean_path}" if not clean_path.startswith("local:") else clean_path
    db_path_without_prefix = clean_path
    
    print(f"Проверяем файл: {original_path}")
    print(f"Персона: {person_name or 'любая'}")
    print("-" * 80)
    
    # Подключаемся к основной БД для получения file_id
    conn = get_connection()
    cur = conn.cursor()
    
    # Находим file_id - пробуем оба варианта пути
    cur.execute("SELECT id FROM files WHERE path = ? OR path = ?", (db_path_with_prefix, db_path_without_prefix))
    file_row = cur.fetchone()
    if not file_row:
        print(f"❌ Файл не найден в БД (пробовали: '{db_path_with_prefix}' и '{db_path_without_prefix}')")
        conn.close()
        return
    
    # Получаем реальный путь из БД для информации
    cur.execute("SELECT id, path FROM files WHERE id = ?", (file_row["id"],))
    file_info = cur.fetchone()
    print(f"✅ file_id: {file_info['id']}, path в БД: {file_info['path']}")
    
    file_id = file_row["id"]
    print(f"✅ file_id: {file_id}")
    conn.close()
    
    # Подключаемся к FaceStore
    fs = FaceStore()
    try:
        fs_cur = fs.conn.cursor()
        
        # Находим персону, если указано имя (поиск без учета регистра)
        person_id = None
        if person_name:
            # Сначала пробуем точное совпадение
            fs_cur.execute("SELECT id, name FROM persons WHERE name = ?", (person_name,))
            person_row = fs_cur.fetchone()
            if not person_row:
                # Если не найдено, пробуем без учета регистра
                fs_cur.execute("SELECT id, name FROM persons WHERE LOWER(name) = LOWER(?)", (person_name,))
                person_row = fs_cur.fetchone()
            if person_row:
                person_id = person_row["id"]
                print(f"✅ Персона найдена: {person_row['name']} (id={person_id})")
            else:
                print(f"⚠️  Персона не найдена по имени '{person_name}', но покажем все привязки для файла")
                print("   Доступные персоны с похожими именами:")
                fs_cur.execute("SELECT id, name FROM persons WHERE LOWER(name) LIKE LOWER(?) LIMIT 10", (f"%{person_name}%",))
                similar = fs_cur.fetchall()
                for p in similar:
                    print(f"     - {p['name']} (id={p['id']})")
        
        print("\n" + "=" * 80)
        print("1. ПРЯМОУГОЛЬНИКИ (photo_rectangles) для этого файла:")
        print("=" * 80)
        
        fs_cur.execute("""
            SELECT 
                pr.id,
                pr.is_face,
                pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h,
                pr.run_id,
                pr.archive_scope,
                pr.ignore_flag
            FROM photo_rectangles pr
            WHERE pr.file_id = ?
            ORDER BY pr.face_index, pr.id
        """, (file_id,))
        
        rectangles = fs_cur.fetchall()
        print(f"Найдено прямоугольников: {len(rectangles)}")
        for rect in rectangles:
            print(f"  - rectangle_id={rect['id']}, is_face={rect['is_face']}, "
                  f"bbox=({rect['bbox_x']}, {rect['bbox_y']}, {rect['bbox_w']}, {rect['bbox_h']}), "
                  f"run_id={rect['run_id']}, archive_scope={rect['archive_scope']}, "
                  f"ignore_flag={rect['ignore_flag']}")
        
        print("\n" + "=" * 80)
        print("2. РУЧНЫЕ ПРИВЯЗКИ (person_rectangle_manual_assignments):")
        print("=" * 80)
        
        query = """
            SELECT 
                fpma.rectangle_id,
                fpma.person_id,
                p.name as person_name,
                fpma.source,
                fpma.created_at,
                pr.is_face
            FROM person_rectangle_manual_assignments fpma
            JOIN photo_rectangles pr ON fpma.rectangle_id = pr.id
            JOIN persons p ON fpma.person_id = p.id
            WHERE pr.file_id = ?
        """
        params = [file_id]
        
        if person_id:
            query += " AND fpma.person_id = ?"
            params.append(person_id)
        
        query += " ORDER BY fpma.created_at DESC"
        
        fs_cur.execute(query, params)
        manual_assignments = fs_cur.fetchall()
        print(f"Найдено ручных привязок: {len(manual_assignments)}")
        for ma in manual_assignments:
            print(f"  - rectangle_id={ma['rectangle_id']}, person_id={ma['person_id']}, "
                  f"person_name='{ma['person_name']}', is_face={ma['is_face']}, "
                  f"source={ma['source']}, created_at={ma['created_at']}")
        
        print("\n" + "=" * 80)
        print("3. ПРИВЯЗКИ ЧЕРЕЗ КЛАСТЕРЫ (face_cluster_members -> face_clusters):")
        print("=" * 80)
        
        query = """
            SELECT 
                fcm.rectangle_id,
                fc.person_id,
                p.name as person_name,
                fc.id as cluster_id
            FROM face_cluster_members fcm
            JOIN face_clusters fc ON fcm.cluster_id = fc.id
            JOIN photo_rectangles pr ON fcm.rectangle_id = pr.id
            LEFT JOIN persons p ON fc.person_id = p.id
            WHERE pr.file_id = ?
              AND fc.person_id IS NOT NULL
        """
        params = [file_id]
        
        if person_id:
            query += " AND fc.person_id = ?"
            params.append(person_id)
        
        fs_cur.execute(query, params)
        cluster_assignments = fs_cur.fetchall()
        print(f"Найдено привязок через кластеры: {len(cluster_assignments)}")
        for ca in cluster_assignments:
            print(f"  - rectangle_id={ca['rectangle_id']}, person_id={ca['person_id']}, "
                  f"person_name='{ca['person_name']}', cluster_id={ca['cluster_id']}")
        
        print("\n" + "=" * 80)
        print("4. ПРЯМОУГОЛЬНИКИ БЕЗ ЛИЦА (person_rectangles):")
        print("=" * 80)
        
        query = """
            SELECT 
                pr.id as person_rectangle_id,
                pr.person_id,
                p.name as person_name,
                pr.pipeline_run_id,
                pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h,
                pr.created_at
            FROM person_rectangles pr
            JOIN persons p ON pr.person_id = p.id
            WHERE pr.file_id = ?
        """
        params = [file_id]
        
        if person_id:
            query += " AND pr.person_id = ?"
            params.append(person_id)
        
        query += " ORDER BY pr.created_at DESC"
        
        fs_cur.execute(query, params)
        person_rectangles = fs_cur.fetchall()
        print(f"Найдено person_rectangles: {len(person_rectangles)}")
        for pr in person_rectangles:
            print(f"  - person_rectangle_id={pr['person_rectangle_id']}, person_id={pr['person_id']}, "
                  f"person_name='{pr['person_name']}', pipeline_run_id={pr['pipeline_run_id']}, "
                  f"bbox=({pr['bbox_x']}, {pr['bbox_y']}, {pr['bbox_w']}, {pr['bbox_h']}), "
                  f"created_at={pr['created_at']}")
        
        print("\n" + "=" * 80)
        print("ИТОГО:")
        print("=" * 80)
        print(f"  - Всего прямоугольников: {len(rectangles)}")
        print(f"  - Ручных привязок: {len(manual_assignments)}")
        print(f"  - Привязок через кластеры: {len(cluster_assignments)}")
        print(f"  - person_rectangles: {len(person_rectangles)}")
        
        # Проверяем, есть ли прямоугольники с is_face=0
        non_face_rects = [r for r in rectangles if r['is_face'] == 0]
        if non_face_rects:
            print(f"\n⚠️  Найдено прямоугольников с is_face=0: {len(non_face_rects)}")
            for nfr in non_face_rects:
                # Проверяем, есть ли привязка для этого прямоугольника
                fs_cur.execute("""
                    SELECT person_id, p.name as person_name
                    FROM person_rectangle_manual_assignments fpma
                    JOIN persons p ON fpma.person_id = p.id
                    WHERE fpma.rectangle_id = ?
                """, (nfr['id'],))
                assignment = fs_cur.fetchone()
                if assignment:
                    print(f"    - rectangle_id={nfr['id']} привязан к персоне: {assignment['person_name']} (id={assignment['person_id']})")
                else:
                    print(f"    - rectangle_id={nfr['id']} НЕ привязан к персоне")
        
    finally:
        fs.close()


def main():
    parser = argparse.ArgumentParser(description="Проверяет привязки персоны к файлу в БД")
    parser.add_argument("file_path", help="Путь к файлу (может быть с префиксом local:)")
    parser.add_argument("--person", help="Имя персоны для фильтрации")
    
    args = parser.parse_args()
    check_person_assignment(args.file_path, args.person)


if __name__ == "__main__":
    main()
