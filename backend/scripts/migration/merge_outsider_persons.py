#!/usr/bin/env python3
"""Миграция: объединение персон 'Посторонний' (ID: 25) и 'Посторонние' (ID: 6)

Переносит все привязки с персоны ID 25 на персону ID 6, затем удаляет персону ID 25.
"""
import sqlite3
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("data/photosorter.db")
SOURCE_PERSON_ID = 26  # "Посторонние" - откуда переносим
TARGET_PERSON_ID = 6   # "Посторонний" - куда переносим

def main():
    parser = argparse.ArgumentParser(description='Объединение персон Посторонний')
    parser.add_argument('--yes', action='store_true', help='Автоматическое подтверждение')
    args = parser.parse_args()
    if not DB_PATH.exists():
        print(f"❌ БД не найдена: {DB_PATH}")
        return 1
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    print("=" * 60)
    print("МИГРАЦИЯ: Объединение персон 'Посторонний'")
    print("=" * 60)
    
    # Проверяем, что персоны существуют
    cur.execute("SELECT id, name FROM persons WHERE id IN (?, ?)", (SOURCE_PERSON_ID, TARGET_PERSON_ID))
    persons = {row['id']: row['name'] for row in cur.fetchall()}
    
    if SOURCE_PERSON_ID not in persons:
        print(f"❌ Персона ID {SOURCE_PERSON_ID} не найдена")
        conn.close()
        return 1
    
    if TARGET_PERSON_ID not in persons:
        print(f"❌ Персона ID {TARGET_PERSON_ID} не найдена")
        conn.close()
        return 1
    
    print(f"\n📋 Персоны:")
    print(f"  Источник: ID {SOURCE_PERSON_ID} - '{persons[SOURCE_PERSON_ID]}'")
    print(f"  Цель: ID {TARGET_PERSON_ID} - '{persons[TARGET_PERSON_ID]}'")
    
    # Подсчитываем привязки
    print("\n📊 Подсчет привязок:")
    
    # Ручные привязки (person_rectangle_manual_assignments)
    cur.execute("SELECT COUNT(*) as cnt FROM person_rectangle_manual_assignments WHERE person_id = ?", (SOURCE_PERSON_ID,))
    manual_count = cur.fetchone()['cnt']
    print(f"  Ручные привязки (person_rectangle_manual_assignments): {manual_count}")
    
    # Привязки через кластеры (face_clusters)
    cur.execute("SELECT COUNT(*) as cnt FROM face_clusters WHERE person_id = ?", (SOURCE_PERSON_ID,))
    cluster_count = cur.fetchone()['cnt']
    print(f"  Кластеры (face_clusters): {cluster_count}")
    
    # Привязки через person_rectangles
    cur.execute("SELECT COUNT(*) as cnt FROM person_rectangles WHERE person_id = ?", (SOURCE_PERSON_ID,))
    person_rect_count = cur.fetchone()['cnt']
    print(f"  Прямоугольники персон (person_rectangles): {person_rect_count}")
    
    # Привязки через file_persons
    cur.execute("SELECT COUNT(*) as cnt FROM file_persons WHERE person_id = ?", (SOURCE_PERSON_ID,))
    file_persons_count = cur.fetchone()['cnt']
    print(f"  Привязки файлов (file_persons): {file_persons_count}")
    
    total = manual_count + cluster_count + person_rect_count + file_persons_count
    print(f"\n  Всего привязок для переноса: {total}")
    
    if total == 0:
        print("\n✅ Нет привязок для переноса. Можно удалить персону ID", SOURCE_PERSON_ID)
        if not args.yes:
            confirm = input("\nУдалить персону ID {}? (yes/no): ".format(SOURCE_PERSON_ID))
            if confirm.lower() != 'yes':
                print("Отменено")
                conn.close()
                return 0
        
        cur.execute("DELETE FROM persons WHERE id = ?", (SOURCE_PERSON_ID,))
        conn.commit()
        print(f"✅ Персона ID {SOURCE_PERSON_ID} удалена")
        conn.close()
        return 0
    
    # Подтверждение
    print(f"\n⚠️  Будет перенесено {total} привязок с персоны ID {SOURCE_PERSON_ID} на персону ID {TARGET_PERSON_ID}")
    if not args.yes:
        confirm = input("Продолжить? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Отменено")
            conn.close()
            return 0
    
    # Создаем резервную копию
    backup_path = DB_PATH.parent / f"photosorter_backup_before_merge_outsider_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    print(f"\n💾 Создаю резервную копию: {backup_path}")
    import shutil
    shutil.copy2(DB_PATH, backup_path)
    print("✅ Резервная копия создана")
    
    # Начинаем транзакцию
    print("\n🔄 Начинаю перенос привязок...")
    
    try:
        # 1. Переносим ручные привязки
        if manual_count > 0:
            print(f"  Переносим {manual_count} ручных привязок...")
            cur.execute("""
                UPDATE person_rectangle_manual_assignments 
                SET person_id = ? 
                WHERE person_id = ?
            """, (TARGET_PERSON_ID, SOURCE_PERSON_ID))
            print(f"  ✅ Перенесено {cur.rowcount} ручных привязок")
        
        # 2. Переносим кластеры
        if cluster_count > 0:
            print(f"  Переносим {cluster_count} кластеров...")
            cur.execute("""
                UPDATE face_clusters 
                SET person_id = ? 
                WHERE person_id = ?
            """, (TARGET_PERSON_ID, SOURCE_PERSON_ID))
            print(f"  ✅ Перенесено {cur.rowcount} кластеров")
        
        # 3. Переносим person_rectangles
        if person_rect_count > 0:
            print(f"  Переносим {person_rect_count} прямоугольников персон...")
            cur.execute("""
                UPDATE person_rectangles 
                SET person_id = ? 
                WHERE person_id = ?
            """, (TARGET_PERSON_ID, SOURCE_PERSON_ID))
            print(f"  ✅ Перенесено {cur.rowcount} прямоугольников персон")
        
        # 4. Переносим file_persons
        if file_persons_count > 0:
            print(f"  Переносим {file_persons_count} привязок файлов...")
            cur.execute("""
                UPDATE file_persons 
                SET person_id = ? 
                WHERE person_id = ?
            """, (TARGET_PERSON_ID, SOURCE_PERSON_ID))
            print(f"  ✅ Перенесено {cur.rowcount} привязок файлов")
        
        # 5. Удаляем старую персону
        print(f"\n🗑️  Удаляю персону ID {SOURCE_PERSON_ID}...")
        cur.execute("DELETE FROM persons WHERE id = ?", (SOURCE_PERSON_ID,))
        print(f"  ✅ Персона ID {SOURCE_PERSON_ID} удалена")
        
        # Коммитим транзакцию
        conn.commit()
        print("\n✅ Миграция завершена успешно!")
        
        # Проверяем результат
        print("\n📊 Проверка результата:")
        cur.execute("SELECT COUNT(*) as cnt FROM person_rectangle_manual_assignments WHERE person_id = ?", (TARGET_PERSON_ID,))
        final_manual = cur.fetchone()['cnt']
        cur.execute("SELECT COUNT(*) as cnt FROM face_clusters WHERE person_id = ?", (TARGET_PERSON_ID,))
        final_cluster = cur.fetchone()['cnt']
        cur.execute("SELECT COUNT(*) as cnt FROM person_rectangles WHERE person_id = ?", (TARGET_PERSON_ID,))
        final_person_rect = cur.fetchone()['cnt']
        cur.execute("SELECT COUNT(*) as cnt FROM file_persons WHERE person_id = ?", (TARGET_PERSON_ID,))
        final_file_persons = cur.fetchone()['cnt']
        
        print(f"  Персона ID {TARGET_PERSON_ID} теперь имеет:")
        print(f"    Ручные привязки: {final_manual}")
        print(f"    Кластеры: {final_cluster}")
        print(f"    Прямоугольники персон: {final_person_rect}")
        print(f"    Привязки файлов: {final_file_persons}")
        
        # Проверяем, что старая персона удалена
        cur.execute("SELECT COUNT(*) as cnt FROM persons WHERE id = ?", (SOURCE_PERSON_ID,))
        old_exists = cur.fetchone()['cnt']
        if old_exists > 0:
            print(f"\n⚠️  ВНИМАНИЕ: Персона ID {SOURCE_PERSON_ID} все еще существует!")
        else:
            print(f"\n✅ Персона ID {SOURCE_PERSON_ID} успешно удалена")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Ошибка при миграции: {e}")
        print("Откат изменений...")
        conn.close()
        return 1
    
    conn.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
