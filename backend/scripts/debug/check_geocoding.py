#!/usr/bin/env python3
"""Проверка наличия данных геокодирования в базе."""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "backend"))

from common.db import DedupStore, PipelineStore

def main():
    ps = PipelineStore()
    try:
        # Получаем последний прогон
        cur = ps.conn.cursor()
        cur.execute("""
            SELECT id, face_run_id, root_path, status
            FROM pipeline_runs
            WHERE kind = 'local_sort'
            ORDER BY id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            print("Нет прогонов в базе")
            return
        
        pipeline_run_id = row[0]
        face_run_id = row[1]
        root_path = row[2]
        status = row[3]
        
        print(f"Последний прогон: {pipeline_run_id} (статус: {status})")
        print(f"face_run_id: {face_run_id}")
        print(f"root_path: {root_path}")
        print()
        
        ds = DedupStore()
        try:
            cur = ds.conn.cursor()
            # Статистика по геокодированию
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN place_country IS NOT NULL AND place_country != '' THEN 1 END) as with_country,
                    COUNT(CASE WHEN place_city IS NOT NULL AND place_city != '' THEN 1 END) as with_city,
                    COUNT(CASE WHEN taken_at IS NOT NULL AND taken_at != '' THEN 1 END) as with_date,
                    COUNT(CASE WHEN gps_lat IS NOT NULL AND gps_lon IS NOT NULL THEN 1 END) as with_gps
                FROM files
                WHERE faces_run_id = ? AND status != 'deleted'
            """, (face_run_id,))
            stats = cur.fetchone()
            
            print("Статистика по файлам:")
            print(f"  Всего файлов: {stats[0]}")
            print(f"  С country: {stats[1]}")
            print(f"  С city: {stats[2]}")
            print(f"  С датой (taken_at): {stats[3]}")
            print(f"  С GPS координатами: {stats[4]}")
            print()
            
            # Проверяем конкретный файл
            file_name = "20240713_135315.jpg"
            cur.execute("""
                SELECT path, place_country, place_city, taken_at, gps_lat, gps_lon, faces_run_id
                FROM files
                WHERE name = ? AND status != 'deleted'
                LIMIT 5
            """, (file_name,))
            rows = cur.fetchall()
            
            if rows:
                print(f"Найден файл {file_name}:")
                for r in rows:
                    print(f"  Путь: {r[0]}")
                    print(f"  faces_run_id: {r[6]}")
                    print(f"  GPS в БД: lat={r[4]}, lon={r[5]}")
                    print(f"  Country: {r[1]}")
                    print(f"  City: {r[2]}")
                    print(f"  Taken_at: {r[3]}")
                    print()
            else:
                print(f"Файл {file_name} не найден в базе данных")
                print()
            
            # Примеры файлов с GPS (даже без геокодирования)
            cur.execute("""
                SELECT path, place_country, place_city, taken_at, gps_lat, gps_lon
                FROM files
                WHERE faces_run_id = ? 
                  AND gps_lat IS NOT NULL 
                  AND gps_lon IS NOT NULL
                  AND status != 'deleted'
                LIMIT 10
            """, (face_run_id,))
            rows = cur.fetchall()
            
            if rows:
                print("Примеры файлов с GPS координатами (первые 10):")
                for r in rows:
                    print(f"  {r[0]}")
                    print(f"    GPS: {r[4]}, {r[5]}")
                    print(f"    country: {r[1]}, city: {r[2]}")
                    print(f"    taken_at: {r[3]}")
                    print()
            else:
                print("Файлов с GPS координатами не найдено")
                print()
                print("Проверяю файлы с GPS, но без геокодирования:")
                cur.execute("""
                    SELECT path, gps_lat, gps_lon, taken_at
                    FROM files
                    WHERE faces_run_id = ? 
                      AND gps_lat IS NOT NULL 
                      AND gps_lon IS NOT NULL
                      AND (place_country IS NULL OR place_country = '')
                      AND status != 'deleted'
                    LIMIT 5
                """, (face_run_id,))
                gps_rows = cur.fetchall()
                if gps_rows:
                    print(f"Найдено {len(gps_rows)} файлов с GPS, но без геокодирования:")
                    for r in gps_rows:
                        print(f"  {r[0]}: GPS {r[1]}, {r[2]}, дата: {r[3]}")
                else:
                    print("Файлов с GPS не найдено")
            
        finally:
            ds.close()
    finally:
        ps.close()

if __name__ == "__main__":
    main()
