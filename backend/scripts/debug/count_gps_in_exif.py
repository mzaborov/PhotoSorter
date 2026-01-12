#!/usr/bin/env python3
"""Подсчёт файлов с GPS в EXIF."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Ошибка: PIL/Pillow не установлен")
    sys.exit(1)

from common.db import DedupStore, PipelineStore

def _try_exif_gps_latlon(img: Image.Image) -> tuple[float, float] | None:
    """Извлекает GPS координаты из EXIF."""
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    try:
        gps = exif.get_ifd(0x8825)
    except Exception:
        gps = exif.get(34853)
    if not gps:
        return None
    try:
        lat = gps.get(2)
        lat_ref = gps.get(1)
        lon = gps.get(4)
        lon_ref = gps.get(3)
    except Exception:
        return None
    if not lat or not lon:
        return None
    def _gps_to_deg(v):
        try:
            d = float(v[0])
            m = float(v[1])
            s = float(v[2])
            return d + (m / 60.0) + (s / 3600.0)
        except Exception:
            return None
    lat_deg = _gps_to_deg(lat)
    lon_deg = _gps_to_deg(lon)
    if lat_deg is None or lon_deg is None:
        return None
    if isinstance(lat_ref, str) and lat_ref.upper() == "S":
        lat_deg = -lat_deg
    if isinstance(lon_ref, str) and lon_ref.upper() == "W":
        lon_deg = -lon_deg
    return (float(lat_deg), float(lon_deg))

def main():
    ps = PipelineStore()
    try:
        cur = ps.conn.cursor()
        cur.execute("""
            SELECT id, face_run_id, root_path
            FROM pipeline_runs
            WHERE kind = 'local_sort'
            ORDER BY id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            print("Нет прогонов в базе")
            return
        pr = {"id": row[0], "face_run_id": row[1], "root_path": row[2]}
        
        face_run_id = pr.get("face_run_id")
        root_path = str(pr.get("root_path") or "")
        
        print(f"Прогон: {pr['id']}")
        print(f"face_run_id: {face_run_id}")
        print(f"root_path: {root_path}")
        print()
        
        if not face_run_id:
            print("face_run_id не установлен")
            return
        
        ds = DedupStore()
        try:
            cur = ds.conn.cursor()
            cur.execute("""
                SELECT path
                FROM files
                WHERE faces_run_id = ? AND status != 'deleted'
                ORDER BY path
            """, (int(face_run_id),))
            rows = cur.fetchall()
            
            total = len(rows)
            print(f"Всего файлов в базе: {total}")
            print()
            
            gps_in_exif = 0
            gps_in_db = 0
            checked = 0
            errors = 0
            
            for i, r in enumerate(rows, 1):
                path = str(r[0] or "")
                
                if i % 1000 == 0:
                    print(f"Проверено: {i}/{total} (GPS в EXIF: {gps_in_exif}, GPS в БД: {gps_in_db}, ошибки: {errors})")
                
                # Проверяем GPS в БД
                cur2 = ds.conn.cursor()
                cur2.execute("""
                    SELECT gps_lat, gps_lon
                    FROM files
                    WHERE path = ?
                """, (path,))
                row_db = cur2.fetchone()
                if row_db and row_db[0] is not None and row_db[1] is not None:
                    gps_in_db += 1
                
                # Преобразуем путь в локальный
                if path.startswith("local:"):
                    local_path = path[6:]
                elif path.startswith("disk:"):
                    continue  # Пропускаем YaDisk
                else:
                    local_path = path
                
                if not local_path or not os.path.isfile(local_path):
                    continue
                
                # Проверяем GPS в EXIF
                try:
                    with Image.open(local_path) as img:
                        gps = _try_exif_gps_latlon(img)
                        if gps and len(gps) >= 2:
                            gps_in_exif += 1
                    checked += 1
                except Exception:
                    errors += 1
                    continue
            
            print()
            print(f"Итого:")
            print(f"  Проверено файлов: {checked}")
            print(f"  С GPS в EXIF: {gps_in_exif} ({100.0 * gps_in_exif / max(1, checked):.1f}%)")
            print(f"  С GPS в БД: {gps_in_db} ({100.0 * gps_in_db / max(1, total):.1f}%)")
            print(f"  Ошибок: {errors}")
        
        finally:
            ds.close()
    finally:
        ps.close()

if __name__ == "__main__":
    main()
