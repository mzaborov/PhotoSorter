#!/usr/bin/env python3
"""Проверка GPS координат в конкретном файле."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from common.db import DedupStore
from PIL import Image
import os

def _gps_to_deg(v):
    try:
        d = float(v[0])
        m = float(v[1])
        s = float(v[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None

def _try_exif_gps_latlon(img: Image.Image) -> tuple[float, float] | None:
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    gps = exif.get(34853)  # GPS IFD tag
    if not gps:
        return None
    try:
        lat = gps.get(2)      # GPSLatitude
        lat_ref = gps.get(1)  # GPSLatitudeRef: 'N' or 'S'
        lon = gps.get(4)       # GPSLongitude
        lon_ref = gps.get(3)  # GPSLongitudeRef: 'E' or 'W'
    except Exception:
        return None
    if not lat or not lon:
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

def _try_exif_datetime_iso(img: Image.Image) -> str | None:
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    dt = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime
    if not dt or not isinstance(dt, str):
        return None
    s = dt.strip()
    try:
        if len(s) >= 19 and s[4] == ":" and s[7] == ":" and s[10] in (" ", "T"):
            yyyy = s[0:4]
            mm = s[5:7]
            dd = s[8:10]
            hh = s[11:13]
            mi = s[14:16]
            ss = s[17:19]
            if yyyy.isdigit() and mm.isdigit() and dd.isdigit() and hh.isdigit() and mi.isdigit() and ss.isdigit():
                return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}Z"
    except Exception:
        return None
    return None

def main():
    file_path = r"C:\tmp\Photo\_no_faces\20240713_135315.jpg"
    
    if not os.path.isfile(file_path):
        print(f"Файл не найден: {file_path}")
        return
    
    print(f"Проверяю файл: {file_path}")
    print()
    
    # Проверяем EXIF напрямую
    try:
        with Image.open(file_path) as pil:
            gps = _try_exif_gps_latlon(pil)
            date = _try_exif_datetime_iso(pil)
            
            print("EXIF данные:")
            if gps:
                print(f"  GPS: {gps[0]:.6f}, {gps[1]:.6f}")
            else:
                print("  GPS: не найдено")
            
            if date:
                print(f"  Дата: {date}")
            else:
                print("  Дата: не найдена")
            print()
            
            # Проверяем, что в базе данных
            ds = DedupStore()
            try:
                cur = ds.conn.cursor()
                # Ищем файл по имени
                file_name = os.path.basename(file_path)
                cur.execute("""
                    SELECT path, gps_lat, gps_lon, place_country, place_city, taken_at
                    FROM files
                    WHERE name = ? AND status != 'deleted'
                    LIMIT 5
                """, (file_name,))
                rows = cur.fetchall()
                
                print("Данные в базе:")
                if rows:
                    for r in rows:
                        print(f"  Путь: {r[0]}")
                        print(f"  GPS в БД: lat={r[1]}, lon={r[2]}")
                        print(f"  Country: {r[3]}")
                        print(f"  City: {r[4]}")
                        print(f"  Taken_at: {r[5]}")
                        print()
                else:
                    print("  Файл не найден в базе данных")
            finally:
                ds.close()
            
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
