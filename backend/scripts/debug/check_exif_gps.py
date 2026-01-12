#!/usr/bin/env python3
"""Проверка GPS координат в EXIF файлов."""
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Ошибка: не установлен PIL/Pillow")
    sys.exit(1)
import os

def _gps_to_deg(gps_tuple):
    """Конвертирует GPS tuple (degrees, minutes, seconds) в десятичные градусы."""
    if not gps_tuple or len(gps_tuple) < 3:
        return None
    try:
        deg = float(gps_tuple[0])
        minutes = float(gps_tuple[1])
        seconds = float(gps_tuple[2])
        return deg + minutes / 60.0 + seconds / 3600.0
    except Exception:
        return None

def _try_exif_gps_latlon(img: Image.Image) -> tuple[float, float] | None:
    """Извлекает GPS координаты из EXIF."""
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
    return (lat_deg, lon_deg)

def _try_exif_datetime_iso(img: Image.Image) -> str | None:
    """Извлекает дату из EXIF."""
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    # DateTimeOriginal (tag 36867)
    dt = exif.get(36867)
    if dt:
        return str(dt)
    # DateTime (tag 306)
    dt = exif.get(306)
    if dt:
        return str(dt)
    return None

def main():
    root_path = r"C:\tmp\Photo"
    if not os.path.isdir(root_path):
        print(f"Директория не найдена: {root_path}")
        return
    
    # Проверяем первые 30 файлов
    checked = 0
    with_gps = 0
    without_gps = 0
    with_date = 0
    
    print(f"Проверяю файлы в {root_path}...")
    print()
    
    for root, dirs, files in os.walk(root_path):
        # Пропускаем служебные папки
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        
        for file in files:
            if checked >= 30:
                break
            
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue
            
            # Проверяем только изображения
            ext = os.path.splitext(file)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.heic', '.heif'):
                continue
            
            checked += 1
            try:
                with Image.open(file_path) as pil:
                    gps = _try_exif_gps_latlon(pil)
                    date = _try_exif_datetime_iso(pil)
                    
                    rel_path = os.path.relpath(file_path, root_path)
                    
                    if gps and len(gps) >= 2 and gps[0] is not None and gps[1] is not None:
                        with_gps += 1
                        print(f"✓ {rel_path}")
                        print(f"  GPS: {gps[0]:.6f}, {gps[1]:.6f}")
                        if date:
                            print(f"  Дата: {date}")
                            with_date += 1
                        print()
                    else:
                        without_gps += 1
                        if checked <= 10:  # Показываем первые 10 без GPS
                            print(f"✗ {rel_path} - нет GPS", end="")
                            if date:
                                print(f" (дата: {date})")
                                with_date += 1
                            else:
                                print()
            except Exception as e:
                without_gps += 1
                if checked <= 5:
                    rel_path = os.path.relpath(file_path, root_path)
                    print(f"✗ {rel_path} - ошибка: {e}")
        
        if checked >= 30:
            break
    
    print()
    print(f"Проверено файлов: {checked}")
    print(f"  С GPS: {with_gps} ({with_gps * 100.0 / checked if checked > 0 else 0:.1f}%)")
    print(f"  Без GPS: {without_gps} ({without_gps * 100.0 / checked if checked > 0 else 0:.1f}%)")
    print(f"  С датой: {with_date} ({with_date * 100.0 / checked if checked > 0 else 0:.1f}%)")

if __name__ == "__main__":
    main()
