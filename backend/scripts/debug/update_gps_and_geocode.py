#!/usr/bin/env python3
"""Обновление GPS координат и геокодирования для уже обработанных файлов."""
import sys
import os
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path

# Загружаем переменные окружения из secrets.env
try:
    from dotenv import load_dotenv
    load_dotenv("secrets.env")
    load_dotenv(".env")  # fallback
except ImportError:
    pass  # dotenv не обязателен

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from common.db import DedupStore, PipelineStore
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Предупреждение: PIL/Pillow не установлен. Установите: pip install Pillow")
    print("Скрипт будет работать только для обновления геокодирования файлов с уже имеющимися GPS в базе.")
import argparse

def _strip_local_prefix(p: str) -> str:
    """Убирает префикс 'local:' из пути."""
    if p.startswith("local:"):
        return p[6:]
    return p

def _gps_to_deg(v):
    try:
        d = float(v[0])
        m = float(v[1])
        s = float(v[2])
        return d + (m / 60.0) + (s / 3600.0)
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
    # Получаем GPS IFD через get_ifd (правильный способ в Pillow)
    try:
        gps = exif.get_ifd(0x8825)  # GPS IFD tag (34853 в десятичной системе)
    except Exception:
        # Fallback: пробуем через get
        gps = exif.get(34853)
    if not gps:
        return None
    try:
        lat = gps.get(2)      # GPSLatitude
        lat_ref = gps.get(1)   # GPSLatitudeRef: 'N' or 'S'
        lon = gps.get(4)       # GPSLongitude
        lon_ref = gps.get(3)   # GPSLongitudeRef: 'E' or 'W'
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
    """Извлекает дату из EXIF."""
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

def _geo_cache_key(lat: float, lon: float) -> str:
    """Формирует ключ кэша для GPS координат."""
    return f"{float(lat):.4f},{float(lon):.4f}"

def _yandex_country_city(lat: float, lon: float, y_key: str, y_url: str, y_lang: str) -> tuple[str | None, str | None, str | None]:
    """Геокодирование через Yandex API."""
    if not y_key:
        return (None, None, None)
    try:
        q = {
            "apikey": y_key,
            "geocode": f"{float(lon)},{float(lat)}",
            "format": "json",
            "results": "1",
            "lang": y_lang,
        }
        url = str(y_url).rstrip("?") + "?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers={"User-Agent": "PhotoSorter/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read()
        txt = raw.decode("utf-8", errors="replace")
        obj = json.loads(txt)
        comps = (
            obj.get("response", {})
            .get("GeoObjectCollection", {})
            .get("featureMember", [{}])[0]
            .get("GeoObject", {})
            .get("metaDataProperty", {})
            .get("GeocoderMetaData", {})
            .get("Address", {})
            .get("Components", [])
        )
        country = None
        city = None
        if isinstance(comps, list):
            for c in comps:
                if not isinstance(c, dict):
                    continue
                kind = str(c.get("kind") or "").strip().lower()
                name = str(c.get("name") or "").strip()
                if not name:
                    continue
                if kind == "country" and not country:
                    country = name
                if kind == "locality" and not city:
                    city = name
                if kind == "province" and not city:
                    city = name
        return (country, city, txt)
    except Exception:
        return (None, None, None)

def main():
    parser = argparse.ArgumentParser(description="Обновление GPS координат и геокодирования")
    parser.add_argument("--pipeline-run-id", type=int, help="ID прогона (если не указан, используется последний)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет обновлено")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество файлов для обработки")
    args = parser.parse_args()
    
    # Настройки Yandex Geocoder
    y_key = (os.getenv("YANDEX_GEOCODER_API_KEY") or "").strip()
    y_url = (os.getenv("YANDEX_GEOCODER_URL") or "https://geocode-maps.yandex.ru/1.x/").strip()
    y_lang = (os.getenv("YANDEX_GEOCODER_LANG") or "ru_RU").strip()
    try:
        y_rps = float(os.getenv("YANDEX_GEOCODER_RPS") or 3.0)
    except Exception:
        y_rps = 3.0
    y_last_call = 0.0
    
    def _rate_limit_geocode() -> None:
        nonlocal y_last_call
        if not y_key or y_rps <= 0:
            return
        now = time.time()
        min_dt = 1.0 / float(y_rps)
        dt = now - float(y_last_call)
        if dt < min_dt:
            time.sleep(float(min_dt - dt))
        y_last_call = time.time()
    
    ps = PipelineStore()
    try:
        if args.pipeline_run_id:
            pr = ps.get_run_by_id(run_id=args.pipeline_run_id)
        else:
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
            pr = {"id": row[0], "face_run_id": row[1], "root_path": row[2], "status": row[3]}
        
        if not pr:
            print("Прогон не найден")
            return
        
        pipeline_run_id = pr["id"]
        face_run_id = pr.get("face_run_id")
        root_path = str(pr.get("root_path") or "")
        
        print(f"Прогон: {pipeline_run_id} (статус: {pr.get('status')})")
        print(f"face_run_id: {face_run_id}")
        print(f"root_path: {root_path}")
        print()
        
        if not face_run_id:
            print("face_run_id не установлен (шаг 3 не выполнен)")
            return
        
        face_run_id_i = int(face_run_id)
        
        if not y_key:
            print("Предупреждение: YANDEX_GEOCODER_API_KEY не установлен, геокодирование не будет выполняться")
        
        ds = DedupStore()
        try:
            cur = ds.conn.cursor()
            # Получаем все файлы для этого прогона
            cur.execute("""
                SELECT path, name, parent_path, gps_lat, gps_lon, place_country, place_city, taken_at
                FROM files
                WHERE faces_run_id = ? AND status != 'deleted'
                ORDER BY path
            """, (face_run_id_i,))
            rows = cur.fetchall()
            
            total = len(rows)
            print(f"Найдено файлов: {total}")
            if args.limit:
                rows = rows[:args.limit]
                print(f"Ограничение: обработаем первые {len(rows)} файлов")
            print()
            
            updated_gps = 0
            updated_geocode = 0
            errors = 0
            skipped_not_found = 0
            
            for i, r in enumerate(rows, 1):
                path = str(r[0] or "")
                name = str(r[1] or "")
                parent_path = str(r[2] or "")
                gps_lat_db = r[3]
                gps_lon_db = r[4]
                place_country_db = r[5]
                place_city_db = r[6]
                
                if i % 100 == 0:
                    print(f"Обработано: {i}/{len(rows)} (GPS: {updated_gps}, геокодирование: {updated_geocode}, ошибки: {errors}, не найдено: {skipped_not_found})")
                
                # Преобразуем путь в локальный
                if path.startswith("local:"):
                    local_path = _strip_local_prefix(path)
                elif path.startswith("disk:"):
                    # Файлы на YaDisk пропускаем (нужно скачивать)
                    skipped_not_found += 1
                    continue
                else:
                    local_path = path
                
                if not local_path or not os.path.isfile(local_path):
                    # Файл не найден - пропускаем
                    skipped_not_found += 1
                    continue
                
                # Проверяем, нужно ли обновлять GPS
                need_gps_update = (gps_lat_db is None or gps_lon_db is None)
                need_geocode = (gps_lat_db is not None and gps_lon_db is not None) and (place_country_db is None or place_country_db == "")
                
                if not need_gps_update and not need_geocode:
                    # Уже всё есть - пропускаем
                    continue
                
                # Извлекаем GPS из EXIF (если PIL доступен)
                gps_lat = None
                gps_lon = None
                exif_taken = None
                
                if need_gps_update and PIL_AVAILABLE:
                    try:
                        with Image.open(local_path) as pil:
                            gps = _try_exif_gps_latlon(pil)
                            exif_taken = _try_exif_datetime_iso(pil)
                            
                            if gps and len(gps) >= 2:
                                gps_lat = float(gps[0])
                                gps_lon = float(gps[1])
                    except Exception as e:
                        errors += 1
                        if errors <= 10:
                            print(f"  [ERROR] {name}: ошибка чтения EXIF {type(e).__name__}: {e}")
                
                # Используем GPS из базы, если EXIF не удалось прочитать
                if gps_lat is None or gps_lon is None:
                    if gps_lat_db is not None and gps_lon_db is not None:
                        gps_lat = float(gps_lat_db)
                        gps_lon = float(gps_lon_db)
                    else:
                        # GPS нет ни в EXIF, ни в базе - пропускаем
                        continue
                
                # Обновляем GPS в базе, если извлекли из EXIF
                if need_gps_update and gps_lat is not None and gps_lon is not None:
                    if not args.dry_run:
                        ds.set_taken_at_and_gps(
                            path=path,
                            taken_at=exif_taken,
                            gps_lat=gps_lat,
                            gps_lon=gps_lon
                        )
                    updated_gps += 1
                    if updated_gps <= 10:  # Показываем первые 10
                        print(f"  [GPS] {name}: GPS {gps_lat:.6f}, {gps_lon:.6f}")
                    # После обновления GPS нужно выполнить геокодирование
                    need_geocode = True
                
                # Геокодирование, если нужно и есть API ключ
                if need_geocode and y_key and gps_lat is not None and gps_lon is not None:
                    k = _geo_cache_key(gps_lat, gps_lon)
                    cached = ds.geocode_cache_get(key=k)
                    
                    if cached and (cached.get("country") or cached.get("city")):
                        # Используем кэш
                        ctry = str(cached.get("country") or "") or None
                        city = str(cached.get("city") or "") or None
                        if not args.dry_run:
                            ds.set_place(
                                path=path,
                                country=ctry,
                                city=city,
                                source=str(cached.get("source") or "yandex")
                            )
                        updated_geocode += 1
                        if updated_geocode <= 10:
                            print(f"    [GEO] {ctry}, {city} (из кэша)")
                    else:
                        # Выполняем геокодирование
                        try:
                            _rate_limit_geocode()
                            ctry, city, rawj = _yandex_country_city(gps_lat, gps_lon, y_key, y_url, y_lang)
                            if ctry or city:
                                if not args.dry_run:
                                    ds.geocode_cache_upsert(
                                        key=k,
                                        lat=gps_lat,
                                        lon=gps_lon,
                                        country=ctry,
                                        city=city,
                                        source="yandex",
                                        raw_json=rawj
                                    )
                                    ds.set_place(
                                        path=path,
                                        country=ctry,
                                        city=city,
                                        source="yandex"
                                    )
                                updated_geocode += 1
                                if updated_geocode <= 10:
                                    print(f"    [GEO] {ctry}, {city} (геокодировано)")
                        except Exception as geocode_err:
                                        errors += 1
                                        if errors <= 10:
                                            print(f"    [ERROR] Ошибка геокодирования: {geocode_err}")
            
            print()
            print(f"Итого:")
            print(f"  Обновлено GPS: {updated_gps}")
            print(f"  Обновлено геокодирование: {updated_geocode}")
            print(f"  Ошибок: {errors}")
            print(f"  Файлов не найдено/пропущено: {skipped_not_found}")
            if args.dry_run:
                print()
                print("(DRY-RUN: изменения не сохранены)")
        
        finally:
            ds.close()
    finally:
        ps.close()

if __name__ == "__main__":
    main()
