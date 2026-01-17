#!/usr/bin/env python3
"""Скачивает модель ArcFace с рабочего источника."""

import sys
import urllib.request
import zipfile
import tempfile
import os
from pathlib import Path

target_dir = Path(__file__).resolve().parents[3] / "models" / "face_recognition"
target_dir.mkdir(parents=True, exist_ok=True)
target_path = target_dir / "w600k_r50.onnx"

if target_path.exists():
    size_mb = target_path.stat().st_size / 1024 / 1024
    if size_mb > 10:
        print(f"✓ Модель уже существует: {target_path}")
        print(f"  Размер: {size_mb:.1f} MB")
        sys.exit(0)

print("Скачивание модели ArcFace...")
print(f"Целевой путь: {target_path}")

# Пробуем скачать архив buffalo_l и извлечь модель
url = "https://github.com/deepinsight/insightface/releases/download/v0.7.3/buffalo_l.zip"

print(f"\nСкачивание архива: {url}")
print("Это может занять несколько минут (~100MB)")

try:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        tmp_zip_path = tmp_zip.name
    
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = "=" * filled + "-" * (bar_length - filled)
            print(f"\r  [{bar}] {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, tmp_zip_path, reporthook=progress_hook)
    print()
    
    # Распаковываем архив
    print("Распаковка архива...")
    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
        # Ищем w600k_r50.onnx в архиве
        for name in zip_ref.namelist():
            if 'w600k_r50.onnx' in name or name.endswith('w600k_r50.onnx'):
                print(f"Найден файл: {name}")
                with zip_ref.open(name) as source:
                    with open(target_path, 'wb') as target:
                        target.write(source.read())
                print(f"✓ Модель извлечена: {target_path}")
                size_mb = target_path.stat().st_size / 1024 / 1024
                print(f"  Размер: {size_mb:.1f} MB")
                os.unlink(tmp_zip_path)
                sys.exit(0)
        
        # Если не нашли напрямую, пробуем найти в подпапках
        print("Поиск модели в архиве...")
        for name in zip_ref.namelist():
            if name.endswith('.onnx'):
                print(f"  Найден ONNX файл: {name}")
                if 'r50' in name.lower() or 'arcface' in name.lower():
                    print(f"  Используем: {name}")
                    with zip_ref.open(name) as source:
                        with open(target_path, 'wb') as target:
                            target.write(source.read())
                    print(f"✓ Модель извлечена: {target_path}")
                    size_mb = target_path.stat().st_size / 1024 / 1024
                    print(f"  Размер: {size_mb:.1f} MB")
                    os.unlink(tmp_zip_path)
                    sys.exit(0)
    
    print("⚠ Модель w600k_r50.onnx не найдена в архиве")
    os.unlink(tmp_zip_path)
    
except urllib.error.HTTPError as e:
    print(f"\n✗ HTTP ошибка: {e.code} {e.reason}")
    if e.code == 404:
        print("Архив не найден по этой ссылке")
except Exception as e:
    print(f"\n✗ Ошибка: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n✗ Не удалось скачать модель автоматически")
print("\nАльтернативный способ:")
print("1. Откройте: https://github.com/deepinsight/insightface/releases")
print("2. Найдите и скачайте архив с моделью")
print(f"3. Извлеките w600k_r50.onnx в: {target_path}")
sys.exit(1)
