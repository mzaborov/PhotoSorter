#!/usr/bin/env python3
"""Скачивает модель через gdown (Google Drive)."""

import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Установка gdown...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

target_dir = Path(__file__).resolve().parents[3] / "models" / "face_recognition"
target_dir.mkdir(parents=True, exist_ok=True)
target_path = target_dir / "w600k_r50.onnx"

if target_path.exists():
    size_mb = target_path.stat().st_size / 1024 / 1024
    print(f"✓ Модель уже существует: {target_path}")
    print(f"  Размер: {size_mb:.1f} MB")
    if size_mb > 10:
        sys.exit(0)

# Google Drive ID для модели (нужно найти правильный)
# Пробуем известные ID
drive_ids = [
    # Это примеры, нужно найти реальные ID
    "1H37LK9FfzNz4qX9y7jS6_P9V4vJ8xQ",  # Пример (не работает)
]

print("Попытка скачать модель с Google Drive...")
print(f"Целевой путь: {target_path}")

# Вместо этого, попробуем использовать готовую ссылку
# или скачать через другой способ

print("\n⚠ Автоматическое скачивание через gdown требует правильный Google Drive ID")
print("Попробуйте скачать модель вручную:")
print("1. Откройте: https://github.com/deepinsight/insightface")
print("2. Найдите модель w600k_r50.onnx")
print(f"3. Сохраните в: {target_path}")

# Или пробуем через другой источник
print("\nПробую альтернативный способ...")
try:
    # Используем прямую ссылку на модель (если доступна)
    url = "https://github.com/deepinsight/insightface/raw/master/python-package/insightface/model_zoo/models/w600k_r50.onnx"
    import urllib.request
    
    print(f"Скачивание с: {url}")
    urllib.request.urlretrieve(url, target_path)
    size_mb = target_path.stat().st_size / 1024 / 1024
    print(f"✓ Модель скачана: {target_path}")
    print(f"  Размер: {size_mb:.1f} MB")
    sys.exit(0)
except Exception as e:
    print(f"✗ Ошибка: {type(e).__name__}: {e}")

sys.exit(1)
