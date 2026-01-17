#!/usr/bin/env python3
"""Скачивает модель ArcFace напрямую."""

import sys
import urllib.request
from pathlib import Path

target_dir = Path(__file__).resolve().parents[3] / "models" / "face_recognition"
target_dir.mkdir(parents=True, exist_ok=True)
target_path = target_dir / "w600k_r50.onnx"

if target_path.exists():
    size_mb = target_path.stat().st_size / 1024 / 1024
    print(f"✓ Модель уже существует: {target_path}")
    print(f"  Размер: {size_mb:.1f} MB")
    if size_mb > 10:
        sys.exit(0)

# Пробуем разные URL
urls = [
    # HuggingFace (прямая ссылка на файл)
    "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    # Альтернативные источники
    "https://github.com/deepinsight/insightface/releases/download/v0.7.3/buffalo_l.zip",
]

print("Попытка скачать модель напрямую...")
print(f"Целевой путь: {target_path}")

for url in urls:
    try:
        print(f"\nПробую: {url}")
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = "=" * filled + "-" * (bar_length - filled)
                print(f"\r  [{bar}] {percent}%", end="", flush=True)
        
        if url.endswith('.zip'):
            # Если это архив, скачиваем во временный файл
            import tempfile
            import zipfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                urllib.request.urlretrieve(url, tmp.name, reporthook=progress_hook)
                print()
                # Распаковываем
                with zipfile.ZipFile(tmp.name, 'r') as zip_ref:
                    # Ищем w600k_r50.onnx в архиве
                    for name in zip_ref.namelist():
                        if name.endswith('w600k_r50.onnx'):
                            with zip_ref.open(name) as source, open(target_path, 'wb') as target:
                                target.write(source.read())
                            print(f"✓ Модель извлечена из архива")
                            import os
                            os.unlink(tmp.name)
                            sys.exit(0)
        else:
            urllib.request.urlretrieve(url, target_path, reporthook=progress_hook)
            print(f"\n✓ Модель скачана: {target_path}")
            size_mb = target_path.stat().st_size / 1024 / 1024
            print(f"  Размер: {size_mb:.1f} MB")
            sys.exit(0)
    except Exception as e:
        print(f"\n  ✗ Ошибка: {type(e).__name__}: {e}")
        continue

print("\n✗ Не удалось скачать модель автоматически")
print("\nРучная установка:")
print("1. Откройте: https://huggingface.co/deepinsight/insightface/tree/main/models/buffalo_l")
print("2. Скачайте w600k_r50.onnx")
print(f"3. Сохраните в: {target_path}")
sys.exit(1)
