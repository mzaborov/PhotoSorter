#!/usr/bin/env python3
"""
Скачивает модель распознавания AntelopeV2 (InsightFace) для экспериментов с альт-эмбеддингами.
Источник: immich-app/antelopev2 на Hugging Face (recognition/model.onnx).

После скачивания:
  python backend/scripts/tools/backfill_archive_embeddings_alt.py --model-path models/face_recognition/antelopev2/model.onnx --model-key antelopev2
  python backend/scripts/tools/tune_face_clustering.py --embedding-source alt --model-key antelopev2
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_DIR = REPO_ROOT / "models" / "face_recognition" / "antelopev2"
TARGET_FILE = TARGET_DIR / "model.onnx"
# Прямая ссылка на recognition model (261 MB)
URL = "https://huggingface.co/immich-app/antelopev2/resolve/main/recognition/model.onnx"


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.exists():
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            print(f"Модель уже есть: {TARGET_FILE} ({size_mb:.1f} MB)")
            return 0
        print(f"Файл есть, но маленький ({size_mb:.1f} MB), перекачиваю...")

    print(f"Скачиваю AntelopeV2 recognition -> {TARGET_FILE}")
    print(f"  {URL}")

    def progress(count, block_size, total):
        if total <= 0:
            return
        pct = min(100, count * block_size * 100 // total)
        bar = "=" * (pct * 40 // 100) + "-" * (40 - pct * 40 // 100)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(URL, TARGET_FILE, reporthook=progress)
        print()
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print(f"Готово: {TARGET_FILE} ({size_mb:.1f} MB)")
        return 0
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("\nРучная установка:")
        print("  1. Откройте https://huggingface.co/immich-app/antelopev2/tree/main/recognition")
        print("  2. Скачайте model.onnx")
        print(f"  3. Сохраните как {TARGET_FILE}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
