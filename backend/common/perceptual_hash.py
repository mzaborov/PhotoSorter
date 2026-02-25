"""
Перцептивный хеш изображений для группировки визуально похожих фото (дубли разного размера/качества).
Используется Pillow + imagehash (pHash).
"""
from __future__ import annotations

from pathlib import Path


def compute_phash_hex(local_path: str | Path) -> str | None:
    """
    Вычисляет pHash изображения по локальному пути к файлу.
    Возвращает hex-строку хеша или None при ошибке (не изображение, битый файл и т.д.).
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return None

    path = Path(local_path)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as img:
            # Конвертируем в RGB, если нужно (для PNG с прозрачностью и т.д.)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            w, h = img.size
            # imagehash.phash может падать на очень маленьких изображениях; доводим до минимум 8x8
            if w < 8 or h < 8:
                scale = max(8 / w, 8 / h) if w and h else 1
                new_w = max(8, int(w * scale))
                new_h = max(8, int(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            h = imagehash.phash(img)
        return str(h)
    except Exception:
        return None
