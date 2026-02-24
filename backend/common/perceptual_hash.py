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
            h = imagehash.phash(img)
        return str(h)
    except Exception:
        return None
