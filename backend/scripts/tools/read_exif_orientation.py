"""
Читает EXIF Orientation из JPEG по пути. Без внешних зависимостей (только стандартная библиотека).
Запуск: python backend/scripts/tools/read_exif_orientation.py --path "C:\\tmp\\Photo\\_faces\\IMG-20250311-WA0035.jpg"
Путь может быть с префиксом local: или без.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# EXIF tag ids
TAG_ORIENTATION = 0x0112  # 274
TAG_EXIF_IFD = 0x8769  # указатель на Exif IFD

ORIENTATION_NAMES = {
    1: "Normal (0°)",
    2: "Mirror horizontal",
    3: "Rotate 180°",
    4: "Mirror vertical",
    5: "Mirror horizontal + rotate 270° CW",
    6: "Rotate 90° CW (вправо)",
    7: "Mirror horizontal + rotate 90° CW",
    8: "Rotate 90° CCW (влево)",
}


def read_exif_orientation(path: Path, debug: bool = False) -> int | None:
    """
    Читает EXIF Orientation из JPEG файла без Pillow.
    Возвращает 1-8 или None, если EXIF/ориентация не найдена.
    """
    data = path.read_bytes()
    if debug:
        print(f"Размер файла: {len(data)} байт, начало: {data[:4].hex()}")
    if len(data) < 4 or data[:2] != b"\xFF\xD8":
        return None  # не JPEG
    pos = 2
    app1_count = 0
    while pos + 10 <= len(data):
        if data[pos] != 0xFF:
            if debug and app1_count == 0:
                print(f"Выход: на позиции {pos} не 0xFF (0x{data[pos]:02x})")
            break
        marker = data[pos + 1]
        pos += 2
        if marker == 0x00:  # JPEG stuffed zero, skip
            continue
        if marker == 0xE1:  # APP1
            seg_len = struct.unpack(">H", data[pos : pos + 2])[0]
            pos += 2
            if debug:
                app1_count += 1
                payload_preview = data[pos : pos + 14].hex() if pos + 14 <= len(data) else ""
                print(f"APP1 #{app1_count}: seg_len={seg_len}, payload[0:14] hex={payload_preview}")
            if seg_len < 8 or pos + seg_len - 2 > len(data):
                pos += max(0, seg_len - 2)
                continue
            # После длины идёт "Exif\0\0" (6 байт), затем TIFF header (II или MM)
            has_exif = pos + 6 <= len(data) and data[pos : pos + 4].lower() == b"exif" and data[pos + 4 : pos + 6] == b"\x00\x00"
            if has_exif:
                tiff_start = pos + 6
                has_tiff = tiff_start + 8 <= len(data) and data[tiff_start : tiff_start + 2] in (b"II", b"MM")
                if debug:
                    print(f"  Найдено 'Exif\\0\\0', TIFF II/MM={has_tiff}")
                if has_tiff:
                    if debug:
                        endian = "<" if data[tiff_start : tiff_start + 2] == b"II" else ">"
                        ifd0_off = struct.unpack(f"{endian}I", data[tiff_start + 4 : tiff_start + 8])[0]
                        print(f"  IFD0 offset={ifd0_off}")
                    result = _parse_tiff_ifd_for_orientation(data, tiff_start)
                    if result is not None:
                        return result
                    if debug:
                        print("  В IFD0/Exif IFD тег Orientation не найден")
            elif debug and app1_count <= 3:
                print(f"  Не Exif (первые 6 байт: {data[pos:pos+6]!r})")
            pos += seg_len - 2
            continue
        if marker == 0x01 or (marker >= 0xD0 and marker <= 0xD9):
            pos += 2
            continue
        if pos + 2 > len(data):
            break
        seg_len = struct.unpack(">H", data[pos : pos + 2])[0]
        pos += seg_len
    if debug and app1_count == 0:
        print("Сегментов APP1 не найдено (EXIF обычно в APP1)")
    return None


def _parse_tiff_ifd_for_orientation(data: bytes, tiff_start: int) -> int | None:
    if tiff_start + 8 > len(data):
        return None
    # TIFF header: II (little) or MM (big), 42, offset to first IFD
    byte_order = data[tiff_start : tiff_start + 2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return None
    ifd0_offset = struct.unpack(f"{endian}I", data[tiff_start + 4 : tiff_start + 8])[0]
    # Сначала ищем в IFD0
    result = _scan_ifd_for_orientation(data, tiff_start, tiff_start + ifd0_offset, endian)
    if result is not None:
        return result
    # Часто Orientation в Exif IFD (tag 0x8769)
    exif_ifd_offset = _get_exif_ifd_offset(data, tiff_start, tiff_start + ifd0_offset, endian)
    if exif_ifd_offset is not None:
        return _scan_ifd_for_orientation(data, tiff_start, tiff_start + exif_ifd_offset, endian)
    return None


def _scan_ifd_for_orientation(
    data: bytes, tiff_start: int, ifd_pos: int, endian: str
) -> int | None:
    if ifd_pos + 2 > len(data):
        return None
    num_entries = struct.unpack(f"{endian}H", data[ifd_pos : ifd_pos + 2])[0]
    ifd_pos += 2
    for _ in range(num_entries):
        if ifd_pos + 12 > len(data):
            break
        tag = struct.unpack(f"{endian}H", data[ifd_pos : ifd_pos + 2])[0]
        type_ = struct.unpack(f"{endian}H", data[ifd_pos + 2 : ifd_pos + 4])[0]
        count = struct.unpack(f"{endian}I", data[ifd_pos + 4 : ifd_pos + 8])[0]
        value = data[ifd_pos + 8 : ifd_pos + 12]
        ifd_pos += 12
        if tag != TAG_ORIENTATION or type_ != 3 or count != 1:
            continue
        orient = struct.unpack(f"{endian}H", value[:2])[0]
        if 1 <= orient <= 8:
            return orient
    return None


def _get_exif_ifd_offset(
    data: bytes, tiff_start: int, ifd_pos: int, endian: str
) -> int | None:
    if ifd_pos + 2 > len(data):
        return None
    num_entries = struct.unpack(f"{endian}H", data[ifd_pos : ifd_pos + 2])[0]
    ifd_pos += 2
    for _ in range(num_entries):
        if ifd_pos + 12 > len(data):
            break
        tag = struct.unpack(f"{endian}H", data[ifd_pos : ifd_pos + 2])[0]
        type_ = struct.unpack(f"{endian}H", data[ifd_pos + 2 : ifd_pos + 4])[0]
        value = data[ifd_pos + 8 : ifd_pos + 12]
        ifd_pos += 12
        if tag == TAG_EXIF_IFD and type_ == 4 and len(value) >= 4:  # LONG
            return struct.unpack(f"{endian}I", value[:4])[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Читает EXIF Orientation из JPEG (без Pillow)")
    parser.add_argument("--path", "-p", required=True, help="Путь к файлу (можно с префиксом local:)")
    parser.add_argument("--debug", "-d", action="store_true", help="Вывести отладочную информацию")
    args = parser.parse_args()

    path_raw = (args.path or "").strip()
    if path_raw.startswith("local:"):
        path_raw = path_raw[6:].lstrip()
    path = Path(path_raw)
    if not path.is_file():
        print(f"Файл не найден: {path}")
        sys.exit(2)

    orientation_raw = read_exif_orientation(path, debug=args.debug)
    if orientation_raw is None:
        print("EXIF не найден или Orientation отсутствует")
        sys.exit(0)

    name = ORIENTATION_NAMES.get(orientation_raw, "?")
    print(f"EXIF Orientation: {orientation_raw} — {name}")


if __name__ == "__main__":
    main()
