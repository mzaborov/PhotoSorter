from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # backend/logic/gold/store.py -> repo root
    return Path(__file__).resolve().parents[3]


def gold_cases_dir() -> Path:
    return _repo_root() / "regression" / "cases"


def gold_faces_manual_rects_path() -> Path:
    # Рядом с faces_gold — отдельный файл с manual-rectangles (NDJSON, 1 JSON per line).
    return gold_cases_dir() / "faces_manual_rects_gold.ndjson"


def gold_faces_video_frames_path() -> Path:
    # Рядом с faces_gold — отдельный файл с разметкой 3 кадров видео (NDJSON, 1 JSON per line).
    return gold_cases_dir() / "faces_video_frames_gold.ndjson"


def gold_file_map() -> dict[str, Path]:
    d = gold_cases_dir()
    return {
        "cats_gold": d / "cats_gold.txt",
        "quarantine_gold": d / "quarantine_gold.txt",
        "no_faces_gold": d / "no_faces_gold.txt",
        "people_no_face_gold": d / "people_no_face_gold.txt",
        "faces_gold": d / "faces_gold.txt",
        "drawn_faces_gold": d / "drawn_faces_gold.txt",
    }


def gold_read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def gold_write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(txt, encoding="utf-8")


def gold_merge_append_only(path: Path, new_items: list[str]) -> int:
    existing = gold_read_lines(path)
    seen = set(existing)
    added = 0
    for it in new_items:
        s = (it or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        existing.append(s)
        seen.add(s)
        added += 1
    gold_write_lines(path, existing)
    return added


def _is_windows_abs_path(p: str) -> bool:
    # C:\... or C:/...
    return bool(p) and len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/")


def gold_normalize_path(raw_path: str) -> dict[str, str]:
    """
    gold txt может содержать:
    - disk:/... (как в YaDisk)
    - local:C:\\... (как в БД)
    - C:\\... (старые записи)
    Возвращаем:
      raw_path: как в файле (для delete)
      path: нормализованный (для preview)
    """
    raw = (raw_path or "").strip()
    if raw.startswith("disk:") or raw.startswith("local:"):
        return {"raw_path": raw, "path": raw}
    if _is_windows_abs_path(raw) or raw.startswith("\\\\"):
        return {"raw_path": raw, "path": "local:" + raw}
    return {"raw_path": raw, "path": raw}


def gold_expected_tab_by_path(*, include_drawn_faces: bool = False) -> dict[str, str]:
    """
    Для подсветки "sorted past gold" на /faces:
    если путь есть в gold, но попал в другую эффективную вкладку — подсветим.

    Возвращает mapping: normalized_path -> expected_tab (faces|no_faces|people_no_face|animals|quarantine).
    """
    m = gold_file_map()
    # порядок важен: если вдруг дубли, берём более "сильный" сигнал (ручные категории выше)
    order: list[tuple[str, str]] = [
        ("faces_gold", "faces"),
        ("no_faces_gold", "no_faces"),
        ("people_no_face_gold", "people_no_face"),
        ("cats_gold", "animals"),
        ("quarantine_gold", "quarantine"),
    ]
    if include_drawn_faces:
        order.append(("drawn_faces_gold", "faces"))

    out: dict[str, str] = {}
    for name, tab in order:
        p = m.get(name)
        if not p:
            continue
        for raw in gold_read_lines(p):
            nm = gold_normalize_path(raw)
            path = nm["path"]
            if path and path not in out:
                out[path] = tab
    return out


def gold_read_ndjson_by_path(path: Path) -> dict[str, dict[str, Any]]:
    """
    Читает NDJSON и возвращает mapping: path -> record.
    Формат записи ожидаем: {"path": "...", ...}.
    """
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        p = str(obj.get("path") or "").strip()
        if not p:
            continue
        out[p] = obj
    return out


def gold_write_ndjson_by_path(path: Path, items_by_path: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for p in sorted(items_by_path.keys()):
        obj = items_by_path[p]
        obj2 = dict(obj)
        obj2["path"] = p
        lines.append(json.dumps(obj2, ensure_ascii=False))
    txt = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(txt, encoding="utf-8")



