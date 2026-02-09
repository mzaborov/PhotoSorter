"""
Логика определения целевой папки по правилам из таблицы folders (content_rule).
Используется веб-API (faces) и скриптами без зависимости от FastAPI.
Правила берутся только из БД (folders.content_rule), без хардкода имён папок.
"""
from __future__ import annotations

import os
import re
from typing import Any

from common.db import get_outsider_person_id

# Форматы content_rule в таблице folders (примеры из UI):
# animals, any_people, only_one_from_group:Дети:9, multiple_from_group:Дети,
# contains_group:Я и Супруга; после миграции: contains_group_id:5, only_one_from_group_id:3:9

_RULE_ONLY_ONE = re.compile(r"^\s*only_one_from_group\s*:\s*([^:]+)(?:\s*:\s*(\d+))?\s*$", re.IGNORECASE)
_RULE_MULTIPLE = re.compile(r"^\s*multiple_from_group\s*:\s*(.+)\s*$", re.IGNORECASE)
_RULE_CONTAINS = re.compile(r"^\s*contains_group\s*:\s*(.+)\s*$", re.IGNORECASE)
_RULE_ONLY_ONE_ID = re.compile(r"^\s*only_one_from_group_id\s*:\s*(\d+)(?:\s*:\s*(\d+))?\s*$", re.IGNORECASE)
_RULE_MULTIPLE_ID = re.compile(r"^\s*multiple_from_group_id\s*:\s*(\d+)\s*$", re.IGNORECASE)
_RULE_CONTAINS_ID = re.compile(r"^\s*contains_group_id\s*:\s*(\d+)\s*$", re.IGNORECASE)


def _get_person_ids_by_group(conn, group_name: str) -> list[int]:
    """Список person_id персон с persons.\"group\" = group_name."""
    if not (group_name or "").strip():
        return []
    cur = conn.cursor()
    cur.execute(
        '''SELECT id FROM persons WHERE "group" = ? ORDER BY group_order ASC, id ASC''',
        (group_name.strip(),),
    )
    return [int(row[0]) for row in cur.fetchall()]


def _get_person_ids_by_group_id(conn, group_id: int) -> list[int]:
    """Список person_id персон из группы по person_groups.id (через persons.\"group\" = person_groups.name)."""
    if group_id is None:
        return []
    cur = conn.cursor()
    cur.execute(
        '''SELECT p.id FROM persons p
           JOIN person_groups g ON g.id = ? AND TRIM(COALESCE(p."group", '')) = TRIM(g.name)
           ORDER BY p.group_order ASC, p.id ASC''',
        (int(group_id),),
    )
    return [int(row[0]) for row in cur.fetchall()]


def get_all_person_names_for_file(
    conn,
    *,
    file_id: int,
    pipeline_run_id: int,
    face_run_id: int,
) -> list[str]:
    """Имена всех персон на файле (исключая Постороннего). Оставляем для обратной совместимости."""
    rows = _get_person_names_and_groups_for_file(
        conn, file_id=file_id, pipeline_run_id=pipeline_run_id, face_run_id=face_run_id
    )
    return [name for name, _ in rows]


def _get_person_names_and_groups_for_file(
    conn,
    *,
    file_id: int,
    pipeline_run_id: int,
    face_run_id: int,
) -> list[tuple[str, str]]:
    """(name, group) для каждой персоны на файле. group может быть \"\"."""
    try:
        outsider_id = get_outsider_person_id(conn, create_if_missing=False)
    except Exception:
        outsider_id = None

    cur = conn.cursor()
    cur.execute(
        """
        SELECT fp.person_id, p.name, COALESCE(p."group", '') FROM file_persons fp
        JOIN persons p ON p.id = fp.person_id
        WHERE fp.pipeline_run_id = ? AND fp.file_id = ?
        """,
        (pipeline_run_id, file_id),
    )
    from_fp = []
    for r in cur.fetchall():
        pid, name, grp = r[0], (r[1] or "").strip(), (r[2] or "").strip()
        if not pid or (outsider_id is not None and pid == outsider_id) or not name:
            continue
        from_fp.append((name, grp))

    cur.execute(
        """
        SELECT DISTINCT COALESCE(fr.manual_person_id, fc.person_id) AS pid
        FROM photo_rectangles fr
        LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
        WHERE fr.file_id = ? AND (fr.run_id = ? OR (SELECT COALESCE(TRIM(f2.inventory_scope), '') FROM files f2 WHERE f2.id = fr.file_id) = 'archive')
          AND (fr.manual_person_id IS NOT NULL OR fc.person_id IS NOT NULL)
        """,
        (file_id, face_run_id),
    )
    pids = [r[0] for r in cur.fetchall() if r[0] and (outsider_id is None or r[0] != outsider_id)]
    from_rect: list[tuple[str, str]] = []
    if pids:
        cur.execute(
            'SELECT name, COALESCE("group", \'\') FROM persons WHERE id IN (' + ",".join("?" * len(pids)) + ")",
            pids,
        )
        from_rect = [((r[0] or "").strip(), (r[1] or "").strip()) for r in cur.fetchall() if r[0]]

    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for name, grp in from_fp + from_rect:
        if name and name not in seen:
            seen.add(name)
            result.append((name, grp))
    return result


def _get_person_ids_for_file(
    conn,
    *,
    file_id: int,
    pipeline_run_id: int,
    face_run_id: int,
) -> set[int]:
    """Множество person_id персон на файле (для правил contains_group)."""
    try:
        outsider_id = get_outsider_person_id(conn, create_if_missing=False)
    except Exception:
        outsider_id = None
    cur = conn.cursor()
    cur.execute(
        """
        SELECT fp.person_id FROM file_persons fp
        WHERE fp.pipeline_run_id = ? AND fp.file_id = ?
        """,
        (pipeline_run_id, file_id),
    )
    from_fp = [r[0] for r in cur.fetchall() if r[0] and (outsider_id is None or r[0] != outsider_id)]
    cur.execute(
        """
        SELECT DISTINCT COALESCE(fr.manual_person_id, fc.person_id) AS pid
        FROM photo_rectangles fr
        LEFT JOIN face_clusters fc ON fc.id = fr.cluster_id
        WHERE fr.file_id = ? AND (fr.run_id = ? OR (SELECT COALESCE(TRIM(f2.inventory_scope), '') FROM files f2 WHERE f2.id = fr.file_id) = 'archive')
          AND (fr.manual_person_id IS NOT NULL OR fc.person_id IS NOT NULL)
        """,
        (file_id, face_run_id),
    )
    from_rect = [r[0] for r in cur.fetchall() if r[0] and (outsider_id is None or r[0] != outsider_id)]
    return set(from_fp + from_rect)


def _match_content_rule(
    conn,
    content_rule: str,
    person_names_and_groups: list[tuple[str, str]],
    file_person_ids: set[int],
) -> bool:
    """
    Совпадает ли правило папки с персонами на файле.
    content_rule: animals, any_people, only_one_from_group:Group[:id], multiple_from_group:Group, contains_group:Group.
    """
    rule = (content_rule or "").strip()
    if not rule:
        return False
    if rule == "animals" or rule == "any_people":
        return False  # обрабатываются отдельно (не по списку персон)

    # Правила по id группы (после миграции)
    m = _RULE_ONLY_ONE_ID.match(rule)
    if m:
        group_id = int(m.group(1))
        optional_id = m.group(2)
        group_ids = set(_get_person_ids_by_group_id(conn, group_id))
        on_file = group_ids & file_person_ids
        if len(on_file) != 1:
            return False
        if optional_id is not None:
            try:
                pid = int(optional_id)
                return pid in on_file
            except ValueError:
                pass
        return True

    m = _RULE_MULTIPLE_ID.match(rule)
    if m:
        group_id = int(m.group(1))
        group_ids = set(_get_person_ids_by_group_id(conn, group_id))
        return len(group_ids & file_person_ids) >= 2

    m = _RULE_CONTAINS_ID.match(rule)
    if m:
        group_id = int(m.group(1))
        group_ids = set(_get_person_ids_by_group_id(conn, group_id))
        return bool(group_ids and (group_ids & file_person_ids))

    # Правила по имени группы (до миграции)
    m = _RULE_ONLY_ONE.match(rule)
    if m:
        group_name = m.group(1).strip()
        optional_id = m.group(2)  # число после второго двоеточия (person_id или порядок)
        group_ids = set(_get_person_ids_by_group(conn, group_name))
        on_file = group_ids & file_person_ids
        # Ровно один из группы на файле (другие персоны на файле допустимы — Санек + Рид Анна → Санек)
        if len(on_file) != 1:
            return False
        if optional_id is not None:
            try:
                pid = int(optional_id)
                return pid in on_file
            except ValueError:
                pass
        return True

    m = _RULE_MULTIPLE.match(rule)
    if m:
        group_name = m.group(1).strip()
        count = sum(1 for _, grp in person_names_and_groups if grp == group_name)
        return count >= 2

    m = _RULE_CONTAINS.match(rule)
    if m:
        group_name = m.group(1).strip()
        group_ids = set(_get_person_ids_by_group(conn, group_name))
        # Хотя бы один из группы на файле (для папки «Миша и Аня» — один из пары достаточно)
        return bool(group_ids and (group_ids & file_person_ids))

    return False


def _find_folder_by_rule_value(target_folders: list[dict[str, Any]], rule_value: str) -> str | None:
    """Имя папки (name), у которой content_rule равен rule_value (без учёта пробелов)."""
    rule_value = (rule_value or "").strip().lower()
    for f in target_folders:
        r = (f.get("content_rule") or "").strip().lower()
        if r == rule_value:
            name = (f.get("name") or "").strip()
            if name:
                return name
    return None


def _get_target_path_for_folder(
    target_folders: list[dict[str, Any]],
    folder_name: str,
    root_path: str,
) -> str:
    """
    Возвращает целевой путь для папки. target_folder в БД всегда под корнем прогона (local:root/...);
    путь disk:/Фото/... используется только при «Перенести в архив», не пишется в target_folder.
    При локальном корне (root_path startswith local:) — всегда local:base_path/folder_name.
    При disk-корне — path из таблицы folders или base_path/folder_name.
    """
    folder_name = (folder_name or "").strip()
    base_path = root_path.rstrip("/") if root_path.startswith("disk:") else (root_path[6:] if root_path.startswith("local:") else root_path)
    # Корень прогона локальный (local: или путь без disk:) — target_folder всегда под корнем (без disk:). В прогоне root_path часто без префикса local:
    is_local_root = root_path.startswith("local:") or (root_path and not root_path.startswith("disk:"))
    if is_local_root:
        return f"local:{os.path.join(base_path, *folder_name.split('/'))}"
    # Корень на диске — можно брать path из таблицы folders
    for f in target_folders or []:
        if (f.get("name") or "").strip() == folder_name:
            path_val = (f.get("path") or "").strip()
            if path_val.startswith("disk:"):
                return path_val
            break
    if root_path.startswith("disk:"):
        return f"{base_path}/{folder_name}"
    return f"local:{os.path.join(base_path, folder_name)}"


def resolve_target_folder_for_faces(
    conn,
    file_id: int,
    pipeline_run_id: int,
    face_run_id: int,
    target_folders: list[dict[str, Any]],
) -> str:
    """
    По файлу и целевым папкам из таблицы folders (role=target, sort_order) возвращает имя папки:
    первое совпадение по content_rule; иначе папка с правилом any_people (fallback).
    Правила только из БД (only_one_from_group, multiple_from_group, contains_group, any_people).
    """
    person_names_and_groups = _get_person_names_and_groups_for_file(
        conn, file_id=file_id, pipeline_run_id=pipeline_run_id, face_run_id=face_run_id
    )
    file_person_ids = _get_person_ids_for_file(
        conn, file_id=file_id, pipeline_run_id=pipeline_run_id, face_run_id=face_run_id
    )

    fallback_name = _find_folder_by_rule_value(target_folders, "any_people")

    person_names = [n for n, _ in person_names_and_groups]

    for folder in target_folders:
        name = (folder.get("name") or "").strip()
        if not name:
            continue
        rule = (folder.get("content_rule") or "").strip()
        if not rule or rule.lower() == "any_people":
            continue
        if _match_content_rule(conn, rule, person_names_and_groups, file_person_ids):
            return name
        # Старый формат: только:[X], вместе:[A,B], содержит:[A,B]
        for rule_type, rule_names in parse_content_rule(rule):
            if folder_rules_match(rule_type, rule_names, person_names):
                return name

    return fallback_name or "Другие люди"


def determine_target_folder(
    *,
    path: str,
    effective_tab: str,
    root_path: str,
    preclean_kind: str | None = None,
    person_name: str | None = None,
    target_folders: list[dict[str, Any]] | None = None,
    group_path: str | None = None,
) -> str | None:
    """
    Определяет целевую папку для файла. Имена папок для animals и fallback «лица»
    берутся из target_folders по content_rule (animals, any_people), без хардкода.
    Для no_faces: при наличии group_path — поездки в Путешествия/... (path из folders, может быть disk:/Фото/...);
    остальные группы (Технологии, Чеки, Мемы, Дом и ремонт и т.д.) — только локальная раскладка (local:root/группа), на ЯД не заливаются, из прогона убираются после локального перемещения; без группы — None.
    """
    if root_path.startswith("disk:"):
        base_path = root_path.rstrip("/")
    elif root_path.startswith("local:"):
        base_path = root_path[6:]
    else:
        base_path = root_path

    if preclean_kind:
        if preclean_kind == "non_media":
            folder_name = "_non_media"
        elif preclean_kind == "broken_media":
            folder_name = "_broken_media"
        else:
            return None
    elif effective_tab == "faces":
        pn = (person_name or "").strip()
        folder_name = pn if pn else (_find_folder_by_rule_value(target_folders or [], "any_people") or "Другие люди")
    elif effective_tab == "quarantine":
        folder_name = "_quarantine"
    elif effective_tab == "animals":
        folder_name = _find_folder_by_rule_value(target_folders or [], "animals") if target_folders else "_animals"
        if not folder_name:
            folder_name = "_animals"
    elif effective_tab == "people_no_face":
        folder_name = "_people_no_face"
    elif effective_tab == "no_faces":
        # С группой: поездки — в Путешествия/... (path из folders, disk или local); остальные группы — только локальная раскладка (local:root/группа), на ЯД не заливаются.
        # В БД/UI группа может быть «Поездки» или «Путешествия» — в целевой пути всегда «Путешествия».
        # group_path из file_groups может приходить с префиксом _no_faces/ — убираем, чтобы target_folder в БД был правильным (без _no_faces)
        gp = (group_path or "").strip()
        if gp.startswith("_no_faces/"):
            gp = gp[len("_no_faces/") :].strip()
        if not gp:
            return None
        # Поездка: группа «Путешествия»/«Поездки» или «.../XXX» или название начинается с года (2025 Гончарка Москва)
        is_trip = (
            gp in ("Путешествия", "Поездки")
            or gp.startswith("Путешествия/")
            or gp.startswith("Поездки/")
            or bool(re.match(r"^\d{4}\s", gp))
        )
        if is_trip:
            if gp in ("Путешествия", "Поездки"):
                parts = ["Путешествия"]
            elif gp.startswith("Путешествия/"):
                parts = ["Путешествия"] + [p for p in gp[13:].split("/") if p.strip()]
            elif gp.startswith("Поездки/"):
                parts = ["Путешествия"] + [p for p in gp[9:].split("/") if p.strip()]
            else:
                parts = ["Путешествия", gp]
            if not parts:
                return None
            if root_path.startswith("disk:"):
                return base_path + "/" + "/".join(parts)
            return "local:" + os.path.join(base_path, *parts)
        # Не поездка: Технологии, Чеки, Мемы, Дом и ремонт, Здоровье и т.д. — только локальная раскладка (local:root/группа); на ЯД не заливаются, после перемещения локально убираются из прогона
        parts = [p for p in gp.split("/") if p.strip()]
        if not parts:
            return None
        return "local:" + os.path.join(base_path, *parts)
    else:
        return None

    # Служебные папки — всегда под корнем прогона; папки из таблицы (лица, животные) — path из БД, если disk:
    if folder_name in ("_non_media", "_broken_media", "_quarantine", "_people_no_face"):
        if root_path.startswith("disk:"):
            return f"{base_path}/{folder_name}"
        return f"local:{os.path.join(base_path, folder_name)}"
    return _get_target_path_for_folder(target_folders or [], folder_name, root_path)


# Обратная совместимость: старый парсер только/вместе/содержит (если в БД остались такие правила)
def parse_content_rule(content_rule: str | None) -> list[tuple[str, list[str]]]:
    """Парсит старый формат только:[X], вместе:[A,B], содержит:[A,B]. Для совместимости."""
    _old = re.compile(r"^\s*(только|вместе|содержит)\s*:\s*\[(.*)\]\s*$", re.IGNORECASE)
    if not (content_rule or "").strip():
        return []
    result: list[tuple[str, list[str]]] = []
    for part in (content_rule or "").split(";"):
        part = part.strip()
        if not part:
            continue
        m = _old.match(part)
        if not m:
            continue
        rule_type = m.group(1).strip().lower()
        names = [n.strip() for n in (m.group(2) or "").split(",") if n.strip()]
        if names:
            result.append((rule_type, names))
    return result


def folder_rules_match(rule_type: str, rule_names: list[str], person_names: list[str]) -> bool:
    """Совпадение для старого формата (только/вместе/содержит). Для совместимости."""
    names_set = set(person_names)
    rule_set = set(rule_names)
    if rule_type == "только":
        return len(person_names) == 1 and person_names[0] in rule_set
    if rule_type == "вместе":
        return len(names_set & rule_set) >= 2
    if rule_type == "содержит":
        return rule_set <= names_set
    return False
