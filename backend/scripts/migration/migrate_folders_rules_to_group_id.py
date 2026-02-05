#!/usr/bin/env python3
"""
Миграция правил папок с имени группы на id группы.

1. Заполняет таблицу person_groups из DISTINCT persons."group" (непустые).
2. Заменяет в folders.content_rule:
   - contains_group:Семья       -> contains_group_id:5
   - only_one_from_group:Дети:9 -> only_one_from_group_id:3:9
   - multiple_from_group:Дети   -> multiple_from_group_id:3

Запуск из корня репозитория:
  python backend/scripts/migration/migrate_folders_rules_to_group_id.py
  python backend/scripts/migration/migrate_folders_rules_to_group_id.py --dry-run
"""

import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

try:
    from backend.common.db import get_connection
except ImportError:
    from common.db import get_connection

# Форматы правил (как в sort_rules.py)
_RULE_CONTAINS = re.compile(r"^\s*contains_group\s*:\s*(.+)\s*$", re.IGNORECASE)
_RULE_ONLY_ONE = re.compile(r"^\s*only_one_from_group\s*:\s*([^:]+)(?:\s*:\s*(\d+))?\s*$", re.IGNORECASE)
_RULE_MULTIPLE = re.compile(r"^\s*multiple_from_group\s*:\s*(.+)\s*$", re.IGNORECASE)


def ensure_person_groups_filled(conn) -> dict[str, int]:
    """
    Заполняет person_groups из persons."group". Возвращает словарь name -> id.
    Таблица person_groups создаётся в db.py при инициализации БД (запустите приложение до миграции).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT TRIM("group") AS g, MIN(group_order) AS ord
        FROM persons
        WHERE TRIM(COALESCE("group", '')) != ''
        GROUP BY TRIM("group")
    """)
    name_to_id: dict[str, int] = {}
    for name, sort_order in cur.fetchall():
        name = (name or "").strip()
        if not name:
            continue
        cur.execute(
            "INSERT OR IGNORE INTO person_groups (name, sort_order) VALUES (?, ?)",
            (name, sort_order),
        )
        cur.execute("SELECT id FROM person_groups WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            name_to_id[name] = int(row[0])
    return name_to_id


def migrate_rule(rule: str, name_to_id: dict[str, int]) -> str | None:
    """
    Заменяет правило по имени группы на правило по id. Возвращает новую строку или None, если менять не нужно.
    """
    rule = (rule or "").strip()
    if not rule or rule.lower() in ("animals", "any_people"):
        return None

    m = _RULE_CONTAINS.match(rule)
    if m:
        name = m.group(1).strip()
        gid = name_to_id.get(name)
        if gid is not None:
            return f"contains_group_id:{gid}"
        return None

    m = _RULE_ONLY_ONE.match(rule)
    if m:
        name = m.group(1).strip()
        opt = m.group(2)
        gid = name_to_id.get(name)
        if gid is not None:
            if opt is not None:
                return f"only_one_from_group_id:{gid}:{opt}"
            return f"only_one_from_group_id:{gid}"
        return None

    m = _RULE_MULTIPLE.match(rule)
    if m:
        name = m.group(1).strip()
        gid = name_to_id.get(name)
        if gid is not None:
            return f"multiple_from_group_id:{gid}"
        return None

    return None


def apply_migration(conn, name_to_id: dict[str, int], dry_run: bool) -> int:
    """
    Заменяет в folders.content_rule правила по имени группы на правила по id.
    Возвращает количество обновлённых папок.
    """
    cur = conn.cursor()
    cur.execute("SELECT id, name, content_rule FROM folders WHERE content_rule IS NOT NULL AND TRIM(content_rule) != ''")
    rows = cur.fetchall()
    updated = 0
    for folder_id, folder_name, content_rule in rows:
        new_rule = migrate_rule(content_rule, name_to_id)
        if new_rule is not None:
            print(f"  folder id={folder_id} name={folder_name!r}: {content_rule!r} -> {new_rule!r}")
            if not dry_run:
                cur.execute("UPDATE folders SET content_rule = ? WHERE id = ?", (new_rule, folder_id))
            updated += 1
    if not dry_run and updated:
        conn.commit()
    return updated


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    conn = get_connection()
    try:
        name_to_id = ensure_person_groups_filled(conn)
        if not dry_run:
            conn.commit()
        print("Person groups (name -> id):")
        for name, gid in sorted(name_to_id.items(), key=lambda x: (x[1], x[0])):
            print(f"  {gid}: {name!r}")
        updated = apply_migration(conn, name_to_id, dry_run)
        if dry_run:
            conn.rollback()
            print("[DRY-RUN] No changes written.")
        print(f"Groups in person_groups: {len(name_to_id)}, folders updated: {updated}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
