from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "photosorter.db"


def _effective_tab(r: sqlite3.Row) -> str:
    """
    Mirrors the current "effective category" logic used by the UI.
    Tabs: faces, quarantine, no_faces, animals, people_no_face
    """
    faces_count = int(r["faces_count"] or 0)
    ml = (r["faces_manual_label"] or "").strip().lower()
    if ml == "faces":
        return "faces"
    if ml == "no_faces":
        return "no_faces"
    if int(r["people_no_face_manual"] or 0) == 1:
        return "people_no_face"
    if faces_count <= 0 and int(r["animals_auto"] or 0) == 1 and (r["animals_kind"] or "").strip():
        return "animals"
    # many_small_faces показываем внутри tab=faces (2-й уровень), не как отдельный top-tab quarantine
    q_reason = (r["faces_quarantine_reason"] or "").strip().lower()
    if int(r["faces_auto_quarantine"] or 0) == 1 and q_reason != "many_small_faces":
        return "quarantine"
    return "faces" if faces_count > 0 else "no_faces"


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug helper: show effective UI tab for provided paths.")
    ap.add_argument("--db", default=str(_db_path()), help="Path to photosorter.db")
    ap.add_argument("--paths-file", required=True, help="Text file with one path per line (local:... or disk:...)")
    args = ap.parse_args()

    paths_file = Path(args.paths_file)
    if not paths_file.exists():
        raise SystemExit(f"paths-file not found: {paths_file}")

    paths = []
    for line in paths_file.read_text(encoding="utf-8").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        paths.append(s)
    if not paths:
        print("No paths provided.")
        return 0

    con = sqlite3.connect(str(Path(args.db)))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    q = f"""
    SELECT
      path,
      COALESCE(faces_count, 0) AS faces_count,
      COALESCE(faces_manual_label, '') AS faces_manual_label,
      COALESCE(faces_auto_quarantine, 0) AS faces_auto_quarantine,
      COALESCE(faces_quarantine_reason, '') AS faces_quarantine_reason,
      COALESCE(animals_auto, 0) AS animals_auto,
      COALESCE(animals_kind, '') AS animals_kind,
      COALESCE(people_no_face_manual, 0) AS people_no_face_manual,
      COALESCE(people_no_face_person, '') AS people_no_face_person
    FROM files
    WHERE path IN ({",".join(["?"] * len(paths))})
    """
    rows = cur.execute(q, paths).fetchall()
    by = {r["path"]: r for r in rows}

    for p in paths:
        r = by.get(p)
        if not r:
            print("MISSING", p)
            continue
        tab = _effective_tab(r)
        extra = ""
        if tab == "quarantine":
            extra = f" reason={r['faces_quarantine_reason']!s}"
        elif tab == "animals":
            extra = f" kind={r['animals_kind']!s}"
        elif tab == "people_no_face":
            extra = f" person={r['people_no_face_person']!s}"
        print(tab, "| faces", int(r["faces_count"] or 0), "|", p, extra)

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



