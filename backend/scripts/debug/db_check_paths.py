from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _db_path() -> Path:
    # use project-local DB location
    # backend/scripts/debug/* -> repo root = parents[3]
    return Path(__file__).resolve().parents[3] / "data" / "photosorter.db"


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug helper: dump files fields for provided local:/disk: paths.")
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

    # Show last pipeline run for quick context
    try:
        pr = cur.execute("select id, root_path, face_run_id, status from pipeline_runs order by id desc limit 1").fetchone()
        print("last pipeline_run:", dict(pr) if pr else None)
    except Exception as e:  # noqa: BLE001
        print("last pipeline_run: <unavailable>", f"{type(e).__name__}: {e}")

    q = f"""
    SELECT
      path,
      faces_run_id, COALESCE(faces_count, 0) AS faces_count,
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

    print("\nrows:")
    for p in paths:
        r = by.get(p)
        if not r:
            print("MISSING", p)
            continue
        print(
            p,
            "| run",
            r["faces_run_id"],
            "| faces",
            r["faces_count"],
            "| manual",
            (r["faces_manual_label"] or ""),
            "| quarantine",
            int(r["faces_auto_quarantine"] or 0),
            (r["faces_quarantine_reason"] or ""),
            "| animal",
            int(r["animals_auto"] or 0),
            (r["animals_kind"] or ""),
            "| people_no_face",
            int(r["people_no_face_manual"] or 0),
            (r["people_no_face_person"] or ""),
        )

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






