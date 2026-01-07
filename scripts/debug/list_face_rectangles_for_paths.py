from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "photosorter.db"


def _get_face_run_id(con: sqlite3.Connection, *, pipeline_run_id: int | None) -> int | None:
    if pipeline_run_id is None:
        return None
    cur = con.cursor()
    row = cur.execute("SELECT face_run_id FROM pipeline_runs WHERE id = ? LIMIT 1", (int(pipeline_run_id),)).fetchone()
    if not row:
        return None
    v = row[0]
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug helper: list face_rectangles (bbox/score) for provided paths.")
    ap.add_argument("--db", default=str(_db_path()), help="Path to photosorter.db")
    ap.add_argument("--paths-file", required=True, help="Text file with one path per line (local:... or disk:...)")
    ap.add_argument("--pipeline-run-id", type=int, default=None, help="Pipeline run id to resolve face_run_id (preferred)")
    ap.add_argument("--face-run-id", type=int, default=None, help="Face run id (overrides pipeline-run-id)")
    args = ap.parse_args()

    paths_file = Path(args.paths_file)
    if not paths_file.exists():
        raise SystemExit(f"paths-file not found: {paths_file}")

    paths: list[str] = []
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

    face_run_id = int(args.face_run_id) if args.face_run_id else _get_face_run_id(con, pipeline_run_id=args.pipeline_run_id)
    if face_run_id is None:
        # fallback: last face_run_id from pipeline_runs
        try:
            row = con.execute("SELECT face_run_id FROM pipeline_runs WHERE face_run_id IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
            face_run_id = int(row[0]) if row and row[0] is not None else None
        except Exception:
            face_run_id = None

    if face_run_id is None:
        raise SystemExit("face_run_id not found (provide --pipeline-run-id or --face-run-id)")

    print(f"Using face_run_id={face_run_id}")

    # Fetch rectangles
    q = f"""
    SELECT
      run_id, file_path, face_index,
      bbox_x, bbox_y, bbox_w, bbox_h,
      confidence, presence_score,
      COALESCE(is_manual, 0) AS is_manual
    FROM face_rectangles
    WHERE run_id = ? AND file_path IN ({",".join(["?"] * len(paths))})
    ORDER BY file_path ASC, COALESCE(is_manual, 0) ASC, face_index ASC, id ASC
    """
    rows = con.execute(q, [int(face_run_id), *paths]).fetchall()
    by: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by.setdefault(str(r["file_path"]), []).append(r)

    for p in paths:
        rs = by.get(p, [])
        print("\n==", p, "rectangles", len(rs))
        for r in rs:
            conf = r["confidence"]
            pres = r["presence_score"]
            print(
                f"  idx={int(r['face_index'])} manual={int(r['is_manual'])} "
                f"bbox=({int(r['bbox_x'])},{int(r['bbox_y'])},{int(r['bbox_w'])},{int(r['bbox_h'])}) "
                f"conf={None if conf is None else round(float(conf), 4)} "
                f"presence={None if pres is None else round(float(pres), 4)}"
            )

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






