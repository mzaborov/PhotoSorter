from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# Allow running as: python scripts/regression/run_regression_checks.py ...
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from DB.db import DedupStore  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    path: str


def _read_cases_file(p: Path) -> list[str]:
    out: list[str] = []
    txt = p.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in txt:
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _effective_tab_for_row(row: dict) -> str:
    """
    Keep this logic in sync with UI semantics.
    We only use DB fields that exist in current schema.

    Tabs (current):
      - faces
      - no_faces
      - quarantine
      - animals
      - people_no_face
    """
    # manual overrides
    if (row.get("people_no_face_manual") or 0) == 1:
        return "people_no_face"
    manual = (row.get("faces_manual_label") or "").strip()
    if manual == "no_faces":
        return "no_faces"
    if manual == "faces":
        return "faces"

    # auto
    if (row.get("animals_auto") or 0) == 1:
        return "animals"
    if (row.get("faces_auto_quarantine") or 0) == 1:
        return "quarantine"

    # fallback: by faces_count
    fc = int(row.get("faces_count") or 0)
    return "faces" if fc > 0 else "no_faces"


def _bool01(x) -> int:
    try:
        return 1 if int(x or 0) != 0 else 0
    except Exception:
        return 0


def _auto_tab_for_row(row: dict) -> str:
    """
    "Авто-таб": игнорируем любые manual overrides, проверяем только auto-флаги и faces_count.
    Это полезно для регрессии именно авто-эвристик (cats/quarantine).
    """
    if _bool01(row.get("animals_auto")) == 1:
        return "animals"
    if _bool01(row.get("faces_auto_quarantine")) == 1:
        return "quarantine"
    fc = int(row.get("faces_count") or 0)
    return "faces" if fc > 0 else "no_faces"


def _fetch_rows_by_paths(ds: DedupStore, paths: list[str]) -> dict[str, dict]:
    if not paths:
        return {}
    placeholders = ",".join(["?"] * len(paths))
    sql = f"""
        SELECT
          path,
          faces_count,
          faces_manual_label,
          faces_auto_quarantine,
          faces_quarantine_reason,
          animals_auto,
          animals_kind,
          people_no_face_manual
        FROM yd_files
        WHERE path IN ({placeholders})
    """
    rows = ds.conn.execute(sql, paths).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        out[str(r["path"])] = dict(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run regression checks from regression/cases/*.txt")
    ap.add_argument("--cases-dir", default=str(_REPO_ROOT / "regression" / "cases"), help="Directory with cases .txt files")
    ap.add_argument(
        "--mode",
        choices=("effective", "auto"),
        default="effective",
        help="effective: like UI (manual overrides win). auto: only auto flags + faces_count.",
    )
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first mismatch")
    args = ap.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists() or not cases_dir.is_dir():
        print(f"ERROR: cases dir not found: {cases_dir}")
        return 2

    # map file name -> expected tab (positive expectations only)
    expected_by_file = {
        "cats.txt": "animals",
        "quarantine_manual.txt": "quarantine",
        "no_faces.txt": "no_faces",
        "people_no_face.txt": "people_no_face",
        "drawn_faces.txt": "quarantine",
    }

    files = sorted([p for p in cases_dir.glob("*.txt") if p.is_file()])
    if not files:
        print(f"ERROR: no .txt files in {cases_dir}")
        return 2

    ds = DedupStore()
    try:
        total = 0
        mism = 0
        missing = 0

        for p in files:
            exp = expected_by_file.get(p.name)
            if not exp:
                # unknown file: skip, but announce so user can map it later
                print(f"SKIP {p.name}: no expected mapping (add to expected_by_file)")
                continue

            paths = _read_cases_file(p)
            rows = _fetch_rows_by_paths(ds, paths)

            for path in paths:
                total += 1
                row = rows.get(path)
                if row is None:
                    missing += 1
                    print(f"MISS [{p.name}] {path}: not found in DB")
                    if args.fail_fast:
                        return 3
                    continue
                got = _effective_tab_for_row(row) if args.mode == "effective" else _auto_tab_for_row(row)
                if got != exp:
                    mism += 1
                    print(f"FAIL [{p.name}] {path}: expected={exp} got={got} row={row}")
                    if args.fail_fast:
                        return 1

        print(f"OK: mode={args.mode} total={total} mismatches={mism} missing={missing}")
        return 0 if (mism == 0 and missing == 0) else 1
    finally:
        ds.close()


if __name__ == "__main__":
    raise SystemExit(main())


