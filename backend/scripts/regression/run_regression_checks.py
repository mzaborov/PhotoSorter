from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# Allow running as: python backend/scripts/regression/run_regression_checks.py ...
# NOTE: this file is located at backend/scripts/regression/run_regression_checks.py
# parents: [regression, scripts, backend, <repo_root>, ...]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = _REPO_ROOT / "backend"
# Needed for "common.*" imports (common lives under backend/common)
for p in (str(_BACKEND_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


from common.db import DedupStore  # noqa: E402


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
    # manual quarantine (run-scoped)
    fc = int(row.get("faces_count") or 0)
    if (row.get("quarantine_manual") or 0) == 1 and fc > 0:
        return "quarantine"
    # manual animals (ground truth)
    if (row.get("animals_manual") or 0) == 1:
        return "animals"

    # auto
    if (row.get("animals_auto") or 0) == 1:
        return "animals"
    # many_small_faces показываем внутри tab=faces (2-й уровень), не как отдельный top-tab quarantine
    q_reason = (row.get("faces_quarantine_reason") or "").strip().lower()
    # quarantine only if there are faces (faces_count>0); otherwise treat as no_faces
    if (row.get("faces_auto_quarantine") or 0) == 1 and q_reason != "many_small_faces" and fc > 0:
        return "quarantine"

    # fallback: by faces_count
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
    q_reason = (row.get("faces_quarantine_reason") or "").strip().lower()
    fc = int(row.get("faces_count") or 0)
    if _bool01(row.get("faces_auto_quarantine")) == 1 and q_reason != "many_small_faces" and fc > 0:
        return "quarantine"
    return "faces" if fc > 0 else "no_faces"


def _fetch_rows_by_paths(ds: DedupStore, paths: list[str], pipeline_run_id: int | None) -> dict[str, dict]:
    if not paths:
        return {}
    placeholders = ",".join(["?"] * len(paths))
    if pipeline_run_id is not None:
        sql = f"""
            SELECT
              f.path AS path,
              f.faces_count AS faces_count,
              COALESCE(m.faces_manual_label, '') AS faces_manual_label,
              COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
              f.faces_auto_quarantine AS faces_auto_quarantine,
              f.faces_quarantine_reason AS faces_quarantine_reason,
              f.animals_auto AS animals_auto,
              f.animals_kind AS animals_kind,
              COALESCE(m.animals_manual, 0) AS animals_manual,
              COALESCE(m.animals_manual_kind, '') AS animals_manual_kind,
              COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE f.path IN ({placeholders})
        """
        rows = ds.conn.execute(sql, [int(pipeline_run_id), *paths]).fetchall()
    else:
        # No pipeline_run_id -> treat as "no manual labels" to avoid mixing runs.
    sql = f"""
        SELECT
          path,
          faces_count,
              '' AS faces_manual_label,
              0 AS quarantine_manual,
          faces_auto_quarantine,
          faces_quarantine_reason,
          animals_auto,
          animals_kind,
              0 AS animals_manual,
              '' AS animals_manual_kind,
              0 AS people_no_face_manual
        FROM files
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
    ap.add_argument(
        "--pipeline-run-id",
        type=int,
        default=None,
        help="If set, use run-scoped manual labels (files_manual_labels) for effective-mode checks.",
    )
    args = ap.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists() or not cases_dir.is_dir():
        print(f"ERROR: cases dir not found: {cases_dir}")
        return 2

    # map file name -> expected tab (positive expectations only)
    expected_by_file = {
        "cats_gold.txt": "animals",
        "quarantine_gold.txt": "quarantine",
        "faces_gold.txt": "faces",
        "no_faces_gold.txt": "no_faces",
        "people_no_face_gold.txt": "people_no_face",
        "drawn_faces_gold.txt": "quarantine",
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
            rows = _fetch_rows_by_paths(ds, paths, int(args.pipeline_run_id) if args.pipeline_run_id is not None else None)

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


