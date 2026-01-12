from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Import root for `common.*` is backend/ (not repo root).
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


from common.db import DedupStore, PipelineStore, init_db  # noqa: E402


def _root_like_for_pipeline_run_id(pipeline_run_id: int) -> str | None:
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        return None
    root_path = str(pr.get("root_path") or "")
    if not root_path:
        return None
    if root_path.startswith("disk:"):
        rp = root_path.rstrip("/")
        return rp + "/%"
    rp_abs = os.path.abspath(root_path).rstrip("\\/") + "\\"
    return "local:" + rp_abs + "%"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Migrate legacy manual labels stored in files.* into run-scoped files_manual_labels for a given pipeline_run_id.\n"
            "By default: dry-run (prints counts). Use --apply to write."
        )
    )
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--apply", action="store_true", help="Actually write to DB (otherwise dry-run)")
    ap.add_argument("--limit-samples", type=int, default=20)
    args = ap.parse_args()

    init_db()

    pipeline_run_id = int(args.pipeline_run_id)
    root_like = _root_like_for_pipeline_run_id(pipeline_run_id)
    if root_like is None:
        raise SystemExit(f"pipeline_run_id not found: {pipeline_run_id}")

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        cur.execute(
            """
            SELECT
              path,
              COALESCE(faces_manual_label, '') AS faces_manual_label,
              COALESCE(people_no_face_manual, 0) AS people_no_face_manual,
              COALESCE(people_no_face_person, '') AS people_no_face_person,
              COALESCE(animals_manual, 0) AS animals_manual,
              COALESCE(animals_manual_kind, '') AS animals_manual_kind,
              COALESCE(faces_auto_quarantine, 0) AS faces_auto_quarantine,
              COALESCE(faces_quarantine_reason, '') AS faces_quarantine_reason
            FROM files
            WHERE status != 'deleted'
              AND path LIKE ?
              AND (
                lower(trim(coalesce(faces_manual_label,''))) IN ('faces','no_faces')
                OR COALESCE(people_no_face_manual, 0) = 1
                OR COALESCE(animals_manual, 0) = 1
                OR (COALESCE(faces_auto_quarantine,0)=1 AND lower(trim(coalesce(faces_quarantine_reason,'')))='manual')
              )
            ORDER BY path ASC
            """,
            (root_like,),
        )
        rows = [dict(r) for r in cur.fetchall()]

        to_set_faces = 0
        to_set_people = 0
        to_set_animals = 0
        to_set_quarantine = 0

        samples: list[str] = []

        for r in rows:
            path = str(r.get("path") or "")
            if not path:
                continue
            faces_lab = str(r.get("faces_manual_label") or "").strip().lower()
            if faces_lab in ("faces", "no_faces"):
                to_set_faces += 1
            if int(r.get("people_no_face_manual") or 0) == 1:
                to_set_people += 1
            if int(r.get("animals_manual") or 0) == 1:
                to_set_animals += 1
            if int(r.get("faces_auto_quarantine") or 0) == 1 and str(r.get("faces_quarantine_reason") or "").strip().lower() == "manual":
                to_set_quarantine += 1
            if len(samples) < int(args.limit_samples):
                samples.append(path)

        print(f"pipeline_run_id={pipeline_run_id}")
        print(f"root_like={root_like}")
        print(f"rows_with_legacy_manual_any={len(rows)}")
        print(f"legacy_faces_manual_label={to_set_faces}")
        print(f"legacy_people_no_face_manual={to_set_people}")
        print(f"legacy_animals_manual={to_set_animals}")
        print(f"legacy_quarantine_manual_by_reason={to_set_quarantine}")
        if samples:
            print("\nSAMPLES:")
            for s in samples:
                print(f"- {s}")

        if not args.apply:
            print("\nDRY-RUN (no changes). Use --apply to write.")
            return 0

        applied = 0
        for r in rows:
            path = str(r.get("path") or "")
            if not path:
                continue

            # Ensure row exists
            cur.execute(
                "INSERT OR IGNORE INTO files_manual_labels(pipeline_run_id, path) VALUES (?, ?)",
                (pipeline_run_id, path),
            )

            faces_lab = str(r.get("faces_manual_label") or "").strip().lower()
            if faces_lab in ("faces", "no_faces"):
                cur.execute(
                    """
                    UPDATE files_manual_labels
                    SET faces_manual_label = ?, faces_manual_at = COALESCE(faces_manual_at, CURRENT_TIMESTAMP)
                    WHERE pipeline_run_id = ? AND path = ?
                      AND (faces_manual_label IS NULL OR trim(coalesce(faces_manual_label,'')) = '')
                    """,
                    (faces_lab, pipeline_run_id, path),
                )
                applied += int(cur.rowcount or 0)

            if int(r.get("people_no_face_manual") or 0) == 1:
                cur.execute(
                    """
                    UPDATE files_manual_labels
                    SET people_no_face_manual = 1, people_no_face_person = ?
                    WHERE pipeline_run_id = ? AND path = ?
                      AND COALESCE(people_no_face_manual, 0) = 0
                    """,
                    ((str(r.get("people_no_face_person") or "").strip() or None), pipeline_run_id, path),
                )
                applied += int(cur.rowcount or 0)

            if int(r.get("animals_manual") or 0) == 1:
                kind = str(r.get("animals_manual_kind") or "").strip() or None
                cur.execute(
                    """
                    UPDATE files_manual_labels
                    SET animals_manual = 1, animals_manual_kind = ?, animals_manual_at = COALESCE(animals_manual_at, CURRENT_TIMESTAMP)
                    WHERE pipeline_run_id = ? AND path = ?
                      AND COALESCE(animals_manual, 0) = 0
                    """,
                    (kind, pipeline_run_id, path),
                )
                applied += int(cur.rowcount or 0)

            if int(r.get("faces_auto_quarantine") or 0) == 1 and str(r.get("faces_quarantine_reason") or "").strip().lower() == "manual":
                cur.execute(
                    """
                    UPDATE files_manual_labels
                    SET quarantine_manual = 1, quarantine_manual_at = COALESCE(quarantine_manual_at, CURRENT_TIMESTAMP)
                    WHERE pipeline_run_id = ? AND path = ?
                      AND COALESCE(quarantine_manual, 0) = 0
                    """,
                    (pipeline_run_id, path),
                )
                applied += int(cur.rowcount or 0)

        ds.conn.commit()
        print(f"\nAPPLIED updates={applied}")
        return 0
    finally:
        ds.close()


if __name__ == "__main__":
    raise SystemExit(main())

