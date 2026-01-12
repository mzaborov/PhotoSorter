from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
import sys


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _pct(n: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * float(n) / float(total)):.1f}%"


def _bucket_prefix(path: str) -> str:
    """
    Грубый "префикс папки" для локальных путей, чтобы быстро увидеть hot spots.
    Пример: local:C:\\tmp\\Photo\\IMG.jpg -> local:C:\\tmp\\Photo\\
    """
    p = str(path or "")
    if p.startswith("local:"):
        s = p[len("local:") :]
        # оставить 3 сегмента максимум
        parts = s.replace("/", "\\").split("\\")
        if len(parts) >= 3:
            return "local:" + "\\".join(parts[:3]) + "\\"
        return "local:" + s
    if p.startswith("disk:"):
        # disk:/Фото/.. -> disk:/Фото/
        parts = p.split("/")
        if len(parts) >= 3:
            return "/".join(parts[:3]) + "/"
        return p
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Animals stats for a given pipeline_run_id.\n"
            "Focus: cat <-> no_faces ошибок. Uses animals_manual as ground truth and animals_auto as model output."
        )
    )
    ap.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--limit-prefixes", type=int, default=15)
    args = ap.parse_args()

    # Ensure DB schema is up-to-date (adds missing columns safely).
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from common.db import init_db  # type: ignore

        init_db()
    except Exception:
        # best-effort: stats can still run if schema is already up-to-date
        pass

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = _connect(db)
    try:
        cur = conn.cursor()
        pr = cur.execute(
            "SELECT id, kind, root_path, face_run_id FROM pipeline_runs WHERE id=? LIMIT 1",
            (int(args.pipeline_run_id),),
        ).fetchone()
        if not pr:
            raise SystemExit(f"pipeline_run_id not found: {args.pipeline_run_id}")
        face_run_id = int(pr["face_run_id"] or 0)
        if not face_run_id:
            raise SystemExit("face_run_id is not set on this pipeline run yet")

        # Only files seen by this run (faces_run_id == face_run_id)
        rows = cur.execute(
            """
            SELECT
              f.path AS path,
              COALESCE(f.faces_count, 0) AS faces_count,
              COALESCE(f.animals_auto, 0) AS animals_auto,
              COALESCE(f.animals_kind, '') AS animals_kind,
              COALESCE(m.animals_manual, 0) AS animals_manual,
              COALESCE(m.animals_manual_kind, '') AS animals_manual_kind
            FROM files f
            LEFT JOIN files_manual_labels m
              ON m.pipeline_run_id = ? AND m.path = f.path
            WHERE f.faces_run_id = ?
              AND f.status != 'deleted'
            """,
            (int(args.pipeline_run_id), int(face_run_id)),
        ).fetchall()

        total = len(rows)
        print("pipeline_run:", dict(pr))
        print("rows_total:", total)
        if total == 0:
            return 0

        # Ground truth: manual cat
        gt_total = 0
        gt_detected = 0  # animals_auto=1
        gt_missed_to_no_faces = 0  # animals_auto=0 AND faces_count=0
        gt_missed_to_faces = 0  # animals_auto=0 AND faces_count>0 (YuNet false faces etc.)

        # Model positives without GT (likely false positives, or unlabeled GT)
        pred_total = 0
        pred_unlabeled = 0  # animals_auto=1 but animals_manual=0

        # Hotspots: where cats are missed into no_faces
        miss_prefix = Counter()
        miss_samples: dict[str, list[str]] = defaultdict(list)

        for r in rows:
            path = str(r["path"] or "")
            fc = int(r["faces_count"] or 0)
            pred = int(r["animals_auto"] or 0) == 1
            gt = int(r["animals_manual"] or 0) == 1

            if pred:
                pred_total += 1
                if not gt:
                    pred_unlabeled += 1

            if not gt:
                continue

            gt_total += 1
            if pred:
                gt_detected += 1
                continue

            # missed
            if fc <= 0:
                gt_missed_to_no_faces += 1
                pref = _bucket_prefix(path)
                miss_prefix[pref] += 1
                if len(miss_samples[pref]) < 5:
                    miss_samples[pref].append(path)
            else:
                gt_missed_to_faces += 1

        print("\nGROUND TRUTH (animals_manual=1):", gt_total)
        print("  recall (auto detects GT cats):", f"{gt_detected}/{gt_total} ({_pct(gt_detected, gt_total)})")
        print(
            "  misses -> no_faces (core 'cat<->no_faces' error):",
            f"{gt_missed_to_no_faces}/{gt_total} ({_pct(gt_missed_to_no_faces, gt_total)})",
        )
        print("  misses -> faces (cat but faces_count>0):", f"{gt_missed_to_faces}/{gt_total} ({_pct(gt_missed_to_faces, gt_total)})")

        print("\nMODEL POSITIVES (animals_auto=1):", pred_total)
        print(
            "  positives without GT label (animals_manual=0):",
            f"{pred_unlabeled}/{pred_total} ({_pct(pred_unlabeled, pred_total)})",
            " (может быть FP, а может 'ещё не размечено')",
        )

        if gt_missed_to_no_faces > 0:
            print("\nTOP hotspots for misses -> no_faces (by path prefix):")
            for pref, n in miss_prefix.most_common(int(args.limit_prefixes)):
                print(f"  {n} ({_pct(n, gt_missed_to_no_faces)})  {pref}")
                for s in miss_samples.get(pref, []):
                    print(f"    - {s}")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

