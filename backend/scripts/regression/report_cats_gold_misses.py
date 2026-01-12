from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Import root for `common.*` is backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


from common.db import DedupStore, PipelineStore, init_db  # noqa: E402


@dataclass(frozen=True)
class MissRow:
    path: str
    got: str
    faces_count: int
    animals_auto: int
    faces_auto_quarantine: int
    faces_quarantine_reason: str
    faces_manual_label: str
    people_no_face_manual: int
    quarantine_manual: int


def _read_cases_file(p: Path) -> list[str]:
    out: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


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


def _bucket_prefix(path: str) -> str:
    """
    Грубый "префикс папки" для локальных путей, чтобы быстро увидеть hot spots.
    Пример: local:C:\\tmp\\Photo\\IMG.jpg -> local:C:\\tmp\\Photo\\
    """
    p = str(path or "")
    if p.startswith("local:"):
        s = p[len("local:") :]
        parts = s.replace("/", "\\").split("\\")
        if len(parts) >= 3:
            return "local:" + "\\".join(parts[:3]) + "\\"
        return "local:" + s
    if p.startswith("disk:"):
        parts = p.split("/")
        if len(parts) >= 3:
            return "/".join(parts[:3]) + "/"
        return p
    return p


def _effective_tab(*, row: dict[str, Any]) -> str:
    """
    Semantics must match /api/faces (run-scoped manual labels).
    """
    if int(row.get("people_no_face_manual") or 0) == 1:
        return "people_no_face"
    manual = str(row.get("faces_manual_label") or "").strip().lower()
    if manual == "faces":
        return "faces"
    if manual == "no_faces":
        return "no_faces"
    fc = int(row.get("faces_count") or 0)
    if int(row.get("quarantine_manual") or 0) == 1 and fc > 0:
        return "quarantine"
    if int(row.get("animals_manual") or 0) == 1:
        return "animals"
    if int(row.get("animals_auto") or 0) == 1:
        return "animals"
    q_reason = str(row.get("faces_quarantine_reason") or "").strip().lower()
    if int(row.get("faces_auto_quarantine") or 0) == 1 and q_reason != "many_small_faces" and fc > 0:
        return "quarantine"
    return "faces" if fc > 0 else "no_faces"


def main() -> int:
    ap = argparse.ArgumentParser(description="Report misses for cats_gold (expected=animals) for a given pipeline_run_id.")
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--cases-dir", default=str(Path("regression") / "cases"))
    ap.add_argument("--out", default=str(Path(".cursor") / "reports" / "cats_gold_misses.txt"))
    ap.add_argument("--limit-prefixes", type=int, default=20)
    ap.add_argument("--limit-samples", type=int, default=5)
    args = ap.parse_args()

    init_db()

    run_id = int(args.pipeline_run_id)
    root_like = _root_like_for_pipeline_run_id(run_id)
    if root_like is None:
        raise SystemExit(f"pipeline_run_id not found: {run_id}")

    cases_dir = Path(args.cases_dir)
    cats_file = cases_dir / "cats_gold.txt"
    if not cats_file.exists():
        raise SystemExit(f"cats_gold not found: {cats_file}")

    cats = _read_cases_file(cats_file)
    # Restrict to run scope (root_like)
    if root_like.endswith("%"):
        pref = root_like[:-1]
        cats = [p for p in cats if str(p).startswith(pref)]

    ds = DedupStore()
    try:
        placeholders = ",".join(["?"] * len(cats)) if cats else "''"
        rows: list[dict[str, Any]] = []
        if cats:
            sql = f"""
                SELECT
                  f.path AS path,
                  COALESCE(f.faces_count, 0) AS faces_count,
                  COALESCE(f.animals_auto, 0) AS animals_auto,
                  COALESCE(f.faces_auto_quarantine, 0) AS faces_auto_quarantine,
                  COALESCE(f.faces_quarantine_reason, '') AS faces_quarantine_reason,
                  COALESCE(m.faces_manual_label, '') AS faces_manual_label,
                  COALESCE(m.people_no_face_manual, 0) AS people_no_face_manual,
                  COALESCE(m.quarantine_manual, 0) AS quarantine_manual,
                  COALESCE(m.animals_manual, 0) AS animals_manual
                FROM files f
                LEFT JOIN files_manual_labels m
                  ON m.pipeline_run_id = ? AND m.path = f.path
                WHERE f.path IN ({placeholders})
            """
            rows = [dict(r) for r in ds.conn.execute(sql, [run_id, *cats]).fetchall()]
        by_path = {str(r.get("path") or ""): r for r in rows}
    finally:
        ds.close()

    total = len(cats)
    missing_in_db = [p for p in cats if p not in by_path]

    got_counter = Counter()
    exp_to_got = Counter()
    fc_bucket = Counter()
    q_reason_counter = Counter()
    screen_like_counter = Counter()
    prefix_miss = Counter()
    prefix_samples: dict[str, list[str]] = defaultdict(list)
    miss_rows: list[MissRow] = []

    for p in cats:
        row = by_path.get(p)
        if not row:
            continue
        got = _effective_tab(row=row)
        if got == "animals":
            continue
        fc = int(row.get("faces_count") or 0)
        got_counter[got] += 1
        exp_to_got[f"animals→{got}"] += 1
        fc_bucket[f"faces_count={fc}"] += 1
        qar = str(row.get("faces_quarantine_reason") or "").strip() or "(none)"
        if int(row.get("faces_auto_quarantine") or 0) == 1:
            q_reason_counter[qar] += 1
            if qar == "screen_like":
                screen_like_counter[got] += 1
        pref = _bucket_prefix(p)
        prefix_miss[pref] += 1
        if len(prefix_samples[pref]) < int(args.limit_samples):
            prefix_samples[pref].append(p)
        miss_rows.append(
            MissRow(
                path=p,
                got=got,
                faces_count=fc,
                animals_auto=int(row.get("animals_auto") or 0),
                faces_auto_quarantine=int(row.get("faces_auto_quarantine") or 0),
                faces_quarantine_reason=str(row.get("faces_quarantine_reason") or ""),
                faces_manual_label=str(row.get("faces_manual_label") or ""),
                people_no_face_manual=int(row.get("people_no_face_manual") or 0),
                quarantine_manual=int(row.get("quarantine_manual") or 0),
            )
        )

    mism = len(miss_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        def w(s: str = "") -> None:
            f.write(s + "\n")

        w(f"cats_gold misses report (effective) for pipeline_run_id={run_id}")
        w(f"cases_dir={cases_dir}")
        w(f"root_like={root_like}")
        w("")
        w(f"cats_gold_total_in_scope={total}")
        w(f"misses_total={mism}")
        w(f"miss_rate={(100.0 * mism / total):.1f}%" if total else "miss_rate=0.0%")
        w(f"missing_in_db={len(missing_in_db)}")
        w("")
        if missing_in_db:
            w("MISSING IN DB (first 20):")
            for p in missing_in_db[:20]:
                w(f"- {p}")
            w("")

        w("Breakdown: expected animals -> got")
        for k, n in exp_to_got.most_common():
            w(f"- {k}: {n} ({(100.0*n/mism):.1f}%)" if mism else f"- {k}: {n}")
        w("")

        w("Breakdown by got (top)")
        for got, n in got_counter.most_common():
            w(f"- got={got}: {n} ({(100.0*n/mism):.1f}%)" if mism else f"- got={got}: {n}")
        w("")

        w("Faces_count buckets (for misses)")
        for k, n in fc_bucket.most_common():
            w(f"- {k}: {n} ({(100.0*n/mism):.1f}%)" if mism else f"- {k}: {n}")
        w("")

        w("Auto-quarantine reasons among misses where faces_auto_quarantine=1")
        total_q = sum(q_reason_counter.values())
        w(f"faces_auto_quarantine=1 among misses: {total_q}/{mism}" if mism else f"faces_auto_quarantine=1 among misses: {total_q}")
        for k, n in q_reason_counter.most_common():
            w(f"- {k}: {n} ({(100.0*n/total_q):.1f}%)" if total_q else f"- {k}: {n}")
        w("")

        w("screen_like among misses (split by got)")
        total_sl = sum(screen_like_counter.values())
        w(f"screen_like among misses: {total_sl}/{mism}" if mism else f"screen_like among misses: {total_sl}")
        for k, n in screen_like_counter.most_common():
            w(f"- got={k}: {n} ({(100.0*n/total_sl):.1f}%)" if total_sl else f"- got={k}: {n}")
        w("")

        w("TOP hotspots by path prefix (misses)")
        for pref, n in prefix_miss.most_common(int(args.limit_prefixes)):
            w(f"- {n} ({(100.0*n/mism):.1f}%)  {pref}" if mism else f"- {n}  {pref}")
            for s in prefix_samples.get(pref, []):
                w(f"  - {s}")
        w("")

        # A few concrete "what to fix" hints based on dominant patterns
        w("SUGGESTED FIX DIRECTIONS (based on stats)")
        if exp_to_got.get("animals→no_faces", 0) > 0:
            w("- Dominant miss: animals→no_faces. Likely animals_auto not set for many cats (YOLO not triggered or thresholding too strict).")
        if total_q and q_reason_counter.get("screen_like", 0) > 0:
            w("- Many misses have faces_auto_quarantine=1 reason=screen_like: consider suppressing screen_like if cat is detected, or run cat detection earlier and clear screen_like on cat.")
        if exp_to_got.get("animals→faces", 0) > 0:
            w("- Some cats go to faces: consider allowing animals_auto even when faces_count>0 (or adjust priority/conditions with person-vs-cat check).")
        w("")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

