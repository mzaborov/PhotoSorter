from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from common.db import DedupStore, PipelineStore  # noqa: E402


def _root_like_for_pipeline_run(pipeline_run_id: int) -> str | None:
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise SystemExit(f"pipeline_run_id not found: {pipeline_run_id}")
    root_path = str(pr.get("root_path") or "")
    if not root_path:
        return None
    if root_path.startswith("disk:"):
        rp = root_path.rstrip("/")
        return rp + "/%"
    rp_abs = os.path.abspath(root_path).rstrip("\\/") + "\\"
    return "local:" + rp_abs + "%"


def _read_existing(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _merge_append_only(path: Path, new_items: list[str]) -> int:
    """
    Не затираем старое содержимое: дописываем только новые пути (которых ещё не было).
    Возвращает число добавленных строк.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(path)
    seen = set(existing)
    added = 0
    for it in new_items:
        s = (it or "").strip()
        if not s or s in seen:
            continue
        existing.append(s)
        seen.add(s)
        added += 1
    path.write_text("\n".join(existing) + ("\n" if existing else ""), encoding="utf-8")
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description="Export regression cases (*.txt) from SQLite")
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "regression" / "cases"))
    args = ap.parse_args()

    root_like = _root_like_for_pipeline_run(int(args.pipeline_run_id))
    out_dir = Path(args.out_dir)

    ds = DedupStore()
    try:
        cur = ds.conn.cursor()

        def q(sql: str, params: list[object]) -> list[str]:
            cur.execute(sql, params)
            return [str(r[0]) for r in cur.fetchall()]

        base_where = ["status != 'deleted'"]
        base_params: list[object] = []
        if root_like:
            base_where.append("path LIKE ?")
            base_params.append(root_like)
        base_where_sql = " AND ".join(base_where)

        # cats_gold.txt: animals_auto=1 (gold)
        cats = q(
            f"SELECT path FROM files WHERE {base_where_sql} AND COALESCE(animals_auto,0)=1 ORDER BY path ASC",
            list(base_params),
        )

        # faces_gold.txt: manually marked as "faces" (Нормальные лица)
        faces_gold = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND lower(trim(coalesce(faces_manual_label,''))) = 'faces'
            ORDER BY path ASC
            """,
            list(base_params),
        )

        # no_faces_gold.txt: gold: no faces (auto semantics depends on regression strategy)
        no_faces = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND lower(trim(coalesce(faces_manual_label,''))) = 'no_faces'
            ORDER BY path ASC
            """,
            list(base_params),
        )

        # people_no_face_gold.txt: gold: people no face (manual list for now)
        people_no_face = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND COALESCE(people_no_face_manual,0) = 1
            ORDER BY path ASC
            """,
            list(base_params),
        )

        # quarantine_gold.txt: gold quarantine list (currently sourced from manual quarantine marks in DB)
        quarantine_gold = q(
            f"""
            SELECT path
            FROM files
            WHERE {base_where_sql}
              AND COALESCE(faces_auto_quarantine,0) = 1
              AND lower(trim(coalesce(faces_quarantine_reason,''))) = 'manual'
            ORDER BY path ASC
            """,
            list(base_params),
        )
    finally:
        ds.close()

    a_cats = _merge_append_only(out_dir / "cats_gold.txt", cats)
    a_faces = _merge_append_only(out_dir / "faces_gold.txt", faces_gold)
    a_no_faces = _merge_append_only(out_dir / "no_faces_gold.txt", no_faces)
    a_people_no_face = _merge_append_only(out_dir / "people_no_face_gold.txt", people_no_face)
    a_quarantine_gold = _merge_append_only(out_dir / "quarantine_gold.txt", quarantine_gold)

    print(f"UPDATED {out_dir}\\cats_gold.txt +{a_cats} (db_total={len(cats)})")
    print(f"UPDATED {out_dir}\\faces_gold.txt +{a_faces} (db_total={len(faces_gold)})")
    print(f"UPDATED {out_dir}\\no_faces_gold.txt +{a_no_faces} (db_total={len(no_faces)})")
    print(f"UPDATED {out_dir}\\people_no_face_gold.txt +{a_people_no_face} (db_total={len(people_no_face)})")
    print(f"UPDATED {out_dir}\\quarantine_gold.txt +{a_quarantine_gold} (db_total={len(quarantine_gold)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


