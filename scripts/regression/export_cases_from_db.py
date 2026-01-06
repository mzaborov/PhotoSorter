from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from DB.db import DedupStore, PipelineStore  # noqa: E402


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

        # cats.txt: animals_auto=1
        cats = q(
            f"SELECT path FROM yd_files WHERE {base_where_sql} AND COALESCE(animals_auto,0)=1 ORDER BY path ASC",
            list(base_params),
        )

        # no_faces.txt: manual no_faces
        no_faces = q(
            f"""
            SELECT path
            FROM yd_files
            WHERE {base_where_sql}
              AND lower(trim(coalesce(faces_manual_label,''))) = 'no_faces'
            ORDER BY path ASC
            """,
            list(base_params),
        )

        # people_no_face.txt: manual people-no-face
        people_no_face = q(
            f"""
            SELECT path
            FROM yd_files
            WHERE {base_where_sql}
              AND COALESCE(people_no_face_manual,0) = 1
            ORDER BY path ASC
            """,
            list(base_params),
        )

        # quarantine_manual.txt: manual quarantine marks (faces_auto_quarantine + reason='manual')
        quarantine_manual = q(
            f"""
            SELECT path
            FROM yd_files
            WHERE {base_where_sql}
              AND COALESCE(faces_auto_quarantine,0) = 1
              AND lower(trim(coalesce(faces_quarantine_reason,''))) = 'manual'
            ORDER BY path ASC
            """,
            list(base_params),
        )
    finally:
        ds.close()

    a_cats = _merge_append_only(out_dir / "cats.txt", cats)
    a_no_faces = _merge_append_only(out_dir / "no_faces.txt", no_faces)
    a_people_no_face = _merge_append_only(out_dir / "people_no_face.txt", people_no_face)
    a_quarantine_manual = _merge_append_only(out_dir / "quarantine_manual.txt", quarantine_manual)

    print(f"UPDATED {out_dir}\\cats.txt +{a_cats} (db_total={len(cats)})")
    print(f"UPDATED {out_dir}\\no_faces.txt +{a_no_faces} (db_total={len(no_faces)})")
    print(f"UPDATED {out_dir}\\people_no_face.txt +{a_people_no_face} (db_total={len(people_no_face)})")
    print(f"UPDATED {out_dir}\\quarantine_manual.txt +{a_quarantine_manual} (db_total={len(quarantine_manual)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


