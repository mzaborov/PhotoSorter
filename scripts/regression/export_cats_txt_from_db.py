from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from DB.db import DedupStore, PipelineStore  # noqa: E402


def _detect_root_like(pipeline_run_id: int) -> str | None:
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
    ap = argparse.ArgumentParser(description="Export cats list (animals_auto=1) from SQLite to regression/cases/cats.txt")
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--out", default=str(_REPO_ROOT / "regression" / "cases" / "cats.txt"))
    args = ap.parse_args()

    root_like = _detect_root_like(int(args.pipeline_run_id))
    ds = DedupStore()
    try:
        cur = ds.conn.cursor()
        where = ["status != 'deleted'", "COALESCE(animals_auto,0) = 1"]
        params: list[object] = []
        if root_like:
            where.append("path LIKE ?")
            params.append(root_like)
        cur.execute(
            f"""
            SELECT path
            FROM yd_files
            WHERE {' AND '.join(where)}
            ORDER BY path ASC
            """,
            params,
        )
        paths = [str(r["path"]) for r in cur.fetchall()]
    finally:
        ds.close()

    out = Path(args.out)
    added = _merge_append_only(out, paths)
    print(f"UPDATED {out} +{added} (db_total={len(paths)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


