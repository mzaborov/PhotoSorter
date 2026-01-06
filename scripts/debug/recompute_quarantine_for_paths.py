from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
from PIL import Image, ImageOps  # type: ignore[import-untyped]


# allow running as: python scripts/debug/recompute_quarantine_for_paths.py ...
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DB.db import DedupStore, PipelineStore, FaceStore  # noqa: E402


def _looks_like_screen_ui(img_bgr_full: np.ndarray) -> bool:
    h0, w0 = img_bgr_full.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return False
    ar = float(max(h0, w0)) / float(min(h0, w0))
    if not (1.00 <= ar <= 2.60):
        return False
    # Downscale
    max_dim = 720
    img = img_bgr_full
    m = max(h0, w0)
    if m > max_dim:
        sc = float(max_dim) / float(m)
        nh = max(1, int(round(h0 * sc)))
        nw = max(1, int(round(w0 * sc)))
        img = cv2.resize(img_bgr_full, (nw, nh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.count_nonzero(g >= 235)) / float(g.size)
    black_ratio = float(np.count_nonzero(g <= 20)) / float(g.size)
    v = float(np.median(g))
    lo = max(0, int(0.66 * v))
    hi = min(255, int(1.33 * v))
    e = cv2.Canny(g, lo, hi)
    er_total = float(np.count_nonzero(e)) / float(e.size)
    h = int(e.shape[0])
    top = e[: max(1, int(round(h * 0.12))), :]
    bot = e[h - max(1, int(round(h * 0.12))) :, :]
    er_top = float(np.count_nonzero(top)) / float(top.size)
    er_bot = float(np.count_nonzero(bot)) / float(bot.size)
    if er_total >= 0.085:
        return True
    if er_total >= 0.040 and (er_top >= 0.040 or er_bot >= 0.030):
        return True
    if er_total >= 0.035 and er_bot >= 0.050:
        return True
    if er_total >= 0.010 and (er_top >= 0.040 and er_bot >= 0.030):
        return True
    if er_total >= 0.025 and (white_ratio >= 0.10 and black_ratio >= 0.05):
        return True
    if er_total >= 0.025 and white_ratio >= 0.18:
        return True
    if er_total >= 0.045 and black_ratio >= 0.15:
        return True
    if er_total >= 0.055 and ar >= 1.60 and black_ratio >= 0.03:
        return True
    if er_total >= 0.020 and _looks_like_text_heavy(g):
        return True
    return False


def _looks_like_text_heavy(gray: np.ndarray) -> bool:
    try:
        if gray.ndim != 2:
            return False
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return False
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            5,
        )
        bw = cv2.medianBlur(bw, 3)
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        small = 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 10 or area > 400:
                continue
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if ww <= 0 or hh <= 0:
                continue
            ar = float(ww) / float(hh)
            if 0.15 <= ar <= 8.0:
                small += 1
        return small >= 140
    except Exception:
        return False


def _qr_found_best_effort(img_bgr_full: np.ndarray, *, qr, qr_max_dim: int = 1200) -> bool:
    if qr is None:
        return False
    h0, w0 = img_bgr_full.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return False

    def _try_one(qimg: np.ndarray) -> bool:
        try:
            if hasattr(qr, "detectAndDecodeMulti"):
                _ok, decoded, _points, _ = qr.detectAndDecodeMulti(qimg)
                return bool(decoded and any(str(x or "").strip() for x in decoded))
            decoded, _points, _ = qr.detectAndDecode(qimg)
            return bool(str(decoded or "").strip())
        except Exception:
            return False

    for target in (qr_max_dim, 900):
        qimg = img_bgr_full
        mh = max(int(h0), int(w0))
        if mh > target and mh > 0:
            sc = float(target) / float(mh)
            qh = max(1, int(round(h0 * sc)))
            qw = max(1, int(round(w0 * sc)))
            qimg = cv2.resize(img_bgr_full, (qw, qh), interpolation=cv2.INTER_AREA)
        if _try_one(qimg):
            return True
        try:
            g = cv2.cvtColor(qimg, cv2.COLOR_BGR2GRAY)
        except Exception:
            g = None
        if g is not None and _try_one(g):
            return True
        if g is not None:
            try:
                th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
                if _try_one(th):
                    return True
            except Exception:
                pass
    return False


def _read_paths(paths_file: Path) -> list[str]:
    paths: list[str] = []
    for line in paths_file.read_text(encoding="utf-8").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        paths.append(s)
    return paths


def _strip_local(p: str) -> str:
    return p[len("local:") :] if p.startswith("local:") else p


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute faces_auto_quarantine for given paths (qr/screen_like/many_small_faces).")
    ap.add_argument("--paths-file", required=True)
    ap.add_argument("--pipeline-run-id", type=int, required=True, help="Used to resolve face_run_id for bbox-based rules")
    ap.add_argument("--apply", action="store_true", help="Actually update DB (default: dry-run print only)")
    args = ap.parse_args()

    paths_file = Path(args.paths_file)
    paths = _read_paths(paths_file)
    if not paths:
        print("No paths.")
        return 0

    ps = PipelineStore()
    pr = ps.get_run_by_id(run_id=int(args.pipeline_run_id))
    ps.close()
    if not pr:
        raise SystemExit(f"pipeline_run_id not found: {args.pipeline_run_id}")
    face_run_id = int(pr.get("face_run_id") or 0)
    if not face_run_id:
        raise SystemExit("face_run_id is not set for pipeline run")

    fs = FaceStore()
    ds = DedupStore()
    qr = None
    try:
        qr = cv2.QRCodeDetector()
    except Exception:
        qr = None

    # bbox-based thresholds (match pipeline defaults)
    many_faces_min = 6
    many_faces_max_dim_px = 80

    try:
        for p in paths:
            if not p.startswith("local:"):
                print("skip non-local", p)
                continue
            abspath = _strip_local(p)
            if not Path(abspath).exists():
                print("missing file", p)
                continue

            # read image
            try:
                with Image.open(abspath) as im0:
                    im = ImageOps.exif_transpose(im0)
                    im = im.convert("RGB")
                    arr = np.asarray(im, dtype=np.uint8)
                bgr = arr[:, :, ::-1].copy()
            except Exception as e:  # noqa: BLE001
                print("read_error", p, f"{type(e).__name__}: {e}")
                continue

            # bbox-based many_small_faces
            try:
                rects = fs.list_rectangles(run_id=face_run_id, file_path=p)
                auto = [r for r in rects if int(r.get("is_manual") or 0) == 0]
                max_dim = 0
                for r in auto:
                    max_dim = max(max_dim, int(max(int(r.get("bbox_w") or 0), int(r.get("bbox_h") or 0))))
                many_small = (len(auto) >= many_faces_min and max_dim <= many_faces_max_dim_px)
            except Exception:
                many_small = False

            qr_found = _qr_found_best_effort(bgr, qr=qr)
            screen_like = _looks_like_screen_ui(bgr)

            is_q = False
            reason = None
            if qr_found:
                is_q = True
                reason = "qr"
            elif screen_like:
                is_q = True
                reason = "screen_like"
            elif many_small:
                is_q = True
                reason = "many_small_faces"

            if args.apply:
                ds.set_faces_auto_quarantine(path=p, is_quarantine=bool(is_q), reason=reason)
            print(("SET" if args.apply else "DRY"), p, "->", int(is_q), reason)
    finally:
        ds.close()
        fs.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


