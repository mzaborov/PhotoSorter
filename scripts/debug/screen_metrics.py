from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
from PIL import Image, ImageOps  # type: ignore[import-untyped]


def _read_paths(paths_file: Path) -> list[str]:
    out: list[str] = []
    for line in paths_file.read_text(encoding="utf-8").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _strip_local(p: str) -> str:
    return p[len("local:") :] if p.startswith("local:") else p


def screen_metrics(img_bgr_full: np.ndarray) -> dict[str, float]:
    h0, w0 = img_bgr_full.shape[:2]
    ar = float(max(h0, w0)) / float(min(h0, w0)) if min(h0, w0) > 0 else 0.0
    max_dim = 720
    img = img_bgr_full
    m = max(h0, w0)
    if m > max_dim and m > 0:
        sc = float(max_dim) / float(m)
        nh = max(1, int(round(h0 * sc)))
        nw = max(1, int(round(w0 * sc)))
        img = cv2.resize(img_bgr_full, (nw, nh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(g))
    white_ratio = float(np.count_nonzero(g >= 235)) / float(g.size)
    black_ratio = float(np.count_nonzero(g <= 20)) / float(g.size)
    v = float(np.median(g))
    lo = max(0, int(0.66 * v))
    hi = min(255, int(1.33 * v))
    e = cv2.Canny(g, lo, hi)
    er_total = float(np.count_nonzero(e)) / float(e.size)
    h = int(e.shape[0])
    band = max(1, int(round(h * 0.12)))
    top = e[:band, :]
    bot = e[h - band :, :]
    er_top = float(np.count_nonzero(top)) / float(top.size)
    er_bot = float(np.count_nonzero(bot)) / float(bot.size)
    return {
        "ar": ar,
        "mean": mean,
        "white_ratio": white_ratio,
        "black_ratio": black_ratio,
        "er_total": er_total,
        "er_top": er_top,
        "er_bot": er_bot,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Print screen-like edge metrics for paths.")
    ap.add_argument("--paths-file", required=True)
    args = ap.parse_args()

    paths = _read_paths(Path(args.paths_file))
    for p in paths:
        if not p.startswith("local:"):
            print("skip", p)
            continue
        fp = Path(_strip_local(p))
        if not fp.exists():
            print("MISSING", p)
            continue
        with Image.open(fp) as im0:
            im = ImageOps.exif_transpose(im0).convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        bgr = arr[:, :, ::-1].copy()
        m = screen_metrics(bgr)
        print(
            f"{p} ar={m['ar']:.3f} mean={m['mean']:.1f} white={m['white_ratio']:.3f} black={m['black_ratio']:.3f} "
            f"er_total={m['er_total']:.4f} er_top={m['er_top']:.4f} er_bot={m['er_bot']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


