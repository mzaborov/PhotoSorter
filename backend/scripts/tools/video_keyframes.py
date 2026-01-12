from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _video_duration_seconds(cap) -> float | None:
    try:
        import cv2  # noqa: WPS433

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0.0 and n > 0.0:
            return float(n) / float(fps)
    except Exception:
        return None
    return None


def _pick_times(dur: float, *, samples: int) -> list[float]:
    d = float(max(0.0, dur))
    if d <= 0.0:
        return [0.0]
    if d < 2.0 or samples <= 1:
        return [min(0.5 * d, max(0.0, d - 0.5))]
    t1 = min(max(1.0, 0.05 * d), max(0.0, d - 0.5))
    t2 = min(0.5 * d, max(0.0, d - 0.5))
    if samples == 2:
        return [t1, t2]
    t3 = min(0.95 * d, max(0.0, d - 0.5))
    out: list[float] = []
    for t in (t1, t2, t3):
        if not out or abs(out[-1] - t) > 0.05:
            out.append(t)
    return out


def _extract_frame(*, path: str, t_sec: float, out_path: str, max_dim: int) -> None:
    import cv2  # noqa: WPS433

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError("video_open_failed")
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("frame_read_failed")
        h, w = frame.shape[:2]
        md = int(max_dim or 0)
        if md and md > 0 and (w > md or h > md):
            if w >= h:
                nw = md
                nh = int(round((h / w) * nw))
            else:
                nh = md
                nw = int(round((w / h) * nh))
            if nw >= 2 and nh >= 2:
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        ok2 = cv2.imwrite(str(outp), frame)
        if not ok2:
            raise RuntimeError("frame_write_failed")
    finally:
        try:
            cap.release()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute keyframe times and/or extract keyframes for a local video.")
    ap.add_argument("--path", required=True, help="Absolute local path to video file")
    ap.add_argument("--samples", type=int, default=3, help="1..3")
    ap.add_argument("--mode", choices=("meta", "extract"), default="meta")
    ap.add_argument("--frame-idx", type=int, default=1, help="1..3 (for extract)")
    ap.add_argument("--max-dim", type=int, default=960)
    ap.add_argument("--out", default="", help="Output image path (for extract)")
    args = ap.parse_args()

    p = str(args.path or "")
    if not p:
        raise SystemExit("empty --path")
    if not os.path.isfile(p):
        raise SystemExit(f"not found: {p}")
    samples = max(1, min(3, int(args.samples or 3)))

    import cv2  # noqa: WPS433

    cap = cv2.VideoCapture(p)
    try:
        if not cap.isOpened():
            raise SystemExit("video_open_failed")
        dur = _video_duration_seconds(cap) or 0.0
    finally:
        try:
            cap.release()
        except Exception:
            pass

    times = _pick_times(float(dur), samples=samples)
    # normalize to 3 slots (idx=1..3) when samples=3; for 1/2 we still return only those
    if args.mode == "meta":
        print(json.dumps({"ok": True, "duration_sec": float(dur), "samples": samples, "times_sec": times}, ensure_ascii=False))
        return 0

    idx = int(args.frame_idx or 0)
    if idx <= 0:
        raise SystemExit("bad --frame-idx")
    if idx > len(times):
        # for safety: clamp to last available time
        idx = len(times)
    if not args.out:
        raise SystemExit("missing --out")
    t = float(times[idx - 1])
    _extract_frame(path=p, t_sec=t, out_path=str(args.out), max_dim=int(args.max_dim or 0))
    print(json.dumps({"ok": True, "frame_idx": int(idx), "t_sec": float(t), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

