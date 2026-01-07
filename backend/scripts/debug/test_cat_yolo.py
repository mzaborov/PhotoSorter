from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_path() -> Path:
    return _repo_root() / "data" / "models" / "yolov5s.onnx"


def _letterbox(img_bgr: np.ndarray, *, new_size: int = 640, color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, int, int]:
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return img_bgr, 1.0, 0, 0
    r = min(float(new_size) / float(h), float(new_size) / float(w))
    nh = int(round(h * r))
    nw = int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR)
    pad_w = new_size - nw
    pad_h = new_size - nh
    dw = int(round(pad_w / 2.0))
    dh = int(round(pad_h / 2.0))
    out = cv2.copyMakeBorder(resized, dh, pad_h - dh, dw, pad_w - dw, cv2.BORDER_CONSTANT, value=color)
    return out, r, dw, dh


def _iter_rows(o: np.ndarray) -> np.ndarray:
    # Normalize output to (N, D)
    if o.ndim == 3:
        o = o[0]
        if o.shape[0] in (84, 85):  # (D, N)
            o = o.transpose(1, 0)
    return o if o.ndim == 2 else np.empty((0, 0), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug helper: run YOLO (COCO) and report cat scores on provided images.")
    ap.add_argument("--model", default=str(_model_path()), help="Path to ONNX model (default: data/models/yolov5s.onnx)")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--min-area-ratio", type=float, default=0.002, help="Min box area ratio (box_area / image_area)")
    ap.add_argument("files", nargs="+", help="Image files (absolute or relative paths)")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    import onnxruntime as ort  # type: ignore[import-untyped]

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_type = (sess.get_inputs()[0].type or "").lower()
    use_fp16 = "float16" in input_type

    # COCO: cat class index = 15 (0-based)
    cat_idx = 15

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print("MISSING", p)
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print("BAD_IMAGE", p)
            continue
        h0, w0 = img.shape[:2]
        lb, r, dw, dh = _letterbox(img, new_size=640)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        if use_fp16:
            x = x.astype(np.float16)
        outs = sess.run(None, {input_name: x})
        out = outs[0] if outs else None
        if out is None:
            print("NO_OUTPUT", p)
            continue
        o = _iter_rows(np.asarray(out))
        if o.size == 0:
            print("BAD_SHAPE", p, "shape", getattr(out, "shape", None))
            continue

        d = int(o.shape[1])
        best = 0.0
        best_area_ratio = 0.0
        kept = 0
        for row in o:
            if d == 84:
                x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                conf = float(row[4 + cat_idx])
            elif d == 85:
                x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                obj = float(row[4])
                conf = obj * float(row[5 + cat_idx])
            else:
                continue
            if conf < float(args.conf):
                continue

            # xywh -> xyxy (letterboxed)
            x1 = x_c - bw / 2.0
            y1 = y_c - bh / 2.0
            x2 = x_c + bw / 2.0
            y2 = y_c + bh / 2.0
            # unletterbox -> original
            x1 = (x1 - float(dw)) / float(r)
            y1 = (y1 - float(dh)) / float(r)
            x2 = (x2 - float(dw)) / float(r)
            y2 = (y2 - float(dh)) / float(r)
            x1i = max(0, min(int(round(x1)), w0 - 1))
            y1i = max(0, min(int(round(y1)), h0 - 1))
            x2i = max(0, min(int(round(x2)), w0 - 1))
            y2i = max(0, min(int(round(y2)), h0 - 1))
            if x2i <= x1i or y2i <= y1i:
                continue
            area_ratio = float((x2i - x1i) * (y2i - y1i)) / float(w0 * h0)
            if area_ratio < float(args.min_area_ratio):
                continue
            kept += 1
            if conf > best:
                best = conf
                best_area_ratio = area_ratio

        verdict = "CAT" if kept > 0 else "NO_CAT"
        print(f"{verdict} best={best:.3f} kept={kept} best_area_ratio={best_area_ratio:.4f} file={p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


