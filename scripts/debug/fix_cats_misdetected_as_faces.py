from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
from PIL import Image, ImageOps  # type: ignore[import-untyped]


# run as: .venv-face\\Scripts\\python.exe scripts/debug/fix_cats_misdetected_as_faces.py ...
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DB.db import DedupStore, FaceStore, PipelineStore  # noqa: E402


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
    if o.ndim == 3:
        o = o[0]
        if o.shape[0] in (84, 85):
            o = o.transpose(1, 0)
    return o if o.ndim == 2 else np.empty((0, 0), dtype=np.float32)


def _yolo_has_class(*, sess, input_name: str, img_bgr_full: np.ndarray, class_idx: int, conf_threshold: float, min_area_ratio: float) -> bool:
    h0, w0 = img_bgr_full.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return False
    lb, r, dw, dh = _letterbox(img_bgr_full, new_size=640)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    try:
        in_type = (sess.get_inputs()[0].type or "").lower()
        if "float16" in in_type:
            x = x.astype(np.float16)
    except Exception:
        pass
    outs = sess.run(None, {input_name: x})
    if not outs:
        return False
    o = _iter_rows(np.asarray(outs[0]))
    if o.size == 0:
        return False
    d = int(o.shape[1])
    for row in o:
        if d == 84:
            x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            conf = float(row[4 + class_idx])
        elif d == 85:
            x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            obj = float(row[4])
            conf = obj * float(row[5 + class_idx])
        else:
            continue
        if conf < float(conf_threshold):
            continue
        x1 = x_c - bw / 2.0
        y1 = y_c - bh / 2.0
        x2 = x_c + bw / 2.0
        y2 = y_c + bh / 2.0
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
        if area_ratio < float(min_area_ratio):
            continue
        return True
    return False


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix: cats misdetected as faces by YuNet (set animals_auto and clear auto face rectangles).")
    ap.add_argument("--paths-file", required=True)
    ap.add_argument("--pipeline-run-id", type=int, required=True)
    ap.add_argument("--model", default=str(_REPO_ROOT / "data" / "models" / "yolov5s.onnx"))
    ap.add_argument("--cat-conf", type=float, default=0.35)
    ap.add_argument("--person-conf", type=float, default=0.35)
    ap.add_argument("--min-area-ratio", type=float, default=0.002)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    paths = _read_paths(Path(args.paths_file))
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

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"YOLO model not found: {model}")

    import onnxruntime as ort  # type: ignore[import-untyped]

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    ds = DedupStore()
    fs = FaceStore()
    try:
        for p in paths:
            if not p.startswith("local:"):
                print("skip non-local", p)
                continue
            abspath = _strip_local(p)
            if not Path(abspath).exists():
                print("missing", p)
                continue
            try:
                with Image.open(abspath) as im0:
                    im = ImageOps.exif_transpose(im0).convert("RGB")
                    arr = np.asarray(im, dtype=np.uint8)
                bgr = arr[:, :, ::-1].copy()
            except Exception as e:  # noqa: BLE001
                print("read_error", p, f"{type(e).__name__}: {e}")
                continue

            has_cat = _yolo_has_class(
                sess=sess,
                input_name=input_name,
                img_bgr_full=bgr,
                class_idx=15,  # COCO cat
                conf_threshold=float(args.cat_conf),
                min_area_ratio=float(args.min_area_ratio),
            )
            has_person = _yolo_has_class(
                sess=sess,
                input_name=input_name,
                img_bgr_full=bgr,
                class_idx=0,  # COCO person
                conf_threshold=float(args.person_conf),
                min_area_ratio=float(args.min_area_ratio),
            )

            verdict = "KEEP"  # default: do nothing
            if has_cat and not has_person:
                verdict = "CAT"

            if args.apply and verdict == "CAT":
                # 1) clear auto face rectangles (they are actually cat faces)
                cur = fs.conn.cursor()
                cur.execute(
                    "DELETE FROM face_rectangles WHERE run_id = ? AND file_path = ? AND COALESCE(is_manual, 0) = 0",
                    (int(face_run_id), str(p)),
                )
                fs.conn.commit()
                # 2) set faces_count=0, keep faces_run_id/scanned_at as is
                cur2 = ds.conn.cursor()
                cur2.execute("UPDATE files SET faces_count = 0 WHERE path = ?", (str(p),))
                ds.conn.commit()
                # 3) set animals
                ds.set_animals_auto(path=str(p), is_animal=True, kind="cat")

            print(("SET" if args.apply else "DRY"), p, "cat", int(has_cat), "person", int(has_person), "->", verdict)
    finally:
        fs.close()
        ds.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())






