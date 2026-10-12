"""
Compares a trained YOLO26 detector with the YOLOv8 model it would replace.

    python training/compare_models.py windows [--photos DIR]
    python training/compare_models.py solar

1. mAP on the dataset's held-out test split (images neither model trained on for YOLO26;
   YOLOv8 was trained elsewhere, so this favours YOLO26 somewhat).
2. Optional: detections per photo on an independent folder of house photos, at the app's
   confidence (0.25), as a sanity check outside the training distribution.
Both models are loaded with the app's safe loader (weights-only unpickling).
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OLD = {"windows": "facade_windows.pt", "solar": "solar_panel_yolov8.pt"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(OLD))
    ap.add_argument("--photos", type=Path, help="folder of extra photos to count detections on")
    args = ap.parse_args()

    os.environ.setdefault("ECO_VISION_WARMUP", "0")
    from backend.engine.safe_models import load_yolo

    data = HERE / "datasets" / args.dataset / "data.yaml"
    models = {
        "YOLOv8 (current)": load_yolo(ROOT / "models" / OLD[args.dataset]),
        "YOLO26 (trained)": load_yolo(ROOT / "models" / f"yolo26_{args.dataset}.pt"),
    }
    print(f"\n== {args.dataset}: held-out test split ==")
    for name, m in models.items():
        r = m.val(data=str(data), split="test", single_cls=True, imgsz=640, device="cpu", plots=False, verbose=False,
                  project=str(HERE / "runs"), name=f"compare-{args.dataset}", exist_ok=True)
        print(f"{name:18s} mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}  precision={r.box.mp:.3f}  recall={r.box.mr:.3f}")

    if args.photos:
        print(f"\n== detections per photo (conf 0.25): {args.photos} ==")
        photos = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        for name, m in models.items():
            t = time.perf_counter()
            counts = [len(m.predict(str(p), conf=0.25, imgsz=640, verbose=False)[0].boxes) for p in photos]
            ms = (time.perf_counter() - t) * 1000 / max(1, len(photos))
            print(f"{name:18s} {counts}  ({ms:.0f} ms/photo)")


if __name__ == "__main__":
    main()
