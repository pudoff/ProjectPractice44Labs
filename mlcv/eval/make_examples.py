"""Copy review images of selected cases from an evaluation output into mlcv/examples as small JPEGs.

    python mlcv/eval/make_examples.py [--src mlcv/eval/out]"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "Шайба.png": ["prep_compare", "overlay", "geometry", "compare"],
    "Шайба.pdf": ["overlay"],
    "Опора.png": ["overlay", "geometry"],
    "control.png.png": ["prep_compare", "overlay", "geometry"],
    "control.scan.jpeg": ["prep_compare", "geometry"],
}
NAMES = {"Шайба.png": "shaiba_png", "Шайба.pdf": "shaiba_pdf", "Опора.png": "opora_png",
         "control.png.png": "control_png", "control.scan.jpeg": "control_scan"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "mlcv/eval/out"))
    ap.add_argument("--max-side", type=int, default=1800)
    a = ap.parse_args()
    dst = ROOT / "mlcv/examples"
    dst.mkdir(exist_ok=True)
    for case, images in CASES.items():
        for name in images:
            src = Path(a.src) / case / f"{name}.png"
            if not src.exists():
                print("missing", src)
                continue
            img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
            f = min(1.0, a.max_side / max(img.shape[:2]))
            if f < 1:
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            out = dst / f"{NAMES[case]}_{name}.jpg"
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(out))
            print(out.relative_to(ROOT), out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
