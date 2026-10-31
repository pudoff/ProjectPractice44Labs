"""Algorithmic smoke, no OCR or weights. Not evidence of end-to-end OCR."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from runs.klapan_png2dxf.pipeline_b import pipeline
import ocr

outdir = ROOT / "logs" / "smoke-no-ocr"
outdir.mkdir(parents=True, exist_ok=True)
img = cv2.imread(str(ROOT / "runs" / "клапан.png"), cv2.IMREAD_GRAYSCALE)
assert img is not None
new_w = 800
preview = cv2.resize(img, (new_w, round(img.shape[0] * new_w / img.shape[1])),
                     interpolation=cv2.INTER_AREA)
input_path = outdir / "input-preview.png"
cv2.imwrite(str(input_path), preview)
pipeline.run_ocr = lambda gray, cal: []
ocr.ocr_window = lambda *args, **kwargs: []
result = pipeline.run(str(input_path), str(outdir / "output.dxf"),
                      report_path=str(outdir / "report.json"))
print(json.dumps({"total_s": result["total_s"],
                  "counts": result["counts"],
                  "verify": result["verify"]}, ensure_ascii=False))
