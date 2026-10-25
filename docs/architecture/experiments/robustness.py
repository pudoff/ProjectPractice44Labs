"""Robustness of pipeline B: (1) the test sheet degraded like an archive scan, (2) synthetic scanned PDFs.

Degradations (seeded): 200 dpi, rotation, blur, noise, uneven background, JPEG q=55;
variant 'cut' additionally crops the image inside the sheet trim line (paper edge, no drawn border),
which is how real archive scans look.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
PIPE = ROOT / "runs/klapan_png2dxf/pipeline_b/pipeline.py"
CMP = ROOT / "tools/compare_dxf.py"
OUT = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "robustness")


def load(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)


def save(p, img, q=None):
    ok, buf = cv2.imencode(".jpg" if q else ".png", img, [cv2.IMWRITE_JPEG_QUALITY, q] if q else [])
    buf.tofile(str(p))


def degrade(g, dpi=200, angle=0.8, cut_mm=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ppm = 23.495
    if cut_mm:
        c = int(round((1.0 + cut_mm) * ppm))
        g = g[c:-c, c:-c]
    f = dpi / 600.0
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    g = cv2.GaussianBlur(g, (0, 0), 0.8).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    paper = 232 + 12 * np.sin(xx / w * 3.1) * np.cos(yy / h * 2.3)
    g = paper * (g / 255.0) + 25 * (1 - g / 255.0)
    g += rng.normal(0, 7, g.shape)
    spots = (rng.random(g.shape) < 0.0004)
    g[cv2.dilate(spots.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] -= 90
    g = np.clip(g, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def metrics(md):
    rows = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and "|" in line[2:]:
            k, v = [c.strip() for c in line.strip("|").split("|")[:2]]
            rows[k] = v
    return rows


def run_case(name, png, ref, extra=()):
    dxf = OUT / f"{name}.dxf"
    t = time.time()
    p = subprocess.run([PY, str(PIPE), str(png), str(dxf), "--report", str(OUT / f"{name}.run.json"), *extra],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    row = {"case": name, "wall_s": round(time.time() - t, 1), "ok": p.returncode == 0}
    if p.returncode != 0:
        row["error"] = (p.stderr or "?").strip().splitlines()[-1][:300]
        return row
    run = json.load(open(OUT / f"{name}.run.json", encoding="utf-8"))
    row.update({"calibration": {k: run["calibration"][k] for k in ("sheet", "px_per_mm_x", "calibrated_on")},
                "skew": run["skew_deg"], "counts": run["counts"], "dims_with_value": run["dims_with_value"],
                "title": run["title_fields"]})
    if ref:
        c = subprocess.run([PY, str(CMP), str(ref), str(dxf), "--out", str(OUT / f"cmp_{name}"), "--align", "geom"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        row["cmp"] = metrics(OUT / f"cmp_{name}" / "report.md") if c.returncode == 0 else {"error": c.stderr[-300:]}
    return row


def main():
    OUT.mkdir(exist_ok=True)
    src = load(ROOT / "runs/клапан.png")
    ref = ROOT / "dataset/clear_sum_dataset/vse_dxf/клапан.dxf"   # read only by compare_dxf.py (final stage)
    res = []
    for name, kw in (("scan200_border", dict(cut_mm=0.0)), ("scan200_cut", dict(cut_mm=1.5))):
        save(OUT / f"{name}.png", degrade(src, **kw))
        res.append(run_case(name, OUT / f"{name}.png", ref))
        print(res[-1]["case"], res[-1].get("cmp", {}).get("geometry F1 @0.5 mm"), res[-1].get("error", ""), flush=True)
    import pypdfium2 as pdfium
    for k in ("001", "002", "003"):
        pdf = ROOT / f"dataset/data/pdf/drawing_{k}.pdf"
        page = pdfium.PdfDocument(str(pdf))[0]
        img = page.render(scale=200 / 72, grayscale=True).to_numpy()
        img = img if img.ndim == 2 else img[..., 0]
        save(OUT / f"synthetic_{k}.png", img)
        res.append(run_case(f"synthetic_{k}", OUT / f"synthetic_{k}.png", ROOT / f"dataset/data/dxf/drawing_{k}.dxf"))
        print(res[-1]["case"], res[-1].get("cmp", {}).get("geometry F1 @0.5 mm"), res[-1].get("error", ""), flush=True)
    json.dump(res, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
