"""Overlay a DXF on the source raster (same sheet mm calibration) + raster-based metrics.

  python overlay.py DXF --box x0 y0 x1 y1 [--ppm 20] --out crop.png     visual crop
  python overlay.py DXF --metrics --out metrics.json [--ppm 8]          whole-sheet recall/precision

Colours in crops: black = raster and DXF agree, grey = only on raster (missing in DXF),
red = only in DXF (extra / misplaced).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import ezdxf
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ezdxf.addons.drawing import Frontend, RenderContext  # noqa: E402
from ezdxf.addons.drawing.config import (BackgroundPolicy, ColorPolicy, Configuration,  # noqa: E402
                                         LineweightPolicy)
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from meas import CAL, OX, OY, SX, SY, gray  # noqa: E402


def render(doc, box, ppm, lw_mm=0.25):
    x0, y0, x1, y1 = box
    W, H = int(round((x1 - x0) * ppm)), int(round((y1 - y0) * ppm))
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    cfg = Configuration(color_policy=ColorPolicy.BLACK, background_policy=BackgroundPolicy.WHITE)
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=cfg).draw_layout(doc.modelspace(), finalize=False)
    fig.set_size_inches(W / dpi, H / dpi)
    ax.autoscale(False)
    ax.set_aspect("auto", adjustable="box")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_position([0, 0, 1, 1])
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if g.shape != (H, W):
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    ink = (g < 128).astype(np.uint8)
    k = int(round(lw_mm * ppm / 2))
    if k >= 1:
        ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return ink > 0


def raster(box, ppm):
    x0, y0, x1, y1 = box
    W, H = int(round((x1 - x0) * ppm)), int(round((y1 - y0) * ppm))
    M = np.array([[SX / ppm, 0, OX + x0 * SX], [0, SY / ppm, OY - y1 * SY]], dtype=np.float64)
    g = cv2.warpAffine(gray(), M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA, borderValue=255)
    return g < 160


def crop(doc, box, ppm, out):
    d = render(doc, box, ppm)
    r = raster(box, ppm)
    img = np.full(d.shape + (3,), 255, np.uint8)
    img[r] = (170, 170, 170)
    img[d] = (0, 0, 255)
    img[r & d] = (0, 0, 0)
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(out)


def text_mask(shape, ppm, pad=0.6):
    """Union of OCR text boxes (cached by pipeline_b OCR) -> mask in the metric raster."""
    import json as _json
    cache = Path(__file__).with_name("ocr_boxes.json")
    if not cache.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline_b"))
        from ocr import run_ocr
        items = run_ocr(gray(), CAL)
        _json.dump([{"text": t.text, "poly": t.poly.tolist()} for t in items], open(cache, "w", encoding="utf-8"),
                   ensure_ascii=False)
    m = np.zeros(shape, np.uint8)
    H = shape[0]
    for it in _json.load(open(cache, encoding="utf-8")):
        poly = np.array(it["poly"])
        c = poly.mean(0)
        poly = c + (poly - c) * 1.0 + np.sign(poly - c) * pad
        pts = np.column_stack([poly[:, 0] * ppm, H - poly[:, 1] * ppm]).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m > 0


def metrics(doc, ppm, tols=(0.15, 0.3, 0.5), graphics_only=False):
    box = (0, 0, CAL["sheet_w"], CAL["sheet_h"])
    d = skeletonize(render(doc, box, ppm))
    r = skeletonize(raster(box, ppm))
    if graphics_only:
        tm = text_mask(d.shape, ppm)
        d &= ~tm
        r &= ~tm
    dt_d = cv2.distanceTransform((~d).astype(np.uint8), cv2.DIST_L2, 5) / ppm
    dt_r = cv2.distanceTransform((~r).astype(np.uint8), cv2.DIST_L2, 5) / ppm
    res = {"ppm": ppm, "raster_skel_px": int(r.sum()), "dxf_skel_px": int(d.sum())}
    for t in tols:
        rec = float((dt_d[r] <= t).mean())
        prec = float((dt_r[d] <= t).mean())
        res[f"recall@{t}"] = round(rec, 4)
        res[f"precision@{t}"] = round(prec, 4)
        res[f"f1@{t}"] = round(2 * rec * prec / (rec + prec + 1e-12), 4)
    miss = r & (dt_d > 0.5)
    extra = d & (dt_r > 0.5)
    return res, miss, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf")
    ap.add_argument("--box", nargs=4, type=float)
    ap.add_argument("--ppm", type=float, default=None)
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--graphics-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = ezdxf.readfile(a.dxf)
    if a.metrics:
        ppm = a.ppm or 8
        res, miss, extra = metrics(doc, ppm, graphics_only=a.graphics_only)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
        vis = np.full(miss.shape + (3,), 255, np.uint8)
        vis[cv2.dilate(miss.astype(np.uint8), np.ones((3, 3))) > 0] = (120, 120, 120)
        vis[cv2.dilate(extra.astype(np.uint8), np.ones((3, 3))) > 0] = (0, 0, 255)
        ok, buf = cv2.imencode(".png", vis)
        buf.tofile(a.out.replace(".json", "_diff.png"))
        print(json.dumps(res))
    else:
        x0, y0, x1, y1 = a.box
        ppm = a.ppm or min(25, 1800 / max(x1 - x0, y1 - y0))
        crop(doc, a.box, ppm, a.out)
        print(a.out)


if __name__ == "__main__":
    sys.exit(main())
