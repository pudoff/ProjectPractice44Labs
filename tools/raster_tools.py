"""Raster drawing measurement helpers (OpenCV + Pillow).

Subcommands (all coordinates in SHEET millimetres, origin = bottom-left of the outer
sheet edge, Y up - the same system the DXF will use):

  info  IMG                         size, guessed DPI for A-series sheets
  calib IMG --sheet A3 --out c.json detect the sheet frame, write px<->mm calibration
  crop  IMG --calib c.json --x0 .. --y0 .. --x1 .. --y1 .. --out crop.png [--grid 5]
                                    crop a region (mm), upscale for reading, draw mm grid
                                    with labels every 10 mm (thin cyan, labels red)
  probe IMG --calib c.json --x .. --y ..
                                    print px position and local stroke thickness (mm)

Read crops with the Read tool: keep crops <= ~2000 px on the long side, otherwise the
viewer downsamples and you lose the detail you cropped for.
"""
import argparse
import json
import math
import sys

import cv2
import numpy as np

SHEETS = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (210, 297)}


def load_gray(path):
    data = np.fromfile(path, dtype=np.uint8)  # works with Cyrillic paths on Windows
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"cannot read {path}")
    return img


def save_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(path)


def cmd_info(a):
    g = load_gray(a.img)
    h, w = g.shape
    print(f"size {w}x{h}px  (all A-sizes share one aspect ratio: take the format from the title block,"
          " e.g. 'Формат А3', then the DPI follows)")
    for name, (sw, sh) in SHEETS.items():
        for W, H in ((sw, sh), (sh, sw)):
            dx, dy = w / W * 25.4, h / H * 25.4
            if abs(dx - dy) / dx < 0.01:
                print(f"  matches {name} {W}x{H} mm at {dx:.1f} x {dy:.1f} dpi")


def cmd_calib(a):
    g = load_gray(a.img)
    h, w = g.shape
    sw, sh = SHEETS[a.sheet]
    if (w > h) != (sw > sh):
        sw, sh = sh, sw
    bw = (g < 128).astype(np.uint8)
    # Outer drawn border: first/last rows/cols with long dark runs.
    rows = np.where(bw.sum(axis=1) > 0.5 * w)[0]
    cols = np.where(bw.sum(axis=0) > 0.5 * h)[0]
    res = {"img_w": w, "img_h": h, "sheet": a.sheet, "sheet_w": sw, "sheet_h": sh}
    # Default: image == full sheet (common for exported rasters).
    sx, sy = w / sw, h / sh
    res.update({"px_per_mm_x": sx, "px_per_mm_y": sy, "origin_px": [0, h]})
    if len(rows) and len(cols):
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
        res["outer_line_px"] = {"left": c0, "right": c1, "top": r0, "bottom": r1}
        if (c1 - c0) > 0.9 * w and (r1 - r0) > 0.9 * h:
            # The outermost long lines are the sheet trim edge (sheet_w x sheet_h). Rendered/scanned
            # images usually carry a margin around it, so calibrate on the line, not on the image size.
            sx, sy = (c1 - c0) / sw, (r1 - r0) / sh
            res.update({"px_per_mm_x": sx, "px_per_mm_y": sy, "origin_px": [c0, r1],
                        "calibrated_on": "outer sheet line"})
        else:
            res["calibrated_on"] = "image size (no full-sheet outer line found - check manually)"
        if abs(sx - sy) / sx > 0.003:
            res["warning"] = "x/y scale differ >0.3 % - scan is stretched or skewed; rectify first"
    # skew estimate from the longest horizontal line
    lines = cv2.HoughLinesP(bw * 255, 1, np.pi / 7200, threshold=200, minLineLength=w // 3, maxLineGap=5)
    if lines is not None:
        angs = [math.degrees(math.atan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)
                if abs(y2 - y1) < abs(x2 - x1)]
        if angs:
            res["skew_deg"] = round(float(np.median(angs)), 4)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(res, indent=2))


def mm_to_px(c, x, y):
    ox, oy = c["origin_px"]
    return ox + x * c["px_per_mm_x"], oy - y * c["px_per_mm_y"]


def cmd_crop(a):
    c = json.load(open(a.calib, encoding="utf-8"))
    g = load_gray(a.img)
    px0, py1 = mm_to_px(c, a.x0, a.y0)
    px1, py0 = mm_to_px(c, a.x1, a.y1)
    px0, py0 = max(0, int(px0)), max(0, int(py0))
    px1, py1 = min(c["img_w"], int(px1)), min(c["img_h"], int(py1))
    crop = g[py0:py1, px0:px1]
    scale = min(a.max_side / max(crop.shape), 4.0)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    col = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if a.grid > 0:
        k = c["px_per_mm_x"] * scale
        start_x = math.ceil(a.x0 / a.grid) * a.grid
        start_y = math.ceil(a.y0 / a.grid) * a.grid
        xm = start_x
        while xm <= a.x1:
            X = int((xm - a.x0) * k)
            major = abs(xm % 10) < 1e-6
            cv2.line(col, (X, 0), (X, col.shape[0]), (255, 200, 0) if major else (255, 235, 180), 1)
            if major:
                cv2.putText(col, f"{xm:g}", (X + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
            xm += a.grid
        ym = start_y
        while ym <= a.y1:
            Y = int(col.shape[0] - (ym - a.y0) * k)
            major = abs(ym % 10) < 1e-6
            cv2.line(col, (0, Y), (col.shape[1], Y), (255, 200, 0) if major else (255, 235, 180), 1)
            if major:
                cv2.putText(col, f"{ym:g}", (2, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
            ym += a.grid
        # redraw ink on top of the grid so strokes stay readable
        ink = crop < 128
        col[ink] = (0, 0, 0)
    save_png(a.out, col)
    print(f"{a.out}: {col.shape[1]}x{col.shape[0]} px, {scale:.2f}x, region mm x[{a.x0},{a.x1}] y[{a.y0},{a.y1}]")


def cmd_probe(a):
    c = json.load(open(a.calib, encoding="utf-8"))
    g = load_gray(a.img)
    px, py = mm_to_px(c, a.x, a.y)
    bw = (g < 128).astype(np.uint8)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    r = 6
    win = dist[int(py) - r:int(py) + r + 1, int(px) - r:int(px) + r + 1]
    t = 2 * float(win.max()) / c["px_per_mm_x"] if win.size else 0
    print(f"px=({px:.1f},{py:.1f}) ink={'yes' if bw[int(py), int(px)] else 'no'} stroke~{t:.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("info"); p.add_argument("img")
    p = sp.add_parser("calib"); p.add_argument("img"); p.add_argument("--sheet", default="A3"); p.add_argument("--out", required=True)
    p = sp.add_parser("crop"); p.add_argument("img"); p.add_argument("--calib", required=True)
    for k in ("x0", "y0", "x1", "y1"):
        p.add_argument("--" + k, type=float, required=True)
    p.add_argument("--out", required=True); p.add_argument("--grid", type=float, default=5); p.add_argument("--max-side", type=int, default=1800)
    p = sp.add_parser("probe"); p.add_argument("img"); p.add_argument("--calib", required=True)
    p.add_argument("--x", type=float, required=True); p.add_argument("--y", type=float, required=True)
    a = ap.parse_args()
    {"info": cmd_info, "calib": cmd_calib, "crop": cmd_crop, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
