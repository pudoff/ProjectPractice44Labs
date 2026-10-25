"""Raster loading, binarisation, skew estimate and px<->mm calibration on the sheet trim line."""
import math

import cv2
import numpy as np

SHEETS = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (210, 297)}


def load_gray(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def binarize(gray):
    """Otsu on a lightly blurred image; ink = 1."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw.astype(np.uint8)


def estimate_skew(bw):
    h, w = bw.shape
    small = cv2.resize(bw * 255, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    lines = cv2.HoughLinesP((small > 64).astype(np.uint8) * 255, 1, np.pi / 3600, threshold=200,
                            minLineLength=small.shape[1] // 3, maxLineGap=4)
    if lines is None:
        return 0.0
    angs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(a) < 5:
            angs.append(a)
        elif abs(abs(a) - 90) < 5:
            angs.append(a - 90 * np.sign(a))
    return float(np.median(angs)) if angs else 0.0


def deskew(gray, angle):
    if abs(angle) < 0.02:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def calibrate(bw, sheet=None):
    """Find the outermost long horizontal/vertical lines (sheet trim line) and map them to the
    nominal sheet size. If `sheet` is None it is guessed from the aspect ratio (A3/A4/...)."""
    h, w = bw.shape
    rows = np.where(bw.sum(axis=1) > 0.5 * w)[0]
    cols = np.where(bw.sum(axis=0) > 0.5 * h)[0]
    if len(rows) and len(cols) and (cols.max() - cols.min()) > 0.9 * w and (rows.max() - rows.min()) > 0.9 * h:
        c0, c1, r0, r1 = _stroke_centre(cols, 0), _stroke_centre(cols, -1), _stroke_centre(rows, 0), _stroke_centre(rows, -1)
        on = "sheet trim line"
    else:
        c0, c1, r0, r1 = 0, w - 1, 0, h - 1
        on = "image size (no trim line found)"
    if sheet is None:
        sheet = guess_sheet(c1 - c0, r1 - r0)
    sw, sh = SHEETS[sheet]
    if (c1 - c0 > r1 - r0) != (sw > sh):
        sw, sh = sh, sw
    # The outermost long lines are either the sheet trim line or, when the scan is cut along the trim,
    # the inner frame (GOST 2.301 margins: 20 mm left, 5 mm elsewhere). Pick the hypothesis whose
    # x and y scales agree.
    hyps = [(on, sw, sh, 0.0, 0.0)]
    if on == "sheet trim line":
        hyps.append(("inner frame (GOST 2.301 margins)", sw - 25.0, sh - 10.0, 20.0, 5.0))
    best = min(hyps, key=lambda t: abs((c1 - c0) / t[1] / ((r1 - r0) / t[2]) - 1))
    on, fw, fh, mx, my = best
    sx, sy = (c1 - c0) / fw, (r1 - r0) / fh
    return {"sheet": sheet, "sheet_w": sw, "sheet_h": sh, "px_per_mm_x": sx, "px_per_mm_y": sy,
            "origin_px": [c0 - mx * sx, r1 + my * sy], "calibrated_on": on, "img_w": w, "img_h": h}


def _stroke_centre(idx, end):
    """Centre of the contiguous run of long-line rows/cols at the start (end=0) or end (end=-1):
    a thick frame line must be measured at its middle, not at its outer edge."""
    seq = list(idx) if end == 0 else list(idx)[::-1]
    run = [seq[0]]
    for v in seq[1:]:
        if abs(v - run[-1]) != 1:
            break
        run.append(v)
    return float(np.mean(run))


def guess_sheet(wpx, hpx):
    """All A formats share the aspect; without OCR the most common archive format is chosen by
    size class assuming 150-600 dpi scans (A3 default for landscape, A4 for portrait)."""
    return "A3" if wpx > hpx else "A4"


def to_mm(cal, px, py):
    return (np.asarray(px) - cal["origin_px"][0]) / cal["px_per_mm_x"], (cal["origin_px"][1] - np.asarray(py)) / cal["px_per_mm_y"]


def to_px(cal, x, y):
    return cal["origin_px"][0] + np.asarray(x) * cal["px_per_mm_x"], cal["origin_px"][1] - np.asarray(y) * cal["px_per_mm_y"]


def stroke_width_mm(bw, cal, sample=400000, pct=20):
    """Thin-line stroke width (mm): low percentile of 2 x distance-to-background on the skeleton.
    The median is dominated by thick contour lines on some sheets and must not be used."""
    from skimage.morphology import skeletonize
    d = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    s = skeletonize(bw.astype(bool))
    v = d[s]
    if len(v) > sample:
        v = v[:: len(v) // sample]
    return float(np.percentile(2 * v, pct)) / cal["px_per_mm_x"] if len(v) else 0.25
