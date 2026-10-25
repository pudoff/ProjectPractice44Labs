import json
import math
from pathlib import Path

import cv2
import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
IMG = ROOT / "runs" / "клапан.png"
CAL = json.load(open(RUN / "work" / "calib.json", encoding="utf-8"))
SX, SY = CAL["px_per_mm_x"], CAL["px_per_mm_y"]
OX, OY = CAL["origin_px"]

_gray = None
_ink = None
_dist = None


def gray():
    global _gray
    if _gray is None:
        _gray = cv2.imdecode(np.fromfile(str(IMG), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return _gray


def ink():
    global _ink
    if _ink is None:
        _ink = (gray() < 128).astype(np.uint8)
    return _ink


def dist():
    """Distance (px) from every pixel to the nearest ink pixel."""
    global _dist
    if _dist is None:
        _dist = cv2.distanceTransform(1 - ink(), cv2.DIST_L2, 5)
    return _dist


def to_px(x, y):
    return OX + x * SX, OY - y * SY


def to_mm(px, py):
    return (px - OX) / SX, (OY - py) / SY


def vline(x, y0, y1, search=0.6):
    """Sub-pixel x (mm) of a vertical stroke near x between y0..y1."""
    px0, _ = to_px(x - search, 0)
    px1, _ = to_px(x + search, 0)
    _, py0 = to_px(0, max(y0, y1))
    _, py1 = to_px(0, min(y0, y1))
    win = ink()[int(py0):int(py1), int(px0):int(px1) + 1].astype(float)
    prof = win.mean(axis=0)
    if prof.sum() == 0:
        return None
    cols = np.arange(int(px0), int(px0) + len(prof))
    m = prof > 0.5 * prof.max()
    return to_mm((cols[m] * prof[m]).sum() / prof[m].sum(), 0)[0]


def hline(y, x0, x1, search=0.6):
    _, py0 = to_px(0, y + search)
    _, py1 = to_px(0, y - search)
    px0, _ = to_px(min(x0, x1), 0)
    px1, _ = to_px(max(x0, x1), 0)
    win = ink()[int(py0):int(py1) + 1, int(px0):int(px1)].astype(float)
    prof = win.mean(axis=1)
    if prof.sum() == 0:
        return None
    rows = np.arange(int(py0), int(py0) + len(prof))
    m = prof > 0.5 * prof.max()
    return to_mm(0, (rows[m] * prof[m]).sum() / prof[m].sum())[1]


def ink_points(x0, y0, x1, y1):
    """Ink pixel centres (mm) inside a mm box."""
    px0, py1 = to_px(x0, y0)
    px1, py0 = to_px(x1, y1)
    sub = ink()[int(py0):int(py1), int(px0):int(px1)]
    ys, xs = np.nonzero(sub)
    xm, ym = to_mm(xs + int(px0), ys + int(py0))
    return np.column_stack([xm, ym])


def fit_circle(pts):
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    return c[0], c[1], r


def fit_line(pts):
    vx, vy, x, y = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return (float(x), float(y)), math.degrees(math.atan2(vy, vx))


def score_points(pts, tol_mm=0.15):
    """Share of mm points that lie on ink (within tol)."""
    d = dist()
    px = OX + pts[:, 0] * SX
    py = OY - pts[:, 1] * SY
    ok = (px >= 0) & (py >= 0) & (px < d.shape[1]) & (py < d.shape[0])
    v = np.full(len(pts), 1e9)
    v[ok] = d[py[ok].astype(int), px[ok].astype(int)]
    return float((v <= tol_mm * SX).mean())


def arc_pts(cx, cy, r, a0, a1, n=60):
    a = np.radians(np.linspace(a0, a1, n))
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def seg_pts(p, q, step=0.1):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(2, int(np.linalg.norm(q - p) / step))
    t = np.linspace(0, 1, n)[:, None]
    return p + (q - p) * t


_sdist = None


def skel_dist():
    """Distance (px) to the skeleton (stroke centre line) of the ink."""
    global _sdist
    if _sdist is None:
        from skimage.morphology import skeletonize
        sk = skeletonize(ink().astype(bool)).astype(np.uint8)
        _sdist = cv2.distanceTransform(1 - sk, cv2.DIST_L2, 5)
    return _sdist


def mean_skel_dist(pts, cap_mm=0.5):
    d = skel_dist()
    px = np.clip((OX + pts[:, 0] * SX).round().astype(int), 0, d.shape[1] - 1)
    py = np.clip((OY - pts[:, 1] * SY).round().astype(int), 0, d.shape[0] - 1)
    v = np.minimum(d[py, px] / SX, cap_mm)
    return float(v.mean())


_skel = None


def skel_points(x0, y0, x1, y1):
    global _skel
    if _skel is None:
        from skimage.morphology import skeletonize
        _skel = skeletonize(ink().astype(bool))
    px0, py1 = to_px(x0, y0)
    px1, py0 = to_px(x1, y1)
    sub = _skel[int(py0):int(py1), int(px0):int(px1)]
    ys, xs = np.nonzero(sub)
    xm, ym = to_mm(xs + int(px0), ys + int(py0))
    return np.column_stack([xm, ym])


def trace_curve(guide, band=0.6, step=1.0):
    """Skeleton points near a rough guide polyline (mm), resampled every `step` mm along the guide."""
    g = np.asarray(guide, float)
    seg_len = np.linalg.norm(np.diff(g, axis=0), axis=1)
    s_cum = np.concatenate([[0], np.cumsum(seg_len)])
    box = (g[:, 0].min() - band - 1, g[:, 1].min() - band - 1, g[:, 0].max() + band + 1, g[:, 1].max() + band + 1)
    pts = skel_points(*box)
    out = []
    for s in np.arange(0, s_cum[-1] + 1e-9, step):
        i = min(np.searchsorted(s_cum, s, side="right") - 1, len(g) - 2)
        t = (s - s_cum[i]) / max(seg_len[i], 1e-9)
        q = g[i] + (g[i + 1] - g[i]) * t
        d = np.linalg.norm(pts - q, axis=1)
        m = d < band
        if m.any():
            out.append(pts[m][np.argmin(d[m])])
    return np.array(out)
