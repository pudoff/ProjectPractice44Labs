"""Filled symbols on the raster: dimension arrowheads, filled datum triangles, dots.

Filled shapes survive a morphological opening with a disk wider than a line stroke; lines do not.
For an arrowhead (isosceles triangle, base width wb, length L) the opened blob is the triangle with
the tip cut where the width equals the disk diameter d, so the full length is L = core / (1 - d / wb).
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from raster import to_mm


@dataclass
class Arrow:
    tip: np.ndarray        # mm
    dir: np.ndarray        # unit vector pointing from the base to the tip
    length: float          # mm
    width: float           # mm
    used: bool = False
    crossed: bool = False

    @property
    def base(self):
        return self.tip - self.dir * self.length


@dataclass
class Blob:
    kind: str              # 'datum' | 'dot'
    poly: np.ndarray       # mm outline
    center: np.ndarray


def detect(bw, cal, d_open_mm=0.42):
    ppm = cal["px_per_mm_x"]
    r = max(2, int(round(d_open_mm * ppm / 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(opened, connectivity=8)
    arrows, blobs = [], []
    erase = np.zeros_like(bw)
    min_area = (0.25 * ppm) ** 2
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or max(w, h) > 12 * ppm:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        pts = np.column_stack([xs + x, ys + y]).astype(float)
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        evals, evecs = np.linalg.eigh(cov)
        e = evecs[:, 1]
        t = (pts - c) @ e
        s = (pts - c) @ np.array([-e[1], e[0]])
        tmin, tmax = t.min(), t.max()
        core = (tmax - tmin) / ppm
        sl = max(1.0, 0.15 * (tmax - tmin))
        w_lo = np.ptp(s[t < tmin + sl]) / ppm if (t < tmin + sl).any() else 0
        w_hi = np.ptp(s[t > tmax - sl]) / ppm if (t > tmax - sl).any() else 0
        wb = max(np.ptp(s) / ppm, 1e-3)
        aspect = core / wb
        fill = area / (ppm * ppm) / max(core * wb, 1e-6)
        if aspect < 1.4:
            if fill > 0.6 and core < 1.5 and area / ppm ** 2 > 0.5:
                kind = "dot"
            elif 0.35 < fill < 0.75 and core > 2.0:
                kind = "datum"
            else:
                continue
            cnts, _ = cv2.findContours((lab[y:y + h, x:x + w] == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cpts = cnts[0].reshape(-1, 2).astype(float) + [x, y]
            if kind == "datum":
                cpts = cv2.approxPolyDP(cpts.astype(np.float32), 0.3 * ppm, True).reshape(-1, 2)
            mx, my = to_mm(cal, cpts[:, 0], cpts[:, 1])
            cx, cy = to_mm(cal, c[0], c[1])
            blobs.append(Blob(kind, np.column_stack([mx, my]), np.array([cx, cy])))
            cv2.drawContours(erase, [cnts[0] + [x, y]], -1, 1, -1)
            if kind == "dot":
                cv2.drawContours(erase, [cnts[0] + [x, y]], -1, 1, 3)   # grow only this dot, not the whole mask
            continue
        if not (1.4 <= aspect <= 6.0 and 0.35 <= fill <= 0.9 and 0.4 <= wb <= 2.5):
            continue
        # base = wider end; direction from base towards the narrow end
        if w_lo >= w_hi:
            tb, dirv = tmin, e
        else:
            tb, dirv = tmax, -e
        base_px = c + e * tb
        narrow_px = c + e * (tmax if w_lo >= w_hi else tmin)
        tip_px, crossed = refine_tip(bw, ppm, narrow_px, dirv, max_mm=0.6 * core + 0.6)
        bx, by = to_mm(cal, base_px[0], base_px[1])
        tx, ty = to_mm(cal, tip_px[0], tip_px[1])
        tip, base = np.array([tx, ty]), np.array([bx, by])
        d = (tip - base) / np.linalg.norm(tip - base)
        arrows.append(Arrow(tip, d, float(np.linalg.norm(tip - base)), float(wb), crossed=crossed))
        nrm = np.array([-dirv[1], dirv[0]])
        tri = np.array([tip_px + dirv * 1.0, base_px - dirv * 1.5 + nrm * (wb * ppm / 2 + 2),
                        base_px - dirv * 1.5 - nrm * (wb * ppm / 2 + 2)])
        cv2.fillPoly(erase, [np.round(tri).astype(np.int32)], 1)
    return arrows, blobs, erase


def refine_tip(bw, ppm, start_px, dirv, max_mm=2.0, step_mm=0.04):
    """Walk from the narrow end of the opened arrow core along the arrow axis.
    The tip is where the axis meets a crossing line (ink on both sides at +-0.8 mm across)
    or where the ink on the axis ends; capped at max_mm."""
    h, w = bw.shape
    nrm = np.array([-dirv[1], dirv[0]])

    def ink(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 0 <= x < w and 0 <= y < h and bw[y, x] > 0

    cross = []
    last_on = 0.0
    t = 0.0
    while t <= max_mm:
        p = start_px + dirv * t * ppm
        on = ink(p)
        if on:
            last_on = t
            side = all(any(ink(p + nrm * k * o * ppm) for o in (0.7, 0.8, 0.9)) for k in (1, -1))
            if side:
                cross.append(t)
            elif cross:
                break
        elif not cross:
            break
        t += step_mm
    tt = float(np.mean(cross)) if cross else last_on + 0.08
    return start_px + dirv * tt * ppm, bool(cross)
