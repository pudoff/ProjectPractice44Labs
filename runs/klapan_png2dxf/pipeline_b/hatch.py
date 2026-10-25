"""Section hatching: families of parallel equidistant segments -> HATCH regions.

Region boundary: flood fill of the background bounded by the contour strokes (hatch strokes removed),
seeded at hatch-segment midpoints; falls back to a morphological closing of the segments when the
fill leaks through a gap in the contour.
"""
import math

import cv2
import numpy as np

from postproc import angle, length


def find_families(lines, angles=(45.0, 135.0), ang_tol=1.5, sp_range=(1.2, 6.0), min_lines=4):
    regions = []
    rest = set(range(len(lines)))
    for fam in angles:
        idx = [i for i, e in enumerate(lines) if e["type"] == "line" and length(e) >= 0.8
               and abs((angle(e) - fam + 90) % 180 - 90) < ang_tol]
        if len(idx) < min_lines:
            continue
        t = math.radians(fam)
        n = np.array([-math.sin(t), math.cos(t)])
        d = np.array([math.cos(t), math.sin(t)])
        offs = np.array([lines[i]["p"] @ n for i in idx])
        ext0 = [(min(lines[i]["p"] @ d, lines[i]["q"] @ d), max(lines[i]["p"] @ d, lines[i]["q"] @ d)) for i in idx]
        # spacing = most frequent distance to the nearest parallel neighbour that overlaps along the
        # hatch direction (local: hatches of different views must not interleave in one statistic)
        gaps = []
        for a_ in range(len(idx)):
            best = None
            for b_ in range(len(idx)):
                do = offs[b_] - offs[a_]
                if do <= 0.1 or do > sp_range[1]:
                    continue
                if min(ext0[a_][1], ext0[b_][1]) - max(ext0[a_][0], ext0[b_][0]) > 0.5:
                    best = do if best is None else min(best, do)
            if best is not None:
                gaps.append(best)
        gaps = np.array(gaps)
        gaps = gaps[(gaps >= sp_range[0]) & (gaps <= sp_range[1])]
        if len(gaps) < 3:
            continue
        hist, edges = np.histogram(gaps, bins=np.arange(sp_range[0], sp_range[1] + 0.05, 0.05))
        sp = float(np.median(gaps[(gaps >= edges[hist.argmax()] - 0.1) & (gaps <= edges[hist.argmax() + 1] + 0.1)]))
        # union-find over lines that are neighbours (offset differs by ~sp and overlap along d)
        parent = list(range(len(idx)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        ext = [(min(lines[i]["p"] @ d, lines[i]["q"] @ d), max(lines[i]["p"] @ d, lines[i]["q"] @ d)) for i in idx]
        order = np.argsort(offs)
        for a_ in range(len(order)):
            ia = order[a_]
            for b_ in range(a_ + 1, len(order)):
                ib = order[b_]
                do = offs[ib] - offs[ia]
                if do > 1.6 * sp:
                    break
                if abs(do - sp) < 0.25 * sp or do < 0.1:
                    if min(ext[ia][1], ext[ib][1]) - max(ext[ia][0], ext[ib][0]) > -1.5 * sp:
                        parent[find(ia)] = find(ib)
        groups = {}
        for k in range(len(idx)):
            groups.setdefault(find(k), []).append(k)
        for g in groups.values():
            if len(g) < min_lines:
                continue
            o = offs[g]
            # circular mean: offsets mod spacing wrap around, a plain median breaks near the 0/sp boundary
            phase = float(np.angle(np.mean(np.exp(2j * np.pi * o / sp))) / (2 * np.pi) * sp) % sp
            res = np.abs((np.mod(o - phase, sp) + sp / 2) % sp - sp / 2)
            members = [idx[k] for k, r in zip(g, res) if r < 0.2]
            if len(members) < min_lines:
                continue
            regions.append({"angle": fam, "spacing": sp, "phase_off": phase, "lines": members,
                            "normal": n, "dir": d})
            rest -= set(members)
    return regions, sorted(rest)


def region_boundary(region, lines, ink_nohatch, cal, to_px, to_mm):
    """Closed polygon(s) in mm for a hatch region."""
    ppm = cal["px_per_mm_x"]
    pts = []
    for i in region["lines"]:
        pts += [lines[i]["p"], lines[i]["q"]]
    pts = np.array(pts)
    x0, y0 = pts.min(0) - 3
    x1, y1 = pts.max(0) + 3
    px0, py1 = to_px(cal, x0, y0)
    px1, py0 = to_px(cal, x1, y1)
    px0, py0 = max(0, int(px0)), max(0, int(py0))
    px1, py1 = min(ink_nohatch.shape[1], int(px1)), min(ink_nohatch.shape[0], int(py1))
    sub = (ink_nohatch[py0:py1, px0:px1] > 0).astype(np.uint8)
    walls = cv2.dilate(sub, np.ones((3, 3), np.uint8))
    fill = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), np.uint8)
    region_mask = np.zeros_like(sub)
    hull_area = cv2.contourArea(cv2.convexHull(np.round(np.column_stack(to_px(cal, pts[:, 0], pts[:, 1])) -
                                                        [px0, py0]).astype(np.float32)))
    for i in region["lines"]:
        m = (lines[i]["p"] + lines[i]["q"]) / 2
        sx, sy = to_px(cal, m[0], m[1])
        sx, sy = int(round(sx)) - px0, int(round(sy)) - py0
        if not (0 <= sx < sub.shape[1] and 0 <= sy < sub.shape[0]) or walls[sy, sx] or region_mask[sy, sx]:
            continue
        f = fill.copy()
        cv2.floodFill(walls.copy(), f, (sx, sy), 1, flags=8 | (1 << 8) | cv2.FLOODFILL_MASK_ONLY)
        region_mask |= f[1:-1, 1:-1]
    leaked = region_mask.sum() > 2.5 * max(hull_area, 1) or region_mask[0, :].any() or region_mask[-1, :].any() \
        or region_mask[:, 0].any() or region_mask[:, -1].any()
    if leaked:
        region_mask = np.zeros_like(sub)
        for i in region["lines"]:
            a = np.array(to_px(cal, *lines[i]["p"])) - [px0, py0]
            b = np.array(to_px(cal, *lines[i]["q"])) - [px0, py0]
            cv2.line(region_mask, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 1, 3)
        k = int(region["spacing"] * ppm * 1.3) | 1
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    else:
        region_mask = cv2.dilate(region_mask, np.ones((5, 5), np.uint8))      # to the stroke centre
    cnts, hier = cv2.findContours(region_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c, h in zip(cnts, hier[0] if hier is not None else []):
        if cv2.contourArea(c) < (1.0 * ppm) ** 2:
            continue
        c = cv2.approxPolyDP(c, 0.12 * ppm, True).reshape(-1, 2).astype(float) + [px0, py0]
        xs, ys = to_mm(cal, c[:, 0], c[:, 1])
        polys.append((np.column_stack([xs, ys]), h[3] >= 0))
    return polys, leaked
