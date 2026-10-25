"""Raster -> centre-line chains -> straight segments / circular arcs (sheet mm)."""
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton(bw):
    return skeletonize(bw.astype(bool)).astype(np.uint8)


def trace_chains(sk, min_len=3):
    """Split a 1-px skeleton into pixel chains between junctions/endpoints.
    Returns list of (N,2) arrays of (row, col)."""
    h, w = sk.shape
    pad = np.pad(sk, 1)
    deg = np.zeros_like(pad, dtype=np.uint8)
    for dr, dc in NB:
        deg += np.roll(np.roll(pad, dr, 0), dc, 1)
    deg = deg[1:-1, 1:-1] * sk
    node = (sk == 1) & (deg != 2)
    rows, cols = np.nonzero(sk)
    idx = -np.ones(sk.shape, dtype=np.int64)
    idx[rows, cols] = np.arange(len(rows))
    visited_edge = set()
    chains = []
    is_node = node[rows, cols]

    def neighbours(i):
        r, c = rows[i], cols[i]
        out = []
        for dr, dc in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                j = idx[rr, cc]
                if j >= 0:
                    out.append(j)
        return out

    def walk(start, nxt):
        path = [start, nxt]
        prev, cur = start, nxt
        while not is_node[cur]:
            nb = [j for j in neighbours(cur) if j != prev and j not in path[-3:]]
            if not nb:
                break
            # prefer 4-connected continuation
            nb.sort(key=lambda j: abs(rows[j] - rows[cur]) + abs(cols[j] - cols[cur]))
            prev, cur = cur, nb[0]
            path.append(cur)
            if cur == start:
                break
        return path

    for s in np.nonzero(is_node)[0]:
        for n in neighbours(s):
            key = (min(s, n), max(s, n))
            if key in visited_edge:
                continue
            p = walk(s, n)
            for a, b in zip(p[:-1], p[1:]):
                visited_edge.add((min(a, b), max(a, b)))
            if len(p) >= min_len:
                chains.append(np.column_stack([rows[p], cols[p]]))
    # pure loops (no nodes): circles etc.
    seen = np.zeros(len(rows), bool)
    for a, b in visited_edge:
        seen[a] = seen[b] = True
    for s in np.nonzero(~seen & ~is_node)[0]:
        if seen[s]:
            continue
        nb = neighbours(s)
        if not nb:
            continue
        p = walk(s, nb[0])
        for q in p:
            seen[q] = True
        if len(p) >= min_len:
            chains.append(np.column_stack([rows[p], cols[p]]))
    return chains


def fit_circle(pts):
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = c[2] + c[0] ** 2 + c[1] ** 2
    if r2 <= 0:
        return None
    return c[0], c[1], math.sqrt(r2)


def line_dev(pts):
    a, b = pts[0], pts[-1]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:
        return np.linalg.norm(pts - a, axis=1)
    n = np.array([-d[1], d[0]]) / L
    return np.abs((pts - a) @ n)


def split_primitives(pts, tol, min_arc_pts=8):
    """Recursive split of a centre-line polyline (mm) into ('line', p, q) / ('arc', c, r, a0, a1)."""
    if len(pts) < 2:
        return []
    dev = line_dev(pts)
    if dev.max() <= tol:
        return [("line", pts[0], pts[-1])]
    if len(pts) >= min_arc_pts:
        fc = fit_circle(pts)
        if fc is not None:
            cx, cy, r = fc
            rd = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
            if rd.max() <= tol and r < 500:
                a = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
                ua = np.degrees(np.unwrap(np.radians(a)))
                sweep = ua[-1] - ua[0]
                if abs(sweep) > 8:
                    a0, a1 = (ua[0], ua[-1]) if sweep > 0 else (ua[-1], ua[0])
                    return [("arc", np.array([cx, cy]), r, a0 % 360, a1 % 360 if (a1 - a0) < 359 else (a0 + 359.9) % 360,
                             abs(sweep))]
    k = int(np.argmax(dev))
    k = min(max(k, 1), len(pts) - 2)
    return split_primitives(pts[:k + 1], tol, min_arc_pts) + split_primitives(pts[k:], tol, min_arc_pts)


def chains_to_mm(chains, calib):
    sx, sy = calib["px_per_mm_x"], calib["px_per_mm_y"]
    ox, oy = calib["origin_px"]
    out = []
    for ch in chains:
        r, c = ch[:, 0].astype(float), ch[:, 1].astype(float)
        out.append(np.column_stack([(c - ox) / sx, (oy - r) / sy]))
    return out


def smooth(pts, k=2):
    if len(pts) < 2 * k + 3:
        return pts
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    xs = np.convolve(pts[:, 0], ker, mode="same")
    ys = np.convolve(pts[:, 1], ker, mode="same")
    xs[:k], ys[:k] = pts[:k, 0], pts[:k, 1]
    xs[-k:], ys[-k:] = pts[-k:, 0], pts[-k:, 1]
    return np.column_stack([xs, ys])


def vectorize(bw, calib, tol_mm=0.08):
    sk = skeleton(bw)
    chains = trace_chains(sk)
    prims = []
    for ch in chains_to_mm(chains, calib):
        prims.extend(split_primitives(smooth(ch), tol_mm))
    return prims, sk
