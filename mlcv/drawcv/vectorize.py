"""Binary image -> 1 px skeleton -> pixel chains between junctions -> straight segments and circular arcs.
Ported from the prototype runs/klapan_png2dxf/pipeline_b/vectorize.py (validated on 9 real sheets)."""
import math

import numpy as np
from skimage.morphology import skeletonize

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton(bw):
    return skeletonize(bw.astype(bool)).astype(np.uint8)


def trace_chains(sk, min_len=3):
    """Split a 1 px skeleton into pixel chains between junctions/endpoints. Returns (N, 2) arrays of (row, col)."""
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
    is_node = node[rows, cols]
    visited_edge = set()
    chains = []

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
    seen = np.zeros(len(rows), bool)
    for a, b in visited_edge:
        seen[a] = seen[b] = True
    for s in np.nonzero(~seen & ~is_node)[0]:       # closed loops without junctions (isolated circles)
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
    a = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(1)
    c, *_ = np.linalg.lstsq(a, b, rcond=None)
    r2 = c[2] + c[0] ** 2 + c[1] ** 2
    if r2 <= 0:
        return None
    return c[0], c[1], math.sqrt(r2)


def line_dev(pts):
    a, b = pts[0], pts[-1]
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-9:
        return np.linalg.norm(pts - a, axis=1)
    n = np.array([-d[1], d[0]]) / length
    return np.abs((pts - a) @ n)


def split_primitives(pts, tol, min_arc_pts=8, max_radius=500.0):
    """Recursive split of a centre-line polyline (mm) into line / arc pieces with fit residuals."""
    if len(pts) < 2:
        return []
    dev = line_dev(pts)
    if dev.max() <= tol:
        return [{"kind": "line", "p": pts[0].copy(), "q": pts[-1].copy(), "rms": float(np.sqrt(np.mean(dev ** 2))),
                 "n": len(pts)}]
    if len(pts) >= min_arc_pts:
        fc = fit_circle(pts)
        if fc is not None:
            cx, cy, r = fc
            rd = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
            if rd.max() <= tol and r < max_radius:
                a = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
                ua = np.degrees(np.unwrap(np.radians(a)))
                sweep = ua[-1] - ua[0]
                sagitta = r * (1 - math.cos(math.radians(min(abs(sweep), 180.0)) / 2))
                # a nearly straight chain with a skeleton hook at a junction fits a shallow arc within tol
                # (sagitta ~0.1 mm): an arc must be visibly curved, otherwise the chain is split into lines
                if abs(sweep) > 8 and sagitta >= 2 * tol:
                    a0, a1 = (ua[0], ua[-1]) if sweep > 0 else (ua[-1], ua[0])
                    return [{"kind": "arc", "c": np.array([cx, cy]), "r": float(r), "a0": a0 % 360,
                             "sweep": float(min(abs(sweep), 360.0)), "rms": float(np.sqrt(np.mean(rd ** 2))),
                             "n": len(pts)}]
    k = int(np.argmax(dev))
    k = min(max(k, 1), len(pts) - 2)
    return (split_primitives(pts[:k + 1], tol, min_arc_pts, max_radius)
            + split_primitives(pts[k:], tol, min_arc_pts, max_radius))


def smooth(pts, k=2):
    if len(pts) < 2 * k + 3:
        return pts
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    xs = np.convolve(pts[:, 0], ker, mode="same")
    ys = np.convolve(pts[:, 1], ker, mode="same")
    xs[:k], ys[:k] = pts[:k, 0], pts[:k, 1]
    xs[-k:], ys[-k:] = pts[-k:, 0], pts[-k:, 1]
    return np.column_stack([xs, ys])


def vectorize(bw, px_to_mm, tol_mm, max_radius):
    """px_to_mm: Transform work_px -> sheet_mm. Returns (raw pieces in mm, skeleton)."""
    sk = skeleton(bw)
    pieces = []
    for ch in trace_chains(sk):
        pts = px_to_mm.apply(np.column_stack([ch[:, 1], ch[:, 0]]).astype(float))
        pieces.extend(split_primitives(smooth(pts), tol_mm, max_radius=max_radius))
    return pieces, sk
