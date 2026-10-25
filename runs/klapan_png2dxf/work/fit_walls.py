import math

import numpy as np

from meas import fit_circle, fit_line, ink_points

out = open("fit_walls.txt", "w", encoding="utf-8")


def band_line(p, ang, t0, t1, w=0.2):
    """Ink points within w mm of a predicted line p + t*dir, t in [t0, t1]."""
    d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    n = np.array([-d[1], d[0]])
    a, b = np.array(p) + d * t0, np.array(p) + d * t1
    box = ink_points(min(a[0], b[0]) - 1, min(a[1], b[1]) - 1, max(a[0], b[0]) + 1, max(a[1], b[1]) + 1)
    rel = box - np.array(p)
    s, o = rel @ d, rel @ n
    m = (s >= t0) & (s <= t1) & (np.abs(o) < w)
    return box[m], o[m]


def report_line(name, p, ang, t0, t1):
    pts, off = band_line(p, ang, t0, t1)
    (x, y), a = fit_line(pts)
    if a < 0:
        a += 180
    # signed offset of fitted line from predicted, measured at p
    d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    n = np.array([-d[1], d[0]])
    off0 = (np.array([x, y]) - np.array(p)) @ n
    out.write(f"{name}: n={len(pts)} angle={a:.3f} offset_from_pred={off0:+.3f} mm (median {np.median(off):+.3f})\n")


YA, X0 = 160.41, 54.30
pc = (X0 + 10, YA + 21)
report_line("upper outer wall (pred thru (64.3,181.41) 60°)", pc, 60, 3, 40)
report_line("upper inner wall (pred +4/sin60)", (pc[0] + 4 / math.sin(math.radians(60)), pc[1]), 60, 3, 40)
pcl = (X0 + 10, YA - 21)
report_line("lower outer wall (pred thru (64.3,139.41) -60°)", pcl, -60, 3, 40)
report_line("lower inner wall", (pcl[0] + 4 / math.sin(math.radians(60)), pcl[1]), -60, 3, 40)


def report_circle(name, box):
    pts = ink_points(*box)
    cx, cy, r = fit_circle(pts)
    out.write(f"{name}: c=({cx:.3f},{cy:.3f}) r={r:.3f} (stroke-centre approx, n={len(pts)})\n")


report_circle("groove top", (94.95, 233.6, 97.65, 235.2))
report_circle("groove bot", (94.95, 85.6, 97.65, 87.2))
report_circle("callout A circle", (92.5, 233.0, 93.3, 236.0))
out.close()
