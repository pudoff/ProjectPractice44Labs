import math

import numpy as np

from geom import ang, fillet_lc, fillet_ll, unit
from meas import arc_pts, mean_skel_dist, score_points

YA, X0 = 160.41, 54.30
XR = X0 + 46            # rim right face
XS = XR - 5.5           # rim step
XL = XS - 4             # rim left face
out = open("fit_fillets.txt", "w", encoding="utf-8")


def arc_between(c, t1, t2, r):
    a1, a2 = ang(c, t1), ang(c, t2)
    da = (a2 - a1 + 540) % 360 - 180
    return arc_pts(c[0], c[1], r, a1, a1 + da, 80)


def best(name, fn, radii):
    res = []
    for r in radii:
        c, t1, t2 = fn(r)
        pts = arc_between(c, t1, t2, r)
        res.append((-mean_skel_dist(pts), r, c, t1, t2))
    res.sort(key=lambda v: -v[0])
    s, r, c, t1, t2 = res[0]
    out.write(f"{name}: best r={r} score={s:.3f} c=({c[0]:.3f},{c[1]:.3f}) t1=({t1[0]:.3f},{t1[1]:.3f}) t2=({t2[0]:.3f},{t2[1]:.3f}) | "
              + " ".join(f"{rr}:{-ss:.3f}" for ss, rr, *_ in sorted(res, key=lambda v: v[1])) + "\n")


R_LIST = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
for sgn, tag in ((1, "top"), (-1, "bot")):
    y = lambda r_: YA + sgn * r_
    up = unit(60 * sgn)
    outer_p = np.array([X0 + 10, y(21)])
    inner_p = outer_p + np.array([4 / math.sin(math.radians(60)), 0])
    # bead: R2 arc centre (X0+2, y(14)); concave fillet to groove line r=14
    cc = np.array([X0 + 2, y(14)])
    best(f"{tag} bead->groove concave", lambda r: fillet_lc((X0, y(14)), (1, 0), cc, 2, r, (X0 + 4, y(14) + sgn * r)), [2.5, 3, 3.5, 4, 4.5, 5])
    best(f"{tag} groove->face R3", lambda r: fillet_ll((X0, y(14)), (1, 0), (X0 + 10, YA), (0, 1), r, (X0 + 8, y(16))), [2, 2.5, 3, 3.5])
    best(f"{tag} face->outer wall", lambda r: fillet_ll((X0 + 10, YA), (0, 1), outer_p, up, r, (X0 + 13, y(20))), R_LIST)
    best(f"{tag} outer wall->rim face", lambda r: fillet_ll(outer_p, up, (XL, YA), (0, 1), r, (XL - 2, y(70))), R_LIST)
    best(f"{tag} inner wall->bore 135", lambda r: fillet_ll(inner_p, up, (XR, y(67.5)), (1, 0), r, (XR - 3, y(64))), R_LIST)
    best(f"{tag} inner wall->hub R7", lambda r: fillet_ll(inner_p, up, (X0 + 32, y(11)), (1, 0), r, (X0 + 25, y(16))), [5, 6, 7, 8])
out.close()
