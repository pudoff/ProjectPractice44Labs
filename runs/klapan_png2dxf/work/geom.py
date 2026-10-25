import math

import numpy as np


def unit(a_deg):
    a = math.radians(a_deg)
    return np.array([math.cos(a), math.sin(a)])


def line_x(p, d, x):
    t = (x - p[0]) / d[0]
    return np.asarray(p, float) + t * np.asarray(d, float)


def line_y(p, d, y):
    t = (y - p[1]) / d[1]
    return np.asarray(p, float) + t * np.asarray(d, float)


def intersect(p1, d1, p2, d2):
    p1, d1, p2, d2 = map(lambda v: np.asarray(v, float), (p1, d1, p2, d2))
    A = np.column_stack([d1, -d2])
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def fillet_ll(p1, d1, p2, d2, r, hint):
    """Fillet of radius r between lines (p1,d1) and (p2,d2); centre on the side of `hint`.
    Returns centre, tangent point on line 1, tangent point on line 2."""
    p1, d1, p2, d2, hint = map(lambda v: np.asarray(v, float), (p1, d1, p2, d2, hint))
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    n1 = np.array([-d1[1], d1[0]])
    n2 = np.array([-d2[1], d2[0]])
    if (hint - p1) @ n1 < 0:
        n1 = -n1
    if (hint - p2) @ n2 < 0:
        n2 = -n2
    c = intersect(p1 + r * n1, d1, p2 + r * n2, d2)
    t1 = c - r * n1
    t2 = c - r * n2
    return c, t1, t2


def fillet_lc(p, d, cc, rc, r, hint, external=True):
    """Fillet radius r between line (p,d) and circle (cc, rc), tangent externally (or internally).
    Centre lies on the hint side of the line; picks the solution closest to hint."""
    p, d, cc, hint = map(lambda v: np.asarray(v, float), (p, d, cc, hint))
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    if (hint - p) @ n < 0:
        n = -n
    q = p + r * n
    R = rc + r if external else abs(rc - r)
    # points on line q + t d at distance R from cc
    w = q - cc
    b = 2 * (w @ d)
    c = w @ w - R * R
    disc = b * b - 4 * c
    sols = [(-b + s * math.sqrt(max(disc, 0))) / 2 for s in (1, -1)]
    cands = [q + t * d for t in sols]
    c0 = min(cands, key=lambda v: np.linalg.norm(v - hint))
    t_line = c0 - r * n
    u = (c0 - cc) / np.linalg.norm(c0 - cc)
    t_circ = cc + rc * u if external else cc + rc * (u if r < rc else -u)
    return c0, t_line, t_circ


def ang(c, p):
    return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
