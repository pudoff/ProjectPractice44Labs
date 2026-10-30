"""Hatch line families: >= N parallel lines with a regular local spacing.

Supported case: inclined hatching (angle not within 5 degrees of 0/90) made of solid parallel lines.
Horizontal/vertical families are left alone, because stacked dimension lines and title-block rows look the
same; this limitation is reported, not hidden."""
import math

import numpy as np

from .primitives import angle_diff, line_angle

AXIS_EXCLUSION_DEG = 5.0
ANGLE_TOL_DEG = 1.0
REGULARITY = 0.6


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def find_hatch_groups(lines, min_lines, spacing_range):
    smin, smax = spacing_range
    cands = [e for e in lines if min(angle_diff(line_angle(e), 0), angle_diff(line_angle(e), 90)) > AXIS_EXCLUSION_DEG]
    cands.sort(key=line_angle)
    clusters, cur = [], []
    for e in cands:
        if cur and angle_diff(line_angle(e), line_angle(cur[-1])) > ANGLE_TOL_DEG:
            clusters.append(cur)
            cur = []
        cur.append(e)
    if cur:
        clusters.append(cur)
    if len(clusters) > 1 and angle_diff(line_angle(clusters[0][0]), line_angle(clusters[-1][-1])) <= ANGLE_TOL_DEG:
        clusters[0] = clusters.pop() + clusters[0]

    groups = []
    for cl in clusters:
        if len(cl) < min_lines:
            continue
        ang = math.radians(line_angle(cl[len(cl) // 2]))
        d = np.array([math.cos(ang), math.sin(ang)])
        n = np.array([-d[1], d[0]])
        off = np.array([((e["p"] + e["q"]) / 2) @ n for e in cl])
        iv = np.array([sorted((e["p"] @ d, e["q"] @ d)) for e in cl])
        parent = list(range(len(cl)))
        order = np.argsort(off)
        for a_i, i in enumerate(order):
            for j in order[a_i + 1:]:
                if off[j] - off[i] > smax:
                    break
                if off[j] - off[i] < smin * 0.5:
                    continue
                gap = max(iv[i][0], iv[j][0]) - min(iv[i][1], iv[j][1])
                if gap <= smax:
                    parent[_find(parent, i)] = _find(parent, j)
        comps = {}
        for i in range(len(cl)):
            comps.setdefault(_find(parent, i), []).append(i)
        for members in comps.values():
            if len(members) < min_lines:
                continue
            o = np.sort(off[members])
            diffs = np.diff(o)
            diffs = diffs[diffs > smin * 0.5]
            if len(diffs) < min_lines - 1:
                continue
            spacing = float(np.median(diffs))
            if not smin <= spacing <= smax:
                continue
            regular = float(np.mean(np.abs(diffs - spacing) <= 0.3 * spacing))
            if regular < REGULARITY:
                continue
            groups.append({"lines": [cl[i] for i in members], "angle_deg": round(math.degrees(ang) % 180, 3),
                           "spacing_mm": round(spacing, 4), "regularity": round(regular, 3)})
    return groups
