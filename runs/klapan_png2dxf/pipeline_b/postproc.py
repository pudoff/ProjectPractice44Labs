"""Clean-up of vectorised primitives: collinear merge, angle snapping, dashed/axis chains."""
import math

import numpy as np

SNAP_ANGLES = np.arange(0, 180, 15.0)


def as_dicts(prims):
    out = []
    for p in prims:
        if p[0] == "line":
            out.append({"type": "line", "p": np.asarray(p[1], float), "q": np.asarray(p[2], float)})
        else:
            out.append({"type": "arc", "c": np.asarray(p[1], float), "r": float(p[2]), "a0": float(p[3]),
                        "a1": float(p[4]), "sweep": float(p[5])})
    return out


def length(e):
    if e["type"] == "line":
        return float(np.linalg.norm(e["q"] - e["p"]))
    return math.radians(e["sweep"]) * e["r"]


def angle(e):
    d = e["q"] - e["p"]
    return math.degrees(math.atan2(d[1], d[0])) % 180


def snap_angles(lines, tol=0.35):
    for e in lines:
        if e["type"] != "line":
            continue
        L = length(e)
        if L < 1.0:
            continue
        a = angle(e)
        cand = SNAP_ANGLES[np.argmin(np.abs((SNAP_ANGLES - a + 90) % 180 - 90))]
        tol_l = max(tol, math.degrees(math.atan(0.07 / (L / 2))))
        if abs((cand - a + 90) % 180 - 90) < tol_l:
            m = (e["p"] + e["q"]) / 2
            d = np.array([math.cos(math.radians(cand)), math.sin(math.radians(cand))])
            s = np.sign((e["q"] - e["p"]) @ d) or 1
            e["p"], e["q"] = m - d * s * L / 2, m + d * s * L / 2
    return lines


def _line_key(e):
    a = angle(e)
    t = math.radians(a)
    n = np.array([-math.sin(t), math.cos(t)])
    return a, float(e["p"] @ n), n, np.array([math.cos(t), math.sin(t)])


def merge_collinear(prims, ang_tol=0.6, off_tol=0.12, gap=0.35):
    """Join line pieces that are collinear and touch/overlap (split by skeleton junctions)."""
    lines = [e for e in prims if e["type"] == "line"]
    others = [e for e in prims if e["type"] != "line"]
    keyed = sorted((_line_key(e) + (e,) for e in lines), key=lambda k: (round(k[0] / 2), k[1]))
    used = [False] * len(keyed)
    out = []
    for i, (a, o, n, d, e) in enumerate(keyed):
        if used[i]:
            continue
        group = [e]
        used[i] = True
        for j in range(i + 1, len(keyed)):
            if used[j]:
                continue
            a2, o2, _, _, e2 = keyed[j]
            if abs((a2 - a + 90) % 180 - 90) < ang_tol and abs(o2 - o) < off_tol:
                group.append(e2)
                used[j] = True
        # split the group into contiguous runs along d
        iv = sorted([tuple(sorted(((g["p"] @ d), (g["q"] @ d)))) for g in group])
        runs = []
        for s0, s1 in iv:
            if runs and s0 <= runs[-1][1] + gap:
                runs[-1][1] = max(runs[-1][1], s1)
            else:
                runs.append([s0, s1])
        off = np.mean([g["p"] @ n for g in group])
        for s0, s1 in runs:
            out.append({"type": "line", "p": n * off + d * s0, "q": n * off + d * s1})
    return out + others


def dashed_chains(lines, max_dash=14.0, max_gap=4.0, min_pieces=4, ang_tol=0.8, off_tol=0.15, coverage=None):
    """Find collinear runs of short dashes. Returns (chains, remaining). A chain is
    {'p','q','kind': 'AXIS' | 'HIDDEN'}."""
    short = [e for e in lines if e["type"] == "line" and 0.3 < length(e) <= max_dash]
    rest_ids = {id(e) for e in short}
    keyed = sorted((_line_key(e) + (e,) for e in short), key=lambda k: (round(k[0]), k[1]))
    used = set()
    chains = []
    for i, (a, o, n, d, e) in enumerate(keyed):
        if id(e) in used:
            continue
        grp = [e]
        for j in range(i + 1, len(keyed)):
            a2, o2, _, _, e2 = keyed[j]
            if id(e2) not in used and abs((a2 - a + 90) % 180 - 90) < ang_tol and abs(o2 - o) < off_tol:
                grp.append(e2)
        iv = sorted([(min(g["p"] @ d, g["q"] @ d), max(g["p"] @ d, g["q"] @ d), g) for g in grp], key=lambda t: t[0])
        run = [iv[0]]
        runs = []
        for k in iv[1:]:
            g = k[0] - run[-1][1]
            if g < 0.4:
                run[-1] = (run[-1][0], max(run[-1][1], k[1]), run[-1][2], *run[-1][3:], k[2])
            elif g <= max_gap:
                run.append(k)
            else:
                runs.append(run)
                run = [k]
        runs.append(run)
        for run in runs:
            if len(run) < min_pieces:
                continue
            lens = np.array([r[1] - r[0] for r in run])
            gaps = np.array([b[0] - a_[1] for a_, b in zip(run[:-1], run[1:])])
            if gaps.std() > 0.5 or gaps.min() < 0.6:
                continue
            off = np.mean([r[2]["p"] @ n for r in run])
            if coverage is not None and coverage(n * off + d * run[0][0], n * off + d * run[-1][1]) > 0.93:
                continue          # gaps are filled with ink: a solid line split by crossings, not a dashed one
            kind = "AXIS" if (lens.max() > 2.5 * max(lens.min(), 0.3) and lens.max() > 3) else "HIDDEN"
            chains.append({"p": n * off + d * run[0][0], "q": n * off + d * run[-1][1], "kind": kind})
            for r in run:
                for g in r[2:]:
                    used.add(id(g))
    remaining = [e for e in lines if id(e) not in used]
    return chains, remaining
