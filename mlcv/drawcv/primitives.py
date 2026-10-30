"""Clean-up and classification of vectorised pieces (sheet mm).

Each primitive is a dict: kind line (p, q) | circle (c, r) | arc (c, r, a0, a1 counter-clockwise), plus
role, status, confidence and the measurements behind them. Nothing is deleted silently: short noise, duplicates,
text residue and unsupported arcs stay in the list with their role/status so metrics can count them."""
import math

import numpy as np

SNAP_ANGLES = np.arange(0, 180, 15.0)
SAMPLE_STEP_MM = 0.25


# ------------------------------------------------------------------ geometry helpers

def length(e):
    if e["kind"] == "line":
        return float(np.linalg.norm(e["q"] - e["p"]))
    if e["kind"] == "circle":
        return 2 * math.pi * e["r"]
    return math.radians(sweep(e)) * e["r"]


def sweep(e):
    return (e["a1"] - e["a0"]) % 360 or 360.0


def line_angle(e):
    d = e["q"] - e["p"]
    return math.degrees(math.atan2(d[1], d[0])) % 180


def angle_diff(a, b, period=180.0):
    return abs((a - b + period / 2) % period - period / 2)


def sample(e, step=SAMPLE_STEP_MM):
    if e["kind"] == "line":
        n = max(2, int(math.ceil(length(e) / step)) + 1)
        t = np.linspace(0, 1, n)[:, None]
        return e["p"] + (e["q"] - e["p"]) * t
    sw = 360.0 if e["kind"] == "circle" else sweep(e)
    a0 = 0.0 if e["kind"] == "circle" else e["a0"]
    n = max(3, int(math.ceil(math.radians(sw) * e["r"] / step)) + 1)
    a = np.radians(a0 + np.linspace(0, sw, n, endpoint=e["kind"] != "circle"))
    return e["c"] + e["r"] * np.column_stack([np.cos(a), np.sin(a)])


def endpoints(e):
    if e["kind"] == "line":
        return e["p"], e["q"]
    if e["kind"] == "arc":
        return (e["c"] + e["r"] * np.array([math.cos(math.radians(e["a0"])), math.sin(math.radians(e["a0"]))]),
                e["c"] + e["r"] * np.array([math.cos(math.radians(e["a1"])), math.sin(math.radians(e["a1"]))]))
    return None


# ------------------------------------------------------------------ lines

def _line_key(e):
    a = line_angle(e)
    t = math.radians(a)
    n = np.array([-math.sin(t), math.cos(t)])
    return a, float(e["p"] @ n), n, np.array([math.cos(t), math.sin(t)])


def merge_collinear(lines, ang_tol, off_tol, gap, inked=None, inked_gap=2.0):
    """Join collinear touching/overlapping line pieces (skeleton junctions split straight lines). A larger gap
    (up to inked_gap mm, e.g. the distorted skeleton at a crossing) is joined only when inked(p, q) >= 0.9 -
    dashed lines have empty gaps and stay apart."""
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
            if angle_diff(a2, a) < ang_tol and abs(o2 - o) < off_tol:
                group.append(e2)
                used[j] = True
        iv = sorted(((min(g["p"] @ d, g["q"] @ d), max(g["p"] @ d, g["q"] @ d), g) for g in group),
                    key=lambda t: (t[0], t[1]))
        off0 = float(np.mean([g["p"] @ n for g in group]))
        runs = []
        for s0, s1, g in iv:
            near = runs and s0 <= runs[-1][1] + gap
            if (not near and runs and inked is not None and s0 <= runs[-1][1] + inked_gap
                    and inked(n * off0 + d * runs[-1][1], n * off0 + d * s0) >= 0.9):
                near = True
            if near:
                runs[-1][1] = max(runs[-1][1], s1)
                runs[-1][2].append(g)
            else:
                runs.append([s0, s1, [g]])
        for s0, s1, members in runs:
            weights = np.array([max(length(g), 1e-6) for g in members])
            off = float(np.average([g["p"] @ n for g in members], weights=weights))
            rms = float(np.sqrt(np.average([g["rms"] ** 2 for g in members], weights=weights)))
            out.append({"kind": "line", "p": n * off + d * s0, "q": n * off + d * s1, "rms": rms,
                        "pieces": sum(g.get("pieces", 1) for g in members)})
    return out


def snap_angles(lines, tol):
    """Snap nearly axis-parallel / 15-degree lines exactly (rotation about the midpoint, length kept)."""
    for e in lines:
        L = length(e)
        if L < 1.0:
            continue
        a = line_angle(e)
        cand = SNAP_ANGLES[np.argmin([angle_diff(s, a) for s in SNAP_ANGLES])]
        tol_l = max(tol, math.degrees(math.atan(0.07 / (L / 2))))
        if angle_diff(cand, a) < tol_l and angle_diff(cand, a) > 0:
            m = (e["p"] + e["q"]) / 2
            d = np.array([math.cos(math.radians(cand)), math.sin(math.radians(cand))])
            s = np.sign((e["q"] - e["p"]) @ d) or 1
            e["p"], e["q"] = m - d * s * L / 2, m + d * s * L / 2
            e["snapped_deg"] = round(float(angle_diff(cand, a)), 4)
    return lines


def dashed_chains(lines, max_dash=30.0, max_gap=6.0, min_pieces=3, off_tol=0.25, coverage=None):
    """Collinear runs of short dashes -> one line with role axis (dash-dot) or hidden (dashed).

    Grouping is anchored on the longest dash: a piece joins when both its ends lie within off_tol of that line
    (the direction of a 2 mm dash is too noisy to compare angles). A break filled with ink (the line passes under
    a crossing contour) does not count as a dash gap. The gaps must be regular and empty.
    Returns (chains, remaining lines)."""
    short = sorted((e for e in lines if 0.3 < length(e) <= max_dash), key=lambda e: -length(e))
    used = set()
    chains = []
    for e in short:
        if id(e) in used:
            continue
        d = (e["q"] - e["p"]) / max(length(e), 1e-9)
        n = np.array([-d[1], d[0]])
        o = float(e["p"] @ n)
        grp = [g for g in short if id(g) not in used
               and abs(g["p"] @ n - o) <= off_tol and abs(g["q"] @ n - o) <= off_tol]
        if len(grp) < min_pieces:
            continue
        pts = np.array([p for g in grp for p in (g["p"], g["q"])])
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre)
        d = vt[0] if vt[0] @ d >= 0 else -vt[0]
        n = np.array([-d[1], d[0]])
        off = float(centre @ n)
        iv = sorted(((min(g["p"] @ d, g["q"] @ d), max(g["p"] @ d, g["q"] @ d), g) for g in grp), key=lambda t: t[0])
        run, runs = [list(iv[0])], []
        for s0, s1, g in iv[1:]:
            gap = s0 - run[-1][1]
            inked = gap >= 0.4 and coverage is not None and coverage(n * off + d * run[-1][1], n * off + d * s0) >= 0.9
            if gap < 0.4 or inked:
                run[-1][1] = max(run[-1][1], s1)
                run[-1].append(g)
            elif gap <= max_gap:
                run.append([s0, s1, g])
            else:
                runs.append(run)
                run = [[s0, s1, g]]
        runs.append(run)
        for run in runs:
            if len(run) < min_pieces:
                continue
            lens = np.array([r[1] - r[0] for r in run])
            gaps = np.array([b[0] - a_[1] for a_, b in zip(run[:-1], run[1:])])
            if gaps.std() > 0.5 or gaps.min() < 0.6:
                continue
            p, q = n * off + d * run[0][0], n * off + d * run[-1][1]
            if coverage is not None and coverage(p, q) > 0.93:
                continue          # gaps are filled with ink: a solid line split by crossings, not a dashed one
            kind = "axis" if (lens.max() > 2.5 * max(lens.min(), 0.3) and lens.max() > 3) else "hidden"
            members = [g for r in run for g in r[2:]]
            chains.append({"kind": "line", "p": p, "q": q, "rms": float(max(g["rms"] for g in members)),
                           "pieces": len(members), "role": kind, "dash_count": len(run)})
            used.update(id(g) for g in members)
    return chains, [e for e in lines if id(e) not in used]


# ------------------------------------------------------------------ arcs and circles

def merge_arcs(arcs, lines, center_tol, radius_tol, stroke_mm, ink_on_arc, width_on_arc, refine=None):
    """Arcs of one circle split by crossing lines are merged. A circle/arc hypothesis from the fragments is
    accepted only where the ink confirms it: ink_on_arc(c, r, a0, a1) is the share of samples along the arc
    lying on ink at least min_w wide (a thick contour arc is not bridged over a thin axis dash on the same circle).
    Fragments of different line weight are not merged. Short line pieces lying on an accepted circle/arc (chords of tiny fragments) are absorbed.
    Concentric dashed arcs (>= 4 dashes, regular gaps, no ink in the gaps) give a dashed circle/arc.
    Returns (rounds, absorbed line ids)."""
    order = sorted(range(len(arcs)), key=lambda i: -arcs[i]["sweep_fit"] * arcs[i]["r"])
    on_tol = max(0.1, 0.6 * stroke_mm)
    clusters = []
    for i in order:
        e = arcs[i]
        pts = sample({"kind": "arc", "c": e["c"], "r": e["r"], "a0": e["a0"], "a1": (e["a0"] + e["sweep_fit"]) % 360}, 0.5)
        e["w"] = width_on_arc(e["c"], e["r"], e["a0"], e["a0"] + e["sweep_fit"])
        for cl in clusters:
            ref = cl[0]
            # a short fragment has an unreliable centre: test its points against the cluster's best circle
            if (abs(e["r"] - ref["r"]) < max(radius_tol, 0.25 * ref["r"])
                    and 0.6 <= e["w"] / max(ref["w"], 1e-6) <= 1.67
                    and np.abs(np.hypot(*(pts - ref["c"]).T) - ref["r"]).max() <= on_tol):
                cl.append(e)
                break
        else:
            clusters.append([e])
    out, absorbed = [], set()
    for cl in clusters:
        w = np.array([(e["sweep_fit"] ** 2) * e["r"] for e in cl])
        c = np.average([e["c"] for e in cl], axis=0, weights=w)
        r = float(np.average([e["r"] for e in cl], weights=w))
        rms = float(np.sqrt(np.average([e["rms"] ** 2 for e in cl], weights=w)))
        iv = [(e["a0"], e["a0"] + e["sweep_fit"]) for e in cl]
        if refine is not None:
            c, r, rms = refine(c, r, rms, iv)
        min_w = 0.7 * cl[0]["w"]
        tol = max(0.1, 0.6 * stroke_mm)
        for ln in lines:
            if id(ln) in absorbed or np.linalg.norm(ln["q"] - ln["p"]) > max(1.0, 0.5 * r):
                continue
            pts = np.array([ln["p"], (ln["p"] + ln["q"]) / 2, ln["q"]])
            if np.all(np.abs(np.hypot(*(pts - c).T) - r) <= tol):
                a = np.degrees(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
                ua = np.degrees(np.unwrap(np.radians(a)))
                iv.append((min(ua[0], ua[2]), max(ua[0], ua[2])))
                absorbed.add(id(ln))
                cl.append(ln)
        join_gap = max(3.0, math.degrees(2.5 * stroke_mm / max(r, 1e-6)))
        merged = _merge_intervals(iv, join_gap)
        covered = sum(b - a for a, b in merged)
        base = {"c": c, "r": r, "rms": rms, "pieces": len(cl), "w": cl[0]["w"]}
        if covered >= 350 or (covered >= 90 and ink_on_arc(c, r, 0.0, 360.0, min_w) >= 0.97):
            out.append(dict(base, kind="circle"))
            continue
        if len(merged) >= 4:
            lens = np.array([b - a for a, b in merged])
            gaps = np.array([(merged[(k + 1) % len(merged)][0] - merged[k][1]) % 360 for k in range(len(merged))])
            inner = np.sort(gaps)[:-1] if covered < 300 else gaps
            gap_ink = np.mean([ink_on_arc(c, r, merged[k][1], merged[k][1] + gaps[k]) for k in range(len(merged))])
            if inner.max() < 25 and inner.std() < max(2.0, 0.5 * inner.mean()) and gap_ink < 0.5:
                role = "axis" if lens.max() > 2.5 * max(lens.min(), 0.5) else "hidden"
                big_gap = int(np.argmax(gaps))
                if gaps[big_gap] > 3 * np.median(gaps):        # dashed arc, not a full circle
                    a0 = merged[(big_gap + 1) % len(merged)][0] % 360
                    a1 = merged[big_gap][1] % 360
                    out.append(dict(base, kind="arc", a0=a0, a1=a1, role=role, dash_count=len(merged)))
                else:
                    out.append(dict(base, kind="circle", role=role, dash_count=len(merged)))
                continue
        # bridge gaps that are inked (the arc continues under a crossing line)
        bridged = [list(merged[0])]
        for a, b in merged[1:]:
            if ink_on_arc(c, r, bridged[-1][1], a, min_w) >= 0.9:
                bridged[-1][1] = b
            else:
                bridged.append([a, b])
        if len(bridged) > 1 and ink_on_arc(c, r, bridged[-1][1], bridged[0][0] + 360, min_w) >= 0.9:
            last = bridged.pop()
            bridged[0] = [last[0] - 360, bridged[0][1]]
        if len(bridged) == 1 and bridged[0][1] - bridged[0][0] >= 350:
            out.append(dict(base, kind="circle"))
            continue
        for a, b in bridged:
            out.append(dict(base, kind="arc", a0=a % 360, a1=b % 360))
    return out, absorbed


def _merge_intervals(iv, gap):
    """Merge angular intervals (degrees, a < b, may exceed 360) on the circle."""
    norm = sorted(((a % 360), (a % 360) + (b - a)) for a, b in iv)
    merged = []
    for a, b in norm:
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if len(merged) > 1 and merged[-1][1] >= 360 + merged[0][0] - gap:
        last = merged.pop()
        merged[0] = [last[0] - 360, max(merged[0][1], last[1] - 360)]
    return [(a, b) for a, b in merged]


# ------------------------------------------------------------------ measurements

def measure(e, px_from_mm, ink_dist, ink_dt, px_per_mm):
    """Ink support (share of samples within 1 px + half stroke of ink), stroke width and its taper."""
    pts_mm = sample(e)
    px = px_from_mm.apply(pts_mm)
    h, w = ink_dist.shape
    xi = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
    d = ink_dist[yi, xi]
    widths = 2 * ink_dt[yi, xi] / px_per_mm
    e["support"] = float(np.mean(d <= 1.5))
    on = widths[d <= 1.5]
    e["stroke_mm"] = float(np.median(on)) if len(on) else 0.0
    if len(on) >= 6 and np.std(on) > 1e-9:
        t = np.linspace(0, 1, len(on))
        with np.errstate(invalid="ignore", divide="ignore"):
            e["taper"] = float(np.nan_to_num(abs(np.corrcoef(t, on)[0, 1])))
    else:
        e["taper"] = 0.0
    return pts_mm


def confidence(e, tol_mm, min_len):
    fit = math.exp(-e["rms"] / max(tol_mm, 1e-6) * 0.5)
    size = min(1.0, length(e) / (3 * min_len))
    return round(float(e["support"] * fit * size), 4)


# ------------------------------------------------------------------ duplicates

def mark_duplicates(prims, tol):
    """Mark overlapping copies (same line/circle found twice). The longer/better one stays."""
    vis = [e for e in prims if e["status"] == "ok"]
    lines = sorted([e for e in vis if e["kind"] == "line"], key=lambda e: -length(e))
    keyed = [(line_angle(e),) + _line_key(e)[1:] + (e,) for e in lines]
    for i, (a, o, n, d, e) in enumerate(keyed):
        if e["status"] != "ok":
            continue
        s0, s1 = sorted((e["p"] @ d, e["q"] @ d))
        for a2, o2, n2, d2, e2 in keyed[i + 1:]:
            if e2["status"] != "ok" or angle_diff(a2, a) > 1.0:
                continue
            if abs((e2["p"] @ n) - o) > tol or abs((e2["q"] @ n) - o) > tol:
                continue
            t0, t1 = sorted((e2["p"] @ d, e2["q"] @ d))
            overlap = min(s1, t1) - max(s0, t0)
            if overlap >= 0.8 * (t1 - t0):
                e2["status"] = "excluded"
                e2["role"] = "duplicate"
                e2["duplicate_of"] = e
    rounds = sorted([e for e in vis if e["kind"] in ("circle", "arc")],
                    key=lambda e: (e["kind"] != "circle", -e.get("support", 1) * e.get("fit", 1), -length(e)))
    for i, e in enumerate(rounds):
        if e["status"] != "ok":
            continue
        for e2 in rounds[i + 1:]:
            if e2["status"] != "ok" or e2.get("role") != e.get("role") and {e2.get("role"), e.get("role")} & {"axis", "hidden"}:
                continue
            band = max(tol, 0.6 * max(e.get("stroke_mm", 0), e2.get("stroke_mm", 0)))
            pts = sample(e2, 0.5)
            if np.abs(np.hypot(*(pts - e["c"]).T) - e["r"]).max() <= band and (e["kind"] == "circle" or _points_on_arc(pts, e)):
                e2["status"] = "excluded"
                e2["role"] = "duplicate"
                e2["duplicate_of"] = e
    for ln in [e for e in vis if e["kind"] == "line" and e["status"] == "ok"]:
        pts = sample(ln, 0.5)
        for rd in rounds:
            if rd["status"] != "ok":
                continue
            dist = np.abs(np.hypot(*(pts - rd["c"]).T) - rd["r"])
            if dist.max() <= max(tol, 0.4) and (rd["kind"] == "circle" or _points_on_arc(pts, rd)):
                ln["status"] = "excluded"
                ln["role"] = "duplicate"
                ln["duplicate_of"] = rd
                break


def _points_on_arc(pts, arc):
    a = np.degrees(np.arctan2(pts[:, 1] - arc["c"][1], pts[:, 0] - arc["c"][0]))
    return bool(np.all((a - arc["a0"]) % 360 <= sweep(arc) + 1.0))


def _arc_inside(inner, outer):
    s = sweep(outer)
    return ((inner["a0"] - outer["a0"]) % 360 <= s + 1) and ((inner["a1"] - outer["a0"]) % 360 <= s + 1)
