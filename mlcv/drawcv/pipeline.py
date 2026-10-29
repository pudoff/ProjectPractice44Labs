"""CV stage: page -> primitives (lines, circles, arcs) with confidence -> part contours and defects -> cv.json."""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point

from . import __version__
from .calibrate import to_sheet_transform
from .hatch import find_hatch_groups
from .ingest import load_page
from .output import build_cv_json
from .params import Params
from .preprocess import prepare
from .primitives import (confidence, dashed_chains, endpoints, length, mark_duplicates, measure, merge_arcs,
                         merge_collinear, sample, snap_angles, sweep)
from .textmask import erase_text, find_text_regions
from .topology import build as build_topology
from .vectorize import fit_circle, vectorize

ROLE_ORDER = ["part", "thin", "axis", "hidden", "hatch", "frame", "title_block", "arrowhead", "text_residue",
              "junction_artifact", "duplicate", "noise", "low_support"]
TITLE_BLOCK_MM = (185.0, 55.0)
DESIGNATION_BOX_MM = (70.0, 14.0)
FRAME_MARGINS = (20.0, 5.0, 5.0, 5.0)


class Timer:
    def __init__(self):
        self.t = time.perf_counter()
        self.stages = {}

    def __call__(self, name):
        now = time.perf_counter()
        self.stages[name] = round(now - self.t, 3)
        self.t = now


def run(input_path, out_dir=None, params=None, page_index=0):
    """Process one page. Returns the cv.json dict; writes cv.json and images into out_dir when given."""
    params = params or Params()
    T = Timer()
    page = load_page(input_path, page_index, params.pdf_render_dpi)
    T("ingest")
    prep = prepare(page, params)
    T("preprocess")

    stated = prep.px_per_mm_estimate if page.dpi else None
    pdf_origin = None
    if page.pdf_page_mm is not None:
        pdf_origin = prep.source_to_work.apply([-0.5, page.gray.shape[0] - 0.5])
    from .calibrate import calibrate
    cal = calibrate(prep.bw, params, page.pdf_page_mm, pdf_origin, stated)
    px_per_mm = float(np.mean(cal["px_per_mm"]))
    work_to_mm = to_sheet_transform(cal["px_per_mm"], cal["origin_px"])
    mm_to_work = work_to_mm.inverse()
    if prep.px_per_mm_estimate is None and params.despeckle_mm2 > 0:
        _despeckle(prep, params.despeckle_mm2 * px_per_mm ** 2)
    T("calibrate")

    regions, provider, text_note = find_text_regions(prep, prep.gray, params, px_per_mm)
    clean, erased, region_mask = erase_text(prep.bw, regions, px_per_mm, params.text_margin_mm)
    T("text_regions")

    ex = _extract(clean, work_to_mm, px_per_mm, params)
    prims, thin_mm = ex["prims"], ex["thin_mm"]
    T("vectorize")
    separable, thick_mm, threshold = _lineweights(prims, thin_mm, cal)
    _classify(prims, params, cal, region_mask, mm_to_work, thin_mm, thick_mm, threshold, separable)
    groups = find_hatch_groups([e for e in prims if e["status"] == "ok" and e["kind"] == "line" and e["role"] in ("thin", "part")],
                               params.hatch_min_lines, params.hatch_spacing_mm)
    hatch_px = 0
    if groups:
        # second pass without the hatch lines: hatch ending in a corner otherwise bends the contour skeleton
        hatch_lines = [e for g in groups for e in g["lines"]]
        clean, hatch_px = _erase_hatch(clean, hatch_lines + _family_neighbours(groups), mm_to_work, px_per_mm,
                                       thin_mm, thick_mm)
        ex = _extract(clean, work_to_mm, px_per_mm, params, thin_mm=thin_mm)
        _classify(ex["prims"], params, cal, region_mask, mm_to_work, thin_mm, thick_mm, threshold, separable)
        prims = ex["prims"] + hatch_lines
        T("hatch_second_pass")
    for g in groups:
        for e in g["lines"]:
            e["role"] = "hatch"
    widths_at = ex["widths_at"]
    pieces, sk = ex["pieces"], ex["sk"]
    mark_duplicates(prims, params.duplicate_tol_mm)
    for e in prims:
        e["confidence"] = confidence(e, params.tol_fit_mm, params.min_len_mm)
    _assign_ids(prims)
    T("classify")

    part = [e for e in prims if e["status"] == "ok" and e["role"] == "part"]
    others = [e for e in prims if e["status"] == "ok" and e["role"] != "part"]
    snap_tol = params.snap_tol_mm + params.snap_stroke_factor * (thick_mm if separable else thin_mm)
    extended = _extend_ends(part, widths_at, thick_mm if separable else thin_mm, snap_tol, params.extend_max_mm)
    topo = build_topology(part, others, snap_tol, params.gap_search_mm)
    topo["defects"] += extended
    role_rank = {"outer_boundary": 0, "circle": 1, "region": 2, "open_chain": 3}
    topo["contours"].sort(key=lambda c: (role_rank[c["role"]], c["bbox_mm"][0], c["bbox_mm"][1], -c["area_mm2"]))
    for i, c in enumerate(topo["contours"]):
        c["id"] = f"contour_{i}"
    hatches = _resolve_hatches(groups, topo["contours"], topo["defects"], thin_mm)
    T("topology")

    lineweights = {"thin_mm": round(thin_mm, 4), "thick_mm": round(thick_mm, 4), "threshold_mm": round(threshold, 4),
                   "separable": separable}
    stats = {"tolerances_mm": {"snap": round(snap_tol, 4), "snap_formula": "snap_tol_mm + snap_stroke_factor * thick_mm",
                               "gap_search": params.gap_search_mm, "fit": params.tol_fit_mm,
                               "duplicate": params.duplicate_tol_mm, "min_length": params.min_len_mm},
             "text_provider": provider, "text_note": text_note, "lineweights": lineweights,
             "raw_pieces": len(pieces), "skeleton_px": int(sk.sum()), "erased_text_px": int(erased.sum()),
             "erased_hatch_px": int(hatch_px)}
    cv = build_cv_json(page, prep, cal, work_to_mm, params, prims, regions, topo, hatches, groups, stats, T.stages,
                       __version__)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if params.save_debug_images:
            from .overlay import save_all
            save_all(out, page, prep, clean, erased, regions, prims, topo, hatches, cv, work_to_mm, params)
            T("images")
            cv["timings_s"] = dict(T.stages, total=round(sum(T.stages.values()), 3))
        (out / "cv.json").write_text(json.dumps(cv, ensure_ascii=False, indent=1), encoding="utf-8")
    return cv


def _extract(clean, work_to_mm, px_per_mm, params, thin_mm=None):
    """Vectorise a binary working image into measured primitives (roles not yet assigned)."""
    mm_to_work = work_to_mm.inverse()
    pieces, sk = vectorize(clean, work_to_mm, params.tol_fit_mm, params.arc_max_radius_mm)
    dt = cv2.distanceTransform(clean, cv2.DIST_L2, 3)
    ink_dist = cv2.distanceTransform((1 - clean).astype(np.uint8), cv2.DIST_L2, 3)
    skel_w = 2 * dt[sk > 0]
    if thin_mm is None:
        thin_mm = float(np.percentile(skel_w, 20)) / px_per_mm if len(skel_w) else 0.25

    raw_lines = [p for p in pieces if p["kind"] == "line"]
    raw_arcs = [dict(p, sweep_fit=p["sweep"]) for p in pieces if p["kind"] == "arc"]

    def coverage(p, q):
        e = {"kind": "line", "p": p, "q": q}
        px = mm_to_work.apply(sample(e, 0.1))
        h, w = clean.shape
        xi = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        return float(np.mean(ink_dist[yi, xi] <= 1.0))

    lines = merge_collinear(raw_lines, params.merge_angle_deg, params.merge_offset_mm, params.merge_gap_mm,
                            inked=coverage, inked_gap=max(1.0, 6 * thin_mm))
    snap_angles(lines, params.snap_angle_deg)

    def _arc_px(c, r, a0, a1):
        e = {"kind": "arc", "c": np.asarray(c), "r": r, "a0": a0 % 360, "a1": a1 % 360}
        if a1 - a0 >= 359.999:
            e = {"kind": "circle", "c": np.asarray(c), "r": r}
        px = mm_to_work.apply(sample(e, 0.1))
        h, w = clean.shape
        return (np.clip(np.round(px[:, 1]).astype(int), 0, h - 1), np.clip(np.round(px[:, 0]).astype(int), 0, w - 1))

    def ink_on_arc(c, r, a0, a1, min_w=0.0):
        """Share of samples along the arc on ink at least min_w mm wide."""
        if a1 - a0 <= 0:
            return 1.0
        yi, xi = _arc_px(c, r, a0, a1)
        on = ink_dist[yi, xi] <= 1.5
        if min_w > 0:
            on &= 2 * _local_max(dt, yi, xi) / px_per_mm >= min_w
        return float(np.mean(on))

    def widths_along(c, r, a0, a1):
        """Stroke width (mm) at 0.1 mm samples along the arc, 0 where there is no ink."""
        yi, xi = _arc_px(c, r, a0, a1)
        on = ink_dist[yi, xi] <= 1.5
        return np.where(on, 2 * _local_max(dt, yi, xi) / px_per_mm, 0.0)

    def widths_at(pts_mm):
        px = mm_to_work.apply(pts_mm)
        h, w = clean.shape
        xi = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        return np.where(ink_dist[yi, xi] <= 1.5, 2 * _local_max(dt, yi, xi) / px_per_mm, 0.0)

    def width_on_arc(c, r, a0, a1):
        yi, xi = _arc_px(c, r, a0, a1)
        on = ink_dist[yi, xi] <= 1.5
        return float(np.median(2 * _local_max(dt, yi, xi)[on] / px_per_mm)) if on.any() else 0.0

    sk_rc = np.column_stack(np.nonzero(sk))
    skel_mm = work_to_mm.apply(sk_rc[:, ::-1].astype(float)) if len(sk_rc) else np.zeros((0, 2))
    skel_tree = cKDTree(skel_mm) if len(skel_mm) else None

    def refine_circle(c, r, rms, intervals=None):
        """Least-squares circle through skeleton points in a narrowing band around the hypothesis, restricted to
        the angular ranges of the fragments, with median-deviation outlier rejection."""
        if skel_tree is None:
            return c, r, rms
        band = max(0.3, 2 * thin_mm, 0.02 * r)
        for _ in range(4):
            idx = skel_tree.query_ball_point(c, r + band)
            if len(idx) < 12:
                break
            P = skel_mm[idx]
            dev = np.hypot(*(P - c).T) - r
            sel = np.abs(dev) < band
            if intervals:
                ang = np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
                ok = np.zeros(len(P), bool)
                for a0, a1 in intervals:
                    # junction zones at the ends of a fragment distort the skeleton: fit on the middle part
                    trim = min(0.15 * (a1 - a0), math.degrees(3.0 / max(r, 1e-6)))
                    ok |= (ang - a0 - trim) % 360 <= max(a1 - a0 - 2 * trim, 1.0)
                sel &= ok
            if sel.sum() < 12:
                break
            d = dev[sel]
            mad = np.median(np.abs(d - np.median(d))) + 1e-6
            keep = np.abs(d - np.median(d)) <= max(3 * mad, 0.05)
            fc = fit_circle(P[sel][keep])
            if fc is None or abs(fc[2] - r) > band:
                break
            c, r = np.array(fc[:2]), float(fc[2])
            rms = float(np.sqrt(np.mean((np.hypot(*(P[sel][keep] - c).T) - r) ** 2)))
            band = max(0.15, 0.6 * band)
        return c, r, rms

    chains, lines = dashed_chains(lines, max_dash=params.dash_max_mm, max_gap=params.dash_gap_max_mm,
                                  min_pieces=params.dash_min_pieces, coverage=coverage)
    rounds, absorbed = merge_arcs(raw_arcs, lines, center_tol=max(0.15, 1.5 * thin_mm), radius_tol=max(0.15, thin_mm),
                                  stroke_mm=thin_mm, ink_on_arc=ink_on_arc, width_on_arc=width_on_arc,
                                  refine=refine_circle)
    lines = [e for e in lines if id(e) not in absorbed]
    _snap_concentric(rounds, skel_tree, skel_mm, thin_mm)
    for e in rounds:
        if e["kind"] == "arc" and e.get("concentric_with") is not None:
            _trim_arc(e, widths_along, thin_mm)
    rounds = _join_concentric(rounds, ink_on_arc)
    for e in rounds:
        if e["kind"] == "arc" and e.get("concentric_with") is not None:
            _trim_arc(e, widths_along, thin_mm)
    prims = lines + chains + rounds
    for e in prims:
        e.setdefault("role", None)
        e["status"] = "ok"
        e["reasons"] = []
        measure(e, mm_to_work, ink_dist, dt, px_per_mm)
        e["length_mm"] = length(e)
        e["fit"] = math.exp(-e["rms"] / params.tol_fit_mm * 0.5)
    return {"prims": prims, "pieces": pieces, "sk": sk, "thin_mm": thin_mm, "widths_at": widths_at}



def _family_neighbours(groups):
    """Lines one step beyond the outermost members of each hatch family, clipped to the family's hull grown by 1.5
    step: short hatch lines cut by a corner merge into the contour in the first pass and are not found as members."""
    from shapely.geometry import LineString, MultiPoint

    out = []
    for g in groups:
        ang = math.radians(g["angle_deg"])
        d = np.array([math.cos(ang), math.sin(ang)])
        n = np.array([-d[1], d[0]])
        offs = [((e["p"] + e["q"]) / 2) @ n for e in g["lines"]]
        hull = MultiPoint([tuple(p) for e in g["lines"] for p in (e["p"], e["q"])]).convex_hull.buffer(1.5 * g["spacing_mm"])
        span = max(hull.bounds[2] - hull.bounds[0], hull.bounds[3] - hull.bounds[1]) * 2
        centre = np.array(hull.centroid.coords[0])
        t0 = centre @ d
        for o in (min(offs) - g["spacing_mm"], max(offs) + g["spacing_mm"]):
            base = n * o + d * t0
            seg = LineString([tuple(base - d * span), tuple(base + d * span)]).intersection(hull)
            if seg.is_empty or seg.geom_type != "LineString" or seg.length < 0.3:
                continue
            (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
            out.append({"kind": "line", "p": np.array([x0, y0]), "q": np.array([x1, y1])})
    return out


def _erase_hatch(clean, hatch_lines, mm_to_work, px_per_mm, thin_mm, thick_mm):
    """Remove hatch line pixels from the working copy, except pixels belonging to thick strokes (an opening
    with a disc of 0.7 x thick width keeps thick lines and drops thin ones)."""
    band = np.zeros_like(clean)
    w_px = max(2, int(round((thin_mm * 1.6) * px_per_mm)) + 2)
    for e in hatch_lines:
        d = (e["q"] - e["p"]) / max(np.linalg.norm(e["q"] - e["p"]), 1e-9)
        a, b = mm_to_work.apply(np.array([e["p"] - d * thick_mm, e["q"] + d * thick_mm]))
        cv2.line(band, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 1, w_px)
    k = max(3, int(round(0.7 * thick_mm * px_per_mm)))
    thick = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    erase = (band > 0) & (clean > 0) & (thick == 0)
    out = clean.copy()
    out[erase] = 0
    return out, int(erase.sum())


def _snap_concentric(rounds, tree, skel_mm, thin_mm):
    """A short arc has a poorly conditioned centre (35 deg of a R40 arc with 0.05 mm noise fits R38).
    If centring it on an existing circle (or long arc) fits its ink as well as the free fit, use that centre."""
    if tree is None:
        return
    centres = [e for e in rounds if e["kind"] == "circle" or (e["kind"] == "arc" and sweep(e) >= 180)]
    for e in rounds:
        if e["kind"] != "arc" or sweep(e) >= 120 or e.get("role") in ("axis", "hidden"):
            continue
        idx = tree.query_ball_point(e["c"], e["r"] + 0.3)
        if len(idx) < 10:
            continue
        P = skel_mm[idx]
        d = np.hypot(*(P - e["c"]).T)
        ang = np.degrees(np.arctan2(P[:, 1] - e["c"][1], P[:, 0] - e["c"][0]))
        sw = sweep(e)
        trim = min(0.15 * sw, math.degrees(3.0 / max(e["r"], 1e-6)))
        sel = (np.abs(d - e["r"]) < 0.3) & ((ang - e["a0"] - trim) % 360 <= sw - 2 * trim)
        if sel.sum() < 10:
            continue
        P = P[sel]
        fc = fit_circle(P)
        if fc is None:
            continue
        free = float(np.sqrt(np.mean((np.hypot(*(P - fc[:2]).T) - fc[2]) ** 2)))
        best = None
        for o in centres:
            reach = (0.25 if sw < 45 else 0.1) * e["r"] + 0.5      # the shorter the arc, the less its centre is known
            if o is e or np.linalg.norm(o["c"] - e["c"]) > reach:
                continue
            dd = np.hypot(*(P - o["c"]).T)
            r = float(np.median(dd))
            rms = float(np.sqrt(np.mean((dd - r) ** 2)))
            if rms <= max(1.5 * free, free + 0.03) and (best is None or rms < best[0]):
                best = (rms, o, r)
        if best is None:
            continue
        p0, p1 = endpoints(e)
        c = best[1]["c"].copy()
        e.update(c=c, r=best[2], rms=best[0], concentric_with=best[1])
        e["a0"] = math.degrees(math.atan2(p0[1] - c[1], p0[0] - c[0])) % 360
        e["a1"] = math.degrees(math.atan2(p1[1] - c[1], p1[0] - c[0])) % 360


def _extend_ends(part, widths_at, thick_mm, snap_tol, max_ext):
    """Skeleton ends stop short of junctions. An end with nothing within snap_tol is extended along its own line
    or circle while the ink under it is thick, until it meets other part geometry (max max_ext mm).
    Each extension is reported (code end_extended, severity info); a real break (no ink) is never bridged."""
    if not part or max_ext <= 0:
        return []
    pts, owner = [], []
    for e in part:
        s_ = sample(e, 0.1)
        pts.append(s_)
        owner += [id(e)] * len(s_)
    pts = np.vstack(pts)
    owner = np.array(owner)
    tree = cKDTree(pts)
    out = []
    step = 0.1
    for e in part:
        if e["kind"] == "circle":
            continue
        for which in ("start", "end"):
            p0, p1 = endpoints(e)
            end = p0 if which == "start" else p1
            near = [k for k in tree.query_ball_point(end, snap_tol) if owner[k] != id(e)]
            if near:
                continue
            n = int(max_ext / step)
            if e["kind"] == "line":
                d = (e["p"] - e["q"]) if which == "start" else (e["q"] - e["p"])
                d = d / (np.linalg.norm(d) + 1e-12)
                cand = end + np.outer(np.arange(1, n + 1) * step, d)
            else:
                da = math.degrees(step / e["r"]) * (-1 if which == "start" else 1)
                base = e["a0"] if which == "start" else e["a1"]
                ang = np.radians(base + da * np.arange(1, n + 1))
                cand = e["c"] + e["r"] * np.column_stack([np.cos(ang), np.sin(ang)])
            w = widths_at(cand)
            hit = None
            for k in range(n):
                if w[k] < 0.7 * thick_mm:
                    break
                others = [j for j in tree.query_ball_point(cand[k], 0.5 * snap_tol) if owner[j] != id(e)]
                if others:
                    hit = k
                    break
            if hit is None:
                # free end: the skeleton stops about half a stroke short of the ink end (round/square caps)
                run_ = 0
                while run_ < n and w[run_] >= 0.7 * thick_mm:
                    run_ += 1
                corr = run_ * step - thick_mm / 2
                if corr > 0.05:
                    _move_end(e, which, cand[min(n - 1, max(0, int(round(corr / step)) - 1))], corr)
                    e.setdefault("extended_mm", {})[which] = round(corr, 3)
                continue
            length_mm = (hit + 1) * step
            if e["kind"] == "line":
                e["p" if which == "start" else "q"] = cand[hit].copy()
            else:
                key = "a0" if which == "start" else "a1"
                e[key] = (e[key] + math.degrees(length_mm / e["r"]) * (-1 if which == "start" else 1)) % 360
            e.setdefault("extended_mm", {})[which] = round(length_mm, 3)
            e["length_mm"] = length(e)
            out.append({"code": "end_extended", "severity": "info", "at_mm": [round(float(v), 4) for v in cand[hit]],
                        "distance_mm": round(length_mm, 3), "object_ids": [e["id"]],
                        "message": f"{which} extended {length_mm:.2f} mm along its own curve over thick ink"})
    # a corner may have lost up to the length of an excluded junction arc (< 3 strokes): allow 4 strokes there
    out += _close_corners(part, widths_at, thick_mm, snap_tol, max(max_ext, 4 * thick_mm))
    return out


def _move_end(e, which, point, length_mm):
    if e["kind"] == "line":
        e["p" if which == "start" else "q"] = np.asarray(point, float).copy()
    else:
        key = "a0" if which == "start" else "a1"
        e[key] = (e[key] + math.degrees(length_mm / e["r"]) * (-1 if which == "start" else 1)) % 360
    e["length_mm"] = length(e)


def _close_corners(part, widths_at, thick_mm, snap_tol, max_ext):
    """Two free line ends whose extensions meet within max_ext mm: at an acute corner the skeleton of both
    lines stops early. The corner is restored only if both extensions run over thick ink."""
    ends = []
    for e in part:
        if e["kind"] != "line":
            continue
        for which in ("p", "q"):
            ends.append((e, which))
    free = []
    for e, which in ends:
        pt = e[which]
        if not any(o is not e and _dist_to_prim(pt, o) <= snap_tol for o in part):
            free.append((e, which))
    out, used = [], set()
    for i, (e1, w1) in enumerate(free):
        for e2, w2 in free[i + 1:]:
            if e1 is e2 or (id(e1), w1) in used or (id(e2), w2) in used:
                continue
            a, b = e1[w1], e2[w2]
            if np.linalg.norm(a - b) > 2 * max_ext:
                continue
            d1 = e1[w1] - e1["q" if w1 == "p" else "p"]
            d2 = e2[w2] - e2["q" if w2 == "p" else "p"]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9 * np.linalg.norm(d1) * np.linalg.norm(d2):
                continue
            t = ((b - a)[0] * d2[1] - (b - a)[1] * d2[0]) / den
            x = a + t * d1
            l1, l2 = np.linalg.norm(x - a), np.linalg.norm(x - b)
            if l1 > max_ext or l2 > max_ext or (x - a) @ d1 < -1e-9 or (x - b) @ d2 < -1e-9:
                continue
            ok = True
            for start, length_ in ((a, l1), (b, l2)):
                if length_ > 0.05:
                    n = max(2, int(length_ / 0.1))
                    pts = start + np.outer(np.linspace(0, 1, n), x - start)
                    ok &= bool(np.mean(widths_at(pts) >= 0.7 * thick_mm) >= 0.9)
            if not ok:
                continue
            e1[w1], e2[w2] = x.copy(), x.copy()
            used.update({(id(e1), w1), (id(e2), w2)})
            for e, length_ in ((e1, l1), (e2, l2)):
                e.setdefault("extended_mm", {})["start" if e is e1 and w1 == "p" or e is e2 and w2 == "p" else "end"] = round(float(length_), 3)
                e["length_mm"] = length(e)
            out.append({"code": "corner_restored", "severity": "info", "at_mm": [round(float(v), 4) for v in x],
                        "distance_mm": round(float(max(l1, l2)), 3), "object_ids": [e1["id"], e2["id"]],
                        "message": f"corner restored: ends extended {l1:.2f} / {l2:.2f} mm over thick ink to their intersection"})
    return out


def _dist_to_prim(pt, e):
    if e["kind"] == "line":
        d = e["q"] - e["p"]
        t = np.clip((pt - e["p"]) @ d / max(d @ d, 1e-12), 0, 1)
        return float(np.linalg.norm(e["p"] + t * d - pt))
    r = abs(np.linalg.norm(pt - e["c"]) - e["r"])
    if e["kind"] == "circle":
        return float(r)
    a = math.degrees(math.atan2(pt[1] - e["c"][1], pt[0] - e["c"][0]))
    if (a - e["a0"]) % 360 <= sweep(e):
        return float(r)
    return float(min(np.linalg.norm(pt - q) for q in endpoints(e)))


def _join_concentric(rounds, ink_on_arc):
    """Arcs snapped to the same centre with the same radius are one arc where the gap between them is inked
    with the arcs' own line weight (a crossing line split them)."""
    groups, rest = {}, []
    for e in rounds:
        if e["kind"] == "arc" and e.get("concentric_with") is not None:
            key = (id(e["concentric_with"]), round(e["r"] / 0.1))
            groups.setdefault(key, []).append(e)
        else:
            rest.append(e)
    for arcs in groups.values():
        arcs.sort(key=lambda e: e["a0"])
        merged = [arcs[0]]
        for e in arcs[1:]:
            m = merged[-1]
            gap_from = m["a0"] + sweep(m)
            gap_to = e["a0"] if e["a0"] >= m["a0"] else e["a0"] + 360
            w = 0.7 * min(m["w"], e["w"])
            if w > 0 and (gap_to - gap_from <= 0 or ink_on_arc(m["c"], m["r"], gap_from, gap_to, w) >= 0.9):
                end = max(gap_from, gap_to + sweep(e))
                m["a1"] = end % 360
                m["pieces"] = m.get("pieces", 1) + e.get("pieces", 1)
            else:
                merged.append(e)
        rest += merged
    return rest


def _trim_arc(e, widths_along, thin_mm):
    """After the centre changed, keep the longest run of the arc covered by ink of its own (thick) weight;
    breaks shorter than 1.5 strokes (a crossing line) are tolerated. The arc is only shortened, never extended."""
    sw = sweep(e)
    w = widths_along(e["c"], e["r"], e["a0"], e["a0"] + sw)
    if len(w) < 5 or not (w > 0).any():
        return
    thick = float(np.percentile(w[w > 0], 90))
    on = w >= 0.7 * thick
    step_mm = math.radians(sw) * e["r"] / (len(w) - 1)
    max_gap = int(max(1.5 * thick, 3 * thin_mm) / step_mm)
    runs, start, gap = [], None, 0
    for k, v in enumerate(on):
        if v:
            if start is None:
                start = k
            last, gap = k, 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, last))
                start = None
    if start is not None:
        runs.append((start, last))
    if not runs:
        return
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if (b - a) * step_mm < 1.0:
        return
    k = sw / (len(w) - 1)
    e["a0"], e["a1"] = (e["a0"] + a * k) % 360, (e["a0"] + b * k) % 360


def _local_max(img, yi, xi):
    """Value of img at the samples, taking the maximum over the 3x3 neighbourhood (samples may sit off-centre)."""
    h, w = img.shape
    best = img[yi, xi]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            best = np.maximum(best, img[np.clip(yi + dy, 0, h - 1), np.clip(xi + dx, 0, w - 1)])
    return best


def _despeckle(prep, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(prep.bw, connectivity=8)
    small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
    if len(small):
        mask = np.isin(lab, small)
        prep.bw[mask] = 0
        prep.speck_mask = mask
        prep.specks_removed = int(len(small))


def _lineweights(prims, thin_mm, cal):
    """Thick (contour) vs thin line width from the length-weighted 90th percentile of stroke widths of the drawing
    body. The frame and title block are excluded: they are long and thick and on some sheets (an A1 drawing reduced
    to A3) thicker than the part contour, which would then be called thin."""
    cand = [e for e in prims if e["length_mm"] > 3 and e["support"] > 0.8 and _in_body(e, cal)]
    if len(cand) < 5:
        cand = [e for e in prims if e["length_mm"] > 3 and e["support"] > 0.8]
    if not cand:
        return False, thin_mm, thin_mm * 1.5
    w = np.array([e["stroke_mm"] for e in cand])
    L = np.array([e["length_mm"] for e in cand])
    order = np.argsort(w)
    cum = np.cumsum(L[order]) / L.sum()
    thick = float(w[order][np.searchsorted(cum, 0.9)])
    separable = thick >= 1.5 * thin_mm
    threshold = math.sqrt(thin_mm * thick) if separable else thin_mm * 1.5
    return separable, thick, threshold


def _classify(prims, params, cal, region_mask, mm_to_work, thin_mm, thick_mm, threshold, separable):
    W, H = cal["sheet_mm"]
    use_frame = cal["method"] != "image_size"
    ml, mb, mr, mt = FRAME_MARGINS
    pad = 0.8
    tb = (W - mr - TITLE_BLOCK_MM[0] - pad, mb - pad, W - mr + pad, mb + TITLE_BLOCK_MM[1] + pad)
    db = (ml - pad, H - mt - DESIGNATION_BOX_MM[1] - pad, ml + DESIGNATION_BOX_MM[0] + pad, H - mt + pad)
    h, w = region_mask.shape
    for e in prims:
        pts = sample(e)
        if e["length_mm"] < params.min_len_mm:
            e.update(role="noise", status="excluded")
            e["reasons"].append(f"shorter than {params.min_len_mm} mm")
            continue
        px = mm_to_work.apply(pts)
        xi = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        if np.mean(region_mask[yi, xi]) >= 0.6:
            e.update(role="text_residue", status="excluded")
            e["reasons"].append("lies inside a text region")
            continue
        if e["support"] < (0.8 if e["kind"] != "line" and e["role"] not in ("axis", "hidden") else 0.5):
            e.update(role="low_support", status="excluded")
            e["reasons"].append(f"ink support {e['support']:.2f} too low")
            continue
        if use_frame:
            x, y = pts[:, 0], pts[:, 1]
            outside = (x < ml + pad) | (x > W - mr - pad) | (y < mb + pad) | (y > H - mt - pad)
            if outside.all():
                e["role"] = "frame"
                continue
            if _inside(pts, tb) or _inside(pts, db):
                e["role"] = "title_block"
                continue
        if e["kind"] == "arc" and e["r"] < 4 * thick_mm and e["length_mm"] < 3 * thick_mm:
            e.update(role="junction_artifact", status="excluded")
            e["reasons"].append("short small-radius arc where thick lines meet (skeleton corner rounding)")
            continue
        if e["kind"] == "arc":
            bad = []
            if e["r"] < params.arc_min_radius_mm:
                bad.append(f"radius {e['r']:.2f} < {params.arc_min_radius_mm} mm")
            if sweep(e) < params.arc_min_sweep_deg:
                bad.append(f"sweep {sweep(e):.1f} < {params.arc_min_sweep_deg} deg")
            if bad:
                e["status"] = "unsupported"
                e["reasons"] += bad
        if e["role"] in ("axis", "hidden"):
            continue
        if (e["kind"] == "line" and e["length_mm"] < 6 and e.get("taper", 0) > 0.8
                and e["stroke_mm"] > 1.4 * thin_mm):
            e.update(role="arrowhead", status="excluded")
            e["reasons"].append("tapered filled stroke (dimension arrow)")
            continue
        e["role"] = "part" if (not separable or e["stroke_mm"] >= threshold) else "thin"


def _in_body(e, cal):
    if cal["method"] == "image_size":
        return True
    W, H = cal["sheet_mm"]
    ml, mb, mr, mt = FRAME_MARGINS
    pts = sample(e, 1.0)
    inside = ((pts[:, 0] > ml + 1) & (pts[:, 0] < W - mr - 1) & (pts[:, 1] > mb + 1) & (pts[:, 1] < H - mt - 1)).all()
    tb = (W - mr - TITLE_BLOCK_MM[0] - 1, mb - 1, W - mr + 1, mb + TITLE_BLOCK_MM[1] + 1)
    return bool(inside) and not _inside(pts, tb)


def _inside(pts, box):
    x0, y0, x1, y1 = box
    return bool(((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).all())


def _assign_ids(prims):
    """Deterministic ids: per kind, ordered by status, role and position."""
    def key(e):
        anchor = e["p"] if e["kind"] == "line" else e["c"]
        return (e["status"] != "ok", ROLE_ORDER.index(e["role"]) if e["role"] in ROLE_ORDER else 99,
                round(float(anchor[0]), 2), round(float(anchor[1]), 2), round(e["length_mm"], 2))
    counters = {}
    for e in sorted(prims, key=key):
        k = e["kind"]
        e["id"] = f"{k}_{counters.get(k, 0)}"
        counters[k] = counters.get(k, 0) + 1
    prims.sort(key=lambda e: (e["kind"], int(e["id"].split("_")[1])))


def _resolve_hatches(groups, contours, defects, thin_mm):
    """A hatch boundary is taken only in the simple case: one minimal closed contour holds >= 90 % of the hatch
    lines and >= 60 % of their ends lie on its boundary. Otherwise the boundary is left as a question."""
    closed = [c for c in contours if c["closed"] and c["faces_inside"] <= 1]
    out = []
    for k, g in enumerate(groups):
        g["id"] = f"hatchgroup_{k}"
        mids = [Point(*((e["p"] + e["q"]) / 2)) for e in g["lines"]]
        best = None
        for c in closed:
            inside = sum(c["_poly"].contains(m) for m in mids) / len(mids)
            if inside >= 0.9 and (best is None or c["area_mm2"] < best["area_mm2"]):
                best = c
        ok = False
        if best is not None:
            bnd = best["_poly"].boundary
            ends = [Point(*p) for e in g["lines"] for p in (e["p"], e["q"])]
            on = sum(bnd.distance(p) <= 0.5 + 2 * thin_mm for p in ends) / len(ends)
            g["ends_on_boundary"] = round(on, 3)
            ok = on >= 0.6
        if ok:
            holes = [c for c in contours if c is not best and c["closed"] and best["_poly"].contains(c["_poly"])
                     and not any(c["_poly"].contains(m) for m in mids)]
            g.update(boundary="resolved", contour_id=best.get("id"), _contour=best, _holes=holes)
            for e in g["lines"]:
                e["status"] = "converted"
            out.append(g)
        else:
            g.update(boundary="unresolved")
            defects.append({"code": "hatch_boundary_unresolved", "severity": "warning",
                            "at_mm": [round(float(np.mean([m.x for m in mids])), 4), round(float(np.mean([m.y for m in mids])), 4)],
                            "object_ids": [e["id"] for e in g["lines"]],
                            "message": f"{len(g['lines'])} hatch lines at {g['angle_deg']} deg: no closed part contour "
                                       "bounds them in the supported simple way; hatch lines kept as lines, needs review"})
    return out
