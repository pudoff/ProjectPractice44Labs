"""Variant B: automatic raster -> ESKD DXF pipeline (no sheet-specific input).

usage: python pipeline.py INPUT.png OUTPUT.dxf [--sheet A3] [--report report.json]
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))

import dims as dimsmod  # noqa: E402
import hatch as hatchmod  # noqa: E402
import symbols  # noqa: E402
from eskd_dxf import new_drawing, verify  # noqa: E402
from ocr import DIAM, run_ocr  # noqa: E402
from postproc import angle, as_dicts, dashed_chains, length, merge_collinear, snap_angles  # noqa: E402
from raster import SHEETS, binarize, calibrate, deskew, estimate_skew, load_gray, stroke_width_mm, to_mm, to_px  # noqa: E402
from vectorize import vectorize  # noqa: E402


GOST_H = (2.5, 3.5, 5.0, 7.0, 10.0, 14.0)


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0
        self.stages = {}

    def __call__(self, name):
        now = time.time()
        self.stages[name] = round(now - self.last, 2)
        self.last = now


# ---------------------------------------------------------------- masks

def text_erase_mask(bw, cal, texts, min_score=0.5, pad_mm=0.4, inside=0.9):
    """Erase ink components that lie (almost) completely inside recognised text boxes."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    erase = np.zeros_like(bw)
    ppm = cal["px_per_mm_x"]
    for t in texts:
        t.glyph_h = None
        if t.score < min_score:
            continue
        c = t.center
        poly = c + (t.poly - c) * 1.0
        d = t.poly[1] - t.poly[0]
        d = d / (np.linalg.norm(d) + 1e-9)
        nrm = np.array([-d[1], d[0]])
        poly = np.array([p + (np.sign((p - c) @ d) * d + np.sign((p - c) @ nrm) * nrm) * pad_mm for p in t.poly])
        px, py = to_px(cal, poly[:, 0], poly[:, 1])
        pp = np.column_stack([px, py])
        x0, y0 = np.floor(pp.min(0)).astype(int)
        x1, y1 = np.ceil(pp.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, bw.shape[1] - 1), min(y1, bw.shape[0] - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(m, [np.round(pp - [x0, y0]).astype(np.int32)], 1)
        sub = lab[y0:y1 + 1, x0:x1 + 1]
        ids, cnt = np.unique(sub[(m > 0) & (sub > 0)], return_counts=True)
        across = []
        vertical = abs(math.sin(math.radians(t.angle))) > 0.7
        for i, k in zip(ids, cnt):
            if k >= inside * stats[i, cv2.CC_STAT_AREA] and stats[i, cv2.CC_STAT_AREA] < (12 * ppm) ** 2:
                X, Y, W, H = stats[i, :4]
                erase[Y:Y + H, X:X + W][lab[Y:Y + H, X:X + W] == i] = 1
                acr, alo = (W, H) if vertical else (H, W)
                if alo <= 1.6 * acr:
                    across.append(acr / ppm)
        box_h = t.height * 0.57
        if len(across) >= 3:
            a_ = np.array(across)
            a_ = a_[a_ > 0.35 * a_.max()]
            g = float(np.percentile(a_, 80)) - 0.25
            # long lines: the OCR line box is reliable; short labels/numbers: single glyph heights
            t.glyph_h = box_h if len(t.text) >= 6 or abs(g - box_h) > 0.45 * box_h else g
        else:
            t.glyph_h = box_h
    return erase


# ---------------------------------------------------------------- line surgery

def cut_interval(e, a, b):
    """Remove the part of line e between points a and b (both on/near its supporting line).
    Returns list of remaining pieces."""
    d = e["q"] - e["p"]
    L = np.linalg.norm(d)
    if L < 1e-9:
        return []
    u = d / L
    s0, s1 = sorted(((a - e["p"]) @ u, (b - e["p"]) @ u))
    out = []
    if s0 > 0.3:
        out.append({"type": "line", "p": e["p"].copy(), "q": e["p"] + u * min(s0, L)})
    if s1 < L - 0.3:
        out.append({"type": "line", "p": e["p"] + u * max(s1, 0), "q": e["q"].copy()})
    return [o for o in out if np.linalg.norm(o["q"] - o["p"]) > 0.3]


def remove_collinear_run(lines, a, b, extend=1.0, off_tol=0.2):
    """Remove all pieces of lines collinear with segment a-b that overlap [a, b] (grown through
    contiguous pieces). Returns new list."""
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    lo, hi = 0.0, float(np.linalg.norm(b - a))
    coll = []
    for k, e in enumerate(lines):
        if e["type"] != "line":
            continue
        if abs((e["p"] - a) @ n) < off_tol and abs((e["q"] - a) @ n) < off_tol:
            s0, s1 = sorted(((e["p"] - a) @ u, (e["q"] - a) @ u))
            coll.append((s0, s1, k))
    coll.sort()
    changed = True
    while changed:
        changed = False
        for s0, s1, k in coll:
            if s0 <= hi + extend and s1 >= lo - extend and (s0 < lo or s1 > hi):
                lo, hi = min(lo, s0), max(hi, s1)
                changed = True
    kill = {k for s0, s1, k in coll if s0 >= lo - 1e-6 and s1 <= hi + 1e-6}
    return [e for k, e in enumerate(lines) if k not in kill]


# ---------------------------------------------------------------- snapping to dimension values

def snap_to_dims(lines, arcs, dims_lin, tol=0.25):
    """Least-squares adjustment of x / y coordinate clusters so that recognised linear dimensions
    hold their exact values (classic 'geometry follows dimensions')."""
    report = []
    for axis in (0, 1):
        # constraints along this axis: dims whose direction is along the axis
        cons = []
        for d in dims_lin:
            if d.get("value") is None:
                continue
            ang = d["angle"] % 180
            if axis == 0 and min(ang, 180 - ang) > 0.5:
                continue
            if axis == 1 and abs(ang - 90) > 0.5:
                continue
            cons.append(d)
        if not cons:
            continue
        coords = []
        for e in lines:
            for k in ("p", "q"):
                coords.append(e[k][axis])
        for e in arcs:
            coords.append(e["c"][axis])
        vals = np.sort(np.array(coords))
        clusters = []
        for v in vals:
            if clusters and v - clusters[-1][-1] < 0.06:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        cvals = np.array([np.mean(c) for c in clusters])
        idx = {}
        rows = []
        for d in cons:
            a, b = d["p1"][axis], d["p2"][axis]
            ia, ib = int(np.argmin(np.abs(cvals - a))), int(np.argmin(np.abs(cvals - b)))
            if abs(cvals[ia] - a) > tol or abs(cvals[ib] - b) > tol or ia == ib:
                continue
            sgn = 1 if cvals[ib] > cvals[ia] else -1
            rows.append((ia, ib, sgn * d["value"]))
            idx.setdefault(ia, len(idx))
            idx.setdefault(ib, len(idx))
        if not rows:
            continue
        m = len(idx)
        A = np.zeros((len(rows) + m, m))
        y = np.zeros(len(rows) + m)
        for r, (ia, ib, v) in enumerate(rows):
            A[r, idx[ib]], A[r, idx[ia]], y[r] = 1, -1, v
        lam = 0.05
        for c, j in idx.items():
            A[len(rows) + j, j], y[len(rows) + j] = lam, lam * cvals[c]
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        new = {c: sol[j] for c, j in idx.items()}
        shift = {c: new[c] - cvals[c] for c in new}
        mx = max(abs(v) for v in shift.values())
        if mx > 0.6:
            continue
        def move(v):
            c = int(np.argmin(np.abs(cvals - v)))
            if c in shift and abs(cvals[c] - v) < 0.07:
                return v + shift[c]
            return v
        for e in lines:
            for k in ("p", "q"):
                e[k][axis] = move(e[k][axis])
        for e in arcs:
            e["c"][axis] = move(e["c"][axis])
        for d in dims_lin:
            for k in ("p1", "p2", "base"):
                d[k][axis] = move(d[k][axis])
        report.append({"axis": "xy"[axis], "constraints": len(rows), "max_shift_mm": round(float(mx), 4)})
    return report


# ---------------------------------------------------------------- leftovers

def _in_poly(pt, poly):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def drop_text_strokes(prims, texts, max_len=3.0, pad=0.3):
    """Short primitives lying inside a recognised text box are remains of glyphs."""
    boxes = []
    for t in texts:
        if t.score < 0.5:
            continue
        c = t.center
        boxes.append(c + (t.poly - c) * (1 + pad / max(t.height, 1)))
    out = []
    for e in prims:
        if length(e) <= max_len:
            pts = [e["p"], e["q"]] if e["type"] == "line" else [e["c"] + e["r"] * np.array(
                [math.cos(math.radians(a)), math.sin(math.radians(a))]) for a in (e["a0"], e["a1"])]
            if any(all(_in_poly(p, b) for p in pts) for b in boxes):
                continue
        out.append(e)
    return out


def roughness_marks(lines, tol=0.2):
    """GOST 2.309 roughness sign: two strokes from a common apex at 60°, lengths ~1:2."""
    ends = []
    for k, e in enumerate(lines):
        ends.append((k, e["p"], e["q"]))
        ends.append((k, e["q"], e["p"]))
    out = set()
    for i in range(len(ends)):
        k1, a1, b1 = ends[i]
        for j in range(i + 1, len(ends)):
            k2, a2, b2 = ends[j]
            if k1 == k2 or np.linalg.norm(a1 - a2) > tol:
                continue
            v1, v2 = b1 - a1, b2 - a2
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            s, lg = min(l1, l2), max(l1, l2)
            if not (2.0 <= s <= 12 and 1.5 <= lg / s <= 2.7):
                continue
            cosang = v1 @ v2 / (l1 * l2)
            if abs(math.degrees(math.acos(np.clip(cosang, -1, 1))) - 60) < 6:
                out |= {k1, k2}
    return out


def place_radius_dim(msp, d, iters=3):
    """Radius DIMENSION whose arrow is at the detected arc point and whose text lands on the OCR box.
    ezdxf derives the arc point from `location`, so defpoint4 is set explicitly and the location is
    corrected by the offset between the rendered and the wanted text centre."""
    from ezdxf.math import Vec3
    tip = d["c"] + d["r"] * np.array([math.cos(math.radians(d["angle"])), math.sin(math.radians(d["angle"]))])
    want = np.asarray(d["loc"], float) if d["loc"] is not None else None
    loc = want.copy() if want is not None else None
    for _ in range(iters):
        dd = msp.add_radius_dim(center=tuple(d["c"]), radius=d["r"], angle=d["angle"], text=d["text"],
                                location=tuple(loc) if loc is not None else None, dimstyle="ESKD",
                                override={"dimtih": 1, "dimtoh": 1}, dxfattribs={"layer": "DIMS"})
        dd.dimension.dxf.defpoint4 = Vec3(float(tip[0]), float(tip[1]), 0)
        dd.render()
        if want is None:
            return
        mt = [e for e in dd.dimension.get_geometry_block() if e.dxftype() == "MTEXT"]
        got = np.array([mt[0].dxf.insert.x, mt[0].dxf.insert.y]) if mt else want
        err = want - got
        if np.linalg.norm(err) < 0.3:
            return
        msp.delete_entity(dd.dimension)
        loc = loc + err


# ---------------------------------------------------------------- title block

def title_fields(texts, frame):
    """GOST 2.104 form 1 cells relative to the lower-right corner of the frame."""
    if frame is None:
        return {}
    xr, yb = frame[2], frame[1]
    cells = {"designation": (xr - 120, yb + 40, xr, yb + 55), "name": (xr - 120, yb + 15, xr - 50, yb + 40),
             "material": (xr - 120, yb, xr - 50, yb + 15), "litera": (xr - 50, yb + 20, xr - 35, yb + 35),
             "mass": (xr - 35, yb + 20, xr - 18, yb + 35), "scale": (xr - 18, yb + 20, xr, yb + 35),
             "developer": (xr - 168, yb + 25, xr - 145, yb + 30), "checker": (xr - 168, yb + 20, xr - 145, yb + 25),
             "format": (xr - 50, yb - 5, xr, yb)}
    out = {}
    for k, (x0, y0, x1, y1) in cells.items():
        got = [t for t in texts if x0 <= t.center[0] <= x1 and y0 <= t.center[1] <= y1]
        got.sort(key=lambda t: (-t.center[1], t.center[0]))
        if got:
            out[k] = " ".join(t.text for t in got)
    return out


def detect_frame(lines, sheet_w, sheet_h):
    """Inner frame = longest horizontal/vertical lines near the GOST 2.301 margins."""
    hs = [e for e in lines if e["type"] == "line" and abs(e["p"][1] - e["q"][1]) < 0.2 and length(e) > 0.4 * sheet_w]
    vs = [e for e in lines if e["type"] == "line" and abs(e["p"][0] - e["q"][0]) < 0.2 and length(e) > 0.4 * sheet_h]
    if not hs or not vs:
        return None
    xs = sorted(e["p"][0] for e in vs)
    ys = sorted(e["p"][1] for e in hs)
    inner_x = [x for x in xs if 5 < x < sheet_w - 2]
    inner_y = [y for y in ys if 2 < y < sheet_h - 2]
    if len(inner_x) < 2 or len(inner_y) < 2:
        return None
    return (min(inner_x), min(inner_y), max(inner_x), max(inner_y))


# ---------------------------------------------------------------- main

def run(img_path, out_path, sheet=None, report_path=None):
    T = Timer()
    gray = load_gray(img_path)
    bw = binarize(gray)
    skew = estimate_skew(bw)
    if abs(skew) > 0.02:
        gray = deskew(gray, skew)
        bw = binarize(gray)
    cal = calibrate(bw, sheet)
    T("load+binarize+deskew+calibrate")

    texts = run_ocr(gray, cal)
    fmt = None
    for t in texts:
        m = re.search(r"Формат\s*[AА]\s*([0-4])", t.text.replace(" ", " "))
        if m:
            fmt = "A" + m.group(1)
    if fmt and fmt in SHEETS and fmt != cal["sheet"] and sheet is None:
        cal = calibrate(bw, fmt)
        texts = run_ocr(gray, cal)
    T("ocr")

    stroke = stroke_width_mm(bw, cal)
    # filled symbols must survive an opening wider than a line stroke (0.42 mm on clean 600 dpi renders)
    arrows, blobs, sym_erase = symbols.detect(bw, cal, d_open_mm=max(0.42, 1.6 * stroke))
    T("symbols")

    txt_erase = text_erase_mask(bw, cal, texts)
    ink = (bw & (1 - txt_erase) & (1 - sym_erase)).astype(np.uint8)
    prims, _ = vectorize(ink, cal)
    T("vectorize")

    prims = as_dicts(prims)
    prims = [e for e in prims if length(e) >= 0.35]
    lines = [e for e in prims if e["type"] == "line"]
    arcs = [e for e in prims if e["type"] == "arc"]
    lines = snap_angles(merge_collinear(lines))
    chains, lines = dashed_chains(lines, coverage=lambda a, b: dimsmod.ink_cover(bw, cal, to_px, a, b, step=0.1))
    axes = [c for c in chains if c["kind"] == "AXIS"]
    T("postprocess")

    regions, rest_idx = hatchmod.find_families(lines)
    hatch_ids = {i for r in regions for i in r["lines"]}
    nohatch = ink.copy()
    ppm = cal["px_per_mm_x"]
    for i in hatch_ids:
        a = np.array(to_px(cal, *lines[i]["p"]))
        b = np.array(to_px(cal, *lines[i]["q"]))
        cv2.line(nohatch, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 0, max(3, int(0.4 * ppm)))
    hatches = []
    for r in regions:
        polys, leaked = hatchmod.region_boundary(r, lines, nohatch, cal, to_px, to_mm)
        if polys:
            hatches.append({"region": r, "polys": polys, "leaked": leaked})
    lines = [e for i, e in enumerate(lines) if i not in hatch_ids]
    T("hatch")

    dims_lin, used_ext, used_texts = dimsmod.build(arrows, lines, texts, axes, bw, cal, to_px)
    # remove dimension lines and the used parts of extension lines
    for d in dims_lin:
        a_, b_ = arrows[d["arrows"][0]], arrows[d["arrows"][1]]
        lines = remove_collinear_run(lines, a_.tip, b_.tip)
    new_lines = []
    ext_cut = []
    for d in dims_lin:
        for e in d["ext"]:
            if e:
                ext_cut.append((e[1], e[0]))
    for e in lines:
        pieces = [e]
        for over, defp in ext_cut:
            u = defp - over
            if np.linalg.norm(u) < 0.5:
                continue
            u = u / np.linalg.norm(u)
            nn = np.array([-u[1], u[0]])
            nxt = []
            for pc in pieces:
                if abs((pc["p"] - over) @ nn) < 0.2 and abs((pc["q"] - over) @ nn) < 0.2:
                    nxt += cut_interval(pc, over, defp)
                else:
                    nxt.append(pc)
            pieces = nxt
        new_lines += pieces
    lines = new_lines
    from ocr import ocr_window
    rad = dimsmod.radius_dims(arrows, arcs, texts, used_texts, ocr_window=lambda box: ocr_window(gray, cal, box))
    angd = dimsmod.angular_dims(arrows, arcs, texts, used_texts)
    used_arc_ids = {id(e) for e in arcs for d in angd if np.allclose(e["c"], d["c"]) and abs(e["r"] - d["r"]) < 1e-6}
    arcs = [e for e in arcs if id(e) not in used_arc_ids]
    used_l = {}
    leads = dimsmod.leaders(arrows, lines, used_l)
    for d in rad:
        a_ = arrows[d["arrow"]]
        leads_r = dimsmod.leaders([a_.__class__(a_.tip, a_.dir, a_.length, a_.width)], lines, used_l)
    lines = [e for k, e in enumerate(lines) if k not in used_l]
    T("dimensions")

    snap_rep = snap_to_dims(lines, arcs, dims_lin)
    T("snap")

    lines = drop_text_strokes(lines, texts)
    arcs = drop_text_strokes(arcs, texts)
    rough_ids = roughness_marks(lines)
    T("cleanup")

    frame = detect_frame(lines, cal["sheet_w"], cal["sheet_h"])
    fields = title_fields(texts, frame)

    # ------------------------------------------------------------ write DXF
    doc = new_drawing(font="isocpeur.ttf", oblique=15.0, dim_text_height=5.0, arrow=3.0)
    msp = doc.modelspace()
    for k, e in enumerate(lines):
        L = length(e)
        if L < 0.5:
            continue
        layer = "DIMS" if k in rough_ids else ("MAIN" if L >= 1.5 else "THIN")
        msp.add_line(tuple(e["p"]), tuple(e["q"]), dxfattribs={"layer": layer})
    for e in arcs:
        if e["sweep"] >= 359:
            msp.add_circle(tuple(e["c"]), e["r"], dxfattribs={"layer": "MAIN"})
        else:
            msp.add_arc(tuple(e["c"]), e["r"], e["a0"], e["a1"], dxfattribs={"layer": "MAIN"})
    for c in chains:
        att = {"layer": "AXIS"} if c["kind"] == "AXIS" else {"layer": "THIN", "linetype": "DASHED"}
        msp.add_line(tuple(c["p"]), tuple(c["q"]), dxfattribs=att)
    for h in hatches:
        r = h["region"]
        ent = msp.add_hatch(dxfattribs={"layer": "THIN"})
        ent.set_pattern_fill("ANSI31", scale=r["spacing"] / 3.175, angle=r["angle"] - 45.0)
        from ezdxf.math import Vec2
        ph = r["phase_off"] * r["normal"]
        ent.pattern.lines[0].base_point = Vec2(float(ph[0]), float(ph[1]))
        for poly, is_hole in h["polys"]:
            ent.paths.add_polyline_path([tuple(p) for p in poly], is_closed=True)
    for b in blobs:
        ent = msp.add_hatch(dxfattribs={"layer": "DIMS"})
        ent.set_solid_fill(color=256)
        ent.paths.add_polyline_path([tuple(p) for p in b.poly], is_closed=True)
    for d in dims_lin:
        ang_d = (d["angle"] - 90.001) % 180 - 89.999
        dd = msp.add_linear_dim(base=tuple(d["base"]), p1=tuple(d["p1"]), p2=tuple(d["p2"]), angle=ang_d,
                                text=d["text"], location=tuple(d["loc"]) if d["loc"] is not None else None,
                                dimstyle="ESKD", override={"dimtix": 0, "dimtofl": 1, "dimlfac": 1.0 / d.get("scale", 1.0)},
                                dxfattribs={"layer": "DIMS"})
        dd.render()
    for d in rad:
        place_radius_dim(msp, d)
    for d in angd:
        a1 = math.degrees(math.atan2(*(d["t1"] - d["c"])[::-1]))
        a2 = math.degrees(math.atan2(*(d["t2"] - d["c"])[::-1]))
        if (a2 - a1) % 360 > 180:
            a1, a2 = a2, a1
        dd = msp.add_angular_dim_cra(center=tuple(d["c"]), radius=d["r"], start_angle=a1, end_angle=a2, distance=0,
                                     text=d["text"], location=tuple(d["loc"]) if d["loc"] is not None else None,
                                     dimstyle="ESKD", dxfattribs={"layer": "DIMS"})
        dd.render()
    for ld in leads:
        msp.add_leader([tuple(map(float, p)) for p in ld["pts"]], dimstyle="ESKD", dxfattribs={"layer": "DIMS"})
    for k, t in enumerate(texts):
        if k in used_texts or t.score < 0.45:
            continue
        s = re.sub(r"^/(?=\d)", "", t.text).replace(DIAM, "%%c").replace("±", "%%p").replace("°", "%%d")
        h_est = t.glyph_h if getattr(t, "glyph_h", None) else t.height * 0.55
        h = min(GOST_H, key=lambda v: abs(math.log(v / max(h_est, 0.5))))
        up = (t.poly[0] - t.poly[3]) / (np.linalg.norm(t.poly[0] - t.poly[3]) + 1e-9)
        p0 = t.poly[3] + up * max(0.0, (t.height - h) / 2)
        ent = msp.add_text(s, height=h, rotation=t.angle,
                           dxfattribs={"layer": "TEXT", "style": "ESKD"})
        ent.set_placement(tuple(map(float, p0)))
    for k, v in fields.items():
        doc.header.custom_vars.append(k, v)
    doc.header.custom_vars.append("source", Path(img_path).name)
    doc.header.custom_vars.append("project", "PNG->DXF, joint work with 44labs")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_path)
    T("write")
    problems = verify(str(out_path))
    rep = {"input": str(img_path), "output": str(out_path), "calibration": cal, "skew_deg": skew, "stroke_mm": stroke,
           "format_from_ocr": fmt, "stages_s": T.stages, "total_s": round(time.time() - T.t0, 2),
           "counts": {"texts": len(texts), "arrows": len(arrows), "blobs": len(blobs), "lines": len(lines),
                      "arcs": len(arcs), "axis_or_dashed": len(chains), "hatches": len(hatches),
                      "dims_linear": len(dims_lin), "dims_radius": len(rad), "dims_angular": len(angd),
                      "leaders": len(leads)},
           "dims_with_value": sum(1 for d in dims_lin if d["value"] is not None),
           "snap": snap_rep, "title_fields": fields, "verify": problems or "OK"}
    if report_path:
        json.dump(rep, open(report_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    rep = run(a.image, a.out, a.sheet, a.report)
    print(json.dumps({k: rep[k] for k in ("total_s", "stages_s", "counts", "verify")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
