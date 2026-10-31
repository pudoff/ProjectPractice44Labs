"""Measure the CV module on sheets with a reference DXF.

    python mlcv/eval/evaluate.py [--set default|quick|random] [--n 8 --seed 44] [--out DIR] [--only NAME ...]

For every input (PNG / JPEG / PDF of a real sheet, the synthetic control sheet and its simulated scan) the module
runs as the backend would call it, then its exported primitives are compared with the reference DXF:
  found / missed reference primitives (by kind), extra / duplicate / short detected primitives, coordinate errors
  (circle centre and radius, line ends, curve deviation), closed contours against contours assembled from the
  reference geometry, gaps (on real sheets every gap is false: the drawings are complete), stage times.
Registration with the reference: a translation only (phase correlation + ICP); the residual scale is reported,
never applied. The calibration method of the module and any manual calibration are listed per row.
Writes OUT/<case>/ (module output + compare.png + metrics.json), OUT/results.json and OUT/table.md."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from ezdxf.render import hatching
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mlcv"))
sys.path.insert(0, str(ROOT / "tools"))

from compare_dxf import entity_class, iter_flat  # noqa: E402
from drawcv import Params, run  # noqa: E402
from drawcv.primitives import sample as prim_sample  # noqa: E402
from drawcv.synth import make_all, write_scaled_pdf  # noqa: E402
from drawcv.topology import build as build_topology  # noqa: E402

TOL = 0.5
TOL_FINE = 0.25
COVER = 0.8
PNG = ROOT / "dataset/clear_sum_dataset/vse_png"
DXF = ROOT / "dataset/clear_sum_dataset/vse_dxf"
PDF = ROOT / "dataset/dataset2"

# The test sheet (клапан) is never used here: its reference may only be read by tools/compare_dxf.py.
SHEETS = {
    "default": [
        ("Шайба", ["png", "jpeg", "pdf"]),
        ("Сфера1", ["png", "pdf"]),
        ("Опора", ["png", "pdf"]),
        ("Фланец", ["png", "pdf"]),
        ("прокладка", ["png", "pdf"]),
        ("плита", ["png"]),
    ],
    "quick": [("Шайба", ["png", "jpeg", "pdf"])],
    "random": None,                     # filled by random_sheets(): sheets never used while developing
}
DEV_SHEETS = {"клапан", "Шайба", "Сфера1", "Опора", "Фланец", "прокладка", "плита"}


def random_sheets(n, seed):
    """n random real sheets (fixed seed => repeatable) not used in development; multi-sheet references skipped."""
    import random

    from ezdxf import bbox

    names = sorted(p.stem for p in PNG.glob("*.png") if p.stem not in DEV_SHEETS and (DXF / f"{p.stem}.dxf").exists())
    random.Random(seed).shuffle(names)
    out = []
    for name in names:
        try:
            doc = ezdxf.readfile(str(DXF / f"{name}.dxf"))
            ext = bbox.extents(e for e in iter_flat(doc.modelspace()) if e.dxftype() in ("LINE", "ARC", "CIRCLE", "LWPOLYLINE"))
        except Exception:
            continue
        if not ext.has_data or ext.size.x > 1250 or ext.size.y > 900:
            continue
        out.append((name, ["png", "pdf"] if (PDF / f"{name}.pdf").exists() else ["png"]))
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------ inputs

def prepare_inputs(set_name, work, n=8, seed=44):
    cases = []
    sheets = random_sheets(n, seed) if set_name == "random" else SHEETS[set_name]
    for name, fmts in sheets:
        for fmt in fmts:
            if fmt == "png":
                src = PNG / f"{name}.png"
            elif fmt == "pdf":
                src = PDF / f"{name}.pdf"
            else:
                src = work / "inputs" / f"{name}.jpg"
                if not src.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    img = cv2.imdecode(np.fromfile(str(PNG / f"{name}.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
                    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(src))
            cases.append({"case": f"{name}.{fmt}", "sheet": name, "format": fmt, "input": src,
                          "reference": DXF / f"{name}.dxf",
                          "note": "JPEG q90 re-encoded from the dataset PNG" if fmt == "jpeg" else ""})
    if set_name == "random":
        return cases
    synth = make_all(work / "synth")
    img = cv2.imdecode(np.fromfile(str(synth / "control.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    write_scaled_pdf(img, 300, synth / "control_print97.pdf", 0.97)
    write_scaled_pdf(img, 300, synth / "control_print_aniso.pdf", 0.99, 1.0)
    write_scaled_pdf(img, 300, synth / "control_print_aniso3.pdf", 0.97, 1.0)
    for fname, fmt, note in (("control.png", "png", "synthetic, exact truth"),
                             ("control.jpg", "jpeg", "synthetic, JPEG q90"),
                             ("control.pdf", "pdf", "synthetic, raster PDF"),
                             ("control_print97.pdf", "pdf", "synthetic PDF printed at 97 %"),
                             ("control_print_aniso.pdf", "pdf", "synthetic PDF printed 99 % x 100 % (anisotropic)"),
                             ("control_print_aniso3.pdf", "pdf", "synthetic PDF printed 97 % x 100 % (must be unknown)"),
                             ("control_scan.jpg", "jpeg", "simulated scan: 1.2 deg, blur, noise, shading, JPEG q75")):
        cases.append({"case": f"control.{Path(fname).stem.replace('control', '').strip('_') or fmt}.{fmt}",
                      "sheet": "control", "format": fmt, "input": synth / fname, "reference": synth / "control.dxf",
                      "truth": synth / "truth.json", "note": note})
    return cases


# ------------------------------------------------------------------ reference

def sample(e, step=0.25):
    """Points along a primitive; reference splines/ellipses ("curve") are resampled polylines."""
    if e["kind"] != "curve":
        return prim_sample(e, step)
    pts = e["pts"]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, seg[-1], max(2, int(np.ceil(seg[-1] / step)) + 1))
    return np.column_stack([np.interp(t, seg, pts[:, 0]), np.interp(t, seg, pts[:, 1])])


def _prim(kind, **kw):
    d = {"kind": kind}
    for k, v in kw.items():
        d[k] = np.asarray(v, float)[:2] if k in ("p", "q", "c") else v
    return d


def reference(path):
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    geo, annot, hatch = [], [], []
    texts = []

    def add(e, cls, into):
        t = e.dxftype()
        if t == "LINE":
            p, q = np.array(e.dxf.start)[:2], np.array(e.dxf.end)[:2]
            if np.linalg.norm(q - p) > 1e-6:
                into.append(dict(_prim("line", p=p, q=q), cls=cls))
        elif t == "CIRCLE":
            into.append(dict(_prim("circle", c=e.dxf.center, r=float(e.dxf.radius)), cls=cls))
        elif t == "ARC":
            into.append(dict(_prim("arc", c=e.dxf.center, r=float(e.dxf.radius), a0=float(e.dxf.start_angle) % 360,
                                   a1=float(e.dxf.end_angle) % 360), cls=cls))
        elif t in ("LWPOLYLINE", "POLYLINE"):
            for v in e.virtual_entities():
                add(v, cls, into)
        elif t in ("ELLIPSE", "SPLINE"):
            # one primitive (break lines are splines): flattened only for sampling
            pts = np.array([(v.x, v.y) for v in e.flattening(0.05)])
            if len(pts) >= 2:
                into.append({"kind": "curve", "pts": pts, "cls": cls, "source": t})

    for e in iter_flat(msp):
        t = e.dxftype()
        if t == "DIMENSION" or t == "LEADER":
            try:
                for v in e.virtual_entities():
                    if v.dxftype() in ("LINE", "ARC", "LWPOLYLINE", "POLYLINE"):
                        add(v, "DIMS", annot)
            except Exception:
                pass
        elif t == "HATCH":
            try:
                for a, b in hatching.hatch_entity(e):
                    hatch.append(_prim("line", p=np.array(a)[:2], q=np.array(b)[:2]))
            except Exception:
                pass
        elif t in ("TEXT", "MTEXT"):
            try:
                from ezdxf import bbox
                ext = bbox.extents([e])
                if ext.has_data:
                    texts.append((ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y))
            except Exception:
                pass
        else:
            n0 = len(geo)
            add(e, entity_class(e, doc), geo)
            if e.dxf.get("layer", "0") == "0":
                for g in geo[n0:]:
                    g["layer0"] = True
    promoted = False
    if not any(g["cls"] == "MAIN" for g in geo):
        # some archive DXFs keep the contour on layer "0" with the default weight and leave "Wide" empty
        for g in geo:
            if g.get("layer0"):
                g["cls"] = "MAIN"
                promoted = True
    return geo, annot, hatch, texts, promoted


# ------------------------------------------------------------------ registration

def samples_of(prims, step=0.25):
    pts, owner = [], []
    for k, e in enumerate(prims):
        s = sample(e, step)
        pts.append(s)
        owner += [k] * len(s)
    return (np.vstack(pts) if pts else np.zeros((0, 2))), np.array(owner, int)


def register(ref_pts, det_pts):
    """Translation t with det + t ~ ref: phase correlation on 0.5 mm rasters, then ICP on near pairs."""
    res = 0.5
    r0, d0 = ref_pts.min(axis=0), det_pts.min(axis=0)
    size = np.ceil(np.maximum(ref_pts.max(axis=0) - r0, det_pts.max(axis=0) - d0) / res).astype(int) + 8
    size = np.minimum(size, 6000)

    def raster(pts, origin):
        img = np.zeros((size[1], size[0]), np.float32)
        ij = np.clip(((pts - origin) / res).astype(int), 0, size - 1)
        img[ij[:, 1], ij[:, 0]] = 1
        return cv2.GaussianBlur(img, (0, 0), 1.0)

    win = cv2.createHanningWindow((size[0], size[1]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(raster(det_pts, d0), raster(ref_pts, r0), win)
    t = r0 - d0 + np.array([dx, dy]) * -res
    tree = cKDTree(ref_pts)
    best = None
    for cand in (t, r0 - d0 + np.array([dx, dy]) * res, r0 - d0):
        d, _ = tree.query(det_pts + cand)
        score = float(np.mean(d < 1.0))
        if best is None or score > best[0]:
            best = (score, cand)
    t = best[1]
    for radius in (3.0, 1.0, 0.5, 0.5):
        d, idx = tree.query(det_pts + t)
        sel = d < radius
        if sel.sum() < 20:
            break
        t = t + np.median(ref_pts[idx[sel]] - (det_pts[sel] + t), axis=0)
    d, idx = tree.query(det_pts + t)
    sel = d < 0.5
    scale = None
    if sel.sum() > 50:
        a = det_pts[sel] + t
        b = ref_pts[idx[sel]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        scale = float(np.sqrt(((b - cb) ** 2).sum() / ((a - ca) ** 2).sum()))
    return t, scale, float(np.mean(d < 0.5))


# ------------------------------------------------------------------ matching

def coverage(prims, tree, tol):
    out = []
    for e in prims:
        s = sample(e, 0.25)
        d, _ = tree.query(s)
        out.append(float(np.mean(d <= tol)))
    return out


def shift(e, t):
    e = dict(e)
    for k in ("p", "q", "c"):
        if k in e:
            e[k] = np.asarray(e[k], float) + t
    if "pts" in e:
        e["pts"] = e["pts"] + t
    return e


def detected(cv, t):
    """Exported primitives of cv.json moved into reference coordinates."""
    out = []
    for item in cv["primitives"]:
        k = item["kind"]
        if k == "line":
            e = _prim("line", p=item["start"], q=item["end"])
        elif k == "circle":
            e = _prim("circle", c=item["center"], r=item["radius"])
        else:
            e = _prim("arc", c=item["center"], r=item["radius"], a0=item["start_angle"], a1=item["end_angle"])
        e.update(id=item["id"], layer=item["layer"], role=cv["objects"][item["id"]]["role"])
        out.append(shift(e, t))
    # hatch lines turned into a HATCH entity are module output too (references often draw hatching as LINEs)
    for oid, o in cv["objects"].items():
        if o["status"] == "converted":
            g = o["geometry_mm"]
            e = _prim("line", p=g["start"], q=g["end"])
            e.update(id=oid, layer="THIN", role="hatch")
            out.append(shift(e, t))
    return out


def length_of(e):
    if e["kind"] == "curve":
        return float(np.sum(np.linalg.norm(np.diff(e["pts"], axis=0), axis=1)))
    if e["kind"] == "line":
        return float(np.linalg.norm(e["q"] - e["p"]))
    sw = 360.0 if e["kind"] == "circle" else ((e["a1"] - e["a0"]) % 360 or 360.0)
    return math.radians(sw) * e["r"]


def in_sheet_body(e, frame_box, title_box):
    """Reference primitive belongs to the drawing body: inside the frame and outside the title block."""
    s = sample(e, 1.0)
    x0, y0, x1, y1 = frame_box
    inside = (s[:, 0] > x0 + 1) & (s[:, 0] < x1 - 1) & (s[:, 1] > y0 + 1) & (s[:, 1] < y1 - 1)
    if not inside.all():
        return False
    tx0, ty0, tx1, ty1 = title_box
    in_tb = (s[:, 0] >= tx0 - 1) & (s[:, 0] <= tx1 + 1) & (s[:, 1] >= ty0 - 1) & (s[:, 1] <= ty1 + 1)
    return not in_tb.all()


def evaluate_case(case, out_root):
    out = out_root / case["case"]
    t0 = time.perf_counter()
    cv = run(case["input"], out, Params())
    wall = time.perf_counter() - t0
    geo, annot, hatch_ref, texts, promoted = reference(case["reference"])
    ref_pts, ref_owner = samples_of(geo)
    det0 = detected(cv, np.zeros(2))
    det_pts0, _ = samples_of(det0)
    t, scale, overlap = register(ref_pts, det_pts0)
    det = detected(cv, t)
    det_pts, det_owner = samples_of(det)

    W, H = cv["calibration"]["sheet_mm"]
    frame_box = (20 + t[0], 5 + t[1], W - 5 + t[0], H - 5 + t[1])
    title_box = (W - 5 - 185 + t[0], 5 + t[1], W - 5 + t[0], 60 + t[1])
    body = [in_sheet_body(e, frame_box, title_box) for e in geo]

    det_tree = cKDTree(det_pts) if len(det_pts) else None
    by_kind = {}
    rows = []
    for e, b in zip(geo, body):
        kind = e["kind"]
        same = [d for d in det if d["kind"] == kind or (kind == "arc" and d["kind"] == "circle") or kind == "curve"]
        sp, _ = samples_of(same)
        cov_same = coverage([e], cKDTree(sp), TOL)[0] if len(sp) else 0.0
        cov_any = coverage([e], det_tree, TOL)[0] if det_tree is not None else 0.0
        cov_fine = coverage([e], det_tree, TOL_FINE)[0] if det_tree is not None else 0.0
        row = {"kind": kind, "cls": e["cls"], "body": b, "found": cov_same >= COVER, "covered": cov_any >= COVER,
               "covered_fine": cov_fine >= COVER, "length": length_of(e)}
        if row["found"] and kind in ("circle", "arc"):
            cands = [d for d in same if np.linalg.norm(d["c"] - e["c"]) < max(1.0, 0.1 * e["r"]) and abs(d["r"] - e["r"]) < max(0.5, 0.05 * e["r"])]
            if cands:
                d = min(cands, key=lambda d: np.linalg.norm(d["c"] - e["c"]) + abs(d["r"] - e["r"]))
                row.update(center_err=float(np.linalg.norm(d["c"] - e["c"])), radius_err=float(d["r"] - e["r"]))
        if row["found"] and kind == "line":
            best = None
            for d in same:
                if d["kind"] != "line":
                    continue
                c1 = coverage([e], cKDTree(sample(d, 0.25)), TOL)[0]
                # ends are comparable only for a detected line of about the same length (not a merged longer one)
                if c1 >= 0.9 and abs(length_of(d) - length_of(e)) <= 2.0 and (best is None or c1 > best[0]):
                    best = (c1, d)
            if best:
                d = best[1]
                ends = [min(np.linalg.norm(pe - d["p"]), np.linalg.norm(pe - d["q"])) for pe in (e["p"], e["q"])]
                row["end_err"] = float(max(ends))
        if row["covered"]:
            dd, _ = det_tree.query(sample(e, 0.25))
            row["dev"] = float(np.median(dd))
        rows.append(row)
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += row["found"]
        by_kind[kind][1] += 1

    # detected side
    ref_tree = cKDTree(ref_pts) if len(ref_pts) else None
    ann_pts, _ = samples_of(annot)
    hat_pts, _ = samples_of(hatch_ref)
    ann_tree = cKDTree(ann_pts) if len(ann_pts) else None
    hat_tree = cKDTree(hat_pts) if len(hat_pts) else None
    det_rows = []
    for k, d in enumerate(det):
        s = sample(d, 0.25)
        cat = "extra"
        if ref_tree is not None and np.mean(ref_tree.query(s)[0] <= TOL) >= COVER:
            cat = "geometry"
        elif ann_tree is not None and np.mean(ann_tree.query(s)[0] <= TOL) >= COVER:
            cat = "dimension_graphics"
        elif (hat_tree is not None and np.mean(hat_tree.query(s)[0] <= TOL) >= COVER) or d["role"] == "hatch":
            cat = "hatch_pattern"          # a hatch line need not coincide with the reference pattern phase
        elif any(np.mean((s[:, 0] >= x0 - 1) & (s[:, 0] <= x1 + 1) & (s[:, 1] >= y0 - 1) & (s[:, 1] <= y1 + 1)) >= COVER
                 for x0, y0, x1, y1 in texts):
            cat = "text_strokes"
        dup = False
        if length_of(d) > 0:
            longer = [j for j in range(len(det)) if j != k and length_of(det[j]) >= length_of(d)]
            if longer:
                lp, _ = samples_of([det[j] for j in longer])
                dup = bool(np.mean(cKDTree(lp).query(s)[0] <= 0.25) >= 0.8)
        row = {"id": d["id"], "kind": d["kind"], "role": d["role"], "category": cat, "duplicate": dup,
               "short": length_of(d) < 2.0, "length": length_of(d)}
        if cat == "geometry":
            # line class of the reference primitive under the middle of the detected one vs the module's layer
            mid = s[len(s) // 2]
            row["ref_class"] = geo[ref_owner[ref_tree.query(mid)[1]]]["cls"]
            row["our_class"] = "HIDDEN" if d["role"] == "hidden" else d["layer"]
        det_rows.append(row)

    contours = contour_metrics(cv, geo, body, t)
    defects = cv["summary"]["defects"]
    truth = check_truth(cv, case) if case.get("truth") else None
    contours["reference_main_from_layer0"] = promoted
    m = summarise(case, cv, rows, det_rows, by_kind, contours, defects, t, scale, overlap, wall, truth)
    compare_image(geo, annot, det, det_rows, out / "compare.png", t)
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def contour_metrics(cv, geo, body, t):
    """Closed contours assembled from the reference MAIN geometry of the drawing body vs closed contours of cv."""
    part = []
    for k, (e, b) in enumerate(zip(geo, body)):
        if b and e["cls"] == "MAIN" and e["kind"] != "curve":
            part.append(dict(e, id=f"ref_{k}", role="part"))
    gt = build_topology(part, [], 0.05, 0.0)["contours"] if part else []
    gt_closed = [c for c in gt if c["closed"] and c["role"] in ("outer_boundary", "circle", "region")]
    det_closed = []
    for c in cv["contours"]:
        if c["closed"]:
            pts = contour_points(c) + t
            if len(pts) >= 3:
                poly = Polygon(pts)
                det_closed.append(poly if poly.is_valid else poly.buffer(0))
    found = 0
    for g in gt_closed:
        gp = g["_poly"]
        if any(_iou(gp, d) >= 0.9 for d in det_closed):
            found += 1
    return {"reference_closed": len(gt_closed), "found": found, "detected_closed": len(det_closed),
            "detected_open": sum(not c["closed"] for c in cv["contours"])}


def contour_points(c):
    pts = []
    for e in c["edges"]:
        if e["kind"] == "line":
            pts += [e["start"], e["end"]]
        else:
            a0, a1 = e["start_angle"], e["end_angle"]
            sw = (a1 - a0) % 360 or 360.0
            angs = np.radians(a0 + np.linspace(0, sw, max(3, int(sw / 3))))
            arc = np.column_stack([np.cos(angs), np.sin(angs)]) * e["radius"] + e["center"]
            pts += (arc[::-1] if e["reversed"] else arc).tolist()
    return np.array(pts, float)


def _iou(a, b):
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def check_truth(cv, case):
    """Synthetic sheet: expected closed profiles, the deliberate gap and self-intersection."""
    truth = json.loads(Path(case["truth"]).read_text(encoding="utf-8"))
    res = {}
    closed = [c for c in cv["contours"] if c["closed"]]
    for prof in truth["closed_profiles"]:
        x0, y0, x1, y1 = prof["bbox"]
        hit = [c for c in closed if abs(c["bbox_mm"][0] - x0) < 1.5 and abs(c["bbox_mm"][1] - y0) < 1.5
               and abs(c["bbox_mm"][2] - x1) < 1.5 and abs(c["bbox_mm"][3] - y1) < 1.5]
        res[f"{prof['name']}_closed"] = bool(hit)
        if hit:
            res[f"{prof['name']}_area_err_mm2"] = round(min(abs(c["area_mm2"] - prof["area_mm2"]) for c in hit), 3)
    for gap in truth["gaps"]:
        a, b = np.array(gap["between"])
        hits = [d for d in cv["defects"] if d["code"] == "gap"
                and min(np.linalg.norm(np.array(d["at_mm"]) - a), np.linalg.norm(np.array(d["at_mm"]) - b)) < 2.0]
        res[f"{gap['name']}_gap_found"] = bool(hits)
        if hits:
            res[f"{gap['name']}_gap_mm"] = hits[0]["distance_mm"]
    for si in truth["self_intersections"]:
        hits = [d for d in cv["defects"] if d["code"] == "self_intersection"
                and np.linalg.norm(np.array(d["at_mm"]) - si["at"]) < 2.0]
        res[f"{si['name']}_self_intersection_found"] = bool(hits)
    circles = [o for o in cv["objects"].values() if o["kind"] == "circle" and o["status"] == "ok" and o["role"] == "part"]
    errs = []
    for c in truth["circles"]:
        best = min((np.linalg.norm(np.array(o["geometry_mm"]["center"]) - c["center"]) for o in circles), default=None)
        errs.append(best)
    res["circle_center_err_mm"] = [None if e is None else round(float(e), 4) for e in errs]
    hatch = cv["hatches"]
    res["hatch_resolved"] = bool(hatch)
    res["axes_in_contours"] = any(cv["objects"][oid]["role"] == "axis" for c in cv["contours"] for oid in c["object_ids"])
    return res


def summarise(case, cv, rows, det_rows, by_kind, contours, defects, t, scale, overlap, wall, truth):
    body = [r for r in rows if r["body"]]

    def stat(vals):
        vals = [v for v in vals if v is not None]
        return {"n": len(vals), "mean": round(float(np.mean(vals)), 4) if vals else None,
                "p95": round(float(np.percentile(vals, 95)), 4) if vals else None,
                "max": round(float(np.max(np.abs(vals))), 4) if vals else None}

    cats = {}
    for d in det_rows:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    summary_cv = cv["summary"]
    excluded = {k: v for k, v in summary_cv["by_kind_role_status"].items() if not k.endswith(":ok")}
    return {
        "case": case["case"], "sheet": case["sheet"], "format": case["format"], "note": case.get("note", ""),
        "calibration": {"method": cv["calibration"]["method"], "manual": cv["calibration"]["manual"],
                        "scale_status": cv["scale_status"], "px_per_mm": cv["calibration"]["px_per_mm"],
                        "xy_agreement": cv["calibration"].get("xy_agreement"),
                        "pdf_page_check": cv["calibration"].get("pdf_page_check")},
        "registration": {"translation_mm": [round(float(v), 3) for v in t], "residual_scale": scale,
                         "share_within_0.5mm": round(overlap, 4)},
        "reference": {"primitives": len(rows), "body_primitives": len(body)},
        "found": {"all": sum(r["found"] for r in rows), "body": sum(r["found"] for r in body),
                  "body_covered_any_kind": sum(r["covered"] for r in body),
                  "body_covered_0.25mm": sum(r["covered_fine"] for r in body),
                  "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())}},
        "missed_body": [{"kind": r["kind"], "cls": r["cls"], "length": round(r["length"], 2)} for r in body if not r["found"]][:40],
        "layers": layer_metrics(det_rows),
        "detected": {"exported": len(det_rows), "categories": cats,
                     "duplicates": sum(d["duplicate"] for d in det_rows),
                     "short_lt_2mm": sum(d["short"] for d in det_rows),
                     "extra_short_lt_2mm": sum(d["short"] and d["category"] == "extra" for d in det_rows),
                     "excluded_by_module": excluded},
        "errors_mm": {"circle_center": stat([r.get("center_err") for r in rows if r["kind"] == "circle"]),
                      "circle_radius": stat([r.get("radius_err") for r in rows if r["kind"] == "circle"]),
                      "arc_center": stat([r.get("center_err") for r in rows if r["kind"] == "arc"]),
                      "arc_radius": stat([r.get("radius_err") for r in rows if r["kind"] == "arc"]),
                      "line_end": stat([r.get("end_err") for r in rows if r["kind"] == "line"]),
                      "curve_deviation_median": stat([r.get("dev") for r in rows])},
        "contours": contours,
        "defects": defects,
        "truth": truth,
        "timings_s": cv["timings_s"],
        "wall_s": round(wall, 2),
    }


def layer_metrics(det_rows):
    """Agreement of the module's layer (MAIN / THIN / AXIS, hidden = THIN + DASHED) with the class of the matched
    reference line. Reference classes come from layer names / linetypes (tools/compare_dxf.py: entity_class)."""
    pairs = [(d["ref_class"], d["our_class"]) for d in det_rows if "ref_class" in d]
    pairs = [(r, o) for r, o in pairs if r in ("MAIN", "THIN", "AXIS", "HIDDEN")]
    conf = {}
    for r, o in pairs:
        conf[f"{r}->{o}"] = conf.get(f"{r}->{o}", 0) + 1
    ok = sum(r == o for r, o in pairs)
    return {"n": len(pairs), "accuracy": round(ok / len(pairs), 4) if pairs else None, "confusion": dict(sorted(conf.items()))}


def compare_image(geo, annot, det, det_rows, path, t, px_per_mm=3.0):
    pts = [sample(e, 0.5) for e in geo + det]
    allp = np.vstack(pts) if pts else np.zeros((1, 2))
    lo, hi = allp.min(axis=0) - 5, allp.max(axis=0) + 5
    w, h = (np.ceil((hi - lo) * px_per_mm)).astype(int) + 1
    img = np.full((h, w, 3), 255, np.uint8)

    def poly(e, color, th=1):
        s = (sample(e, 0.3) - lo) * px_per_mm
        s[:, 1] = h - s[:, 1]
        cv2.polylines(img, [np.round(s * 16).astype(np.int32).reshape(-1, 1, 2)], e["kind"] == "circle", color, th,
                      cv2.LINE_AA, shift=4)

    for e in annot:
        poly(e, (210, 210, 210))
    for e in geo:
        poly(e, (60, 170, 60), 3)
    colors = {"geometry": (40, 40, 220), "dimension_graphics": (200, 120, 0), "hatch_pattern": (0, 160, 160),
              "text_strokes": (160, 160, 160), "extra": (0, 140, 255)}
    for e, r in zip(det, det_rows):
        poly(e, colors[r["category"]], 1)
    y = 20
    for name, c in [("reference", (60, 170, 60))] + list(colors.items()):
        cv2.line(img, (10, y), (40, y), c, 3)
        cv2.putText(img, name, (48, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 20
    cv2.imencode(".png", img)[1].tofile(str(path))


# ------------------------------------------------------------------ table

def fmt_err(s):
    return "—" if not s or s["n"] == 0 else f"{s['mean']:.3f} / {s['max']:.3f}"


def table(results):
    lines = ["| вход | калибровка | эталон (тело) | найдено тело | по видам | слои верно | лишние | дубли | <2 мм | центр окр. ср/макс | радиус окр. ср/макс | концы отрезков ср/макс | замкнутые: эталон → найдено (+лишние) | разрывы | время, с |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in results:
        if "error" in m:
            lines.append(f"| {m['case']} | ОШИБКА: {m['error']} |" + " |" * 13)
            continue
        c = m["calibration"]
        cal = f"{c['method']}{' (ручная)' if c['manual'] else ''}, {c['scale_status']}"
        if m["registration"]["residual_scale"]:
            cal += f", масштаб×{m['registration']['residual_scale']:.4f}"
        f = m["found"]
        d = m["detected"]
        k = m["contours"]
        gaps = m["defects"].get("gap", 0)
        si = m["defects"].get("self_intersection", 0)
        lines.append(
            f"| {m['case']} | {cal} | {m['reference']['body_primitives']} | {f['body']} ({f['body'] / max(1, m['reference']['body_primitives']):.0%}) "
            f"| {', '.join(f'{kk} {vv}' for kk, vv in f['by_kind'].items())} "
            f"| {'—' if m['layers']['accuracy'] is None else format(m['layers']['accuracy'], '.0%')} "
            f"| {d['categories'].get('extra', 0)} | {d['duplicates']} "
            f"| {d['short_lt_2mm']} | {fmt_err(m['errors_mm']['circle_center'])} | {fmt_err(m['errors_mm']['circle_radius'])} "
            f"| {fmt_err(m['errors_mm']['line_end'])} | {k['reference_closed']} → {k['found']} (+{max(0, k['detected_closed'] - k['found'])}){'¹' if k.get('reference_main_from_layer0') else ''} "
            f"| {gaps} разр., {si} самопер. | {m['timings_s']['total']} |")
    lines += ["", "¹ в эталоне нет слоя основной линии: основными считаются кривые слоя «0»."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="default", choices=sorted(SHEETS))
    ap.add_argument("--out", default=str(ROOT / "mlcv/eval/out"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--n", type=int, default=8, help="random set: number of sheets")
    ap.add_argument("--seed", type=int, default=44, help="random set: seed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = prepare_inputs(a.set, out, a.n, a.seed)
    if a.only:
        cases = [c for c in cases if any(o in c["case"] for o in a.only)]
    results = []
    for case in cases:
        try:
            m = evaluate_case(case, out)
        except Exception as exc:
            m = {"case": case["case"], "error": f"{type(exc).__name__}: {exc}"}
        results.append(m)
        print(case["case"], "error" if "error" in m else
              f"body {m['found']['body']}/{m['reference']['body_primitives']} extra {m['detected']['categories'].get('extra', 0)} "
              f"closed {m['contours']['found']}/{m['contours']['reference_closed']} {m['timings_s']['total']} s", flush=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "table.md").write_text(table(results) + "\n", encoding="utf-8")
    print(out / "table.md")


if __name__ == "__main__":
    main()
