"""cv.json assembly.

Top-level keys read by the backend (backend/app/pipeline.py, models with extra="forbid"):
  schema_version, status, coordinate_space, scale_status, primitives (line/circle/arc, layer MAIN/THIN/AXIS),
  hatches (outer/holes/pattern/scale/angle), issues (code/severity/message/object_id).
Everything else is extra detail for review, metrics and OCR linking: objects (role, confidence, source pixels),
points (ids of measurable points), contours, defects, text_regions, calibration, transforms, params, timings."""
import math

import numpy as np

from .primitives import sweep
from .transform import apply_to_geometry

SCHEMA_VERSION = "0.1"
CV_FORMAT = "drawcv-1"
LAYER_BY_ROLE = {"part": "MAIN", "thin": "THIN", "axis": "AXIS", "hidden": "THIN", "hatch": "THIN"}
LINETYPE_BY_ROLE = {"axis": "CENTER", "hidden": "DASHED"}


def r4(v):
    return round(float(v), 4)


def pt(p):
    return [r4(p[0]), r4(p[1])]


def geometry(e):
    if e["kind"] == "line":
        return {"start": pt(e["p"]), "end": pt(e["q"])}
    g = {"center": pt(e["c"]), "radius": r4(e["r"])}
    if e["kind"] == "arc":
        g.update(start_angle=r4(e["a0"] % 360), end_angle=r4(e["a1"] % 360))
    return g


def layer_of(e, threshold):
    if e["role"] in LAYER_BY_ROLE:
        return LAYER_BY_ROLE[e["role"]]
    return "MAIN" if e["stroke_mm"] >= threshold else "THIN"


def exported(e):
    """Primitives handed to the DXF exporter: visible geometry, not noise/duplicates/text/arrows/unsupported."""
    return e["status"] == "ok"


def build_cv_json(page, prep, cal, work_to_mm, params, prims, regions, topo, hatches, groups, stats, timings, version):
    source_to_mm = prep.source_to_work.then(work_to_mm)
    mm_to_source = source_to_mm.inverse()
    threshold = stats["lineweights"]["threshold_mm"]

    primitives, objects = [], {}
    for e in prims:
        g = geometry(e)
        obj = {"kind": e["kind"], "role": e["role"], "status": e["status"], "confidence": e["confidence"],
               "support": r4(e["support"]), "rms_mm": r4(e["rms"]), "stroke_mm": r4(e["stroke_mm"]),
               "length_mm": r4(e["length_mm"]), "pieces": int(e.get("pieces", 1)), "layer": layer_of(e, threshold),
               "geometry_mm": g, "geometry_source_px": _round_geom(apply_to_geometry(mm_to_source, g)),
               "reasons": e["reasons"]}
        if e["kind"] == "arc":
            obj["sweep_deg"] = r4(sweep(e))
        if e["role"] in LINETYPE_BY_ROLE:
            obj["linetype"] = LINETYPE_BY_ROLE[e["role"]]
        for key in ("dash_count", "snapped_deg", "extended_mm"):
            if key in e:
                obj[key] = e[key]
        if "concentric_with" in e:
            obj["concentric_with"] = e["concentric_with"]["id"]
        if "duplicate_of" in e:
            obj["duplicate_of"] = e["duplicate_of"]["id"]
        objects[e["id"]] = obj
        if exported(e):
            primitives.append({"id": e["id"], "kind": e["kind"], "layer": obj["layer"], **g})

    points = _points(prims, params.snap_tol_mm)
    contours = [_contour_json(c) for c in topo["contours"]]
    defects = []
    for k, d in enumerate(sorted(topo["defects"], key=lambda d: (d["code"], d["at_mm"]))):
        item = {"id": f"defect_{k}", **{key: v for key, v in d.items() if not key.startswith("_")}}
        defects.append(item)

    hatch_items = []
    for k, g in enumerate(hatches):
        outer = g["_contour"]["_pts"]
        hatch_items.append({"id": f"hatch_{k}", "outer": [pt(p) for p in _thin_ring(outer)],
                            "holes": [[pt(p) for p in _thin_ring(h["_pts"])] for h in g["_holes"]],
                            "pattern": "ANSI31", "scale": r4(g["spacing_mm"] / 3.175),
                            "angle": r4(g["angle_deg"] - 45)})
        g["hatch_id"] = f"hatch_{k}"
    hatch_groups = [{"id": g["id"], "line_ids": [e["id"] for e in g["lines"]], "angle_deg": g["angle_deg"],
                     "spacing_mm": g["spacing_mm"], "regularity": g["regularity"], "boundary": g["boundary"],
                     "ends_on_boundary": g.get("ends_on_boundary"), "contour_id": g.get("contour_id"),
                     "hatch_id": g.get("hatch_id"),
                     "hole_contour_ids": [h["id"] for h in g.get("_holes", [])]} for g in groups]

    text_regions = []
    work_to_source = prep.source_to_work.inverse()
    for r in regions:
        item = {"id": r["id"], "provider": r["provider"], "score": None if r.get("score") is None else r4(r["score"]),
                "poly_source_px": [pt(p) for p in work_to_source.apply(r["poly_work_px"])],
                "poly_mm": [pt(p) for p in work_to_mm.apply(r["poly_work_px"])],
                "erased_components": r.get("erased_components", 0), "kept_components": r.get("kept_components", 0)}
        for key in ("ocr_id", "text"):
            if r.get(key) is not None:
                item[key] = r[key]
        text_regions.append(item)

    issues = _issues(prims, defects, stats, cal)
    counts = {}
    for e in prims:
        key = f"{e['kind']}:{e['role']}:{e['status']}"
        counts[key] = counts.get(key, 0) + 1
    summary = {"exported_primitives": len(primitives), "objects": len(objects),
               "by_kind_role_status": dict(sorted(counts.items())),
               "closed_contours": sum(c["closed"] for c in contours),
               "open_chains": sum(not c["closed"] for c in contours),
               "defects": {code: sum(d["code"] == code for d in defects) for code in sorted({d["code"] for d in defects})},
               "text_regions": len(text_regions), "hatches_resolved": len(hatch_items),
               "hatch_groups_unresolved": sum(g["boundary"] == "unresolved" for g in groups)}

    return {
        "schema_version": SCHEMA_VERSION,
        "format": CV_FORMAT,
        "status": "ok",
        "module": {"name": "drawcv", "version": version},
        "source": page.info(),
        "coordinate_space": "sheet_mm",
        "units": "mm",
        "scale_status": cal["scale_status"],
        "calibration": cal,
        "preprocess": prep.info(),
        "transforms": {"source_px_to_work_px": prep.source_to_work.to_list(),
                       "work_px_to_sheet_mm": work_to_mm.to_list(),
                       "source_px_to_sheet_mm": source_to_mm.to_list(),
                       "sheet_mm_to_source_px": mm_to_source.to_list()},
        "params": params.to_dict(),
        "stats": stats,
        "summary": summary,
        "timings_s": dict(timings, total=round(sum(timings.values()), 3)),
        "primitives": primitives,
        "hatches": hatch_items,
        "issues": issues,
        "objects": objects,
        "points": points,
        "contours": contours,
        "defects": defects,
        "hatch_groups": hatch_groups,
        "text_regions": text_regions,
    }


def _round_geom(g):
    out = {}
    for k, v in g.items():
        out[k] = pt(v) if isinstance(v, (list, tuple, np.ndarray)) else r4(v)
    return out


def _thin_ring(pts, tol=0.02):
    """Drop consecutive duplicate and collinear vertices of a sampled ring."""
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    keep = []
    n = len(out)
    for i in range(n):
        a, b, c = out[i - 1], out[i], out[(i + 1) % n]
        cross = abs((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0])
        if cross > 1e-6 or n <= 3:
            keep.append(b)
    return keep if len(keep) >= 3 else out


def _points(prims, tol):
    """Measurable points for linking (OCR dimensions refer to them): ends, centres and junctions of visible
    primitives, clustered within the join tolerance. Each point lists the anchors that coincide there."""
    anchors = []
    for e in prims:
        if e["status"] != "ok":
            continue
        if e["kind"] == "line":
            anchors += [(e["p"], e["id"], "start"), (e["q"], e["id"], "end"), ((e["p"] + e["q"]) / 2, e["id"], "mid")]
        else:
            anchors.append((e["c"], e["id"], "center"))
            if e["kind"] == "arc":
                for name, a in (("start", e["a0"]), ("end", e["a1"])):
                    anchors.append((e["c"] + e["r"] * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]),
                                    e["id"], name))
            else:
                for name, a in (("quadrant_0", 0), ("quadrant_90", 90), ("quadrant_180", 180), ("quadrant_270", 270)):
                    anchors.append((e["c"] + e["r"] * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]),
                                    e["id"], name))
    if not anchors:
        return []
    from scipy.spatial import cKDTree
    xy = np.array([a[0] for a in anchors])
    parent = list(range(len(anchors)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(xy).query_pairs(tol):
        if anchors[i][2] != "mid" and anchors[j][2] != "mid":
            parent[find(i)] = find(j)
    groups = {}
    for i in range(len(anchors)):
        groups.setdefault(find(i), []).append(i)
    pts = []
    for members in groups.values():
        c = xy[members].mean(axis=0)
        refs = sorted(({"object_id": anchors[i][1], "anchor": anchors[i][2]} for i in members),
                      key=lambda r: (r["object_id"], r["anchor"]))
        kinds = {r["anchor"] for r in refs}
        kind = "center" if kinds == {"center"} else "mid" if kinds == {"mid"} else (
            "junction" if len({r["object_id"] for r in refs}) > 1 else "end" if kinds & {"start", "end"} else "quadrant")
        pts.append({"xy": pt(c), "kind": kind, "refs": refs})
    pts.sort(key=lambda p: (p["xy"][0], p["xy"][1], p["kind"]))
    for k, p in enumerate(pts):
        p["id"] = f"pt_{k}"
    return [{"id": p["id"], **{k: v for k, v in p.items() if k != "id"}} for p in pts]


def _contour_json(c):
    return {"id": c["id"], "role": c["role"], "closed": c["closed"], "edges": c["edges"],
            "object_ids": sorted({e["object_id"] for e in c["edges"]}, key=lambda s: (s.split("_")[0], int(s.split("_")[1]))),
            "area_mm2": c["area_mm2"], "length_mm": c["length_mm"], "closure_error_mm": c["closure_error_mm"],
            "faces_inside": c["faces_inside"], "bbox_mm": c["bbox_mm"],
            "parent_id": c["parent"]["id"] if c.get("parent") else None}


def _issues(prims, defects, stats, cal):
    """Backend Issue list (code, severity warning|error, message, object_id)."""
    issues = []
    for d in defects:
        if d["severity"] not in ("warning", "error"):
            continue
        where = f" at ({d['at_mm'][0]:.2f}, {d['at_mm'][1]:.2f}) mm"
        issues.append({"code": d["code"], "severity": d["severity"], "message": d["message"] + where,
                       "object_id": d["object_ids"][0] if d.get("object_ids") else None})
    for e in prims:
        if e["status"] == "unsupported":
            issues.append({"code": "arc_unsupported", "severity": "warning",
                           "message": "arc not exported: " + "; ".join(e["reasons"]), "object_id": e["id"]})
    if not stats["lineweights"]["separable"]:
        issues.append({"code": "lineweight_not_separable", "severity": "warning",
                       "message": "thick and thin lines have the same width: all solid lines treated as part contour",
                       "object_id": None})
    if stats.get("text_note"):
        issues.append({"code": "text_detector_fallback", "severity": "warning", "message": stats["text_note"],
                       "object_id": None})
    if cal["method"] == "image_size":
        issues.append({"code": "calibration_failed", "severity": "error",
                       "message": "sheet frame not found: coordinates assume the image is the whole sheet",
                       "object_id": None})
    if cal["method"] == "pdf_page":
        issues.append({"code": "pdf_scale_unverified", "severity": "error",
                       "message": "no sheet frame found: PDF page size used, the content may be printed scaled",
                       "object_id": None})
    check = cal.get("pdf_page_check")
    if check and abs(check["content_scale_vs_page"] - 1) > 0.001:
        pct = (1 / check["content_scale_vs_page"] - 1) * 100
        issues.append({"code": "pdf_content_scaled", "severity": "warning",
                       "message": f"PDF content is printed at {100 + pct:.2f} % of the page; calibrated on the drawn frame",
                       "object_id": None})
    if cal.get("anisotropy"):
        issues.append({"code": "anisotropic_scale", "severity": "warning",
                       "message": f"x and y scales differ by {cal['anisotropy'] * 100:.2f} % (printed/scanned unevenly); "
                                  "separate x/y calibration from the frame verified by the title block",
                       "object_id": None})
    if cal.get("manual"):
        issues.append({"code": "manual_calibration", "severity": "warning",
                       "message": f"manual calibration {cal['px_per_mm'][0]:.4f} px/mm", "object_id": None})
    return issues
