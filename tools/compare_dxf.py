"""Compare a generated DXF drawing against a reference DXF.

Usage:
    python compare_dxf.py REF.dxf GEN.dxf --out DIR [--tol 0.5] [--align bbox|geom|none]

Writes DIR/report.json, DIR/report.md and DIR/overlay.png (ref = green, gen = red).

Metrics:
  * geometry   - point-cloud precision/recall/F1 (curves sampled every 0.2 mm) at
                 several tolerances, plus primitive-level recall/precision
                 (a primitive counts as matched when >=90 % of its samples lie
                 within tol of the other drawing).
  * dimensions - DIMENSION entities matched by type + displayed text + position.
  * text       - TEXT/MTEXT matched by position, compared by content (CER).
  * layers     - semantic class (MAIN/THIN/AXIS/HIDDEN/DIMS/TEXT/HATCH) agreement
                 for matched primitives; reference layers are mapped heuristically.
  * hatches    - count and total boundary area.
All coordinates are in drawing units (mm). Reports are written as UTF-8 files;
do not rely on the Windows console for Cyrillic.
"""
import argparse
import difflib
import json
import math
import re
import sys
from pathlib import Path

import ezdxf
import numpy as np
from ezdxf import bbox, path as ezpath
from scipy.spatial import cKDTree

STEP = 0.2  # mm between curve samples
CURVES = {"LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ELLIPSE"}

# Heuristic mapping of foreign layer names to semantic classes (lower-case keys).
LAYER_MAP = {
    "main": "MAIN", "tost": "MAIN", "wide": "MAIN", "основная": "MAIN", "осн": "MAIN",
    "thin": "THIN", "slim": "THIN", "тонкая": "THIN", "0": "THIN",
    "axis": "AXIS", "ax": "AXIS", "осевая": "AXIS",
    "hidden": "HIDDEN", "пунктир": "HIDDEN",
    "dims": "DIMS", "dim": "DIMS", "размер": "DIMS", "размерная": "DIMS",
    "text": "TEXT", "техт": "TEXT",
    "hatch": "HATCH", "штриховка": "HATCH",
}


def entity_class(e, doc):
    """Semantic class of an entity from its type, linetype and layer."""
    t = e.dxftype()
    if t == "DIMENSION":
        return "DIMS"
    if t in ("TEXT", "MTEXT"):
        return "TEXT"
    if t == "HATCH":
        return "HATCH"
    lt = (e.dxf.get("linetype") or "BYLAYER").upper()
    layer = e.dxf.layer
    if lt in ("BYLAYER", "BYBLOCK") and layer in doc.layers:
        lt = doc.layers.get(layer).dxf.linetype.upper()
    if any(k in lt for k in ("CENTER", "DASHDOT", "ОСЕВ", "AXLED")):
        return "AXIS"
    if any(k in lt for k in ("DASHED", "HIDDEN", "ISO02", "ПУНКТ")):
        return "HIDDEN"
    return LAYER_MAP.get(layer.lower(), "THIN")


def sample_entity(e):
    try:
        p = ezpath.make_path(e)
    except Exception:
        return None
    pts = []
    for sub in p.sub_paths() if p.has_sub_paths else [p]:
        v = np.array([(q.x, q.y) for q in sub.flattening(0.01, segments=4)])
        if len(v) < 2:
            continue
        for a, b in zip(v[:-1], v[1:]):
            n = max(1, int(math.ceil(np.linalg.norm(b - a) / STEP)))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + (b - a) * t)
        pts.append(v[-1:])
    return np.vstack(pts) if pts else None


NONPLOT_LAYERS = {"defpoints"}


def iter_flat(layout):
    """Model space entities with block references exploded (dimension blocks excluded).
    Entities on the non-plotting DEFPOINTS layer are skipped: they never appear on paper/scans
    (archive DXFs keep the sheet trim rectangle there)."""
    for e in layout:
        if e.dxf.get("layer", "0").lower() in NONPLOT_LAYERS:
            continue
        if e.dxftype() == "INSERT":
            try:
                yield from (v for v in e.virtual_entities() if v.dxf.get("layer", "0").lower() not in NONPLOT_LAYERS)
            except Exception:
                continue
        else:
            yield e


def plain_text(e):
    if e.dxftype() == "MTEXT":
        s = e.plain_text()
    else:
        s = e.dxf.text
    return normalize_text(s)


def normalize_text(s):
    s = s.replace("%%c", "⌀").replace("%%C", "⌀").replace("%%d", "°").replace("%%D", "°")
    s = s.replace("%%p", "±").replace("%%P", "±").replace("Ø", "⌀").replace("ø", "⌀")
    s = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s)
    s = s.replace("\\P", " ")
    return re.sub(r"\s+", " ", s).strip()


def dim_text(d):
    """Displayed dimension text: taken from the rendered *D block when present,
    otherwise rebuilt from measurement * DIMLFAC and the text override."""
    dimtype = d.dxf.dimtype & 7
    try:
        m = d.get_measurement()
        val = float(m) if not hasattr(m, "x") else 0.0
    except Exception:
        val = 0.0
    if dimtype in (2, 5):
        val = min(val, 360.0 - val) if val > 180 else val
    else:
        try:
            val *= d.override().get("dimlfac", None) or 1.0
        except Exception:
            pass
    block = d.get_geometry_block()
    if block is not None:
        for e in block:
            if e.dxftype() == "MTEXT":
                return normalize_text(e.plain_text()), dimtype, val
            if e.dxftype() == "TEXT":
                return normalize_text(e.dxf.text), dimtype, val
    num = f"{val:.3f}".rstrip("0").rstrip(".").replace(".", ",")
    raw = d.dxf.get("text") or "<>"
    if raw.strip() == "":
        raw = "<>"
    txt = raw.replace("<>", num)
    if dimtype == 4 and not txt.upper().startswith("R"):
        txt = "R" + txt
    if dimtype == 3 and "%%c" not in txt.lower() and "⌀" not in txt:
        txt = "⌀" + txt
    if dimtype in (2, 5) and "°" not in txt and "%%d" not in txt.lower():
        txt += "°"
    return normalize_text(txt), dimtype, val


def load(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    ents = list(iter_flat(msp))
    return doc, msp, ents


def collect(doc, ents, offset):
    curves, texts, dims, hatches = [], [], [], []
    for e in ents:
        t = e.dxftype()
        if t in CURVES:
            pts = sample_entity(e)
            if pts is not None and len(pts):
                curves.append({"type": t, "cls": entity_class(e, doc), "layer": e.dxf.layer, "pts": pts - offset})
        elif t in ("TEXT", "MTEXT"):
            ins = np.array(e.dxf.insert)[:2] - offset
            txt = plain_text(e)
            h = e.dxf.get("char_height", 0) if t == "MTEXT" else e.dxf.get("height", 0)
            texts.append({"text": txt, "pos": ins, "center": text_center(e, txt, h) - offset,
                          "layer": e.dxf.layer, "h": h})
        elif t == "DIMENSION":
            txt, dt, val = dim_text(e)
            pos = np.array(e.dxf.get("text_midpoint", e.dxf.defpoint))[:2] - offset
            dims.append({"text": txt, "type": dt, "value": val, "pos": pos, "layer": e.dxf.layer})
        elif t == "HATCH":
            area = 0.0
            try:
                for p in ezpath.from_hatch(e):
                    v = np.array([(q.x, q.y) for q in p.flattening(0.05)])
                    if len(v) > 2:
                        area += abs(0.5 * np.sum(v[:-1, 0] * v[1:, 1] - v[1:, 0] * v[:-1, 1]))
            except Exception:
                pass
            hatches.append({"pattern": e.dxf.pattern_name, "area": area, "layer": e.dxf.layer})
    return curves, texts, dims, hatches


def pr_f1(ref_pts, gen_pts, tol):
    if len(ref_pts) == 0 or len(gen_pts) == 0:
        return 0.0, 0.0, 0.0
    d_rg, _ = cKDTree(gen_pts).query(ref_pts)
    d_gr, _ = cKDTree(ref_pts).query(gen_pts)
    rec = float(np.mean(d_rg <= tol))
    prec = float(np.mean(d_gr <= tol))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def primitive_match(src, other_tree, tol, frac=0.9):
    flags = []
    for c in src:
        d, _ = other_tree.query(c["pts"])
        flags.append(bool(np.mean(d <= tol) >= frac))
    return flags


def text_center(e, txt, h):
    """Font-independent estimate of the centre of a TEXT/MTEXT line (0.7*h per character),
    taking the alignment into account, so left/centre aligned texts at the same place match."""
    rot = math.radians(e.dxf.get("rotation", 0) or 0)
    if e.dxftype() == "MTEXT":
        d = np.array(e.dxf.get("text_direction", (math.cos(rot), math.sin(rot), 0)))[:2]
        d = d / (np.linalg.norm(d) or 1)
        att = e.dxf.get("attachment_point", 1)
        col, row = (att - 1) % 3, (att - 1) // 3
    else:
        d = np.array([math.cos(rot), math.sin(rot)])
        col = {0: 0, 1: 1, 2: 2, 3: 1, 4: 1, 5: 1}.get(e.dxf.get("halign", 0), 0)
        row = {0: 2, 1: 2, 2: 1, 3: 0}.get(e.dxf.get("valign", 0), 2)
    n = np.array([-d[1], d[0]])
    anchor = np.array(e.dxf.insert)[:2]
    if e.dxftype() == "TEXT" and (e.dxf.get("halign", 0) or e.dxf.get("valign", 0)):
        anchor = np.array(e.dxf.get("align_point", e.dxf.insert))[:2]
    w = 0.7 * (h or 2.5) * max(len(txt), 1)
    return anchor + d * w * (0.5 - col / 2) + n * (h or 2.5) * (row / 2 - 0.5)


def match_texts(ref, gen, radius=6.0, key="pos"):
    used, rows = set(), []
    gtree = cKDTree(np.array([g[key] for g in gen])) if gen else None
    for r in ref:
        best, best_s, best_j = None, -1.0, None
        if gtree is not None:
            for j in gtree.query_ball_point(r[key], radius):
                if j in used:
                    continue
                s = difflib.SequenceMatcher(None, r["text"], gen[j]["text"]).ratio()
                if s > best_s:
                    best, best_s, best_j = gen[j], s, j
        if best_j is not None:
            used.add(best_j)
        cer = cer_of(r["text"], best["text"]) if best else 1.0
        rows.append({"ref": r["text"], "gen": best["text"] if best else None, "cer": cer,
                     "exact": best is not None and best["text"] == r["text"]})
    return rows, len(gen) - len(used)


def cer_of(a, b):
    if not a:
        return 0.0 if not b else 1.0
    m = difflib.SequenceMatcher(None, a, b)
    edits = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in m.get_opcodes() if op != "equal")
    return min(1.0, edits / len(a))


_LOOKALIKE = str.maketrans({"М": "M", "Х": "x", "х": "x", "×": "x", "Н": "H", "А": "A", "В": "B", "С": "C", "Е": "E",
                            "К": "K", "Р": "P", "Т": "T", "О": "O", "о": "o", "р": "p", "с": "c", "е": "e"})


def canon_dim(t):
    """Text of a dimension with diameter signs unified and Cyrillic/Latin look-alikes folded."""
    t = t.replace("∅", "⌀").replace("Ø", "⌀").replace("ø", "⌀").translate(_LOOKALIKE)
    return re.sub(r"\s+", "", t)


def dim_number(t):
    m = re.search(r"\d+(?:[.,]\d+)?", t)
    return float(m.group(0).replace(",", ".")) if m else None


def match_dims_by_value(ref, gen, radius=15.0):
    """Lenient matching: same leading numeric value, position within radius (prefix/notation ignored)."""
    used, found = set(), 0
    for r in ref:
        v = dim_number(r["text"])
        if v is None:
            continue
        cands = [(np.linalg.norm(g["pos"] - r["pos"]), j) for j, g in enumerate(gen)
                 if j not in used and dim_number(g["text"]) is not None and abs(dim_number(g["text"]) - v) < 1e-3]
        cands = [c for c in cands if c[0] <= radius]
        if cands:
            used.add(min(cands)[1])
            found += 1
    return found


def match_dims(ref, gen, radius=15.0):
    used, rows = set(), []
    for r in ref:
        cands = [(np.linalg.norm(g["pos"] - r["pos"]), j) for j, g in enumerate(gen)
                 if j not in used and canon_dim(g["text"]) == canon_dim(r["text"])]
        cands = [c for c in cands if c[0] <= radius]
        if cands:
            _, j = min(cands)
            used.add(j)
            rows.append({"ref": r["text"], "type": r["type"], "found": True, "type_ok": gen[j]["type"] == r["type"]})
        else:
            rows.append({"ref": r["text"], "type": r["type"], "found": False, "type_ok": False})
    extra = [gen[j]["text"] for j in range(len(gen)) if j not in used]
    return rows, extra


def frame_offset(msp, align):
    if align == "none":
        return np.zeros(2)
    if align == "geom":
        # curves only: TEXT extents depend on the fonts installed on the machine running the comparison
        ext = bbox.extents(e for e in iter_flat(msp) if e.dxftype() in CURVES)
    else:
        ext = bbox.extents(e for e in msp if e.dxftype() != "DIMENSION")
    return np.array([ext.extmin.x, ext.extmin.y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("gen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--align", choices=["bbox", "geom", "none"], default="bbox")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rdoc, rmsp, rents = load(a.ref)
    gdoc, gmsp, gents = load(a.gen)
    roff, goff = frame_offset(rmsp, a.align), frame_offset(gmsp, a.align)
    rc, rt, rd, rh = collect(rdoc, rents, roff)
    gc, gt, gd, gh = collect(gdoc, gents, goff)

    rpts = np.vstack([c["pts"] for c in rc]) if rc else np.zeros((0, 2))
    gpts = np.vstack([c["pts"] for c in gc]) if gc else np.zeros((0, 2))
    geo = {}
    for tol in (0.1, 0.25, 0.5, 1.0):
        p, r, f = pr_f1(rpts, gpts, tol)
        geo[f"tol_{tol}"] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    rtree = cKDTree(rpts) if len(rpts) else None
    gtree = cKDTree(gpts) if len(gpts) else None
    rflags = primitive_match(rc, gtree, a.tol) if gtree is not None else [False] * len(rc)
    gflags = primitive_match(gc, rtree, a.tol) if rtree is not None else [False] * len(gc)

    # layer agreement: for each matched gen primitive, compare its class with the nearest ref primitive class
    cls_ok = cls_n = 0
    confusion = {}
    if rc and gc:
        owner = np.concatenate([np.full(len(c["pts"]), i) for i, c in enumerate(rc)])
        for c, ok in zip(gc, gflags):
            if not ok:
                continue
            _, idx = rtree.query(c["pts"][len(c["pts"]) // 2])
            rcls = rc[owner[idx]]["cls"]
            key = f"{rcls}->{c['cls']}"
            confusion[key] = confusion.get(key, 0) + 1
            cls_n += 1
            cls_ok += rcls == c["cls"]

    trows, textra = match_texts(rt, gt)
    crows, cextra = match_texts(rt, gt, key="center")
    drows, dextra = match_dims(rd, gd)
    dval = match_dims_by_value(rd, gd)

    by_type = {}
    for c, ok in zip(rc, rflags):
        s = by_type.setdefault(c["type"], [0, 0])
        s[0] += ok
        s[1] += 1

    report = {
        "ref": a.ref, "gen": a.gen, "tol_mm": a.tol,
        "gen_header": {"version": gdoc.dxfversion, "insunits": gdoc.header.get("$INSUNITS"),
                       "codepage": gdoc.header.get("$DWGCODEPAGE"),
                       "layers": [l.dxf.name for l in gdoc.layers]},
        "counts": {"ref_curves": len(rc), "gen_curves": len(gc), "ref_texts": len(rt), "gen_texts": len(gt),
                   "ref_dims": len(rd), "gen_dims": len(gd), "ref_hatches": len(rh), "gen_hatches": len(gh)},
        "geometry_points": geo,
        "primitives": {"recall": round(sum(rflags) / max(1, len(rflags)), 4),
                       "precision": round(sum(gflags) / max(1, len(gflags)), 4),
                       "recall_by_ref_type": {k: f"{v[0]}/{v[1]}" for k, v in by_type.items()}},
        "layers": {"class_accuracy": round(cls_ok / max(1, cls_n), 4), "confusion": confusion},
        "dimensions": {"recall": round(sum(r["found"] for r in drows) / max(1, len(drows)), 4),
                       "recall_by_value": round(dval / max(1, len(drows)), 4),
                       "type_accuracy": round(sum(r["type_ok"] for r in drows) / max(1, len(drows)), 4),
                       "missing": [r["ref"] for r in drows if not r["found"]], "extra": dextra},
        "text": {"exact_rate": round(sum(r["exact"] for r in trows) / max(1, len(trows)), 4),
                 "mean_cer": round(float(np.mean([r["cer"] for r in trows])) if trows else 0.0, 4),
                 "unmatched_gen": textra,
                 "worst": sorted(trows, key=lambda r: -r["cer"])[:15],
                 "by_center": {"exact_rate": round(sum(r["exact"] for r in crows) / max(1, len(crows)), 4),
                               "mean_cer": round(float(np.mean([r["cer"] for r in crows])) if crows else 0.0, 4),
                               "unmatched_gen": cextra,
                               "worst": sorted(crows, key=lambda r: -r["cer"])[:20]}},
        "hatches": {"ref_area": round(sum(h["area"] for h in rh), 1), "gen_area": round(sum(h["area"] for h in gh), 1)},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# DXF comparison\n", f"ref: `{a.ref}`  \ngen: `{a.gen}`  \ntolerance: {a.tol} mm\n",
          "| metric | value |", "|---|---|"]
    for k, v in geo.items():
        md.append(f"| geometry F1 @{k[4:]} mm | {v['f1']:.3f} (P {v['precision']:.3f} / R {v['recall']:.3f}) |")
    md += [f"| primitive recall / precision | {report['primitives']['recall']:.3f} / {report['primitives']['precision']:.3f} |",
           f"| layer class accuracy | {report['layers']['class_accuracy']:.3f} |",
           f"| dimension recall | {report['dimensions']['recall']:.3f} ({len(rd)} ref, {len(gd)} gen) |",
           f"| dimension recall by value (notation ignored) | {report['dimensions']['recall_by_value']:.3f} |",
           f"| dimension type accuracy | {report['dimensions']['type_accuracy']:.3f} |",
           f"| text exact / mean CER | {report['text']['exact_rate']:.3f} / {report['text']['mean_cer']:.3f} |",
           f"| text exact / mean CER (matched by line centre) | {report['text']['by_center']['exact_rate']:.3f} / {report['text']['by_center']['mean_cer']:.3f} |",
           f"| hatch area ref / gen | {report['hatches']['ref_area']} / {report['hatches']['gen_area']} |",
           "", "Missing dimensions: " + ", ".join(report["dimensions"]["missing"]),
           "", "Extra dimensions: " + ", ".join(report["dimensions"]["extra"]), ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        w = max(rpts[:, 0].max() if len(rpts) else 1, gpts[:, 0].max() if len(gpts) else 1)
        h = max(rpts[:, 1].max() if len(rpts) else 1, gpts[:, 1].max() if len(gpts) else 1)
        fig = plt.figure(figsize=(w / 25.4, h / 25.4), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        from matplotlib.collections import LineCollection

        def segments(curves):
            segs = []
            for c in curves:
                p = c["pts"]
                if len(p) < 2:
                    continue
                keep = np.linalg.norm(p[1:] - p[:-1], axis=1) <= STEP * 1.5
                segs.extend(np.stack([p[:-1][keep], p[1:][keep]], axis=1))
            return segs

        ax.add_collection(LineCollection(segments(rc), colors="#1a9850", linewidths=0.6))
        ax.add_collection(LineCollection(segments(gc), colors="#d73027", linewidths=0.6, alpha=0.7))
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(out / "overlay.png")
    except Exception as ex:  # overlay is optional
        print("overlay skipped:", ex, file=sys.stderr)

    print(json.dumps({k: report[k] for k in ("geometry_points", "primitives", "dimensions")}, ensure_ascii=True)[:2000])


if __name__ == "__main__":
    main()
