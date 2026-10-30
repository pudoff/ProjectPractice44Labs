"""Review images.

overlay.png       - every object (including excluded noise, duplicates, text residue) drawn over the ORIGINAL page
                    through sheet_mm -> source_px, so mistakes are visible next to the ink they came from;
geometry.png      - part geometry only: closed contours, open chains, gaps, crossings, resolved hatch boundaries;
prep_compare.png  - before/after preparation and a zoom where text was masked (red = erased in the working copy only)."""
import math

import cv2
import numpy as np

from .primitives import sample

# BGR
COLORS = {"part": (40, 40, 220), "thin": (200, 120, 0), "axis": (180, 0, 180), "hidden": (0, 140, 140),
          "hatch": (0, 160, 0), "frame": (150, 150, 150), "title_block": (150, 150, 150),
          "noise": (0, 165, 255), "duplicate": (0, 215, 255), "text_residue": (60, 200, 60),
          "arrowhead": (120, 120, 0), "low_support": (0, 100, 255), "junction_artifact": (0, 165, 255), "unsupported": (255, 0, 255)}
LEGEND = ["part", "thin", "axis", "hidden", "hatch", "frame", "noise", "duplicate", "text_residue", "unsupported"]


def _fit(img, max_side):
    f = min(1.0, max_side / max(img.shape[:2]))
    if f < 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img, f


def _poly(img, pts, color, thick=1, closed=False):
    p = np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA, shift=4)


def _legend(img, items, origin=(10, 10), scale=0.6):
    x, y = origin
    pad = int(22 * scale / 0.6)
    cv2.rectangle(img, (x - 4, y - 4), (x + int(190 * scale / 0.6), y + pad * len(items) + 4), (255, 255, 255), -1)
    for k, (name, color) in enumerate(items):
        yy = y + pad * k + pad // 2
        cv2.line(img, (x + 2, yy), (x + 28, yy), color, 3)
        cv2.putText(img, name, (x + 36, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def overlay_source(page, prims, regions, cv, mm_to_source, max_side, f_out):
    base, f = _fit(page.gray, max_side)
    img = cv2.cvtColor((base * 0.35 + 165).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for r in cv["text_regions"]:
        _poly(img, np.array(r["poly_source_px"]) * f, (60, 200, 60), 1, closed=True)
    order = sorted(prims, key=lambda e: e["status"] == "ok")
    for e in order:
        key = "unsupported" if e["status"] == "unsupported" else e["role"]
        color = COLORS.get(key, (0, 0, 0))
        thick = 2 if e["role"] == "part" and e["status"] == "ok" else 1
        pts = mm_to_source.apply(sample(e, 0.5)) * f
        _poly(img, pts, color, thick, closed=e["kind"] == "circle")
        if e["status"] != "ok" and e["role"] in ("noise", "duplicate", "low_support"):
            c = pts[len(pts) // 2]
            cv2.circle(img, (int(c[0]), int(c[1])), 4, color, 1, cv2.LINE_AA)
    _legend(img, [(n, COLORS[n]) for n in LEGEND])
    cv2.imencode(".png", img)[1].tofile(str(f_out))


def geometry_preview(cv, topo, hatches, out, px_per_mm=4.0):
    W, H = cv["calibration"]["sheet_mm"]
    w, h = int(math.ceil(W * px_per_mm)) + 1, int(math.ceil(H * px_per_mm)) + 1
    img = np.full((h, w, 3), 255, np.uint8)

    def to_px(pts):
        pts = np.asarray(pts, float)
        return np.column_stack([pts[:, 0] * px_per_mm, (H - pts[:, 1]) * px_per_mm])

    for g in hatches:
        ring = to_px(g["_contour"]["_pts"]).astype(np.int32)
        cv2.fillPoly(img, [ring], (235, 215, 190))
        for hole in g["_holes"]:
            cv2.fillPoly(img, [to_px(hole["_pts"]).astype(np.int32)], (255, 255, 255))
    for c in topo["contours"]:
        color = {"outer_boundary": (30, 140, 30), "region": (120, 170, 60), "circle": (30, 140, 30),
                 "open_chain": (40, 40, 220)}[c["role"]]
        _poly(img, to_px(c["_pts"]), color, 2 if c["role"] != "region" else 1, closed=c["closed"])
    for d in cv["defects"]:
        p = to_px([d["at_mm"]])[0]
        if d["code"] == "gap":
            q = to_px([d["to_mm"]])[0]
            cv2.circle(img, (int(p[0]), int(p[1])), 9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{d['distance_mm']:.2f}", (int(p[0]) + 10, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)
        elif d["code"] == "self_intersection":
            x, y = int(p[0]), int(p[1])
            cv2.line(img, (x - 7, y - 7), (x + 7, y + 7), (200, 0, 200), 2, cv2.LINE_AA)
            cv2.line(img, (x - 7, y + 7), (x + 7, y - 7), (200, 0, 200), 2, cv2.LINE_AA)
        elif d["code"] == "open_end":
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 140, 255), 1, cv2.LINE_AA)
        elif d["code"] == "hatch_boundary_unresolved":
            cv2.putText(img, "?", (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    _legend(img, [("closed contour", (30, 140, 30)), ("open chain", (40, 40, 220)), ("gap (mm)", (0, 0, 255)),
                  ("self-intersection", (200, 0, 200)), ("open end", (0, 140, 255)), ("hatch area", (235, 215, 190))],
            scale=0.5)
    cv2.imencode(".png", img)[1].tofile(str(out))


def prep_compare(page, prep, clean, erased, regions, out, max_side=2400):
    left, f1 = _fit(page.gray, max_side // 2)
    right_src = ((1 - prep.bw) * 255).astype(np.uint8)
    right, _ = _fit(right_src, max_side // 2)
    right = cv2.resize(right, (int(round(right.shape[1] * left.shape[0] / right.shape[0])), left.shape[0]),
                       interpolation=cv2.INTER_AREA)
    top = np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), np.full((left.shape[0], 8, 3), 128, np.uint8),
                     cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)])
    cv2.putText(top, "source", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    cv2.putText(top, f"prepared (skew {prep.skew_deg:.2f} deg, x{prep.resample:.3f})", (left.shape[1] + 18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    # zoom: densest text region cluster, source vs working copy with erased ink in red
    zoom = None
    if regions:
        best = max(regions, key=lambda r: r.get("erased_components", 0))
        c = best["poly_work_px"].mean(axis=0)
        half = 350
        h, w = prep.bw.shape
        x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half * 0.6))
        x1, y1 = int(min(w, c[0] + half)), int(min(h, c[1] + half * 0.6))
        work_gray = prep.gray[y0:y1, x0:x1]
        vis = cv2.cvtColor(((1 - clean[y0:y1, x0:x1]) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis[erased[y0:y1, x0:x1]] = (0, 0, 230)
        zoom = np.hstack([cv2.cvtColor(work_gray, cv2.COLOR_GRAY2BGR), np.full((y1 - y0, 8, 3), 128, np.uint8), vis])
        f = top.shape[1] / zoom.shape[1]
        zoom = cv2.resize(zoom, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST if f > 1 else cv2.INTER_AREA)
        cv2.putText(zoom, "zoom: prepared image | working copy, red = text masked (source untouched)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    img = top if zoom is None else np.vstack([top, np.full((8, top.shape[1], 3), 128, np.uint8), zoom])
    cv2.imencode(".png", img)[1].tofile(str(out))


def save_all(out, page, prep, clean, erased, regions, prims, topo, hatches, cv, work_to_mm, params):
    mm_to_source = work_to_mm.inverse().then(prep.source_to_work.inverse())
    overlay_source(page, prims, regions, cv, mm_to_source, params.overlay_max_side_px, out / "overlay.png")
    geometry_preview(cv, topo, hatches, out / "geometry.png")
    prep_compare(page, prep, clean, erased, regions, out / "prep_compare.png")
