"""Stable reference points for OCR: what every point of cv.json means and which distances can be measured.

points[] stays the strict list the backend validates; the explanation lives next to it:
  point_info{pt_id}  - point in source pixels (after rotation / crop the page transform maps it back), what it means
                       for each object ("start of line_3", "centre of hole circle_1"), contour vertices, the object it
                       lies on at a T-junction, and the observation before the join when the point was moved;
  measurable[]       - candidates of measured sections: line lengths, diameters, radii, distances between parallel lines,
                       and the feet of thin (extension) lines on part objects, where a dimension often really measures."""
import math

import numpy as np
from scipy.spatial import cKDTree

from .topology import COINCIDENT_MM

ANCHOR_WORDS = {"start": "начало", "end": "конец", "mid": "середина", "center": "центр"}
KIND_WORDS = {"line": "отрезка", "arc": "дуги", "circle": "окружности"}
AXIS_TOL_DEG = 0.5
PARALLEL_DEG = 0.5
MIN_OVERLAP = 0.3
FOOT_REACH_MM = 2.5                  # extension lines stop short of the contour by up to this much
FOOT_MIN_ANGLE_DEG = 30.0


def _anchor_words(obj_id, kind, anchor, hole):
    if anchor.startswith("quadrant_"):
        return f"точка окружности {obj_id} на {anchor.split('_')[1]}°"
    if kind == "circle" and anchor == "center" and hole:
        return f"центр отверстия {obj_id}"
    return f"{ANCHOR_WORDS[anchor]} {KIND_WORDS[kind]} {obj_id}"


def build_point_info(points, prims, contours, closures, mm_to_source):
    by_id = {e["id"]: e for e in prims}
    holes = {c["edges"][0]["object_id"] for c in contours if c["role"] == "circle" and c.get("parent_id")}
    vertices, owners = [], []
    for c in contours:
        if not c["closed"] or c["role"] == "circle":
            continue
        for edge in c["edges"]:
            vertices.append(edge["start"])
            owners.append(c["id"])
    tree = cKDTree(np.array(vertices)) if vertices else None
    hosts = {}
    for c in closures:
        if c["status"] == "joined":
            for m in c["members"]:
                hosts[(m["object_id"], m["end"])] = (c["id"], c["hosts"])
    info = {}
    for p in points:
        links, on, closure = [], set(), None
        for r in p["refs"]:
            e = by_id[r["object_id"]]
            link = {**r, "kind": e["kind"], "role": e["role"],
                    "meaning": _anchor_words(r["object_id"], e["kind"], r["anchor"], r["object_id"] in holes)}
            move = next((a for a in e.get("adjustments", []) if a["end"] == r["anchor"]), None)
            if move:
                link.update(observed_mm=move["from_mm"], moved_mm=move["moved_mm"], move_reason=move["reason"])
            links.append(link)
            if (r["object_id"], r["anchor"]) in hosts:
                closure, host_ids = hosts[(r["object_id"], r["anchor"])]
                on.update(host_ids)
        contour_ids = []
        if tree is not None:
            contour_ids = sorted({owners[i] for i in tree.query_ball_point(p["xy"], 2 * COINCIDENT_MM)})
        words = "; ".join(link["meaning"] for link in links)
        if contour_ids and p["kind"] in ("junction", "end"):
            words = f"вершина контура {', '.join(contour_ids)}: {words}"
        if on:
            words += f"; лежит на {', '.join(sorted(on))}"
        item = {"source_px": [round(float(v), 3) for v in mm_to_source.apply(np.array(p["xy"], float))],
                "meaning": words, "links": links}
        if contour_ids:
            item["contour_ids"] = contour_ids
        if on:
            item["lies_on"] = sorted(on)
        if closure:
            item["closure_id"] = closure
        info[p["id"]] = item
    return info


def _direction(angle):
    a = angle % 180
    if min(a, 180 - a) <= AXIS_TOL_DEG:
        return "horizontal"
    if abs(a - 90) <= AXIS_TOL_DEG:
        return "vertical"
    return "oblique"


def build_measurable(points, prims):
    """Measured-section candidates in sheet mm of the part geometry (the view scale is applied by whoever reads the
    printed number). Each one names the points it runs between, so a dimension can refer to exact geometry."""
    where = {(r["object_id"], r["anchor"]): p["id"] for p in points for r in p["refs"]}
    out = []
    part = [e for e in prims if e["status"] == "ok" and e["role"] == "part"]
    lines = []
    for e in part:
        if e["kind"] == "line":
            d = e["q"] - e["p"]
            angle = math.degrees(math.atan2(d[1], d[0]))
            lines.append((e, angle % 180))
            out.append({"kind": "line_length", "object_ids": [e["id"]], "p1": where.get((e["id"], "start")),
                        "p2": where.get((e["id"], "end")), "value_mm": round(float(np.linalg.norm(d)), 4),
                        "direction": _direction(angle)})
        elif e["kind"] == "circle":
            out.append({"kind": "circle_diameter", "object_ids": [e["id"]], "center": where.get((e["id"], "center")),
                        "value_mm": round(2 * float(e["r"]), 4)})
        else:
            out.append({"kind": "arc_radius", "object_ids": [e["id"]], "center": where.get((e["id"], "center")),
                        "p1": where.get((e["id"], "start")), "p2": where.get((e["id"], "end")),
                        "value_mm": round(float(e["r"]), 4)})
    out += _parallel_distances(lines, where)
    out += _extension_feet(prims, part, where)
    for k, m in enumerate(out):
        m["id"] = f"measure_{k}"
    return out


def _extension_feet(prims, part, where):
    """Where a thin line, continued from its end, meets a part line nearby: the point on the part a dimension refers to
    when it does not start at a corner."""
    targets = [e for e in part if e["kind"] == "line"]
    out = []
    for t in prims:
        if t["status"] != "ok" or t["role"] != "thin" or t["kind"] != "line":
            continue
        direction = (t["q"] - t["p"]) / (np.linalg.norm(t["q"] - t["p"]) + 1e-12)
        for name, end, step in (("start", t["p"], -direction), ("end", t["q"], direction)):
            best = None
            for e in targets:
                d = e["q"] - e["p"]
                den = step[0] * d[1] - step[1] * d[0]
                n = float(np.linalg.norm(d))
                if abs(den) < math.sin(math.radians(FOOT_MIN_ANGLE_DEG)) * n:
                    continue
                w = e["p"] - end
                along_thin = (w[0] * d[1] - w[1] * d[0]) / den
                along_part = (w[0] * step[1] - w[1] * step[0]) / den
                if -0.5 <= along_thin <= FOOT_REACH_MM and 0.0 <= along_part <= 1.0 and (best is None or abs(along_thin) < abs(best[0])):
                    best = (float(along_thin), e, end + step * along_thin)
            if best:
                out.append({"kind": "extension_foot", "object_ids": [t["id"], best[1]["id"]], "p1": where.get((t["id"], name)),
                            "xy_mm": [round(float(v), 4) for v in best[2]], "on_object_id": best[1]["id"],
                            "from_end": name, "value_mm": round(abs(best[0]), 4)})
    return out


def _parallel_distances(lines, where):
    """For each line, the nearest parallel line on each side with a common stretch: width-like dimensions."""
    found = {}
    for i, (a, ang_a) in enumerate(lines):
        d = (a["q"] - a["p"]) / (np.linalg.norm(a["q"] - a["p"]) + 1e-12)
        n = np.array([-d[1], d[0]])
        ta0, ta1 = sorted((float(a["p"] @ d), float(a["q"] @ d)))
        nearest = {}
        for j, (b, ang_b) in enumerate(lines):
            if i == j or abs((ang_a - ang_b + 90) % 180 - 90) > PARALLEL_DEG:
                continue
            tb0, tb1 = sorted((float(b["p"] @ d), float(b["q"] @ d)))
            common = min(ta1, tb1) - max(ta0, tb0)
            if common < MIN_OVERLAP * min(ta1 - ta0, tb1 - tb0):
                continue
            off = float(((b["p"] + b["q"]) / 2 - a["p"]) @ n)
            side = off > 0
            if abs(off) > 1e-3 and (side not in nearest or abs(off) < abs(nearest[side][0])):
                nearest[side] = (off, j)
        for off, j in nearest.values():
            found[(min(i, j), max(i, j))] = abs(off)
    out = []
    for (i, j), dist in sorted(found.items()):
        a, b = lines[i][0], lines[j][0]
        d = (a["q"] - a["p"]) / (np.linalg.norm(a["q"] - a["p"]) + 1e-12)
        pairs = [(abs(float((pa - pb) @ d)), na, nb) for pa, na in ((a["p"], "start"), (a["q"], "end"))
                 for pb, nb in ((b["p"], "start"), (b["q"], "end"))]
        offset, na, nb = min(pairs, key=lambda t: t[0])
        out.append({"kind": "parallel_distance", "object_ids": [a["id"], b["id"]], "p1": where.get((a["id"], na)),
                    "p2": where.get((b["id"], nb)), "value_mm": round(dist, 4), "offset_along_mm": round(offset, 4),
                    "direction": _direction(lines[i][1] + 90)})
    return out
