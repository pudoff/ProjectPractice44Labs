"""Connected contours of the part geometry and geometric defects (all distances in sheet mm).

Only primitives with role "part" (solid thick lines, arcs, circles) take part: text, axes, dimension and hatch
lines are never joined to the part. Ends closer than snap_tol are joined; T-junctions and proper crossings split
edges; nothing is closed beyond snap_tol - such places are reported as gaps. Arcs stay arcs: a contour edge
references its primitive and the parameter range used."""
import math

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon

from .primitives import sample

EPS = 1e-9


# ------------------------------------------------------------------ edge geometry

class Edge:
    """A piece of a primitive: line (t in mm from p) or arc (t in degrees from a0, counter-clockwise)."""

    def __init__(self, prim, t0, t1):
        self.prim, self.t0, self.t1 = prim, t0, t1
        self.kind = "line" if prim["kind"] == "line" else "arc"

    def point(self, t):
        e = self.prim
        if self.kind == "line":
            d = e["q"] - e["p"]
            return e["p"] + d / (np.linalg.norm(d) + EPS) * t
        a = math.radians(self.a0 + t)
        return e["c"] + e["r"] * np.array([math.cos(a), math.sin(a)])

    @property
    def a0(self):
        return self.prim.get("a0", 0.0)

    @property
    def a(self):
        return self.point(self.t0)

    @property
    def b(self):
        return self.point(self.t1)

    def tangent_angle(self, at_start):
        """Direction of departure from the start (at_start) or from the end going backwards."""
        if self.kind == "line":
            d = self.b - self.a if at_start else self.a - self.b
            return math.atan2(d[1], d[0])
        p = self.a if at_start else self.b
        r = p - self.prim["c"]
        base = math.atan2(r[1], r[0])
        return base + (math.pi / 2 if at_start else -math.pi / 2)

    def polyline(self, forward=True, step=0.25):
        if self.kind == "line":
            pts = np.array([self.a, self.b])
        else:
            n = max(2, int(math.ceil(math.radians(self.t1 - self.t0) * self.prim["r"] / step)) + 1)
            pts = np.array([self.point(t) for t in np.linspace(self.t0, self.t1, n)])
        return pts if forward else pts[::-1]

    def dense(self, step):
        """Points every `step` mm along the edge (polyline() keeps only the vertices of a line)."""
        n = max(2, int(math.ceil(self.length() / step)) + 1)
        return np.array([self.point(t) for t in np.linspace(self.t0, self.t1, n)])

    def length(self):
        return (self.t1 - self.t0) if self.kind == "line" else math.radians(self.t1 - self.t0) * self.prim["r"]

    def project(self, pt):
        """(parameter, distance) of the closest point of this edge's primitive to pt."""
        e = self.prim
        if self.kind == "line":
            d = e["q"] - e["p"]
            L = np.linalg.norm(d) + EPS
            t = float(np.clip((pt - e["p"]) @ d / L, 0, L))
            return t, float(np.linalg.norm(self.point(t) - pt))
        v = pt - e["c"]
        ang = (math.degrees(math.atan2(v[1], v[0])) - self.a0) % 360
        span = self.full_span()
        if ang > span:
            ang = 0.0 if (360 - ang) < (ang - span) else span
        return ang, float(np.linalg.norm(self.point(ang) - pt))

    def full_span(self):
        e = self.prim
        if self.kind == "line":
            return float(np.linalg.norm(e["q"] - e["p"]))
        return 360.0 if e["kind"] == "circle" else ((e["a1"] - e["a0"]) % 360 or 360.0)


def _intersections(e1, e2):
    """Proper intersection points of two primitives as (t1, t2) parameter pairs."""
    p1, p2 = e1.prim, e2.prim
    out = []
    if e1.kind == "line" and e2.kind == "line":
        a, b, c, d = p1["p"], p1["q"], p2["p"], p2["q"]
        r, s = b - a, d - c
        den = r[0] * s[1] - r[1] * s[0]
        if abs(den) < EPS:
            return []
        u = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
        v = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / den
        if 0 <= u <= 1 and 0 <= v <= 1:
            x = a + u * r
            out.append((e1.project(x)[0], e2.project(x)[0], x))
        return out
    if e1.kind == "arc" and e2.kind == "line":
        return [(t1, t2, x) for t2, t1, x in _intersections(e2, e1)]
    if e1.kind == "line":
        a, b = p1["p"], p1["q"]
        c, rad = p2["c"], p2["r"]
        d = b - a
        f = a - c
        qa, qb, qc = d @ d, 2 * f @ d, f @ f - rad * rad
        disc = qb * qb - 4 * qa * qc
        if disc < 0 or qa < EPS:
            return []
        for s in ((-qb - math.sqrt(disc)) / (2 * qa), (-qb + math.sqrt(disc)) / (2 * qa)):
            if 0 <= s <= 1:
                x = a + s * d
                t2, dist = e2.project(x)
                if dist < 1e-6:
                    out.append((e1.project(x)[0], t2, x))
        return out
    c1, r1, c2, r2 = p1["c"], p1["r"], p2["c"], p2["r"]
    dd = np.linalg.norm(c2 - c1)
    if dd < EPS or dd > r1 + r2 or dd < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd)
    h = math.sqrt(max(r1 * r1 - a * a, 0))
    m = c1 + a * (c2 - c1) / dd
    perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / dd
    for x in (m + h * perp, m - h * perp):
        t1, d1 = e1.project(x)
        t2, d2 = e2.project(x)
        if d1 < 1e-6 and d2 < 1e-6:
            out.append((t1, t2, x))
    return out


# ------------------------------------------------------------------ main

def build(part, others, snap_tol, gap_search):
    """part: primitives with role part (status ok). others: remaining visible primitives (for gap context).
    Returns dict with contours, defects, nodes."""
    defects = []
    whole = [Edge(p, 0.0, Edge(p, 0, 0).full_span()) for p in part]

    # split points: T-junctions (an end within snap_tol of another primitive's interior) and proper crossings
    splits = {i: set() for i in range(len(whole))}
    ends = []
    for i, e in enumerate(whole):
        if not (e.prim["kind"] == "circle"):
            ends += [(i, e.a), (i, e.b)]
    samples, owner = _sample_edges(whole, 0.5)
    tree = cKDTree(samples) if len(samples) else None
    for i, pt in ends:
        if tree is None:
            break
        for j in {owner[k] for k in tree.query_ball_point(pt, snap_tol + 0.5)}:
            if j == i:
                continue
            t, dist = whole[j].project(pt)
            span = whole[j].full_span()
            interior = min(t, span - t) * (1.0 if whole[j].kind == "line" else math.radians(1) * whole[j].prim["r"]) > snap_tol
            if dist <= snap_tol and (interior or whole[j].prim["kind"] == "circle"):
                splits[j].add(t)
    crossings = []
    trims = [{} for _ in whole]
    for i in range(len(whole)):
        if tree is None:
            break
        near = set()
        for pt in sample(whole[i].prim, 0.5):
            near.update(owner[k] for k in tree.query_ball_point(pt, 0.6))
        for j in near:
            if j <= i:
                continue
            for t1, t2, x in _intersections(whole[i], whole[j]):
                ei, ej = whole[i], whole[j]
                near_i = _near_end(ei, t1, snap_tol)
                near_j = _near_end(ej, t2, snap_tol)
                if near_i is None and near_j is None:
                    splits[i].add(t1)
                    splits[j].add(t2)
                    crossings.append((x, ei.prim, ej.prim))
                    continue
                # an end overshooting another edge by less than the tolerance is a junction: trim it there
                for k, near, t in ((i, near_i, t1), (j, near_j, t2)):
                    if near is None:
                        splits[k].add(t)
                    else:
                        trims[k][near] = t
    for x, a, b in crossings:
        defects.append({"code": "self_intersection", "severity": "warning", "at_mm": _r(x),
                        "object_ids": [a["id"], b["id"]],
                        "message": "part edges cross each other (not at their ends)"})

    edges = []
    loops = []                                    # unsplit circles
    for i, e in enumerate(whole):
        span = e.full_span()
        ts = sorted(t for t in splits[i] if 0 < t < span)
        if e.prim["kind"] == "circle":
            if not ts:
                loops.append(e)
                continue
            # rotate the parameter origin to the first split so the pieces are plain arcs
            first = ts[0]
            ts = [t - first for t in ts] + [360.0]
            e.prim = dict(e.prim, a0=(e.prim.get("a0", 0.0) + first) % 360, kind="circle")
            bounds = list(zip([0.0] + ts[1:-1], ts[1:]))
        else:
            lo, hi = trims[i].get("start", 0.0), trims[i].get("end", span)
            cuts = [lo] + _dedupe([t for t in ts if lo < t < hi], 1e-6) + [hi]
            bounds = list(zip(cuts[:-1], cuts[1:]))
        for t0, t1 in bounds:
            piece = Edge(e.prim, t0, t1)
            if piece.length() > 1e-6:
                edges.append(piece)

    # nodes: cluster edge ends within snap_tol
    pts = np.array([p for ed in edges for p in (ed.a, ed.b)]) if edges else np.zeros((0, 2))
    node_of, nodes, spread = _cluster(pts, snap_tol)
    for k, ed in enumerate(edges):
        ed.u, ed.v = node_of[2 * k], node_of[2 * k + 1]
    edges = [ed for ed in edges if not (ed.u == ed.v and ed.kind == "line")]
    edges = _dedupe_edges(edges, defects)

    deg = np.zeros(len(nodes), int)
    for ed in edges:
        deg[ed.u] += 1
        deg[ed.v] += 1
    dangling = [n for n in range(len(nodes)) if deg[n] == 1]
    defects += _crossing_nodes(edges, nodes, deg, [d["at_mm"] for d in defects if d["code"] == "self_intersection"])

    # prune trees: what is left lies on cycles
    alive = list(edges)
    open_edges = []
    d = deg.copy()
    changed = True
    while changed:
        changed = False
        keep = []
        for ed in alive:
            if d[ed.u] == 1 or d[ed.v] == 1:
                d[ed.u] -= 1
                d[ed.v] -= 1
                open_edges.append(ed)
                changed = True
            else:
                keep.append(ed)
        alive = keep

    faces = _faces(alive, nodes)
    contours = []
    comp = _components(alive, len(nodes))
    by_comp = {}
    for f in faces:
        by_comp.setdefault(comp[f["half"][0][0].u if f["half"][0][1] else f["half"][0][0].v], []).append(f)
    for faces_c in by_comp.values():
        outer = min(faces_c, key=lambda f: f["area"])
        bounded = [f for f in faces_c if f is not outer and f["area"] > 0]
        if outer["area"] < 0:
            outer_ring = [(ed, not fw) for ed, fw in reversed(outer["half"])]
            contours.append(_contour(outer_ring, "outer_boundary", nodes, spread, len(bounded)))
        if len(bounded) > 1 or outer["area"] >= 0:
            for f in bounded:
                contours.append(_contour(f["half"], "region", nodes, spread, 1))
    for e in loops:
        contours.append(_contour([(e, True)], "circle", nodes, spread, 1))
    for chain in _chains(open_edges):
        contours.append(_contour(chain, "open_chain", nodes, spread, 0, closed=False))

    _containment(contours)

    defects += _gaps(dangling, nodes, edges, part, others, snap_tol, gap_search)
    return {"contours": contours, "defects": defects, "nodes": nodes, "edges": edges}


def _near_end(e, t, tol):
    """'start' / 'end' if parameter t lies within tol (mm along the edge) of that end of a non-circle edge."""
    if e.prim["kind"] == "circle":
        return None
    k = 1.0 if e.kind == "line" else math.radians(1) * e.prim["r"]
    if abs(t) * k <= tol:
        return "start"
    if abs(e.full_span() - t) * k <= tol:
        return "end"
    return None


def _crossing_nodes(edges, nodes, deg, known, tol_deg=10.0):
    """A node where two edge pairs pass straight through and alternate around it is a crossing even when the
    skeleton broke the lines there (X junction): report it as a self-intersection."""
    out = []
    for n in np.nonzero(deg == 4)[0]:
        if any(np.linalg.norm(np.asarray(k) - nodes[n]) < 1.0 for k in known):
            continue
        dirs = []
        for ed in edges:
            if ed.u == n:
                dirs.append((math.degrees(ed.tangent_angle(True)) % 360, ed))
            if ed.v == n:
                dirs.append((math.degrees(ed.tangent_angle(False)) % 360, ed))
        if len(dirs) != 4:
            continue
        dirs.sort(key=lambda t: t[0])
        straight = all(abs((dirs[k + 2][0] - dirs[k][0]) % 360 - 180) <= tol_deg for k in (0, 1))
        if straight:
            out.append({"code": "self_intersection", "severity": "warning", "at_mm": _r(nodes[n]),
                        "object_ids": sorted({d[1].prim["id"] for d in dirs}),
                        "message": "two part edges pass straight through one node (X crossing)"})
    return out


def _r(p):
    return [round(float(p[0]), 4), round(float(p[1]), 4)]


def _dedupe(ts, tol):
    out = []
    for t in ts:
        if not out or t - out[-1] > tol:
            out.append(t)
    return out


def _sample_edges(edges, step):
    pts, owner = [], []
    for i, e in enumerate(edges):
        s = sample(e.prim, step)
        pts.append(s)
        owner += [i] * len(s)
    return (np.vstack(pts) if pts else np.zeros((0, 2))), owner


def _cluster(pts, tol):
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        for i, j in cKDTree(pts).query_pairs(tol):
            parent[find(i)] = find(j)
    roots = {}
    node_of = []
    for i in range(n):
        node_of.append(roots.setdefault(find(i), len(roots)))
    nodes = np.zeros((len(roots), 2))
    members = {}
    for i, k in enumerate(node_of):
        members.setdefault(k, []).append(pts[i])
    spread = np.zeros(len(roots))
    for k, m in members.items():
        m = np.array(m)
        nodes[k] = m.mean(axis=0)
        spread[k] = float(np.max(np.linalg.norm(m[:, None] - m[None], axis=2))) if len(m) > 1 else 0.0
    return node_of, nodes, spread


def _dedupe_edges(edges, defects):
    """Two edges between the same nodes with the same shape: keep one, report the duplicate."""
    out, seen = [], {}
    for ed in edges:
        mid = ed.point((ed.t0 + ed.t1) / 2)
        key = (min(ed.u, ed.v), max(ed.u, ed.v))
        dup = next((o for o in seen.get(key, []) if np.linalg.norm(o.point((o.t0 + o.t1) / 2) - mid) < 0.2), None)
        if dup is not None:
            defects.append({"code": "duplicate_edge", "severity": "warning", "at_mm": _r(mid),
                            "object_ids": [dup.prim["id"], ed.prim["id"]], "message": "the same edge found twice"})
            continue
        seen.setdefault(key, []).append(ed)
        out.append(ed)
    return out


def _faces(edges, nodes):
    """Planar face traversal; each half-edge (edge, forward) is used once. Faces keep the area sign:
    bounded faces are counter-clockwise (area > 0), the outer face of a component is clockwise."""
    out_at = {}
    for ed in edges:
        out_at.setdefault(ed.u, []).append((ed.tangent_angle(True), ed, True))
        out_at.setdefault(ed.v, []).append((ed.tangent_angle(False), ed, False))
    for v in out_at:
        out_at[v].sort(key=lambda t: t[0])
    used = set()
    faces = []
    for ed in edges:
        for fw in (True, False):
            if (id(ed), fw) in used:
                continue
            half, cur = [], (ed, fw)
            for _ in range(4 * len(edges) + 4):
                if (id(cur[0]), cur[1]) in used:
                    break
                used.add((id(cur[0]), cur[1]))
                half.append(cur)
                e, f = cur
                v = e.v if f else e.u
                lst = out_at[v]
                twin = next(k for k, (_, e2, f2) in enumerate(lst) if e2 is e and f2 != f)
                _, e3, f3 = lst[(twin - 1) % len(lst)]          # next clockwise after the reversed edge
                cur = (e3, f3)
            pts = np.vstack([e.polyline(f) for e, f in half])
            area = 0.5 * float(np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
                               + pts[-1, 0] * pts[0, 1] - pts[0, 0] * pts[-1, 1])
            faces.append({"half": half, "area": area})
    return faces


def _components(edges, n):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for ed in edges:
        parent[find(ed.u)] = find(ed.v)
    return [find(i) for i in range(n)]


def _chains(open_edges):
    """Group pruned edges into connected open chains (in traversal order where possible)."""
    remaining = list(open_edges)
    chains = []
    while remaining:
        chain = [(remaining.pop(0), True)]
        grown = True
        while grown:
            grown = False
            end = chain[-1][0].v if chain[-1][1] else chain[-1][0].u
            start = chain[0][0].u if chain[0][1] else chain[0][0].v
            for k, ed in enumerate(remaining):
                if ed.u == end or ed.v == end:
                    chain.append((ed, ed.u == end))
                elif ed.v == start or ed.u == start:
                    chain.insert(0, (ed, ed.v == start))
                else:
                    continue
                remaining.pop(k)
                grown = True
                break
        chains.append(chain)
    return chains


def _contour(half, role, nodes, spread, faces_inside, closed=True):
    pts = np.vstack([e.polyline(f) for e, f in half])
    edges = []
    closure = 0.0
    for e, f in half:
        item = {"object_id": e.prim["id"], "kind": e.kind, "reversed": not f}
        if e.kind == "line":
            item.update(start=_r(e.a if f else e.b), end=_r(e.b if f else e.a))
        else:
            item.update(start=_r(e.a if f else e.b), end=_r(e.b if f else e.a), center=_r(e.prim["c"]),
                        radius=round(float(e.prim["r"]), 4),
                        start_angle=round((e.a0 + e.t0) % 360, 4), end_angle=round((e.a0 + e.t1) % 360, 4))
        edges.append(item)
        for n in ((getattr(e, "u", None), getattr(e, "v", None))):
            if n is not None:
                closure = max(closure, float(spread[n]))
    if role == "circle":
        e = half[0][0]
        poly = Polygon(sample(e.prim, 0.25))
        closure = 0.0
    else:
        poly = Polygon(pts) if closed and len(pts) >= 3 else LineString(pts)
        if closed and not poly.is_valid:          # a ring touching itself (bridge between cycles): area of the cleaned shape
            poly = poly.buffer(0)
    area = abs(poly.area) if closed else 0.0
    return {"role": role, "closed": closed, "edges": edges, "area_mm2": round(float(area), 4),
            "length_mm": round(float(sum(LineString(e.polyline()).length for e, _ in half)), 4),
            "closure_error_mm": round(closure, 4), "faces_inside": faces_inside,
            "bbox_mm": [round(float(v), 4) for v in poly.bounds], "_poly": poly, "_pts": pts}


def _containment(contours):
    closed = [c for c in contours if c["closed"] and c["role"] in ("outer_boundary", "circle")]
    for c in contours:
        c["parent"] = None
        if not c["closed"]:
            continue
        rep = c["_poly"].representative_point()
        best = None
        for o in closed:
            if o is c or o["_poly"].area <= c["_poly"].area:
                continue
            if o["_poly"].contains(rep) and (c["role"] != "region" or not _same_component(o, c)):
                if best is None or o["_poly"].area < best["_poly"].area:
                    best = o
        c["parent"] = best


def _same_component(a, b):
    return bool({e["object_id"] for e in a["edges"]} & {e["object_id"] for e in b["edges"]})


def _gaps(dangling, nodes, edges, part, others, snap_tol, gap_search):
    """A dangling end is a gap when other part geometry lies within gap_search; it is never closed here."""
    out = []
    if not dangling:
        return out
    samples, owner = [], []
    for ed in edges:
        s = ed.dense(0.1)
        samples.append(s)
        owner += [ed] * len(s)
    samples = np.vstack(samples)
    tree = cKDTree(samples)
    other_pts, other_owner = [], []
    for o in others:
        s = sample(o, 0.25)
        other_pts.append(s)
        other_owner += [o] * len(s)
    otree = cKDTree(np.vstack(other_pts)) if other_pts else None
    for n in dangling:
        p = nodes[n]
        own = [ed for ed in edges if ed.u == n or ed.v == n][0]
        best = None
        for k in tree.query_ball_point(p, gap_search):
            ed = owner[k]
            if ed is own or ed.u == n or ed.v == n:
                continue
            dist = float(np.linalg.norm(samples[k] - p))
            if dist > snap_tol and (best is None or dist < best[0]):
                best = (dist, samples[k], ed)
        near = None
        if otree is not None:
            dist, k = otree.query(p)
            if dist <= gap_search:
                near = {"object_id": other_owner[k]["id"], "role": other_owner[k]["role"], "distance_mm": round(float(dist), 4)}
        if best is not None:
            item = {"code": "gap", "severity": "warning", "at_mm": _r(p), "to_mm": _r(best[1]),
                    "distance_mm": round(best[0], 4), "object_ids": [own.prim["id"], best[2].prim["id"]],
                    "message": f"open end {best[0]:.2f} mm from other part geometry (join tolerance {snap_tol} mm); not closed automatically"}
        else:
            item = {"code": "open_end", "severity": "info", "at_mm": _r(p), "object_ids": [own.prim["id"]],
                    "message": f"open end, no part geometry within {gap_search} mm"}
        if near:
            item["nearest_non_part"] = near
        out.append(item)
    return out
