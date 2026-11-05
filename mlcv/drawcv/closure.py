"""Consistent coordinates for joined part contours (sheet mm).

Ends of part primitives that lie within the join tolerance are moved to one common point, or stay where they are and
the contour stays open. Nothing is closed by tolerance alone: after this step the contours are rebuilt with
COINCIDENT_MM, so "closed" means the neighbouring edges really meet.

The common point is chosen so that the shape changes least:
  - an end on the inside of another object (T-junction): the point lies on that object, which does not move;
  - arcs keep centre and radius (they are measured much better than the ends): the point lies on their circle;
  - lines extend or trim along themselves to the intersection; only when there is none (collinear, tangent) an end
    moves across the line: the moved line must stay on its ink (without a raster: within closure_max_turn_deg).
A move is refused when it would need more than the tolerance, turn or collapse an object, run over white paper or
cross another object. Every move and every refusal is recorded with its reason; the observation before the moves is
kept on the primitive."""
import math

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .primitives import length, sample, sweep
from .topology import COINCIDENT_MM, Edge, build

PARALLEL_DEG = 5.0
PARALLEL_MIN_ACROSS_MM = 0.15
MIN_LENGTH_MM = 0.5
REACH = 1.25                         # a move may be up to this many join tolerances (cluster diameter limit too)
T_PASSES = 6
HOLE_MIN_AREA_RATIO = 0.8
MAX_TURN_ON_INK_DEG = 15.0
LINE_ON_INK = 0.9


def end_point(e, which):
    if e["kind"] == "line":
        return e["p"] if which == "start" else e["q"]
    a = math.radians(e["a0"] if which == "start" else e["a1"])
    return e["c"] + e["r"] * np.array([math.cos(a), math.sin(a)])


def into(e, which):
    """Unit direction from the end into the primitive."""
    if e["kind"] == "line":
        d = (e["q"] - e["p"]) if which == "start" else (e["p"] - e["q"])
        return d / (np.linalg.norm(d) + 1e-12)
    a = math.radians(e["a0"] if which == "start" else e["a1"])
    t = np.array([-math.sin(a), math.cos(a)])
    return t if which == "start" else -t


def nearest_on_carrier(e, xy):
    """Closest point of the infinite line or the full circle."""
    if e["kind"] == "line":
        d = e["q"] - e["p"]
        return e["p"] + d * (((xy - e["p"]) @ d) / max(d @ d, 1e-12))
    v = xy - e["c"]
    n = np.linalg.norm(v)
    return e["c"] + e["r"] * (v / n if n > 1e-12 else np.array([1.0, 0.0]))


def tangent_at(e, xy):
    if e["kind"] == "line":
        d = e["q"] - e["p"]
        return d / (np.linalg.norm(d) + 1e-12)
    v = xy - e["c"]
    return np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-12)


def carrier_intersections(a, b):
    if a["kind"] == "line" and b["kind"] == "line":
        d1, d2 = a["q"] - a["p"], b["q"] - b["p"]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < math.sin(math.radians(1.0)) * np.linalg.norm(d1) * np.linalg.norm(d2):
            return []
        w = b["p"] - a["p"]
        return [a["p"] + d1 * ((w[0] * d2[1] - w[1] * d2[0]) / den)]
    if a["kind"] != "line" and b["kind"] == "line":
        a, b = b, a
    if a["kind"] == "line":
        d = a["q"] - a["p"]
        f = a["p"] - b["c"]
        qa, qb, qc = d @ d, 2 * f @ d, f @ f - b["r"] ** 2
        disc = qb * qb - 4 * qa * qc
        if disc < 0 or qa < 1e-12:
            return []
        root = math.sqrt(disc)
        return [a["p"] + d * ((-qb - root) / (2 * qa)), a["p"] + d * ((-qb + root) / (2 * qa))]
    dd = float(np.linalg.norm(b["c"] - a["c"]))
    if dd < 1e-9 or dd > a["r"] + b["r"] or dd < abs(a["r"] - b["r"]):
        return []
    k = (a["r"] ** 2 - b["r"] ** 2 + dd ** 2) / (2 * dd)
    h = math.sqrt(max(a["r"] ** 2 - k ** 2, 0.0))
    u = (b["c"] - a["c"]) / dd
    m = a["c"] + k * u
    n = np.array([-u[1], u[0]])
    return [m + h * n, m - h * n]


def refit_arc(e, which, target):
    """The arc with its radius through the moved end and its other end: the last resort for arcs that touch without an
    intersection. The radius is kept (three-point fits of short arcs are ill-conditioned); the centre shifts a little.
    Returns (c, r, a0, a1) or None for a degenerate chord."""
    other = end_point(e, "end" if which == "start" else "start")
    start, end = (np.asarray(target, float), other) if which == "start" else (other, np.asarray(target, float))
    chord = end - start
    n = float(np.linalg.norm(chord))
    if n < 1e-9:
        return None
    r = max(float(e["r"]), n / 2)
    h = math.sqrt(max(r * r - n * n / 4, 0.0))
    left = np.array([-chord[1], chord[0]]) / n
    centre = (start + end) / 2 + left * (h if sweep(e) <= 180 else -h)   # counter-clockwise from start to end
    return (centre, r, math.degrees(math.atan2(start[1] - centre[1], start[0] - centre[0])) % 360,
            math.degrees(math.atan2(end[1] - centre[1], end[0] - centre[0])) % 360)


class Joint:
    def __init__(self, members):
        self.members = members           # [(primitive, "start" | "end")]
        self.hosts = []                  # primitives whose inside the point must stay on
        self.reason = None
        self.refused = []                # [(primitive, which, reason)]
        self.target = None
        self.joined = False
        self.undone_reason = None


class Reconciler:
    def __init__(self, part, snap_tol, widths_at=None, thick_mm=None, max_turn_deg=3.0, min_ink=0.75):
        self.part = part
        self.tol = float(snap_tol)
        self.widths_at = widths_at
        self.thick = thick_mm
        self.max_turn = math.radians(max_turn_deg)
        self.min_ink = min_ink
        self.saved = {id(e): {k: (e[k].copy() if isinstance(e[k], np.ndarray) else e[k])
                              for k in ("p", "q", "c", "r", "a0", "a1") if k in e} for e in part}
        self.geoms = [LineString(sample(e, 0.25)) for e in part]
        self.index = STRtree(self.geoms)

    # ---------------------------------------------------------------- clusters of ends and hosts
    def clusters(self):
        ends = [(e, w) for e in self.part if e["kind"] != "circle" for w in ("start", "end")]
        if not ends:
            return []
        xy = np.array([end_point(e, w) for e, w in ends])
        pairs = sorted(cKDTree(xy).query_pairs(self.tol), key=lambda ij: (round(float(np.linalg.norm(xy[ij[0]] - xy[ij[1]])), 9), ij))
        group = list(range(len(ends)))
        members = {i: [i] for i in range(len(ends))}
        for i, j in pairs:
            a, b = group[i], group[j]
            if a == b:
                continue
            both = members[a] + members[b]
            if len({id(ends[k][0]) for k in both}) < len(both):      # both ends of one object never meet in one point
                continue
            if np.max(np.linalg.norm(xy[both][:, None] - xy[both][None], axis=2)) > REACH * self.tol:
                continue
            for k in members[b]:
                group[k] = a
            members[a] = both
            del members[b]
        joints = [Joint([ends[k] for k in sorted(m)]) for _, m in sorted(members.items())]
        samples, owner = [], []
        for k, e in enumerate(self.part):
            s = sample(e, 0.25)
            samples.append(s)
            owner += [k] * len(s)
        tree = cKDTree(np.vstack(samples))
        for joint in joints:
            mine = {id(e) for e, _ in joint.members}
            found = {}
            for e, w in joint.members:
                p = end_point(e, w)
                for k in {owner[i] for i in tree.query_ball_point(p, self.tol + 0.3)}:
                    h = self.part[k]
                    if id(h) in mine:
                        continue
                    edge = Edge(h, 0.0, Edge(h, 0, 0).full_span())
                    t, dist = edge.project(p)
                    if dist > self.tol:
                        continue
                    if h["kind"] != "circle":
                        scale = 1.0 if edge.kind == "line" else math.radians(1) * h["r"]
                        if min(t, edge.full_span() - t) * scale <= self.tol:
                            continue                                # near its own end: that end decides the join
                    if self._parallel(into(e, w), tangent_at(h, p)) and dist > PARALLEL_MIN_ACROSS_MM:
                        continue                                    # runs alongside, does not end on it
                    found[k] = h
            joint.hosts = [found[k] for k in sorted(found)]
        return [j for j in joints if len(j.members) + len(j.hosts) >= 2]

    @staticmethod
    def _parallel(u, v):
        return abs(u[0] * v[1] - u[1] * v[0]) < math.sin(math.radians(PARALLEL_DEG))

    # ---------------------------------------------------------------- the common point
    def solve(self, joint):
        """(point, reason) or (None, reason)."""
        lines = [(e, w) for e, w in joint.members if e["kind"] == "line"]
        arcs = [(e, w) for e, w in joint.members if e["kind"] == "arc"]
        pts = np.array([end_point(e, w) for e, w in joint.members])
        weights = np.array([max(length(e), 1e-3) for e, _ in joint.members])
        centre = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        hosts = joint.hosts
        if len(hosts) >= 2:
            near = [p for a in range(len(hosts)) for b in range(a + 1, len(hosts))
                    for p in carrier_intersections(hosts[a], hosts[b]) if np.linalg.norm(p - centre) <= REACH * self.tol]
            if not near:                                            # two neighbours without a common point: the closest
                hosts = [min(hosts, key=lambda h: float(np.linalg.norm(nearest_on_carrier(h, centre) - centre)))]
                joint.hosts = hosts
            else:
                return min(near, key=lambda p: float(np.linalg.norm(p - centre))), "t_junction"
        rigid = hosts + [e for e, _ in arcs]
        if len(rigid) >= 2:
            near = [p for a in range(len(rigid)) for b in range(a + 1, len(rigid))
                    for p in carrier_intersections(rigid[a], rigid[b]) if np.linalg.norm(p - centre) <= REACH * self.tol]
            if near:
                return min(near, key=lambda p: float(np.linalg.norm(p - centre))), "t_junction" if hosts else "arc_intersection"
            base = hosts[0] if hosts else max((e for e, _ in arcs), key=length)
            return nearest_on_carrier(base, centre), "arc_refit"
        if len(rigid) == 1:
            r = rigid[0]
            near = [p for e, w in lines for p in carrier_intersections(e, r)
                    if np.linalg.norm(p - end_point(e, w)) <= REACH * self.tol]
            if near and len(lines) == 1:
                end = end_point(*lines[0])
                return min(near, key=lambda p: float(np.linalg.norm(p - end))), "t_junction" if hosts else "line_arc_intersection"
            return nearest_on_carrier(r, centre), "t_junction" if hosts else "tangent_joint"
        if len(lines) == 2:
            (e1, w1), (e2, w2) = lines
            if not self._parallel(into(e1, w1), into(e2, w2)):
                x = carrier_intersections(e1, e2)
                if x and max(np.linalg.norm(x[0] - end_point(e1, w1)), np.linalg.norm(x[0] - end_point(e2, w2))) <= REACH * self.tol:
                    return x[0], "corner_intersection"
            return centre, "collinear_joint"
        # three or more lines: least-squares point of the carriers when they are not all parallel
        a = np.zeros((2, 2))
        b = np.zeros(2)
        for e, w in lines:
            n = into(e, w)
            n = np.array([-n[1], n[0]])
            m = np.outer(n, n)
            a += m
            b += m @ end_point(e, w)
        if np.linalg.cond(a) < 1e4:
            x = np.linalg.solve(a, b)
            if np.linalg.norm(x - centre) <= REACH * self.tol:
                return x, "corner_intersection"
        return centre, "collinear_joint"

    # ---------------------------------------------------------------- is a move allowed
    def refusal(self, joint, e, which, target):
        old = end_point(e, which)
        step = target - old
        moved = float(np.linalg.norm(step))
        if moved > REACH * self.tol:
            return "too_far"
        if e["kind"] == "line":
            other = end_point(e, "end" if which == "start" else "start")
            axis = old - other
            l0 = float(np.linalg.norm(axis))
            axis = axis / (l0 + 1e-12)
            l1 = float((target - other) @ axis)
            if l1 < 0.5 * l0 or l1 < MIN_LENGTH_MM:
                return "would_collapse"
            across = abs(step[0] * axis[1] - step[1] * axis[0])
            if across > max(0.1, l0 * math.tan(self.max_turn)):
                # short pieces have noisy directions: the test is whether the moved line still lies on its ink
                if self.widths_at is None or across > l0 * math.tan(math.radians(MAX_TURN_ON_INK_DEG)):
                    return "would_turn_line"
                along = other + np.outer(np.linspace(0, 1, max(4, int(l1 / 0.1))), target - other)
                if np.mean(self.widths_at(along) > 0) < LINE_ON_INK:
                    return "would_leave_ink"
        else:
            radial = abs(float(np.linalg.norm(target - e["c"])) - e["r"])
            if radial > 1e-6:
                fit = refit_arc(e, which, target)
                if fit is None or abs(fit[1] - e["r"]) > max(0.05, 0.01 * e["r"]) or \
                        np.linalg.norm(fit[0] - e["c"]) > max(0.1, 0.02 * e["r"]):
                    return "arc_refit_too_large"
            else:
                a = math.degrees(math.atan2(target[1] - e["c"][1], target[0] - e["c"][0]))
                old_a = e["a0"] if which == "start" else e["a1"]
                delta = (a - old_a + 180) % 360 - 180
                new_sweep = sweep(e) + (-delta if which == "start" else delta)
                if new_sweep < 0.5 * sweep(e) or new_sweep >= 360 or math.radians(new_sweep) * e["r"] < MIN_LENGTH_MM:
                    return "would_collapse"
        if moved > 0.1 and self.widths_at is not None:
            n = max(3, int(moved / 0.05))
            line = old + np.outer(np.linspace(0, 1, n), step)
            if np.mean(self.widths_at(line) >= 0.5 * self.thick) < self.min_ink:
                return "no_ink"
        if moved > 0.1:
            seg = LineString([tuple(old), tuple(target)])
            mine = {id(x) for x, _ in joint.members} | {id(h) for h in joint.hosts}
            for k in self.index.query(seg.buffer(0.01)):
                h = self.part[int(k)]
                if id(h) in mine:
                    continue
                g = self.geoms[int(k)]
                if seg.distance(g) <= 1e-3 and g.distance(Point(old)) > 1e-3:
                    return "would_cross"
        return None

    def apply(self, e, which, target):
        if e["kind"] == "line":
            e["p" if which == "start" else "q"] = np.asarray(target, float).copy()
        elif abs(float(np.linalg.norm(target - e["c"])) - e["r"]) > 1e-9:
            e["c"], e["r"], e["a0"], e["a1"] = refit_arc(e, which, target)
            e["refit"] = True
        else:
            a = math.degrees(math.atan2(target[1] - e["c"][1], target[0] - e["c"][0])) % 360
            e["a0" if which == "start" else "a1"] = a
        e["length_mm"] = length(e)

    # ---------------------------------------------------------------- run
    def run(self):
        joints = self.clusters()
        joints.sort(key=lambda j: bool(j.hosts))             # hosts settle first, T-junctions follow them
        for joint in joints:
            self._join(joint)
        t_joints = [j for j in joints if j.joined and j.hosts]
        for _ in range(T_PASSES):                            # a host line turned a little: follow it
            worst = 0.0
            for joint in t_joints:
                target, _reason = self.solve(joint)
                if target is None:
                    continue
                for e, w in joint.members:
                    worst = max(worst, float(np.linalg.norm(end_point(e, w) - target)))
                    self.apply(e, w, target)
                joint.target = target
            if worst < 1e-7:
                break
        return joints

    def _join(self, joint):
        duplicates = [(e, w) for e, w in joint.members for f, v in joint.members if e is not f and self._overlap(e, w, f, v)]
        for e, w in {(id(x[0]), x[1]): x for x in duplicates}.values():
            joint.refused.append((e, w, "parallel_overlap"))
        if duplicates:
            joint.members = [m for m in joint.members if not any(m[0] is d[0] and m[1] == d[1] for d in duplicates)]
        while len(joint.members) + len(joint.hosts) >= 2 and joint.members:
            target, reason = self.solve(joint)
            bad = [(self.refusal(joint, e, w, target), e, w) for e, w in joint.members]
            bad = [b for b in bad if b[0]]
            if not bad:
                for e, w in joint.members:
                    self.apply(e, w, target)
                joint.target, joint.reason, joint.joined = target, reason, True
                return
            worst = max(bad, key=lambda b: float(np.linalg.norm(end_point(b[1], b[2]) - target)))
            joint.refused.append((worst[1], worst[2], worst[0]))
            joint.members = [m for m in joint.members if not (m[0] is worst[1] and m[1] == worst[2])]

    def _overlap(self, e, w, f, v):
        """Two lines leaving the point in the same direction would run on top of each other (a closed slit)."""
        if e["kind"] != "line" or f["kind"] != "line":
            return False
        u1, u2 = into(e, w), into(f, v)
        return u1 @ u2 > 0 and self._parallel(u1, u2) and float(np.linalg.norm(end_point(e, w) - end_point(f, v))) > PARALLEL_MIN_ACROSS_MM

    def rollback(self, joint):
        for e, w in joint.members:
            s = self.saved[id(e)]
            if e.get("refit"):
                e.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()})
                e.pop("refit", None)
            elif e["kind"] == "line":
                e["p" if w == "start" else "q"] = s["p" if w == "start" else "q"].copy()
            else:
                e["a0" if w == "start" else "a1"] = s["a0" if w == "start" else "a1"]
            e["length_mm"] = length(e)
        joint.joined = False

    # ---------------------------------------------------------------- report
    def record(self, joints):
        """closures[] for cv.json and adjustments[] on the moved primitives."""
        out = []
        for joint in joints:
            ats = [end_point(e, w) for e, w in joint.members] + [end_point(e, w) for e, w, _ in joint.refused]
            if not ats:
                continue
            at = joint.target if joint.joined else np.mean(ats, axis=0)
            item = {"status": "joined" if joint.joined else "refused", "at_mm": _r(at), "tolerance_mm": round(self.tol, 4),
                    "reason": joint.reason if joint.joined else joint.refused[0][2],
                    "members": [], "hosts": [h["id"] for h in joint.hosts], "refused": [
                        {"object_id": e["id"], "end": w, "at_mm": _r(end_point(e, w)), "reason": why} for e, w, why in joint.refused]}
            if joint.joined:
                for e, w in joint.members:
                    move = self._move(e, w)
                    item["members"].append({"object_id": e["id"], "end": w, **move})
            item["max_moved_mm"] = max([m["moved_mm"] for m in item["members"]], default=0.0)
            out.append(item)
        out.sort(key=lambda c: (c["at_mm"], c["status"]))
        for k, c in enumerate(out):
            c["id"] = f"closure_{k}"
            for m in c["members"]:
                if m["moved_mm"] <= 1e-6:
                    continue
                obj = next(e for e in self.part if e["id"] == m["object_id"])
                obj.setdefault("adjustments", []).append({**{key: m[key] for key in m if key != "object_id"},
                                                          "reason": c["reason"], "closure_id": c["id"]})
        return out

    def _move(self, e, w):
        old = dict(self.saved[id(e)])
        prev = end_point({**e, **old}, w)
        new = end_point(e, w)
        step = new - prev
        moved = float(np.linalg.norm(step))
        d = into({**e, **old}, w)
        along = float(abs(step @ d))
        across = float(abs(step[0] * d[1] - step[1] * d[0]))
        return {"from_mm": _r(prev), "to_mm": _r(new), "moved_mm": round(moved, 6),
                "along_mm": round(along, 6), "across_mm": round(across, 6)}


def _r(p):
    return [round(float(p[0]), 6), round(float(p[1]), 6)]


def close_contours(part, others, snap_tol, gap_search, widths_at=None, thick_mm=None, max_turn_deg=3.0, min_ink=0.75):
    """Move part ends to common points, rebuild the contours exactly, verify. Returns the topology result with
    "closures" and "closure_checks" added. Joins that shrink a hole or create a new crossing are undone."""
    before = build(part, others, snap_tol, gap_search, strict_rings=False)
    worker = Reconciler(part, snap_tol, widths_at, thick_mm, max_turn_deg, min_ink)
    joints = worker.run()
    topo = build(part, others, COINCIDENT_MM, gap_search)
    undone = _verify(joints, before, topo, worker)
    if undone:
        topo = build(part, others, COINCIDENT_MM, gap_search)
    for joint in joints:
        if joint.undone_reason:
            joint.refused += [(e, w, joint.undone_reason) for e, w in joint.members]
    closures = worker.record(joints)
    topo["closures"] = closures
    edges_ok = True
    residuals = []
    for c in closures:
        if c["status"] != "joined":
            continue
        ends = np.array([m["to_mm"] for m in c["members"]])
        residual = float(np.max(np.linalg.norm(ends - ends.mean(axis=0), axis=1)))
        c["residual_mm"] = round(residual, 7)
        residuals.append(residual)
        edges_ok &= residual <= 1e-4
    legacy = [c for c in before["contours"] if c["closed"]]
    topo["closure_checks"] = {
        "edges_meet": bool(edges_ok),
        "max_residual_mm": round(max(residuals, default=0.0), 7),
        "joined": sum(c["status"] == "joined" for c in closures),
        "refused": sum(bool(c["refused"]) for c in closures),
        "undone": undone,
        "closed_by_tolerance_only": len(legacy),
        "closed_by_tolerance_with_real_gap": sum(c["closure_error_mm"] > 1e-4 for c in legacy),
        "closed_in_coordinates": sum(c["closed"] for c in topo["contours"]),
    }
    return topo


def _chord_area(points):
    x, y = np.asarray(points, float).T
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _verify(joints, before, after, worker):
    """Undo joints that close up a contour or create a crossing that was not there. The vertices of each contour of
    the tolerance view are moved exactly as the joints moved them: this measures the moves alone, whatever a refused
    join or a split into regions did to the final contours."""
    undone = 0
    joined = [j for j in joints if j.joined]
    if joined:
        old = np.array([end_point({**e, **worker.saved[id(e)]}, w) for j in joined for e, w in j.members])
        owner = [j for j in joined for _ in j.members]
        tree = cKDTree(old)
        for b in before["contours"]:
            if not b["closed"] or b["role"] == "circle" or len(b["edges"]) < 3:
                continue
            ring = np.array([e["start"] for e in b["edges"]], float)
            area0 = _chord_area(ring)
            # the polygon of the vertices must describe the contour: not a ring with bridges or mostly of arcs
            if area0 <= 0.5 or not b["_simple"] or abs(area0 - b["area_mm2"]) > 0.3 * b["area_mm2"]:
                continue
            dist, idx = tree.query(ring)
            hit = np.nonzero(dist <= worker.tol)[0]
            moved = {int(k): owner[idx[k]] for k in hit}
            while moved:
                now = ring.copy()
                for k, j in moved.items():
                    now[k] = j.target
                if _chord_area(now) >= HOLE_MIN_AREA_RATIO * area0:
                    break
                # the joint that takes the contour's area most: undo it alone and measure again
                worst = None
                for j in {id(j): j for j in moved.values()}.values():
                    test = now.copy()
                    for k, other in moved.items():
                        if other is j:
                            test[k] = ring[k]
                    gain = _chord_area(test)
                    if worst is None or gain > worst[0]:
                        worst = (gain, j)
                j = worst[1]
                if j.joined:
                    j.undone_reason = "would_close_contour"
                    worker.rollback(j)
                    undone += 1
                moved = {k: o for k, o in moved.items() if o is not j}
    near = worker.tol * REACH + 0.1
    known = [np.asarray(d["at_mm"]) for d in before["defects"] if d["code"] == "self_intersection"]
    for d in after["defects"]:
        if d["code"] != "self_intersection" or any(np.linalg.norm(k - d["at_mm"]) < 1.0 for k in known):
            continue
        ids = set(d["object_ids"])
        for j in joined:
            if j.joined and ids & {e["id"] for e, _ in j.members} and np.linalg.norm(j.target - d["at_mm"]) <= 2 * near:
                j.undone_reason = "would_cross"
                worker.rollback(j)
                undone += 1
    return undone
