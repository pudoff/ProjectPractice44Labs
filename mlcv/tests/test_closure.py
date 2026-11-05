"""Joined ends get one common point or stay apart; every move and refusal is recorded (hand-made primitives)."""
import math

import numpy as np
import pytest

from drawcv.closure import Joint, Reconciler, close_contours, end_point, refit_arc
from drawcv.topology import COINCIDENT_MM, build


def line(i, p, q):
    return {"id": i, "kind": "line", "p": np.array(p, float), "q": np.array(q, float), "role": "part"}


def arc(i, c, r, a0, a1):
    return {"id": i, "kind": "arc", "c": np.array(c, float), "r": r, "a0": a0, "a1": a1, "role": "part"}


def circle(i, c, r):
    return {"id": i, "kind": "circle", "c": np.array(c, float), "r": r, "role": "part"}


def ends_of(edges):
    return [(np.array(e["end"]), np.array(f["start"])) for e, f in zip(edges, edges[1:] + edges[:1])]


def closed(topo):
    return [c for c in topo["contours"] if c["closed"]]


def test_corner_moves_to_the_intersection_of_the_lines():
    """Ends 0.1 mm off: lines extend along themselves, so their directions stay and the contour really closes."""
    prims = [line("a", (0, 0), (9.9, 0)), line("b", (10, 0.1), (10, 10)), line("c", (10, 10), (0, 10)), line("d", (0, 10), (0, 0))]
    topo = close_contours(prims, [], 0.3, 3.0)
    assert np.allclose(prims[0]["q"], [10, 0]) and np.allclose(prims[1]["p"], [10, 0])
    ring = closed(topo)[0]
    assert ring["closure_error_mm"] <= COINCIDENT_MM and ring["area_mm2"] == pytest.approx(100, abs=0.01)
    for a, b in ends_of(ring["edges"]):
        assert np.linalg.norm(a - b) <= 1e-4
    assert topo["closure_checks"]["edges_meet"]


def test_observation_move_and_reason_are_kept():
    prims = [line("a", (0, 0), (9.9, 0)), line("b", (10, 0.1), (10, 10)), line("c", (10, 10), (0, 10)), line("d", (0, 10), (0, 0))]
    topo = close_contours(prims, [], 0.3, 3.0)
    moved = prims[0]["adjustments"][0]
    assert moved["end"] == "end" and moved["from_mm"] == [9.9, 0.0] and moved["to_mm"] == [10.0, 0.0]
    assert moved["moved_mm"] == pytest.approx(0.1) and moved["along_mm"] == pytest.approx(0.1) and moved["across_mm"] == pytest.approx(0)
    assert moved["reason"] == "corner_intersection"
    closure = next(c for c in topo["closures"] if c["id"] == moved["closure_id"])
    assert closure["status"] == "joined" and {m["object_id"] for m in closure["members"]} == {"a", "b"}


def test_real_white_gap_stays_open_with_a_reason():
    """Within tolerance but on white paper: the old behaviour closed it silently."""
    prims = [line("a", (0, 0), (9.6, 0)), line("b", (10, 0), (10, 10)), line("c", (10, 10), (0, 10)), line("d", (0, 10), (0, 0))]
    paper = lambda pts: np.where(np.asarray(pts)[:, 0] > 9.6, 0.0, 0.5)   # no ink over the last 0.4 mm of the bottom edge
    topo = close_contours(prims, [], 0.5, 3.0, widths_at=paper, thick_mm=0.5)
    assert not closed(topo)
    assert np.allclose(prims[0]["q"], [9.6, 0]), "the end was not moved"
    refused = [r for c in topo["closures"] for r in c["refused"]]
    assert refused and refused[0]["reason"] == "no_ink"
    gap = [d for d in topo["defects"] if d["code"] == "gap"]
    assert gap and gap[0]["distance_mm"] == pytest.approx(0.4, abs=0.1)


def test_t_junction_end_goes_on_the_host_and_the_host_does_not_move():
    bottom = line("a", (0, 0), (10, 0))
    prims = [bottom, line("b", (10, 0), (10, 10)), line("c", (10, 10), (0, 10)), line("d", (0, 10), (0, 0)), line("e", (5, 0.2), (5, 10))]
    topo = close_contours(prims, [], 0.3, 3.0)
    assert np.allclose(prims[4]["p"], [5, 0]) and np.allclose(bottom["p"], [0, 0]) and np.allclose(bottom["q"], [10, 0])
    regions = [c for c in topo["contours"] if c["role"] == "region"]
    assert len(regions) == 2 and all(c["area_mm2"] == pytest.approx(50, abs=0.01) for c in regions)
    joint = next(c for c in topo["closures"] if c["reason"] == "t_junction")
    assert joint["hosts"] == ["a"]


def test_parallel_lines_side_by_side_are_not_joined():
    """Two lines running next to each other within the tolerance: a slit, not a corner."""
    a, b = line("a", (0, 0), (10, 0)), line("b", (0, 0.4), (10, 0.4))
    topo = close_contours([a, b], [], 0.6, 3.0)
    assert np.allclose(a["p"], [0, 0]) and np.allclose(b["p"], [0, 0.4])
    assert not closed(topo)
    assert any(r["reason"] == "parallel_overlap" for c in topo["closures"] for r in c["refused"])


def test_both_ends_of_a_short_object_stay_separate():
    """A 1 mm chamfer shorter than the tolerance chain must not collapse into one point."""
    prims = [line("a", (0, 0), (10, 0)), line("ch", (10, 0), (10.7, 0.7)), line("b", (10.7, 0.7), (10.7, 10)),
             line("c", (10.7, 10), (0, 10)), line("d", (0, 10), (0, 0))]
    topo = close_contours(prims, [], 0.8, 3.0)
    assert np.allclose(prims[1]["p"], [10, 0]) and np.allclose(prims[1]["q"], [10.7, 0.7])
    assert {e["object_id"] for e in closed(topo)[0]["edges"]} == {"a", "ch", "b", "c", "d"}


def test_a_move_across_another_object_is_refused():
    a, b, c = line("a", (0, 0), (5, 0)), line("b", (5, 0.6), (5, 10)), line("c", (4, 0.3), (6, 0.3))
    worker = Reconciler([a, b, c], 0.8)
    assert worker.refusal(Joint([(a, "end"), (b, "start")]), b, "start", np.array([5.0, 0.0])) == "would_cross"


def test_a_join_that_closes_up_a_contour_is_undone():
    """The moves are forced inward: the check measures the moved vertices and restores the observation."""
    prims = [line("a", (0, 0), (2, 0)), line("b", (2, 0), (2, 2)), line("c", (2, 2), (0, 2)), line("d", (0, 2), (0, 0))]
    from drawcv.closure import _verify

    before = build(prims, [], 0.8, 3.0, strict_rings=False)
    worker = Reconciler(prims, 0.8)
    joints = worker.run()
    joint = next(j for j in joints if np.allclose(j.target, [2, 2]))
    joint.target = np.array([1.2, 1.2])
    for e, w in joint.members:
        worker.apply(e, w, joint.target)
    after = build(prims, [], COINCIDENT_MM, 3.0)
    assert _verify(joints, before, after, worker) >= 1
    assert joint.undone_reason == "would_close_contour"
    assert np.allclose(prims[1]["q"], [2, 2]) and np.allclose(prims[2]["p"], [2, 2])


def test_arc_keeps_centre_and_radius_when_it_meets_a_line():
    prims = [line("a", (0, 0), (10, 0)), arc("b", (10, 5), 5, 272, 88), line("c", (10, 10), (0, 10)), line("d", (0, 10), (0, 0))]
    topo = close_contours(prims, [], 0.6, 3.0)
    assert prims[1]["r"] == 5 and np.allclose(prims[1]["c"], [10, 5])
    ring = closed(topo)[0]
    assert next(e for e in ring["edges"] if e["object_id"] == "b")["kind"] == "arc"
    for a, b in ends_of(ring["edges"]):
        assert np.linalg.norm(a - b) <= 1e-4


def test_touching_arcs_without_an_intersection_keep_their_radius():
    a = arc("a", (0, 0), 5, 0, 90)
    fit = refit_arc(a, "end", np.array([0.02, 5.03]))
    assert fit[1] == pytest.approx(5) and np.linalg.norm(fit[0]) < 0.05
    assert (fit[2] + 180) % 360 - 180 == pytest.approx(0, abs=0.5) and fit[3] == pytest.approx(90, abs=0.5)
    moved = dict(a, c=fit[0], r=fit[1], a0=fit[2], a1=fit[3])
    assert np.allclose(end_point(moved, "end"), [0.02, 5.03]) and np.allclose(end_point(moved, "start"), [5, 0])


def test_a_hole_stays_a_hole():
    prims = [line("a", (0, 0), (20, 0)), line("b", (20, 0), (20, 20)), line("c", (20, 20), (0, 20)), line("d", (0, 20), (0, 0)),
             circle("h", (10, 10), 1.0)]
    topo = close_contours(prims, [], 0.8, 3.0)
    roles = {c["role"]: c for c in topo["contours"]}
    assert roles["circle"]["area_mm2"] == pytest.approx(math.pi, abs=0.05)
    assert roles["circle"]["parent"]["role"] == "outer_boundary"


def test_ring_of_no_area_never_gives_nan():
    """It used to reach cv.json as bbox NaN, and the backend then rejected the whole sheet."""
    prims = [line("a", (0, 0), (10, 0)), line("b", (10, 0), (5, 0)), arc("c", (2.5, 0), 2.5, 0, 180)]
    prims[2]["a0"], prims[2]["a1"] = 180, 0
    topo = build(prims, [], 0.05, 1.0)
    for c in topo["contours"]:
        assert all(math.isfinite(v) for v in c["bbox_mm"])


def test_tolerance_view_reports_the_real_distance():
    """closure_error_mm is what the coordinates say, not the spread of a tolerance cluster."""
    prims = [line("a", (0, 0), (9.9, 0)), line("b", (10, 0.1), (10, 10)), line("c", (10, 10), (0, 10)), line("d", (0, 10), (0, 0))]
    ring = closed(build(prims, [], 0.3, 3.0))[0]
    assert ring["closure_error_mm"] == pytest.approx(math.hypot(0.1, 0.1), abs=1e-6)
