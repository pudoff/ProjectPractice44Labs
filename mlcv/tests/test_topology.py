"""Contour assembly on hand-made primitives (no raster)."""
import numpy as np

from drawcv.topology import build


def line(i, p, q):
    return {"id": i, "kind": "line", "p": np.array(p, float), "q": np.array(q, float), "role": "part"}


def arc(i, c, r, a0, a1):
    return {"id": i, "kind": "arc", "c": np.array(c, float), "r": r, "a0": a0, "a1": a1, "role": "part"}


def square(dx=0.0, gap=0.0):
    return [line("a", (0, 0), (10, 0)), line("b", (10, 0), (10, 10)), line("c", (10, 10), (gap, 10)),
            line("d", (0, 10 - dx), (0, 0))]


def test_closed_square_within_tolerance():
    prims = square()
    prims[1]["q"] = np.array([10.05, 10.04])           # ends off by 0.06 mm: joined at tolerance 0.3
    r = build(prims, [], 0.3, 3.0)
    closed = [c for c in r["contours"] if c["closed"]]
    assert len(closed) == 1 and abs(closed[0]["area_mm2"] - 100) < 1
    assert closed[0]["closure_error_mm"] <= 0.3
    assert not [d for d in r["defects"] if d["code"] == "gap"]


def test_gap_is_reported_not_closed():
    r = build(square(gap=1.0), [], 0.3, 3.0)
    assert not [c for c in r["contours"] if c["closed"]]
    gaps = [d for d in r["defects"] if d["code"] == "gap"]
    assert gaps and abs(gaps[0]["distance_mm"] - 1.0) < 0.05


def test_t_junction_splits_edge():
    prims = square() + [line("e", (5, 0), (5, 10))]
    r = build(prims, [], 0.3, 3.0)
    regions = [c for c in r["contours"] if c["role"] == "region"]
    assert len(regions) == 2 and all(abs(c["area_mm2"] - 50) < 1 for c in regions)


def test_crossing_is_self_intersection():
    prims = [line("a", (0, 0), (10, 10)), line("b", (10, 10), (10, 0)), line("c", (10, 0), (0, 10)),
             line("d", (0, 10), (0, 0))]
    r = build(prims, [], 0.3, 3.0)
    si = [d for d in r["defects"] if d["code"] == "self_intersection"]
    assert si and np.allclose(si[0]["at_mm"], [5, 5], atol=0.01)


def test_arcs_stay_arcs():
    prims = [line("a", (0, 0), (10, 0)), arc("b", (10, 5), 5, 270, 90), line("c", (10, 10), (0, 10)),
             line("d", (0, 10), (0, 0))]
    r = build(prims, [], 0.3, 3.0)
    closed = [c for c in r["contours"] if c["closed"]][0]
    kinds = {e["object_id"]: e["kind"] for e in closed["edges"]}
    assert kinds["b"] == "arc"
    arc_edge = next(e for e in closed["edges"] if e["object_id"] == "b")
    assert arc_edge["radius"] == 5 and "center" in arc_edge
    assert abs(closed["area_mm2"] - (100 + np.pi * 25 / 2)) < 0.5


def test_other_geometry_is_not_joined():
    axis = {"id": "x", "kind": "line", "p": np.array([10.2, 5.0]), "q": np.array([20.0, 5.0]), "role": "axis"}
    r = build(square(gap=1.0), [axis], 0.3, 3.0)
    assert all("x" not in [e["object_id"] for e in c["edges"]] for c in r["contours"])
