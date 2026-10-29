"""End-to-end checks on the synthetic control sheet (exact truth) in all input formats."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from drawcv import Params, run
from drawcv.transform import Transform


def part_circles(cv):
    return [o["geometry_mm"] for o in cv["objects"].values()
            if o["kind"] == "circle" and o["status"] == "ok" and o["role"] == "part"]


def nearest(items, centre):
    return min(items, key=lambda g: np.linalg.norm(np.array(g["center"]) - centre))


@pytest.mark.parametrize("name", ["control.png", "control.jpg", "control.pdf"])
def test_formats_same_result(results, truth_data, name):
    cv, _ = results(name)
    assert cv["status"] == "ok"
    assert cv["scale_status"] == "known"
    assert cv["calibration"]["px_per_mm"][0] == pytest.approx(300 / 25.4, rel=0.01)
    for c in truth_data["circles"]:
        g = nearest(part_circles(cv), c["center"])
        assert np.linalg.norm(np.array(g["center"]) - c["center"]) < 0.15
        assert abs(g["radius"] - c["radius"]) < 0.05


def test_formats_agree_with_each_other(results):
    a, _ = results("control.png")
    b, _ = results("control.jpg")
    c, _ = results("control.pdf")
    ca, cb, cc = part_circles(a), part_circles(b), part_circles(c)
    assert len(ca) == len(cb) == len(cc)
    for g in ca:
        for other in (cb, cc):
            h = nearest(other, g["center"])
            assert np.linalg.norm(np.array(g["center"]) - h["center"]) < 0.1


def test_rotated_scan_coordinates(results, truth_data, synth_dir):
    """1.2 deg rotated, blurred, noisy scan: deskew is undone, coordinates stay in sheet mm, and objects mapped
    back through sheet_mm -> source_px land on the ink of the original (rotated) image."""
    cv, _ = results("control_scan.jpg")
    assert abs(abs(cv["preprocess"]["skew_deg"]) - 1.2) < 0.15
    for c in truth_data["circles"]:
        g = nearest(part_circles(cv), c["center"])
        assert np.linalg.norm(np.array(g["center"]) - c["center"]) < 0.3
        assert abs(g["radius"] - c["radius"]) < 0.1
    src = cv2.imdecode(np.fromfile(str(synth_dir / "control_scan.jpg"), np.uint8), cv2.IMREAD_GRAYSCALE)
    to_src = Transform(cv["transforms"]["sheet_mm_to_source_px"])
    for c in truth_data["circles"]:
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        pts = np.array(c["center"]) + c["radius"] * np.column_stack([np.cos(ang), np.sin(ang)])
        px = np.round(to_src.apply(pts)).astype(int)
        dark = [src[max(0, y - 2):y + 3, max(0, x - 2):x + 3].min() < 128 for x, y in px]
        assert np.mean(dark) > 0.95


def test_transforms_are_consistent(results):
    cv, _ = results("control.png")
    t = cv["transforms"]
    a = Transform(t["source_px_to_work_px"]).then(Transform(t["work_px_to_sheet_mm"]))
    assert np.allclose(a.m, np.array(t["source_px_to_sheet_mm"]), atol=1e-6)
    assert np.allclose(Transform(t["sheet_mm_to_source_px"]).m, a.inverse().m, atol=1e-6)


def test_source_untouched_and_text_reviewable(results, synth_dir):
    before = hashlib.sha256((synth_dir / "control.png").read_bytes()).hexdigest()
    cv, out = results("control.png")
    assert cv["source"]["sha256"] == before
    assert hashlib.sha256((synth_dir / "control.png").read_bytes()).hexdigest() == before
    assert cv["text_regions"], "labels must be found"
    for r in cv["text_regions"]:
        assert len(r["poly_source_px"]) >= 4 and len(r["poly_mm"]) >= 4
    assert (out / "prep_compare.png").exists() and (out / "overlay.png").exists() and (out / "geometry.png").exists()
    # the 2.5 mm label "0,5" (comma = small dot) must not be removed as a speck
    assert cv["preprocess"]["specks_removed"] == 0


def test_primitives_have_type_coordinates_confidence(results):
    cv, _ = results("control.png")
    kinds = {p["kind"] for p in cv["primitives"]}
    assert {"line", "circle", "arc"} <= kinds
    for oid, o in cv["objects"].items():
        assert 0.0 <= o["confidence"] <= 1.0
        assert o["role"] and o["status"] in ("ok", "excluded", "unsupported", "converted")
        assert "geometry_mm" in o and "geometry_source_px" in o
    noise = sum(1 for o in cv["objects"].values() if o["role"] == "noise")
    assert cv["summary"]["by_kind_role_status"].get("line:noise:excluded", 0) + \
        cv["summary"]["by_kind_role_status"].get("arc:noise:excluded", 0) == noise


def test_rounded_corner_is_an_arc(results, truth_data):
    cv, _ = results("control.png")
    t = truth_data["arcs"][0]
    arcs = [o["geometry_mm"] for o in cv["objects"].values() if o["kind"] == "arc" and o["status"] == "ok" and o["role"] == "part"]
    g = nearest(arcs, t["center"])
    assert np.linalg.norm(np.array(g["center"]) - t["center"]) < 0.2
    assert abs(g["radius"] - t["radius"]) < 0.1


@pytest.mark.parametrize("name", ["control.png", "control.jpg", "control.pdf", "control_scan.jpg"])
def test_profiles_gaps_crossings(results, truth_data, name):
    cv, _ = results(name)
    closed = [c for c in cv["contours"] if c["closed"]]
    plate = truth_data["closed_profiles"][0]
    hit = [c for c in closed if np.allclose(c["bbox_mm"], plate["bbox"], atol=1.5)]
    assert hit, "plate profile must be closed"
    assert abs(hit[0]["area_mm2"] - plate["area_mm2"]) < 0.01 * plate["area_mm2"]
    assert hit[0]["closure_error_mm"] <= cv["stats"]["tolerances_mm"]["snap"]
    gap = truth_data["gaps"][0]
    gaps = [d for d in cv["defects"] if d["code"] == "gap"
            and min(np.linalg.norm(np.array(d["at_mm"]) - p) for p in gap["between"]) < 2.0]
    assert gaps and 1.0 < gaps[0]["distance_mm"] < 2.5
    si = truth_data["self_intersections"][0]
    assert any(d["code"] == "self_intersection" and np.linalg.norm(np.array(d["at_mm"]) - si["at"]) < 2.0
               for d in cv["defects"])
    # the broken profile is not closed silently
    x0, y0 = 40.0, 110.0
    assert not any(np.allclose(c["bbox_mm"][:2], [x0, y0], atol=1.5) and np.allclose(c["bbox_mm"][2:], [90, 140], atol=1.5)
                   for c in closed)


@pytest.mark.parametrize("name", ["control.png", "control.jpg", "control.pdf"])
def test_axes_and_text_not_joined(results, name):
    cv, _ = results(name)
    for c in cv["contours"]:
        for oid in c["object_ids"]:
            assert cv["objects"][oid]["role"] == "part"
    assert any(o["role"] == "axis" for o in cv["objects"].values())


@pytest.mark.parametrize("name", ["control.png", "control.jpg", "control.pdf"])
def test_simple_hatch_boundary(results, truth_data, name):
    cv, _ = results(name)
    assert len(cv["hatches"]) == 1
    h = cv["hatches"][0]
    xs, ys = [p[0] for p in h["outer"]], [p[1] for p in h["outer"]]
    assert np.allclose([min(xs), min(ys), max(xs), max(ys)], truth_data["hatch"]["bbox"], atol=0.8)
    group = cv["hatch_groups"][0]
    assert group["spacing_mm"] == pytest.approx(truth_data["hatch"]["spacing_mm"], abs=0.1)
    assert group["angle_deg"] == pytest.approx(truth_data["hatch"]["angle_deg"], abs=1.0)


@pytest.mark.parametrize("name,scale", [("control_print_9948.pdf", 0.9948), ("control_print_102.pdf", 1.02)])
def test_pdf_printed_scaled(results, truth_data, name, scale):
    """PDF printed with "fit to page": calibration on the drawn frame removes any print percentage."""
    cv, _ = results(name)
    assert cv["scale_status"] == "known"
    assert cv["calibration"]["method"] != "pdf_page"
    for c in truth_data["circles"]:
        g = nearest(part_circles(cv), c["center"])
        assert abs(g["radius"] - c["radius"]) < 0.06
    check = cv["calibration"].get("pdf_page_check")
    assert check and check["content_scale_vs_page"] == pytest.approx(1 / scale, abs=0.004)
    assert any(i["code"] == "pdf_content_scaled" for i in cv["issues"])


def test_pdf_uneven_print_calibrated_per_axis(results, truth_data):
    """Printed 99 % x 100 %: the title block confirms the frame, x and y keep their own scales, a warning is issued."""
    cv, _ = results("control_print_aniso1.pdf")
    assert cv["scale_status"] == "known"
    assert cv["calibration"]["anisotropy"] == pytest.approx(-0.01, abs=0.002)
    assert any(i["code"] == "anisotropic_scale" for i in cv["issues"])
    for c in truth_data["circles"]:
        g = nearest(part_circles(cv), c["center"])
        assert np.linalg.norm(np.array(g["center"]) - c["center"]) < 0.2


def test_pdf_strongly_uneven_print_is_not_trusted(results):
    """97 % x 100 % is beyond any plausible printer error: the scale is reported unknown, not guessed."""
    cv, _ = results("control_print_aniso.pdf")
    assert cv["scale_status"] == "unknown"
    assert any(i["severity"] == "error" for i in cv["issues"])


def test_manual_calibration_is_marked(synth_dir, tmp_path):
    p = Params(manual_px_per_mm=300 / 25.4, save_debug_images=False, text_provider="none")
    cv = run(synth_dir / "control.png", tmp_path, p)
    assert cv["calibration"]["manual"] is True and cv["calibration"]["method"] == "manual"
    assert any(i["code"] == "manual_calibration" for i in cv["issues"])


def test_deterministic(synth_dir, tmp_path):
    p = Params(save_debug_images=False)
    a = run(synth_dir / "control.png", tmp_path / "a", p)
    b = run(synth_dir / "control.png", tmp_path / "b", p)
    for cv in (a, b):
        cv.pop("timings_s")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
