"""Reference points for OCR: ids, what each point means, coordinates after rotation and crop."""
import cv2
import numpy as np
import pytest

from drawcv import Params, run
from drawcv.topology import COINCIDENT_MM
from drawcv.transform import Transform


def anchor_xy(cv, ref):
    p = next(x for x in cv["primitives"] if x["id"] == ref["object_id"])
    a = ref["anchor"]
    if p["kind"] == "line":
        return {"start": np.array(p["start"]), "end": np.array(p["end"]),
                "mid": (np.array(p["start"]) + p["end"]) / 2}[a]
    if a == "center":
        return np.array(p["center"])
    angle = p["start_angle"] if a == "start" else p["end_angle"] if a == "end" else float(a.split("_")[1])
    return np.array(p["center"]) + p["radius"] * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])


@pytest.mark.parametrize("name", ["control.png", "control.pdf", "control_scan.jpg"])
def test_one_point_is_never_given_to_objects_that_stay_apart(results, name):
    cv, _ = results(name)
    assert cv["points"]
    for p in cv["points"]:
        where = [anchor_xy(cv, r) for r in p["refs"]]
        for a in where:
            assert np.linalg.norm(a - p["xy"]) <= 2 * COINCIDENT_MM, (p["id"], p["refs"])


def test_each_point_explains_each_of_its_links(results):
    cv, _ = results("control.png")
    assert set(cv["point_info"]) == {p["id"] for p in cv["points"]}
    words = " ".join(i["meaning"] for i in cv["point_info"].values())
    assert "начало отрезка" in words and "конец отрезка" in words and "центр отверстия" in words and "вершина контура" in words
    for p in cv["points"]:
        info = cv["point_info"][p["id"]]
        assert [(x["object_id"], x["anchor"]) for x in info["links"]] == [(r["object_id"], r["anchor"]) for r in p["refs"]]
        assert all(x["meaning"] and x["object_id"] in x["meaning"] for x in info["links"])


def test_moved_point_keeps_the_observation(results):
    cv, _ = results("control.png")
    moved = [link for i in cv["point_info"].values() for link in i["links"] if "moved_mm" in link]
    assert moved
    for link in moved:
        assert len(link["observed_mm"]) == 2 and link["move_reason"]
        adj = next(a for a in cv["objects"][link["object_id"]]["adjustments"] if a["end"] == link["anchor"])
        assert adj["from_mm"] == link["observed_mm"]
    obj = next(o for o in cv["objects"].values() if "adjustments" in o)
    assert obj["observed_geometry_mm"] != obj["geometry_mm"]


def test_points_map_back_to_ink_of_the_source(results, synth_dir):
    cv, _ = results("control_scan.jpg")
    src = cv2.imdecode(np.fromfile(str(synth_dir / "control_scan.jpg"), np.uint8), cv2.IMREAD_GRAYSCALE)
    to_src = Transform(cv["transforms"]["sheet_mm_to_source_px"])
    dark = []
    for p in cv["points"]:
        info = cv["point_info"][p["id"]]
        assert np.allclose(info["source_px"], to_src.apply(np.array(p["xy"])), atol=0.01)
        if p["kind"] in ("junction", "end"):
            x, y = np.round(info["source_px"]).astype(int)
            dark.append(src[max(0, y - 4):y + 5, max(0, x - 4):x + 5].min() < 128)
    assert np.mean(dark) > 0.95


@pytest.fixture(scope="module")
def variants(synth_dir, tmp_path_factory):
    """The control sheet as a file rotated by 90 degrees and as a cropped file."""
    d = tmp_path_factory.mktemp("variants")
    img = cv2.imdecode(np.fromfile(str(synth_dir / "control.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    cv2.imencode(".png", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))[1].tofile(str(d / "rotated.png"))
    cv2.imencode(".png", img[20:-10, 25:-15])[1].tofile(str(d / "cropped.png"))
    p = Params(save_debug_images=False)
    return {"img": img, "base": run(synth_dir / "control.png", None, p),
            "rotated": run(d / "rotated.png", None, Params(save_debug_images=False, rotate_deg=90)),
            "cropped": run(d / "cropped.png", None, p)}


def circles(cv):
    return sorted(np.array(o["geometry_mm"]["center"]).tolist() for o in cv["objects"].values()
                  if o["kind"] == "circle" and o["status"] == "ok" and o["role"] == "part")


@pytest.mark.parametrize("name", ["rotated", "cropped"])
def test_sheet_coordinates_survive_rotation_and_crop(variants, name):
    assert variants[name]["scale_status"] == "known"
    a, b = circles(variants["base"]), circles(variants[name])
    assert len(a) == len(b)
    assert np.allclose(a, b, atol=0.3)


def test_ids_of_a_rotated_file_are_the_same_objects(variants):
    base = {p["id"]: p for p in variants["base"]["primitives"]}
    rotated = {p["id"]: p for p in variants["rotated"]["primitives"]}
    assert set(base) == set(rotated)
    for i, p in base.items():
        q = rotated[i]
        assert p["kind"] == q["kind"]
        if p["kind"] == "line":
            assert np.allclose(p["start"], q["start"], atol=0.3) and np.allclose(p["end"], q["end"], atol=0.3)


def test_cropped_points_follow_the_crop_in_source_pixels(variants):
    """Same sheet point, source pixels shifted exactly by the crop."""
    base, cropped = variants["base"], variants["cropped"]
    to_a, to_b = Transform(base["transforms"]["sheet_mm_to_source_px"]), Transform(cropped["transforms"]["sheet_mm_to_source_px"])
    centre = np.array(circles(base)[0])
    near = min(circles(cropped), key=lambda c: np.linalg.norm(np.array(c) - centre))
    shift = to_a.apply(centre) - to_b.apply(np.array(near))
    assert np.allclose(shift, [25, 20], atol=2.0)


def test_measurable_sections_name_real_points(results):
    cv, _ = results("control.png")
    ids = {p["id"] for p in cv["points"]}
    kinds = {m["kind"] for m in cv["measurable"]}
    assert {"line_length", "circle_diameter", "arc_radius", "parallel_distance"} <= kinds
    assert len({m["id"] for m in cv["measurable"]}) == len(cv["measurable"])
    for m in cv["measurable"]:
        assert m["value_mm"] > 0
        for key in ("p1", "p2", "center"):
            if key in m:
                assert m[key] in ids
    holes = [o["geometry_mm"]["radius"] * 2 for o in cv["objects"].values() if o["kind"] == "circle" and o["status"] == "ok" and o["role"] == "part"]
    diameters = [m["value_mm"] for m in cv["measurable"] if m["kind"] == "circle_diameter"]
    assert np.allclose(sorted(holes), sorted(diameters), atol=0.001)
