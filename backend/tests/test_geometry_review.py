"""Regressions for review CR-02..07: exercise accepted graphs and read real DXF."""
import copy
import math

import ezdxf
from PIL import Image
import pytest

from app.domain.drawing import DrawingGraph
from app.services.dxf import DxfExporter
from app.services.image import ImageService
from app.services.recognition import assemble_graph


def graph(**fields):
    return DrawingGraph.model_validate({"coordinate_space": "part_mm", **fields})


def line(id="edge", end=(46, 0), **extra):
    return dict(id=id, kind="line", start=(0, 0), end=end, **extra)


@pytest.mark.parametrize("primitives,hatches,code", [
    ([line(end=(0, 0))], [], "invalid_primitive"),
    ([dict(id="arc", kind="arc", center=[0, 0], radius=5, start_angle=0, end_angle=360)], [], "invalid_primitive"),
    ([dict(id="bowtie", kind="polyline", closed=True, vertices=[[0, 0], [10, 10], [0, 10], [10, 0]])], [], "invalid_primitive"),
    ([], [dict(id="h", outer=[[0, 0], [10, 0], [10, 10], [0, 10]],
               holes=[[[20, 20], [21, 20], [21, 21], [20, 21]]])], "invalid_hatch_hole"),
    ([], [dict(id="h", outer=[[0, 0], [10, 10], [0, 10], [10, 0]])], "invalid_boundary"),
    ([], [dict(id="h", outer=[[0, 0], [10, 0], [10, 10], [0, 10]],
               holes=[[[2, 2], [5, 2], [5, 5], [2, 5]], [[4, 4], [6, 4], [6, 6], [4, 6]]])], "invalid_hatch_hole"),
])
def test_invalid_geometry_cannot_be_exported_as_success(tmp_path, primitives, hatches, code):
    path = tmp_path / "bad.dxf"
    issues = DxfExporter().export(graph(primitives=primitives, hatches=hatches), path)
    assert code in {issue.code for issue in issues}
    assert not path.exists()


@pytest.mark.parametrize("length,nominal,scale,mismatch", [(45.95, 46, 1, True), (46, 46, 1, False), (23, 46, .5, False)])
def test_dimension_compares_physical_value(tmp_path, length, nominal, scale, mismatch):
    drawing = graph(coordinate_space="sheet_mm", primitives=[line(end=(length, 0))], dimensions=[
        dict(id="dim", p1=(0, 0), p2=(length, 0), base=(0, -5), nominal_mm=nominal, view_scale=scale)])
    path = tmp_path / "dim.dxf"
    issues = DxfExporter().export(drawing, path)
    assert any(i.code == "dimension_mismatch" for i in issues) is mismatch
    dimension = ezdxf.readfile(path).modelspace().query("DIMENSION")[0]
    assert dimension.dxf.text == "<>"
    assert math.isclose(dimension.get_measurement(), length)


def test_natural_part_coordinates_are_not_scaled_twice(tmp_path):
    drawing = graph(primitives=[line()], dimensions=[
        dict(id="dim", p1=[0, 0], p2=[46, 0], base=[0, -5], nominal_mm=92, view_scale=.5)])
    path = tmp_path / "scale-conflict.dxf"
    issues = DxfExporter().export(drawing, path)
    assert any(i.code == "coordinate_scale_conflict" for i in issues)
    assert not ezdxf.readfile(path).modelspace().query("DIMENSION")


@pytest.mark.parametrize("p2,nominal", [([0, 0], None), ([0, 10], None), ([46, 0], -46)])
def test_invalid_dimension_is_reported_and_omitted(tmp_path, p2, nominal):
    drawing = graph(primitives=[line()], dimensions=[
        dict(id="dim", p1=[0, 0], p2=p2, base=[0, -5], nominal_mm=nominal)])
    path = tmp_path / "invalid-dimension.dxf"
    issues = DxfExporter().export(drawing, path)
    assert any(i.code == "invalid_dimension" and i.object_id == "dim" for i in issues)
    assert not ezdxf.readfile(path).modelspace().query("DIMENSION")


def linked_plate():
    return graph(
        primitives=[dict(id="plate", kind="polyline", closed=True, vertices=[[0, 0], [46, 0], [46, 20], [0, 20]])],
        points=[dict(id="a", xy=[0, 0], refs=[dict(object_id="plate", anchor="vertex_0")]),
                dict(id="b", xy=[46, 0], refs=[dict(object_id="plate", anchor="vertex_1")])],
        contours=[dict(id="border", object_ids=["plate"], closed=True)],
        dimensions=[dict(id="width", p1_ref="a", p2_ref="b", base=[0, -5])],
        hatches=[dict(id="section", outer_ref="border")],
    )


def test_semantic_links_survive_roundtrip_and_follow_geometry_edit(tmp_path):
    drawing = DrawingGraph.model_validate_json(linked_plate().model_dump_json())
    drawing.primitives[0].vertices[1] = (47, 0)
    drawing.primitives[0].vertices[2] = (47, 20)
    path = tmp_path / "edited.dxf"
    assert not DxfExporter().export(drawing, path)
    result = ezdxf.readfile(path)
    assert result.modelspace().query("DIMENSION")[0].get_measurement() == 47
    hatch = result.modelspace().query("HATCH")[0]
    assert max(edge.end[0] for edge in hatch.paths.paths[0].edges) == 47
    # Source observations are not overwritten by recomputed geometry.
    assert drawing.points[1].xy == (46, 0)


@pytest.mark.parametrize("field,wrong", [("p1_ref", "missing"), ("p2_ref", "missing")])
def test_unknown_point_reference_is_rejected(field, wrong):
    value = linked_plate().model_dump()
    value["dimensions"][0][field] = wrong
    with pytest.raises(ValueError, match="unknown point"):
        DrawingGraph.model_validate(value)


def test_unknown_contour_and_primitive_reference_are_rejected():
    value = linked_plate().model_dump()
    value["hatches"][0]["outer_ref"] = "missing"
    with pytest.raises(ValueError, match="unknown contour"):
        DrawingGraph.model_validate(value)
    value = linked_plate().model_dump()
    value["points"][0]["refs"][0]["object_id"] = "missing"
    with pytest.raises(ValueError, match="unknown primitive"):
        DrawingGraph.model_validate(value)


def test_cv_metadata_preserved_and_hatch_connected():
    drawing = linked_plate().model_dump()
    cv = {"status": "ok", "scale_status": "known", "coordinate_space": "sheet_mm",
          "primitives": drawing["primitives"], "points": drawing["points"], "contours": drawing["contours"],
          "hatches": [{"id": "section", "outer": [[0, 0], [46, 0], [46, 20], [0, 20]]}],
          "hatch_groups": [{"hatch_id": "section", "contour_id": "border", "hole_contour_ids": []}],
          "objects": {"plate": {"role": "part", "confidence": .95}},
          "calibration": {"manual": False}, "transforms": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
          "module": {"name": "drawcv", "version": "test"}}
    before = copy.deepcopy(cv)
    result = assemble_graph(cv, {"status": "ok", "title_block": {"material": "Сталь 20"}}, "sha")
    restored = DrawingGraph.model_validate_json(result.model_dump_json())
    assert restored.points and restored.contours
    assert restored.calibration == cv["calibration"]
    assert restored.transforms == cv["transforms"]
    assert restored.objects == cv["objects"]
    assert restored.title_block["material"] == "Сталь 20"
    assert restored.hatches[0].outer_ref == "border"
    assert restored.provenance["cv"]["module"]["name"] == "drawcv"
    assert cv == before


def test_topological_closure_does_not_mask_coordinate_gap(tmp_path):
    coords = [((0, 0), (10, 0)), ((10, 0), (10.05, 10.04)), ((10, 10), (0, 10)), ((0, 10), (0, 0))]
    drawing = graph(primitives=[dict(id=f"p{i}", kind="line", start=a, end=b) for i, (a, b) in enumerate(coords)],
                    contours=[dict(id="almost", closed=True, object_ids=[f"p{i}" for i in range(4)])],
                    hatches=[dict(id="h", outer_ref="almost")])
    path = tmp_path / "gap.dxf"
    issues = DxfExporter().export(drawing, path)
    assert "contour_gap" in {i.code for i in issues}
    assert not ezdxf.readfile(path).modelspace().query("HATCH")
    assert drawing.primitives[1].end == (10.05, 10.04)


def test_native_arcs_remain_native_hatch_edges(tmp_path):
    drawing = graph(primitives=[dict(id="circle", kind="circle", center=[0, 0], radius=10)],
                    contours=[dict(id="ring", closed=True, object_ids=["circle"])],
                    hatches=[dict(id="h", outer_ref="ring")])
    path = tmp_path / "round.dxf"
    assert not DxfExporter().export(drawing, path)
    edge = ezdxf.readfile(path).modelspace().query("HATCH")[0].paths.paths[0].edges[0]
    assert edge.EDGE_TYPE == "ArcEdge"
    assert edge.radius == 10


@pytest.mark.parametrize("end_angle,reversed", [(0, False), (180, False), (180, True)])
def test_cv_arc_edge_can_reference_a_circle(tmp_path, end_angle, reversed):
    primitives = [dict(id="circle", kind="circle", center=[0, 0], radius=10)]
    edges = [dict(object_id="circle", kind="arc", start_angle=0, end_angle=end_angle, reversed=reversed)]
    if end_angle:
        a, b = ((10, 0), (-10, 0)) if reversed else ((-10, 0), (10, 0))
        primitives.append(dict(id="chord", kind="line", start=a, end=b))
        edges.append(dict(object_id="chord", kind="line", start=a, end=b))
    drawing = graph(primitives=primitives, contours=[dict(id="ring", closed=True, edges=edges)],
                    hatches=[dict(id="h", outer_ref="ring")])
    path = tmp_path / "cv-circle.dxf"
    assert not DxfExporter().export(drawing, path)
    edge = ezdxf.readfile(path).modelspace().query("HATCH")[0].paths.paths[0].edges[0]
    assert edge.EDGE_TYPE == "ArcEdge"
    assert edge.end_angle == (end_angle or 360)
    assert edge.ccw is not reversed


@pytest.mark.parametrize("profile,layer", [("eskd_tz", "THIN"), ("editable", "HATCH")])
def test_export_profiles_and_axis_linetype(tmp_path, profile, layer):
    drawing = linked_plate()
    drawing.export_profile = profile
    drawing.primitives.append(type(drawing.primitives[0]).model_validate(line(id="axis", layer="AXIS")))
    path = tmp_path / "layers.dxf"
    assert not DxfExporter().export(drawing, path)
    doc = ezdxf.readfile(path)
    assert doc.layers.get("AXIS").dxf.linetype == "CENTER"
    assert doc.modelspace().query("HATCH")[0].dxf.layer == layer


def test_transparent_png_and_pixel_limit(tmp_path, monkeypatch):
    from app.config import get_settings
    source = tmp_path / "transparent.png"
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    image.putpixel((10, 10), (0, 0, 0, 255))
    image.save(source)
    result = ImageService().prepare(source, tmp_path / "run")
    with Image.open(result) as rgb:
        assert rgb.getpixel((0, 0)) == (255, 255, 255)
        assert rgb.getpixel((10, 10)) == (0, 0, 0)
    monkeypatch.setattr(get_settings(), "max_image_pixels", 100)
    with pytest.raises(ValueError, match="пикселей"):
        ImageService().prepare(source, tmp_path / "too_big")


def test_legacy_graph_migrates_to_current_version():
    result = DrawingGraph.model_validate({"schema_version": "0.1", "coordinate_space": "part_mm", "primitives": [line()]})
    assert result.schema_version == "0.2"
