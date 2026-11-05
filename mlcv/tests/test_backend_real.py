"""cv.json against the real backend and OCR code of this repository: DrawingGraph, GeometryResolver, DXF export.

test_backend_contract.py keeps its own copies of the models and runs in the drawcv image alone; these tests need the
backend/ and mlocr/ folders and their packages (requirements-dev.txt), otherwise they are skipped."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "backend").is_dir():
    pytest.skip("backend/ is not available here", allow_module_level=True)
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
_tmp = Path(tempfile.mkdtemp(prefix="drawcv-backend-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{(_tmp / 'test.sqlite3').as_posix()}")
os.environ.setdefault("STORAGE_ROOT", str(_tmp / "storage"))
os.environ.setdefault("REQUIRE_LOCAL_MODELS", "false")

pytest.importorskip("pydantic_settings")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("pymupdf")
ezdxf = pytest.importorskip("ezdxf")

from app.services.dxf import DxfExporter  # noqa: E402
from app.services.geometry import GeometryResolver  # noqa: E402
from app.services.recognition import assemble_graph  # noqa: E402
from mlocr.src.adapter import to_drawing  # noqa: E402
from mlocr.src.normalization import enrich  # noqa: E402


def empty_ocr(cv):
    return {"status": "ok", "source": cv["raster_identity"], "texts": [], "issues": [], "title_block": {}}


@pytest.mark.parametrize("name", ["control.png", "control.jpg", "control_scan.jpg"])
def test_every_closed_contour_and_point_is_accepted(results, name):
    cv, _ = results(name)
    graph = assemble_graph(cv, to_drawing(empty_ocr(cv), cv), "test")
    resolver = GeometryResolver(graph)
    closed = [c for c in graph.contours if c.closed]
    assert len(closed) >= 2
    for c in closed:
        resolver.contour(c.id)                       # raises on a real gap or an invalid ring
    for p in graph.points:
        resolver.point(p.id)                         # raises when the anchors of a point do not coincide


def test_export_has_no_geometry_errors_and_writes_the_hatch(results, tmp_path):
    cv, _ = results("control.png")
    graph = assemble_graph(cv, to_drawing(empty_ocr(cv), cv), "test")
    path = tmp_path / "out.dxf"
    errors = [i for i in DxfExporter().export(graph, path) if i.severity == "error"]
    assert not errors, errors
    msp = ezdxf.readfile(path).modelspace()
    assert len(msp.query("HATCH")) == 1 and msp.query("LINE") and msp.query("CIRCLE")


def test_move_records_reach_the_drawing(results):
    cv, _ = results("control.png")
    graph = assemble_graph(cv, to_drawing(empty_ocr(cv), cv), "test")
    assert graph.provenance["cv"]["closures"]
    assert any("adjustments" in o and "observed_geometry_mm" in o for o in graph.objects.values())


def test_a_dimension_links_to_a_shared_corner(results):
    """A linear dimension between two points of a real sheet: linking must not fail because a corner is shared by two
    lines, now that the anchors of a point really coincide."""
    cv, _ = results("control.png")
    primitives = {p["id"]: p for p in cv["primitives"]}
    shared = next(p for p in cv["points"] if p["kind"] == "junction" and len(p["refs"]) == 2
                  and all(cv["objects"][r["object_id"]]["role"] == "part" and primitives[r["object_id"]]["kind"] == "line" for r in p["refs"]))
    coords = {tuple(primitives[r["object_id"]][r["anchor"]]) for r in shared["refs"]}
    assert len(coords) == 1, "two lines of one point have one written coordinate"


def test_ocr_text_is_positioned_through_the_transform(results):
    cv, _ = results("control.png")
    poly = [[1000, 1000], [1100, 1000], [1100, 1040], [1000, 1040]]
    obs = empty_ocr(cv)
    obs["texts"] = [enrich({"id": "t1", "text": "ГОСТ", "score": 1.0, "purpose": "note", "poly_source_px": poly})]
    out = to_drawing(obs, cv)
    assert out["texts"][0]["height"] > 0 and out["status"] == "ok"
