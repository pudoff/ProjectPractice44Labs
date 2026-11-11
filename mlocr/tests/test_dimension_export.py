"""DIMENSION export: the measured value follows the geometry, not the OCR number."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in (str(ROOT), str(ROOT / "backend"), str(ROOT / "mlcv")):
    if extra not in sys.path:
        sys.path.insert(0, extra)
_tmp = Path(tempfile.mkdtemp(prefix="mlocr-dim-export-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{(_tmp / 't.sqlite3').as_posix()}")
os.environ.setdefault("STORAGE_ROOT", str(_tmp / "storage"))
os.environ.setdefault("REQUIRE_LOCAL_MODELS", "false")

ezdxf = pytest.importorskip("ezdxf")
pytest.importorskip("pydantic_settings")
pytest.importorskip("sqlalchemy")
pytest.importorskip("drawcv")

from app.services.dxf import DxfExporter  # noqa: E402
from app.services.recognition import assemble_graph  # noqa: E402
from mlocr.src.adapter import to_drawing  # noqa: E402
from mlocr.src.normalization import enrich  # noqa: E402


def make_case():
    identity = {'page': 'page-1', 'width_px': 400, 'height_px': 100, 'raster_sha256': 'fixture'}
    text = enrich({'id': 't1', 'text': '100', 'poly_source_px': [[150, 17], [250, 17], [250, 33], [150, 33]],
                   'purpose': 'dimension_or_scale_candidate'})
    cv = {'status': 'ok', 'raster_identity': identity, 'coordinate_space': 'sheet_mm', 'scale_status': 'known',
          'calibration': {'view_scale': 1.0}, 'transforms': {'source_px_to_sheet_mm': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
          'points': [{'id': 'p1', 'xy': [100, 0], 'refs': [{'object_id': 'part', 'anchor': 'start'}]},
                     {'id': 'p2', 'xy': [200, 0], 'refs': [{'object_id': 'part', 'anchor': 'end'}]}],
          'primitives': [{'id': 'part', 'kind': 'line', 'start': [100, 0], 'end': [200, 0]},
                         {'id': 'dim', 'kind': 'line', 'start': [100, 25], 'end': [200, 25]},
                         {'id': 'e1', 'kind': 'line', 'start': [100, 0], 'end': [100, 30]},
                         {'id': 'e2', 'kind': 'line', 'start': [200, 0], 'end': [200, 30]}],
          'objects': {'part': {'role': 'part'}, 'dim': {'role': 'thin'}, 'e1': {'role': 'thin'}, 'e2': {'role': 'thin'}}}
    obs = {'status': 'ok', 'source': identity, 'texts': [text], 'issues': [], 'title_block': {}}
    return obs, cv


def exported_dimension(graph, path):
    issues = DxfExporter().export(graph, path)
    assert path.exists()
    errors = [i for i in issues if i.severity == "error"]
    assert not errors
    dims = list(ezdxf.readfile(str(path)).modelspace().query("DIMENSION"))
    assert len(dims) == 1
    return dims[0]


def test_exported_dimension_measures_geometry_and_reexport_follows_changes(tmp_path):
    obs, cv = make_case()
    ocr = to_drawing(obs, cv)
    assert len(ocr['dimensions']) == 1 and ocr['dimensions'][0]['display'] == '<>'
    graph = assemble_graph(cv, ocr, 'fixture')
    dim = exported_dimension(graph, tmp_path / 'first.dxf')
    assert dim.dxf.text == '<>'                      # the OCR number is not baked into the entity text
    assert float(dim.get_measurement()) == pytest.approx(100.0, abs=0.01)

    part = next(p for p in graph.primitives if p.id == 'part')
    part.end = (207.0, 0.0)                          # edit DrawingGraph, then export again
    path2 = tmp_path / 'second.dxf'
    issues = DxfExporter().export(graph, path2)
    dim2 = list(ezdxf.readfile(str(path2)).modelspace().query("DIMENSION"))[0]
    assert dim2.dxf.text == '<>'
    assert float(dim2.get_measurement()) == pytest.approx(107.0, abs=0.01)   # the measured value, not the printed 100
    assert any(i.code == 'dimension_mismatch' for i in issues)   # the OCR nominal is kept for comparison only
