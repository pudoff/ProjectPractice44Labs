from copy import deepcopy

import ezdxf
import pytest
from pydantic import ValidationError

from app.domain.drawing import AnchorRef, DrawingGraph
from app.services.dxf import DxfExporter, printed_decimals
from app.services.geometry import GeometryResolver
from app.services.recognition import assemble_graph, recognize
from mlocr.src.adapter import to_drawing
from mlocr.tests.test_regressions import dimension_case
from test_ocr_integration import setup_adapters


def test_interior_line_feet_follow_geometry_after_graph_roundtrip(tmp_path):
    observations, cv = dimension_case(.5)
    cv['points'] = []
    cv['primitives'][0].update(start=[-5, 0], end=[55, 0])
    result = to_drawing(observations, cv)
    assert len(result['dimensions']) == 1
    graph = DrawingGraph.model_validate(assemble_graph(cv, result, 'fixture').model_dump())
    assert all(point.refs and point.refs[0].anchor == 'segment' for point in graph.points)
    part = next(p for p in graph.primitives if p.id == 'part')
    part.start, part.end = (-5, 5), (55, 5)
    resolved = GeometryResolver(graph).dimension(graph.dimensions[0])
    assert resolved.p1 == pytest.approx((0, 5)) and resolved.p2 == pytest.approx((50, 5))
    path = tmp_path / 'edited.dxf'
    issues = DxfExporter().export(graph, path)
    assert not [issue for issue in issues if issue.severity == 'error']
    dim = ezdxf.readfile(path).modelspace().query('DIMENSION')[0]
    assert dim.dxf.defpoint2.y == pytest.approx(5)
    assert dim.dxf.defpoint3.y == pytest.approx(5)


@pytest.mark.parametrize('anchor,parameter', [('segment', None), ('segment', -0.1), ('segment', 1.1), ('start', .5)])
def test_invalid_segment_anchor_is_rejected(anchor, parameter):
    with pytest.raises(ValidationError):
        AnchorRef(object_id='part', anchor=anchor, parameter=parameter)


@pytest.mark.parametrize('text,decimals', [('8отв.Ø100.5', 1), ('8 отв. Ø100,50', 2), ('Ø12H7', 0), ('50±0.1', 0)])
def test_precision_comes_from_nominal_not_hole_count_or_tolerance(text, decimals):
    assert printed_decimals(text) == decimals


def test_precision_survives_raw_hole_abbreviation_with_mixed_alphabets():
    assert printed_decimals('8omb.100,50', 100.5) == 2


@pytest.mark.parametrize('text', ['50', '50.0', None])
def test_printed_precision_does_not_suppress_geometry_mismatch(tmp_path, text):
    graph = DrawingGraph.model_validate({'coordinate_space': 'part_mm',
        'primitives': [{'id': 'part', 'kind': 'line', 'start': [0, 0], 'end': [49.6, 0]}],
        'dimensions': [{'id': 'dim', 'p1': [0, 0], 'p2': [49.6, 0], 'base': [25, 10],
                        'nominal_mm': 50, 'recognized_text': text}]})
    issues = DxfExporter().export(graph, tmp_path / 'mismatch.dxf')
    assert any(issue.code == 'dimension_mismatch' for issue in issues)


def test_failed_raw_ocr_is_saved_and_reaches_backend_report(tmp_path, monkeypatch):
    source, observations, cv = setup_adapters(tmp_path, monkeypatch)
    observations.update(status='error', error='All OCR passes failed', texts=[],
                        issues=[{'code': 'ocr_page_failed', 'severity': 'error', 'message': 'All passes failed'}])
    graph = recognize(source, tmp_path / 'failed-run', 'fixture')
    assert (tmp_path / 'failed-run/ocr-observations.json').exists()
    assert any(issue.code == 'ocr_page_failed' for issue in graph.issues)
    assert any(issue.code == 'ocr_error' for issue in graph.issues)
    assert not graph.dimensions
