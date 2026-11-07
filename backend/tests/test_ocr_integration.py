"""Contract/export checks; real model runs are separate from these unit fixtures."""
from copy import deepcopy
import ezdxf
from app.services.recognition import assemble_graph
from app.services.dxf import DxfExporter
from mlocr.tests.test_regressions import dimension_case
from mlocr.src.adapter import to_drawing


def test_ocr_cv_link_exports_native_dimension_and_preserves_evidence(tmp_path):
    observations, cv = dimension_case(.5)
    ocr = to_drawing(observations, cv)
    graph = assemble_graph(cv, ocr, 'test-source')
    path = tmp_path/'linked.dxf'
    issues = DxfExporter().export(graph, path)
    assert not issues
    doc = ezdxf.readfile(path)
    entity = doc.modelspace().query('DIMENSION')[0]
    assert entity.get_measurement() == 50
    assert entity.get_acad_dstyle(doc.dimstyles.get(entity.dxf.dimstyle))['dimlfac'] == 2
    assert len(doc.modelspace().query('LINE')) == 1  # Auxiliary lines belong to DIMENSION now.
    assert graph.provenance['ocr']['dimension_links'][0]['confirmed']['evidence']['dimension_line'] == 'dim'
    assert graph.provenance['ocr']['observations'][0]['raw_text'] == '100'


def test_clustered_cv_point_uses_exact_measured_edge_without_moving_it(tmp_path):
    observations, cv = dimension_case(.5)
    cv['primitives'].append({'id': 'neighbor', 'kind': 'line', 'start': [0.1, -20], 'end': [0.1, 0]})
    cv['objects']['neighbor'] = {'role': 'part'}
    cv['points'][0]['refs'].append({'object_id': 'neighbor', 'anchor': 'end'})
    cv['points'][0]['xy'] = [.05, 0]
    ocr = to_drawing(observations, cv)
    graph = assemble_graph(cv, ocr, 'fixture')
    assert graph.dimensions[0].p1_ref.startswith('ocr-anchor-')
    assert cv['primitives'][0]['start'] == [0, 0]  # Never snap to the printed number.
    assert cv['primitives'][-1]['end'] == [.1, 0]
    issues = DxfExporter().export(graph, tmp_path/'exact.dxf')
    assert not any(i.code in {'ambiguous_point', 'dimension_mismatch'} for i in issues)
