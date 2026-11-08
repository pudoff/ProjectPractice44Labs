from copy import deepcopy
import ezdxf
from app.services.recognition import assemble_graph
from app.services.dxf import DxfExporter
from mlocr.tests.test_regressions import dimension_case
from mlocr.src.adapter import to_drawing
import numpy as np
import pytest
from PIL import Image
from app.config import get_settings
from app.services.recognition import recognize
from mlocr.src.inputs import page_identity


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


def setup_adapters(tmp_path, monkeypatch):
    observations, cv = dimension_case(.5)
    source = tmp_path / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(source)
    identity = page_identity(np.asarray(Image.open(source).convert('RGB')))
    observations.update(schema_version='ocr-observations-1', module={'name': 'mlocr', 'version': 'fixture-1'}, params={'passes': [0, 90]})
    observations['source'] = deepcopy(identity)
    cv.update(raster_identity=deepcopy(identity), module={'name': 'drawcv', 'version': 'fixture-2'}, params={'manual_px_per_mm': 1})
    monkeypatch.setattr(get_settings(), 'recognition_profile', 'current')
    monkeypatch.setattr(get_settings(), 'ocr_adapter', 'fixtures:ocr')
    monkeypatch.setattr(get_settings(), 'cv_adapter', 'fixtures:cv')
    monkeypatch.setattr('app.services.recognition.call_adapter', lambda target, image: deepcopy(observations if target == 'fixtures:ocr' else cv))
    return source, observations, cv


def test_common_prepared_page_provenance_raw_value_and_geometry_recalculation(tmp_path, monkeypatch):
    source, observations, cv = setup_adapters(tmp_path, monkeypatch)
    observations['texts'][0]['raw_text'] = '100,0'
    graph = recognize(source, tmp_path / 'run', 'source-hash')
    assert graph.dimensions[0].recognized_text == '100,0'
    assert graph.dimensions[0].nominal_mm == 100
    assert graph.dimensions[0].p1_ref and graph.dimensions[0].p2_ref
    assert graph.provenance['adapters'] == {'profile': 'current', 'ocr': 'fixtures:ocr', 'cv': 'fixtures:cv'}
    for module in ('ocr', 'cv'):
        assert graph.provenance[module]['module']['version'].startswith('fixture-')
        assert graph.provenance[module]['params']
    graph.primitives[0].end = (51, 0)
    target = tmp_path / 'edited.dxf'
    issues = DxfExporter().export(graph, target)
    entity = ezdxf.readfile(target).modelspace().query('DIMENSION')[0]
    assert entity.get_measurement() == 51
    assert entity.dxf.text == '<>'
    assert any(issue.code == 'dimension_mismatch' for issue in issues)


@pytest.mark.parametrize('module,field,value', [
    ('ocr', 'width_px', 201), ('ocr', 'page', 'page-2'), ('ocr', 'raster_sha256', 'other'),
    ('cv', 'height_px', 101), ('both', 'raster_sha256', 'same-but-stale'),
])
def test_reject_wrong_page_or_stale_raster(tmp_path, monkeypatch, module, field, value):
    source, observations, cv = setup_adapters(tmp_path, monkeypatch)
    if module in ('ocr', 'both'):
        observations['source'][field] = value
    if module in ('cv', 'both'):
        cv['raster_identity'][field] = value
    graph = recognize(source, tmp_path / 'run', 'source-hash')
    assert not graph.texts and not graph.dimensions
    assert any(issue.code == 'ocr_error' for issue in graph.issues)
    if module in ('cv', 'both'):
        assert not graph.primitives and graph.scale_status == 'unknown'


def test_uncertain_reading_remains_text_and_requires_review(tmp_path):
    observations, cv = dimension_case(.5)
    observations['texts'][0]['needs_review'] = True
    ocr = to_drawing(observations, cv)
    graph = assemble_graph(cv, ocr, 'source')
    path = tmp_path / 'uncertain.dxf'
    issues = DxfExporter().export(graph, path)
    assert not ezdxf.readfile(path).modelspace().query('DIMENSION')
    assert ezdxf.readfile(path).modelspace().query('TEXT')[0].dxf.text == '100'
    assert any(issue.code == 'ocr_value_needs_review' for issue in issues)


def test_different_measured_points_are_not_selected_by_text_proximity():
    from mlocr.src.normalization import enrich
    observations, cv = dimension_case(.5)
    for primitive in deepcopy(cv['primitives']):
        primitive['id'] += '-other'
        for point in ('start', 'end'):
            primitive[point][1] += 5
        cv['primitives'].append(primitive)
        cv['objects'][primitive['id']] = {'role': 'part' if primitive['id'] == 'part-other' else 'thin'}
    observations['texts'][0] = enrich({'id': 'ambiguous', 'text': '100',
        'poly_source_px': [[23, 11], [27, 11], [27, 13], [23, 13]], 'purpose': 'dimension_or_scale_candidate'})
    ocr = to_drawing(observations, cv)
    assert not ocr['dimensions']
    assert ocr['texts'] and any(issue['code'] == 'dimension_unlinked' for issue in ocr['issues'])


def test_unknown_numeric_dimension_value_is_reported(tmp_path):
    from app.domain.drawing import DrawingGraph
    graph = DrawingGraph.model_validate({
        'coordinate_space': 'part_mm',
        'primitives': [{'id': 'part', 'kind': 'line', 'start': [0, 0], 'end': [50, 0]}],
        'dimensions': [{'id': 'unknown', 'recognized_text': '?', 'nominal_mm': None,
                        'p1': [0, 0], 'p2': [50, 0], 'base': [0, -5]}],
    })
    issues = DxfExporter().export(graph, tmp_path / 'unknown-number.dxf')
    assert any(issue.code == 'unknown_dimension_value' for issue in issues)


def test_geometry_edit_cannot_keep_a_dimension_on_a_free_ocr_point(tmp_path):
    observations, cv = dimension_case(.5)
    ocr = to_drawing(observations, cv)
    cv['points'][-1]['refs'] = []
    graph = assemble_graph(cv, ocr, 'source')
    original = tmp_path / 'original.dxf'
    assert not DxfExporter().export(graph, original)
    assert ezdxf.readfile(original).modelspace().query('DIMENSION')
    graph.primitives[0].end = (51, 0)
    target = tmp_path / 'edited.dxf'
    issues = DxfExporter().export(graph, target)
    assert any(issue.code == 'dimension_recalculation_required' for issue in issues)
    assert not ezdxf.readfile(target).modelspace().query('DIMENSION')


def test_style_edit_does_not_invalidate_free_measurement(tmp_path):
    observations, cv = dimension_case(.5)
    ocr = to_drawing(observations, cv)
    cv['points'][-1]['refs'] = []
    graph = assemble_graph(cv, ocr, 'source')
    graph.primitives[0].layer = 'THIN'
    assert not DxfExporter().export(graph, tmp_path / 'style-only.dxf')


@pytest.mark.parametrize('ocr_mode', ['ready', 'error'])
@pytest.mark.parametrize('wrong_module', ['ocr', 'cv'])
def test_identity_required_for_all_adapter_formats(tmp_path, monkeypatch, ocr_mode, wrong_module):
    source, observations, cv = setup_adapters(tmp_path, monkeypatch)
    observations['texts'] = [{'id': 'ready', 'value': '100', 'insert': [0, 0], 'height': 3}]
    observations['schema_version'] = 'ready-graph'
    observations['status'] = 'ok' if ocr_mode == 'ready' else 'error'
    if wrong_module == 'ocr':
        observations['source']['raster_sha256'] = 'other-page'
    else:
        cv['raster_identity']['raster_sha256'] = 'other-page'
    graph = recognize(source, tmp_path / 'run', 'source-hash')
    if wrong_module == 'ocr':
        assert not graph.texts and not graph.dimensions
        assert any(i.code == 'ocr_error' for i in graph.issues)
    else:
        assert not graph.primitives
        assert any(i.code == 'cv_error' for i in graph.issues)


@pytest.mark.parametrize('offset', [0, .2, 1])
def test_parallel_or_duplicate_contour_cannot_supply_unique_foot(offset):
    observations, cv = dimension_case(.5)
    other = deepcopy(cv['primitives'][0])
    other['id'] = 'competing-part'
    other['start'][1] += offset
    other['end'][1] += offset
    cv['primitives'].append(other)
    cv['objects'][other['id']] = {'role': 'part'}
    output = to_drawing(observations, cv)
    assert not output['dimensions']
    assert any(i['code'] == 'dimension_unlinked' for i in output['issues'])
