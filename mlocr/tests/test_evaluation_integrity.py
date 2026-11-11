from copy import deepcopy
import csv
import json

import numpy as np
import pytest
from PIL import Image

from mlocr.evaluation import dim_link_cases
from mlocr.evaluation.matching import assign_pairs, match_dimensions, real_observations
from mlocr.evaluation.score_on_references import score_sheet
from mlocr.evaluation.score_qa_task1 import score
from mlocr.src.inputs import page_identity
from mlocr.src.normalization import enrich
from mlocr.tests.test_regressions import dimension_case


def test_assignment_maximizes_unique_matches_before_minimizing_distance():
    assert set(assign_pairs(2, 2, [(0, 0, .01), (0, 1, .2), (1, 0, .1)])) == {(0, 1), (1, 0)}


def test_dimension_matching_uses_both_coordinates_and_counts_duplicates_once():
    dim = {'p1_ref': 'a', 'p2_ref': 'b', 'angle': 0}
    ref = {'p1': np.array([0, 0]), 'p2': np.array([100, 0]), 'u': np.array([1., 0.])}
    points = {'a': {'xy': [0, 100]}, 'b': {'xy': [100, 100]}}
    assert not dim_link_cases.on_reference(dim, points, [ref], np.zeros(2))
    points = {'a': {'xy': [0, 0]}, 'b': {'xy': [100, 0]}}
    assert len(match_dimensions([dim, dim], points, [ref], np.zeros(2))) == 1
    reversed_dim = {**dim, 'p1_ref': 'b', 'p2_ref': 'a'}
    assert match_dimensions([reversed_dim], points, [ref], np.zeros(2)) == [(0, 0)]


def test_real_evaluation_preserves_failures_and_rejects_wrong_page():
    raw, cv = dimension_case()
    raw.update(status='error', issues=[{'code': 'ocr_page_failed', 'severity': 'error', 'message': 'Failed'}])
    result = real_observations(raw, cv)
    assert result['status'] == 'error' and result['issues'] == raw['issues']
    assert result['source'] == raw['source']
    original = deepcopy(raw)
    raw['source'] = {**raw['source'], 'raster_sha256': 'OTHER', 'page': 'page-2'}
    with pytest.raises(ValueError, match='source raster'):
        dim_link_cases.real_observations(None, cv, raw)
    assert result['source'] == original['source']


@pytest.mark.parametrize('reference_text', ['99', '5'])
def test_bad_and_single_character_readings_stay_in_reference_denominator(reference_text):
    raw, cv = dimension_case()
    raw['texts'] = [enrich({'id': 'good', 'text': '50', 'poly_source_px': [[0, 0], [5, 0], [5, 4], [0, 4]]}),
                    enrich({'id': 'bad', 'text': 'x', 'poly_source_px': [[30, 0], [35, 0], [35, 4], [30, 4]]})]
    refs = [{'text': '50', 'xy': np.array([0., 0.])}, {'text': reference_text, 'xy': np.array([30., 0.])}]
    rows = score_sheet(raw, cv, {'registration': {'translation_mm': [0, 0]}}, refs)
    assert len(rows) == 2 and sum(row['exact'] for row in rows) == 1
    assert all(row['match_status'] == 'matched' for row in rows)
    raw['texts'].append(deepcopy(raw['texts'][0]))
    rows = score_sheet(raw, cv, {'registration': {'translation_mm': [0, 0]}}, refs)
    assert sum(row['exact'] for row in rows) == 1
    assert sum(row['match_status'] == 'extra' for row in rows) == 1


def test_real_dimension_evaluation_does_not_read_scale_from_reference(tmp_path, monkeypatch):
    raw, cv = dimension_case(.5)
    cv['calibration']['view_scale'] = None
    case = tmp_path / 'case'
    case.mkdir()
    (case / 'cv.json').write_text(json.dumps(cv), encoding='utf-8')
    (case / 'metrics.json').write_text(json.dumps({'case': 'fixture', 'sheet': 'fixture',
        'registration': {'translation_mm': [0, 0]}}), encoding='utf-8')
    raw_path = tmp_path / 'raw.json'
    raw_path.write_text(json.dumps(raw), encoding='utf-8')
    ref = {'p1': np.array([0., 0.]), 'p2': np.array([50., 0.]), 'u': np.array([1., 0.]),
           'text_at': np.array([25., 12.]), 'nominal': 100., 'factor': 2., 'diameter': False}
    cv['transforms']['sheet_mm_to_source_px'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    (case / 'cv.json').write_text(json.dumps(cv), encoding='utf-8')
    monkeypatch.setattr(dim_link_cases.base, 'reference_dimensions', lambda _: [ref])
    result = dim_link_cases.check_case(case, raw_path)
    assert result['modes']['ideal']['reference_assisted'] is True
    assert result['modes']['ideal']['calibration']['view_scale'] == .5
    real = result['modes']['real']
    assert not real['reference_assisted'] and real['calibration']['view_scale'] is None
    raw['source']['raster_sha256'] = 'OTHER'
    raw_path.write_text(json.dumps(raw), encoding='utf-8')
    assert 'source raster' in dim_link_cases.check_case(case, raw_path)['modes']['real']['error']
    monkeypatch.setattr(dim_link_cases.base, 'reference_dimensions', lambda _: [ref, {**ref, 'factor': 1.}])
    assert 'exactly one reference scale' in dim_link_cases.check_case(case, None)['modes']['ideal']['error']


def make_qa(tmp_path):
    images, results = tmp_path / 'images', tmp_path / 'observations'
    images.mkdir()
    results.mkdir()
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    Image.fromarray(image).save(images / 'fixture.png')
    payload = {'input': 'fixture.png', 'source': page_identity(image), 'status': 'ok', 'issues': [],
               'module': {'name': 'fixture', 'version': '1'}, 'params': {'passes': [0, 90, 180]},
               'texts': [enrich({'id': 't', 'text': '50', 'score': .99,
                    'poly_source_px': [[0, 0], [20, 0], [20, 10], [0, 10]]}),
                    enrich({'id': 'extra', 'text': '25', 'score': .99,
                    'poly_source_px': [[50, 60], [70, 60], [70, 70], [50, 70]]})]}
    raw_path = results / 'fixture.ocr.json'
    raw_path.write_text(json.dumps(payload), encoding='utf-8')
    worksheet = tmp_path / 'qa.csv'
    with worksheet.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'page', 'text_id', 'bbox_xyxy_px', 'ground_truth'])
        writer.writeheader()
        writer.writerows([{'image': 'fixture.png', 'page': 'page-1', 'text_id': 'a',
                           'bbox_xyxy_px': '0,0,20,10', 'ground_truth': '50'},
                          {'image': 'fixture.png', 'page': 'page-1', 'text_id': 'b',
                           'bbox_xyxy_px': '75,25,95,35', 'ground_truth': '90'}])
    return worksheet, results, images, raw_path


def test_qa_scoring_does_not_modify_sources_and_declares_partial_scope(tmp_path):
    worksheet, results, images, raw = make_qa(tmp_path)
    before = {path: path.read_bytes() for path in (worksheet, raw)}
    report = score(worksheet, results, images, tmp_path / 'run')
    assert report['exact_text_accuracy'] == .5 and report['missed_labelled_rows'] == 1
    assert report['unassigned_prediction_regions'] == 1 and report['false_positive_regions'] is None
    assert report['scope'] == 'reviewed_regions_only' and report['cer'] == .5
    assert all(path.read_bytes() == data for path, data in before.items())
    complete = score(worksheet, results, images, tmp_path / 'complete-run', complete_pages=True)
    assert complete['false_positive_regions'] == 1 and complete['cer'] == 1.
    with pytest.raises(ValueError, match='refusing to overwrite'):
        score(worksheet, results, images, tmp_path / 'run')


def test_qa_score_rejects_wrong_raster_and_missing_results(tmp_path):
    worksheet, results, images, raw = make_qa(tmp_path)
    payload = json.loads(raw.read_text(encoding='utf-8'))
    payload['source']['raster_sha256'] = 'OTHER'
    raw.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='source raster'):
        score(worksheet, results, images, tmp_path / 'wrong-run')
    assert not (tmp_path / 'wrong-run').exists()
    with pytest.raises(ValueError, match='No .*observations'):
        score(worksheet, tmp_path / 'missing', images, tmp_path / 'empty-run')


def test_qa_score_cannot_reuse_prediction_for_two_reference_regions(tmp_path):
    worksheet, results, images, raw = make_qa(tmp_path)
    text = worksheet.read_text(encoding='utf-8').replace('75,25,95,35', '0,0,20,10').replace(',90', ',50')
    worksheet.write_text(text, encoding='utf-8')
    report = score(worksheet, results, images, tmp_path / 'duplicate-run')
    assert report['exact_text_rows'] == 1 and report['missed_labelled_rows'] == 1


def test_reference_cli_publishes_complete_denominators_and_rejects_empty_input(tmp_path, monkeypatch, capsys):
    import sys
    import ezdxf
    from mlocr.evaluation import score_on_references as scorer

    _, results, _, original = make_qa(tmp_path)
    raw_dir, case_root, ref_dir = tmp_path / 'raw', tmp_path / 'cv', tmp_path / 'dxf'
    raw_dir.mkdir()
    ref_dir.mkdir()
    case = case_root / 'fixture.png'
    case.mkdir(parents=True)
    raw = json.loads(original.read_text(encoding='utf-8'))
    (raw_dir / 'fixture.json').write_text(json.dumps(raw), encoding='utf-8')
    cv = {'raster_identity': raw['source'], 'calibration': {'sheet_mm': [100, 80]},
          'transforms': {'source_px_to_sheet_mm': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
    (case / 'cv.json').write_text(json.dumps(cv), encoding='utf-8')
    (case / 'metrics.json').write_text(json.dumps({'registration': {'translation_mm': [0, 0]}}), encoding='utf-8')
    doc = ezdxf.new()
    doc.modelspace().add_text('50', dxfattribs={'insert': (0, 0), 'height': 4})
    doc.modelspace().add_text('90', dxfattribs={'insert': (75, 25), 'height': 4})
    doc.saveas(ref_dir / 'fixture.dxf')
    report = tmp_path / 'summary.json'
    argv = ['score', '--raw', str(raw_dir), '--cv-dirs', str(case_root), '--dxf-dir', str(ref_dir), '--json', str(report)]
    monkeypatch.setattr(sys, 'argv', argv)
    scorer.main()
    summary = json.loads(report.read_text(encoding='utf-8'))
    assert summary['reference_texts'] == 2 and summary['predicted_texts'] == 2
    assert summary['exact_share'] == .5 and summary['missed_texts'] == 1 and summary['extra_texts'] == 1
    assert summary['cer'] == 1. and summary['readings_matched'] == 1
    monkeypatch.setattr(sys, 'argv', ['score', '--raw', str(tmp_path / 'absent')])
    with pytest.raises(SystemExit) as error:
        scorer.main()
    assert error.value.code == 2


def test_dimension_cli_rejects_empty_evaluation(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'argv', ['dim_link_cases', str(tmp_path / 'case')])
    monkeypatch.setattr(dim_link_cases, 'check_case', lambda *_: {'modes': {}, 'reference_dims': 0})
    with pytest.raises(SystemExit) as error:
        dim_link_cases.main()
    assert error.value.code == 2
