"""Regression cases from the 6f52633 review; no OCR weights are required."""
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from mlocr.src import cli, dimension_links
from mlocr.src.adapter import to_drawing
from mlocr.src.normalization import enrich, normalize_text
from mlocr.tests.test_regressions import dimension_case


@pytest.mark.parametrize('raw,expected', [
    ('ИвановA', 'Иванов A'), ('СтальA36', 'Сталь A36'), ('abcИванов', 'abc Иванов'),
    ('M12', 'M12'), ('10m', '10m'), ('мама', 'мама'), ('Сmаль 20', 'Сталь 20'),
])
def test_mixed_script_boundaries_do_not_delete_letters(raw, expected):
    assert normalize_text(raw)[0] == expected
    assert enrich({'text': raw})['raw_text'] == raw


@pytest.mark.parametrize('fail_count', [0, 1, 3])
def test_pass_failures_reported_even_when_no_text(tmp_path, monkeypatch, fail_count):
    path = tmp_path / 'blank.png'
    Image.new('RGB', (80, 60), 'white').save(path)

    class Engine:
        calls = 0

        def predict(self, image):
            self.calls += 1
            if self.calls <= fail_count:
                raise RuntimeError('controlled failure')
            return []

    monkeypatch.setattr(cli, 'get_engine', lambda *_: EngineInstance)
    EngineInstance = Engine()
    result = cli.recognize_file(path)
    assert result['status'] == ('error' if fail_count == 3 else 'ok')
    assert len(result['pages'][0]['pass_failures']) == fail_count
    assert bool(result['issues']) == bool(fail_count)
    assert not result['texts']


@pytest.mark.parametrize('original,parts,accepted', [
    ('100±0.1', ['100', '0.1'], False), ('100±0.1', ['100', '±0.1'], True),
    ('Ø12H7', ['12', 'H7'], False), ('8отв.Ø100', ['отв.', 'Ø100'], False),
    ('Ø12H7', ['Ø12', 'H7'], True), ('100+0.1', ['100', '-0.1'], False),
])
def test_cell_split_preserves_engineering_notation(original, parts, accepted):
    assert cli._split_preserves_reading(original, [{'text': s, 'score': .99} for s in parts]) is accepted


def test_gridline_reread_preserves_parent_warnings_and_limits_calls(monkeypatch):
    image = np.full((200, 220, 3), 255, dtype=np.uint8)
    image[150:185, 110:112] = 0
    original = {'text': 'Разраб.Иванов', 'score': .9,
                'polygon': np.array([[20, 150], [200, 150], [200, 185], [20, 185]]),
                'needs_review': True, 'issues': [{'code': 'ocr_pass_failed', 'message': 'parent warning'}]}
    calls = []

    def read(*args):
        calls.append(args[-1])
        return [{'text': 'Разраб.' if len(calls) <= 2 else 'Иванов', 'score': .99,
                 'polygon': np.array([[0, 0], [40, 0], [40, 30], [0, 30]]), 'pass_angle_deg': args[-1]}]

    monkeypatch.setattr(cli, '_read_pass', read)
    split = cli._split_title_block_gridlines(None, image, [deepcopy(original)], max_crop_passes=4)
    assert len(split) == 2
    assert all(item['needs_review'] and item['issues'][0]['code'] == 'ocr_pass_failed' for item in split)
    calls.clear()
    unsplit = cli._split_title_block_gridlines(None, image, [deepcopy(original)], max_crop_passes=1)
    assert len(calls) == 1 and len(unsplit) == 1
    assert any(i['code'] == 'ocr_retry_budget_exhausted' for i in unsplit[0]['issues'])


def test_no_measured_part_preserves_text_and_lines():
    observations, cv = dimension_case(.5)
    cv['points'] = []
    cv['primitives'] = [p for p in cv['primitives'] if p['id'] != 'part']
    del cv['objects']['part']
    cv['primitives'].append({'id': 'remote', 'kind': 'line', 'start': [0, 100], 'end': [50, 100]})
    cv['objects']['remote'] = {'role': 'part'}
    result = to_drawing(observations, cv)
    assert not result['dimensions'] and result['texts'][0]['value'] == '100'
    assert any(issue['code'] == 'dimension_unlinked' for issue in result['issues'])
    assert not any(o.get('represented_by_dimension') for o in cv['objects'].values())


@pytest.mark.parametrize('existing_scale', [None, .5])
def test_conflicting_printed_scales_block_all_dimension_search(monkeypatch, existing_scale):
    observations, cv = dimension_case(.5)
    cv['calibration']['view_scale'] = existing_scale
    for index, scale in enumerate(['1:1', '1:2']):
        observations['texts'].append(enrich({'id': f'scale-{index}', 'text': scale,
            'poly_source_px': [[80, 50], [90, 50], [90, 55], [80, 55]]}))

    def forbidden(*args, **kwargs):
        pytest.fail('conflicting scales must not start geometry search')

    monkeypatch.setattr(dimension_links, 'find_dimension', forbidden)
    result = to_drawing(observations, cv)
    assert not result['dimensions'] and len(result['texts']) == 3
    assert any(issue['code'] == 'multiple_view_scales' for issue in result['issues'])


def test_large_input_limit_precedes_scale_voting(monkeypatch):
    observations, cv = dimension_case(.5)
    cv['calibration'].pop('view_scale')
    cv['primitives'] = [deepcopy(cv['primitives'][0]) for _ in range(8001)]
    monkeypatch.setattr(dimension_links, 'find_dimension', lambda *args: pytest.fail('search before limit'))
    result = to_drawing(observations, cv)
    assert not result['dimensions'] and result['texts']
    assert any(issue['code'] == 'dimension_link_limit' for issue in result['issues'])


def test_missing_dimension_line_is_conservatively_preserved_as_text():
    observations, cv = dimension_case()
    cv['primitives'] = [p for p in cv['primitives'] if p['id'] != 'dim']
    del cv['objects']['dim']
    result = to_drawing(observations, cv)
    assert not result['dimensions'] and result['texts']


def test_arrow_near_one_hypothesis_does_not_hide_different_measured_part():
    observations, cv = dimension_case()
    for primitive in deepcopy(cv['primitives']):
        original_id = primitive['id']
        primitive['id'] += '-other'
        for key in ('start', 'end'):
            primitive[key][1] += 7
        cv['primitives'].append(primitive)
        cv['objects'][primitive['id']] = {'role': 'part' if original_id == 'part' else 'thin'}
    cv['objects']['arrow'] = {'role': 'arrowhead', 'kind': 'line',
                              'geometry_mm': {'start': [-1, 10], 'end': [1, 10]}}
    result = to_drawing(observations, cv)
    assert not result['dimensions'] and result['texts']


def test_unsupported_curve_anchor_is_not_replaced_with_free_point():
    observations, cv = dimension_case()
    cv['primitives'][0] = {'id': 'part', 'kind': 'circle', 'center': [25, 0], 'radius': 25}
    cv['points'] = []
    result = to_drawing(observations, cv)
    assert not result['dimensions'] and result['texts']


@pytest.mark.parametrize('value,display', [('8отв.Ø100', '8отв.%%c<>'), ('Ø100H7', '%%c<>H7'), ('100±0.1', '<>%%p0.1')])
def test_dimension_display_preserves_all_supported_notation(value, display):
    observations, cv = dimension_case()
    observations['texts'][0] = enrich({**observations['texts'][0], 'text': value, 'raw_text': value})
    assert to_drawing(observations, cv)['dimensions'][0]['display'] == display
