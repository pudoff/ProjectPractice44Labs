import copy
from pathlib import Path
import numpy as np
import pymupdf
import pytest
from PIL import Image
from mlocr.src.cli import _read_pass, _merge_passes, load_pages
from mlocr.src.normalization import normalize_text, enrich
from mlocr.src.semantics import parse_engineering_notation
from mlocr.src.pdf_text import extract_pdf_text_page, merge_pdf_and_ocr, build_title_block
from mlocr.src.adapter import to_drawing
from mlocr.src.glyphs import recover_diameter


@pytest.mark.parametrize('text,expected', [('Сmаль 20', 'Сталь 20'), ('2оmв.', '2отв.'), ('меmка', 'метка'),
                                        ('M12x1.5', 'M12x1.5'), ('10 m', '10 m'), ('diameter', 'diameter'),
                                        ('мама', 'мама'), ('ø20', 'Ø20'), ('φ20', 'Ø20'), ('∅20', 'Ø20'),
                                        ('O/20', 'Ø20'), ('020', '020'), ('Форма', 'Форма')])
def test_corrections(text, expected):
    value, changes = normalize_text(text)
    assert value == expected
    assert bool(changes) == (text != expected)
    record = enrich({'text': text})
    assert record['raw_text'] == text


@pytest.mark.parametrize('text', ['1:0', '-1:5', 'R-2', '46±-0,1', 'M12x0', 'Ø-2', '9'*400])
def test_invalid_numbers(text):
    result = parse_engineering_notation(text)
    assert result['kind'] == 'invalid' and result['warning']


def test_spaces_do_not_join_numbers():
    result = parse_engineering_notation('10 20')
    assert result['nominal'] is None and result['warning']


def test_filtered_polygons_follow_recognition():
    first, second = [[0, 0], [10, 0], [10, 10], [0, 10]], [[90, 0], [100, 0], [100, 10], [90, 10]]
    class Engine:
        def predict(self, image):
            yield {'res': {'dt_polys': [first, second], 'rec_polys': [second], 'rec_texts': ['46'], 'rec_scores': [.9]}}
    result = _read_pass(Engine(), np.zeros((120, 120, 3)), 0)
    assert result[0]['polygon'].tolist() == second


def test_conflicting_numbers_require_review():
    polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    result = _merge_passes([{'text': '46', 'score': .89, 'polygon': polygon, 'pass_angle_deg': 0},
                            {'text': '48', 'score': .90, 'polygon': polygon, 'pass_angle_deg': 90}])[0]
    assert result['needs_review'] and len(result['alternatives']) == 2
    merged = merge_pdf_and_ocr([{'id': 'p', 'text': '46', 'poly_source_px': polygon.tolist()}],
                                [{'id': 'o', 'text': '48', 'poly_source_px': polygon.tolist()}])[0]
    assert merged['needs_review'] and merged['issues']


def test_transparent_png_and_exif(tmp_path):
    rgba = Image.new('RGBA', (20, 10), (0, 0, 0, 0))
    rgba.putpixel((5, 5), (0, 0, 0, 255))
    path = tmp_path/'alpha.png'
    rgba.save(path)
    array = load_pages(path, 1, 300)[0][1]
    assert array[0, 0].tolist() == [255]*3 and array[5, 5].tolist() == [0]*3
    exif = Image.Exif(); exif[274] = 6
    path = tmp_path/'rotated.jpg'
    rgba.convert('RGB').save(path, exif=exif)
    assert load_pages(path, 1, 300)[0][1].shape[:2] == (20, 10)


def test_hidden_pdf_layer_is_not_authority(tmp_path):
    path = tmp_path/'hidden.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=200)
        page.insert_text((50, 60), '99', render_mode=3)
        doc.save(path)
    assert extract_pdf_text_page(path, 0, 72, 200, 200) == []


def test_pdf_pixel_limit_before_render(tmp_path):
    path = tmp_path/'huge.pdf'
    with pymupdf.open() as doc:
        doc.new_page(width=5000, height=5000)
        doc.save(path)
    with pytest.raises(ValueError, match='pixel limit'):
        load_pages(path, 1, 600)


def dimension_case(scale=.5):
    identity = {'page': 'page-1', 'width_px': 200, 'height_px': 100, 'raster_sha256': 'fixture'}
    text = enrich({'id': 't1', 'text': '100', 'poly_source_px': [[23, 11], [27, 11], [27, 13], [23, 13]],
                   'purpose': 'dimension_or_scale_candidate'})
    length = 100*scale
    cv = {'status': 'ok', 'raster_identity': identity, 'coordinate_space': 'sheet_mm', 'scale_status': 'known',
          'calibration': {'view_scale': scale}, 'transforms': {'source_px_to_sheet_mm': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
          'points': [{'id': 'p1', 'xy': [0, 0], 'refs': [{'object_id': 'part', 'anchor': 'start'}]},
                     {'id': 'p2', 'xy': [length, 0], 'refs': [{'object_id': 'part', 'anchor': 'end'}]}],
          'primitives': [{'id': 'part', 'kind': 'line', 'start': [0, 0], 'end': [length, 0]},
                         {'id': 'dim', 'kind': 'line', 'start': [0, 10], 'end': [length, 10], 'layer': 'THIN'},
                         {'id': 'e1', 'kind': 'line', 'start': [0, 0], 'end': [0, 12], 'layer': 'THIN'},
                         {'id': 'e2', 'kind': 'line', 'start': [length, 0], 'end': [length, 12], 'layer': 'THIN'}],
          'objects': {'part': {'role': 'part'}, 'dim': {'role': 'thin'}, 'e1': {'role': 'thin'}, 'e2': {'role': 'thin'}}}
    obs = {'status': 'ok', 'source': identity, 'texts': [text], 'issues': [], 'title_block': {}}
    return obs, cv


@pytest.mark.parametrize('scale', [1, .5, 2])
def test_real_evidence_and_view_scale(scale):
    obs, cv = dimension_case(scale)
    output = to_drawing(obs, cv)
    dimension = output['dimensions'][0]
    assert dimension['nominal_mm'] == 100 and dimension['view_scale'] == scale
    assert dimension['p1_ref'] == 'p1' and dimension['p2_ref'] == 'p2'
    assert output['texts'] == []
    assert output['dimension_links'][0]['confirmed']['evidence']['extension_lines'] == ['e1', 'e2']
    cv['primitives'] = [p for p in cv['primitives'] if p['id'] != 'e2']
    assert to_drawing(obs, cv)['dimensions'] == []


def test_identity_unknown_scale_and_ambiguity():
    obs, cv = dimension_case()
    cv['raster_identity'] = dict(cv['raster_identity'], page='page-2')
    with pytest.raises(ValueError, match='source raster'):
        to_drawing(obs, cv)
    cv['raster_identity'] = obs['source']
    cv['calibration']['view_scale'] = None
    assert not to_drawing(obs, cv)['dimensions']
    cv['calibration']['view_scale'] = .5
    obs['texts'][0]['needs_review'] = True
    assert not to_drawing(obs, cv)['dimensions']


def test_author_row_is_read():
    items = [enrich({'id': 'label', 'text': 'Разраб.', 'poly_source_px': [[550, 720], [600, 720], [600, 735], [550, 735]]}),
             enrich({'id': 'name', 'text': 'Иванов И.И.', 'poly_source_px': [[605, 720], [680, 720], [680, 735], [605, 735]]})]
    result = build_title_block([{'page': 'page-1', 'width_px': 1000, 'height_px': 1000, 'texts': items}])
    assert result['page-1']['fields']['authors']['value'][0]['name'] == 'Иванов И.И.'


def test_dropped_diameter_recovers_from_actual_pixels():
    image = np.asarray(Image.open(Path(__file__).resolve().parents[1]/'evaluation/crops/diameter36-regression.png').convert('RGB'))
    h, w = image.shape[:2]
    item = enrich({'text': '36'})
    recover_diameter(item, image, [[0, 0], [w, 0], [w, h], [0, h]])
    assert item['text'] == 'Ø36' and item['raw_text'] == '36' and item['needs_review']
    assert item['notation']['kind'] == 'diameter'


def test_hole_count_abbreviation_preserves_count_and_recovers_diameter():
    image = np.asarray(Image.open(Path(__file__).resolve().parents[1]/'evaluation/crops/real-2.png').convert('RGB'))
    h, w = image.shape[:2]
    item = enrich({'text': '8omb.4'})
    recover_diameter(item, image, [[0, 0], [w, 0], [w, h], [0, h]])
    assert item['text'] == '8отв.Ø4' and item['raw_text'] == '8omb.4'
    assert item['notation']['nominal'] == 4 and item['notation']['hole_count'] == 8


def test_partial_rotation_reading_is_not_duplicated_but_changed_digit_is_reviewed():
    large = np.array([[0, 0], [100, 0], [100, 15], [0, 15]])
    small = np.array([[60, 0], [100, 0], [100, 15], [60, 15]])
    result = _merge_passes([{'text': 'Ст3 ГОСТ 380-2005', 'score': .9, 'polygon': large, 'pass_angle_deg': 0},
                            {'text': '2005', 'score': .99, 'polygon': small, 'pass_angle_deg': 90}])
    assert len(result) == 1 and result[0]['text'] == 'Ст3 ГОСТ 380-2005'
    assert not result[0].get('needs_review')
    result = _merge_passes([{'text': '100', 'score': .89, 'polygon': large, 'pass_angle_deg': 0},
                            {'text': '1000', 'score': .90, 'polygon': large, 'pass_angle_deg': 90}])
    assert result[0]['needs_review']


def test_text_polygon_corner_order_does_not_rotate_horizontal_text():
    obs, cv = dimension_case()
    obs['texts'][0].update(text='Сталь', poly_source_px=[[10, 20], [10, 10], [50, 10], [50, 20]])
    obs['texts'][0]['notation'] = {'kind': 'unknown'}
    output = to_drawing(obs, cv)['texts'][0]
    assert abs(output['rotation']) < 1e-6 and abs(output['height']-10) < 1e-6


def test_detector_box_is_taller_than_the_text_and_text_stays_in_the_box():
    obs, cv = dimension_case()
    obs['texts'][0].update(text='Сталь', source='paddleocr', poly_source_px=[[10, 10], [50, 10], [50, 20], [10, 20]])
    obs['texts'][0]['notation'] = {'kind': 'unknown'}
    output = to_drawing(obs, cv)['texts'][0]
    assert abs(output['height']-6.2) < 1e-6
    assert abs(output['insert'][1]-(10+(10-6.2)/2)) < 1e-6 and abs(output['insert'][0]-10) < 1e-6


@pytest.mark.parametrize('number', ['036', '836', '8036', '88'])
def test_zeros_and_eights_are_not_diameter_prefixes(number):
    import cv2
    canvas = np.full((90, 240, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, number, (10, 65), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1.8, (0, 0, 0), 2)
    item = enrich({'text': number})
    recover_diameter(item, canvas, [[0, 0], [240, 0], [240, 90], [0, 90]])
    assert item['text'] == number


def _shared_corners_case(offset):
    """Both measured points are corners of two part lines; no line runs between them."""
    identity = {'page': 'page-1', 'width_px': 200, 'height_px': 100, 'raster_sha256': 'fixture'}
    text = enrich({'id': 't1', 'text': '50', 'poly_source_px': [[23, 11], [27, 11], [27, 13], [23, 13]],
                   'purpose': 'dimension_or_scale_candidate'})
    cv = {'status': 'ok', 'raster_identity': identity, 'coordinate_space': 'sheet_mm', 'scale_status': 'known',
          'calibration': {'view_scale': 1}, 'transforms': {'source_px_to_sheet_mm': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
          'points': [{'id': 'p1', 'xy': [0, 0], 'refs': [{'object_id': 'a', 'anchor': 'start'}, {'object_id': 'b', 'anchor': 'end'}]},
                     {'id': 'p2', 'xy': [50, 0], 'refs': [{'object_id': 'c', 'anchor': 'start'}, {'object_id': 'd', 'anchor': 'end'}]}],
          'primitives': [{'id': 'a', 'kind': 'line', 'start': [0, 0], 'end': [0, -5]},
                         {'id': 'b', 'kind': 'line', 'start': [-5, 0], 'end': [offset, 0]},
                         {'id': 'c', 'kind': 'line', 'start': [50, 0], 'end': [50, -5]},
                         {'id': 'd', 'kind': 'line', 'start': [55, 0], 'end': [50, 0]},
                         {'id': 'dim', 'kind': 'line', 'start': [0, 10], 'end': [50, 10]},
                         {'id': 'e1', 'kind': 'line', 'start': [0, 0], 'end': [0, 12]},
                         {'id': 'e2', 'kind': 'line', 'start': [50, 0], 'end': [50, 12]}],
          'objects': {'a': {'role': 'part'}, 'b': {'role': 'part'}, 'c': {'role': 'part'}, 'd': {'role': 'part'},
                      'dim': {'role': 'thin'}, 'e1': {'role': 'thin'}, 'e2': {'role': 'thin'}}}
    return {'status': 'ok', 'source': identity, 'texts': [text], 'issues': [], 'title_block': {}}, cv


def test_shared_corner_links_when_its_anchors_coincide_and_not_when_they_diverge():
    obs, cv = _shared_corners_case(0.0)
    assert len(to_drawing(obs, cv)['dimensions']) == 1
    obs, cv = _shared_corners_case(0.1)                  # the lines really stay apart: which end is measured is unknown
    assert to_drawing(obs, cv)['dimensions'] == []


def test_crowded_sheet_does_not_hit_the_limit_with_points_that_cannot_anchor():
    obs, cv = dimension_case()
    cv['objects']['thin_many'] = {'role': 'thin'}
    cv['points'] += [{'id': f'x{k}', 'xy': [k * .1, 40], 'refs': [{'object_id': 'thin_many', 'anchor': 'mid'}]} for k in range(900)]
    assert not any(i['code'] == 'dimension_link_limit' for i in to_drawing(obs, cv)['issues'])
