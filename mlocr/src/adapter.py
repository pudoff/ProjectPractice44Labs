"""Observation adapter and a separate, geometry-aware DrawingGraph projection."""
import math
from copy import deepcopy
import numpy as np
from .cli import recognize_file
from .inputs import assert_same_page
from .dimension_links import propose_dimension_links

# The PaddleOCR detector box is taller than the characters: reference TEXT height / box height is 0.60-0.65
# (5 mm text has a 9 mm box). With the box height the text in the DXF runs over its neighbours.
DETECTOR_BOX_TO_TEXT_HEIGHT = 0.62


def run_ocr(image):
    return recognize_file(image)


def to_drawing(observations, cv):
    assert_same_page(observations, cv)
    result = {key: deepcopy(observations[key]) for key in ('status', 'module', 'source', 'title_block', 'params', 'timings_s', 'issues') if key in observations}
    result.update(schema_version='0.2', texts=[], dimensions=[], observations=deepcopy(observations['texts']))
    matrix = cv.get('transforms', {}).get('source_px_to_sheet_mm')
    if not matrix or cv.get('scale_status') != 'known':
        result['issues'].append({'code': 'ocr_unknown_sheet_scale', 'severity': 'error', 'message': 'Text cannot be positioned in millimetres until sheet calibration is known.'})
        return result
    # Only a single unambiguous scale is usable for the MVP single-view case.
    # Multiple printed scales require explicit assignment of a scale to each view.
    scales = {t['notation']['nominal']/t['notation']['scale_denominator'] for t in observations['texts']
              if t.get('notation', {}).get('kind') == 'scale' and not t.get('needs_review') and not t['notation'].get('warning')}
    if cv.get('calibration', {}).get('view_scale') is None and len(scales) == 1:
        cv.setdefault('calibration', {})['view_scale'] = next(iter(scales))
        cv['calibration']['view_scale_basis'] = 'single_ocr_scale_candidate'
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid CV coordinate transform')
    for item in observations['texts']:
        poly = np.asarray(item['poly_source_px'], dtype=float)
        if poly.shape != (4, 2) or not np.isfinite(poly).all():
            raise ValueError('Invalid OCR text polygon')
        xy = np.column_stack((poly, np.ones(4))) @ matrix.T
        edges = np.roll(xy[:, :2], -1, axis=0)-xy[:, :2]
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        angle = (math.degrees(math.atan2(edge[1], edge[0]))+90) % 180-90
        u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        v = np.array([-u[1], u[0]])
        along, across = xy[:, :2] @ u, xy[:, :2] @ v
        box_height = float(np.ptp(across))
        if box_height <= 0:
            continue
        height = box_height*(DETECTOR_BOX_TO_TEXT_HEIGHT if item.get('source') == 'paddleocr' else 1)
        insert = u*along.min()+v*(across.min()+(box_height-height)/2)   # stays in the middle of its box
        result['texts'].append({'id': item['id'], 'value': item['text'], 'insert': insert.tolist(),
                                'height': height, 'rotation': angle})
    try:
        links = propose_dimension_links(observations, cv)
    except ValueError as exc:
        result['issues'].append({'code': 'dimension_link_limit', 'severity': 'warning', 'message': str(exc)})
        links = []
    result['dimension_links'] = links
    linked = set()
    for link in links:
        chosen = link['confirmed']
        if chosen is None:
            continue
        evidence = chosen['evidence']
        point_ids = []
        primitives = {p['id']: p for p in cv['primitives']}
        for index, (original, anchor) in enumerate(zip((chosen['p1_id'], chosen['p2_id']), evidence['part_anchors'])):
            point = next(p for p in cv['points'] if p['id'] == original)
            if point.get('refs') == [anchor]:
                point_ids.append(original)
                continue
            # CV clusters near endpoints; backend requires an exact anchor.
            # Use the identified measured part edge, never average two distinct
            # coordinates or silently move the underlying geometry.
            identity = f"ocr-anchor-{link['text_id']}-{index}"
            xy = primitives[anchor['object_id']][anchor['anchor']]
            cv['points'] = [p for p in cv['points'] if p['id'] != identity]
            cv['points'].append({'id': identity, 'xy': xy, 'kind': 'ocr_dimension_anchor', 'refs': [anchor]})
            point_ids.append(identity)
        link['resolved_point_ids'] = point_ids
        result['dimensions'].append({'id': 'dim-'+link['text_id'], 'kind': 'linear', 'p1_ref': point_ids[0],
                                     'p2_ref': point_ids[1], 'base': evidence['base'], 'angle': evidence['angle'],
                                     'nominal_mm': link['nominal_mm'], 'recognized_text': link['recognized_text'],
                                     'view_scale': chosen['view_scale'], 'link_status': 'resolved'})
        for object_id in [evidence['dimension_line'], *evidence['extension_lines']]:
            cv['objects'][object_id]['represented_by_dimension'] = 'dim-'+link['text_id']
        linked.add(link['text_id'])
    result['texts'] = [t for t in result['texts'] if t['id'] not in linked]
    for item in observations['texts']:
        kind = item.get('notation', {}).get('kind')
        if kind in {'linear', 'linear_tolerance', 'diameter', 'radius'} and item['id'] not in linked and item.get('purpose') != 'title_block_candidate':
            result['issues'].append({'code': 'dimension_unlinked', 'severity': 'warning', 'object_id': item['id'],
                                     'message': 'Text preserved; dimension geometry/scale is not uniquely confirmed.'})
    return result
