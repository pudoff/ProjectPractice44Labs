import json
import pytest
from drawcv.pipeline import _lineweights, run
from drawcv import Params
from PIL import Image


def test_frame_dominated_stroke_estimate_does_not_hide_thin_body_lines(monkeypatch):
    monkeypatch.setattr('drawcv.pipeline._in_body', lambda entity, calibration: True)
    prims = [{'length_mm': length, 'support': 1., 'stroke_mm': width}
             for width, length in [(0.24, 17), (0.24, 17), (0.24, 50), (0.73, 50), (0.73, 50), (0.73, 20), (0.73, 20)]]
    separable, thick, threshold = _lineweights(prims, .73, {})
    assert separable and .24 < threshold < .73 and thick == .73


def test_observation_mask_from_another_page_is_rejected_before_geometry(tmp_path):
    image = tmp_path/'page.png'
    Image.new('RGB', (200, 300), 'white').save(image)
    mask = tmp_path/'ocr.json'
    mask.write_text(json.dumps({'schema_version': 'ocr-observations-1', 'pages': [{'page': 'page-2'}],
                                'source': {'page': 'page-2'}, 'texts': []}))
    with pytest.raises(ValueError, match='another raster/page'):
        run(image, params=Params(text_provider='file', ocr_json=str(mask)))
