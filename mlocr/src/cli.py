"""Minimal offline OCR CLI for image crops and rendered PDF pages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image
from .inputs import iter_pages, page_identity
from .normalization import enrich, normalize_text
from .assets import local_model_dirs
from .glyphs import recover_diameter
from functools import lru_cache
import os
import time
from .semantics import classify_purpose, parse_engineering_notation
from .pdf_text import build_title_block, extract_pdf_text_page, merge_pdf_and_ocr
from .dimension_links import propose_dimension_links


def load_pages(path: Path, page_number: int | None, dpi: int) -> list[tuple[str, np.ndarray]]:
    return list(iter_pages(path, page_number, dpi))


def _box_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    ax0, ay0 = poly_a.min(axis=0)
    ax1, ay1 = poly_a.max(axis=0)
    bx0, by0 = poly_b.min(axis=0)
    bx1, by1 = poly_b.max(axis=0)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def _read_pass(engine, image: np.ndarray, pass_angle: int) -> list[dict]:
    height, _ = image.shape[:2]
    if pass_angle == 90:
        passed_image = np.ascontiguousarray(np.rot90(image, k=3))
    else:
        passed_image = image
    records = []
    for batch in engine.predict(passed_image):
        data = batch.json if hasattr(batch, "json") else batch
        if isinstance(data, dict):
            data = data.get("res", data)
        if not isinstance(data, dict):
            continue
        texts, scores = data.get("rec_texts", []), data.get("rec_scores", [])
        polys = data.get("rec_polys")
        if polys is None:
            polys = data.get("dt_polys", [])
        if len(texts) != len(polys) or len(texts) != len(scores):
            raise ValueError('OCR text/score/polygon arrays do not match')
        for index, raw in enumerate(texts):
            text = str(raw).strip()
            if not text or index >= len(polys):
                continue
            polygon = np.asarray(polys[index], dtype=float)
            if pass_angle == 90:
                # Inverse of np.rot90(..., k=3), i.e. clockwise image rotation.
                polygon = np.column_stack((polygon[:, 1], height - 1 - polygon[:, 0]))
            score = float(scores[index]) if index < len(scores) else 0.0
            records.append({"text": text, "score": score, "polygon": polygon,
                            "pass_angle_deg": pass_angle})
    return records


def _merge_passes(records: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for record in records:
        duplicate = next((item for item in merged if _box_iou(item["polygon"], record["polygon"]) >= 0.45
                          or _partial_reading(item, record) or _partial_reading(record, item)), None)
        alternative = {"text": record["text"], "score": record["score"],
                       "pass_angle_deg": record["pass_angle_deg"]}
        if duplicate is None:
            record["alternatives"] = [alternative]
            merged.append(record)
            continue
        if not any(a["text"] == record["text"] and a["pass_angle_deg"] == record["pass_angle_deg"]
                   for a in duplicate["alternatives"]):
            duplicate["alternatives"].append(alternative)
        if _partial_reading(record, duplicate):
            alternative['partial'] = True
            continue
        if _partial_reading(duplicate, record):
            for old in duplicate['alternatives']:
                if old is not alternative:
                    old['partial'] = True
            replace = True
        else:
            replace = record['score'] > duplicate['score']
        if replace:
            alternatives = duplicate["alternatives"]
            duplicate.update(record)
            duplicate["alternatives"] = alternatives
    for item in merged:
        # A failed rotated pass is not an equally plausible reading. Keep it in
        # the evidence, but flag conflicts between credible alternatives.
        credible = [a for a in item['alternatives'] if not a.get('partial')
                    and a['score'] >= max(.7, item['score']-.15)]
        readings = {normalize_text(a['text'])[0] for a in credible}
        if len(readings) > 1:
            item['needs_review'] = True
            item['issues'] = [{'code': 'ocr_reading_conflict', 'message': 'Orientation passes disagree; confirm the source text.'}]
        if item['score'] < .7:
            item['needs_review'] = True
            item.setdefault('issues', []).append({'code': 'ocr_low_confidence', 'message': 'Low-confidence recognition; confirm the source text.'})
    return merged


def _partial_reading(small, large):
    a = ''.join(normalize_text(small['text'])[0].split())
    b = ''.join(normalize_text(large['text'])[0].split())
    if not a or a == b or a not in b or small['pass_angle_deg'] == large['pass_angle_deg']:
        return False
    low, high = small['polygon'].min(axis=0), small['polygon'].max(axis=0)
    left, right = large['polygon'].min(axis=0), large['polygon'].max(axis=0)
    overlap = np.maximum(0, np.minimum(high, right)-np.maximum(low, left))
    area = np.prod(high-low)
    large_area = np.prod(right-left)
    return bool(area > 0 and area < .8*large_area and np.prod(overlap)/area >= .85)


@lru_cache(maxsize=2)
def get_engine(det=None, rec=None):
    det, rec = local_model_dirs(det, rec)
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    from paddleocr import PaddleOCR
    return PaddleOCR(device='cpu', text_detection_model_name='PP-OCRv5_mobile_det',
                     text_detection_model_dir=str(det), text_recognition_model_name='eslav_PP-OCRv5_mobile_rec',
                     text_recognition_model_dir=str(rec), use_doc_orientation_classify=False,
                     use_doc_unwarping=False, use_textline_orientation=False,
                     text_det_limit_side_len=4096, text_det_limit_type='max')


def recognize_file(path, *, page=1, dpi=300, det=None, rec=None, crop_origin=None, source_size=None):
    from mlocr import __version__
    started = time.perf_counter()
    path = Path(path)
    if bool(crop_origin) != bool(source_size):
        raise ValueError('crop-origin and source-size must be provided together')
    result_pages = []
    # Validate/load a page before loading the model; render one page at a time.
    for page_id, image in iter_pages(path, page, dpi):
        engine = get_engine(det, rec)
        height, width = image.shape[:2]
        source_width, source_height = source_size or (width, height)
        if crop_origin and (min(crop_origin) < 0 or crop_origin[0]+width > source_width or crop_origin[1]+height > source_height):
            raise ValueError('Crop exceeds source image bounds')
        observations = _merge_passes(_read_pass(engine, image, 0) + _read_pass(engine, image, 90))
        items = []
        for item in observations:
            polygon = item['polygon'].copy()
            if crop_origin:
                polygon += np.asarray(crop_origin, dtype=float)
            edges = np.roll(polygon, -1, axis=0) - polygon
            delta = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
            angle = float((np.degrees(np.arctan2(delta[1], delta[0])) + 90) % 180 - 90)
            poly = polygon.tolist()
            record = enrich({'id': f'{page_id}-ocr-{len(items)+1:04d}', 'page': page_id,
                             'text': item['text'], 'score': item['score'], 'source': 'paddleocr',
                             'alternatives': item['alternatives'], 'angle_deg': angle,
                             'selection_basis': 'highest_recognition_confidence',
                             'needs_review': item.get('needs_review', False), 'issues': item.get('issues', []),
                             'polygon_px': poly, 'poly_source_px': poly})
            recover_diameter(record, image, item['polygon'])
            record['purpose'] = classify_purpose(record['text'], poly, source_width, source_height)
            items.append(record)
        if path.suffix.lower() == '.pdf':
            pdf = extract_pdf_text_page(path, int(page_id.split('-')[1])-1, dpi, width, height)
            items = merge_pdf_and_ocr(pdf, items)
        identity = page_identity(image, page_id)
        if crop_origin:
            identity.update(raster_sha256=None, crop_origin=crop_origin, width_px=source_width, height_px=source_height)
        result_pages.append({'page': page_id, 'source': identity, 'width_px': source_width, 'height_px': source_height, 'texts': items})
    issues = [{'code': issue['code'], 'message': issue['message'], 'severity': 'warning', 'object_id': item['id']}
              for p in result_pages for item in p['texts'] for issue in item.get('issues', [])]
    return {'schema_version': 'ocr-observations-1', 'status': 'ok', 'module': {'name': 'mlocr', 'version': __version__},
            'input': str(path), 'source': result_pages[0]['source'] if len(result_pages) == 1 else None,
            'pages': result_pages, 'texts': [t for p in result_pages for t in p['texts']],
            'title_block': build_title_block(result_pages), 'issues': issues,
            'params': {'pdf_render_dpi': dpi if path.suffix.lower() == '.pdf' else None, 'det_limit_side_len': 4096, 'passes': [0, 90]},
            'timings_s': {'total': round(time.perf_counter()-started, 3)}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PaddleOCR on an image or PDF")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ocr.json"))
    parser.add_argument("--page", type=int, help="1-based page number for PDF")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rasterization resolution")
    parser.add_argument("--crop-origin", type=int, nargs=2, metavar=("X", "Y"),
                        help="Top-left pixel of an image crop within its source page")
    parser.add_argument("--source-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Full source-page pixel size when input is a crop")
    parser.add_argument("--cv-json", type=Path,
                        help="Optional CV JSON from mlcv; adds review-only dimension-link proposals")
    parser.add_argument("--det-model-dir", type=Path, help="Local PP-OCRv5 detection model directory")
    parser.add_argument("--rec-model-dir", type=Path, help="Local PP-OCRv5 East Slavic recognition model directory")
    args = parser.parse_args()
    if bool(args.crop_origin) != bool(args.source_size):
        parser.error("--crop-origin and --source-size must be provided together")

    payload = recognize_file(args.input, page=args.page, dpi=args.dpi,
                             det=args.det_model_dir, rec=args.rec_model_dir,
                             crop_origin=args.crop_origin, source_size=args.source_size)
    if args.cv_json:
        if len(payload['pages']) != 1:
            parser.error('CV linking requires exactly one matching page')
        cv_data = json.loads(args.cv_json.read_text(encoding='utf-8'))
        from .adapter import to_drawing
        payload['drawing'] = to_drawing(payload, cv_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
