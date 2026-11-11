"""Read-only QA scoring. Partial worksheets measure only their declared regions.

python -m mlocr.evaluation.score_qa_task1 --results DIR --out NEW_RUN_DIR
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from mlocr.evaluation.matching import assign_pairs, edit_distance
from mlocr.src.glyphs import recover_diameter
from mlocr.src.inputs import assert_same_page, iter_pages, page_identity
from mlocr.src.normalization import enrich
from mlocr.src.semantics import parse_engineering_notation

ROOT = Path(__file__).resolve().parents[1]


def bounds(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_units(texts):
    """A complete cell split is one prediction for its original reviewed region."""
    grouped, units = {}, []
    for item in texts:
        parent = item.get('split_from')
        if parent:
            key = json.dumps(parent['polygon'])
            if key not in grouped:
                grouped[key] = {'box': bounds(parent['polygon']), 'items': [], 'method': 'gridline_split'}
                units.append(grouped[key])
            grouped[key]['items'].append(item)
        else:
            units.append({'box': bounds(item['poly_source_px']), 'items': [item], 'method': 'polygon_iou'})
    for unit in units:
        unit['items'].sort(key=lambda item: bounds(item['poly_source_px'])[0])
        unit['text'] = ' '.join(item['text'] for item in unit['items'])
    return units


def score(worksheet, results_dir, images_dir, out, *, complete_pages=False):
    worksheet, results_dir, images_dir, out = map(Path, (worksheet, results_dir, images_dir, out))
    if out.exists():
        raise ValueError(f'Use a new output directory; refusing to overwrite {out}')
    with worksheet.open(encoding='utf-8-sig', newline='') as stream:
        qa_rows = list(csv.DictReader(stream))
    if not qa_rows:
        raise ValueError('QA worksheet is empty')
    files = sorted(results_dir.glob('*.ocr.json'))
    if not files:
        raise ValueError(f'No *.ocr.json observations found in {results_dir}')
    if complete_pages and any(not row.get('ground_truth', '').strip() for row in qa_rows):
        raise ValueError('Complete-page evaluation requires a label for every reference region')
    results, provenance = {}, []
    for path in files:
        payload = json.loads(path.read_text(encoding='utf-8'))
        name = Path(payload['input'].replace('\\', '/')).name
        if name in results:
            raise ValueError(f'Multiple OCR runs for the same image: {name}')
        image_path = images_dir / name
        _, image = next(iter_pages(image_path, 1, 300))
        identity = page_identity(image)
        assert_same_page(payload, {'raster_identity': identity})
        normalized = deepcopy(payload)
        for item in normalized['texts']:
            raw = item.get('raw_text', item['text'])
            item['text'] = raw
            enrich(item)
            recover_diameter(item, image, item['poly_source_px'])
        results[name] = {'payload': normalized, 'units': prediction_units(normalized['texts']), 'path': path}
        provenance.append({'image': name, 'image_file_sha256': digest(image_path), 'source': identity,
                           'observations_file': path.name, 'observations_sha256': digest(path),
                           'module': payload.get('module'), 'params': payload.get('params'),
                           'status': payload.get('status'), 'issues': payload.get('issues', [])})
    reviewed_images = {row['image'] for row in qa_rows}
    if reviewed_images != set(results):
        raise ValueError(f'Worksheet/results image sets differ: missing={sorted(reviewed_images - set(results))}, '
                         f'unreviewed={sorted(set(results) - reviewed_images)}')
    comparison, unassigned, unassigned_chars = [], 0, 0
    for name in sorted(reviewed_images):
        page = results[name]
        rows = [row for row in qa_rows if row['image'] == name]
        units = page['units']
        candidates = []
        for index, row in enumerate(rows):
            source = page['payload']['source']
            if row.get('page') and row['page'] != source['page']:
                raise ValueError(f'Worksheet/OCR page differs: {name}')
            if row.get('raster_sha256') and row['raster_sha256'] != source['raster_sha256']:
                raise ValueError(f'Worksheet/OCR raster hash differs: {name}')
            box = tuple(map(float, row['bbox_xyxy_px'].split(',')))
            for unit_index, unit in enumerate(units):
                overlap = iou(box, unit['box'])
                if overlap >= .15:
                    candidates.append((index, unit_index, 1 - overlap))
        matches = dict(assign_pairs(len(rows), len(units), candidates))
        assigned_units = set(matches.values())
        unassigned += len(units) - len(assigned_units)
        unassigned_chars += sum(len(unit['text']) for index, unit in enumerate(units) if index not in assigned_units)
        for index, row in enumerate(rows):
            unit = units[matches[index]] if index in matches else None
            items = unit['items'] if unit else []
            predicted, truth = unit['text'].strip() if unit else '', row.get('ground_truth', '').strip()
            corrections = [change for item in items for change in item.get('corrections', [])]
            issues = sorted({issue['code'] for item in items for issue in item.get('issues', [])})
            notation, expected = parse_engineering_notation(predicted), parse_engineering_notation(truth)
            comparison.append({'image': name, 'page': row.get('page'), 'text_id': row.get('text_id'),
                'initial_raw': row.get('raw_text'), 'ground_truth': truth, 'final_after_rules': predicted,
                'exact_match': predicted == truth if truth else None,
                'match_method': unit['method'] if unit else 'unmatched',
                'char_errors': edit_distance(predicted, truth) if truth else None,
                'reference_chars': len(truth), 'needs_review': any(item.get('needs_review') for item in items),
                'numeric_error': (notation.get('nominal') != expected['nominal']) if expected.get('nominal') is not None else None,
                'issues': issues, 'corrections': corrections,
                'alternatives': [item.get('alternatives', []) for item in items], 'qa_note': row.get('qa_note', '')})
    labelled = [row for row in comparison if row['ground_truth']]
    if not labelled:
        raise ValueError('No non-empty ground truth labels; quality cannot be scored')
    errors = sum(row['char_errors'] for row in labelled) + (unassigned_chars if complete_pages else 0)
    characters = sum(row['reference_chars'] for row in labelled)
    numeric = [row for row in labelled if row['numeric_error'] is not None]
    italic = [row for row in labelled if any(c.get('rule') == 'cyrillic_italic_t' for c in row['corrections'])]
    metrics = {'scope': 'complete_pages_explicitly_declared' if complete_pages else 'reviewed_regions_only',
        'sheets': len(results), 'qa_rows': len(comparison), 'labelled_rows': len(labelled),
        'exact_text_rows': sum(row['exact_match'] for row in labelled),
        'exact_text_accuracy': sum(row['exact_match'] for row in labelled) / len(labelled),
        'missed_labelled_rows': sum(row['match_method'] == 'unmatched' for row in labelled),
        'unassigned_prediction_regions': unassigned,
        'false_positive_regions': unassigned if complete_pages else None,
        'character_errors': errors, 'reference_characters': characters, 'cer': errors / characters,
        'numeric_labelled_rows': len(numeric), 'numeric_errors': sum(row['numeric_error'] for row in numeric),
        'diameter_ground_truth_rows': sum('Ø' in row['ground_truth'] for row in labelled),
        'diameter_missed': sum('Ø' in row['ground_truth'] and 'Ø' not in row['final_after_rules'] for row in labelled),
        'false_diameter_additions': sum('Ø' not in row['ground_truth'] and 'Ø' in row['final_after_rules'] for row in labelled),
        'labelled_italic_t_corrections': len(italic), 'italic_t_wrong_rows': sum(not row['exact_match'] for row in italic),
        'worksheet_sha256': digest(worksheet), 'provenance': provenance}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'normalized').mkdir()
    for page in results.values():
        (out / 'normalized' / page['path'].name).write_text(json.dumps(page['payload'], ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'task1_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'task1_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                         for row in comparison)
    (out / 'task1_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worksheet', type=Path, default=ROOT / 'qa_samples/qa_review.csv')
    parser.add_argument('--results', type=Path, default=ROOT / 'results/qa_samples/task1_final')
    parser.add_argument('--images', type=Path, default=ROOT / 'qa_samples')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/qa_samples' / datetime.now(timezone.utc).strftime('score-%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--complete-pages', action='store_true', help='QA explicitly confirms every text region on each page is labelled')
    args = parser.parse_args()
    try:
        result = score(args.worksheet, args.results, args.images, args.out, complete_pages=args.complete_pages)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
