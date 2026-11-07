"""Repeatable small OCR baseline. Reviewer labels still need CAD/QA acceptance."""
import argparse
import json
import re
from pathlib import Path
from mlocr.src.cli import recognize_file


def distance(a, b):
    row = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        new = [i]
        for j, right in enumerate(b, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(left != right)))
        row = new
    return row[-1]


def comparable(text):
    return re.sub(r'\s+', '', text).replace('–', '-').replace('—', '-')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, default=Path(__file__).resolve().parents[1]/'mlocr/evaluation/cases.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for case in json.loads(args.cases.read_text(encoding='utf-8')):
        output = recognize_file(args.cases.parent/case['image'])
        (args.out/(case['id']+'.json')).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        # Single-line or rotated single-label crops; deterministic order by boxes.
        items = sorted(output['texts'], key=lambda t: (min(p[0] for p in t['poly_source_px']), min(p[1] for p in t['poly_source_px'])))
        actual = ' '.join(t['text'] for t in items)
        raw = ' '.join(t['raw_text'] for t in items)
        expected = comparable(case['expected'])
        records.append({**case, 'raw': raw, 'actual': actual, 'exact': comparable(actual) == expected,
                        'errors': distance(expected, comparable(actual)), 'raw_errors': distance(expected, comparable(raw)),
                        'reference_characters': len(expected), 'seconds': output['timings_s']['total']})
    chars = sum(r['reference_characters'] for r in records)
    result = {'cases': records, 'normalization_for_metrics': 'ignore whitespace and dash typography only',
              'exact_lines': sum(r['exact'] for r in records), 'total_lines': len(records),
              'cer': sum(r['errors'] for r in records)/chars, 'raw_cer': sum(r['raw_errors'] for r in records)/chars,
              'scope': 'Three reviewer-labelled real crops; not independent CAD acceptance or whole-dataset accuracy.'}
    (args.out/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
