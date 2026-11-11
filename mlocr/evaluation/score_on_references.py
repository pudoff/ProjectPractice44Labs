"""OCR quality against reference DXFs: text readings and title-block fields per sheet.

    python mlocr/evaluation/score_on_references.py [--raw mlocr/evaluation/out/raw] [--cv-dirs mlcv/eval/out ...] [--json F]

Readings come from collect_observations.py and are normalised again by the current code, so a change of the rules is
measured without running the models. The sheet position of a text comes from cv.json of the same sheet; the reference
text at the same place is the truth. Reference title blocks are read from the GOST 2.104 form 1 cells."""
import argparse
import collections
import copy
import difflib
import json
import re
import sys
from pathlib import Path

import ezdxf
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from compare_dxf import iter_flat  # noqa: E402
from mlocr.src.normalization import enrich  # noqa: E402
from mlocr.src.inputs import assert_same_page  # noqa: E402
from mlocr.evaluation.matching import assign_pairs, edit_distance  # noqa: E402
from mlocr.src.title_block import CELLS, LABEL, LEADING_LABEL, SURNAME_COLUMN, SLACK, block_xy, in_cell, title_block_from_cells  # noqa: E402

DXF = ROOT / "dataset/clear_sum_dataset/vse_dxf"
PLACE_MM = 6.0
FOLD = str.maketrans("AaBCcEeHKMOoPpTXxYy", "АаВСсЕеНКМОоРрТХхУу")


def canon(text):
    """Comparable form: DXF control codes, diameter glyphs, look-alike Latin letters, spaces and case do not count."""
    text = re.sub(r"\\[A-Za-z][^;]*;|[{}]", "", text or "")
    for a, b in (("%%c", "Ø"), ("%%C", "Ø"), ("%%d", "°"), ("%%p", "±"), ("⌀", "Ø"), ("∅", "Ø"), ("ø", "Ø"), (",", ".")):
        text = text.replace(a, b)
    return re.sub(r"\s+", "", text.translate(FOLD)).lower()


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def reference_texts(path):
    doc = ezdxf.readfile(str(path))
    out = []
    for e in iter_flat(doc.modelspace()):
        t = e.dxftype()
        try:
            if t == "TEXT":
                value, h = e.dxf.text, float(e.dxf.height)
                align = e.get_placement()[1] if hasattr(e, "get_placement") else e.dxf.insert
                x, y = np.array(align)[:2]
            elif t == "MTEXT":
                value, h = e.plain_text(), float(e.dxf.char_height)
                x, y = np.array(e.dxf.insert)[:2]
            else:
                continue
        except Exception:
            continue
        if value and value.strip():
            out.append({"text": value.strip(), "xy": np.array([x, y], float), "height": h})
    return out


def text_extent(item, matrix):
    poly = np.array(item["poly_source_px"], float)
    return (np.column_stack([poly, np.ones(len(poly))]) @ np.array(matrix).T)[:, :2]


def score_sheet(raw, cv, metrics, refs, normalise=True):
    assert_same_page(raw, cv)
    matrix = cv["transforms"]["source_px_to_sheet_mm"]
    t = np.array(metrics["registration"]["translation_mm"])
    readings, candidates = [], []
    for source in raw["texts"]:
        item = copy.deepcopy(source)
        if normalise:
            item["text"] = item.get("raw_text", item["text"])
            enrich(item)
        poly = text_extent(item, matrix) + t
        low, high = poly.min(axis=0) - PLACE_MM, poly.max(axis=0) + PLACE_MM
        index = len(readings)
        readings.append(item)
        for ref_index, ref in enumerate(refs):
            if np.all(ref['xy'] >= low) and np.all(ref['xy'] <= high):
                candidates.append((index, ref_index, float(np.linalg.norm(poly.mean(axis=0) - ref['xy']))))
    matches = assign_pairs(len(readings), len(refs), candidates)
    by_reference = {ref_index: index for index, ref_index in matches}
    rows = []
    for ref_index, ref in enumerate(refs):
        item = readings[by_reference[ref_index]] if ref_index in by_reference else None
        ours, truth = canon(item['text']) if item else '', canon(ref['text'])
        rows.append({'ocr': item['text'] if item else '', 'raw': item.get('raw_text') if item else None,
                     'ref': ref['text'], 'sim': round(similarity(ours, truth), 3),
                     'exact': bool(item) and ours == truth, 'kind': item.get('notation', {}).get('kind', 'unknown') if item else 'missed',
                     'match_status': 'matched' if item else 'missed', 'char_errors': edit_distance(ours, truth),
                     'reference_chars': len(truth)})
    used = {index for index, _ in matches}
    for index, item in enumerate(readings):
        if index not in used:
            rows.append({'ocr': item['text'], 'raw': item.get('raw_text'), 'ref': '', 'sim': 0., 'exact': False,
                         'kind': item.get('notation', {}).get('kind', 'unknown'), 'match_status': 'extra',
                         'char_errors': len(canon(item['text'])), 'reference_chars': 0})
    return rows


def reference_fields(refs, cv, translation):
    """Reference values of title-block cells. Same layout as the OCR rule: only a filled cell can be scored."""
    W, H = cv["calibration"]["sheet_mm"]
    fields = collections.defaultdict(list)
    for r in refs:
        p = r["xy"] - translation
        for name in CELLS:
            if not in_cell(name, p, W, H):
                continue
            text = LEADING_LABEL.sub("", r["text"].strip())
            if name == "authors":
                # surnames only; part of the dataset masks them with x
                if not (block_xy(p, W, H)[0] <= SURNAME_COLUMN[1] + SLACK) or LABEL.fullmatch(text) or re.fullmatch(r"[xXхХ\s]*", text):
                    continue
            elif name == "document_number" and not re.search(r"\d", text):
                continue
            fields[name].append(text)
    return fields


def file_name_truth(name):
    """Dataset files are named after the drawing: "260620-КМ3.18-00.01.16 - Ребро", "1579138_02", "переходник"."""
    if " - " in name:
        number, title = name.split(" - ", 1)
        return {"document_number": number.strip(), "name": title.strip()}
    if re.match(r"\d{5}", name):
        return {"document_number": re.sub(r"[._]?(Нов|нов).*$", "", name).replace("_", ".")}
    word = re.sub(r"[\s_\-.\d]+$", "", name)
    return {"name": word} if word and not re.search(r"\d", word) else {}


def contains_score(ours, truth):
    """1 when the truth is a part of the field (a field may hold more, e.g. a second line of the name)."""
    a, b = canon(ours or ""), canon(truth)
    if not a:
        return 0.0
    if b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b)).size / max(len(b), 1)


def field_score(ours, truth):
    if not truth:
        return None
    if ours is None:
        return 0.0
    if isinstance(ours, list):
        ours = " ".join(x["name"] if isinstance(x, dict) else str(x) for x in ours)
    joined = canon(" ".join(truth))
    a = canon(str(ours))
    return similarity(a, joined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "mlocr/evaluation/out/raw"))
    ap.add_argument("--cv-dirs", nargs="*", default=[str(ROOT / f"mlcv/eval/{d}") for d in ("out", "out_random", "out_random7", "out_random101", "out_three_sheets")])
    ap.add_argument("--json")
    ap.add_argument("--show", type=int, default=0, help="print N worst differing readings")
    ap.add_argument('--dxf-dir', type=Path, default=DXF, help='Current QA reference DXF directory')
    a = ap.parse_args()
    cases = {}
    for d in a.cv_dirs:
        for m in Path(d).glob("*.png/metrics.json"):
            cases.setdefault(m.parent.name[:-4], m.parent)
    totals = collections.Counter()
    kinds = collections.defaultdict(collections.Counter)
    diffs = collections.Counter()
    fields_total = collections.defaultdict(list)
    per_sheet = []
    file_truth = collections.defaultdict(list)
    filled = collections.Counter()
    for raw_path in sorted(Path(a.raw).glob("*.json")):
        name = raw_path.stem
        if name not in cases:
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        cv = json.loads((cases[name] / "cv.json").read_text(encoding="utf-8"))
        metrics = json.loads((cases[name] / "metrics.json").read_text(encoding="utf-8"))
        refs = reference_texts(a.dxf_dir / f"{name}.dxf")
        rows = score_sheet(raw, cv, metrics, refs)
        totals['ocr_failed_sheets'] += raw.get('status') != 'ok'
        for r in rows:
            totals['reference'] += r['match_status'] != 'extra'
            totals['predicted'] += r['match_status'] != 'missed'
            totals[r['match_status']] += 1
            totals["exact"] += r["exact"]
            totals['char_errors'] += r['char_errors']
            totals['reference_chars'] += r['reference_chars']
            kinds[r["kind"]]["n"] += 1
            kinds[r["kind"]]["exact"] += r["exact"]
            if not r["exact"]:
                diffs[(r["ocr"], r["ref"])] += 1
        totals["sum_sim"] += sum(r["sim"] for r in rows)
        fields = title_block_from_cells(_normalised(raw), cv["transforms"]["source_px_to_sheet_mm"], cv["calibration"]["sheet_mm"])
        truth = reference_fields(refs, cv, np.array(metrics["registration"]["translation_mm"]))
        sheet = {"sheet": name, "matched": sum(r['match_status'] == 'matched' for r in rows),
                 "exact": sum(r["exact"] for r in rows), "fields": {}}
        for field in CELLS:
            s = field_score(fields[field].get("value"), truth.get(field))
            if s is not None:
                fields_total[field].append(s)
                sheet["fields"][field] = {"score": round(s, 3), "ours": fields[field].get("value"), "truth": truth.get(field)}
        layout = (raw.get("title_block") or {}).get("page-1", {}).get("fields", {})
        names = file_name_truth(name)
        for field, truth_value in names.items():
            new = contains_score(fields[field].get("value"), truth_value)
            old = contains_score(layout.get(field, {}).get("value"), truth_value)
            for label, value in (("cells", new), ("layout", old)):
                file_truth[(field, label)].append(value)
            sheet.setdefault("file_name", {})[field] = {"truth": truth_value, "cells": fields[field].get("value"), "layout": layout.get(field, {}).get("value"), "score": round(new, 3)}
        for field in CELLS:
            filled["cells", field] += bool(fields[field].get("value"))
            filled["layout", field] += bool(layout.get(field, {}).get("value"))
        per_sheet.append(sheet)
    if not per_sheet or not totals['reference']:
        ap.error('No matching OCR/CV cases found; check --raw, --cv-dirs, and --dxf-dir')
    n = totals["reference"]
    summary = {"sheets": len(per_sheet), "readings_matched": totals['matched'], "exact": totals["exact"],
               'reference_texts': n, 'predicted_texts': totals['predicted'],
               'missed_texts': totals['missed'], 'extra_texts': totals['extra'],
               'wrong_readings': totals['matched'] - totals['exact'],
               'failed_ocr_sheets': totals['ocr_failed_sheets'],
               'character_errors': totals['char_errors'], 'reference_characters': totals['reference_chars'],
               'cer': totals['char_errors'] / totals['reference_chars'] if totals['reference_chars'] else None,
               "exact_share": round(totals["exact"] / max(n, 1), 4), "mean_similarity": round(totals["sum_sim"] / max(n, 1), 4),
               "by_kind": {k: {"n": v["n"], "exact": round(v["exact"] / v["n"], 3)} for k, v in sorted(kinds.items())},
               "fields": {f: {"sheets_with_truth": len(v), "correct_0.9": sum(x >= 0.9 for x in v), "filled_0.5": sum(x >= 0.5 for x in v),
                              "mean": round(float(np.mean(v)), 3)} for f, v in fields_total.items()},
               "by_file_name": {f"{f}:{label}": {"sheets": len(v), "correct": sum(x >= 0.9 for x in v)} for (f, label), v in sorted(file_truth.items())},
               "filled": {f"{f}:{label}": c for (label, f), c in sorted(filled.items())}}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for (ours, ref), c in diffs.most_common(a.show):
        print(c, repr(ours), "<-", repr(ref))
    if a.json:
        # only the summary: field values of real drawings hold surnames and do not belong in the repository
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def _normalised(raw):
    out = []
    for source in raw["texts"]:
        item = copy.deepcopy(source)
        item["text"] = item.get("raw_text", item["text"])
        enrich(item)
        out.append(item)
    return out


if __name__ == "__main__":
    main()
