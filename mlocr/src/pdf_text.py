"""PDF text-layer observations and conservative title-block candidates."""
from __future__ import annotations

import itertools
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pymupdf

from .semantics import classify_purpose, parse_engineering_notation
from .normalization import enrich, normalize_text
from .title_block import LABEL, LEADING_LABEL, PERSON, match_role


def _transform_bbox(page, bbox, dpi: int, image_width: int, image_height: int):
    x0, y0, x1, y1 = map(float, bbox)
    corners = [
        pymupdf.Point(x0, y0), pymupdf.Point(x1, y0),
        pymupdf.Point(x1, y1), pymupdf.Point(x0, y1),
    ]
    rotated = [point * page.rotation_matrix for point in corners]
    rect = page.rect
    sx, sy = image_width / rect.width, image_height / rect.height
    return [[(p.x - rect.x0) * sx, (p.y - rect.y0) * sy] for p in rotated]


def extract_pdf_text_page(path: Path, page_index: int, dpi: int,
                          image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Extract selectable PDF words grouped by text line and map their boxes to rendered pixels."""
    doc = pymupdf.open(path)
    page = doc[page_index]
    hidden = [span['bbox'] for span in page.get_texttrace() if span.get('type') == 3 or span.get('opacity', 1) == 0]
    line_directions = {}
    blocks = page.get_text("dict", sort=False).get("blocks", [])
    for block_index, block in enumerate(blocks):
        for line_index, line in enumerate(block.get("lines", [])):
            line_directions[(block_index, line_index)] = line.get("dir", (1.0, 0.0))

    grouped = {}
    for word in page.get_text("words", sort=False):
        if len(word) < 8:
            continue
        x0, y0, x1, y1, value, block_index, line_index, word_index = word[:8]
        value = str(value).strip()
        if not value:
            continue
        grouped.setdefault((block_index, line_index), []).append(
            (word_index, value, (x0, y0, x1, y1))
        )

    rows = []
    for (block_index, line_index), words in grouped.items():
        words.sort(key=lambda item: item[0])
        text = " ".join(item[1] for item in words).strip()
        bbox = (
            min(item[2][0] for item in words), min(item[2][1] for item in words),
            max(item[2][2] for item in words), max(item[2][3] for item in words),
        )
        if any(_overlap_score(bbox, hidden_box) > 0.35 for hidden_box in hidden):
            continue  # Invisible/overprinted OCR layers must not override the raster.
        polygon = _transform_bbox(page, bbox, dpi, image_width, image_height)
        dx, dy = line_directions.get((block_index, line_index), (1.0, 0.0))
        origin = pymupdf.Point(0, 0) * page.rotation_matrix
        endpoint = pymupdf.Point(dx, dy) * page.rotation_matrix
        angle = (math.degrees(math.atan2(endpoint.y - origin.y, endpoint.x - origin.x)) + 90) % 180 - 90
        page_id = f"page-{page_index + 1}"
        rows.append(enrich({
            "id": f"{page_id}-pdftext-{len(rows) + 1:04d}",
            "page": page_id,
            "text": text,
            "score": None,
            "alternatives": [],
            "selection_basis": "pdf_text_layer",
            "source": "pdf_text",
            "angle_deg": angle,
            "purpose": None,
            "notation": parse_engineering_notation(text),
            "polygon_px": polygon,
            "poly_source_px": polygon,
            "bbox_pdf_pt": [float(v) for v in bbox],
        }))

    for row in rows:
        row["purpose"] = classify_purpose(
            row["text"], row["poly_source_px"], image_width, image_height
        )
    doc.close()
    return rows


def _bounds(item):
    poly = item.get("poly_source_px") or item.get("polygon_px")
    if not poly:
        return None
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_score(a, b):
    if not a or not b:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def merge_pdf_and_ocr(pdf_items: list[dict], ocr_items: list[dict]) -> list[dict]:
    """Prefer embedded PDF text for the same visible region; preserve OCR readings as evidence."""
    result = [dict(item) for item in pdf_items]
    unmatched = []
    for ocr in ocr_items:
        obox = _bounds(ocr)
        matches = [(_overlap_score(obox, _bounds(pdf)), index) for index, pdf in enumerate(result)]
        score, index = max(matches, default=(0.0, -1))
        if index >= 0 and score >= 0.35:
            target = result[index]
            target["source"] = "pdf_text+ocr"
            target.setdefault("ocr_readings", []).append({
                "text": ocr["text"], "score": ocr.get("score"),
                "alternatives": ocr.get("alternatives", []),
            })
            if normalize_text(ocr["text"])[0] != normalize_text(target["text"])[0]:
                target['needs_review'] = True
                target.setdefault("issues", []).append({
                    "code": "pdf_ocr_disagreement",
                    "message": "Selectable PDF text and raster OCR differ; both readings are retained.",
                })
            if ocr.get('needs_review'):
                target['needs_review'] = True
                target.setdefault('issues', []).extend(ocr.get('issues', []))
        else:
            item = dict(ocr)
            item["source"] = "paddleocr"
            unmatched.append(item)
    combined = result + unmatched
    combined.sort(key=lambda item: (
        min((point[1] for point in item.get("poly_source_px", [])), default=0),
        min((point[0] for point in item.get("poly_source_px", [])), default=0),
        item["id"],
    ))
    return combined


def _center(item):
    box = _bounds(item)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else (0.0, 0.0)


def _unknown(reason: str, source_ids: list[str] | None = None):
    return {"value": None, "status": "unknown", "reason": reason,
            "source_text_ids": source_ids or []}


def _authors(items):
    """Author rows found by their printed role label, wherever the stamp sits on the sheet.

    A label anchors its row: the surname must be the only person-like reading in the next cell
    to the right of that label. The role comes from the printed text, never from position and
    never from a placeholder."""
    values, ids = [], []
    anchors = []
    for item in items:
        role = match_role(item['text'].strip())
        if role:
            anchors.append(([item], _bounds(item), role))
    for first, second in itertools.combinations(items, 2):
        if abs(_center(first)[1] - _center(second)[1]) > 30:
            continue
        a, b = _bounds(first), _bounds(second)
        if min(a[2], b[2]) - max(a[0], b[0]) < -2 or len(first['text']) + len(second['text']) > 24:
            continue
        role = match_role(first['text'] + ' ' + second['text'])
        if role:
            anchors.append(([first, second], (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])), role))
    used = set()
    for label_items, box, role in anchors:
        height = max(box[3] - box[1], 1.0)
        # The surname cell is the next cell of the row (~5-20 mm from the label); a text many
        # box heights away sits in another cell of the stamp (e.g. the part name on wide forms).
        persons = [item for item in items
                   if item not in label_items and item['id'] not in used
                   and PERSON.fullmatch(item['text'].strip()) and not LABEL.fullmatch(item['text'].strip())
                   and _bounds(item)[0] >= box[0] - 2 and _bounds(item)[0] - box[2] <= 5 * height
                   and abs(_center(item)[1] - (box[1] + box[3]) / 2) <= 0.75 * max(height, _bounds(item)[3] - _bounds(item)[1])]
        if len(persons) != 1:
            continue
        person = persons[0]
        used.add(person['id'])
        values.append({'role': role, 'name': person['text'].strip()})
        ids.extend([item['id'] for item in label_items] + [person['id']])
    values.sort(key=lambda entry: next(_center(item)[1] for item in items if item['text'].strip() == entry['name']))
    return ({'value': values, 'status': 'candidate', 'source_text_ids': sorted(set(ids))} if values else
            _unknown('No unambiguous person in a labelled author row.'))


def build_title_block(pages: list[dict]) -> dict:
    """Return reviewable field candidates; never fill a value from layout plausibility alone."""
    output = {}
    for page in pages:
        width, height = page["width_px"], page["height_px"]
        items = page["texts"]
        candidates = []
        for item in items:
            x, y = _center(item)
            if x >= width * 0.52 and y >= height * 0.70:
                candidates.append(item)

        doc_re = re.compile(r"\d[\dA-Za-zXx\.\-_/]{4,}")
        doc_candidates = [
            item for item in candidates
            if doc_re.fullmatch(item["text"].replace(" ", ""))
            and not re.fullmatch(r"0+", item["text"].replace(" ", ""))
        ]
        doc = max(doc_candidates, key=lambda item: _center(item)[1], default=None)
        # Author labels live in the leftmost stamp column, far from the right edge on wide
        # sheets; search by the bottom band of the page, not by a fixed horizontal fraction.
        authors = _authors([item for item in items if _center(item)[1] >= height * 0.60])

        excluded = re.compile(
            r"(?:\u0437\u0430\u0434\u0430\u043d\u0438\u0435|\u043a\u0430\u0444\u0435\u0434\u0440\u0430|"
            r"\u0444\u043e\u0440\u043c\u0430\u0442|\u043b\u0438\u0441\u0442|\u043b\u0438\u0442\.|"
            r"\u043c\u0430\u0441\u0441\u0430|\u043c\u0430\u0441\u0448\u0442\u0430\u0431|"
            r"\u0433\u043e\u0441\u0442|\u0441\u0442\u0430\u043b\u044c|\u0440\u0430\u0437\u0440\u0430\u0431\.|"
            r"\u043f\u0440\u043e\u0432\.)", re.IGNORECASE
        )
        name_candidates = []
        for item in candidates:
            text = item["text"].strip()
            x, y = _center(item)
            if item is doc or item['id'] in authors['source_text_ids'] or item.get("notation", {}).get("kind") != "unknown":
                continue
            if not re.search(r"[A-Za-z\u0400-\u04FF]", text) or excluded.search(text):
                continue
            if x < width * 0.65 or y < height * 0.80:
                continue
            name_candidates.append(item)
        name = max(name_candidates, key=lambda item: (
            (_bounds(item)[2] - _bounds(item)[0]) * (_bounds(item)[3] - _bounds(item)[1])
        ), default=None)

        material_re = re.compile(
            r"(?:\u0441\u0442\u0430\u043b\u044c|\u0441\u0442\.?\s*\d|"
            r"\u0441\u0447\s*\d|\u0430\u043b\s*\d|\u0432\u0447\s*\d|\u0433\u043e\u0441\u0442)",
            re.IGNORECASE
        )
        material = next((item for item in candidates if material_re.search(item["text"])), None)

        fields = {
            "name": ({"value": name["text"], "status": "candidate",
                      "source_text_ids": [name["id"]]} if name else
                     _unknown("No clear title candidate was found in the title-block area.")),
            "document_number": ({"value": doc["text"], "status": "candidate",
                                 "source_text_ids": [doc["id"]]} if doc else
                                _unknown("No readable document number was found in the title-block area.")),
            "material": ({"value": material["text"], "status": "candidate",
                          "source_text_ids": [material["id"]]} if material else
                         _unknown("No readable material field was found in the title-block area.")),
            "authors": authors,
        }
        output[page["page"]] = {"status": "needs_review", "fields": fields}
    return output
