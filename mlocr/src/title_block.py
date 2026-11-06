"""Title-block fields from the fixed cells of GOST 2.104 form 1, using sheet millimetres from CV.

Layout-only guessing is unreliable, but the cells of the form are a standard: 185 x 55 mm in the lower right corner of
the frame. Here x runs from the left edge of the block and y from its bottom edge. Values stay candidates for review:
a cell gives the meaning of a text, it does not make the reading right."""
from __future__ import annotations

import re

import numpy as np

TITLE_W, TITLE_H = 185.0, 55.0
FRAME_RIGHT, FRAME_BOTTOM = 5.0, 5.0
SLACK = 1.0

# cell: (x0, y0, x1, y1) in mm inside the block
CELLS = {
    "document_number": (65.0, 40.0, 185.0, 55.0),
    "name": (65.0, 15.0, 135.0, 40.0),
    "material": (65.0, 0.0, 135.0, 15.0),
    "authors": (0.0, 0.0, 65.0, 30.0),
}
ROLES = (("Разраб.", 25.0, 30.0), ("Пров.", 20.0, 25.0), ("Т.контр.", 15.0, 20.0), ("Н.контр.", 5.0, 10.0), ("Утв.", 0.0, 5.0))
SURNAME_COLUMN = (17.0, 40.0)
LABEL = re.compile(r"(?:разраб|пров|провер|т\.?\s*контр|н\.?\s*контр|нормо?к?|утв|утверд|изм|лист|докум|подп|дата|"
                   r"лит|масса|масштаб|листов|формат|копировал|рук|гр|нач|отдела?|гип|зав)\.?", re.I)
LEADING_LABEL = re.compile(r"^(?:разраб|пров|провер|т\.?\s*контр|н\.?\s*контр|нормо?к?|утв|утверд|нач\.?\s*отдела?)\.?\s*", re.I)
STANDARD = re.compile(r"ГОСТ|ОСТ\s|ТУ\s", re.I)     # a material standard, not a designation of the document
DESIGNATION = re.compile(r"\d[-.]\d")      # numbers with separators: the designation of a tall text reaching into the name cell
PERSON = re.compile(r"[А-ЯЁ][а-яё]{2,}(?:\s*[А-ЯЁ]\.\s*[А-ЯЁ]?\.?)?")


def block_xy(point, sheet_w, sheet_h):
    """Sheet mm -> mm inside the title block."""
    return np.array([point[0] - (sheet_w - FRAME_RIGHT - TITLE_W), point[1] - FRAME_BOTTOM])


def in_cell(name, point, sheet_w, sheet_h):
    x, y = block_xy(point, sheet_w, sheet_h)
    x0, y0, x1, y1 = CELLS[name]
    return x0 - SLACK <= x <= x1 + SLACK and y0 - SLACK <= y <= y1 + SLACK


def _centres(items, matrix):
    m = np.asarray(matrix, float)
    out = []
    for item in items:
        poly = np.asarray(item["poly_source_px"], float)
        xy = (np.column_stack([poly, np.ones(len(poly))]) @ m.T)[:, :2]
        out.append((item, xy.mean(axis=0)))
    return out


def _lines(rows):
    """Texts of one cell in reading order: top to bottom, then left to right."""
    return sorted(rows, key=lambda r: (-round(r[1][1] / 2.5), r[1][0]))


def _candidate(rows):
    if not rows:
        return None
    rows = _lines(rows)
    return {"value": " ".join(r[0]["text"].strip() for r in rows), "status": "candidate",
            "source_text_ids": [r[0]["id"] for r in rows], "basis": "gost_2_104_form_1_cell",
            "needs_review": any(r[0].get("needs_review") for r in rows)}


def _unknown(reason):
    return {"value": None, "status": "unknown", "reason": reason, "source_text_ids": []}


def title_block_from_cells(texts, matrix, sheet_mm):
    """{field: candidate}. Cells are read only where the block fits the sheet; empty cells stay unknown."""
    W, H = sheet_mm
    centres = _centres(texts, matrix)
    cells = {name: [] for name in CELLS}
    for item, xy in centres:
        for name in CELLS:
            if in_cell(name, xy, W, H):
                cells[name].append((item, xy))
    fields = {}
    number = _lines([r for r in cells["document_number"] if re.search(r"\d", r[0]["text"]) and not STANDARD.search(r[0]["text"])])
    # The cell can also hold an organisation code next to the designation: the first reading is the designation.
    fields["document_number"] = (dict(_candidate(number[:1]), also_in_cell=[r[0]["text"].strip() for r in number[1:]]) if number else
                                 _unknown("The designation cell of the title block is empty or unreadable."))
    names = [r for r in cells["name"] if re.search(r"[A-Za-zА-Яа-яЁё]", r[0]["text"]) and not LABEL.fullmatch(r[0]["text"].strip())
             and not DESIGNATION.search(r[0]["text"]) and len(re.findall(r"[А-Яа-яЁё]", r[0]["text"])) >= 3]
    fields["name"] = _candidate(names) or _unknown("The name cell of the title block is empty or unreadable.")
    material = [r for r in cells["material"] if not LABEL.fullmatch(r[0]["text"].strip())]
    fields["material"] = _candidate(material) or _unknown("The material cell of the title block is empty or unreadable.")
    authors, ids = [], []
    for item, xy in cells["authors"]:
        x, y = block_xy(xy, W, H)
        # OCR often reads the label and the surname of a row as one box: "Разраб.Иванов"
        text = LEADING_LABEL.sub("", item["text"].strip())
        if not text or x > SURNAME_COLUMN[1] + SLACK or LABEL.fullmatch(text) or not PERSON.fullmatch(text):
            continue
        role = next((r for r, y0, y1 in ROLES if y0 - SLACK <= y <= y1 + SLACK), None)
        authors.append({"role": role, "name": text})
        ids.append(item["id"])
    fields["authors"] = ({"value": authors, "status": "candidate", "source_text_ids": ids, "basis": "gost_2_104_form_1_cell"}
                         if authors else _unknown("No surname was read in the surname column of the title block."))
    return fields


def merge_title_block(by_layout, observations, matrix, cv):
    """Cells of the standard form where they gave a value, the earlier layout rule elsewhere. Needs a calibrated sheet."""
    sheet = cv.get("calibration", {}).get("sheet_mm")
    if not sheet or cv.get("scale_status") != "known":
        return by_layout
    pages = {}
    for item in observations.get("texts", []):
        pages.setdefault(item.get("page", "page-1"), []).append(item)
    output = {key: value for key, value in (by_layout or {}).items()}
    for page, texts in pages.items():
        cells = title_block_from_cells(texts, matrix, sheet)
        known = (output.get(page) or {}).get("fields", {})
        fields = {}
        for name, value in cells.items():
            fields[name] = value if value.get("value") else known.get(name, value)
        output[page] = {"status": "needs_review", "fields": fields}
    return output
