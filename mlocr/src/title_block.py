"""Title-block fields from the fixed cells of GOST 2.104 form 1, using sheet millimetres from CV.

Layout-only guessing is unreliable, but the cells of the form are a standard: 185 x 55 mm in the lower right corner of
the frame. Here x runs from the left edge of the block and y from its bottom edge. Values stay candidates for review:
a cell gives the meaning of a text, it does not make the reading right."""
from __future__ import annotations

import re
from difflib import get_close_matches

import numpy as np

from .normalization import _LOOK_ALIKE

# Latin letters the recogniser substitutes inside author labels (H.контр., Haч. omdena).
# Cyrillic д must stay д here: label words like "отдела" contain it, unlike vocabulary folding.
_LABEL_FOLD = {**_LOOK_ALIKE, 'д': 'д', 'd': 'д', 'D': 'Д', 'i': 'и', 'I': 'И', 'l': 'л', 'L': 'Л', 'q': 'д'}
_LABEL_TABLE = str.maketrans(_LABEL_FOLD)

# Canonical roles keyed by their folded label; a role comes from what is printed in the label
# column, not from a fixed row position — organisations use their own row sets (ГИП, Нач. отдела…).
ROLE_LABELS = {
    'разраб': 'Разраб.', 'пров': 'Пров.', 'провер': 'Пров.',
    'гип': 'ГИП', 'тконтр': 'Т.контр.', 'нконтр': 'Н.контр.', 'нормоконтр': 'Нормоконтр.',
    'начотдела': 'Нач. отдела', 'начбюро': 'Нач. бюро', 'рукгр': 'Рук. гр.', 'утв': 'Утв.',
}
# Whole-label misreadings observed on real sheets, mapped to the folded key of the intended label.
ROLE_ALIASES = {'гнп': 'гип'}


def fold_label(text: str) -> str:
    """Letters only, caseless, with recogniser lookalikes folded to Cyrillic."""
    folded = text.translate(_LABEL_TABLE).lower()
    return ''.join(ch for ch in folded if 'а' <= ch <= 'я' or ch == 'ё')


def match_role(text: str) -> str | None:
    """The author role printed in a label cell, or None when it cannot be identified reliably."""
    folded = ROLE_ALIASES.get(fold_label(text), None)
    if folded is None:
        folded = fold_label(text)
    if folded in ROLE_LABELS:
        return ROLE_LABELS[folded]
    close = get_close_matches(folded, ROLE_LABELS.keys(), n=1, cutoff=0.72)
    return ROLE_LABELS[close[0]] if close else None

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
# Row-position fallback only: the printed label decides the role when it was read.
ROLES = (("Разраб.", 25.0, 30.0), ("Пров.", 20.0, 25.0), ("Т.контр.", 15.0, 20.0), ("Н.контр.", 5.0, 10.0), ("Утв.", 0.0, 5.0))
SURNAME_COLUMN = (17.0, 40.0)
LABEL = re.compile(r"(?:разраб|пров|провер|т\.?\s*контр|н\.?\s*контр|нормо?к?|утв|утверд|изм|лист|докум|подп|дата|"
                   r"лит|масса|масштаб|листов|формат|копировал|рук|гр|нач|отдела?|гип|зав)\.?", re.I)
LEADING_LABEL = re.compile(r"^(?:разраб|пров|провер|гип|т\.?\s*контр|н\.?\s*контр|нормо?к?|утв|утверд|нач\.?\s*отдела?|"
                           r"нач\.?\s*бюро|рук\.?\s*гр\.?)\.?\s*", re.I)
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


def _row_groups(rows, band_mm=1.2):
    """(item, xy) pairs grouped into rows by vertical centre proximity."""
    groups = []
    for entry in sorted(rows, key=lambda r: r[1][1]):
        if groups and abs(entry[1][1] - np.mean([e[1][1] for e in groups[-1]])) <= band_mm:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


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
    grouped = sorted(_row_groups(cells["authors"]), key=lambda r: -float(np.mean([xy[1] for _, xy in r])))
    for raw_row in grouped:
        row = sorted(((item, block_xy(xy, W, H)) for item, xy in raw_row), key=lambda r: r[1][0])
        persons = [(item, xy) for item, xy in row
                   if SURNAME_COLUMN[0] - SLACK <= xy[0] <= SURNAME_COLUMN[1] + SLACK
                   and PERSON.fullmatch(item["text"].strip())]
        merged = []
        for item, xy in row:
            if xy[0] >= SURNAME_COLUMN[1] + SLACK:
                continue
            rest = LEADING_LABEL.sub('', item["text"].strip())
            if rest and rest != item["text"].strip() and PERSON.fullmatch(rest):
                merged.append((item, xy, rest))
        if len(persons) != 1 and len(merged) != 1:
            continue
        if len(persons) == 1:
            person_item, person_xy = persons[0]
            name = person_item["text"].strip()
        else:
            person_item, person_xy, name = merged[0]
        # The role is what is printed in the label column; a merged "Разраб.Иванов" box
        # contributes only its label prefix. Position is the last resort, not the source.
        label_ids = []
        label_parts = []
        for item, xy in row:
            if xy[0] >= SURNAME_COLUMN[0]:
                continue
            text = item["text"].strip()
            stripped = LEADING_LABEL.sub('', text)
            label_parts.append(text[:len(text) - len(stripped)] if stripped != text else text)
            label_ids.append(item["id"])
        role = match_role(' '.join(label_parts))
        if role is None:
            role = next((r for r, y0, y1 in ROLES if y0 - SLACK <= person_xy[1] <= y1 + SLACK), None)
        authors.append({"role": role, "name": name})
        ids.append(person_item["id"])
        ids.extend(label_ids)
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
