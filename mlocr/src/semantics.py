"""Conservative parsing and purpose hints for raw OCR observations."""
from __future__ import annotations

import re
from typing import Any


_NUMBER = r"[+-]?\d+(?:[.,]\d+)?"


def _number(value: str) -> float:
    return float(value.replace(",", "."))


def parse_engineering_notation(raw_text: str) -> dict[str, Any]:
    """Parse common dimension strings without rewriting the source reading."""
    text = raw_text.strip().replace(" ", "")
    result: dict[str, Any] = {"raw_text": raw_text, "kind": "unknown", "nominal": None,
                              "tolerance": None, "thread_pitch": None, "scale_denominator": None,
                              "warning": None}
    match = re.fullmatch(rf"({_NUMBER})\s*:\s*({_NUMBER})", text)
    if match:
        result.update(kind="scale", nominal=_number(match.group(1)),
                      scale_denominator=_number(match.group(2)))
        return result

    thread_class = r"(?:[A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451]+\d+|\d+[A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451]+)(?:[.,]\d+)?"
    match = re.fullmatch(rf"[\u041cMm\u043c]({_NUMBER})(?:[\u00d7x\u0445X\u0425]({_NUMBER}))?(?:[-\u2013]({thread_class}))?", text)
    if match:
        result.update(kind="thread", nominal=_number(match.group(1)),
                      thread_pitch=_number(match.group(2)) if match.group(2) else None,
                      thread_class=match.group(3))
        return result

    match = re.fullmatch(rf"({_NUMBER})[±]({_NUMBER})", text)
    if match:
        result.update(kind="linear_tolerance", nominal=_number(match.group(1)),
                      tolerance=_number(match.group(2)))
        return result

    match = re.fullmatch(rf"[\u00d8\u2300]({_NUMBER})", text)
    if match:
        result.update(kind="diameter", nominal=_number(match.group(1)))
        return result

    # Empty-set and phi-like glyphs are common font/OCR confusions in drawings.
    # Keep them as candidates; a geometry link or human review must confirm them.
    match = re.fullmatch(rf"[\u2205\u0424\u0444\u03c6\u03a6]({_NUMBER})", text)
    if match:
        result.update(kind="diameter_candidate", nominal=_number(match.group(1)),
                      warning="Diameter-like glyph is ambiguous; confirm from the source glyph and dimension geometry.")
        return result

    match = re.fullmatch(rf"[Rr]({_NUMBER})", text)
    if match:
        result.update(kind="radius", nominal=_number(match.group(1)))
        return result

    match = re.fullmatch(_NUMBER, text)
    if match:
        result.update(kind="linear", nominal=_number(text))
        return result

    result["warning"] = "Unrecognized notation; original OCR text is retained unchanged."
    return result


def classify_purpose(raw_text: str, polygon: list[list[float]] | None,
                     image_width: int, image_height: int) -> str:
    """Return a weak UI hint; callers must not treat this as verified semantics."""
    if polygon and image_width > 0 and image_height > 0:
        center_x = sum(point[0] for point in polygon) / len(polygon)
        center_y = sum(point[1] for point in polygon) / len(polygon)
        if center_x >= image_width * 0.55 and center_y >= image_height * 0.72:
            return "title_block_candidate"
    parsed = parse_engineering_notation(raw_text)
    if parsed["kind"] in {"linear", "linear_tolerance", "diameter", "diameter_candidate", "radius", "thread", "scale"}:
        return "dimension_or_scale_candidate"
    return "ordinary_annotation_candidate"
