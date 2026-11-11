"""Conservative dimension-to-CV-point proposals; ambiguity is explicit."""
from __future__ import annotations

import itertools
import math
from typing import Any
from .dimension_geometry import arrowheads, excluded_dimension_pieces, find_dimension, part_objects, thin_lines
from .inputs import assert_same_page


def _center_px(item: dict) -> tuple[float, float] | None:
    polygon = item.get("poly_source_px") or item.get("polygon_px")
    if not polygon:
        return None
    return (sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon))


def _text_box(text: dict, matrix) -> tuple[tuple[float, float], float]:
    """Size (long, short side, mm) and direction (degrees) of a text box on the sheet."""
    points = [_transform(matrix, tuple(p)) for p in (text.get('poly_source_px') or text.get('polygon_px'))]
    edges = [(points[(k+1) % len(points)][0]-points[k][0], points[(k+1) % len(points)][1]-points[k][1]) for k in range(len(points))]
    lengths = [math.hypot(*e) for e in edges]
    k = lengths.index(max(lengths))
    return (lengths[k], min(lengths)), math.degrees(math.atan2(edges[k][1], edges[k][0]))


def _transform(matrix: list[list[float]], point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2])


def _can_anchor(point, objects):
    """A linear dimension is confirmed only on the start or end of a part object (see _line_evidence). Middles,
    quadrants, centres and thin lines of a crowded sheet must not use up the complexity limit."""
    for ref in point.get("refs", []):
        if objects.get(ref.get("object_id"), {}).get("role", "part") == "part" and ref.get("anchor", "start") in {"start", "end"}:
            return True
    return False


STANDARD_SCALES = (1.0, 0.5, 0.4, 0.25, 0.2, 0.1, 0.05, 2.0, 2.5, 4.0, 5.0, 10.0)      # GOST 2.302, as sheet mm per part mm


def validate_link_budget(cv):
    if len(cv.get('primitives', [])) > 8000:
        raise ValueError('Dimension linking complexity limit exceeded (8000 primitives)')


def printed_view_scales(observations):
    return {item['notation']['nominal'] / item['notation']['scale_denominator']
            for item in observations.get('texts', [])
            if item.get('notation', {}).get('kind') == 'scale'
            and not item.get('needs_review') and not item['notation'].get('warning')}


def vote_view_scale(ocr_page: dict[str, Any], cv: dict[str, Any]) -> float | None:
    """The scale at which most printed sizes find their dimension lines, when OCR did not give one scale. Decided only by a
    clear majority: several sizes, twice as many as any other scale. Only dimension lines with an arrow vote: the cells of
    a table look just like a number between two extension lines at any scale."""
    validate_link_budget(cv)
    if len(printed_view_scales(ocr_page)) > 1:
        return None
    matrix = cv.get("transforms", {}).get("source_px_to_sheet_mm")
    if not matrix:
        return None
    lines, part, residue, arrows = thin_lines(cv), part_objects(cv), excluded_dimension_pieces(cv), arrowheads(cv)
    votes = {}
    for text in ocr_page.get("texts", []):
        notation = text.get("notation") or {}
        center = _center_px(text)
        if (notation.get("kind") not in {"linear", "linear_tolerance", "diameter"} or not notation.get("nominal") or center is None
                or text.get("purpose") == "title_block_candidate" or text.get("needs_review") or notation.get("warning")):
            continue
        size, direction = _text_box(text, matrix)
        for scale in STANDARD_SCALES:
            if find_dimension(_transform(matrix, center), size, direction, notation["nominal"], scale, lines, part, residue, arrows, True)[0]:
                votes[scale] = votes.get(scale, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < 3 or (len(ranked) > 1 and ranked[0][1] < 2 * ranked[1][1]):
        return None
    return ranked[0][0]


def propose_dimension_links(ocr_page: dict[str, Any], cv: dict[str, Any],
                           length_tolerance_mm: float = 0.8,
                           max_candidates: int = 5) -> list[dict[str, Any]]:
    """Link each printed size to its geometry (dimension_geometry.find_dimension).

    A point pair of the right length near the number is kept only as a proposal for review: it is never accepted from
    text proximity and length agreement alone.
    """
    assert_same_page(ocr_page, cv)
    validate_link_budget(cv)
    if len(printed_view_scales(ocr_page)) > 1:
        return []
    matrix = cv.get("transforms", {}).get("source_px_to_sheet_mm")
    if not matrix or cv.get("scale_status") != "known":
        return []
    objects = cv.get("objects", {})
    points = [p for p in cv.get("points", []) if p.get('kind') != 'ocr_dimension_anchor' and _can_anchor(p, objects)]
    if len(points) > 400:       # the point-pair proposals below are quadratic; the dimension-line search does not use them
        points = []
    scale = 1.0 if cv.get('coordinate_space') == 'part_mm' else cv.get('calibration', {}).get('view_scale')
    if scale is None or not math.isfinite(scale) or scale <= 0:
        return []
    # Compute axis-aligned pairs once, rather than rescanning O(n²) points for
    # every OCR label on a page.
    axis_pairs = []
    for first, second in itertools.combinations(points, 2):
        p1, p2 = first.get("xy"), second.get("xy")
        if not p1 or not p2:
            continue
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if min(abs(dx), abs(dy)) > max(0.35, length * 0.01):
            continue
        refs = [*first.get("refs", []), *second.get("refs", [])]
        ref_ids = sorted({ref.get("object_id") for ref in refs if ref.get("object_id")})
        roles = sorted({objects.get(ref_id, {}).get("role", "unknown") for ref_id in ref_ids})
        axis_pairs.append((first, second, float(length), ref_ids, roles))
    output = []
    all_thin, part, residue, arrows = thin_lines(cv), part_objects(cv), excluded_dimension_pieces(cv), arrowheads(cv)
    for text in ocr_page.get("texts", []):
        if text.get("purpose") == "title_block_candidate":
            continue
        notation = text.get("notation") or {}
        if notation.get("kind") not in {"linear", "linear_tolerance", "diameter"}:
            continue
        nominal = notation.get("nominal")
        center_px = _center_px(text)
        if nominal is None or nominal <= 0 or center_px is None or text.get('needs_review') or notation.get('warning'):
            continue
        center_mm = _transform(matrix, center_px)
        size, direction = _text_box(text, matrix)
        evidence, reason = find_dimension(center_mm, size, direction, nominal, scale, all_thin, part, residue, arrows)
        if evidence:
            output.append({"text_id": text.get("id"), "recognized_text": text.get("text"), "nominal_mm": nominal,
                           "notation_kind": notation.get("kind"), "suffix": notation.get("suffix"),
                           "hole_count": notation.get("hole_count"),
                           "status": "resolved", "basis": ["dimension_line_near_number", "extension_lines_at_nominal_distance", "feet_on_part"],
                           "decision_note": "Dimension line and two extension lines at the printed distance found.",
                           "candidates": [], "candidate_count": 1,
                           "confirmed": {"view_scale": scale, "evidence": evidence}})
            continue
        if notation.get("kind") == "diameter":          # the point-pair rules below cover only plain linear sizes
            output.append({"text_id": text.get("id"), "recognized_text": text.get("text"), "nominal_mm": nominal, "status": "unlinked",
                           "basis": [], "decision_note": reason, "candidates": [], "candidate_count": 0, "confirmed": None})
            continue
        tolerance = max(length_tolerance_mm, abs(float(nominal)) * 0.015)
        candidates = []
        for first, second, length, ref_ids, roles in axis_pairs:
            p1, p2 = first.get("xy"), second.get("xy")
            error = abs(length / scale - float(nominal))
            if error > tolerance:
                continue
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            text_distance = math.hypot(midpoint[0] - center_mm[0], midpoint[1] - center_mm[1])
            if text_distance > max(35.0, abs(float(nominal)) * scale * 0.65):
                continue
            candidates.append({
                "p1_id": first["id"], "p2_id": second["id"],
                "p1_mm": [round(float(p1[0]), 4), round(float(p1[1]), 4)],
                "p2_mm": [round(float(p2[0]), 4), round(float(p2[1]), 4)],
                "measured_mm": round(length, 4),
                "part_measured_mm": round(length / scale, 4),
                "view_scale": scale,
                "nominal_error_mm": round(error, 4),
                "text_to_midpoint_mm": round(text_distance, 3),
                "geometry_refs": ref_ids, "geometry_roles": roles,
            })
            if len(candidates) > 250:
                raise ValueError('Dimension linking has too many geometric candidates')
        candidates.sort(key=lambda item: (item["nominal_error_mm"], item["text_to_midpoint_mm"]))
        output.append({
            "dimension_line_search": reason,
            "text_id": text.get("id"), "recognized_text": text.get("text"),
            "nominal_mm": nominal, "status": "needs_review" if candidates else "unlinked",
            "basis": ["axis_aligned_CV_point_pair", "length_matches_nominal_within_tolerance",
                      "text_proximity_used_for_ranking_only"],
            "decision_note": "Proposal only: no dimension line with matching extension lines was found for this number.",
            "candidates": candidates[:max_candidates],
            "candidate_count": len(candidates),
            "confirmed": None,
        })
    return output

