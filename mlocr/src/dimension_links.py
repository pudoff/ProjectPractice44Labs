"""Conservative dimension-to-CV-point proposals; ambiguity is explicit."""
from __future__ import annotations

import itertools
import math
from typing import Any


def _center_px(item: dict) -> tuple[float, float] | None:
    polygon = item.get("poly_source_px") or item.get("polygon_px")
    if not polygon:
        return None
    return (sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon))


def _transform(matrix: list[list[float]], point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2])


def propose_dimension_links(ocr_page: dict[str, Any], cv: dict[str, Any],
                           length_tolerance_mm: float = 0.8,
                           max_candidates: int = 5) -> list[dict[str, Any]]:
    """Find point-pair hypotheses matching a linear nominal and orientation.

    These are diagnostic proposals only. A pair is never accepted from text
    proximity and length agreement alone; the caller must verify dimension-line
    and extension-line evidence before promoting it to an automatic link.
    """
    matrix = cv.get("transforms", {}).get("source_px_to_sheet_mm")
    if not matrix or cv.get("scale_status") != "known":
        return []
    points = cv.get("points", [])
    objects = cv.get("objects", {})
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
    for text in ocr_page.get("texts", []):
        if text.get("purpose") == "title_block_candidate":
            continue
        notation = text.get("notation") or {}
        if notation.get("kind") not in {"linear", "linear_tolerance"}:
            continue
        nominal = notation.get("nominal")
        center_px = _center_px(text)
        if nominal is None or center_px is None:
            continue
        center_mm = _transform(matrix, center_px)
        tolerance = max(length_tolerance_mm, abs(float(nominal)) * 0.015)
        candidates = []
        for first, second, length, ref_ids, roles in axis_pairs:
            p1, p2 = first.get("xy"), second.get("xy")
            error = abs(length - float(nominal))
            if error > tolerance:
                continue
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            text_distance = math.hypot(midpoint[0] - center_mm[0], midpoint[1] - center_mm[1])
            if text_distance > max(35.0, abs(float(nominal)) * 0.65):
                continue
            candidates.append({
                "p1_id": first["id"], "p2_id": second["id"],
                "p1_mm": [round(float(p1[0]), 4), round(float(p1[1]), 4)],
                "p2_mm": [round(float(p2[0]), 4), round(float(p2[1]), 4)],
                "measured_mm": round(length, 4),
                "nominal_error_mm": round(error, 4),
                "text_to_midpoint_mm": round(text_distance, 3),
                "geometry_refs": ref_ids, "geometry_roles": roles,
            })
        candidates.sort(key=lambda item: (item["nominal_error_mm"], item["text_to_midpoint_mm"]))
        output.append({
            "text_id": text.get("id"), "recognized_text": text.get("text"),
            "nominal_mm": nominal, "status": "needs_review" if candidates else "unlinked",
            "basis": ["axis_aligned_CV_point_pair", "length_matches_nominal_within_tolerance",
                      "text_proximity_used_for_ranking_only"],
            "decision_note": "Proposal only: dimension-line and extension-line ownership are not yet verified.",
            "candidates": candidates[:max_candidates],
            "candidate_count": len(candidates),
        })
    return output
