"""Conservative dimension-to-CV-point proposals; ambiguity is explicit."""
from __future__ import annotations

import itertools
import math
from typing import Any
from .inputs import assert_same_page


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
    assert_same_page(ocr_page, cv)
    matrix = cv.get("transforms", {}).get("source_px_to_sheet_mm")
    if not matrix or cv.get("scale_status") != "known":
        return []
    points = [p for p in cv.get("points", []) if p.get('kind') != 'ocr_dimension_anchor']
    if len(points) > 400 or len(cv.get('primitives', [])) > 2000:
        raise ValueError('Dimension linking complexity limit exceeded (400 points / 2000 primitives)')
    scale = 1.0 if cv.get('coordinate_space') == 'part_mm' else cv.get('calibration', {}).get('view_scale')
    if scale is None or not math.isfinite(scale) or scale <= 0:
        return []
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
        if nominal is None or nominal <= 0 or center_px is None or text.get('needs_review') or notation.get('warning'):
            continue
        center_mm = _transform(matrix, center_px)
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
        thin_count = sum(g.get('kind') == 'line' and objects.get(g['id'], {}).get('role') in {'thin', 'dimension', 'extension'}
                         for g in cv.get('primitives', []))
        if len(candidates) * thin_count * thin_count > 2_000_000:
            raise ValueError('Dimension evidence search exceeds complexity limit')
        for candidate in candidates:
            candidate['evidence'] = _line_evidence(candidate, center_mm, cv)
        verified = [c for c in candidates if c['evidence']]
        output.append({
            "text_id": text.get("id"), "recognized_text": text.get("text"),
            "nominal_mm": nominal, "status": "resolved" if len(verified) == 1 else ("needs_review" if candidates else "unlinked"),
            "basis": ["axis_aligned_CV_point_pair", "length_matches_nominal_within_tolerance",
                      "text_proximity_used_for_ranking_only"],
            "decision_note": ("Unique dimension-line and extension-line ownership verified." if len(verified) == 1 else
                              "Proposal only: dimension-line and extension-line ownership are not yet verified."),
            "candidates": candidates[:max_candidates],
            "candidate_count": len(candidates),
            "confirmed": verified[0] if len(verified) == 1 else None,
        })
    return output


def _line_evidence(candidate, center, cv):
    """Minimum linear dimension: part anchors + dimension line + two extensions.

    No inference from proximity/nominal alone. Only straight axis-aligned MVP
    dimensions with a single unambiguous set of supporting lines are accepted.
    """
    objects = cv.get('objects', {})
    points = {p['id']: p for p in cv.get('points', [])}
    line_ids = {g['id'] for g in cv.get('primitives', []) if g['kind'] == 'line'}
    part_refs = []
    for key in ('p1_id', 'p2_id'):
        refs = [ref for ref in points[candidate[key]].get('refs', [])
                if ref['object_id'] in line_ids and objects.get(ref['object_id'], {}).get('role') == 'part'
                and ref.get('anchor') in {'start', 'end'}]
        if not refs:
            return None
        part_refs.append(refs)
    common = {r['object_id'] for r in part_refs[0]} & {r['object_id'] for r in part_refs[1]}
    if len(common) == 1:
        shared = next(iter(common))
        part_refs = [[r for r in refs if r['object_id'] == shared] for refs in part_refs]
    if any(len(refs) != 1 for refs in part_refs):
        return None
    anchors = [refs[0] for refs in part_refs]
    p, q = candidate['p1_mm'], candidate['p2_mm']
    axis = 0 if abs(p[1]-q[1]) < 0.35 else 1
    cross = 1-axis
    if abs(p[cross]-q[cross]) > 0.35:
        return None
    lines = [g for g in cv.get('primitives', []) if g['kind'] == 'line' and
             objects.get(g['id'], {}).get('role') in {'thin', 'dimension', 'extension'}]
    supported = []
    tolerance = 0.5
    for line in lines:
        a, b = line['start'], line['end']
        if abs(a[cross]-b[cross]) > tolerance:
            continue
        base = (a[cross]+b[cross])/2
        lo, hi = sorted((a[axis], b[axis]))
        ends = sorted((p[axis], q[axis]))
        if max(abs(lo-ends[0]), abs(hi-ends[1])) > 1 or not lo-1 <= center[axis] <= hi+1:
            continue
        if abs(center[cross]-base) > 6 or abs(base-p[cross]) < 1:
            continue
        extensions = []
        for endpoint in (p, q):
            matches = []
            for ext in lines:
                if ext['id'] == line['id']:
                    continue
                c, d = ext['start'], ext['end']
                if max(abs(c[axis]-endpoint[axis]), abs(d[axis]-endpoint[axis])) > tolerance:
                    continue
                low, high = sorted((c[cross], d[cross]))
                if low-1 <= endpoint[cross] <= high+1 and low-tolerance <= base <= high+tolerance:
                    matches.append(ext['id'])
            if len(matches) != 1:
                break
            extensions.extend(matches)
        if len(set(extensions)) == 2:
            base_point = list(center)
            base_point[cross] = base
            supported.append({'dimension_line': line['id'], 'extension_lines': extensions,
                              'base': base_point, 'angle': 90*axis, 'part_anchors': anchors})
    return supported[0] if len(supported) == 1 else None
