"""Shared one-to-one evaluation, independent of recognition or reference text values."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from mlocr.src.inputs import assert_same_page
from mlocr.src.normalization import enrich


def assign_pairs(left_count, right_count, candidates):
    """Maximise the number of geometric matches, then minimise their cost.

    candidates contains (left_index, right_index, non-negative_cost). Unmatched
    entries are allowed. A prediction/reference can participate at most once.
    """
    candidates = list(candidates)
    if not left_count or not right_count or not candidates:
        return []
    if any(not math.isfinite(cost) or cost < 0 for _, _, cost in candidates):
        raise ValueError('Matching costs must be finite and non-negative')
    penalty = (max(cost for _, _, cost in candidates) + 1) * (min(left_count, right_count) + 1)
    costs = np.full((left_count, right_count + left_count), penalty * 2)
    costs[:, right_count:] = penalty
    valid = set()
    for left, right, cost in candidates:
        costs[left, right] = min(costs[left, right], cost)
        valid.add((left, right))
    rows, cols = linear_sum_assignment(costs)
    return [(int(left), int(right)) for left, right in zip(rows, cols) if (left, right) in valid]


def match_dimensions(dimensions, points, references, translation, tolerance_mm=.5):
    """Match both measured feet in 2D; nominal-reading accuracy is scored separately."""
    candidates = []
    for index, dim in enumerate(dimensions):
        a = np.asarray(points[dim['p1_ref']]['xy']) + translation
        b = np.asarray(points[dim['p2_ref']]['xy']) + translation
        direction = np.array([math.cos(math.radians(dim['angle'])), math.sin(math.radians(dim['angle']))])
        for ref_index, ref in enumerate(references):
            if abs(float(np.asarray(ref['u']) @ direction)) <= .99:
                continue
            direct = max(np.linalg.norm(a - ref['p1']), np.linalg.norm(b - ref['p2']))
            reverse = max(np.linalg.norm(a - ref['p2']), np.linalg.norm(b - ref['p1']))
            error = float(min(direct, reverse))
            if error <= tolerance_mm:
                candidates.append((index, ref_index, error))
    return assign_pairs(len(dimensions), len(references), candidates)


def real_observations(payload, cv):
    """Reapply text rules without rewriting raster identity, status, or failures."""
    assert_same_page(payload, cv)
    result = deepcopy(payload)
    for item in result['texts']:
        original = deepcopy(item)
        item['text'] = item.get('raw_text', item['text'])
        enrich(item)
        visual = [c for c in original.get('corrections', []) if c.get('rule') == 'diameter_visible_in_raster']
        if visual:
            item.update(text=original['text'], notation=original['notation'])
            item['corrections'].extend(visual)
    return result


def edit_distance(actual, expected):
    row = list(range(len(expected) + 1))
    for index, char in enumerate(actual, 1):
        next_row = [index]
        for other_index, other in enumerate(expected, 1):
            next_row.append(min(row[other_index] + 1, next_row[-1] + 1,
                                row[other_index - 1] + (char != other)))
        row = next_row
    return row[-1]
