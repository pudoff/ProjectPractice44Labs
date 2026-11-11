from __future__ import annotations

import math

import numpy as np

ROW_OFFSET_MM = 0.3
PARALLEL_DEG = 1.0
PERPENDICULAR_DEG = 3.0
CROSS_REACH_MM = 1.5
TEXT_DIST_MM = 9.0
TEXT_SHORT_ASPECT = 1.6
TEXT_ANGLE_DEG = 12.0
FOOT_REACH_MM = 3.0
CORNER_SNAP_MM = 0.3
SAME_POSITION_MM = 0.3
LENGTH_TOL_MIN_MM = 0.3
LENGTH_TOL_SHARE = 0.003
LENGTH_TOL_SHORT = 0.25
THICK_PENALTY_MM = 2.0
ARROW_REACH_MM = 6.0


def _unit(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return d / max(float(np.linalg.norm(d)), 1e-12)


def closed_contour_objects(cv):
    return {oid for c in cv.get('contours', []) if c.get('closed') for oid in c.get('object_ids', [])}


def thin_lines(cv):
    objects = cv.get('objects', {})
    out = []
    for p in cv.get('primitives', []):
        role = objects.get(p['id'], {}).get('role')
        if p['kind'] != 'line' or math.dist(p['start'], p['end']) <= 1e-6:
            continue
        if role in {'thin', 'dimension', 'extension', 'axis'}:
            out.append(p)
        elif role == 'hidden':
            out.append({**p, 'hidden': True})
        elif role == 'part':
            out.append({**p, 'thick': True})
    return out


def arrowheads(cv):
    out = []
    for o in cv.get('objects', {}).values():
        if o.get('role') == 'arrowhead' and o.get('kind') == 'line':
            g = o['geometry_mm']
            out.append((np.asarray(g['start'], float) + np.asarray(g['end'], float)) / 2)
    return np.array(out) if out else np.zeros((0, 2))


def excluded_dimension_pieces(cv):
    out = []
    for oid, o in cv.get('objects', {}).items():
        if o.get('kind') == 'line' and o.get('status') == 'excluded' and o.get('role') in {'text_residue', 'duplicate'}:
            g = o['geometry_mm']
            if math.dist(g['start'], g['end']) > 1e-6:
                out.append({'id': oid, 'kind': 'line', 'start': g['start'], 'end': g['end'], 'excluded': True})
    return out


def part_objects(cv):
    objects = cv.get('objects', {})
    return [p for p in cv.get('primitives', []) if objects.get(p['id'], {}).get('role') == 'part']


def _rows(lines, centre):
    rows = []
    used = set()
    for line in lines:
        if line['id'] in used or line.get('role_axis'):
            continue
        u = _unit(line['start'], line['end'])
        if u[0] < 0 or (abs(u[0]) < 1e-9 and u[1] < 0):
            u = -u
        n = np.array([-u[1], u[0]])
        offset = float(np.asarray(line['start'], float) @ n)
        if abs(float(centre @ n) - offset) > TEXT_DIST_MM:
            continue
        pieces = []
        for other in lines:
            v = _unit(other['start'], other['end'])
            if abs(u[0] * v[1] - u[1] * v[0]) > math.sin(math.radians(PARALLEL_DEG)):
                continue
            if max(abs(float(np.asarray(other['start'], float) @ n) - offset),
                   abs(float(np.asarray(other['end'], float) @ n) - offset)) <= ROW_OFFSET_MM:
                pieces.append(other)
        used.update(p['id'] for p in pieces)
        rows.append((u, n, offset, pieces))
    return rows


def _carrier_candidates(lines, centre, reach, existing=()):
    """Lines near the text that can carry a dimension, with their own carrier offset.

    The text of a large dimension is often placed beside the line, not on it; the search then
    needs the carrier at the line itself. This additional search uses thin visible lines;
    thick lines can still participate in the nearby rows returned by _rows()."""
    found = []
    for line in lines:
        if line.get('excluded') or line.get('thick') or line.get('hidden'):
            continue
        a, b = np.asarray(line['start'], float), np.asarray(line['end'], float)
        if min(np.linalg.norm(a - centre), np.linalg.norm(b - centre)) > reach:
            continue
        v = _unit(a, b)
        u = v
        if u[0] < 0 or (abs(u[0]) < 1e-9 and u[1] < 0):
            u = -u
        n = np.array([-u[1], u[0]])
        offset = float(np.asarray(line['start'], float) @ n)
        if any(abs(u[0] * ou[1] - u[1] * ou[0]) < math.sin(math.radians(PARALLEL_DEG)) and
               abs(offset - oo) <= ROW_OFFSET_MM for ou, _, oo, _ in found):
            continue
        if any(abs(u[0] * eu[1] - u[1] * eu[0]) < math.sin(math.radians(PARALLEL_DEG)) and
               abs(offset - eo) <= ROW_OFFSET_MM for eu, eo in existing):
            continue
        found.append((u, n, offset, line))
    return found


def _crossings(lines, u, n, offset, row_ids, reach=CROSS_REACH_MM):
    found = []
    for line in lines:
        if line['id'] in row_ids:
            continue
        v = _unit(line['start'], line['end'])
        if abs(float(u @ v)) > math.sin(math.radians(PERPENDICULAR_DEG)):
            continue
        a, b = float(np.asarray(line['start'], float) @ n) - offset, float(np.asarray(line['end'], float) @ n) - offset
        low, high = min(a, b), max(a, b)
        if low > reach or high < -reach:
            continue
        s = float((np.asarray(line['start'], float) @ u + np.asarray(line['end'], float) @ u) / 2)
        found.append((s, line))
    found.sort(key=lambda t: (t[0], t[1]['id']))
    merged = []
    for s, line in found:
        if merged and s - merged[-1][0] <= SAME_POSITION_MM:
            if merged[-1][1].get('thick') and not line.get('thick'):     # a thin line at the same place is the extension line
                merged[-1][1] = line
        else:
            merged.append([s, line])
    return [(m[0], m[1]) for m in merged]


def _ray_hits(origin, direction, prim):
    out = []
    if prim['kind'] == 'line':
        a, b = np.asarray(prim['start'], float), np.asarray(prim['end'], float)
        d = b - a
        den = direction[0] * d[1] - direction[1] * d[0]
        if abs(den) < 1e-9:
            return out
        w = a - origin
        r = (w[0] * d[1] - w[1] * d[0]) / den
        t = (w[0] * direction[1] - w[1] * direction[0]) / den
        if r >= 0 and -1e-6 <= t <= 1 + 1e-6:
            out.append(float(r))
        return out
    c = np.asarray(prim['center'], float)
    f = origin - c
    qb = 2 * float(f @ direction)
    qc = float(f @ f) - prim['radius'] ** 2
    disc = qb * qb - 4 * qc
    if disc < 0:
        return out
    for r in ((-qb - math.sqrt(disc)) / 2, (-qb + math.sqrt(disc)) / 2):
        if r < 0:
            continue
        if prim['kind'] == 'arc':
            p = origin + direction * r - c
            angle = math.degrees(math.atan2(p[1], p[0])) % 360
            start = prim['start_angle'] % 360
            if (angle - start) % 360 > ((prim['end_angle'] - start) % 360 or 360):
                continue
        out.append(float(r))
    return out


def foot_of(line, s, u, n, offset, part):
    base = u * s + n * offset
    a, b = np.asarray(line['start'], float), np.asarray(line['end'], float)
    far = a if abs(float(a @ n) - offset) >= abs(float(b @ n) - offset) else b
    length = abs(float(far @ n) - offset)
    if length < 1e-6:
        return None
    direction = n * np.sign(float(far @ n) - offset)
    hits = []
    best = None
    for prim in part:
        for r in _ray_hits(base, direction, prim):
            if length - FOOT_REACH_MM <= r <= length + FOOT_REACH_MM:
                hits.append((r, prim))
                if best is None or abs(r - length) < abs(best[0] - length):
                    best = (r, prim)
    if best is None:
        return None
    xy = base + direction * best[0]
    alternatives = [(r, p) for r, p in hits
                    if p['id'] != best[1]['id'] or abs(r - best[0]) > 1e-4]
    def shared_vertex(p):
        q = best[1]
        if p['kind'] != 'line' or q['kind'] != 'line':
            return False
        if abs(float(_unit(p['start'], p['end']) @ _unit(q['start'], q['end']))) > 1 - 1e-8:
            return False
        return (any(math.dist(xy, p[k]) < 1e-4 for k in ('start', 'end')) and
                any(math.dist(xy, q[k]) < 1e-4 for k in ('start', 'end')))
    return {'xy': xy, 'on': best[1]['id'],
            'ambiguous': any(not shared_vertex(p) for _, p in alternatives)}


def corner_anchor(xy, part):
    candidates = []
    best = None
    for prim in part:
        if prim['kind'] != 'line':
            continue
        for name in ('start', 'end'):
            dist = math.dist(prim[name], xy)
            if dist <= CORNER_SNAP_MM:
                candidate = (dist, {'object_id': prim['id'], 'anchor': name}, prim[name])
                candidates.append(candidate)
                if best is None or dist < best[0]:
                    best = candidate
    if best:
        by_id = {p['id']: p for p in part}
        chosen = by_id[best[1]['object_id']]
        for candidate in candidates:
            if candidate is best:
                continue
            other = by_id[candidate[1]['object_id']]
            same_segment = (other['start'] == chosen['start'] and other['end'] == chosen['end'] or
                            other['start'] == chosen['end'] and other['end'] == chosen['start'])
            if same_segment or (math.dist(candidate[2], best[2]) > 1e-4 and
                                (best[0] > 1e-4 or abs(candidate[0] - best[0]) < 1e-4)):
                raise ValueError('ambiguous_corner_anchor')
    return best


def _arrows_at(arrows, point):
    return len(arrows) > 0 and float(np.min(np.linalg.norm(arrows - point, axis=1))) <= ARROW_REACH_MM


def find_dimension(centre, box_size, text_angle, nominal, scale, lines, part, excluded=(), arrows=None, need_arrows=False):
    length = abs(float(nominal)) * scale
    lines = [*lines, *excluded]
    tolerance = min(max(LENGTH_TOL_MIN_MM, length * LENGTH_TOL_SHARE), LENGTH_TOL_SHORT * length)
    centre = np.asarray(centre, float)
    hypotheses = []
    directional = box_size[0] / max(box_size[1], 1e-6) >= TEXT_SHORT_ASPECT

    def along_text(u):
        angle = math.degrees(math.atan2(u[1], u[0]))
        return not directional or abs((angle - text_angle + 90) % 180 - 90) <= TEXT_ANGLE_DEG

    visible = [line for line in lines if not line.get('hidden')]
    base_rows = [row for row in _rows(visible, centre) if along_text(row[0])]
    rows = [(u, n, offset, pieces, visible) for u, n, offset, pieces in base_rows]
    rows += [(u, n, offset, [line], lines)
             for u, n, offset, line in _carrier_candidates(lines, centre, length + 10.0,
                                                           [(u, o) for u, _, o, _ in base_rows]) if along_text(u)]
    for u, n, offset, pieces, pool in rows:
        ids = {p['id'] for p in pieces}
        kept = sorted(p['id'] for p in pieces if not p.get('excluded'))
        along = float(centre @ u)
        ends = [float(np.asarray(p[k], float) @ u) for p in pieces for k in ('start', 'end')]
        low, high = min(ends), max(ends)
        cross = _crossings(pool, u, n, offset, ids)
        side = abs(float(centre @ n) - offset)
        thick_row = bool(pieces) and all(p.get('thick') or p.get('excluded') for p in pieces)
        for i in range(len(cross)):
            for j in range(i + 1, len(cross)):
                s1, s2 = cross[i][0], cross[j][0]
                if abs((s2 - s1) - length) > tolerance:
                    continue
                if s1 > high + CROSS_REACH_MM or s2 < low - CROSS_REACH_MM:     # the row does not reach this pair
                    continue
                outside = max(s1 - along, along - s2, 0.0)
                if outside > 2 * box_size[0] + 5:
                    continue
                thick = thick_row + bool(cross[i][1].get('thick')) + bool(cross[j][1].get('thick'))
                arrowed = arrows is not None and any(_arrows_at(arrows, u * q + n * offset) for q in (s1, s2))
                if need_arrows and not arrowed:
                    continue
                if cross[i][1].get('excluded') or cross[j][1].get('excluded'):
                    continue
                feet = [foot_of(cross[k][1], cross[k][0], u, n, offset, part) for k in (i, j)]
                if not all(feet):
                    continue
                if any(f.get('ambiguous') for f in feet):
                    return None, 'ambiguous_contour_foot'
                hypotheses.append({'score': side + 0.5 * outside + THICK_PENALTY_MM * thick - (1.0 if arrowed else 0.0), 'u': u, 'n': n, 'offset': offset, 'pieces': kept,
                                   'extensions': [cross[i][1]['id'], cross[j][1]['id']], 'feet': feet, 'along': along,
                                   'arrowed': arrowed,
                                   'pair': (round(float(u[0]), 2), round(float(u[1]), 2), round(s1, 1), round(s2, 1))})
    if not hypotheses:
        return None, 'no_dimension_line_with_matching_extension_lines'
    # Only hypotheses measuring the same points on the same part objects are duplicates.
    by_position = {}
    for h in hypotheses:
        key = tuple(sorted((f['on'], *(round(float(v), 4) for v in f['xy'])) for f in h['feet']))
        if key not in by_position or h['score'] < by_position[key]['score']:
            by_position[key] = h
    hypotheses = sorted(by_position.values(), key=lambda h: (h['score'], h['pair']))
    best = hypotheses[0]
    def measured_feet(hypothesis):
        return sorted(tuple(round(float(v), 4) for v in foot['xy']) for foot in hypothesis['feet'])
    for other in hypotheses[1:]:
        # Arrows and text proximity rank alternatives; neither proves which part is measured.
        if measured_feet(other) != measured_feet(best) or {f['on'] for f in other['feet']} != {f['on'] for f in best['feet']}:
            return None, 'two_dimension_lines_fit_this_number'
    base = best['u'] * best['along'] + best['n'] * best['offset']
    angle = math.degrees(math.atan2(best['u'][1], best['u'][0]))
    feet = []
    for foot in best['feet']:
        try:
            corner = corner_anchor(foot['xy'], part)
        except ValueError:
            return None, 'ambiguous_corner_anchor'
        anchor = corner[1] if corner else None
        if anchor is None:
            primitive = next(p for p in part if p['id'] == foot['on'])
            if primitive['kind'] != 'line':
                return None, 'unsupported_contour_anchor'
            start, end = np.asarray(primitive['start'], float), np.asarray(primitive['end'], float)
            delta = end - start
            fraction = float((foot['xy'] - start) @ delta / (delta @ delta))
            anchor = {'object_id': primitive['id'], 'anchor': 'segment', 'parameter': min(1., max(0., fraction))}
        feet.append({'xy': [round(float(v), 6) for v in (corner[2] if corner else foot['xy'])], 'on': foot['on'],
                     'anchor': anchor})
    return {'dimension_line': best['pieces'][0] if best['pieces'] else None, 'dimension_row': best['pieces'], 'extension_lines': best['extensions'],
            'base': [round(float(v), 6) for v in base], 'angle': round(angle, 6), 'feet': feet}, None
