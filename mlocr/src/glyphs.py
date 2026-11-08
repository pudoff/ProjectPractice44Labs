"""Recover a dropped diameter prefix only from a visible slashed-ring glyph."""
import cv2
import re
import numpy as np
from .semantics import parse_engineering_notation

_INLINE_O_DIAMETER = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])[oоОO](?=\d)")


def text_crop(image, polygon):
    points = np.asarray(polygon, dtype=float)
    edges = np.roll(points, -1, axis=0)-points
    edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    u = edge/max(float(np.linalg.norm(edge)), 1e-6)
    if (abs(u[0]) >= abs(u[1]) and u[0] < 0) or (abs(u[0]) < abs(u[1]) and u[1] > 0):
        u = -u
    v = np.array([-u[1], u[0]])
    x, y = points @ u, points @ v
    width, height = int(np.ceil(np.ptp(x))), int(np.ceil(np.ptp(y)))
    if min(width, height) < 5:
        return None
    matrix = np.array([[u[0], u[1], -x.min()], [v[0], v[1], -y.min()]])
    return cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))


def _glyphs(crop):
    if crop is None:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    roots = [(cv2.boundingRect(c), i) for i, c in enumerate(contours) if hierarchy[0][i][3] == -1]
    # Ignore tiny specks and long annotation rules. Inspect the leading glyph,
    # never an interior zero/eight of the recognized number.
    roots = [(box, i) for box, i in roots if box[2] <= 1.3*box[3]]
    glyph_height = max((box[3] for box, _ in roots), default=0)
    roots = [(box, i) for box, i in roots if box[3] >= .6*glyph_height]
    output = []
    for (x, y, w, h), index in sorted(roots):
        holes = []
        for j, contour in enumerate(contours):
            if hierarchy[0][j][3] == index and cv2.contourArea(contour) > .025*w*h:
                m = cv2.moments(contour)
                holes.append((m['m10']/m['m00'], m['m01']/m['m00']))
        ring = False
        if len(holes) == 2 and .45 <= w/h <= 1.3:
            dx, dy = holes[1][0]-holes[0][0], holes[1][1]-holes[0][1]
            ring = abs(dx) > .22*w and abs(dy) > .06*h and dx*dy > 0
        output.append(((x, y, w, h), ring))
    return output


def leading_slashed_ring(crop):
    glyphs = _glyphs(crop)
    return bool(glyphs and glyphs[0][0][0] <= .3*crop.shape[0] and glyphs[0][1])


def recover_diameter(item, image, local_polygon):
    holes = re.fullmatch(r'(\d+отв\.)\s*(\d+(?:[.,]\d+)?)', item['text'])
    inline_matches = list(_INLINE_O_DIAMETER.finditer(item['text']))
    if item['notation']['kind'] not in {'linear', 'linear_tolerance'} and not holes and not inline_matches:
        return item
    crop = text_crop(image, local_polygon)
    glyphs = _glyphs(crop)
    leading_ring = leading_slashed_ring(crop) if not holes and not inline_matches else False
    found = leading_ring if not holes and not inline_matches else (
        any(ring and box[0] > .5*crop.shape[1] for box, ring in glyphs) if holes else False)
    inline_match = None
    if inline_matches and crop is not None:
        ring_centres = [((box[0] + box[2] / 2) / crop.shape[1], box)
                        for box, ring in glyphs if ring]
        if ring_centres:
            inline_match = min(inline_matches,
                key=lambda match: min(abs((match.start() + .5) / max(len(item['text']), 1) - x)
                                      for x, _ in ring_centres))
            distance = min(abs((inline_match.start() + .5) / max(len(item['text']), 1) - x)
                           for x, _ in ring_centres)
            found = distance <= .14
    if found:
        before = item['text']
        if holes:
            corrected = holes[1]+'Ø'+holes[2]
        elif inline_match:
            corrected = before[:inline_match.start()] + 'Ø' + before[inline_match.end():]
        elif leading_ring and before.startswith('0'):
            # Sometimes the recogniser renders the slashed ring as a literal zero;
            # replace that glyph instead of inserting an extra Ø before it.
            corrected = 'Ø' + before[1:]
        else:
            corrected = 'Ø'+before
        item['corrections'].append({'rule': 'diameter_visible_in_raster', 'before': before, 'after': corrected})
        item['text'] = corrected
        item['notation'] = parse_engineering_notation(corrected)
        item['notation']['raw_text'] = item['raw_text']
        item['needs_review'] = True
        item.setdefault('issues', []).append({'code': 'diameter_recovered', 'message': 'A slashed-ring prefix was recovered from pixels; confirm the original glyph.'})
    return item
