"""Drawing-specific corrections with an audit trail; never guess a missing digit."""
import re

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_DIAMETER = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])(?P<glyph>[Øø⌀∅φΦФф]|[OО0][/\\]|[/\\][OО0])\s*(?=[+\-]?\d)")


def normalize_text(raw: str) -> tuple[str, list[dict]]:
    changes = []
    def word(match):
        original = match.group()
        if 'm' in original and re.search(r'[А-Яа-яЁё]', original):
            corrected = original.replace('m', 'т')
            changes.append({'rule': 'cyrillic_italic_t', 'before': original, 'after': corrected})
            return corrected
        return original
    text = _TOKEN.sub(word, raw.strip())
    def holes(match):
        before = match.group()
        after = match[1]+'отв.'
        if before != after:
            changes.append({'rule': 'drawing_hole_abbreviation', 'before': before, 'after': after})
        return after
    # A whole engineering abbreviation is recognizable even when all three
    # Cyrillic letters were read as Latin lookalikes (e.g. 8omb.).
    text = re.sub(r'(?<!\w)(\d+)\s*[oо][mтt][bв]\.', holes, text)
    def diameter(match):
        if match.group() != 'Ø':
            changes.append({'rule': 'engineering_diameter_glyph', 'before': match.group(), 'after': 'Ø'})
        return 'Ø'
    text = _DIAMETER.sub(diameter, text)
    return text, changes


def enrich(item: dict) -> dict:
    from .semantics import parse_engineering_notation
    raw = item.setdefault('raw_text', item['text'])
    text, corrections = normalize_text(raw)
    item.update(text=text, corrections=corrections, notation=parse_engineering_notation(raw))
    if item['notation']['warning'] and item['notation']['kind'] != 'unknown':
        item.setdefault('issues', []).append({'code': 'invalid_notation', 'message': item['notation']['warning']})
        item['needs_review'] = True
    return item
