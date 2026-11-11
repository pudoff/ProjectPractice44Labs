"""Drawing-specific corrections with an audit trail; never guess a missing digit."""
import re

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_DIAMETER = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])(?P<glyph>[Øø⌀∅φΦФф]|[OО0][/\\]|[/\\][OО0])\s*(?=[+\-]?\d)")
# PaddleOCR can read a dimension arrow touching the number as a dollar sign.
# Treat `$` as Ø only when the whole suffix is a plausible size/fit/tolerance;
# a trailing lone letter (e.g. `$32f`, where `f` is an arrow) stays ambiguous.
_DOLLAR_DIAMETER = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9])\$(?P<value>\d+(?:[.,]\d+)?(?:[A-Za-z]{1,2}\d{1,2})?(?:[+\-]\d+(?:[.,]\d+)?)*(?:±\d+(?:[.,]\d+)?)?)(?![A-Za-zА-Яа-яЁё0-9])"
)
_AMBIGUOUS_DOLLAR = re.compile(r"\$\d+(?:[.,]\d+)?[A-Za-z]")
# Letters the recogniser mixes up in the slanted drawing font. Used only to compare a word with the vocabulary below.
_LOOK_ALIKE = str.maketrans({"A": "А", "a": "а", "B": "В", "b": "в", "C": "С", "c": "с", "E": "Е", "e": "е", "H": "Н", "h": "н", "K": "К", "k": "к",
                             "M": "М", "m": "т", "O": "О", "o": "о", "P": "Р", "p": "р", "T": "Т", "t": "т", "X": "Х", "x": "х", "F": "Г",
                             "Y": "У", "y": "у", "u": "и", "n": "п", "g": "д", "d": "б", "6": "б", "3": "з", "0": "о", "д": "б", "w": "ш"})
# Fixed words of the title block and common part names. A word is replaced only when it folds to exactly one of them.
_VOCABULARY = ("Разраб.", "Пров.", "Утв.", "Лист", "Листов", "Лит.", "Масса", "Масштаб", "Изм.", "Подп.", "Дата", "Инв.", "подл.", "дубл.",
               "Взам.", "Перв.", "примен.", "Справ.", "докум.", "Копировал", "Формат", "Обозначение", "Наименование", "Подпись", "контр.",
               "Шайба", "Гайка", "Болт", "Винт", "Вал", "Втулка", "Опора", "Фланец", "Корпус", "Крышка", "Кольцо", "Прокладка", "Штуцер",
               "Ниппель", "Плита", "Стойка", "Сфера", "Цилиндр", "Переходник", "Рычаг", "Патрубок", "Пробка", "Лист", "Ребро", "Тележка",
               "ГОСТ", "Сталь", "Ст", "Сертификат", "Размер", "Размеры", "справок", "фаски", "скоса", "работа", "Работа", "Работа")


def _fold(word):
    return word.translate(_LOOK_ALIKE).lower().rstrip('.').replace('б', 'в').replace('и', 'п').replace('у', 'п')


_BY_FOLD = {}
for _word in _VOCABULARY:
    _BY_FOLD.setdefault(_fold(_word), set()).add(_word)
# The name of a standard, even when it is glued to its number: F0CT1051-88
_STANDARD = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])[ГF][О0O][СC][ТT](?=[\s\d]|$)")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]*[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9]*\.?")


def _vocabulary_word(match, changes):
    """A word that differs from a known drawing word only by letters the recogniser mixes up. Digits inside a word count
    only when the word has letters of a known word around them; a number on its own is never changed."""
    original = match.group()
    bare = original.rstrip('.')
    if len(bare) < 3 or bare.isdigit():
        return original
    known = _BY_FOLD.get(_fold(bare))
    if not known or len(known) != 1:
        return original
    word = next(iter(known))
    corrected = word if word.endswith('.') or not original.endswith('.') else word + '.'
    if bare[0].islower() and word[0].isupper() and not re.search(r'[A-ZА-ЯЁ]', bare):
        corrected = corrected[0].lower() + corrected[1:]
    if corrected != original:
        changes.append({'rule': 'drawing_vocabulary', 'before': original, 'after': corrected})
    return corrected


def normalize_text(raw: str) -> tuple[str, list[dict]]:
    changes = []
    text = raw.strip()
    # OCR boxes often span two adjacent title-block cells. Put the visible
    # word boundary back when case/punctuation makes it unambiguous.
    def insert_space(pattern, rule):
        nonlocal text
        updated, count = re.subn(pattern, r"\1 \2", text)
        if count:
            changes.append({'rule': rule, 'before': text, 'after': updated})
            text = updated
    # A case jump matters only across scripts (Cyrillic word glued to a Latin one); a merged
    # same-script token ("MaccaMacwmad") stays untouched as OCR evidence.
    insert_space(r"([a-z])([А-ЯЁ])", 'word_boundary_case')
    insert_space(r"([а-яё])([A-Z])", 'word_boundary_case')
    # Split adjacent title-block abbreviations, but keep identifiers such as 03.XXXX.03 intact,
    # and do not break initials: a capital followed by a dot is an initial, not a new sentence.
    insert_space(r"(?<=[A-Za-zА-Яа-яЁё])([.!?])([A-ZА-ЯЁ])(?![.!?])", 'word_boundary_punctuation')
    # Common title-block label captured as one OCR word (e.g. Nдокум.).
    insert_space(r"(№|N)(докум\.)", 'title_block_label_boundary')
    insert_space(r"(Взам\.)(инв\.)", 'title_block_label_boundary')
    # A specification name can be glued to the preceding steel designation.
    insert_space(r"([А-Яа-яЁё])([ГF][О0O][СC][ТT](?=\d))", 'standard_boundary')
    # Numbered notes such as "4фаски" have a clear numeric/word boundary; the hole
    # abbreviation "8отв." is a single token and must not be split.
    insert_space(r"(\d)((?!отв)[а-яё]{3,})", 'number_word_boundary')
    def word(match):
        original = match.group()
        if 'm' in original and re.search(r'[А-Яа-яЁё]', original):
            corrected = original.replace('m', 'т')
            changes.append({'rule': 'cyrillic_italic_t', 'before': original, 'after': corrected})
            return corrected
        return original
    text = _TOKEN.sub(word, text)
    def holes(match):
        before = match.group()
        # Keep the separator as it was read: some sheets print "8отв.", others "4 отв.",
        # and a forced space breaks the diameter recovery for the unspaced form.
        after = match[1] + match[2] + 'отв.'
        if before != after:
            changes.append({'rule': 'drawing_hole_abbreviation', 'before': before, 'after': after})
        return after
    # A whole engineering abbreviation is recognizable even when all three
    # Cyrillic letters were read as Latin lookalikes (e.g. 8omb.).
    text = re.sub(r'(?<!\w)(\d+)(\s*)[oо][mтt][bв]\.', holes, text)
    # The same without the dot before a number: 5omb14
    text = re.sub(r'(?<!\w)(\d+)(\s*)[oо][mтt][bв](?=\d)', holes, text)
    text = re.sub(r'\b[CС][mт](\d)', lambda m: _steel_grade(m, changes), text)
    text = _STANDARD.sub(lambda m: _standard_name(m, changes), text)
    text = _WORD.sub(lambda m: _vocabulary_word(m, changes), text)
    def diameter(match):
        if match.group() != 'Ø':
            changes.append({'rule': 'engineering_diameter_glyph', 'before': match.group(), 'after': 'Ø'})
        return 'Ø'
    def dollar_diameter(match):
        changes.append({'rule': 'engineering_diameter_glyph', 'before': '$', 'after': 'Ø'})
        return 'Ø' + match['value']
    text = _DOLLAR_DIAMETER.sub(dollar_diameter, text)
    text = _DIAMETER.sub(diameter, text)
    return text, changes


def _standard_name(match, changes):
    if match.group() != 'ГОСТ':
        changes.append({'rule': 'standard_name', 'before': match.group(), 'after': 'ГОСТ'})
    return 'ГОСТ'


def _steel_grade(match, changes):
    corrected = 'Ст' + match[1]
    if match.group() != corrected:
        changes.append({'rule': 'steel_grade', 'before': match.group(), 'after': corrected})
    return corrected


def enrich(item: dict) -> dict:
    from .semantics import parse_engineering_notation
    raw = item.setdefault('raw_text', item['text'])
    text, corrections = normalize_text(raw)
    item.update(text=text, corrections=corrections, notation=parse_engineering_notation(raw))
    if item['notation']['warning'] and item['notation']['kind'] != 'unknown':
        item.setdefault('issues', []).append({'code': 'invalid_notation', 'message': item['notation']['warning']})
        item['needs_review'] = True
    if _AMBIGUOUS_DOLLAR.fullmatch(raw):
        item.setdefault('issues', []).append({'code': 'ambiguous_diameter_glyph',
                                               'message': 'A dollar-like glyph before a number may be a dimension arrow; confirm the source image.'})
        item['needs_review'] = True
    return item
