"""Drawing-specific corrections with an audit trail; never guess a missing digit."""
import re

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_DIAMETER = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])(?P<glyph>[Øø⌀∅φΦФф$]|[OО0][/\\]|[/\\][OО0])\s*(?=[+\-]?\d)")
# Letters the recogniser mixes up in the slanted drawing font. Used only to compare a word with the vocabulary below.
_LOOK_ALIKE = str.maketrans({"A": "А", "a": "а", "B": "В", "b": "в", "C": "С", "c": "с", "E": "Е", "e": "е", "H": "Н", "h": "н", "K": "К", "k": "к",
                             "M": "М", "m": "т", "O": "О", "o": "о", "P": "Р", "p": "р", "T": "Т", "t": "т", "X": "Х", "x": "х", "F": "Г",
                             "Y": "У", "y": "у", "u": "и", "n": "п", "g": "д", "6": "б", "3": "з", "0": "о", "д": "б", "w": "ш"})
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
    # The same without the dot before a number: 5omb14
    text = re.sub(r'(?<!\w)(\d+)\s*[oо][mтt][bв](?=\d)', holes, text)
    text = re.sub(r'\b[CС][mт](\d)', lambda m: _steel_grade(m, changes), text)
    text = _STANDARD.sub(lambda m: _standard_name(m, changes), text)
    text = _WORD.sub(lambda m: _vocabulary_word(m, changes), text)
    def diameter(match):
        if match.group() != 'Ø':
            changes.append({'rule': 'engineering_diameter_glyph', 'before': match.group(), 'after': 'Ø'})
        return 'Ø'
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
    return item
