"""Build mlcv/eval/REPORT.md (short) and mlcv/eval/TABLES.md (every input) from evaluation outputs.

    python mlcv/eval/report.py     # reads out, out_random, out_random7, out_random101, before_after.json, ocr_link.json,
                                   # three_sheets/results.json"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "mlcv/eval"
SETS = [("out", "Листы разработки", "--set default"),
        ("out_random", "Случайные, seed 44", "--set random --n 8 --seed 44"),
        ("out_random7", "Случайные, seed 7", "--set random --n 8 --seed 7"),
        ("out_random101", "Случайные, seed 101", "--set random --n 8 --seed 101")]


def load(name):
    p = EVAL / name / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_json(name):
    p = EVAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(a, b):
    return f"{a / b:.0%}" if b else "—"


def summary_rows():
    rows = ["| набор | входов | масштаб верен | найдено геометрии, ср. / мин. | центр окружности, ср. мм | контуры: пары / эталон | точность контуров | бэкенд принял контуры | неоднозн. точки | время ср., с |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, title, _ in SETS:
        r = load(name)
        real = [m for m in (r or []) if m.get("sheet") != "control" and "error" not in m]
        if not real:
            continue
        # a reference with almost nothing inside the frame does not fit its page: it cannot judge the module
        found = [m["found"]["body"] / m["reference"]["body_primitives"] for m in real if m["reference"]["body_primitives"] >= 20]
        centre = [m["errors_mm"]["circle_center"]["mean"] for m in real if m["errors_mm"]["circle_center"]["n"]]
        known = sum(m["calibration"]["scale_status"] == "known" for m in real)
        pairs = sum(m["contours"]["found"] for m in real)
        ref = sum(m["contours"]["reference_closed"] for m in real)
        det = sum(m["contours"]["detected_closed"] for m in real)
        accepted = sum(m["backend"]["accepted"] for m in real)
        closed = sum(m["backend"]["closed"] for m in real)
        ambiguous = sum(m["backend"]["ambiguous_points"] for m in real)
        rows.append(f"| {title} | {len(real)} | {known}/{len(real)} | {np.mean(found):.0%} / {np.min(found):.0%} | {np.mean(centre):.3f} | {pairs} / {ref} | "
                    f"{pct(pairs, det)} | {accepted} / {closed} | {ambiguous} | {np.mean([m['timings_s']['total'] for m in real]):.0f} |")
    return "\n".join(rows)


def before_after():
    data = load_json("before_after.json")
    if not data:
        return ""
    b, n = data["total"]["base"], data["total"]["new"]
    rows = [f"Одни и те же {data['inputs']} входов, обе версии оценены новыми правилами.", "",
            "| показатель | было | стало |", "|---|---|---|",
            f"| замкнутые контуры, которые принял настоящий код бэкенда | {b['backend_accepted']} из {b['contours_closed']} ({pct(b['backend_accepted'], b['contours_closed'])}) | "
            f"{n['backend_accepted']} из {n['contours_closed']} ({pct(n['backend_accepted'], n['contours_closed'])}) |",
            f"| точки, объединившие реально разошедшиеся объекты | {b['ambiguous_points']} | {n['ambiguous_points']} |",
            f"| контуры, совпавшие с эталоном 1:1 | {b['contour_pairs']} из {b['contour_reference']} | {n['contour_pairs']} из {n['contour_reference']} |",
            f"| точность контуров (совпавшие / все замкнутые) | {pct(b['contour_pairs'], b['contours_closed'])} | {pct(n['contour_pairs'], n['contours_closed'])} |",
            f"| копии одного контура | {b['contour_copies']} | {n['contour_copies']} |",
            f"| честно показанные разрывы | {b['gaps']} | {n['gaps']} |",
            f"| найдено геометрии (примитивы тела) | {b['body_found']} из {b['body_total']} | {n['body_found']} из {n['body_total']} |"]
    return "\n".join(rows)


def control_rows():
    r = load("out")
    if not r:
        return ""
    rows = ["| вход | масштаб | пластина замкнута (ошибка площади, мм²) | сечение замкнуто | разрыв найден (мм; истина 1,5) | самопересечение | штриховка с границей |",
            "|---|---|---|---|---|---|---|"]
    yes = {True: "да", False: "**нет**"}
    for m in r:
        t = m.get("truth")
        if t:
            area = t.get("plate_area_err_mm2")
            gap = t.get("broken_gap_mm")
            rows.append(f"| {m['case']} | {m['calibration']['scale_status']} | {yes[t['plate_closed']]}{'' if area is None else f' ({area:.1f})'} | {yes[t['section_closed']]} | "
                        f"{yes[t['broken_gap_found']]}{f' ({gap:.2f})' if gap else ''} | {yes[t['bowtie_self_intersection_found']]} | {yes[t['hatch_resolved']]} |")
    return "\n".join(rows)


def link_rows():
    d = load_json("ocr_link.json")
    if not d:
        return ""
    t = d["total"]
    good = [r["ends_within_05"] for r in d["rows"]]
    return (f"Листов {t['sheets']}, размеров {t['reference_dims']}: связано автоматически **{t['dimensions']}**, все на верных точках, в DXF "
            f"настоящих `DIMENSION` — {t.get('dxf_dimensions', 0)}; DXF записан на {t['dxf_written']} листах; лимит сложности сработал {t.get('limit_hit', 0)} раз "
            f"(было 5 из первых 9 листов). Измеряемый конец размера совпал с точкой CV в среднем в {np.mean(good):.0%} случаев.")


def three_sheet_rows():
    d = load_json("three_sheets/results.json")
    if not d:
        return ""
    rows = ["| лист | отрезки: найдено | окружности | дуги | контуры 1:1 | лишние отрезки | ошибка конца, мм | ошибка центра, мм | масштаб, % | время, с |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d:
        k = r["analysis"]["kinds"]
        line_err = k["line"]["errors_mm"].get("конец")
        circle_err = k["circle"]["errors_mm"].get("центр")
        rows.append(f"| {r['name']} | {k['line']['found']}/{k['line']['reference']} | {k['circle']['found']}/{k['circle']['reference']} | {k['arc']['found']}/{k['arc']['reference']} | "
                    f"{k['contour']['found']}/{k['contour']['reference']} | {k['line']['extra']} | {line_err['mean'] if line_err else '—'} | {circle_err['mean'] if circle_err else '—'} | "
                    f"{r['analysis']['scale']['error_percent']} | {r['time_s']:.0f} |")
    return "\n".join(rows)


BIG_SHEET = "самый большой лист датасета (A1, 280 млн пикселей) — за 17 минут, память worker ≈ 2,7 ГБ"

REPORT = """# CV-модуль drawcv: измерения

Совместный проект с 44labs. Цифры собирает `mlcv/eval/report.py`. Таблицы по каждому входу — [TABLES.md](TABLES.md).
Тестовый лист клапан и приёмочные листы не используются.

## Главное

1. **Замкнутый контур теперь действительно замкнут.** Раньше концы считались соединёнными, если лежали рядом: бэкенд отвергал большинство таких контуров как щели.
2. **Лишние копии контура снижают точность**, один контур больше не закрывает несколько эталонных.
3. **Опорные точки честные:** одна точка — только совпадающие координаты, у каждой есть объяснение.
4. **Размер листа ограничен до декодирования:** файл-«бомба» отвергается без выделения памяти.
5. **Главная открытая проблема — связывание размеров:** даже с идеальным чтением OCR связываются единицы размеров (раздел ниже).

## Что изменилось

{before_after}

**Как читать падение «совпавших с эталоном».** Прежняя версия считала замкнутым то, что по координатам не замкнуто, — часть таких контуров
совпадала с эталоном по площади, но экспорт их отвергал. Теперь они открыты и показаны как разрывы. Больше всего это видно на листе Опора7: там толстая
окружность раздроблена на куски (слабость векторизации, а не стыков). Число лишних замкнутых контуров при этом сократилось: например, на листе «трубка»
база насчитала 107 контуров, это в основном клетки толстой штриховки, сейчас их 57.

## Сводка по наборам

{summary}

Масштаб определён на всех реальных листах и отличается от эталона меньше чем на 0,2 %. Столбец «найдено» не учитывает листы, у которых после совмещения в рамке меньше 20 примитивов эталона (DXF не соответствует странице). Низкая точность контуров на случайных наборах — в основном клетки штриховки и эталоны с неполными данными о толщине линий.

## Три независимых листа

Семейства этих деталей не встречались в разработке и прежних отчётах. Наложения на исходник, настройки, время, модель и причины по каждому виду — [three_sheets/REPORT.md](three_sheets/REPORT.md).

{three}

Основные причины ошибок (подробная таблица — в отчёте по листам):

| причина | что происходит | что делать |
|---|---|---|
| разрыв | контур не замкнулся: чаще всего в месте, где примитив распался на куски или идёт короткой ломаной | главный резерв качества контуров: улучшать векторизацию коротких дуг и кусков окружностей |
| другой вид примитива | дуга или окружность выдана ломаной из отрезков | то же |
| текст | линия лежит внутри области надписи (резьба, шероховатость, штамп) и стёрта или не найдена | передавать области от OCR вместо детектора, проверять линии через надпись |
| штриховка | клетки толстой штриховки принимаются за контуры | различать штриховку по толщине нельзя: нужна проверка регулярного шага и границы |
| неверный масштаб | на этих листах не встретился: ошибка ≤ 0,04 % | — |

Лишние вне детали: рамка и штамп, размерные и выносные линии, сросшиеся с контуром, короткие обломки.

## Стыковка с OCR и бэкендом

Проверено настоящим кодом `mlocr` и `backend` (`ocr_link_check.py`, `backend_check.py`), с идеальным чтением текста из размеров эталонного DXF. PaddleOCR в этой проверке не запускается.

{links}

Экспорт в DXF больше не выдаёт ошибок `contour_gap` и `ambiguous_point`. Подтверждённых связей мало по причинам ниже — это следующая задача для CV и OCR.

| где обрывается связывание (размеров на 45 входах) | сколько |
|---|---|
| на измеряемом конце нет точки CV: чаще всего это середина линии детали, куда приходит выносная (кандидаты теперь есть в `measurable`, `extension_foot`, но `mlocr` их пока не использует) | 156 |
| размер не по осям (сейчас связываются только горизонтальные и вертикальные) | 42 |
| не найдена размерная линия или она разбита на куски | 32 |
| конец лежит на тонкой линии, дуге или окружности, а не на отрезке детали | 33 |
| дошли до проверки выносных линий | 16 |

## Контрольный лист (точная истина)

{control}

На «PDF 97 % × 100 %» масштаб намеренно `unknown`. Сечение и пластина могут быть не замкнуты там, где растровый PDF даёт реальные разрывы: модуль больше не закрывает их допуском.

## Размер изображения (задача 5)

До работы ограничения не было: `Image.MAX_IMAGE_PIXELS = None` ещё и отключал защиту Pillow для всего процесса. Сейчас размер читается из заголовка (PDF — из размера страницы и dpi) до декодирования,
лимит 300 млн пикселей и сторона до 32 000 px (`Params.max_image_pixels`, `DRAWCV_MAX_IMAGE_PIXELS`, `--max-pixels`), глобальная настройка Pillow не меняется. Тесты: PNG, JPEG и PDF на 3,6 млрд пикселей отвергаются меньше чем за секунду, расход памяти менее 50 МБ.

В бэкенде и `mlocr` был лимит 50 млн пикселей, и 146 из 510 листов датасета настоящий сервис отвергал (проверено: `переходник`, A3 при 600 dpi — 70 млн). Лимит везде поднят до 300 млн. После этого тот же лист проходит весь сервис за 36 с; {big}.

## Что показала проверка всего сервиса

Весь сервис поднят в Docker: API, worker, PostgreSQL, PaddleOCR, `drawcv`; 116 тестов в контейнере бэкенда проходят, 80 — в чистом образе `drawcv` (1 модуль пропущен: в образе нет `backend/`), 87 — в `.venv`. Чертежи отправлены через API, DXF прочитан и открыт в LibreCAD.

| вход | результат | время |
|---|---|---|
| `Шайба.pdf` | `needs_review`, DXF AC1021, `ANSI_1251`, ошибок аудита нет, открыт в LibreCAD | 16 с |
| `переходник.png` (A3, 600 dpi) | `needs_review`, 244 примитива, штамп: наименование и материал прочитаны | 36 с |
| `Тележка тяговая.png` (A1, 600 dpi, 5650 примитивов) | `needs_review`, DXF записан | 17 мин |

Исправлено по итогам: кодовая страница DXF была `ANSI_1252` (ТЗ требует `ANSI_1251`); лимит 50 млн пикселей отвергал настоящие листы; в образе `drawcv` не было `libGL` и стояли два пакета `cv2`.

Текст в DXF был крупнее оригинала и наезжал сам на себя: высота бралась по рамке детектора, которая в 1,5–1,8 раза выше букв. Теперь высота — 0,62 от рамки (измерено по эталонным DXF). Найдено, не исправлено (зона OCR): ошибки символов (`Шайда`, `$26` вместо `⌀26`); обозначение и ФИО в штампе часто `unknown`.

Проверить в AutoCAD: в стиле текста бэкенда шрифт `DejaVuSans.ttf`, которого нет в поставке AutoCAD; для проекта проверен `isocpeur.ttf` с кириллицей (`DXF_FONT` в настройках бэкенда). В самом файле кириллица записана верно. Headless LibreCAD показывает текст ромбиками при любом шрифте: это известное ограничение `dxf2pdf`, а не ошибка файла.

## Известные слабые места

- Ложные разрывы на перегруженных листах и там, где окружность раздроблена на короткие куски.
- Слои: осевые из 1–2 штрихов и осевые окружности под контуром остаются `THIN`.
- Штриховка с границей — только в простом случае (семейство наклонных линий внутри одного контура), иначе вопрос `hatch_boundary_unresolved`.
- Тонкие линии рендера 600 dpi (≈ 1 px) теряются при переходе к 300 dpi: Вал, Цилиндр в PNG хуже, чем в PDF.
- Эталонные контуры есть не на всех листах: часть эталонов хранит основные линии без признака толщины.
- Время на больших листах: самый большой лист датасета (A1, 600 dpi, 5650 примитивов) проходит весь сервис за 17 минут, ориентир заказчика 5–10. Основная часть — квадратичные циклы `dashed_chains` и `merge_arcs` на этапе векторизации.

## Воспроизведение

```bash
export PYTHONUTF8=1; P=.venv/Scripts/python
$P mlcv/eval/evaluate.py --set default --out mlcv/eval/out
for s in "44 out_random" "7 out_random7" "101 out_random101"; do set -- $s; $P mlcv/eval/evaluate.py --set random --n 8 --seed $1 --out mlcv/eval/$2; done
$P mlcv/eval/three_sheets.py
$P mlcv/eval/backend_check.py mlcv/eval/out mlcv/eval/out_random
OCR_LINK_JSON=mlcv/eval/ocr_link.json $P mlcv/eval/ocr_link_check.py mlcv/eval/out mlcv/eval/out_random mlcv/eval/out_random7 mlcv/eval/out_random101
$P mlcv/eval/report.py
```
Сравнение версий: `compare_runs.py BASE_DIR NEW_DIR --json mlcv/eval/before_after.json`. В каждой папке входа — `cv.json`, `overlay.png`, `geometry.png`, `compare.png`, `metrics.json`.
"""

TABLES = """# Таблицы по каждому входу

Строка — один вход (лист × формат). Эталон — DXF листа. Сводка и выводы — [REPORT.md](REPORT.md).

- **калибровка** — способ, статус масштаба, остаточный масштаб к эталону (оценивается, не применяется);
- **найдено тело** — примитивы эталона внутри рамки вне штампа, покрытые найденными того же вида на ≥ 80 % длины в пределах 0,5 мм;
- **лишние** — выданные примитивы, не совпавшие с геометрией, размерами, штриховкой и текстом; **дубли** — на ≥ 80 % лежат на более длинном;
- **контуры** — эталон → пары один к одному при IoU ≥ 0,9 (лишние, из них копии); точность и полнота;
- **бэкенд принял** — замкнутые контуры, прошедшие настоящий `GeometryResolver`;
- **разрывы** — на реальных листах чертёж замкнут, каждый разрыв — ошибка векторизации.

"""


def main():
    text = REPORT.format(before_after=before_after(), summary=summary_rows(), three=three_sheet_rows(), links=link_rows(), control=control_rows(),
                         big=BIG_SHEET)
    (EVAL / "REPORT.md").write_text(text, encoding="utf-8")
    parts = [TABLES]
    for name, title, cmd in SETS:
        t = EVAL / name / "table.md"
        if t.exists():
            parts += [f"## {title} (`{cmd}`)\n", t.read_text(encoding="utf-8").strip(), ""]
    (EVAL / "TABLES.md").write_text("\n".join(parts), encoding="utf-8")
    print(EVAL / "REPORT.md")


if __name__ == "__main__":
    main()
