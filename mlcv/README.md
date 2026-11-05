# mlcv — CV-часть сервиса оцифровки чертежей

Совместный проект с **44labs**. Здесь **drawcv**: модуль на OpenCV, который из PNG, JPEG или PDF делает `cv.json` — геометрию
чертежа в миллиметрах листа для бэкенда и OCR. Модуль не читает текст и не определяет масштаб вида: это задача OCR.

| шаг | что делает |
|---|---|
| Подготовка | читает страницу, проверяет размер **до декодирования**, поворачивает, снимает наклон до 5°, выравнивает фон, приводит к 300 dpi, калибрует px → мм по рамке |
| Примитивы | отрезки, окружности, дуги с уверенностью; текст и штриховка отделяются; шум и дубли помечаются, а не прячутся |
| Контуры | концы рядом сводятся в **одну точку с записью причины** или контур остаётся открытым; разрывы и самопересечения показаны |
| Связь с OCR | точки с ID и объяснением («начало отрезка», «центр отверстия»), кандидаты измеряемых участков, координаты в пикселях исходника |

Формат выхода — [CV_JSON.md](CV_JSON.md). Результаты — [eval/REPORT.md](eval/REPORT.md).

## Установка

Python 3.11+ (проверено на 3.14 и закреплённых версиях бэкенда). Всё ставится в окружение проекта `.venv`:

```bash
py -3.14 -m venv .venv                                        # если окружения ещё нет
.venv/Scripts/python -m pip install -r mlcv/requirements-integration.txt
# RapidOCR тянет opencv-python-headless, а нужен один пакет cv2 — как в backend/pyproject.toml:
.venv/Scripts/python -m pip uninstall -y opencv-python-headless opencv-contrib-python
.venv/Scripts/python -m pip install --no-deps opencv-contrib-python==4.10.0.84
export PYTHONUTF8=1                                           # кириллица в консоли Windows
```

Только сам модуль: `requirements.txt`; с тестами и оценкой: `requirements-dev.txt`; с проверкой стыковки с настоящими
`backend/` и `mlocr/`: `requirements-integration.txt` (PaddleOCR для неё не нужен). Linux/macOS — то же с `.venv/bin/python`.
Веса детектора текста RapidOCR (≈ 30 МБ) скачиваются при первом запуске; в Docker-образе — при сборке.

## Запуск

```bash
# DIR/<имя>/cv.json + overlay.png, geometry.png, prep_compare.png, params.json
PYTHONPATH=mlcv .venv/Scripts/python -m drawcv dataset/dataset2/Шайба.pdf лист.png скан.jpg --out out_cv
```

Параметры: `--page N`, `--sheet A3`, `--rotate 90`, `--px-per-mm 11.81 [--origin-px x,y]` (ручная калибровка, помечается в `cv.json`),
`--text-provider rapidocr|components|file|none`, `--ocr-json ocr.json`, `--max-pixels N`, `--params params.json` (любые поля
[drawcv/params.py](drawcv/params.py)), `--no-images`.

Из Python и из бэкенда:

```python
from drawcv import Params, run
cv = run("лист.pdf", out_dir="out/лист", params=Params(snap_tol_mm=0.3))   # dict; то же в out/лист/cv.json
```

```bash
export CV_ADAPTER=drawcv.adapter:run_cv          # значение по умолчанию в бэкенде; картинки проверки — в <run>/cv/
export DRAWCV_PARAMS=/path/params.json           # необязательно
export DRAWCV_MAX_IMAGE_PIXELS=300000000         # необязательно
```

Адаптер не бросает исключений: при ошибке возвращает `{"status": "error", "error": "..."}`.

## Проверка

```bash
cd mlcv && ../.venv/Scripts/python -m pytest -q && cd ..      # все тесты, ≈ 2 мин

P=.venv/Scripts/python
$P mlcv/eval/evaluate.py --set quick                          # Шайба ×3 формата + контрольный лист ×7, против эталонных DXF
$P mlcv/eval/evaluate.py --set default                        # + листы разработки
$P mlcv/eval/evaluate.py --set random --n 8 --seed 44 --out mlcv/eval/out_random   # случайные листы (минимум две выборки)
$P mlcv/eval/three_sheets.py                                  # три независимых листа: виды, причины ошибок, наложения
$P mlcv/eval/backend_check.py mlcv/eval/out                   # что принимает настоящий код бэкенда
$P mlcv/eval/ocr_link_check.py mlcv/eval/out                  # цепочка CV → mlocr → backend → DXF с идеальным чтением
$P mlcv/eval/compare_runs.py СТАРЫЙ_DIR НОВЫЙ_DIR             # сравнение двух версий по строкам
$P mlcv/eval/report.py                                        # REPORT.md и TABLES.md
```

Docker (чистое окружение, без интернета после сборки):

```bash
docker build -t drawcv mlcv
docker run --rm drawcv                                                    # тесты
docker run --rm -v "D:/путь/к/листам:/data" drawcv drawcv /data/лист.pdf --out /data/out
```

В Git Bash перед `docker run -v` нужен `MSYS_NO_PATHCONV=1`. Тесты настоящего бэкенда в образе пропускаются: в нём нет папок `backend/` и `mlocr/`.

## Структура

```
mlcv/
├─ drawcv/
│  ├─ ingest.py      чтение PNG / JPEG / PDF, лимит размера до декодирования
│  ├─ preprocess.py  поворот, наклон, фон, разрешение, бинаризация
│  ├─ calibrate.py   px ↔ мм листа: обрез / рамка ГОСТ по сторонам, PDF, ручная
│  ├─ transform.py   матрицы source_px → work_px → sheet_mm и обратно
│  ├─ textmask.py    области текста (RapidOCR / файл OCR / компоненты), какая модель использована
│  ├─ vectorize.py   скелет → отрезки и дуги
│  ├─ primitives.py  слияние, пунктир, окружности из дуг, толщина, уверенность, дубли
│  ├─ hatch.py       семейства линий штриховки
│  ├─ closure.py     сведение концов в одну точку, проверки, запись правок
│  ├─ topology.py    контуры, разрывы, самопересечения
│  ├─ anchors.py     объяснение точек и кандидаты измеряемых участков для OCR
│  ├─ pipeline.py    весь маршрут и роли объектов
│  ├─ output.py      сборка cv.json
│  ├─ overlay.py     картинки проверки
│  ├─ adapter.py     вызов из бэкенда
│  └─ synth.py       контрольный лист с точной истиной
├─ tests/            pytest: форматы, преобразования, стыки, лимит, опорные точки, настоящий бэкенд
├─ eval/             оценка против эталонов, отчёты; выходы eval/out*/ в git не попадают
├─ examples/         примеры «до/после» и наложений
└─ CV_JSON.md        формат выхода
```

## Результаты в одном абзаце

Масштаб определён на всех реальных листах с ошибкой меньше 0,2 %. Замкнутые контуры, принятые настоящим кодом бэкенда: **было 36 %, стало 100 %** (640 из 641),
точек, объединявших разошедшиеся объекты: **было 1745, стало 0**. Цена честности — показано больше разрывов, контуров совпало с эталоном 121 из 246 вместо 138.
Подробно и с разбором причин — [eval/REPORT.md](eval/REPORT.md), три независимых листа — [eval/three_sheets/REPORT.md](eval/three_sheets/REPORT.md).

Примеры: [до/после подготовки](examples/shaiba_png_prep_compare.jpg), [объекты поверх исходника](examples/opora_png_overlay.jpg),
[контуры и дефекты](examples/opora_png_geometry.jpg), [сравнение с эталоном](examples/shaiba_png_compare.jpg). Пересобрать: `python mlcv/eval/make_examples.py`.

## Что учесть при стыковке

- Лимит размера единый: 300 млн пикселей в бэкенде, `mlocr` и `drawcv` (`MAX_IMAGE_PIXELS`, `DRAWCV_MAX_IMAGE_PIXELS`).
- `id` объектов и точек действует в пределах одного растра (`raster_identity`). Между разными обрезками сопоставляйте по координатам `sheet_mm`.
- Поднимая версию в `pyproject.toml`, обновите её в `backend/uv.lock`: образ бэкенда собирается с `uv sync --locked`.
- Подробности и правила — [CV_JSON.md](CV_JSON.md). Запуск всего сервиса вместе — [главный README](../README.md).
