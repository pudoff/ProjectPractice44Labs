# mlcv — CV/ML-часть сервиса оцифровки чертежей

Совместный проект с **44labs**. В этой папке всё, что относится к компьютерному зрению и машинному обучению;
остальной проект (бэкенд, экспорт DXF, прототипы) этот код не дублирует.

Сейчас здесь **drawcv** — CV-модуль на OpenCV по задаче «Подготовить CV модуль на основе OpenCV»:

1. **Подготовка изображения** — PNG, JPEG и страница PDF идут одним путём; поворот (ручной 90/180/270 и
   автоматическое выравнивание наклона до 5°), выравнивание фона/контраста, пересчёт к рабочему разрешению 300 dpi,
   калибровка px → мм листа по рамке. Все преобразования сохраняются матрицами, любой объект возвращается на
   исходную страницу. Исходник не изменяется; надписи стираются только в рабочей копии, маска сохраняется.
2. **Отрезки, окружности, дуги** с типом, координатами (мм листа и пиксели исходника) и оценкой уверенности;
   области текста (детектор RapidOCR или файл OCR) исключаются; штриховка находится и убирается из контура;
   стрелки, шум, дубли и неподдержанные дуги помечаются, а не прячутся.
3. **Контуры и дефекты** — сшивка концов по явному допуску в мм, удаление дублей, замкнутые контуры и открытые
   цепочки, разрывы и самопересечения; дуги остаются дугами; оси и текст к детали не присоединяются; граница
   штриховки — только в простом случае, иначе вопрос.
4. **Интеграция и измерения** — стабильный вызов для бэкенда (`CV_ADAPTER=drawcv.adapter:run_cv`), точки с
   идентификаторами для связи с OCR, таблица качества по каждому листу против эталонного DXF, время этапов.

```
mlcv/
├─ drawcv/                пакет
│  ├─ ingest.py           чтение PNG / JPEG / PDF (pypdfium2) → одна страница в оттенках серого
│  ├─ preprocess.py       поворот, наклон, фон/контраст, разрешение, бинаризация, «пылинки»
│  ├─ calibrate.py        px ↔ мм листа: обрез / рамка ГОСТ по сторонам, PDF, ручная
│  ├─ transform.py        матрицы source_px → work_px → sheet_mm и обратно
│  ├─ textmask.py         области текста (RapidOCR det / файл OCR / компоненты) и их стирание в рабочей копии
│  ├─ vectorize.py        скелет → цепочки → отрезки и дуги
│  ├─ primitives.py       слияние, пунктир/оси, окружности из дуг, толщина, уверенность, дубли
│  ├─ hatch.py            семейства линий штриховки
│  ├─ topology.py         контуры, разрывы, самопересечения
│  ├─ pipeline.py         весь маршрут, роли объектов, время этапов
│  ├─ output.py           сборка cv.json (контракт бэкенда + подробности)
│  ├─ overlay.py          картинки для проверки
│  ├─ adapter.py          вызов из бэкенда
│  └─ synth.py            контрольный лист с точно известной геометрией (PNG/JPEG/PDF/«скан» + DXF + truth.json)
├─ tests/                 pytest: форматы, преобразования, примитивы, контуры, контракт бэкенда, детерминизм
├─ eval/evaluate.py       оценка против эталонных DXF датасета → таблица
├─ eval/REPORT.md         результаты и ограничения
├─ examples/              примеры «до/после» и наложений
├─ CV_JSON.md             формат cv.json
├─ requirements.txt       зависимости модуля (точные версии)
├─ requirements-dev.txt   + тесты и оценка
├─ pyproject.toml         установка пакетом: pip install ./mlcv
└─ Dockerfile             образ с тестами, работает без интернета
```

## Установка

Python 3.11+ (проверено на 3.14). Всё ставится в окружение проекта `.venv`, не глобально:

```bash
py -3.14 -m venv .venv                                     # если окружения ещё нет
.venv/Scripts/python -m pip install -r mlcv/requirements-dev.txt
export PYTHONUTF8=1                                        # кириллица в консоли Windows
```

Linux/macOS — то же с `.venv/bin/python`. Как пакет (например, в образ бэкенда): `pip install ./mlcv`.
Модели детектора текста RapidOCR (≈30 МБ) скачиваются при первом запуске; в Docker-образе — при сборке.

## Запуск

```bash
# один или несколько листов → DIR/<имя>/cv.json + overlay.png, geometry.png, prep_compare.png, params.json
PYTHONPATH=mlcv .venv/Scripts/python -m drawcv dataset/dataset2/Шайба.pdf лист.png скан.jpg --out out_cv

# параметры: --page N (страница PDF), --sheet A3, --rotate 90, --px-per-mm 11.81 [--origin-px x,y] (ручная
# калибровка, помечается в cv.json), --text-provider rapidocr|components|file|none, --ocr-json ocr.json,
# --params params.json (любые поля drawcv/params.py), --no-images
```

Из Python (стабильный вызов для других модулей):

```python
from drawcv import Params, run
cv = run("лист.pdf", out_dir="out/лист", params=Params(snap_tol_mm=0.3))   # dict; то же в out/лист/cv.json
```

В бэкенде (`backend/app/pipeline.py` вызывает `module:function(image) -> dict`):

```bash
pip install ./mlcv
export CV_ADAPTER=drawcv.adapter:run_cv          # картинки проверки пишутся в <run>/cv/
export DRAWCV_PARAMS=/path/params.json           # необязательно
```

Проверено с настоящими `recognize()` и моделями `DrawingGraph` из ветки `feat/fastapi-dxf-backend`: граф строится
без ошибок валидации. Формат — [CV_JSON.md](CV_JSON.md).

## Проверка

```bash
cd mlcv && ../.venv/Scripts/python -m pytest -q && cd ..        # 41 тест, ≈1,5 мин

# оценка против эталонных DXF; выходы в mlcv/eval/out*/ (не в git), таблица table.md
.venv/Scripts/python mlcv/eval/evaluate.py --set quick                  # Шайба ×3 формата + контрольный лист ×6
.venv/Scripts/python mlcv/eval/evaluate.py --set default                # + Сфера1, Опора, Фланец, прокладка, плита
.venv/Scripts/python mlcv/eval/evaluate.py --set random --n 8 --seed 44 # случайные листы, не участвовавшие в отладке
```

### Docker

```bash
docker build -t drawcv mlcv
docker run --rm drawcv                                                   # тесты в чистом окружении
docker run --rm -v "D:/путь/к/листам:/data" drawcv drawcv /data/лист.pdf --out /data/out
docker run --rm -v "D:/pp/ProjectPractice44Labs:/repo" drawcv \
       python /repo/mlcv/eval/evaluate.py --set quick --out /repo/mlcv/eval/out_docker
```

В Git Bash перед `docker run -v` нужен `MSYS_NO_PATHCONV=1`.

## Результаты

Подробно, со всеми листами и неудачными случаями, — [eval/REPORT.md](eval/REPORT.md).

Сводка по реальным листам (каждая строка — набор; случайные наборы не содержат листов, на которых модуль отлаживался):

| набор | входов | масштаб определён | найдено (тело чертежа), ср. / мин. | центр окружности, ср. | радиус, ср. | замкнутые контуры | время ср., с* |
|---|---|---|---|---|---|---|---|
| листы разработки (Шайба, Сфера1, Опора, Фланец, прокладка, плита) | 12 | 12/12 | 90 % / 59 % | 0,013 мм | 0,010 мм | 101/148 | 10 |
| случайные, seed 44 | 12 | 12/12 | 87 % / 67 % | 0,017 мм | 0,006 мм | 118/237 | 12 |
| случайные, seed 7 | 14 | 14/14 | 80 % / 48 % | 0,024 мм | 0,013 мм | 120/219 | 14 |
| случайные, seed 101 | 12 | 12/12 | 79 % / 65 % | 0,017 мм | 0,002 мм | 91/159 | 16 |

\* четыре оценки шли параллельно; одиночный запуск на листе A3/A4 — 3–6 с на CPU.

Контрольный лист (точная истина) в PNG, JPEG, PDF, PDF с печатью 97 % и 99 %×100 %, «скан» с поворотом 1,2°:
простой профиль замкнут (ошибка площади < 1 %), специально разорванный профиль даёт `gap` 1,5–2,0 мм при истинных
1,5 мм, самопересечение найдено, оси не попадают в контуры; PDF с печатью 97 %×100 % честно получает
`scale_status: unknown`. Слабые места — ложные разрывы на перегруженных листах, мелкие дуги без соосной окружности,
очень тонкие линии рендера 600 dpi — разобраны в [eval/REPORT.md](eval/REPORT.md).

Примеры (`examples/`):

| что | файл |
|---|---|
| до / после подготовки, маска надписей | [shaiba_png_prep_compare.jpg](examples/shaiba_png_prep_compare.jpg), [control_scan_prep_compare.jpg](examples/control_scan_prep_compare.jpg) |
| найденные объекты поверх исходника (включая отброшенные шум/дубли) | [shaiba_png_overlay.jpg](examples/shaiba_png_overlay.jpg), [shaiba_pdf_overlay.jpg](examples/shaiba_pdf_overlay.jpg), [opora_png_overlay.jpg](examples/opora_png_overlay.jpg), [control_png_overlay.jpg](examples/control_png_overlay.jpg) |
| только геометрия детали: контуры, разрывы, самопересечения, штриховка | [shaiba_png_geometry.jpg](examples/shaiba_png_geometry.jpg), [opora_png_geometry.jpg](examples/opora_png_geometry.jpg), [control_png_geometry.jpg](examples/control_png_geometry.jpg), [control_scan_geometry.jpg](examples/control_scan_geometry.jpg) |
| сравнение с эталонным DXF (зелёный — эталон) | [shaiba_png_compare.jpg](examples/shaiba_png_compare.jpg) |

Пересобрать примеры: `python mlcv/eval/make_examples.py` после `evaluate.py --set default`.
