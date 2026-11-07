# ProjectPractice44Labs — оцифровка чертежей: скан / PDF → редактируемый DXF

Совместный проект с **44labs**. Сервис получает скан, PNG/JPEG или PDF чертежа ЕСКД и выдаёт **редактируемый DXF**: геометрия,
штриховки, текст и штамп на русском, размеры как настоящие `DIMENSION`, формат листа и масштаб как в оригинале.

Постановка задачи: [`task/Task.pdf`](task/Task.pdf), ответы заказчика: [`task/additionalComments.txt`](task/additionalComments.txt).

## Как устроен сервис

```
 PNG / JPEG / PDF
        │  POST /api/v1/jobs
        ▼
   ┌─────────┐   очередь в PostgreSQL   ┌────────┐
   │   API   │ ───────────────────────► │ worker │
   └─────────┘                          └───┬────┘
                                            │ готовит PNG страницы (лимит размера)
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                    mlocr — PaddleOCR               mlcv/drawcv — OpenCV
                    текст, штамп, размеры ─области─► геометрия, контуры, точки
                             └──────────┬──────────────────┘
                                        ▼ связывание размеров с точками
                                   DrawingGraph ──► DXF, превью, отчёт
```

| папка | что | документация |
|---|---|---|
| `backend/` | API, worker, база, экспорт DXF | [backend/README.md](backend/README.md) |
| `mlcv/` | CV: модуль `drawcv` → `cv.json` | [mlcv/README.md](mlcv/README.md), [формат](mlcv/CV_JSON.md), [качество](mlcv/eval/REPORT.md) |
| `mlocr/` | OCR: текст, штамп, обозначения | [mlocr/README.md](mlocr/README.md) |
| `tools/`, `docker/` | запись и сравнение DXF, проверка в LibreCAD, конвертация в DWG | |
| `dataset/` | данные, только чтение | |
| `runs/`, `docs/` | прогоны, архитектура, спецификация, ревью | [docs/architecture](docs/architecture/README.md) |
| `service/`, `compose.offline-prototype.yaml` | исторический офлайн-прототип, отдельно от основного сервиса | [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) |

## Запустить всё вместе

Нужен только Docker. Команды проверены на Windows (Git Bash) и подходят для Linux/macOS.

```bash
# 1. Настройки: задать пароль базы в .env
cp .env.example .env

# 2. Собрать образ сервиса (несколько минут, PaddlePaddle большой)
docker compose build api

# 3. Один раз с интернетом: модели в ./models, проверка хешей. При сбое загрузки просто повторить
docker compose run --rm --no-deps model-assets
docker compose run --rm --no-deps model-assets python -m app.infra.model_assets

# 4. Запуск: база, миграции, API, worker. Дальше интернет не нужен
docker compose up -d --no-build
```

Отправить чертёж и забрать результат:

```bash
curl -F "file=@dataset/dataset2/Шайба.pdf;filename=shaiba.pdf" http://localhost:8000/api/v1/jobs   # ответ содержит "id"
curl http://localhost:8000/api/v1/jobs/<id>                                                        # ждать state: succeeded или needs_review
curl -o result.dxf http://localhost:8000/api/v1/jobs/<id>/artifacts/result.dxf
```

Ещё артефакты: `preview.png`, `report.json` (все вопросы к листу), `cv.json`, `ocr.json`, `drawing.json`.
`needs_review` — нормальный итог: DXF записан, но есть места для проверки человеком.

Остановить: `docker compose down` (данные сохраняются в томах), `docker compose down -v` — вместе с данными.
Другой порт: `API_PORT=18000` в `.env`. Полностью автономный комплект: `bash scripts/prepare_online.sh`, затем `bash scripts/start_offline.sh`.

## Проверить, что всё работает

```bash
# Все тесты бэкенда, OCR и прототипа в контейнере, без сети и базы
docker compose -f compose.test.yaml run --build --rm checks

# Тесты CV-модуля в чистом образе
docker build -t drawcv mlcv && docker run --rm drawcv

# Открывается ли DXF в настоящем CAD (LibreCAD, печать в PDF); в Git Bash перед командой MSYS_NO_PATHCONV=1
docker build -t cad-validate docker/cad-validate
docker run --rm -v "D:/путь/к/папке:/work" cad-validate validate.sh /work/result.dxf 420x297
```

В этой проверке текст выглядит ромбиками: headless LibreCAD не загружает шрифты, это не ошибка файла. Кириллицу смотрите в `preview.png`.

Без Docker, для разработки CV и оценки качества, — окружение `.venv` (Python 3.12+, проверено на 3.14):

```bash
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -r mlcv/requirements-integration.txt matplotlib
.venv/Scripts/python -m pip uninstall -y opencv-python-headless opencv-contrib-python   # один пакет cv2, как в бэкенде
.venv/Scripts/python -m pip install --no-deps opencv-contrib-python==4.10.0.84
export PYTHONUTF8=1                                                                      # кириллица в консоли Windows

cd mlcv && ../.venv/Scripts/python -m pytest -q && cd ..                                 # тесты CV, включая настоящий бэкенд
PYTHONPATH=mlcv .venv/Scripts/python -m drawcv лист.pdf --out out_cv                     # только CV: лист → cv.json и картинки
.venv/Scripts/python mlcv/eval/evaluate.py --set quick                                   # качество против эталонных DXF
.venv/Scripts/python mlcv/eval/three_sheets.py                                           # три независимых листа, причины ошибок
.venv/Scripts/python tools/compare_dxf.py REF.dxf out.dxf --out compare_dir --align geom # сравнение DXF с эталоном
```

Варианты на клапане и регрессия прежнего конвейера — [runs/README.md](runs/README.md), [docs/architecture/README.md](docs/architecture/README.md).

## Где мы сейчас

| что | результат | где |
|---|---|---|
| Весь сервис на реальных листах | PDF и PNG проходят от загрузки до DXF; файл открывается в LibreCAD. A3 при 600 dpi — 36 с на CPU | [mlcv/eval/REPORT.md](mlcv/eval/REPORT.md) |
| CV: геометрия | масштаб определён на всех реальных листах с ошибкой меньше 0,2 %; центр окружности ≈ 0,02 мм | там же |
| CV: контуры | «замкнут» теперь значит «рёбра встречаются по координатам»: бэкенд принимает 100 % контуров (было 36 %) | там же |
| Размеры `DIMENSION` | связываются автоматически: 143 из 168 при идеальном чтении, 94 при настоящем OCR, ошибок почти нет; показывают измеренное значение с `%%c`, допусками и `*` | там же |
| OCR и штамп | точных чтений 74 % (было 62 %); наименование найдено на 17 листах из 23 (было 3), обозначение на 16 из 20 (было 11), ФИО заполнено на 19 листах из 33 (было 4) | [mlocr/README.md](mlocr/README.md) |
| CV на трёх независимых листах | разбор пропусков и лишних по видам и причинам, наложения на исходник | [mlcv/eval/three_sheets/REPORT.md](mlcv/eval/three_sheets/REPORT.md) |
| Клапан, ручная перерисовка (вариант A) | 98 % геометрии в пределах 0,25 мм, 28/28 настоящих размеров | [runs/klapan_png2dxf/REPORT.md](runs/klapan_png2dxf/REPORT.md) |
| Архитектура и ревью | этапы, `DrawingGraph`, API, план | [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md), [REVIEW.md](docs/architecture/REVIEW.md) |

**Главные ограничения сервиса сейчас:**

- Около половины размеров при настоящем OCR ещё остаются текстом: радиусы, размеры по выноскам и прочитанные неверно числа. Модель бэкенда пока знает только линейный `DIMENSION`.
- На больших сборочных листах ещё много разрывов; контуры совпадают с эталоном в 54 % случаев: часть линий получает неверную роль из-за толщины.
- OCR всё ещё ошибается в редких символах, а значения штампа нужно проверять: это кандидаты.
- В AutoCAD результат не проверен: AutoCAD здесь нет. Стиль текста — `isocpeur.ttf` с кириллицей, он входит в поставку AutoCAD.
- Самый большой лист (A1 при 600 dpi, 5650 примитивов) проходит весь сервис за 17 минут на CPU при ориентире заказчика 5–10 (память worker ≈ 2,7 ГБ). Обычный лист A3 — меньше минуты.

## Требования к DXF (из ТЗ)

AC1021 (AutoCAD 2007), UTF-8, `$DWGCODEPAGE=ANSI_1251`, миллиметры; слои `MAIN / THIN / AXIS / DIMS / TEXT` (скрытые линии — `THIN` + тип `DASHED`);
размеры — настоящие `DIMENSION`; штамп обязателен (наименование, обозначение, материал, ФИО).
Масштаб ≠ 1:1 — лист в мм бумаги и `DIMLFAC = 1/масштаб` в размерном стиле (так устроены архивные DXF заказчика).

## Лимиты

| настройка | по умолчанию | зачем |
|---|---|---|
| `MAX_IMAGE_PIXELS` | 300 млн пикселей | самый большой лист датасета, A1 при 600 dpi, — 280 млн; проверяется по заголовку файла до декодирования (бэкенд, OCR; у `drawcv` — `DRAWCV_MAX_IMAGE_PIXELS`) |
| `MAX_UPLOAD_BYTES` | 25 МБ | самый большой файл датасета — менее 10 МБ |

## Правила работы с данными

- `dataset/` только читаем; результаты — в `runs/<имя>/`, выходы оценки CV — в `mlcv/eval/out*` (в git не попадают).
- Эталоны тестового листа (`dataset/clear_sum_dataset/vse_dxf/клапан.dxf`, `dataset/dataset2/клапан.*`) используются только скриптом сравнения. 10 приёмочных листов заказчика в разработке не используются.
- Разбиение на обучение и проверку — по детали, а не по файлу: наборы пересекаются.

## Офлайн-прототип

В репозитории сохранён отдельный исторический вариант: `compose.offline-prototype.yaml`, API `/jobs`, только PNG, свои зависимости в корневом `pyproject.toml`.
Не подключайте его к базе основного сервиса: схемы и контракты различаются. При одновременном запуске задайте ему другой `API_PORT`.
Отчёт эксперимента: [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md).
