# ProjectPractice44Labs — оцифровка чертежей: скан / PDF → редактируемый DXF

Совместный проект с **44labs**. Цель — MVP сервиса, который по скану или PDF чертежа конструкторской документации
(ЕСКД) выдаёт **редактируемый DXF**: геометрия, размеры как настоящие `DIMENSION`, штриховки, текст и штамп на русском,
формат листа и масштаб как в оригинале. Файл должен открываться в AutoCAD, LibreCAD, nanoCAD, КОМПАС и
конвертироваться в DWG.

Постановка задачи: [`task/Task.pdf`](task/Task.pdf), ответы заказчика: [`task/additionalComments.txt`](task/additionalComments.txt).

## Где что лежит

```
ProjectPractice44Labs/
├─ task/                    постановка задачи и ответы заказчика
├─ dataset/                 данные (только чтение)
│  ├─ clear_sum_dataset/    127 реальных пар PNG (600 dpi) ↔ DXF
│  ├─ dataset2/, dataset3/  реальные пары PDF ↔ DXF (72 и 52, пересекаются по деталям)
│  └─ data/                 150 синтетических пар PDF-скан ↔ DXF (с дефектами, см. docs)
├─ runs/                    прогоны на конкретных листах
│  ├─ клапан.png            тестовый лист (А3, 1:1)
│  └─ klapan_png2dxf/       оцифровка клапана: итоговые DXF/DWG, отчёт, код вариантов A и B
├─ mlcv/                    CV/ML-часть: модуль drawcv на OpenCV (PNG/JPEG/PDF → примитивы, контуры, дефекты → cv.json),
│                           тесты, оценка против эталонов, Docker-образ — см. mlcv/README.md
├─ docs/architecture/       архитектура сервиса, ревью, эксперименты на 9 листах и «сканах»
├─ архитектура_кодекс/      исходное предложение по архитектуре сервиса (разобрано в docs/architecture/REVIEW.md)
├─ tools/                   общие инструменты: запись DXF по ЕСКД, сравнение DXF, измерения на растре
└─ docker/                  образы проверки: LibreCAD (открытие/печать DXF), ODA File Converter (DXF ↔ DWG)
```

## Главное за 1 минуту

| что | результат | где |
|---|---|---|
| Клапан, вариант A — параметрическая перерисовка от размеров | совпадение геометрии с эталоном 98 % в пределах 0,25 мм, 28/28 настоящих размеров | [`runs/klapan_png2dxf/REPORT.md`](runs/klapan_png2dxf/REPORT.md) |
| Клапан, вариант B — автоматический конвейер | 92 % в пределах 0,25 мм, 93 % размеров по значению, ≈14 с на лист (CPU); после исправлений ревью — 93 % и в 1,4 раза больше найденной штриховки | там же; [`klapan_recheck`](docs/architecture/experiments/klapan_recheck) |
| B на 9 других реальных листах | геометрия 89–98 % в пределах 0,25 мм; размеры ≈ 70 % по значению | [`docs/architecture/experiments/RESULTS.md`](docs/architecture/experiments/RESULTS.md) |
| B на имитации архивного скана 200 dpi | геометрия 79 % (0,5 мм), размеры ≈ 20 % — нужны обучаемые модели | там же |
| CV-модуль drawcv (OpenCV): PNG/JPEG/PDF → примитивы, контуры, дефекты → `cv.json` | на 50 входах реальных листов (из них 38 — случайные, не участвовавшие в отладке) масштаб определён везде, ошибка центра окружности ≈ 0,02 мм, найдено 79–90 % геометрии; контрольный лист: профиль замкнут, разрыв и самопересечение найдены; подключается к бэкенду через `CV_ADAPTER` | [`mlcv/README.md`](mlcv/README.md), [`mlcv/eval/REPORT.md`](mlcv/eval/REPORT.md) |
| Целевая архитектура сервиса (FastAPI + OpenCV + OCR + ML) | этапы, описание чертежа `DrawingGraph`, API, данные и обучение, план | [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) |
| Ревью: что оставить, убрать, изменить | включая разбор исходного предложения по архитектуре | [`docs/architecture/REVIEW.md`](docs/architecture/REVIEW.md) |

Итоговые файлы клапана: `runs/klapan_png2dxf/out/klapan_variant_A.dxf` (рекомендуемый) и `klapan_variant_B.dxf`,
рядом — `.dwg`.

## Требования к DXF (из ТЗ)

AC1021 (AutoCAD 2007), UTF-8, миллиметры; слои `MAIN / THIN / AXIS / DIMS / TEXT` (скрытые линии — `THIN` + тип `DASHED`);
размеры — настоящие `DIMENSION`; штамп обязателен (наименование, обозначение, материал, ФИО).
Масштаб ≠ 1:1 — лист в мм бумаги и `DIMLFAC = 1/масштаб` в размерном стиле (так устроены архивные DXF заказчика).

## Как запустить

Нужны Python 3.12+ (проверено на 3.14) и, для проверки в CAD, Docker.

```bash
# окружение (всё ставится в .venv проекта, точные версии — в mlcv/requirements*.txt)
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -r mlcv/requirements-dev.txt matplotlib
export PYTHONUTF8=1          # кириллица в консоли Windows

# CV-модуль drawcv: PNG / JPEG / PDF → cv.json (примитивы, контуры, дефекты) + картинки проверки
PYTHONPATH=mlcv .venv/Scripts/python -m drawcv dataset/dataset2/Шайба.pdf --out out_cv
cd mlcv && ../.venv/Scripts/python -m pytest -q && cd ..                  # тесты модуля
.venv/Scripts/python mlcv/eval/evaluate.py --set quick                    # качество против эталонных DXF
docker build -t drawcv mlcv && docker run --rm drawcv                      # то же в чистом Docker-образе

# автоматический конвейер (вариант B): PNG → DXF + отчёт со временем этапов
.venv/Scripts/python runs/klapan_png2dxf/pipeline_b/pipeline.py runs/клапан.png out.dxf --report report.json

# вариант A для клапана (пересобрать DXF из параметров)
.venv/Scripts/python runs/klapan_png2dxf/build_variant_a.py

# проверка DXF (версия, кодировка, единицы, слои, размеры) и сравнение с эталоном
.venv/Scripts/python tools/eskd_dxf.py out.dxf
.venv/Scripts/python tools/compare_dxf.py REF.dxf out.dxf --out compare_dir --align geom

# открытие и печать в LibreCAD, конвертация в DWG (Docker; в Git Bash добавить MSYS_NO_PATHCONV=1)
docker build -t cad-validate docker/cad-validate
docker run --rm -v "D:/путь/к/папке:/work" cad-validate validate.sh /work/file.dxf 420x297
docker build -t oda-convert docker/oda-convert
docker run --rm -v "D:/путь/к/папке:/work" oda-convert convert.sh /work/in /work/dwg ACAD2018 DWG "*.dxf"

# регрессия конвейера на 9 листах датасета и на имитации скана
.venv/Scripts/python docs/architecture/experiments/run_b_batch.py --batch_b_new
.venv/Scripts/python docs/architecture/experiments/recompare.py batch_b_new
.venv/Scripts/python docs/architecture/experiments/robustness.py robustness_new
```

## Правила работы с данными

- `dataset/` только читаем; результаты — в `runs/<имя>/` (`work/` — промежуточное, `out/` — итог, `compare/` — сравнение,
  `REPORT.md` — отчёт).
- Эталоны тестового листа (`dataset/clear_sum_dataset/vse_dxf/клапан.dxf`, `dataset/dataset2/клапан.*`) используются только
  скриптом сравнения, не для разработки. 10 приёмочных листов заказчика в разработке не используются.
- Разбиение данных на обучение и проверку — по детали, а не по файлу (наборы пересекаются).

# Сервер API

FastAPI, PostgreSQL, worker и экспорт DrawingGraph → DXF находятся в [backend/README.md](backend/README.md). Порядок первого запуска и применения миграции описан там же.

## Офлайн-прототип и основной backend

В репозитории сохранены два отдельных варианта запуска:

| Вариант | Compose | API | Хранение |
|---|---|---|---|
| Основной backend | `compose.yaml` | `/api/v1/jobs`, PDF/PNG/JPEG, версии и DrawingGraph | Собственные Docker volumes и миграции Alembic |
| Офлайн-прототип из `feature/offline-deployment` | `compose.offline-prototype.yaml` | `/jobs`, PNG | Проект Compose `practice44`, его PostgreSQL volume, каталоги `storage/` и `models/` |

`docker compose ...` использует основной backend. Скрипты `scripts/prepare_online.sh`,
`scripts/start_offline.sh` и `setup-mac.sh` явно выбирают офлайн-прототип. Для его ручного
запуска используйте `docker compose -f compose.offline-prototype.yaml ...`.
Не подключайте эти сервисы к одной базе: у них различаются схемы и контракты API.
При одновременном запуске задайте прототипу другой `API_PORT` в `.env`, например `8001`.
Основной backend пока не получает гарантию автономности от наличия второго Compose-файла.

В общем OCR-модуле `OCR_MODEL_PATH` без `MODEL_DIR` сохраняет существующий режим backend.
Офлайн-прототип задаёт `MODEL_DIR`: все модели проверяются локально, а отсутствие или
повреждение файла вызывает ошибку без перехода на сетевую загрузку. При указании обеих
переменных применяется строгий офлайн-режим. Без обеих переменных прототип использует `./models`.
Отчёт исходного офлайн-эксперимента: [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md).
