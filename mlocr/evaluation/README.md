# Оценка OCR

| файл | что |
|---|---|
| `collect_observations.py` | сохраняет настоящие чтения PaddleOCR по листам датасета; запускается там, где есть PaddleOCR, то есть в образе сервиса |
| `score_on_references.py` | сравнивает чтения с эталонными DXF, а штамп — ещё и с именами файлов; правила нормализации применяются заново, модели не нужны |
| `references_score.json` | последний результат |

Собрать чтения (один раз, ≈ 15 с на лист на CPU; выход в `mlocr/evaluation/out/`, в git не попадает):

```bash
docker run --rm --network none -e PYTHONUTF8=1 -e MODEL_DIR=/models -e DRAWING_OCR_MODEL_DIR=/models/paddle \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True -v "$PWD:/repo" -v "$PWD/models:/models:ro" practice44-backend:0.2.0 \
  python /repo/mlocr/evaluation/collect_observations.py --png /repo/dataset/clear_sum_dataset/vse_png \
  --out /repo/mlocr/evaluation/out/raw "Шайба" "переходник"
```

В Git Bash перед командой нужен `MSYS_NO_PATHCONV=1`. Оценить и проверить всю цепочку с настоящими чтениями (нужны выходы `mlcv/eval/evaluate.py`):

```bash
.venv/Scripts/python mlocr/evaluation/score_on_references.py --show 30
OCR_RAW=mlocr/evaluation/out/raw .venv/Scripts/python mlcv/eval/ocr_link_check.py mlcv/eval/out
```

Эталоны тестового листа клапан сюда не передавать.

## Правила оценки после исправлений 11.11.2026

Сопоставление выполняется по расположению **один к одному**, без отбора по похожести на правильную строку. В `score_on_references.py` знаменатель `exact_share` — все эталонные тексты выбранных листов, включая пропуски и односимвольные обозначения. Лишние чтения считаются отдельно; CER включает ошибки, удаления и лишние символы. Оценка по DXF предполагает, что его тексты действительно являются полным эталоном выбранной области; имена файлов — вспомогательная подсказка, не независимая разметка штампа.

`dim_link_cases.py` и `mlcv/eval/ocr_link_check.py` сравнивают обе координаты обоих измеряемых концов с допуском 0,5 мм после регистрации. Дубликат не добавляет второго успеха. Проверка чтения номинала отделена от правильности геометрической связи.

```powershell
python -m mlocr.evaluation.dim_link_cases path/to/case --ocr path/to/observations.json --out path/to/new-report.json
```

Режим `ideal` использует эталонные подписи и единственный эталонный масштаб; он явно помечен `reference_assisted=true` и не является результатом рабочего конвейера. Несколько эталонных масштабов в этом режиме отклоняются. `real` получает масштаб только из исходного CV/рабочих правил OCR. Номер страницы и хеш OCR сохраняются; несовпадение с CV — ошибка, а не повод заменить идентификатор. В отчёте есть источник масштаба, неверные/лишние и пропущенные связи. Отсутствие выбранных входов не считается успешной оценкой.

## QA-таблица без изменения исходных данных

```powershell
python -m mlocr.evaluation.score_qa_task1 --worksheet mlocr/qa_samples/qa_review.csv --results path/to/raw-runs --images mlocr/qa_samples --out path/to/new-score-run
```

Входные OCR JSON, изображения и таблица не изменяются. Новый каталог содержит нормализованные копии, сравнение JSON/CSV и метрики с хешами входов, параметрами и версией модуля. Уже существующий каталог не перезаписывается. Для запуска нужны настоящие сохранённые `*.ocr.json`, относящиеся к тем же пикселям изображений.

По умолчанию `scope=reviewed_regions_only`: точность и CER относятся только к размеченным строкам, а предсказания вне таблицы показываются как `unassigned_prediction_regions`. Называть их ложными обнаружениями при неполной разметке нельзя. Только после подтверждения QA, что **весь текст каждой страницы** размечен, можно передать `--complete-pages`: тогда лишние области и символы включаются в оценку. Пустой эталон исключается из знаменателя с явным числом размеченных строк. Это не создаёт независимую разметку автоматически.

Обновление только правил нормализации также пишет отдельную таблицу:

```powershell
python -m mlocr.evaluation.refresh_qa_normalization --results path/to/raw-runs --worksheet mlocr/qa_samples/qa_review.csv --output path/to/new-worksheet.csv
```

Исторические `references_score.json` и цифры README набора не пересчитаны автоматически и не отражают новую методику. Сначала нужен новый запуск на зафиксированных входах; старые и новые проценты напрямую не сравниваются.
