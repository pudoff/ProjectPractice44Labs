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
