# Условия и результаты проверки 31.10.2026

Проверяемая ревизия: `0b640d8`, локальный `main`. Код приложения при проверке не изменялся.

## Окружение

Windows, Python 3.13.7. Это диагностическое окружение, а не закреплённый Linux/Python 3.12 проекта.

Использованы доступные локально numpy 1.26.4, OpenCV 4.13.0.92, scipy 1.15.3, shapely 2.1.2, pydantic 2.12.4, pytest 8.4.2. Дополнительные зависимости установлены в отдельный каталог `.local-backups/git-20261031-68e3a399/test-deps`, без изменения файлов зависимостей проекта: ezdxf 1.4.5, python-multipart 0.0.20, pydantic-settings 2.16.2, PyMuPDF 1.28.2, aiosqlite 0.22.1, psycopg/psycopg-binary 3.3.6, pypdfium2 5.14.0. RapidOCR в этом окружении отсутствует; CV использует предусмотренный резервный способ выделения текста `components`.

Проверка не доказывает совместимость всех закреплённых зависимостей между собой. Для этого нужен отдельный прогон в целевом контейнере.

## Штатные наборы тестов

| Команда и рабочий каталог | Результат |
|---|---|
| `python -m pytest -q tests`, корень проекта | **18 passed**, 5.40 с |
| `python -m pytest -q mlcv/tests`, корень проекта | **37 passed, 4 failed**, 240.39 с |
| `python -m pytest -q tests --maxfail=1`, каталог `backend` | Остановлен при подготовке первого теста: `ModuleNotFoundError: No module named 'winloop'` |

Падения CV:

1. `test_formats_agree_with_each_other`: число окружностей для сравниваемых входов различается — 3 против 2; дополнительная окружность имеет радиус около 0.3512 мм.
2. `test_axes_and_text_not_joined[control.png]`: не найден объект с `role == "axis"`.
3. `test_axes_and_text_not_joined[control.jpg]`: то же.
4. `test_axes_and_text_not_joined[control.pdf]`: то же.

Причину этих четырёх падений нельзя однозначно отнести к алгоритму: отличаются версии численных библиотек и способ выделения текста. Нужны повторный прогон с зафиксированными зависимостями и отдельный тест режима `components`.

Backend-тесты задают `use_uvloop=True` и абсолютные пути `/tmp/...`. На этой Windows-машине AnyIO пытается загрузить `winloop`. Это препятствие запуска тестов в окружении ревью, не доказательство ошибки Linux-сервиса. Тесты не переписывались для получения успешного результата.

## Дополнительные проверки

- `reproduce_findings.py` вызывает реальные классы backend и CV на маленьких диагностических входах. Результаты — в `reproductions.json`; созданные тестовые DXF — в `.local-backups/review-20261031/`. Это диагностика найденных проблем, не замена штатному набору тестов.
- `audit_dataset.py` прочитал 510 DXF обновлённого набора, посчитал сущности, единицы, пары по имени и SHA-256-дубликаты. Результат — `dataset-audit.json`.
- `git check-attr filter diff merge -- dataset/clear_sum_dataset/vse_dxf/test.dxf dataset/clear_sum_dataset/vse_png/test.png` вернул `unspecified` для всех атрибутов: правила LFS не применяются к этим путям.
- Полная сборка и запуск текущего стека с PostgreSQL, проверка без интернета и работа в GUI CAD в рамках этого ревью **не выполнялись**.

## Повторение диагностики

Из корня проекта, в Python-окружении с зависимостями backend и CV:

```powershell
python docs/reviews/2026-10-31/reproduce_findings.py
python docs/reviews/2026-10-31/audit_dataset.py
```

В окружении этой проверки дополнительно использовалось:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = (Join-Path (Get-Location).Path '.local-backups/git-20261031-68e3a399/test-deps')
```

Воспроизведение после исправлений ожидаемо изменит `reproductions.json`; текущий файл фиксирует поведение ревизии `0b640d8`.
