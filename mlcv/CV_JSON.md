# Формат `cv.json` (drawcv-1, `schema_version` 0.1)

Выход CV-модуля для одного листа. Файл пишет `drawcv.run()`; бэкенд сохраняет его как `<run>/cv.json`.
Совместная работа с 44labs.

## Что читает бэкенд (строгий контракт)

Бэкенд (`backend/app/pipeline.py`, ветка `feat/fastapi-dxf-backend`) собирает `DrawingGraph` из пяти полей и проверяет
их моделями pydantic с `extra="forbid"`, поэтому в этих списках **нет ни одного лишнего поля**:

| поле | значение |
|---|---|
| `schema_version` | `"0.1"` |
| `status` | `"ok"` или `"error"` (тогда есть `error` — текст ошибки, списки пустые) |
| `coordinate_space` | `"sheet_mm"` — миллиметры листа, начало в левом нижнем углу линии обреза, ось y вверх |
| `scale_status` | `"known"` — калибровка проверена (рамка/обрез с согласием масштабов x и y ≤ 0,3 %, ручная); `"unknown"` — нет |
| `primitives[]` | `{id, kind: line|circle|arc, layer: MAIN|THIN|AXIS, start, end | center, radius, start_angle, end_angle}` — только принятые объекты, дуги против часовой стрелки, углы в градусах |
| `hatches[]` | `{id, outer[[x,y]…], holes[[[x,y]…]…], pattern: "ANSI31", scale, angle}` — только штриховки с найденной границей |
| `issues[]` | `{code, severity: warning|error, message, object_id}` |

Невидимые (штриховые) линии идут в `layer: THIN`; их тип линии `DASHED` записан в `objects[id].linetype`.

## Подробности для проверки, метрик и OCR

| поле | что внутри |
|---|---|
| `source` | имя, формат, sha256 исходника, страница PDF, размер в px, dpi (если ему можно верить), размер страницы PDF в мм, `source_px_to_pdf_pt` |
| `calibration` | `method`: `sheet_trim` / `inner_frame` / `mixed_trim_frame` / `manual` / `pdf_page` / `image_size`; `manual`; `sheet`, `sheet_mm`, `px_per_mm` [x, y], `xy_agreement`, `side_roles`, `pdf_page_check.content_scale_vs_page` (печать PDF «по размеру страницы»); масштаб вида (1:2…) модуль **не** определяет — это поле штампа для OCR |
| `transforms` | матрицы 3×3: `source_px_to_work_px`, `work_px_to_sheet_mm`, `source_px_to_sheet_mm`, `sheet_mm_to_source_px` |
| `preprocess` | угол выравнивания, коэффициент пересчёта разрешения, число удалённых точек-«пылинок» |
| `params` | все параметры запуска (то же лежит в `params.json` рядом) |
| `stats` | толщины тонкой/основной линии и порог между ними, допуски в мм (`tolerances_mm.snap` = `snap_tol_mm + snap_stroke_factor × толщина основной`), провайдер областей текста, стёртые пиксели текста и штриховки |
| `summary` | счётчики по `вид:роль:статус`, замкнутые/открытые контуры, дефекты по кодам |
| `timings_s` | время этапов и `total` |
| `objects{id}` | **все** найденные объекты, включая отброшенные: `kind`, `role`, `status`, `confidence`, `support`, `rms_mm`, `stroke_mm`, `length_mm`, `layer`, `geometry_mm`, `geometry_source_px`, `reasons[]`, `duplicate_of`, `concentric_with`, `extended_mm`, `dash_count`, `linetype` |
| `points[]` | измеряемые точки `{id: pt_N, xy, kind: end|junction|center|mid|quadrant, refs[{object_id, anchor}]}` |
| `contours[]` | `{id, role: outer_boundary|region|circle|open_chain, closed, edges[], object_ids, area_mm2, length_mm, closure_error_mm, faces_inside, bbox_mm, parent_id}`; ребро — `{object_id, kind, reversed, start, end}` (+ `center, radius, start_angle, end_angle` у дуги) |
| `defects[]` | `{id, code, severity, at_mm, object_ids, message, distance_mm?, to_mm?, nearest_non_part?}` |
| `hatch_groups[]` | семейства линий штриховки: `line_ids`, `angle_deg`, `spacing_mm`, `regularity`, `boundary: resolved|unresolved`, `contour_id`, `hatch_id`, `hole_contour_ids` |
| `text_regions[]` | области текста: `poly_source_px`, `poly_mm`, провайдер, сколько компонент стёрто/оставлено (оставлены пересекающие надпись линии); при `ocr_json` — `ocr_id`, `text` |

### Роли и статусы объектов

`role`: `part` (контур детали, основная линия), `thin`, `axis`, `hidden`, `hatch`, `frame`, `title_block`, `arrowhead`,
`text_residue`, `junction_artifact`, `duplicate`, `noise`, `low_support`.
`status`: `ok` (экспортируется), `excluded` (отброшен с причиной в `reasons`), `unsupported` (дуга вне поддержанного
случая: радиус < 1 мм или дуга < 20°), `converted` (линия штриховки, ставшая `HATCH`).

`confidence` = доля точек объекта на чернилах × качество подгонки × поправка на длину (0…1).

### Коды дефектов

| код | значение |
|---|---|
| `gap` | висящий конец детали ближе `gap_search_mm` к другой геометрии детали, но дальше допуска сшивки: **не замыкается** |
| `open_end` | висящий конец без геометрии детали рядом (`severity: info`) |
| `self_intersection` | рёбра детали пересекаются не в концах, или две линии проходят насквозь через один узел |
| `duplicate_edge` | одно и то же ребро найдено дважды |
| `hatch_boundary_unresolved` | у семейства штриховки нет простой замкнутой границы — вопрос, линии остаются линиями |
| `end_extended`, `corner_restored` | конец продлён / угол восстановлен по собственной кривой **только по толстым чернилам** (`info`) |

## Как OCR ссылается на объекты

Размер связывается с измеряемыми точками либо по `points[].id` (`"pt_17"`), либо по паре
`{"object_id": "line_12", "anchor": "start"}` (якоря: `start`, `end`, `mid` у отрезка; `center`, `start`, `end` у дуги;
`center`, `quadrant_0/90/180/270` у окружности). Идентификаторы детерминированы: одинаковый вход и параметры дают
тот же `cv.json` (кроме `timings_s`) — проверяется тестом `test_deterministic`.

Чтобы модуль не путал надписи с геометрией, OCR может передать свои области текста:
`Params(ocr_json="ocr.json")` — список `{id, text, poly_source_px | box_px}` в пикселях исходника.
