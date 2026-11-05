# Формат `cv.json` (`drawcv-1`, `schema_version` 0.1)

Выход CV-модуля для одного листа. Пишет `drawcv.run()`, бэкенд сохраняет как `<run>/cv.json`. Совместная работа с 44labs.
Все координаты — **миллиметры листа**: начало в левом нижнем углу линии обреза, ось y вверх.

## Строгая часть (её проверяет бэкенд)

Бэкенд (`backend/app/services/recognition.py`, модели `backend/app/domain/drawing.py`, `extra="forbid"`) не допускает лишних
полей в этих списках. Проверка настоящим кодом бэкенда — `tests/test_backend_real.py`.

| поле | значение |
|---|---|
| `schema_version`, `status` | `"0.1"`; `"ok"` или `"error"` (тогда есть `error`, списки пустые) |
| `coordinate_space`, `scale_status` | `"sheet_mm"`; `"known"` (рамка/обрез с согласием x и y, ручная) или `"unknown"` |
| `raster_identity` | `{page, width_px, height_px, raster_sha256}` — страница, которую должен читать OCR (`mlocr` сверяет) |
| `primitives[]` | `{id, kind: line\|circle\|arc, layer: MAIN\|THIN\|AXIS, start, end \| center, radius, start_angle, end_angle}` — принятые объекты; дуги против часовой, градусы |
| `points[]` | `{id: pt_N, xy, kind, refs[{object_id, anchor}]}` — опорные точки |
| `contours[]` | `{id, role, closed, edges[], object_ids, area_mm2, length_mm, closure_error_mm, faces_inside, bbox_mm, parent_id}` |
| `hatches[]` | `{id, outer[], holes[], pattern, scale, angle}`; связь с контуром — в `hatch_groups` |
| `issues[]` | `{code, severity: warning\|error, message, object_id}` |

Координаты геометрии записаны с 6 знаками, чтобы рёбра, встретившиеся в модуле, встретились и у бэкенда (его допуск 0,0001 мм).

## Что значит «замкнут»

`closed: true` — соседние рёбра **действительно встречаются по координатам** (`closure_error_mm` ≤ 0,00005 мм) и кольцо
простое. Это не «концы рядом»: концы в пределах допуска либо сведены в одну точку, либо контур остаётся открытым.

| `role` | значение |
|---|---|
| `outer_boundary`, `region`, `circle` | замкнутый контур: внешняя граница компонента, грань внутри него, окружность |
| `open_chain` | открытая цепочка |
| `not_simple` | рёбра образуют кольцо, которое касается себя или без площади; `closed: false`, дефект `contour_not_simple` |

Ребро контура: `{object_id, kind, reversed, start, end}` (+ `center, radius, start_angle, end_angle` у дуги); дуга остаётся дугой.

## Как изменены координаты

Модуль сдвигает только концы объектов детали и всегда записывает, что сделал.

- `objects[id].geometry_mm` — итоговая геометрия (равна `primitives[]`); **`observed_geometry_mm`** — наблюдение до правок,
  есть только у изменённых объектов; **`adjustments[]`** — `{end, from_mm, to_mm, moved_mm, along_mm, across_mm, reason, closure_id}`.
- `closures[]` — каждый стык: `{id, status: joined|refused, at_mm, tolerance_mm, reason, members[], hosts[], refused[], max_moved_mm, residual_mm}`.
  `hosts` — объекты, на внутренней части которых лежит точка (они не двигаются).
- `stats.closure` — итоги и проверки: `edges_meet`, `max_residual_mm`, `joined`, `refused`, `undone`,
  `closed_by_tolerance_only` (сколько было бы «замкнуто» одним допуском), `closed_by_tolerance_with_real_gap`, `closed_in_coordinates`.

| причина правки (`reason`) | что сделано |
|---|---|
| `corner_intersection` | отрезки продлены/подрезаны вдоль себя до пересечения |
| `t_junction` | конец поставлен на другой объект, тот не двигается |
| `collinear_joint` | почти соосные концы сведены; короткий отрезок сдвигается сильнее длинного |
| `line_arc_intersection`, `arc_intersection` | точка на пересечении; центр и радиус дуги не меняются |
| `tangent_joint` | касание без пересечения: конец отрезка перенесён на окружность дуги |
| `arc_refit` | последняя мера: радиус сохранён, центр чуть смещён |

| причина отказа (контур остаётся открытым) | когда |
|---|---|
| `no_ink` | под перемещением белая бумага — реальная щель |
| `too_far` | нужно больше 1,25 допуска |
| `would_turn_line`, `would_leave_ink` | отрезок заметно повернётся или сойдёт со своих чернил |
| `would_collapse` | объект стал бы короче половины длины |
| `would_cross` | перемещение пересекает другой объект |
| `parallel_overlap` | линии идут рядом в одном направлении: это щель, а не угол |
| `arc_refit_too_large` | дугу пришлось бы заметно изменить |
| `would_close_contour` | перемещения уменьшают площадь контура больше чем на 20 % — стык отменён |

Допуск стыка: `stats.tolerances_mm.snap` = `snap_tol_mm + snap_stroke_factor × толщина основной линии`.

## Опорные точки для OCR

Размер ссылается на `points[].id` (`"pt_17"`) или на пару `{"object_id": "line_12", "anchor": "start"}`.
Якоря: `start`, `end`, `mid` у отрезка; `center`, `start`, `end` у дуги; `center`, `quadrant_0/90/180/270` у окружности.

**Одна точка — только совпадающие координаты** (≤ 0,00005 мм): объекты, которые реально разошлись, получают разные точки.

`point_info{pt_id}` объясняет каждую точку:

```json
{"source_px": [1520.4, 980.1],
 "meaning": "вершина контура contour_3: конец отрезка line_5; начало дуги arc_1",
 "links": [{"object_id": "line_5", "anchor": "end", "kind": "line", "role": "part", "meaning": "конец отрезка line_5",
            "observed_mm": [120.31, 44.9], "moved_mm": 0.08, "move_reason": "line_arc_intersection"}],
 "contour_ids": ["contour_3"], "lies_on": [], "closure_id": "closure_12"}
```

`meaning` говорит прямо: «начало отрезка», «центр отверстия» (окружность внутри контура), «вершина контура», «лежит на line_7».

`measurable[]` — кандидаты измеряемых участков в мм листа (масштаб вида применяет тот, кто читает число):
`line_length {p1, p2, value_mm, direction}`, `circle_diameter {center, value_mm}`, `arc_radius {center, p1, p2, value_mm}`,
`parallel_distance {object_ids, p1, p2, value_mm, offset_along_mm}`,
`extension_foot {object_ids, p1, xy_mm, on_object_id}` — основание выносной линии на детали.

**Устойчивость.** Одинаковый вход и параметры дают тот же `cv.json`, кроме `timings_s` (`test_deterministic`). Поворот файла на
90° при `rotate_deg` сохраняет `id` и координаты. Обрезка сохраняет координаты в мм листа (до 0,3 мм), а в пикселях сдвигает их ровно на
обрезку; **номера `id` при этом могут сдвинуться**, поэтому `id` действует в пределах одного растра (это проверяет `raster_identity`),
а между разными обрезками сопоставлять надо по `sheet_mm`. Точка в пикселях исходника — `point_info[...].source_px`
или `transforms.sheet_mm_to_source_px`.

Области текста от OCR: `Params(ocr_json="ocr.json")`, список `{id, text, poly_source_px | box_px}` в пикселях исходника.

## Остальные поля

| поле | что внутри |
|---|---|
| `source` | имя, формат, sha256, страница PDF, размер в px, dpi (если ему можно верить), размер страницы PDF |
| `calibration` | `method`: `sheet_trim` / `inner_frame` / `mixed_trim_frame` / `manual` / `pdf_page` / `image_size`; `px_per_mm [x, y]`, `xy_agreement`, `side_roles`, `pdf_page_check`. Масштаб вида (1:2) модуль не определяет — это поле штампа для OCR |
| `transforms` | матрицы 3×3: `source_px_to_work_px`, `work_px_to_sheet_mm`, `source_px_to_sheet_mm`, `sheet_mm_to_source_px` |
| `preprocess` | угол выравнивания, коэффициент пересчёта, удалённые «пылинки» |
| `params`, `timings_s` | все параметры запуска; время этапов и `total` |
| `stats` | толщины линий, допуски, провайдер областей текста, **`models`** (имя и sha256 модели детектора текста), стёртые пиксели |
| `summary` | счётчики объектов, контуров, дефектов, `closure` |
| `objects{id}` | **все** найденные объекты, включая отброшенные: `kind`, `role`, `status`, `confidence`, `support`, `rms_mm`, `stroke_mm`, `length_mm`, `layer`, `geometry_mm`, `geometry_source_px`, `reasons[]`, `duplicate_of`, `concentric_with`, `extended_mm`, `dash_count`, `linetype` |
| `defects[]` | `{id, code, severity, at_mm, object_ids, message, distance_mm?, to_mm?, nearest_non_part?}` |
| `hatch_groups[]` | семейства линий штриховки: `line_ids`, `angle_deg`, `spacing_mm`, `boundary: resolved|unresolved`, `contour_id`, `hatch_id`, `hole_contour_ids` |
| `text_regions[]` | области текста: `poly_source_px`, `poly_mm`, провайдер; при `ocr_json` — `ocr_id`, `text` |

`role` объекта: `part`, `thin`, `axis`, `hidden`, `hatch`, `frame`, `title_block`, `arrowhead`, `text_residue`, `junction_artifact`,
`duplicate`, `noise`, `low_support`. `status`: `ok`, `excluded` (с причиной), `unsupported` (радиус < 1 мм или дуга < 20°),
`converted` (линия штриховки, ставшая `HATCH`). `confidence` = доля точек на чернилах × качество подгонки × поправка на длину.

| код дефекта | значение |
|---|---|
| `gap` | висящий конец ближе `gap_search_mm` к другой геометрии детали: **не замкнут** |
| `open_end` | висящий конец без геометрии рядом (`info`) |
| `self_intersection` | рёбра детали пересекаются не в концах |
| `duplicate_edge` | одно ребро найдено дважды |
| `contour_not_simple` | кольцо касается себя или без площади (`info`) |
| `hatch_boundary_unresolved` | нет простой замкнутой границы штриховки: линии остаются линиями, нужен вопрос |
| `end_extended`, `corner_restored` | конец продлён / угол восстановлен по своей кривой только по толстым чернилам (`info`) |

## Ошибка при слишком большом листе

Размер проверяется по заголовку файла (у PDF — по размеру страницы и dpi) **до декодирования**. Лимит —
`Params.max_image_pixels`, по умолчанию 300 млн пикселей (самый большой лист датасета, A1 при 600 dpi, — 280 млн), и сторона до 32 000 px.
Переопределение: `--max-pixels` в CLI, `DRAWCV_MAX_IMAGE_PIXELS` у адаптера. Бэкенд и `mlocr` проверяют тот же лимит (`MAX_IMAGE_PIXELS`). При превышении `run()` бросает `ImageTooLarge`,
адаптер возвращает `{"status": "error", "error": "ImageTooLarge: ..."}`.
