"""Resolve semantic references and validate geometry before writing CAD entities.

Tolerances here are numerical (millimetres in the graph), not manufacturing or
recognition tolerances. We never close a real gap or fit geometry to an OCR number.
"""
import math

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from app.domain.drawing import ContourEdge, DrawingGraph


EPS_MM = 1e-4  # CV serializes coordinates with four decimal places.


class GeometryError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def polar(center, radius, degrees):
    a = math.radians(degrees)
    return (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))


def primitive_problem(item):
    if item.kind == "line" and math.dist(item.start, item.end) <= EPS_MM:
        return "Отрезок имеет нулевую длину"
    if item.kind == "arc" and math.isclose((item.end_angle - item.start_angle) % 360, 0, abs_tol=1e-9):
        return "Дуга имеет нулевой угловой размах"
    if item.kind == "polyline":
        if any(math.dist(a, b) <= EPS_MM for a, b in zip(item.vertices, item.vertices[1:])):
            return "Полилиния содержит отрезок нулевой длины"
        if item.closed:
            try:
                polygon(item.vertices)
            except GeometryError as exc:
                return str(exc)
    return None


def polygon(vertices):
    if len(vertices) < 3:
        raise GeometryError("invalid_boundary", "Граница содержит меньше трёх точек")
    result = Polygon(vertices)
    if not result.is_valid or result.area <= EPS_MM ** 2:
        raise GeometryError("invalid_boundary", f"Некорректная граница: {explain_validity(result)}")
    return result


def anchor_position(item, anchor):
    if item.kind == "line":
        values = {"start": item.start, "end": item.end,
                  "mid": tuple((a + b) / 2 for a, b in zip(item.start, item.end))}
        if anchor in values:
            return values[anchor]
    if item.kind in ("circle", "arc"):
        if anchor == "center":
            return item.center
        if item.kind == "arc" and anchor in ("start", "end"):
            return polar(item.center, item.radius, item.start_angle if anchor == "start" else item.end_angle)
        if anchor in ("quadrant_0", "quadrant_90", "quadrant_180", "quadrant_270"):
            return polar(item.center, item.radius, float(anchor.split("_")[1]))
    if item.kind == "polyline" and anchor.startswith("vertex_"):
        index = anchor.removeprefix("vertex_")
        if index.isdigit() and int(index) < len(item.vertices):
            return item.vertices[int(index)]
    raise GeometryError("invalid_point_anchor", f"Недопустимая точка {anchor} объекта {item.id}")


class GeometryResolver:
    def __init__(self, graph: DrawingGraph):
        self.graph = graph
        self.primitives = {p.id: p for p in graph.primitives}
        self.points = {p.id: p for p in graph.points}
        self.contours = {c.id: c for c in graph.contours}
        self.cache = {}

    def point(self, point_id):
        point = self.points[point_id]
        if not point.refs:
            return point.xy
        for ref in point.refs:
            problem = primitive_problem(self.primitives[ref.object_id])
            if problem:
                raise GeometryError("invalid_point_anchor", f"Точка {point_id}: {problem}")
        values = [anchor_position(self.primitives[r.object_id], r.anchor) for r in point.refs]
        if any(math.dist(a, b) > EPS_MM for a in values for b in values):
            raise GeometryError("ambiguous_point", f"Опорные объекты точки {point_id} не совпадают")
        return values[0]

    def dimension(self, item):
        p1 = self.point(item.p1_ref) if item.p1_ref else item.p1
        p2 = self.point(item.p2_ref) if item.p2_ref else item.p2
        return item.model_copy(update={"p1": p1, "p2": p2})

    def _edge(self, edge):
        """Keep CV subsegments while following edits of the referenced primitive.

        objects.geometry_mm is the observation before edits. It defines segment
        parameters; the accepted primitive defines current coordinates.
        """
        item = self.primitives[edge.object_id]
        if item.kind != edge.kind and not (item.kind == "circle" and edge.kind == "arc"):
            raise GeometryError("contour_kind_mismatch", f"Изменился тип объекта {item.id}")
        problem = primitive_problem(item)
        if problem:
            raise GeometryError("invalid_primitive", problem)
        original = self.graph.objects.get(item.id, {}).get("geometry_mm", {})
        if item.kind == "line":
            if edge.start is None or edge.end is None:
                raise GeometryError("invalid_contour_edge", "Не заданы концы ребра")
            a = original.get("start", item.start)
            b = original.get("end", item.end)
            dx, dy = b[0] - a[0], b[1] - a[1]
            length2 = dx * dx + dy * dy
            if length2 <= EPS_MM ** 2:
                raise GeometryError("invalid_contour_edge", "Вырожденное исходное ребро")
            def current(point):
                t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2
                if math.dist(point, (a[0] + t * dx, a[1] + t * dy)) > 2 * EPS_MM:
                    raise GeometryError("stale_contour_edge", f"Граница не лежит на объекте {item.id}")
                tolerance = EPS_MM / math.sqrt(length2)
                if t < -tolerance or t > 1 + tolerance:
                    raise GeometryError("stale_contour_edge", f"Граница выходит за объект {item.id}")
                return tuple(x + t * (y - x) for x, y in zip(item.start, item.end))
            return edge.model_copy(update={"start": current(edge.start), "end": current(edge.end)})
        if item.kind == "circle":
            # drawcv represents circular boundaries as arc edges, including
            # a full circle with equal angles. Partial edges remain partial.
            if edge.kind == "arc":
                if edge.start_angle is None or edge.end_angle is None:
                    raise GeometryError("invalid_contour_edge", "Не заданы углы дуги окружности")
                start = edge.start_angle
                end = start + ((edge.end_angle - start) % 360 or 360)
            else:
                start, end = 0.0, 360.0
        else:
            a0 = original.get("start_angle", item.start_angle)
            a1 = original.get("end_angle", item.end_angle)
            span = (a1 - a0) % 360 or 360
            if edge.start_angle is None or edge.end_angle is None:
                raise GeometryError("invalid_contour_edge", "Не заданы углы дуги границы")
            t0 = ((edge.start_angle - a0) % 360) / span
            subspan = (edge.end_angle - edge.start_angle) % 360 or 360
            t1 = t0 + subspan / span
            if t0 > 1 + 1e-5 or t1 > 1 + 1e-5:
                raise GeometryError("stale_contour_edge", f"Граница выходит за дугу {item.id}")
            current_span = (item.end_angle - item.start_angle) % 360 or 360
            start, end = item.start_angle + t0 * current_span, item.start_angle + t1 * current_span
        p, q = polar(item.center, item.radius, start), polar(item.center, item.radius, end)
        if edge.reversed:
            p, q = q, p
        return edge.model_copy(update={"center": item.center, "radius": item.radius,
                                       "start_angle": start, "end_angle": end, "start": p, "end": q})

    def contour(self, contour_id):
        if contour_id in self.cache:
            return self.cache[contour_id]
        contour = self.contours[contour_id]
        if not contour.closed:
            raise GeometryError("open_contour", f"Контур {contour_id} не замкнут")
        edges = [self._edge(edge) for edge in contour.edges]
        if not edges:
            for object_id in contour.object_ids:
                item = self.primitives[object_id]
                problem = primitive_problem(item)
                if problem:
                    raise GeometryError("invalid_primitive", problem)
                if item.kind == "polyline":
                    vertices = item.vertices + ([item.vertices[0]] if item.closed else [])
                    edges.extend(ContourEdge(object_id=item.id, kind="line", start=a, end=b)
                                 for a, b in zip(vertices, vertices[1:]))
                elif item.kind == "line":
                    edges.append(ContourEdge(object_id=item.id, kind="line", start=item.start, end=item.end))
                else:
                    start = item.start_angle if item.kind == "arc" else 0
                    end = item.end_angle if item.kind == "arc" else 360
                    edges.append(ContourEdge(object_id=item.id, kind=item.kind, center=item.center,
                                 radius=item.radius, start_angle=start, end_angle=end,
                                 start=polar(item.center, item.radius, start), end=polar(item.center, item.radius, end)))
        if not edges:
            raise GeometryError("empty_contour", f"Контур {contour_id} не содержит рёбер")
        gap = max(math.dist(a.end, b.start) for a, b in zip(edges, edges[1:] + edges[:1]))
        if gap > EPS_MM:
            raise GeometryError("contour_gap", f"Контур {contour_id}: разрыв {gap:.6f} мм; не замкнут по координатам")
        vertices = []
        for edge in edges:
            if edge.kind == "line":
                vertices.append(edge.start)
            else:
                span = (edge.end_angle - edge.start_angle) % 360 or 360
                # Sampling is for validation only; DXF receives native arcs below.
                # Bound chord error instead of using a fixed angular step.
                step = 2 * math.acos(max(-1.0, 1 - min(EPS_MM / 2 / edge.radius, 1)))
                count = max(12, math.ceil(math.radians(span) / max(step, 1e-12)))
                if count > 100_000:
                    raise GeometryError("boundary_too_complex", "Граница требует слишком большого числа проверочных точек")
                arc = [polar(edge.center, edge.radius, edge.start_angle + span * i / count) for i in range(count + 1)]
                vertices.extend(arc[::-1] if edge.reversed else arc)
        result = (edges, polygon(vertices))
        self.cache[contour_id] = result
        return result

    def hatch(self, item):
        outer_edges, outer = self.contour(item.outer_ref) if item.outer_ref else (None, polygon(item.outer))
        if item.hole_refs:
            holes = [self.contour(ref) for ref in item.hole_refs]
        else:
            holes = [(None, polygon(vertices)) for vertices in item.holes]
        for _, hole in holes:
            if not outer.contains(hole) or outer.boundary.distance(hole) <= 2 * EPS_MM:
                raise GeometryError("invalid_hatch_hole", "Отверстие штриховки должно быть строго внутри внешней границы")
        for i, (_, hole) in enumerate(holes):
            if any(not hole.disjoint(other) for _, other in holes[i + 1:]):
                raise GeometryError("invalid_hatch_hole", "Отверстия штриховки пересекаются или вложены друг в друга")
        return outer_edges, holes
