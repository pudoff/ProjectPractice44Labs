import math
from pathlib import Path
import os
import tempfile
import ezdxf
from app.config import get_settings
from app.domain.drawing import DrawingGraph, Issue
from app.services.geometry import GeometryError, GeometryResolver, primitive_problem


def new_document(profile="eskd_tz"):
    settings = get_settings()
    doc = ezdxf.new(settings.dxf_version, units=settings.dxf_units)
    doc.header["$INSUNITS"] = settings.dxf_units
    doc.header["$MEASUREMENT"] = settings.dxf_measurement
    for name, color, lineweight in settings.dxf_layers:
        doc.layers.new(name, dxfattribs={"color": color, "lineweight": lineweight})
    if profile == "editable":
        doc.layers.new("HATCH", dxfattribs={"color": 2, "lineweight": 18})
    doc.linetypes.new("CENTER", dxfattribs={"description": "Center line", "pattern": [20.0, 12.0, -3.0, 2.0, -3.0]})
    doc.linetypes.new("HIDDEN", dxfattribs={"description": "Hidden line", "pattern": [6.0, 3.0, -3.0]})
    doc.linetypes.new("DASHED", dxfattribs={"description": "Dashed line", "pattern": [6.0, 3.0, -3.0]})
    doc.layers.get("AXIS").dxf.linetype = "CENTER"
    doc.styles.new("DRAWING", dxfattribs={"font": settings.dxf_font})
    style = doc.dimstyles.new("DRAWING")
    style.dxf.dimtxsty = "DRAWING"
    style.dxf.dimtxt = settings.dxf_text_height
    style.dxf.dimasz = settings.dxf_arrow_size
    style.dxf.dimdec = settings.dxf_dimension_decimals
    style.dxf.dimzin = settings.dxf_dimension_zero_suppression
    style.dxf.dimlfac = 1
    return doc


def add_primitive(msp, item):
    attrs = {"layer": item.layer, "linetype": item.linetype}
    if item.kind == "line":
        msp.add_line(item.start, item.end, dxfattribs=attrs)
    elif item.kind == "circle":
        msp.add_circle(item.center, item.radius, dxfattribs=attrs)
    elif item.kind == "arc":
        msp.add_arc(item.center, item.radius, item.start_angle, item.end_angle, dxfattribs=attrs)
    elif item.kind == "polyline":
        msp.add_lwpolyline(item.vertices, close=item.closed, dxfattribs=attrs)


def add_text(msp, item):
    msp.add_text(item.value, dxfattribs={"layer": "TEXT", "style": "DRAWING", "height": item.height, "insert": item.insert, "rotation": item.rotation})


def add_dimension(msp, item, issues, tolerance=None):
    if tolerance is None:
        tolerance = get_settings().dimension_numeric_tolerance_mm
    if item.link_status == "unresolved":
        issues.append(Issue(code="unresolved_dimension_link", severity="warning", message="Размер не связан с геометрией", object_id=item.id))
        return
    a = math.radians(item.angle)
    measured = abs((item.p2[0] - item.p1[0]) * math.cos(a) + (item.p2[1] - item.p1[1]) * math.sin(a))
    physical = measured / item.view_scale
    if physical <= tolerance:
        issues.append(Issue(code="invalid_dimension", severity="error", message="Измеряемое расстояние равно нулю", object_id=item.id))
        return
    if item.nominal_mm is not None and item.nominal_mm <= 0:
        issues.append(Issue(code="invalid_dimension", severity="error", message="Номинал линейного размера должен быть положительным", object_id=item.id))
        return
    if item.nominal_mm is not None and abs(physical - item.nominal_mm) > tolerance:
        issues.append(Issue(code="dimension_mismatch", severity="warning", message=f"Измерено {physical:.3f} мм; распознано {item.nominal_mm:.3f} мм", object_id=item.id))
    dim = msp.add_linear_dim(base=item.base, p1=item.p1, p2=item.p2, angle=item.angle, text=item.display,
                             dimstyle="DRAWING", override={"dimlfac": 1.0 / item.view_scale},
                             dxfattribs={"layer": "DIMS"})
    dim.render()


def add_hatch(msp, item, resolved, profile):
    hatch = msp.add_hatch(dxfattribs={"layer": "HATCH" if profile == "editable" else "THIN"})
    hatch.set_pattern_fill(item.pattern, scale=item.scale, angle=item.angle)
    outer_edges, holes = resolved
    def boundary(edges, vertices, flags):
        if edges is None:
            hatch.paths.add_polyline_path(vertices, is_closed=True, flags=flags)
        else:
            path = hatch.paths.add_edge_path(flags=flags)
            for edge in edges:
                if edge.kind == "line":
                    path.add_line(edge.start, edge.end)
                else:
                    path.add_arc(edge.center, edge.radius, edge.start_angle, edge.end_angle, ccw=not edge.reversed)
    boundary(outer_edges, item.outer, 1)
    for index, (edges, _) in enumerate(holes):
        boundary(edges, item.holes[index] if edges is None else [], 0)


class DxfExporter:
    def export(self, graph: DrawingGraph, path: Path) -> list[Issue]:
        issues = list(graph.issues)
        if graph.scale_status == "unknown":
            if not any(issue.code == "unknown_scale" for issue in issues):
                issues.append(Issue(code="unknown_scale", severity="error", message="Масштаб детали не определён"))
            return issues
        if not graph.primitives and not graph.hatches:
            issues.append(Issue(code="empty_drawing", severity="error", message="Геометрия чертежа отсутствует"))
            return issues
        doc = new_document(graph.export_profile)
        msp = doc.modelspace()
        resolver = GeometryResolver(graph)
        def problem(code, message, object_id):
            issues.append(Issue(code=code, severity="error", message=message, object_id=object_id))
        for contour in graph.contours:
            if contour.closed:
                try:
                    resolver.contour(contour.id)
                except GeometryError as exc:
                    problem(exc.code, str(exc), contour.id)
        for item in graph.primitives:
            invalid = primitive_problem(item)
            if invalid:
                problem("invalid_primitive", invalid, item.id)
                continue
            add_primitive(msp, item)
        for item in graph.texts:
            add_text(msp, item)
        for item in graph.dimensions:
            if graph.coordinate_space == "part_mm" and item.view_scale != 1:
                problem("coordinate_scale_conflict", "Геометрия уже в миллиметрах детали: view_scale должен быть 1", item.id)
                continue
            try:
                add_dimension(msp, resolver.dimension(item), issues)
            except GeometryError as exc:
                problem(exc.code, str(exc), item.id)
        for item in graph.hatches:
            try:
                resolved = resolver.hatch(item)
                add_hatch(msp, item, resolved, graph.export_profile)
            except GeometryError as exc:
                problem(exc.code, str(exc), item.id)
        if not msp.query("LINE CIRCLE ARC LWPOLYLINE HATCH"):
            problem("empty_drawing", "После проверки не осталось корректной геометрии", None)
            return issues
        if not doc.audit().has_errors:
            # Publish only the complete file. Each job run owns a separate directory.
            handle, temporary = tempfile.mkstemp(prefix=".result-", suffix=".dxf", dir=path.parent)
            os.close(handle)
            try:
                doc.saveas(temporary)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        else:
            issues.append(Issue(code="dxf_audit", severity="error", message="Ошибка структуры DXF"))
        return issues
