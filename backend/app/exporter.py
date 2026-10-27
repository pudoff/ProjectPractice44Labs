import math
from pathlib import Path
import ezdxf
from .drawing import DrawingGraph, Issue


def new_document():
    doc = ezdxf.new("R2007", units=4)
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    for name, color, lineweight in (("MAIN", 7, 50), ("THIN", 3, 25), ("AXIS", 1, 25), ("DIMS", 4, 25), ("TEXT", 7, 25)):
        doc.layers.new(name, dxfattribs={"color": color, "lineweight": lineweight})
    doc.styles.new("DRAWING", dxfattribs={"font": "isocpeur.ttf"})
    style = doc.dimstyles.new("DRAWING")
    style.dxf.dimtxsty = "DRAWING"
    style.dxf.dimtxt = 3.5
    style.dxf.dimasz = 3
    style.dxf.dimdec = 2
    style.dxf.dimzin = 8
    style.dxf.dimlfac = 1
    return doc


def add_primitive(msp, item):
    attrs = {"layer": item.layer}
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


def add_dimension(msp, item, issues, tolerance=0.1):
    if item.link_status == "unresolved":
        issues.append(Issue(code="unresolved_dimension_link", severity="warning", message="Размер не связан с геометрией", object_id=item.id))
        return
    a = math.radians(item.angle)
    measured = abs((item.p2[0] - item.p1[0]) * math.cos(a) + (item.p2[1] - item.p1[1]) * math.sin(a))
    physical = measured / item.view_scale
    if item.nominal_mm is not None and abs(physical - item.nominal_mm) > tolerance:
        issues.append(Issue(code="dimension_mismatch", severity="warning", message=f"Измерено {physical:.3f} мм; распознано {item.nominal_mm:.3f} мм", object_id=item.id))
    dim = msp.add_linear_dim(base=item.base, p1=item.p1, p2=item.p2, angle=item.angle, text=item.display,
                             dimstyle="DRAWING", override={"dimlfac": 1.0 / item.view_scale},
                             dxfattribs={"layer": "DIMS"})
    dim.render()


def add_hatch(msp, item):
    hatch = msp.add_hatch(dxfattribs={"layer": "THIN"})
    hatch.set_pattern_fill(item.pattern, scale=item.scale, angle=item.angle)
    hatch.paths.add_polyline_path(item.outer, is_closed=True, flags=1)
    for hole in item.holes:
        hatch.paths.add_polyline_path(hole, is_closed=True, flags=0)


def export_dxf(graph: DrawingGraph, path: Path) -> list[Issue]:
    issues = list(graph.issues)
    if graph.scale_status == "unknown":
        if not any(issue.code == "unknown_scale" for issue in issues):
            issues.append(Issue(code="unknown_scale", severity="error", message="Масштаб детали не определён"))
        return issues
    if not graph.primitives and not graph.hatches:
        issues.append(Issue(code="empty_drawing", severity="error", message="Геометрия чертежа отсутствует"))
        return issues
    doc = new_document()
    msp = doc.modelspace()
    for item in graph.primitives:
        add_primitive(msp, item)
    for item in graph.texts:
        add_text(msp, item)
    for item in graph.dimensions:
        add_dimension(msp, item, issues)
    for item in graph.hatches:
        add_hatch(msp, item)
    if not doc.audit().has_errors:
        doc.saveas(path)
    else:
        issues.append(Issue(code="dxf_audit", severity="error", message="Ошибка структуры DXF"))
    return issues
