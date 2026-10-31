from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


Point = tuple[float, float]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Issue(Strict):
    code: str
    severity: Literal["warning", "error"]
    message: str
    object_id: str | None = None


class Primitive(Strict):
    id: str
    kind: Literal["line", "circle", "arc", "polyline"]
    layer: Literal["MAIN", "THIN", "AXIS"] = "MAIN"
    linetype: Literal["BYLAYER", "CONTINUOUS", "CENTER", "HIDDEN", "DASHED"] = "BYLAYER"
    start: Point | None = None
    end: Point | None = None
    center: Point | None = None
    radius: float | None = Field(default=None, gt=0)
    start_angle: float | None = None
    end_angle: float | None = None
    vertices: list[Point] | None = None
    closed: bool = False

    @model_validator(mode="after")
    def check_geometry(self):
        geometry_fields = {
            "line": {"start", "end"},
            "circle": {"center", "radius"},
            "arc": {"center", "radius", "start_angle", "end_angle"},
            "polyline": {"vertices", "closed"},
        }
        unexpected = {
            field for field in ("start", "end", "center", "radius", "start_angle", "end_angle", "vertices")
            if getattr(self, field) is not None and field not in geometry_fields[self.kind]
        }
        if self.closed and self.kind != "polyline":
            unexpected.add("closed")
        if unexpected:
            raise ValueError(f"{self.kind} has unrelated geometry fields: {', '.join(sorted(unexpected))}")
        if self.kind == "line" and (self.start is None or self.end is None):
            raise ValueError("line requires start and end")
        if self.kind == "circle" and (self.center is None or self.radius is None):
            raise ValueError("circle requires center and radius")
        if self.kind == "arc" and (self.center is None or self.radius is None or self.start_angle is None or self.end_angle is None):
            raise ValueError("arc requires center, radius and angles")
        if self.kind == "polyline" and (not self.vertices or len(self.vertices) < (3 if self.closed else 2)):
            raise ValueError("polyline requires vertices")
        return self


class Text(Strict):
    id: str
    value: str
    insert: Point
    height: float = Field(gt=0)
    rotation: float = 0


class Dimension(Strict):
    id: str
    kind: Literal["linear"] = "linear"
    p1: Point | None = None
    p2: Point | None = None
    p1_ref: str | None = None
    p2_ref: str | None = None
    base: Point | None = None
    angle: float = 0
    nominal_mm: float | None = None
    recognized_text: str | None = None
    display: str = "<>"
    view_scale: float = Field(default=1, gt=0)
    link_status: Literal["resolved", "unresolved"] = "resolved"

    @model_validator(mode="after")
    def check_points(self):
        if self.link_status == "resolved" and (
            (self.p1 is None and self.p1_ref is None)
            or (self.p2 is None and self.p2_ref is None) or self.base is None
        ):
            raise ValueError("resolved dimension requires p1, p2 and base")
        if "<>" not in self.display:
            raise ValueError("dimension display must preserve measured value with <>")
        return self


class Hatch(Strict):
    id: str
    outer: list[Point] = Field(default_factory=list)
    holes: list[list[Point]] = Field(default_factory=list)
    outer_ref: str | None = None
    hole_refs: list[str] = Field(default_factory=list)
    pattern: str = "ANSI31"
    scale: float = Field(default=1, gt=0)
    angle: float = 0

    @model_validator(mode="after")
    def check_holes(self):
        if not self.outer_ref and len(self.outer) < 3:
            raise ValueError("hatch requires outer vertices or outer_ref")
        if any(len(h) < 3 for h in self.holes):
            raise ValueError("hatch holes require at least three vertices")
        return self


class AnchorRef(Strict):
    object_id: str
    anchor: str


class DrawingPoint(Strict):
    id: str
    xy: Point
    kind: str = "point"
    refs: list[AnchorRef] = Field(default_factory=list)


class ContourEdge(Strict):
    object_id: str
    kind: Literal["line", "arc", "circle"]
    reversed: bool = False
    start: Point | None = None
    end: Point | None = None
    center: Point | None = None
    radius: float | None = Field(default=None, gt=0)
    start_angle: float | None = None
    end_angle: float | None = None


class Contour(Strict):
    id: str
    role: str = "boundary"
    closed: bool = False
    edges: list[ContourEdge] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    area_mm2: float | None = None
    length_mm: float | None = None
    closure_error_mm: float = 0
    faces_inside: int | None = None
    bbox_mm: tuple[float, float, float, float] | None = None
    parent_id: str | None = None


class DrawingGraph(Strict):
    schema_version: Literal["0.2"] = "0.2"
    coordinate_space: Literal["part_mm", "sheet_mm"]
    source_sha256: str | None = None
    scale_status: Literal["known", "unknown"] = "known"
    primitives: list[Primitive] = Field(default_factory=list)
    texts: list[Text] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    hatches: list[Hatch] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    points: list[DrawingPoint] = Field(default_factory=list)
    contours: list[Contour] = Field(default_factory=list)
    objects: dict[str, dict[str, Any]] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    transforms: dict[str, Any] = Field(default_factory=dict)
    title_block: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    export_profile: Literal["eskd_tz", "editable"] = "eskd_tz"

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy(cls, value):
        # 0.1 has coordinate-only dimensions/hatches and remains a supported input.
        if isinstance(value, dict) and value.get("schema_version") == "0.1":
            return {**value, "schema_version": "0.2"}
        return value

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [x.id for group in (self.primitives, self.texts, self.dimensions, self.hatches,
                                  self.points, self.contours) for x in group]
        if len(ids) != len(set(ids)):
            raise ValueError("drawing object ids must be unique")
        primitives = {p.id for p in self.primitives}
        points = {p.id for p in self.points}
        contours = {c.id for c in self.contours}
        for point in self.points:
            for ref in point.refs:
                if ref.object_id not in primitives:
                    raise ValueError(f"point {point.id}: unknown primitive {ref.object_id}")
        for contour in self.contours:
            for ref in set(contour.object_ids) | {e.object_id for e in contour.edges}:
                if ref not in primitives:
                    raise ValueError(f"contour {contour.id}: unknown primitive {ref}")
            if contour.parent_id is not None and contour.parent_id not in contours:
                raise ValueError(f"contour {contour.id}: unknown parent {contour.parent_id}")
        for dimension in self.dimensions:
            for ref in (dimension.p1_ref, dimension.p2_ref):
                if ref is not None and ref not in points:
                    raise ValueError(f"dimension {dimension.id}: unknown point {ref}")
        for hatch in self.hatches:
            for ref in ([hatch.outer_ref] if hatch.outer_ref else []) + hatch.hole_refs:
                if ref not in contours:
                    raise ValueError(f"hatch {hatch.id}: unknown contour {ref}")
        return self
