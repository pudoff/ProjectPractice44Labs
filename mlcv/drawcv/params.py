"""Processing parameters. All geometric tolerances are in sheet millimetres; pixel sizes are derived from the
calibration and the measured thin-line stroke width, never fixed in pixels."""
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Params:
    work_px_per_mm: float = 11.811          # working resolution (300 dpi); images are never upsampled
    pdf_render_dpi: float = 300.0
    max_skew_deg: float = 5.0               # only small scan skew is corrected automatically
    min_skew_deg: float = 0.03
    rotate_deg: int = 0                     # manual 0/90/180/270 rotation applied before deskew
    contrast: str = "normalize"             # "normalize" (flatten background) | "clahe" | "none"
    despeckle_mm2: float = 0.01             # ink specks smaller than this are dropped (counted in the report)

    sheet: str | None = None                # force sheet format (A0..A4), otherwise guessed
    manual_px_per_mm: float | None = None   # manual calibration of the working image
    manual_origin_px: tuple[float, float] | None = None

    text_provider: str = "auto"             # "auto" | "rapidocr" | "components" | "file" | "none"
    ocr_json: str | None = None             # OCR result with text polygons in source pixels
    text_margin_mm: float = 0.4

    tol_fit_mm: float = 0.08                # max deviation of the centre line from a fitted segment/arc
    min_len_mm: float = 1.0                 # shorter primitives are reported as short noise
    merge_angle_deg: float = 0.6
    merge_offset_mm: float = 0.12
    merge_gap_mm: float = 0.35
    snap_angle_deg: float = 0.35
    arc_min_radius_mm: float = 1.0          # smaller arcs are marked unsupported
    arc_min_sweep_deg: float = 20.0
    arc_max_radius_mm: float = 500.0
    duplicate_tol_mm: float = 0.2
    dash_max_mm: float = 30.0               # GOST 2.303: dash-dot dashes 5-30 mm, gaps 3-5 mm
    dash_gap_max_mm: float = 6.0
    dash_min_pieces: int = 3                # a short centre line is long dash - dot - long dash

    hatch_min_lines: int = 4
    hatch_spacing_mm: tuple[float, float] = (0.8, 12.0)

    snap_tol_mm: float = 0.3                # endpoint join tolerance for contours: snap_tol_mm + factor x thick
    snap_stroke_factor: float = 0.75        # stroke width (skeleton ends move by about half a stroke at junctions)
    gap_search_mm: float = 3.0              # dangling ends closer than this to other geometry are reported as gaps
    extend_max_mm: float = 2.0              # an end may be extended along its own curve only over thick ink

    overlay_max_side_px: int = 4000
    save_debug_images: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        values = dict(data)
        for key in ("manual_origin_px", "hatch_spacing_mm"):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        return cls(**values)
