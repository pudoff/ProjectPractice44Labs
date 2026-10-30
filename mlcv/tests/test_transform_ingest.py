import math

import numpy as np
import pytest

from drawcv.calibrate import to_sheet_transform
from drawcv.ingest import load_page
from drawcv.transform import Transform, apply_to_geometry


def test_transform_round_trip():
    t = Transform.from_affine([[0.98, -0.2, 15.0], [0.2, 0.98, -7.0]]).then(Transform.scale(0.5)).then(Transform.translate(3, 4))
    pts = np.random.default_rng(1).uniform(-500, 500, (100, 2))
    assert np.allclose(t.inverse().apply(t.apply(pts)), pts, atol=1e-9)


def test_arc_angles_survive_y_flip():
    """sheet_mm (y up) -> pixels (y down): the arc must still cover the same points."""
    to_mm = to_sheet_transform([10.0, 10.0], [100.0, 900.0])
    to_px = to_mm.inverse()
    arc = {"center": [30.0, 40.0], "radius": 12.0, "start_angle": 20.0, "end_angle": 110.0}
    g = apply_to_geometry(to_px, arc)
    for a in (25.0, 65.0, 105.0):
        p_mm = np.array(arc["center"]) + 12 * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        p_px = to_px.apply(p_mm)
        b = math.degrees(math.atan2(p_px[1] - g["center"][1], p_px[0] - g["center"][0])) % 360
        assert (b - g["start_angle"]) % 360 <= (g["end_angle"] - g["start_angle"]) % 360 + 1e-6
    assert g["radius"] == pytest.approx(120.0)


def test_three_formats_load_alike(synth_dir):
    png = load_page(synth_dir / "control.png")
    jpg = load_page(synth_dir / "control.jpg")
    pdf = load_page(synth_dir / "control.pdf", pdf_render_dpi=300)
    assert png.size_px == jpg.size_px
    assert abs(pdf.size_px[0] - png.size_px[0]) <= 2 and abs(pdf.size_px[1] - png.size_px[1]) <= 2
    assert pdf.pdf_page_mm == pytest.approx((210.0, 297.0), abs=0.2)
    assert png.source_format == "png" and jpg.source_format == "jpeg" and pdf.source_format == "pdf"
    assert len(png.source_sha256) == 64


def test_unsupported_format(tmp_path):
    f = tmp_path / "x.tif"
    f.write_bytes(b"II*\x00")
    with pytest.raises(ValueError):
        load_page(f)
