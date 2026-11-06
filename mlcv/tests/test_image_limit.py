"""The size of a page is checked from the header, before anything is decoded."""
import struct
import time
import tracemalloc
import zlib

import pytest
from PIL import Image

from drawcv import Params, run
from drawcv.adapter import run_cv
from drawcv.ingest import MAX_SIDE_PX, ImageTooLarge, check_size, load_page

HUGE = 60000                         # 3.6 Gpx: decoding would need gigabytes


def png_bomb(path, width=HUGE, height=HUGE):
    """A real small PNG header declaring a giant 8-bit image; the compressed stream would inflate to gigabytes."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(b"\0" * 1000)) + chunk(b"IEND", b""))
    return path


def jpeg_bomb(path, width=HUGE, height=HUGE):
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    sos = b"\xff\xda" + struct.pack(">HB", 8, 1) + b"\x01\x00\x00\x3f\x00"
    path.write_bytes(b"\xff\xd8" + sof + sos + b"\x00" * 16 + b"\xff\xd9")
    return path


def giant_pdf(path, side_pt=14400):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(side_pt, side_pt)                       # 5 x 5 m: 3.6 Gpx at 300 dpi
    with open(path, "wb") as f:
        pdf.save(f)
    pdf.close()
    return path


@pytest.mark.parametrize("make,name", [(png_bomb, "bomb.png"), (jpeg_bomb, "bomb.jpg"), (giant_pdf, "giant.pdf")])
def test_giant_input_is_refused_without_decoding(tmp_path, make, name):
    path = make(tmp_path / name)
    tracemalloc.start()
    t0 = time.perf_counter()
    with pytest.raises(ImageTooLarge, match="exceeds"):
        load_page(path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert time.perf_counter() - t0 < 5
    assert peak < 50 * 1024 * 1024, "nothing may be decoded"


def test_limit_is_a_parameter(tmp_path):
    path = tmp_path / "page.png"
    Image.new("L", (1000, 1000), 255).save(path)
    assert load_page(path, max_pixels=1_000_000).size_px == (1000, 1000)
    with pytest.raises(ImageTooLarge):
        load_page(path, max_pixels=999_999)
    with pytest.raises(ImageTooLarge):
        run(path, params=Params(max_image_pixels=999_999))


def test_pdf_limit_depends_on_render_dpi(synth_dir):
    assert load_page(synth_dir / "control.pdf", pdf_render_dpi=150, max_pixels=2_500_000).size_px
    with pytest.raises(ImageTooLarge):
        load_page(synth_dir / "control.pdf", pdf_render_dpi=300, max_pixels=2_500_000)


def test_a_long_narrow_image_is_refused_by_its_side():
    with pytest.raises(ImageTooLarge, match="side"):
        check_size(MAX_SIDE_PX + 1, 100, 300_000_000)


def test_largest_dataset_sheet_size_is_allowed():
    """A1 at 600 dpi: the largest sheet of the dataset (19892 x 14058 px) must not be refused."""
    check_size(19892, 14058, Params().max_image_pixels)


def test_adapter_returns_an_error_and_reads_the_environment(tmp_path, monkeypatch):
    bomb = png_bomb(tmp_path / "bomb.png")
    cv = run_cv(bomb)
    assert cv["status"] == "error" and "ImageTooLarge" in cv["error"] and cv["primitives"] == []
    page = tmp_path / "page.png"
    Image.new("L", (1000, 1000), 255).save(page)
    monkeypatch.setenv("DRAWCV_MAX_IMAGE_PIXELS", "999")
    cv = run_cv(page)
    assert cv["status"] == "error" and "ImageTooLarge" in cv["error"]


def test_global_pillow_limit_is_not_changed(tmp_path):
    """The backend and OCR run in the same process and rely on Pillow's own protection."""
    before = Image.MAX_IMAGE_PIXELS
    assert before is not None
    with pytest.raises(ImageTooLarge):
        load_page(png_bomb(tmp_path / "bomb.png"))
    assert Image.MAX_IMAGE_PIXELS == before
