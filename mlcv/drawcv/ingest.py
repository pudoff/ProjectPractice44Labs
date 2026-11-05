"""Reading PNG / JPEG / PDF into one grayscale page image. After this stage all formats follow the same path.

The size of the image is checked from the file header (PDF: from the page size) before anything is decoded, so a small
file that unpacks into gigabytes is refused without allocating them."""
import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .transform import Transform

RASTER_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
MM_PER_INCH = 25.4
DEFAULT_MAX_PIXELS = 300_000_000
MAX_SIDE_PX = 32000                  # OpenCV warpAffine / remap refuse sides of 32767 px and more
_header_lock = threading.Lock()


class ImageTooLarge(ValueError):
    """The decoded image would exceed the configured limit; nothing was decoded."""


def check_size(width, height, max_pixels, what="image"):
    pixels = int(width) * int(height)
    if pixels > max_pixels:
        raise ImageTooLarge(f"{what} {int(width)}x{int(height)} px = {pixels / 1e6:.1f} Mpx exceeds the limit of "
                            f"{max_pixels / 1e6:.0f} Mpx (Params.max_image_pixels, DRAWCV_MAX_IMAGE_PIXELS)")
    if max(width, height) > MAX_SIDE_PX:
        raise ImageTooLarge(f"{what} {int(width)}x{int(height)} px: a side exceeds {MAX_SIDE_PX} px")


def open_header(path):
    """Image.open reads only the header. Pillow's own bomb check is global and lower than the sheets of the dataset
    (A1 at 600 dpi), and changing it for the whole process would also change it for the other modules, so it is
    lifted for this call only; check_size() follows before any decoding."""
    with _header_lock:
        saved = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            return Image.open(path)
        finally:
            Image.MAX_IMAGE_PIXELS = saved


@dataclass
class Page:
    gray: np.ndarray
    source_name: str
    source_format: str
    source_sha256: str
    page_index: int = 0
    page_count: int = 1
    dpi: float | None = None                 # physical resolution if the file states it (PDF render dpi or PNG/JPEG metadata)
    pdf_page_mm: tuple[float, float] | None = None
    notes: list = field(default_factory=list)
    raster_identity: dict = field(default_factory=dict)

    @property
    def size_px(self):
        return int(self.gray.shape[1]), int(self.gray.shape[0])

    def source_px_to_pdf_pt(self):
        if self.pdf_page_mm is None:
            return None
        k = 72.0 / self.dpi
        h_pt = self.pdf_page_mm[1] / MM_PER_INCH * 72.0
        return Transform([[k, 0, 0], [0, -k, h_pt], [0, 0, 1]])

    def info(self):
        return {"name": self.source_name, "format": self.source_format, "sha256": self.source_sha256,
                "page_index": self.page_index, "page_count": self.page_count, "size_px": list(self.size_px),
                "dpi": self.dpi, "pdf_page_mm": list(self.pdf_page_mm) if self.pdf_page_mm else None,
                "source_px_to_pdf_pt": self.source_px_to_pdf_pt().to_list() if self.pdf_page_mm else None,
                "notes": self.notes}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_page(path, page_index=0, pdf_render_dpi=300.0, max_pixels=DEFAULT_MAX_PIXELS):
    path = Path(path)
    suffix = path.suffix.lower()
    digest = sha256_of(path)
    if suffix == ".pdf":
        return _load_pdf(path, digest, page_index, pdf_render_dpi, max_pixels)
    if suffix not in RASTER_FORMATS:
        raise ValueError(f"unsupported input format: {suffix} (PNG, JPEG or PDF expected)")
    with open_header(path) as im:
        check_size(im.width, im.height, max_pixels)
        notes = []
        orientation = im.getexif().get(0x0112, 1)
        if orientation != 1:
            notes.append(f"EXIF orientation {orientation} applied; source_px are after it")
        im = ImageOps.exif_transpose(im)
        dpi = im.info.get("dpi")
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            white = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(white, im)
        if im.mode == "I;16" or im.mode == "I":
            arr = np.asarray(im, dtype=np.float64)
            gray = np.clip(arr / max(arr.max(), 1) * 255, 0, 255).astype(np.uint8)
        else:
            gray = np.asarray(im.convert("L"), dtype=np.uint8).copy()
        identity = {'page': 'page-1', 'width_px': im.width, 'height_px': im.height,
                    'raster_sha256': hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()}
    dpi_value = float(dpi[0]) if dpi and dpi[0] and abs(dpi[0] - dpi[1]) < 0.5 and dpi[0] > 50 else None
    if dpi_value:
        from .calibrate import standard_sheet_of
        size_mm = (gray.shape[1] / dpi_value * MM_PER_INCH, gray.shape[0] / dpi_value * MM_PER_INCH)
        if not standard_sheet_of(size_mm, tol=0.03 * max(size_mm)):
            notes.append(f"dpi metadata {dpi_value:g} ignored: page would be {size_mm[0]:.0f}x{size_mm[1]:.0f} mm")
            dpi_value = None
        else:
            notes.append(f"dpi metadata {dpi_value:g} used as a resolution hint")
    return Page(gray=gray, source_name=path.name, source_format=RASTER_FORMATS[suffix], source_sha256=digest,
                dpi=dpi_value, notes=notes, raster_identity=identity)


def _load_pdf(path, digest, page_index, dpi, max_pixels):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        count = len(pdf)
        if not 0 <= page_index < count:
            raise ValueError(f"page {page_index + 1} requested, PDF has {count}")
        page = pdf[page_index]
        w_pt, h_pt = page.get_size()
        check_size(w_pt * dpi / 72.0, h_pt * dpi / 72.0, max_pixels, f"PDF page at {dpi:g} dpi")
        bitmap = page.render(scale=dpi / 72.0, grayscale=True, fill_color=(255, 255, 255, 255))
        gray = np.asarray(bitmap.to_pil().convert("L"), dtype=np.uint8).copy()
    finally:
        pdf.close()
    notes = [f"PDF page rendered at {dpi:g} dpi"]
    if count > 1:
        notes.append(f"multi-page PDF: page {page_index + 1} of {count} processed")
    return Page(gray=gray, source_name=path.name, source_format="pdf", source_sha256=digest, page_index=page_index,
                page_count=count, dpi=dpi, pdf_page_mm=(w_pt / 72.0 * MM_PER_INCH, h_pt / 72.0 * MM_PER_INCH),
                notes=notes)
