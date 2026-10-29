"""Reading PNG / JPEG / PDF into one grayscale page image. After this stage all formats follow the same path."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .transform import Transform

Image.MAX_IMAGE_PIXELS = None
RASTER_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
MM_PER_INCH = 25.4


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


def load_page(path, page_index=0, pdf_render_dpi=300.0):
    path = Path(path)
    suffix = path.suffix.lower()
    digest = sha256_of(path)
    if suffix == ".pdf":
        return _load_pdf(path, digest, page_index, pdf_render_dpi)
    if suffix not in RASTER_FORMATS:
        raise ValueError(f"unsupported input format: {suffix} (PNG, JPEG or PDF expected)")
    with Image.open(path) as im:
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
                dpi=dpi_value, notes=notes)


def _load_pdf(path, digest, page_index, dpi):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        count = len(pdf)
        if not 0 <= page_index < count:
            raise ValueError(f"page {page_index + 1} requested, PDF has {count}")
        page = pdf[page_index]
        w_pt, h_pt = page.get_size()
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
