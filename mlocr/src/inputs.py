"""Bounded page preparation and an identity shared by OCR and CV."""
import hashlib
import math
from pathlib import Path
import numpy as np
import pymupdf
from PIL import Image, ImageOps

MAX_PIXELS = 50_000_000


def page_identity(image, page='page-1'):
    array = np.ascontiguousarray(image, dtype=np.uint8)
    return {'page': page, 'width_px': array.shape[1], 'height_px': array.shape[0],
            'raster_sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def iter_pages(path: Path, page_number: int | None, dpi: int, max_pixels=MAX_PIXELS):
    if not 72 <= dpi <= 600:
        raise ValueError('DPI must be between 72 and 600')
    if path.suffix.lower() == '.pdf':
        with pymupdf.open(path) as doc:
            indexes = [page_number - 1] if page_number is not None else list(range(len(doc)))
            if len(indexes) > 20 or any(i < 0 or i >= len(doc) for i in indexes):
                raise ValueError('Invalid page number or more than 20 selected pages')
            for i in indexes:
                page = doc[i]
                if math.ceil(page.rect.width*dpi/72)*math.ceil(page.rect.height*dpi/72) > max_pixels:
                    raise ValueError('PDF raster exceeds pixel limit; choose a smaller DPI')
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                yield f'page-{i+1}', np.asarray(Image.frombytes('RGB', (pix.width, pix.height), pix.samples))
    else:
        with Image.open(path) as source:
            if source.width*source.height > max_pixels:
                raise ValueError('Image exceeds pixel limit')
            image = ImageOps.exif_transpose(source)
            if image.mode in {'RGBA', 'LA', 'P'}:
                image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image.convert('RGBA'))
            yield 'page-1', np.asarray(image.convert('RGB'))


def assert_same_page(ocr, cv):
    a, b = ocr.get('source', {}), cv.get('raster_identity', {})
    keys = ('raster_sha256', 'page', 'width_px', 'height_px')
    if any(not a.get(key) or a.get(key) != b.get(key) for key in keys):
        raise ValueError('OCR/CV source raster or page differs (identity is required)')
