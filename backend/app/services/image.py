from pathlib import Path

import math
from PIL import Image, ImageOps
import pymupdf

from app.config import get_settings
from app.infra.storage import atomic_json


class ImageService:
    def prepare(self, source: Path, run_path: Path) -> Path:
        target = run_path / "pages" / "page-001.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        limit = get_settings().max_image_pixels
        metadata = {"original_name": source.name, "prepared_name": target.name}
        if source.suffix.lower() == ".pdf":
            with pymupdf.open(source) as document:
                if document.page_count != 1:
                    raise ValueError("Поддерживается только одностраничный PDF")
                page = document[0]
                width, height = math.ceil(page.rect.width * 300 / 72), math.ceil(page.rect.height * 300 / 72)
                if width * height > limit:
                    raise ValueError("Страница PDF превышает допустимое число пикселей")
                page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB, alpha=False).save(target)
                metadata.update(render_dpi=300, page_mm=[page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72])
        else:
            with Image.open(source) as image:
                if image.width * image.height > limit:
                    raise ValueError("Изображение превышает допустимое число пикселей")
                metadata["exif_orientation"] = image.getexif().get(0x0112, 1)
                image = ImageOps.exif_transpose(image)
                dpi = image.info.get("dpi")
                if image.mode in ("RGBA", "LA", "P"):
                    image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
                image.convert("RGB").save(target, **({"dpi": dpi} if dpi else {}))
                metadata["source_dpi"] = dpi
        atomic_json(target.with_suffix(".json"), metadata)
        return target
