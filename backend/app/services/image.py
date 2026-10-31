from pathlib import Path

from PIL import Image
import pymupdf


class ImageService:
    def prepare(self, source: Path, run_path: Path) -> Path:
        target = run_path / "pages" / "page-001.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".pdf":
            with pymupdf.open(source) as document:
                if document.page_count != 1:
                    raise ValueError("Поддерживается только одностраничный PDF")
                page = document[0]
                page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72),
                                colorspace=pymupdf.csRGB, alpha=False).save(target)
        else:
            with Image.open(source) as image:
                image.convert("RGB").save(target)
        return target
