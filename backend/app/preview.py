from pathlib import Path
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.layout import Page, Settings
from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend


def render_preview(dxf_path: Path, png_path: Path):
    doc = ezdxf.readfile(dxf_path)
    backend = PyMuPdfBackend()
    Frontend(RenderContext(doc), backend).draw_layout(doc.modelspace(), finalize=True)
    png_path.write_bytes(backend.get_pixmap_bytes(page=Page(0, 0), settings=Settings(), fmt="png"))
