"""A PDF with a hidden text layer must not override the visible image."""
import tempfile
import unittest
from pathlib import Path

import pymupdf

from mlocr.src.pdf_text import extract_pdf_text_page


def _make_pdf(path: Path, hidden_mode: str, overlap: bool):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((40, 100), 'CT3', fontsize=20)
    point = (40, 100) if overlap else (40, 160)
    if hidden_mode == 'render_mode':
        page.insert_text(point, 'FAKE99', fontsize=20, render_mode=3)
    else:
        page.insert_text(point, 'FAKE99', fontsize=20, fill_opacity=0)
    doc.save(path)
    doc.close()


class HiddenTextLayerTests(unittest.TestCase):
    def _extract(self, mode, overlap):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / 'hidden.pdf'
            _make_pdf(pdf, mode, overlap)
            with pymupdf.open(pdf) as doc:
                pix = doc[0].get_pixmap(dpi=150)
                width, height = pix.width, pix.height
            return extract_pdf_text_page(pdf, 0, 150, width, height)

    def test_hidden_text_elsewhere_is_ignored_and_visible_text_kept(self):
        for mode in ('render_mode', 'opacity'):
            rows = self._extract(mode, overlap=False)
            joined = ' '.join(row['text'] for row in rows)
            self.assertIn('CT3', joined)
            self.assertNotIn('FAKE99', joined)

    def test_hidden_text_over_the_visible_line_drops_the_whole_line(self):
        # An invisible span exactly over a visible word means the text layer is untrustworthy
        # for that line: the PDF item is dropped and the raster OCR reading is used instead.
        for mode in ('render_mode', 'opacity'):
            rows = self._extract(mode, overlap=True)
            joined = ' '.join(row['text'] for row in rows)
            self.assertNotIn('FAKE99', joined)
            self.assertNotIn('CT3', joined)


if __name__ == '__main__':
    unittest.main()
