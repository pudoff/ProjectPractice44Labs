import unittest

from mlocr.src.dimension_links import propose_dimension_links
from mlocr.src.pdf_text import build_title_block, merge_pdf_and_ocr


class PdfTextTests(unittest.TestCase):
    def test_merge_preserves_pdf_glyph_and_ocr_disagreement(self):
        pdf = [{"id": "pdf-1", "text": chr(8709) + "14", "poly_source_px": [[0, 0], [20, 0], [20, 8], [0, 8]],
                "notation": {"kind": "diameter_candidate"}, "purpose": "dimension_or_scale_candidate"}]
        ocr = [{"id": "ocr-1", "text": chr(966) + "14", "score": 0.8,
                "poly_source_px": [[1, 0], [20, 0], [20, 8], [1, 8]], "alternatives": []}]
        result = merge_pdf_and_ocr(pdf, ocr)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], chr(8709) + "14")
        self.assertEqual(result[0]["source"], "pdf_text+ocr")
        self.assertEqual(result[0]["ocr_readings"][0]["text"], chr(966) + "14")
        self.assertEqual(result[0]["issues"][0]["code"], "pdf_ocr_disagreement")

    def test_title_block_missing_fields_are_unknown(self):
        page = {"page": "page-1", "width_px": 1000, "height_px": 700, "texts": []}
        fields = build_title_block([page])["page-1"]["fields"]
        self.assertEqual(fields["material"]["status"], "unknown")
        self.assertIsNone(fields["material"]["value"])
        self.assertEqual(fields["authors"]["status"], "unknown")


class DimensionLinkTests(unittest.TestCase):
    def test_matching_pair_is_proposal_and_not_silently_accepted(self):
        page = {"texts": [{"id": "txt-1", "text": "10", "notation": {"kind": "linear", "nominal": 10.0},
                           "poly_source_px": [[4, 0], [6, 0], [6, 2], [4, 2]]}]}
        cv = {"scale_status": "known", "transforms": {"source_px_to_sheet_mm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
              "points": [{"id": "p1", "xy": [0, 0], "refs": [{"object_id": "l1"}]},
                        {"id": "p2", "xy": [10, 0], "refs": [{"object_id": "l2"}]}],
              "objects": {"l1": {"role": "part"}, "l2": {"role": "part"}}}
        link = propose_dimension_links(page, cv)[0]
        self.assertEqual(link["status"], "needs_review")
        self.assertEqual(link["candidate_count"], 1)
        self.assertEqual(link["candidates"][0]["measured_mm"], 10.0)
        self.assertIn("extension-line ownership", link["decision_note"])

    def test_unknown_scale_prevents_link_proposals(self):
        page = {"texts": [{"id": "txt-1", "text": "10", "notation": {"kind": "linear", "nominal": 10.0},
                           "poly_source_px": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}
        cv = {"scale_status": "unknown", "transforms": {"source_px_to_sheet_mm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
        self.assertEqual(propose_dimension_links(page, cv), [])


if __name__ == "__main__":
    unittest.main()
