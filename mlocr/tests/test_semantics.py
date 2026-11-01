import unittest

from mlocr.src.semantics import parse_engineering_notation


class EngineeringNotationTests(unittest.TestCase):
    def test_examples_from_task(self):
        expected = {
            "46": ("linear", 46.0),
            "Ø20": ("diameter", 20.0),
            "R2": ("radius", 2.0),
            "46±0,1": ("linear_tolerance", 46.0),
            "M12×1,5": ("thread", 12.0),
            "1:5": ("scale", 1.0),
        }
        for source, (kind, nominal) in expected.items():
            with self.subTest(source=source):
                parsed = parse_engineering_notation(source)
                self.assertEqual(parsed["raw_text"], source)
                self.assertEqual(parsed["kind"], kind)
                self.assertEqual(parsed["nominal"], nominal)
                self.assertIsNone(parsed["warning"])

    def test_thread_pitch_and_class_are_separate_from_length(self):
        parsed = parse_engineering_notation("M12×1,5-H7")
        self.assertEqual(parsed["kind"], "thread")
        self.assertEqual(parsed["nominal"], 12.0)
        self.assertEqual(parsed["thread_pitch"], 1.5)
        self.assertEqual(parsed["thread_class"], "H7")

    def test_unknown_notation_is_preserved_without_correction(self):
        parsed = parse_engineering_notation("Ø2O±O,l")
        self.assertEqual(parsed["raw_text"], "Ø2O±O,l")
        self.assertEqual(parsed["kind"], "unknown")
        self.assertIsNotNone(parsed["warning"])


    def test_thread_class_with_grade_before_position(self):
        parsed = parse_engineering_notation("M56x2-7H")
        self.assertEqual(parsed["kind"], "thread")
        self.assertEqual(parsed["nominal"], 56.0)
        self.assertEqual(parsed["thread_pitch"], 2.0)
        self.assertEqual(parsed["thread_class"], "7H")


    def test_greek_phi_remains_ambiguous(self):
        phi72 = chr(966) + "72"
        parsed = parse_engineering_notation(phi72)
        self.assertEqual(parsed["kind"], "diameter_candidate")
        self.assertEqual(parsed["nominal"], 72.0)
        self.assertEqual(parsed["raw_text"], phi72)
        self.assertIsNotNone(parsed["warning"])

    def test_empty_set_and_cyrillic_ef_remain_ambiguous(self):
        for glyph in (chr(8709), chr(1060), chr(1092)):
            with self.subTest(glyph=glyph):
                parsed = parse_engineering_notation(glyph + "14")
                self.assertEqual(parsed["kind"], "diameter_candidate")
                self.assertEqual(parsed["nominal"], 14.0)
                self.assertIsNotNone(parsed["warning"])

    def test_dedicated_diameter_glyphs_are_definite(self):
        for glyph in ("Ø", chr(8960)):
            with self.subTest(glyph=glyph):
                parsed = parse_engineering_notation(glyph + "20")
                self.assertEqual(parsed["kind"], "diameter")
                self.assertEqual(parsed["nominal"], 20.0)


if __name__ == "__main__":
    unittest.main()
