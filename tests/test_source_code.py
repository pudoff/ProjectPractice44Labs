import ast
import pathlib
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
class TestSourceCode(unittest.TestCase):
    def test_python_parse(self):
        for path in [*ROOT.glob("service/*.py"), *ROOT.glob("tools/*.py"),
                     *ROOT.glob("runs/klapan_png2dxf/pipeline_b/*.py")]:
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    def test_python_pin(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))
        self.assertEqual((ROOT / ".python-version").read_text().strip(), "3.12.8")
        for name, pin in config["tool"]["poetry"]["dependencies"].items():
            if name == "python":
                continue
            ver = pin if isinstance(pin, str) else pin["version"]
            self.assertTrue(ver.startswith("=="), (name, ver))
    def test_no_tempdir_path_for_model_assets(self):
        source = (ROOT / "runs/klapan_png2dxf/pipeline_b/ocr.py").read_text()
        self.assertIn("rapidocr_params()", source)
        self.assertNotIn("tempfile", source)

if __name__ == "__main__":
    unittest.main()
