import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service import models


class TestModels(unittest.TestCase):

    def test_explicit_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MODEL_DIR": directory}):
                actual = models.paths(check=False)
                root = Path(directory).resolve()

                expected = {
                    "det": root / "det.onnx",
                    "rec": root / "rec_eslav.onnx",
                    "cls": root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                    "dict": root / "ppocrv5_eslav_dict.txt",
                }

                self.assertEqual(set(actual), set(expected))

                for name, expected_path in expected.items():
                    with self.subTest(model=name):
                        self.assertEqual(
                            actual[name].resolve(),
                            expected_path.resolve(),
                        )

    def test_missing_models_raise_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MODEL_DIR": directory}):
                with self.assertRaises(FileNotFoundError):
                    models.paths(check=True)

    def test_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.onnx"
            content = b"test-model-content"
            path.write_bytes(content)

            expected_hash = hashlib.sha256(content).hexdigest()

            self.assertEqual(models.digest(path), expected_hash)

    def test_model_integrity_and_rapidocr_params(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            sample_models = {
                "det": ("det.onnx", b"test-detection-model"),
                "rec": ("rec_eslav.onnx", b"test-recognition-model"),
                "cls": (
                    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                    b"test-classification-model",
                ),
            }

            expected_hashes = {}

            for name, (filename, content) in sample_models.items():
                (root / filename).write_bytes(content)
                expected_hashes[name] = (
                    filename,
                    hashlib.sha256(content).hexdigest(),
                )

            dictionary = root / "ppocrv5_eslav_dict.txt"
            dictionary.write_text("A\nB\nC\n", encoding="utf-8")

            with (
                patch.dict(os.environ, {"MODEL_DIR": directory}),
                patch.dict(models.MODEL_FILES, expected_hashes, clear=True),
            ):
                verified = models.paths(check=True)

                for name, (filename, _) in expected_hashes.items():
                    with self.subTest(model=name):
                        self.assertEqual(
                            verified[name].resolve(),
                            (root / filename).resolve(),
                        )

                self.assertEqual(
                    verified["dict"].resolve(),
                    dictionary.resolve(),
                )

                params = models.rapidocr_params()

                self.assertEqual(
                    Path(params["Det.model_path"]).resolve(),
                    (root / "det.onnx").resolve(),
                )
                self.assertEqual(
                    Path(params["Rec.model_path"]).resolve(),
                    (root / "rec_eslav.onnx").resolve(),
                )
                self.assertEqual(
                    Path(params["Cls.model_path"]).resolve(),
                    (
                        root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
                    ).resolve(),
                )
                self.assertEqual(
                    Path(params["Rec.rec_keys_path"]).resolve(),
                    dictionary.resolve(),
                )
                self.assertFalse(params["Global.use_cls"])

                # A modified model must fail SHA256 verification.
                (root / "det.onnx").write_bytes(b"corrupted-model")

                with self.assertRaises(ValueError):
                    models.paths(check=True)


if __name__ == "__main__":
    unittest.main()
