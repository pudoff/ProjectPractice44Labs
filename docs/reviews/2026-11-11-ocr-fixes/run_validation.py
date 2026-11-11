"""Repeat the review-fix checks in a disposable local environment, without a live database.

python run_validation.py unit --out NEW_DIRECTORY
python run_validation.py cv --out ANOTHER_DIRECTORY
The CV suite uses the repository's existing local ONNX models, without Paddle inference.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=['unit', 'cv'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / 'backend'), str(ROOT / 'mlcv')]
    os.environ['MODEL_DIR'] = str(ROOT / 'models')
    os.environ['STORAGE_ROOT'] = str(out / 'storage')
    os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///' + (out / 'check.sqlite3').as_posix()
    os.environ.pop('TEST_DATABASE_URL', None)  # Never run destructive database tests against an external DB.
    os.environ['REQUIRE_LOCAL_MODELS'] = 'true' if args.suite == 'cv' else 'false'
    if args.suite == 'cv':
        from app.config import get_settings
        from app.infra.model_assets import configure_local_models
        get_settings().ocr_adapter = None  # This suite requires only the local CV detector, not PaddleOCR.
        configure_local_models()
    versions = {}
    for package in ('pytest', 'numpy', 'ezdxf', 'rapidocr', 'onnxruntime', 'paddleocr', 'paddlepaddle'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    hashes = {}
    for folder in ('backend/app', 'mlocr/src', 'mlocr/evaluation', 'mlcv/drawcv', 'mlcv/eval'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    (out / 'environment.json').write_text(json.dumps({'python': sys.version, 'suite': args.suite,
        'git_head': head, 'versions': versions, 'source_sha256': hashes}, indent=2), encoding='utf-8')
    import pytest
    paths = ['backend/tests', 'tests', 'mlocr/tests'] if args.suite == 'unit' else ['mlcv/tests']
    return int(pytest.main(['-q', *paths, '--junitxml=' + str(out / 'pytest.xml')]))


if __name__ == '__main__':
    raise SystemExit(main())
