"""Verified Paddle assets. Downloading is an explicit preparation operation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

MANIFEST = Path(__file__).resolve().parents[1] / 'models/MANIFEST.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_root():
    return Path(os.getenv('DRAWING_OCR_MODEL_DIR', str(Path(os.getenv('MODEL_DIR', 'models')) / 'paddle')))


def verify(directory, specification):
    for name, expected in specification['files'].items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or invalid OCR model: {path}; run tools.fetch_models --download on the preparation machine')


def local_model_dirs(det=None, rec=None):
    specs = json.loads(MANIFEST.read_text(encoding='utf-8'))['models']
    paths = []
    for override, spec in zip((det, rec), specs):
        directory = Path(override) if override else model_root() / spec['directory']
        verify(directory, spec)
        # Paddle's native Windows reader cannot open some Unicode paths. Copy
        # verified files to a content-addressed local cache; no network fallback.
        if os.name == 'nt' and not str(directory.resolve()).isascii():
            cache = Path(os.getenv('DRAWING_OCR_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'practice44-ocr-models')))
            if not str(cache.resolve()).isascii():
                raise ValueError('Set DRAWING_OCR_CACHE_DIR to an ASCII Windows path')
            target = cache / spec['archive_sha256'].lower()
            target.mkdir(parents=True, exist_ok=True)
            for name, expected in spec['files'].items():
                final = target / name
                if final.is_file() and digest(final) == expected:
                    continue
                with tempfile.NamedTemporaryFile(dir=target, delete=False) as tmp:
                    temporary = Path(tmp.name)
                try:
                    shutil.copyfile(directory / name, temporary)
                    os.replace(temporary, final)
                finally:
                    temporary.unlink(missing_ok=True)
            verify(target, spec)
            directory = target
        paths.append(directory.resolve())
    return paths


def prepare(*, download=False):
    root = model_root()
    specs = json.loads(MANIFEST.read_text(encoding='utf-8'))['models']
    for spec in specs:
        target = root / spec['directory']
        try:
            verify(target, spec)
            continue
        except ValueError:
            if not download:
                raise
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            archive = Path(temp) / 'model.tar'
            with urllib.request.urlopen(spec['source'], timeout=90) as response, archive.open('wb') as out:
                shutil.copyfileobj(response, out)
            if digest(archive) != spec['archive_sha256'].lower():
                raise ValueError('OCR archive SHA256 mismatch')
            with tarfile.open(archive) as stream:
                stream.extractall(Path(temp) / 'unpacked', filter='data')
            unpacked = Path(temp) / 'unpacked' / spec['directory']
            verify(unpacked, spec)
            target.mkdir(parents=True, exist_ok=True)
            for name in spec['files']:
                shutil.copyfile(unpacked / name, target / name)
        verify(target, spec)
    return {spec['name']: spec['files'] for spec in specs}
