import json
import os
import tempfile
from pathlib import Path

from app.config import get_settings
from app.infra.models import Artifact


def storage_root() -> Path:
    return get_settings().storage_root.resolve()


def job_dir(job_id: str) -> Path:
    return storage_root() / "jobs" / job_id


def run_dir(job_id: str, run_id: str) -> Path:
    return job_dir(job_id) / "runs" / run_id


def existing_run_dir(job_id: str, run_id: str, number: int) -> Path:
    """Keep pre-refactor runs readable without moving stored artifacts."""
    path = run_dir(job_id, run_id)
    legacy = job_dir(job_id) / "runs" / f"{number:04d}"
    return legacy if not path.exists() and legacy.exists() else path


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def artifact_from_path(run_id: str, path: Path) -> Artifact:
    return Artifact(run_id=run_id, name=path.name, relative_path=str(path.relative_to(storage_root())))
