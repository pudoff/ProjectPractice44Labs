import hashlib
import json
import os
from pathlib import Path
from .config import STORAGE_ROOT
from .models import Artifact


def job_dir(job_id: str) -> Path:
    return STORAGE_ROOT / "jobs" / job_id


def run_dir(job_id: str, number: int) -> Path:
    return job_dir(job_id) / "runs" / f"{number:04d}"


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def register_artifact(session, run, path: Path) -> Artifact:
    data = path.read_bytes()
    artifact = Artifact(run_id=run.id, name=path.name, relative_path=str(path.relative_to(STORAGE_ROOT)), size=len(data), sha256=hashlib.sha256(data).hexdigest())
    session.add(artifact)
    return artifact
