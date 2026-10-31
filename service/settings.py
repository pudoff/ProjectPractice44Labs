"""Config paths come from dedicated persistent directories, never library caches."""
import os
from pathlib import Path

def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "./storage")).expanduser().resolve()

def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "./models")).expanduser().resolve()

def db_dsn() -> str:
    value = os.environ.get("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL must be set")
    return value

def max_upload_bytes() -> int:
    return int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
