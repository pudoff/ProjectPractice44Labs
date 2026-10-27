import os
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://draw:draw@localhost:5432/draw")
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "var/storage")).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2"))
API_KEY = os.getenv("API_KEY")
