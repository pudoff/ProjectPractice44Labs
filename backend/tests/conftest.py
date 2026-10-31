"""Isolated test storage; never connect to the developer's application database."""
import os
from pathlib import Path
import tempfile

TEST_ROOT = Path(tempfile.mkdtemp(prefix="practice44-backend-tests-"))
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", f"sqlite+aiosqlite:///{(TEST_ROOT / 'test.sqlite3').as_posix()}")
os.environ["STORAGE_ROOT"] = str(TEST_ROOT / "storage")
os.environ["REQUIRE_LOCAL_MODELS"] = "false"
