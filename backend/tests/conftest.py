import os
import asyncio
import sys
from pathlib import Path
import tempfile

TEST_ROOT = Path(tempfile.mkdtemp(prefix="practice44-backend-tests-"))
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", f"sqlite+aiosqlite:///{(TEST_ROOT / 'test.sqlite3').as_posix()}")
os.environ["STORAGE_ROOT"] = str(TEST_ROOT / "storage")
os.environ["REQUIRE_LOCAL_MODELS"] = "false"


import uuid
import pytest
from sqlalchemy import event
from app.config import get_settings
from app.infra.db import Base, engine
from app.infra.run_lock import claims, release_claim


@event.listens_for(engine.sync_engine, "connect")
def sqlite_uuid(dbapi_connection, _record):
    if engine.dialect.name == "sqlite":
        dbapi_connection.create_function("uuid_generate_v7", 0, lambda: str(uuid.uuid4()))


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def test_auth(monkeypatch):
    monkeypatch.setattr(get_settings(), "auth_tokens", {"alice": "a" * 40, "bob": "b" * 40})


_postgres_prepared = False


@pytest.fixture
async def clean_db(anyio_backend):
    global _postgres_prepared
    if engine.dialect.name == "postgresql" and not _postgres_prepared:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "alembic", "upgrade", "head",
            cwd=Path(__file__).resolve().parents[1],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 0, (stdout.decode(), stderr.decode())
        _postgres_prepared = True
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield
    for run_id in list(claims):
        await release_claim(run_id)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()
