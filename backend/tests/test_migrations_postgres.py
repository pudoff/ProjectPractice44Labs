"""Upgrade a pre-ownership database without assigning legacy jobs to a caller."""
import asyncio
import os
from pathlib import Path
import sys
import uuid

import psycopg
from psycopg import sql
import pytest
from sqlalchemy.engine import make_url

from app.infra.db import engine


@pytest.mark.anyio
@pytest.mark.skipif(engine.dialect.name != "postgresql", reason="requires real PostgreSQL")
async def test_upgrade_preserves_history_and_leaves_legacy_jobs_unowned(tmp_path):
    database = "migration_checks_" + uuid.uuid4().hex
    admin_url = make_url(os.environ["TEST_DATABASE_URL"])
    url = admin_url.set(database=database)

    def connect(target):
        return psycopg.connect(target.set(drivername="postgresql").render_as_string(hide_password=False), autocommit=True)

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "DATABASE_URL": url.render_as_string(hide_password=False),
           "PYTHONPATH": os.pathsep.join(map(str, (root, root / "backend", root / "mlcv")))}

    async def alembic(*args):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "alembic", *args, cwd=root / "backend", env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(30):
            stdout, stderr = await process.communicate()
        assert process.returncode == 0, (stdout.decode(), stderr.decode())

    with connect(admin_url) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        await alembic("upgrade", "8d42f1c9a716")
        artifact = tmp_path / "historical.dxf"
        artifact.write_bytes(b"historical DXF bytes must not change during migration")
        before = artifact.read_bytes()
        with connect(url) as connection:
            job_id = connection.execute("""INSERT INTO jobs(filename,source_sha256,source_ext,state,channel,event_id)
                VALUES ('legacy.pdf',%s,'.pdf','succeeded','web','same-event') RETURNING id""", ("a" * 64,)).fetchone()[0]
            run_id = connection.execute("""INSERT INTO runs(job_id,number,state,stage)
                VALUES (%s,1,'succeeded','done') RETURNING id""", (job_id,)).fetchone()[0]
            connection.execute("INSERT INTO artifacts(run_id,name,relative_path) VALUES (%s,'result.dxf',%s)", (run_id, str(artifact)))
        await alembic("upgrade", "head")
        await alembic("check")
        with connect(url) as connection:
            assert connection.execute("SELECT owner_id,state FROM jobs WHERE id=%s", (job_id,)).fetchone() == (None, "succeeded")
            assert connection.execute("SELECT job_id,number,state FROM runs WHERE id=%s", (run_id,)).fetchone() == (job_id, 1, "succeeded")
            assert connection.execute("SELECT relative_path FROM artifacts WHERE run_id=%s", (run_id,)).fetchone()[0] == str(artifact)
            for owner in ("alice", "bob"):
                connection.execute("""INSERT INTO jobs(owner_id,filename,source_sha256,source_ext,channel,event_id)
                    VALUES (%s,'new.pdf',%s,'.pdf','web','same-event')""", (owner, "b" * 64))
        assert artifact.read_bytes() == before
    finally:
        with connect(admin_url) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
