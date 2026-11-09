import asyncio
import json
from pathlib import Path
import sys
import threading

import ezdxf
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from app.infra.db import SessionLocal, engine
from app.infra.models import Artifact, Job, Run
from app.infra.storage import run_dir, storage_root
from app.tasks.job import process_run
from test_access import request, upload
from tools.smoke_backend import drawing

pytestmark = [pytest.mark.anyio, pytest.mark.usefixtures("clean_db"),
              pytest.mark.skipif(engine.dialect.name != "postgresql", reason="requires real PostgreSQL")]

CHILD = r'''
import asyncio, json, sys
from pathlib import Path
from app.tasks import job
from app.infra.db import engine

async def wait(path):
    while not path.exists():
        await asyncio.sleep(.02)

async def main():
    marker, start, finish = map(Path, sys.argv[1:4])
    marker.with_suffix('.ready').touch()
    await wait(start)
    original = job._process_run
    async def paused(session, run, out, work):
        marker.write_text(json.dumps({'run_id': run.id}))
        if len(sys.argv) > 4 and sys.argv[4] == 'pause-after-export':
            import time
            export = job._export
            def paused_export(*args):
                state = export(*args)
                marker.with_suffix('.exported').touch()
                while not finish.exists():
                    time.sleep(.02)
                return state
            job._export = paused_export
        else:
            await wait(finish)
        await original(session, run, out, work)
    job._process_run = paused
    run_id = await job.process_run()
    if not marker.exists():
        marker.write_text(json.dumps({'run_id': run_id}))
    await engine.dispose()

asyncio.run(main())
'''


async def wait_file(path, timeout=30):
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(.02)


async def worker(marker, start, finish, *args):
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", CHILD, str(marker), str(start), str(finish), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )


async def stop(*processes):
    for process in processes:
        if process.returncode is None:
            process.kill()
    for process in processes:
        await process.communicate()


async def succeeded(process):
    async with asyncio.timeout(30):
        stdout, stderr = await process.communicate()
    assert process.returncode == 0, (stdout.decode(), stderr.decode())


async def manual_job(width=46):
    job = (await upload()).json()
    await process_run()  # A deliberately invalid input -> failed initial run.
    response = await request("POST", f"/api/v1/jobs/{job['id']}/drawing", json=drawing(width))
    assert response.status_code == 202, response.text
    return response.json()


async def test_two_simultaneous_workers_claim_once_and_startup_preserves_live_run(tmp_path):
    job = await manual_job()
    run_id = job["runs"][-1]["id"]
    start, finish = tmp_path / "start", tmp_path / "finish"
    markers = [tmp_path / "w1.json", tmp_path / "w2.json"]
    processes = [await worker(marker, start, finish) for marker in markers]
    try:
        for marker in markers:
            await wait_file(marker.with_suffix(".ready"))
        start.touch()
        for marker in markers:
            await wait_file(marker)
        claimed = [json.loads(marker.read_text())["run_id"] for marker in markers]
        assert claimed.count(run_id) == 1 and claimed.count(None) == 1
        # A third worker's startup/recovery must not fail the live worker's run.
        await process_run()
        async with SessionLocal() as session:
            run = await session.get(Run, run_id)
            assert run.state == "queued" and run.error_code is None
            assert run.job.state == "queued"
        status = (await request("GET", f"/api/v1/jobs/{job['id']}")).json()
        assert status["state"] == status["runs"][-1]["state"] == "running"
        assert (await request("POST", f"/api/v1/jobs/{job['id']}/rerun")).status_code == 409
        assert not (run_dir(job["id"], run_id) / "result.dxf").exists()
        finish.touch()
        for process in processes:
            await succeeded(process)
        async with SessionLocal() as session:
            run = await session.get(Run, run_id)
            assert run.state == "succeeded" and run.finished_at
            names = [artifact.name for artifact in run.artifacts]
            assert len(names) == len(set(names))
            assert {"result.dxf", "drawing.json", "report.json"} <= set(names)
        target = run_dir(job["id"], run_id) / "result.dxf"
        assert ezdxf.readfile(target).modelspace().query("DIMENSION")[0].get_measurement() == 46
    finally:
        await stop(*processes)


async def test_worker_killed_after_file_write_restart_and_retry_preserve_history(tmp_path):
    job = await manual_job()
    await process_run()
    job_id = job["id"]
    baseline_id = job["runs"][-1]["id"]
    baseline = run_dir(job_id, baseline_id) / "result.dxf"
    baseline_bytes = baseline.read_bytes()
    response = await request("POST", f"/api/v1/jobs/{job_id}/drawing", json=drawing(47))
    failed_id = response.json()["runs"][-1]["id"]
    start, finish, marker = tmp_path / "start", tmp_path / "finish", tmp_path / "killed.json"
    start.touch()
    process = await worker(marker, start, finish, "pause-after-export")
    try:
        await wait_file(marker.with_suffix(".exported"))
        failed_path = run_dir(job_id, failed_id) / "result.dxf"
        assert not failed_path.exists()  # CPU output is private until publication.
        await process_run()
        async with SessionLocal() as session:
            assert (await session.get(Run, failed_id)).state == "queued"
        process.kill()  # An abrupt worker exit, not an exception simulated in this process.
        await process.communicate()
        # A fresh service process executes the same recovery used by app.worker.
        restart = await worker(tmp_path / "restart.json", start, finish)
        try:
            await succeeded(restart)
        finally:
            await stop(restart)
        async with SessionLocal() as session:
            run = await session.get(Run, failed_id)
            assert run.state == run.job.state == "failed"
            assert run.error_code == "worker_interrupted" and run.finished_at
            assert "retry" in run.error_message
            assert "result.dxf" not in {a.name for a in run.artifacts}
            report = next(a for a in run.artifacts if a.name == "report.json")
            assert json.loads((storage_root() / report.relative_path).read_text())["error"]["code"] == "worker_interrupted"
        # Recovery is idempotent and a retry creates a different output directory.
        await process_run()
        response = await request("POST", f"/api/v1/jobs/{job_id}/drawing", json=drawing(48))
        assert response.status_code == 202, response.text
        retry_id = response.json()["runs"][-1]["id"]
        finish.touch()
        retry = await worker(tmp_path / "retry.json", start, finish)
        try:
            await succeeded(retry)
        finally:
            await stop(retry)
        assert baseline.read_bytes() == baseline_bytes
        assert not failed_path.exists()
        target = run_dir(job_id, retry_id) / "result.dxf"
        assert target != failed_path != baseline
        assert ezdxf.readfile(target).modelspace().query("DIMENSION")[0].get_measurement() == 48
        old = await request("GET", f"/api/v1/jobs/{job_id}/artifacts/result.dxf?run_id={baseline_id}")
        assert old.content == baseline_bytes
        assert (await request("GET", f"/api/v1/jobs/{job_id}/artifacts/result.dxf?run_id={failed_id}")).status_code == 404
        async with SessionLocal() as session:
            job = await session.get(Job, job_id)
            assert [run.state for run in job.runs] == ["failed", "succeeded", "failed", "succeeded"]
            assert len({run.id for run in job.runs}) == 4
            paths = (await session.execute(select(Artifact.relative_path).where(Artifact.name == "result.dxf"))).scalars().all()
            assert len(paths) == len(set(paths)) == 2
            assert all((storage_root() / path).is_file() for path in paths)
    finally:
        await stop(process)


async def test_graceful_cancellation_keeps_row_transaction_until_cpu_thread_finishes(monkeypatch):
    from app.tasks import job as tasks
    job = await manual_job()
    run_id = job["runs"][-1]["id"]
    started, finish = threading.Event(), threading.Event()
    original = tasks._export

    def blocked_export(*args):
        started.set()
        if not finish.wait(30):
            raise TimeoutError("test did not release CPU worker")
        return original(*args)

    monkeypatch.setattr(tasks, "_export", blocked_export)
    task = asyncio.create_task(process_run(run_id))
    try:
        async with asyncio.timeout(20):
            while not started.is_set():
                await asyncio.sleep(.02)
        # A duplicate invocation cannot run a second exporter.
        await process_run(run_id)
        task.cancel()
        await asyncio.sleep(.05)
        assert not task.done()
        await process_run()
        async with SessionLocal() as session:
            assert (await session.get(Run, run_id)).state == "queued"
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    await process_run()
    async with SessionLocal() as session:
        run = await session.get(Run, run_id)
        assert run.state == "failed" and run.error_code == "worker_interrupted"
    assert not (run_dir(job["id"], run_id) / "result.dxf").exists()


async def test_terminated_transaction_cannot_publish_after_retry(monkeypatch):
    from app.tasks import job as tasks
    job = await manual_job()
    run_id = job["runs"][-1]["id"]
    pid = None
    original_process = tasks._process_run

    async def capture_backend(session, *args):
        nonlocal pid
        pid = await session.scalar(text('SELECT pg_backend_pid()'))
        return await original_process(session, *args)

    monkeypatch.setattr(tasks, '_process_run', capture_backend)
    started, finish = threading.Event(), threading.Event()
    original = tasks._export

    def blocked_export(*args):
        state = original(*args)
        started.set()
        if not finish.wait(30):
            raise TimeoutError('CPU worker was not released')
        return state

    monkeypatch.setattr(tasks, '_export', blocked_export)
    task = asyncio.create_task(process_run(run_id))
    try:
        async with asyncio.timeout(20):
            while not started.is_set():
                await asyncio.sleep(.02)
        async with engine.begin() as connection:
            assert await connection.scalar(text('SELECT pg_terminate_backend(:pid)'), {'pid': pid})
        await process_run()
        response = await request('POST', f"/api/v1/jobs/{job['id']}/drawing", json=drawing(48))
        assert response.status_code == 202
        retry_id = response.json()['runs'][-1]['id']
        finish.set()
        with pytest.raises(SQLAlchemyError):
            await task
        monkeypatch.setattr(tasks, '_export', original)
        await process_run()
        async with SessionLocal() as session:
            old = await session.get(Run, run_id)
            assert old.state == 'failed' and old.error_code == 'worker_interrupted'
            assert [a.name for a in old.artifacts] == ['report.json']
            retry = await session.get(Run, retry_id)
            assert retry.state == retry.job.state == 'succeeded'
        assert not (run_dir(job['id'], run_id) / 'result.dxf').exists()
        result = run_dir(job['id'], retry_id) / 'result.dxf'
        assert ezdxf.readfile(result).modelspace().query('DIMENSION')[0].get_measurement() == 48
    finally:
        finish.set()
        if not task.done():
            with pytest.raises(SQLAlchemyError):
                await task


@pytest.mark.parametrize("publication", ["success", "failure_report"])
async def test_cancel_during_publication_drains_writes_before_recovery(monkeypatch, publication):
    from app.tasks import job as tasks

    job = await manual_job()
    await process_run()
    job_id = job['id']
    baseline_id = job['runs'][-1]['id']
    baseline = run_dir(job_id, baseline_id) / 'result.dxf'
    baseline_bytes = baseline.read_bytes()
    response = await request('POST', f'/api/v1/jobs/{job_id}/drawing', json=drawing(47))
    assert response.status_code == 202
    run_id = response.json()["runs"][-1]["id"]
    started, finish, written = threading.Event(), threading.Event(), threading.Event()
    original_json = tasks.atomic_json

    def delay(write, *args):
        started.set()
        if not finish.wait(30):
            raise TimeoutError('publication was not released')
        result = write(*args)
        written.set()
        return result

    def publication_json(path, value):
        status = 'succeeded' if publication == 'success' else 'failed'
        if Path(path) == run_dir(job_id, run_id) / 'report.json' and value.get('status') == status:
            return delay(original_json, path, value)
        return original_json(path, value)

    monkeypatch.setattr(tasks, 'atomic_json', publication_json)
    if publication == 'failure_report':
        def fail_export(*args):
            raise ValueError('forced export failure')
        monkeypatch.setattr(tasks, '_export', fail_export)
    task = asyncio.create_task(process_run(run_id))
    try:
        async with asyncio.timeout(20):
            while not started.is_set():
                await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.05)
        task.cancel()  # A second shutdown request must also preserve the lock.
        await asyncio.sleep(.05)
        assert not task.done()
        assert await process_run(run_id) is None
        await process_run()
        async with SessionLocal() as session:
            assert (await session.get(Run, run_id)).state == 'queued'
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert written.is_set()
    monkeypatch.setattr(tasks, 'atomic_json', original_json)
    await process_run()
    await process_run()
    async with SessionLocal() as session:
        run = await session.get(Run, run_id)
        assert run.state == run.job.state == 'failed'
        assert run.error_code == 'worker_interrupted'
        assert [a.name for a in run.artifacts] == ['report.json']
    report = await request('GET', f'/api/v1/jobs/{job_id}/artifacts/report.json?run_id={run_id}')
    assert report.status_code == 200
    assert report.json()['status'] == 'failed'
    assert report.json()['error']['code'] == 'worker_interrupted'
    for name in ('result.dxf', 'drawing.json', 'preview.png'):
        assert (await request('GET', f'/api/v1/jobs/{job_id}/artifacts/{name}?run_id={run_id}')).status_code == 404
    assert baseline.read_bytes() == baseline_bytes
    previous = await request('GET', f'/api/v1/jobs/{job_id}/artifacts/result.dxf?run_id={baseline_id}')
    assert previous.content == baseline_bytes
