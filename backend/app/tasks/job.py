import asyncio
import json
import logging
import os
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

from app.domain.drawing import DrawingGraph, Issue
from app.services.dxf import DxfExporter
from app.infra.db import SessionLocal
from app.infra.storage import artifact_from_path, atomic_json, job_dir, existing_run_dir
from app.infra.run_lock import RunLock, RunLockLost, claims, processing, release_claim
from app.repositories.jobs import RunRepository
from app.services.recognition import recognize
from app.services.geometry import geometry_signature

log = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(timezone.utc)


async def run_cpu(function, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                break
        except Exception:
            if not cancelled:
                raise
            log.exception("Thread operation failed during worker shutdown")
            break
    if cancelled:
        if task.done() and not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return result


async def recover_interrupted_runs() -> None:
    async with SessionLocal.begin() as session:
        runs = await RunRepository(session).running_for_recovery()
        for run in runs:
            lock = await RunLock.acquire(run.id)
            if lock is None:
                continue  # Another live worker owns this run.
            try:
                path = existing_run_dir(run.job_id, run.id, run.number) / "report.json"
                await run_cpu(atomic_json, path, {"status": "failed", "error": {"code": "worker_interrupted", "message": "Worker stopped before completion; retry creates a new run"}})
                if not any(a.name == "report.json" for a in run.artifacts):
                    session.add(await run_cpu(artifact_from_path, run.id, path))
                run.state = run.job.state = "failed"
                run.stage = "done"
                run.error_code = "worker_interrupted"
                run.error_message = "Worker stopped before completion; retry creates a new run"
                run.finished_at = now()
                await session.flush()
            finally:
                await lock.release()


async def claim_run() -> str | None:
    lock = None
    try:
        async with SessionLocal.begin() as session:
            run = await RunRepository(session).next_queued()
            if run is None:
                return None
            lock = await RunLock.acquire(run.id)
            if lock is None:
                return None
            run.state = run.job.state = "running"
            run.stage = "prepare"
            run.started_at = now()
        claims[run.id] = lock
        return run.id
    except BaseException:
        if lock:
            await lock.release()
        raise


async def _owned_run(session, run_id):
    run = await RunRepository(session).get_for_update(run_id)
    await claims[run_id].assert_owned()
    if run.state != "running":
        raise RunLockLost(run_id)
    return run


async def set_stage(run_id: str, stage: str) -> None:
    async with SessionLocal.begin() as session:
        run = await _owned_run(session, run_id)
        run.stage = stage


def _recognize(source, out, source_hash, parameters):
    if parameters.get("drawing_path"):
        graph = DrawingGraph.model_validate_json((out / "submitted.json").read_text(encoding="utf-8"))
        if graph.source_sha256 and graph.source_sha256 != source_hash:
            raise ValueError("DrawingGraph source_sha256 differs from original")
    else:
        graph = recognize(source, out, source_hash)
    graph.provenance.setdefault("dimension_geometry_sha256", geometry_signature(graph))
    atomic_json(out / "drawing.json", graph.model_dump())
    return graph


def _export(graph, out, source_hash):
    started = time.perf_counter()
    issues = DxfExporter().export(graph, out / "result.dxf")
    if (out / "result.dxf").exists():
        try:
            from app.services.preview import render_preview
            render_preview(out / "result.dxf", out / "preview.png")
        except Exception as exc:
            issues.append(Issue(code="preview_error", severity="warning", message=f"Предпросмотр не создан: {exc}"))
    state = "needs_review" if issues else "succeeded"
    atomic_json(out / "report.json", {"status": state, "schema_version": graph.schema_version,
                                      "issues": [i.model_dump() for i in issues], "source_sha256": source_hash,
                                      "export_profile": graph.export_profile, "build_revision": os.getenv("BUILD_REVISION", "unknown"),
                                      "provenance": graph.provenance, "export_seconds": time.perf_counter() - started})
    return state


async def process_run(run_id: str) -> None:
    if run_id not in claims or run_id in processing:
        return
    processing.add(run_id)
    try:
        with tempfile.TemporaryDirectory(prefix="practice44-run-") as temporary:
            await _process_claimed_run(run_id, Path(temporary))
    except RunLockLost:
        log.warning("run %s lost ownership; discarding computed output", run_id)
    finally:
        processing.discard(run_id)
        await release_claim(run_id)


async def _process_claimed_run(run_id: str, work: Path) -> None:
    async with SessionLocal() as session:
        run = await RunRepository(session).get(run_id)
        job = run.job
        source = job_dir(job.id) / "input" / f"original{job.source_ext}"
        out = existing_run_dir(job.id, run.id, run.number)
        source_hash = job.source_sha256
        parameters = json.loads(run.parameters)
    try:
        if parameters.get("drawing_path"):
            await run_cpu(shutil.copyfile, out / "submitted.json", work / "submitted.json")
        await set_stage(run_id, "recognize")
        graph = await run_cpu(_recognize, source, work, source_hash, parameters)
        await set_stage(run_id, "export")
        state = await run_cpu(_export, graph, work, source_hash)
        async with SessionLocal.begin() as session:
            run = await _owned_run(session, run_id)
            await run_cpu(out.mkdir, parents=True, exist_ok=True)
            for path in work.iterdir():
                if path.is_file() and path.name != "submitted.json":
                    await run_cpu(shutil.copyfile, path, out / path.name)
            for name in ("cv.json", "ocr.json", "ocr-observations.json", "drawing.json", "result.dxf", "preview.png", "report.json"):
                path = out / name
                if await run_cpu(path.exists):
                    session.add(await run_cpu(artifact_from_path, run.id, path))
            run.state = run.job.state = state
            run.stage = "done"
            run.finished_at = now()
    except RunLockLost:
        raise
    except Exception as exc:
        log.exception("run %s failed", run_id)
        async with SessionLocal.begin() as session:
            run = await _owned_run(session, run_id)
            report_path = out / "report.json"
            try:
                await run_cpu(atomic_json, report_path,
                                        {"status": "failed", "error": {"code": type(exc).__name__, "message": str(exc)}})
            except OSError:
                log.exception("could not write failure report for run %s", run_id)
            if await run_cpu(report_path.is_file):
                session.add(await run_cpu(artifact_from_path, run.id, report_path))
            run.state = run.job.state = "failed"
            run.stage = "done"
            run.error_code = type(exc).__name__
            run.error_message = str(exc)[:2000]
            run.finished_at = now()
