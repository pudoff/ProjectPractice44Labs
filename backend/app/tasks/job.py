import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from app.domain.drawing import DrawingGraph, Issue
from app.services.dxf import DxfExporter
from app.infra.db import SessionLocal
from app.infra.storage import artifact_from_path, atomic_json, job_dir, existing_run_dir
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


async def _fail_run(session, run, out, code, message):
    report_path = out / "report.json"
    try:
        await run_cpu(atomic_json, report_path,
                      {"status": "failed", "error": {"code": code, "message": message}})
    except OSError:
        log.exception("could not write failure report for run %s", run.id)
    if await run_cpu(report_path.is_file) and not any(a.name == "report.json" for a in run.artifacts):
        session.add(await run_cpu(artifact_from_path, run.id, report_path))
    run.state = run.job.state = "failed"
    run.stage = "done"
    run.error_code = code
    run.error_message = message[:2000]
    run.finished_at = now()


async def process_run(run_id: str | None = None) -> str | None:
    async with SessionLocal.begin() as session:
        run = await RunRepository(session).next_pending(run_id)
        if run is None:
            return None
        out = existing_run_dir(run.job_id, run.id, run.number)
        if run.state == "running" or await run_cpu((out / "report.json").exists):
            await _fail_run(session, run, out, "worker_interrupted",
                            "Worker stopped before completion; retry creates a new run")
            return run.id
        run.state = "running"
        run.stage = "recognize"
        run.started_at = now()
        try:
            await run_cpu(atomic_json, out / "report.json", {"status": "running", "stage": "recognize"})
            with tempfile.TemporaryDirectory(prefix="practice44-run-") as temporary:
                await _process_run(session, run, out, Path(temporary))
        except Exception as exc:
            await session.execute(text("SELECT 1"))
            log.exception("run %s failed", run.id)
            await _fail_run(session, run, out, type(exc).__name__, str(exc))
        return run.id


async def _process_run(session, run, out, work):
    job = run.job
    source = job_dir(job.id) / "input" / f"original{job.source_ext}"
    parameters = json.loads(run.parameters)
    if parameters.get("drawing_path"):
        await run_cpu(shutil.copyfile, out / "submitted.json", work / "submitted.json")
    graph = await run_cpu(_recognize, source, work, job.source_sha256, parameters)
    await session.execute(text("SELECT 1"))
    run.stage = "export"
    await run_cpu(atomic_json, out / "report.json", {"status": "running", "stage": "export"})
    state = await run_cpu(_export, graph, work, job.source_sha256)
    await session.execute(text("SELECT 1"))
    for path in work.iterdir():
        if path.is_file() and path.name != "submitted.json":
            if path.name == "report.json":
                report = json.loads(await run_cpu(path.read_text, encoding="utf-8"))
                await run_cpu(atomic_json, out / path.name, report)
            else:
                await run_cpu(shutil.copyfile, path, out / path.name)
    for name in ("cv.json", "ocr.json", "ocr-observations.json", "drawing.json", "result.dxf", "preview.png", "report.json"):
        path = out / name
        if await run_cpu(path.exists):
            session.add(await run_cpu(artifact_from_path, run.id, path))
    run.state = run.job.state = state
    run.stage = "done"
    run.finished_at = now()
