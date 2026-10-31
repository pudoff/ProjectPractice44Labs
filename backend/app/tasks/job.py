import asyncio
import json
import logging
from datetime import datetime, timezone

from app.domain.drawing import DrawingGraph, Issue
from app.services.dxf import DxfExporter
from app.infra.db import SessionLocal
from app.infra.storage import artifact_from_path, atomic_json, job_dir, existing_run_dir
from app.infra.models import Run
from app.repositories.jobs import RunRepository
from app.services.recognition import recognize

log = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(timezone.utc)


async def recover_interrupted_runs() -> None:
    async with SessionLocal.begin() as session:
        runs = await RunRepository(session).running_for_recovery()
        for run in runs:
            path = existing_run_dir(run.job_id, run.id, run.number) / "report.json"
            await asyncio.to_thread(atomic_json, path, {"status": "failed", "error": {"code": "worker_interrupted", "message": "Worker stopped before completion"}})
            if not any(a.name == "report.json" for a in run.artifacts):
                session.add(await asyncio.to_thread(artifact_from_path, run.id, path))
            run.state = run.job.state = "failed"
            run.stage = "done"
            run.error_code = "worker_interrupted"
            run.error_message = "Worker stopped before completion"
            run.finished_at = now()


async def claim_run() -> str | None:
    async with SessionLocal.begin() as session:
        run = await RunRepository(session).next_queued()
        if run is None:
            return None
        run.state = run.job.state = "running"
        run.stage = "prepare"
        run.started_at = now()
        return run.id


async def set_stage(run_id: str, stage: str) -> None:
    async with SessionLocal.begin() as session:
        run = await RunRepository(session).get(run_id)
        run.stage = stage


def _recognize(source, out, source_hash, parameters):
    if parameters.get("drawing_path"):
        graph = DrawingGraph.model_validate_json((out / "submitted.json").read_text())
        if graph.source_sha256 and graph.source_sha256 != source_hash:
            raise ValueError("DrawingGraph source_sha256 differs from original")
    else:
        graph = recognize(source, out, source_hash)
    atomic_json(out / "drawing.json", graph.model_dump())
    return graph


def _export(graph, out, source_hash):
    issues = DxfExporter().export(graph, out / "result.dxf")
    if (out / "result.dxf").exists():
        try:
            from app.services.preview import render_preview
            render_preview(out / "result.dxf", out / "preview.png")
        except Exception as exc:
            issues.append(Issue(code="preview_error", severity="warning", message=f"Предпросмотр не создан: {exc}"))
    state = "needs_review" if issues else "succeeded"
    atomic_json(out / "report.json", {"status": state, "schema_version": graph.schema_version,
                                      "issues": [i.model_dump() for i in issues], "source_sha256": source_hash})
    return state


async def process_run(run_id: str) -> None:
    async with SessionLocal() as session:
        run = await RunRepository(session).get(run_id)
        job = run.job
        source = job_dir(job.id) / "input" / f"original{job.source_ext}"
        out = existing_run_dir(job.id, run.id, run.number)
        source_hash = job.source_sha256
        parameters = json.loads(run.parameters)
    try:
        await asyncio.to_thread(out.mkdir, parents=True, exist_ok=True)
        await set_stage(run_id, "recognize")
        graph = await asyncio.to_thread(_recognize, source, out, source_hash, parameters)
        await set_stage(run_id, "export")
        state = await asyncio.to_thread(_export, graph, out, source_hash)
        async with SessionLocal.begin() as session:
            run = await RunRepository(session).get(run_id)
            for name in ("cv.json", "ocr.json", "drawing.json", "result.dxf", "preview.png", "report.json"):
                path = out / name
                if await asyncio.to_thread(path.exists):
                    session.add(await asyncio.to_thread(artifact_from_path, run.id, path))
            run.state = run.job.state = state
            run.stage = "done"
            run.finished_at = now()
    except Exception as exc:
        log.exception("run %s failed", run_id)
        report_path = out / "report.json"
        try:
            await asyncio.to_thread(atomic_json, report_path,
                                    {"status": "failed", "error": {"code": type(exc).__name__, "message": str(exc)}})
        except OSError:
            log.exception("could not write failure report for run %s", run_id)
        async with SessionLocal.begin() as session:
            run = await RunRepository(session).get(run_id)
            if await asyncio.to_thread(report_path.is_file):
                session.add(await asyncio.to_thread(artifact_from_path, run.id, report_path))
            run.state = run.job.state = "failed"
            run.stage = "done"
            run.error_code = type(exc).__name__
            run.error_message = str(exc)[:2000]
            run.finished_at = now()
