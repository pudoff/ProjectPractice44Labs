import json
import logging
import time
from pathlib import Path
from sqlalchemy import select
from .config import POLL_SECONDS
from .db import SessionLocal
from .drawing import DrawingGraph, Issue
from .exporter import export_dxf
from .models import Job, Run, now
from .pipeline import recognize
from .storage import atomic_json, job_dir, register_artifact, run_dir

log = logging.getLogger(__name__)


def recover_interrupted_runs():
    with SessionLocal.begin() as session:
        runs = session.execute(select(Run).where(Run.state == "running").with_for_update(skip_locked=True)).scalars().all()
        for run in runs:
            path = run_dir(run.job_id, run.number) / "report.json"
            atomic_json(path, {"status": "failed", "error": {"code": "worker_interrupted", "message": "Worker stopped before completion"}})
            existing = any(a.name == "report.json" for a in run.artifacts)
            if not existing:
                register_artifact(session, run, path)
            run.state = "failed"
            run.job.state = "failed"
            run.stage = "done"
            run.error_code = "worker_interrupted"
            run.error_message = "Worker stopped before completion"
            run.finished_at = now()


def claim_run():
    with SessionLocal.begin() as session:
        run = session.execute(select(Run).where(Run.state == "queued").order_by(Run.created_at, Run.number).with_for_update(skip_locked=True).limit(1)).scalar_one_or_none()
        if run is None:
            return None
        run.state = "running"
        run.stage = "prepare"
        run.started_at = now()
        run.job.state = "running"
        return run.id


def set_stage(run_id: str, stage: str):
    with SessionLocal.begin() as session:
        session.get(Run, run_id).stage = stage


def process_run(run_id: str):
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        job = run.job
        source = job_dir(job.id) / "input" / f"original{job.source_ext}"
        out = run_dir(job.id, run.number)
        source_hash = job.source_sha256
        parameters = json.loads(run.parameters)
    out.mkdir(parents=True, exist_ok=True)
    try:
        set_stage(run_id, "recognize")
        if parameters.get("drawing_path"):
            graph = DrawingGraph.model_validate_json((out / "submitted.json").read_text())
            if graph.source_sha256 and graph.source_sha256 != source_hash:
                raise ValueError("DrawingGraph source_sha256 differs from original")
        else:
            graph = recognize(source, out, source_hash)
        atomic_json(out / "drawing.json", graph.model_dump())
        set_stage(run_id, "export")
        issues = export_dxf(graph, out / "result.dxf")
        # Preview is generated from the output DXF when available.
        if (out / "result.dxf").exists():
            try:
                from .preview import render_preview
                render_preview(out / "result.dxf", out / "preview.png")
            except Exception as exc:
                issues.append(Issue(code="preview_error", severity="warning", message=f"Предпросмотр не создан: {exc}"))
        state = "needs_review" if issues else "succeeded"
        report = {"status": state, "schema_version": graph.schema_version, "issues": [i.model_dump() for i in issues], "source_sha256": source_hash}
        atomic_json(out / "report.json", report)
        with SessionLocal.begin() as session:
            run = session.get(Run, run_id)
            for name in ("cv.json", "ocr.json", "drawing.json", "result.dxf", "preview.png", "report.json"):
                path = out / name
                if path.exists():
                    register_artifact(session, run, path)
            run.state = state
            run.job.state = state
            run.stage = "done"
            run.schema_version = graph.schema_version
            run.finished_at = now()
    except Exception as exc:
        log.exception("run %s failed", run_id)
        atomic_json(out / "report.json", {"status": "failed", "error": {"code": type(exc).__name__, "message": str(exc)}})
        with SessionLocal.begin() as session:
            run = session.get(Run, run_id)
            register_artifact(session, run, out / "report.json")
            run.state = "failed"
            run.job.state = "failed"
            run.stage = "done"
            run.error_code = type(exc).__name__
            run.error_message = str(exc)[:2000]
            run.finished_at = now()


def main():
    logging.basicConfig(level=logging.INFO)
    recover_interrupted_runs()
    while True:
        run_id = claim_run()
        if run_id:
            process_run(run_id)
        else:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
