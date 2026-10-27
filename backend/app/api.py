import hashlib
import json
import uuid
from pathlib import Path
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from .config import API_KEY, MAX_UPLOAD_BYTES, STORAGE_ROOT
from .db import SessionLocal
from .drawing import DrawingGraph
from .models import Artifact, Job, Run
from .storage import atomic_bytes, job_dir, run_dir, register_artifact

app = FastAPI(title="Drawing to DXF API", version="0.1.0")


async def auth(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")


async def db():
    with SessionLocal() as session:
        yield session


def visible_job(session: Session, job_id: str, channel: str | None, conversation_id: str | None):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.channel != "web" and not API_KEY:
        raise HTTPException(503, "API_KEY is required for bot jobs")
    if job.channel != "web" and (channel != job.channel or conversation_id != job.conversation_id):
        raise HTTPException(404, "Job not found")
    return job


def status_payload(job: Job):
    runs = [{"id": r.id, "number": r.number, "state": r.state, "stage": r.stage, "error_code": r.error_code, "error_message": r.error_message} for r in job.runs]
    return {"id": job.id, "state": job.state, "filename": job.filename, "source_sha256": job.source_sha256, "runs": runs}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"api": app.version, "drawing_schema": "0.1"}


@app.post("/api/v1/jobs", status_code=201, dependencies=[Depends(auth)])
async def create_job(file: UploadFile = File(...), channel: str = Form("web"), conversation_id: str | None = Form(None), event_id: str | None = Form(None), session: Session = Depends(db)):
    if channel not in {"web", "vk"}:
        raise HTTPException(422, "Unsupported channel")
    if channel != "web" and not API_KEY:
        raise HTTPException(503, "API_KEY is required for bot jobs")
    if channel != "web" and (not conversation_id or not event_id):
        raise HTTPException(422, "Bot jobs require conversation_id and event_id")
    if event_id:
        existing = session.execute(select(Job).where(Job.channel == channel, Job.event_id == event_id)).scalar_one_or_none()
        if existing:
            if existing.conversation_id != conversation_id:
                raise HTTPException(409, "Event already belongs to another conversation")
            return status_payload(existing)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(415, "Expected PDF, PNG or JPEG")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    if not data or not ((ext == ".pdf" and data.startswith(b"%PDF-")) or (ext == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n")) or (ext in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8"))):
        raise HTTPException(415, "File contents do not match format")
    job = Job(filename=Path(file.filename or "original").name[:255], source_sha256=hashlib.sha256(data).hexdigest(), source_size=len(data), source_ext=ext, channel=channel, conversation_id=conversation_id, event_id=event_id)
    run = Run(job=job, number=1, state="queued", stage="queued", parameters="{}")
    session.add(job)
    session.flush()
    path = job_dir(job.id) / "input" / f"original{ext}"
    atomic_bytes(path, data)
    register_artifact(session, run, path)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.execute(select(Job).where(Job.channel == channel, Job.event_id == event_id)).scalar_one_or_none()
        if existing and existing.conversation_id == conversation_id:
            return status_payload(existing)
        raise HTTPException(409, "Duplicate event")
    session.refresh(job)
    return status_payload(job)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(auth)])
async def get_job(job_id: str, x_channel: str | None = Header(None), x_conversation_id: str | None = Header(None), session: Session = Depends(db)):
    return status_payload(visible_job(session, job_id, x_channel, x_conversation_id))


@app.post("/api/v1/jobs/{job_id}/rerun", status_code=202, dependencies=[Depends(auth)])
async def rerun(job_id: str, x_channel: str | None = Header(None), x_conversation_id: str | None = Header(None), session: Session = Depends(db)):
    job = visible_job(session, job_id, x_channel, x_conversation_id)
    session.execute(select(Job).where(Job.id == job.id).with_for_update()).scalar_one()
    session.refresh(job)
    if job.state in {"queued", "running"}:
        raise HTTPException(409, "Current run is still active")
    run = Run(job=job, number=max(r.number for r in job.runs) + 1, state="queued", stage="queued", parameters="{}")
    session.add(run)
    job.state = "queued"
    session.commit()
    return status_payload(job)


@app.post("/api/v1/jobs/{job_id}/drawing", status_code=202, dependencies=[Depends(auth)])
async def submit_drawing(job_id: str, graph: DrawingGraph, x_channel: str | None = Header(None), x_conversation_id: str | None = Header(None), session: Session = Depends(db)):
    job = visible_job(session, job_id, x_channel, x_conversation_id)
    session.execute(select(Job).where(Job.id == job.id).with_for_update()).scalar_one()
    session.refresh(job)
    if job.state in {"queued", "running"}:
        raise HTTPException(409, "Current run is still active")
    if graph.source_sha256 and graph.source_sha256 != job.source_sha256:
        raise HTTPException(422, "source_sha256 differs from original")
    number = max(r.number for r in job.runs) + 1
    path = run_dir(job.id, number) / "submitted.json"
    atomic_bytes(path, graph.model_dump_json(indent=2).encode())
    run = Run(job=job, number=number, state="queued", stage="queued", parameters=json.dumps({"drawing_path": True}))
    session.add(run)
    job.state = "queued"
    session.commit()
    return status_payload(job)


@app.get("/api/v1/jobs/{job_id}/artifacts/{name}", dependencies=[Depends(auth)])
async def get_artifact(job_id: str, name: str, run_number: int | None = None, x_channel: str | None = Header(None), x_conversation_id: str | None = Header(None), session: Session = Depends(db)):
    job = visible_job(session, job_id, x_channel, x_conversation_id)
    run = next((r for r in job.runs if r.number == run_number), None) if run_number else job.runs[-1]
    if run is None:
        raise HTTPException(404, "Run not found")
    artifact = session.execute(select(Artifact).where(Artifact.run_id == run.id, Artifact.name == name)).scalar_one_or_none()
    if artifact is None:
        raise HTTPException(404, "Artifact not ready")
    path = (STORAGE_ROOT / artifact.relative_path).resolve()
    if not path.is_relative_to(STORAGE_ROOT) or not path.is_file():
        raise HTTPException(404, "Artifact missing")
    return Response(path.read_bytes(), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{name}"'})
