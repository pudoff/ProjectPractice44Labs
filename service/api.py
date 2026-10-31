"""Small API for the existing PNG/JPG -> DXF prototype."""
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import db
from .settings import data_dir, max_upload_bytes

LOG = logging.getLogger(__name__)
SUPPORTED = {".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    (data_dir() / "uploads").mkdir(parents=True, exist_ok=True)
    (data_dir() / "results").mkdir(parents=True, exist_ok=True)
    db.setup()
    yield

app = FastAPI(title="Practice44 PNG→DXF", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/jobs", status_code=202)
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "input").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(415, "Only PNG/JPEG supported by current raster pipeline")
    job_id = uuid.uuid4()
    input_path = data_dir() / "uploads" / f"{job_id}{ext}"
    output_path = data_dir() / "results" / f"{job_id}.dxf"
    report_path = data_dir() / "results" / f"{job_id}.json"
    total = 0
    try:
        with input_path.open("wb") as sink:
            first = await file.read(1024 * 1024)
            if not first.startswith(SUPPORTED[ext]):
                raise HTTPException(415, "File contents do not match extension")
            sink.write(first)
            total += len(first)
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_upload_bytes():
                    raise HTTPException(413, "File too large")
                sink.write(chunk)
        if total > max_upload_bytes():
            raise HTTPException(413, "File too large")
        db.enqueue(job_id, name, str(input_path), str(output_path), str(report_path))
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    LOG.info("queued job=%s bytes=%s", job_id, total)
    return {"id": str(job_id), "status": "queued", "status_url": f"/jobs/{job_id}"}

@app.get("/jobs/{job_id}")
def status(job_id: uuid.UUID):
    record = db.get_job(job_id)
    if record is None:
        raise HTTPException(404, "Unknown job")
    id_, original, state, error, created, started, finished, output, report = record
    result = {
        "id": id_, "filename": original, "status": state,
        "error": error, "created_at": created, "started_at": started,
        "finished_at": finished,
    }
    if state == "completed":
        result["download_url"] = f"/jobs/{id_}/download"
        if Path(report).is_file():
            result["report"] = json.loads(Path(report).read_text(encoding="utf-8"))
    return result

@app.get("/jobs/{job_id}/download")
def download(job_id: uuid.UUID):
    record = db.get_job(job_id)
    if record is None:
        raise HTTPException(404, "Unknown job")
    if record[2] != "completed":
        raise HTTPException(409, "Job not completed")
    result_file = Path(record[7])
    if not result_file.is_file():
        raise HTTPException(500, "Result missing from persistent storage")
    return FileResponse(result_file, media_type="application/dxf", filename=f"{job_id}.dxf")
