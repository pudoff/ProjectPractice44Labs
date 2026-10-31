import asyncio
import hashlib
import json
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.domain.drawing import DrawingGraph
from app.infra.models import Job, Run
from app.infra.storage import artifact_from_path, atomic_bytes, job_dir, run_dir, storage_root
from app.repositories.jobs import JobRepository
from app.uow.jobs import JobUnitOfWork


class JobError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _status(job: Job) -> dict:
    return {
        "id": job.id, "state": job.state, "filename": job.filename,
        "source_sha256": job.source_sha256,
        "runs": [
            {"id": run.id, "number": run.number, "state": run.state,
             "stage": run.stage, "error_code": run.error_code,
             "error_message": run.error_message,
             "artifacts": [artifact.name for artifact in run.artifacts]}
            for run in job.runs
        ],
    }


async def _find_job(repository: JobRepository, job_id: str, *, lock: bool = False) -> Job:
    job = await repository.get(job_id, lock=lock)
    if job is None:
        raise JobError("not_found", "Job not found")
    return job


def _source_extension(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    signatures = {
        ".pdf": b"%PDF-", ".png": b"\x89PNG\r\n\x1a\n",
        ".jpg": b"\xff\xd8", ".jpeg": b"\xff\xd8",
    }
    if ext not in signatures or not data.startswith(signatures[ext]):
        raise JobError("bad_file", "Expected PDF, PNG or JPEG with matching contents")
    return ext


class CreateJobUseCase:
    def __init__(self, session: AsyncSession) -> None:
        self.uow = JobUnitOfWork(session)

    async def execute(self, filename: str, data: bytes, channel: str,
                      conversation_id: str | None, event_id: str | None) -> dict:
        if not channel or len(channel) > 32:
            raise JobError("invalid", "Invalid channel")
        if event_id:
            existing = await self.uow.jobs.by_event(channel, event_id)
            if existing:
                if existing.conversation_id != conversation_id:
                    raise JobError("conflict", "Event already belongs to another conversation")
                return _status(existing)
        filename = Path(filename or "original").name[:255]
        if len(data) > get_settings().max_upload_bytes:
            raise JobError("too_large", "File too large")
        ext = _source_extension(filename, data)
        job = Job(filename=filename, source_sha256=hashlib.sha256(data).hexdigest(),
                  source_ext=ext, channel=channel, conversation_id=conversation_id,
                  event_id=event_id)
        try:
            self.uow.add(job)
            await self.uow.flush()
            run = Run(job=job, number=1, state="queued", stage="queued", parameters="{}")
            self.uow.add(run)
            await self.uow.flush()
            path = job_dir(job.id) / "input" / f"original{ext}"
            await asyncio.to_thread(atomic_bytes, path, data)
            self.uow.add(await asyncio.to_thread(artifact_from_path, run.id, path))
            await self.uow.commit()
        except IntegrityError as exc:
            await self.uow.rollback()
            constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
            if event_id and constraint == "uq_channel_event":
                existing = await self.uow.jobs.by_event(channel, event_id)
                if existing and existing.conversation_id == conversation_id:
                    return _status(existing)
                raise JobError("conflict", "Event already belongs to another conversation")
            raise
        await self.uow.refresh_runs(job)
        return _status(job)


class FindJobUseCase:
    def __init__(self, session: AsyncSession) -> None:
        self.repository = JobRepository(session)

    async def execute(self, job_id: str) -> dict:
        return _status(await _find_job(self.repository, job_id))


class RerunJobUseCase:
    def __init__(self, session: AsyncSession) -> None:
        self.uow = JobUnitOfWork(session)

    async def execute(self, job_id: str, graph: DrawingGraph | None = None) -> dict:
        job = await _find_job(self.uow.jobs, job_id, lock=True)
        if job.state in {"queued", "running"}:
            raise JobError("conflict", "Current run is still active")
        if graph is not None and graph.source_sha256 and graph.source_sha256 != job.source_sha256:
            raise JobError("invalid", "source_sha256 differs from original")
        number = max(run.number for run in job.runs) + 1
        run = Run(job=job, number=number, state="queued", stage="queued",
                  parameters=json.dumps({"drawing_path": graph is not None}))
        self.uow.add(run)
        await self.uow.flush()
        if graph is not None:
            path = run_dir(job.id, run.id) / "submitted.json"
            await asyncio.to_thread(atomic_bytes, path, graph.model_dump_json(indent=2).encode())
        job.state = "queued"
        await self.uow.commit()
        await self.uow.refresh_runs(job)
        return _status(job)


class GetArtifactUseCase:
    def __init__(self, session: AsyncSession) -> None:
        self.repository = JobRepository(session)

    async def execute(self, job_id: str, name: str, run_number: int | None) -> Path:
        job = await _find_job(self.repository, job_id)
        run = (next((run for run in job.runs if run.number == run_number), None)
               if run_number else (job.runs[-1] if job.runs else None))
        if run is None:
            raise JobError("not_found", "Run not found")
        artifact = await self.repository.artifact(run.id, name)
        if artifact is None:
            raise JobError("not_found", "Artifact not ready")
        root = storage_root()
        path = (root / artifact.relative_path).resolve()
        if not path.is_relative_to(root) or not await asyncio.to_thread(path.is_file):
            raise JobError("not_found", "Artifact missing")
        return path
