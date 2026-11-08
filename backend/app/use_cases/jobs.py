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
from app.services.geometry import GeometryError, GeometryResolver, geometry_signature


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


async def _find_job(repository: JobRepository, job_id: str, owner_id: str, *, lock: bool = False) -> Job:
    job = await repository.get(job_id, owner_id, lock=lock)
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
    def __init__(self, session: AsyncSession, owner_id: str) -> None:
        self.owner_id = owner_id
        self.uow = JobUnitOfWork(session)

    async def execute(self, filename: str, data: bytes, channel: str,
                      conversation_id: str | None, event_id: str | None) -> dict:
        if not channel or len(channel) > 32:
            raise JobError("invalid", "Invalid channel")
        if event_id:
            existing = await self.uow.jobs.by_event(self.owner_id, channel, event_id)
            if existing:
                if existing.conversation_id != conversation_id:
                    raise JobError("conflict", "Event already belongs to another conversation")
                return _status(existing)
        filename = Path(filename or "original").name[:255]
        if len(data) > get_settings().max_upload_bytes:
            raise JobError("too_large", "File too large")
        ext = _source_extension(filename, data)
        job = Job(owner_id=self.owner_id, filename=filename, source_sha256=hashlib.sha256(data).hexdigest(),
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
            if event_id and constraint == "uq_owner_channel_event":
                existing = await self.uow.jobs.by_event(self.owner_id, channel, event_id)
                if existing and existing.conversation_id == conversation_id:
                    return _status(existing)
                raise JobError("conflict", "Event already belongs to another conversation")
            raise
        await self.uow.refresh_runs(job)
        return _status(job)


class FindJobUseCase:
    def __init__(self, session: AsyncSession, owner_id: str) -> None:
        self.owner_id = owner_id
        self.repository = JobRepository(session)

    async def execute(self, job_id: str) -> dict:
        return _status(await _find_job(self.repository, job_id, self.owner_id))


class RerunJobUseCase:
    def __init__(self, session: AsyncSession, owner_id: str) -> None:
        self.owner_id = owner_id
        self.uow = JobUnitOfWork(session)

    async def _dimension_baseline(self, job: Job, graph: DrawingGraph) -> DrawingGraph:
        graph = graph.model_copy(deep=True)
        graph.provenance["dimension_geometry_sha256"] = geometry_signature(graph)
        previous_artifact = next((artifact for run in reversed(job.runs) for artifact in run.artifacts
                                  if artifact.name == "drawing.json"), None)
        if previous_artifact is None:
            return graph
        root = storage_root()
        path = (root / previous_artifact.relative_path).resolve()
        if not path.is_relative_to(root) or not await asyncio.to_thread(path.is_file):
            raise JobError("conflict", "Previous drawing is missing; cannot validate dimension edits")
        previous = DrawingGraph.model_validate_json(await asyncio.to_thread(path.read_text, encoding="utf-8"))
        old_dimensions = {dimension.id: dimension for dimension in previous.dimensions}
        old_resolver, resolver = GeometryResolver(previous), GeometryResolver(graph)
        for dimension in graph.dimensions:
            old = old_dimensions.get(dimension.id)
            if old is None or resolver.dimension_has_anchors(dimension):
                continue
            try:
                before, after = old_resolver.dimension(old), resolver.dimension(dimension)
            except GeometryError:
                continue  # Export reports broken anchors separately.
            if any(
                (ref is None or not resolver.points[ref].refs) and old_xy == new_xy
                for ref, old_xy, new_xy in ((dimension.p1_ref, before.p1, after.p1),
                                            (dimension.p2_ref, before.p2, after.p2))
            ):
                # An unchanged free measurement is inherited, even if a client
                # drops or rewrites provenance. Use the server's saved baseline.
                graph.provenance["dimension_geometry_sha256"] = previous.provenance.get(
                    "dimension_geometry_sha256", geometry_signature(previous))
                break
        return graph

    async def execute(self, job_id: str, graph: DrawingGraph | None = None) -> dict:
        job = await _find_job(self.uow.jobs, job_id, self.owner_id, lock=True)
        if job.state in {"queued", "running"}:
            raise JobError("conflict", "Current run is still active")
        if graph is not None and graph.source_sha256 and graph.source_sha256 != job.source_sha256:
            raise JobError("invalid", "source_sha256 differs from original")
        if graph is not None:
            graph = await self._dimension_baseline(job, graph)
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
    def __init__(self, session: AsyncSession, owner_id: str) -> None:
        self.owner_id = owner_id
        self.repository = JobRepository(session)

    async def execute(self, job_id: str, name: str, run_number: int | None, run_id: str | None = None) -> Path:
        job = await _find_job(self.repository, job_id, self.owner_id)
        if run_number is not None and run_id is not None:
            raise JobError("invalid", "Choose run_number or run_id")
        if run_id is not None:
            run = next((run for run in job.runs if run.id == run_id), None)
        elif run_number is not None:
            run = next((run for run in job.runs if run.number == run_number), None)
        else:
            run = job.runs[-1] if job.runs else None
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
