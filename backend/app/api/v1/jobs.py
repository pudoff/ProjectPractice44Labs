from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.dependencies.jobs import (
    get_artifact_use_case, get_create_job_use_case,
    get_find_job_use_case, get_rerun_job_use_case,
)
from app.domain.drawing import DrawingGraph
from app.schemas.jobs import JobStatusSchema
from app.use_cases.jobs import CreateJobUseCase, FindJobUseCase, GetArtifactUseCase, RerunJobUseCase

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
CreateJobDep = Annotated[CreateJobUseCase, Depends(get_create_job_use_case)]
FindJobDep = Annotated[FindJobUseCase, Depends(get_find_job_use_case)]
RerunJobDep = Annotated[RerunJobUseCase, Depends(get_rerun_job_use_case)]
GetArtifactDep = Annotated[GetArtifactUseCase, Depends(get_artifact_use_case)]


@router.post("", status_code=201, response_model=JobStatusSchema)
async def create_job(use_case: CreateJobDep, file: UploadFile = File(...),
                     channel: str = Form("web"), conversation_id: str | None = Form(None),
                     event_id: str | None = Form(None)):
    data = await file.read(get_settings().max_upload_bytes + 1)
    return await use_case.execute(file.filename or "original", data, channel, conversation_id, event_id)


@router.get("/{job_id}", response_model=JobStatusSchema)
async def get_job(job_id: str, use_case: FindJobDep):
    return await use_case.execute(job_id)


@router.post("/{job_id}/rerun", status_code=202, response_model=JobStatusSchema)
async def rerun(job_id: str, use_case: RerunJobDep):
    return await use_case.execute(job_id)


@router.post("/{job_id}/drawing", status_code=202, response_model=JobStatusSchema)
async def submit_drawing(job_id: str, graph: DrawingGraph, use_case: RerunJobDep):
    return await use_case.execute(job_id, graph)


@router.get("/{job_id}/artifacts/{name}")
async def get_artifact(job_id: str, name: str, use_case: GetArtifactDep, run_number: int | None = None):
    path = await use_case.execute(job_id, name, run_number)
    return FileResponse(path, media_type="application/octet-stream", filename=name)
