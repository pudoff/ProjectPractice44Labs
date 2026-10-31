from app.dependencies.base import SessionDep
from app.use_cases.jobs import CreateJobUseCase, FindJobUseCase, GetArtifactUseCase, RerunJobUseCase


def get_create_job_use_case(session: SessionDep) -> CreateJobUseCase:
    return CreateJobUseCase(session)


def get_find_job_use_case(session: SessionDep) -> FindJobUseCase:
    return FindJobUseCase(session)


def get_rerun_job_use_case(session: SessionDep) -> RerunJobUseCase:
    return RerunJobUseCase(session)


def get_artifact_use_case(session: SessionDep) -> GetArtifactUseCase:
    return GetArtifactUseCase(session)
