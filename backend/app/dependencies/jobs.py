from app.dependencies.base import SessionDep
from app.dependencies.auth import PrincipalDep
from app.use_cases.jobs import CreateJobUseCase, FindJobUseCase, GetArtifactUseCase, RerunJobUseCase


def get_create_job_use_case(session: SessionDep, principal: PrincipalDep) -> CreateJobUseCase:
    return CreateJobUseCase(session, principal.user_id)


def get_find_job_use_case(session: SessionDep, principal: PrincipalDep) -> FindJobUseCase:
    return FindJobUseCase(session, principal.user_id)


def get_rerun_job_use_case(session: SessionDep, principal: PrincipalDep) -> RerunJobUseCase:
    return RerunJobUseCase(session, principal.user_id)


def get_artifact_use_case(session: SessionDep, principal: PrincipalDep) -> GetArtifactUseCase:
    return GetArtifactUseCase(session, principal.user_id)
