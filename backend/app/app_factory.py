from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.v1 import health, jobs
from app.use_cases.jobs import JobError


def create_app() -> FastAPI:
    app = FastAPI(title="Drawing to DXF API")
    app.include_router(health.router)
    app.include_router(jobs.router)

    @app.exception_handler(JobError)
    async def job_error_handler(_request, error: JobError):
        status = {"not_found": 404, "bad_file": 415, "too_large": 413,
                  "invalid": 422, "conflict": 409}[error.code]
        return JSONResponse(status_code=status, content={"detail": error.detail})

    return app
