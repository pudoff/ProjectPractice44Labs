from pydantic import BaseModel


class RunStatusSchema(BaseModel):
    id: str
    number: int
    state: str
    stage: str
    error_code: str | None
    error_message: str | None
    artifacts: list[str]


class JobStatusSchema(BaseModel):
    id: str
    state: str
    filename: str
    source_sha256: str
    runs: list[RunStatusSchema]
