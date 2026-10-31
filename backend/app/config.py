from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://draw:draw@localhost:5432/draw"
    storage_root: Path = Path("var/storage")
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    poll_seconds: float = Field(default=2, gt=0)
    cv_adapter: str | None = None
    ocr_adapter: str | None = None
    builtin_pipeline_path: Path = Path(__file__).resolve().parents[2] / "runs/klapan_png2dxf/pipeline_b/pipeline.py"
    dxf_version: str = "R2007"
    dxf_units: int = 4
    dxf_measurement: int = 1
    dxf_text_height: float = 3.5
    dxf_arrow_size: float = 3.0
    dxf_dimension_decimals: int = 2
    dxf_dimension_zero_suppression: int = 8
    dxf_font: str = "isocpeur.ttf"
    dxf_layers: tuple[tuple[str, int, int], ...] = (
        ("MAIN", 7, 50), ("THIN", 3, 25), ("AXIS", 1, 25),
        ("DIMS", 4, 25), ("TEXT", 7, 25),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
