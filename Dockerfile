FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir poetry==2.2.1

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry check --lock && poetry install --only main --no-root --sync --no-interaction
COPY service/ ./service/
COPY tools/ ./tools/
COPY runs/klapan_png2dxf/pipeline_b/ ./runs/klapan_png2dxf/pipeline_b/
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
