FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1 \
    UV_LINK_MODE=copy

WORKDIR /srv

COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /uvx /bin/
COPY backend/pyproject.toml backend/uv.lock ./

RUN set -eux \
    && uv sync --locked --no-install-project --no-dev

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv \
    PATH="/srv/.venv/bin:$PATH"

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /srv/.venv /srv/.venv
COPY backend/app ./app
COPY backend/migrations ./migrations
COPY backend/alembic.ini ./
COPY backend/scripts/download_ocr_model.py ./scripts/download_ocr_model.py
COPY runs/klapan_png2dxf/pipeline_b ./runs/klapan_png2dxf/pipeline_b
COPY tools/eskd_dxf.py ./tools/eskd_dxf.py

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
