FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy

WORKDIR /srv/backend

COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /uvx /bin/
COPY backend/pyproject.toml backend/uv.lock ./
COPY mlcv/pyproject.toml /srv/mlcv/pyproject.toml
COPY mlcv/drawcv /srv/mlcv/drawcv

RUN set -eux \
    && uv sync --locked --no-install-project --no-dev

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv/backend:/srv \
    PATH="/srv/backend/.venv/bin:$PATH"

WORKDIR /srv/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /srv/backend/.venv /srv/backend/.venv
COPY --from=builder /srv/mlcv /srv/mlcv
COPY backend/app ./app
COPY backend/migrations ./migrations
COPY backend/alembic.ini ./
COPY runs/klapan_png2dxf/pipeline_b /srv/runs/klapan_png2dxf/pipeline_b
COPY tools /srv/tools
# Model utilities only; the main API/worker are in backend/app.
COPY service/__init__.py service/models.py service/settings.py /srv/service/
COPY models/SOURCES.json /srv/models/SOURCES.json
ARG BUILD_REVISION=unknown
ENV BUILD_REVISION=$BUILD_REVISION

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
COPY --from=builder /bin/uv /bin/uv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-install-project
COPY backend/tests ./tests
COPY mlcv/tests /srv/mlcv/tests
COPY tests /srv/tests
COPY service /srv/service
COPY pyproject.toml .python-version /srv/
CMD ["python", "-m", "pytest", "-q", "tests"]
