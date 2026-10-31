#!/usr/bin/env bash
set -euo pipefail
# Absolutely no pulls or builds on this machine.
env_file="${COMPOSE_ENV_FILE:-.env}"
[[ -f "$env_file" ]] || { echo "Copy .env.example to .env (or COMPOSE_ENV_FILE) and set a password" >&2; exit 1; }
[[ -f offline/docker-images.tar ]] || { echo "Missing offline/docker-images.tar" >&2; exit 1; }
[[ -f offline/ocr-models.tar.gz ]] || { echo "Missing offline/ocr-models.tar.gz" >&2; exit 1; }
grep -q 'POSTGRES_PASSWORD=REPLACE_ME' "$env_file" && { echo "Replace POSTGRES_PASSWORD placeholder" >&2; exit 1; }
sha256sum -c offline/SHA256SUMS
mkdir -p models
tar -xzf offline/ocr-models.tar.gz -C models
docker load -i offline/docker-images.tar
export OFFLINE_NETWORK_INTERNAL=true
MODEL_PREPARATION_NETWORK=none docker compose --env-file "$env_file" -f compose.yaml run --rm --no-deps --pull never model-assets python -m app.infra.model_assets
# API and worker wait for successful database migration.
docker compose --env-file "$env_file" -f compose.yaml up -d --no-build --pull never
docker compose --env-file "$env_file" -f compose.yaml ps
