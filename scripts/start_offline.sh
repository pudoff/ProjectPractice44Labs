#!/usr/bin/env bash
set -euo pipefail
# Absolutely no pulls or builds on this machine.
[[ -f .env ]] || { echo "Copy .env.example to .env and set a non-placeholder password" >&2; exit 1; }
[[ -f offline/docker-images.tar ]] || { echo "Missing offline/docker-images.tar" >&2; exit 1; }
[[ -f offline/ocr-models.tar.gz ]] || { echo "Missing offline/ocr-models.tar.gz" >&2; exit 1; }
grep -q 'POSTGRES_PASSWORD=REPLACE_ME' .env && { echo "Replace POSTGRES_PASSWORD placeholder" >&2; exit 1; }
sha256sum -c offline/SHA256SUMS
mkdir -p models storage/uploads storage/results
tar -xzf offline/ocr-models.tar.gz -C models
docker load -i offline/docker-images.tar
docker compose -f compose.offline-prototype.yaml up -d --no-build --pull never
docker compose -f compose.offline-prototype.yaml ps
