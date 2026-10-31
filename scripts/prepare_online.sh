#!/usr/bin/env bash
set -euo pipefail
# Internet required on this machine only. Run at repository root.
command -v poetry >/dev/null || { echo "Install Poetry 2.5.1 first" >&2; exit 1; }
[[ "$(poetry --version)" == *"2.5.1"* ]] || { echo "Expected Poetry 2.5.1" >&2; exit 1; }
poetry lock
poetry check --lock
poetry install --only main --no-root --sync
poetry run python -m tools.fetch_models --download
docker compose -f compose.yaml build api
docker pull postgres:16.6-bookworm
mkdir -p offline
docker save -o offline/docker-images.tar practice44-app:0.1.0 postgres:16.6-bookworm
tar -czf offline/ocr-models.tar.gz -C models det.onnx rec_eslav.onnx ppocrv5_eslav_dict.txt ch_ppocr_mobile_v2.0_cls_mobile.onnx manifest.local.json
sha256sum offline/docker-images.tar offline/ocr-models.tar.gz > offline/SHA256SUMS
echo "Transfer this project (with poetry.lock), offline/, and .env created locally."
