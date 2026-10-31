#!/usr/bin/env bash
set -euo pipefail
# Run at repository root. Only the preparation machine needs Internet.
# Set DOCKER_DEFAULT_PLATFORM to the deployment target before building.
export BUILD_REVISION="$(git rev-parse HEAD)$(test -z "$(git status --porcelain --untracked-files=no)" || printf '+dirty')"
compose_args=(-f compose.yaml)
if [[ -n "${COMPOSE_ENV_FILE:-}" ]]; then compose_args+=(--env-file "$COMPOSE_ENV_FILE"); fi
case "$(uname -s)" in
  MINGW*|MSYS*) export MSYS_NO_PATHCONV=1 ;;
esac
docker compose "${compose_args[@]}" build api
docker compose "${compose_args[@]}" pull db
docker compose "${compose_args[@]}" run --rm --no-deps --pull never model-assets
MODEL_PREPARATION_NETWORK=none docker compose "${compose_args[@]}" run --rm --no-deps --pull never model-assets python -m app.infra.model_assets
mkdir -p offline
images=()
while IFS= read -r image; do images+=("$image"); done < <(docker compose "${compose_args[@]}" config --images | sort -u)
docker save -o offline/docker-images.tar "${images[@]}"
tar -czf offline/ocr-models.tar.gz -C models det.onnx det_cv_v6_small.onnx rec_eslav.onnx ppocrv5_eslav_dict.txt ch_ppocr_mobile_v2.0_cls_mobile.onnx manifest.local.json
docker image inspect "${images[@]}" --format '{{.Id}} {{.Os}}/{{.Architecture}} {{json .RepoTags}}' > offline/images.txt
printf '%s\n' "$BUILD_REVISION" > offline/revision.txt
sha256sum offline/docker-images.tar offline/ocr-models.tar.gz offline/images.txt offline/revision.txt \
  compose.yaml backend/uv.lock scripts/start_offline.sh > offline/SHA256SUMS
echo "Transfer this project, offline/, and a locally configured .env. Main API: /api/v1/jobs."
