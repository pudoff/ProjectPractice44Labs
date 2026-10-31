#!/usr/bin/env bash
set -euo pipefail
URL="${API_URL:-http://127.0.0.1:8000}"
curl --fail "$URL/health"
echo
response="$(curl --fail -sS -F "file=@runs/клапан.png;type=image/png" "$URL/jobs")"
echo "$response"
job_id="$(printf '%s' "$response" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"
echo "Job ID: $job_id"
echo "Poll: curl -sS $URL/jobs/$job_id"
echo "Once completed: curl -f $URL/jobs/$job_id/download -o result.dxf"
