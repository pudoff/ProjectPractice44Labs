#!/usr/bin/env bash
set -euo pipefail
URL="${API_URL:-http://127.0.0.1:8000}"
curl --fail "$URL/health"
echo
input="${1:-runs/клапан.png}"
response="$(curl --fail -sS -F "file=@${input};type=image/png" "$URL/api/v1/jobs")"
echo "$response"
echo "Use id from the response: $URL/api/v1/jobs/{id}"
echo "Report: $URL/api/v1/jobs/{id}/artifacts/report.json"
echo "DXF (if produced): $URL/api/v1/jobs/{id}/artifacts/result.dxf"
echo "Without an OCR adapter the expected recognition status is needs_review."
