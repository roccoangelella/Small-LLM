#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

[[ -f .env ]] || {
  echo "Missing $ROOT/.env" >&2
  exit 1
}

exec uv run \
  --python 3.13 \
  --env-file .env \
  --with-requirements kaggle/requirements-100m-publish.txt \
  python kaggle/build_and_push_1b.py "$@"
