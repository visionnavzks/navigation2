#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

HOST="${HOST:-127.0.0.1}"
FLASK_DEBUG="${FLASK_DEBUG:-0}"

if [[ -z "${PORT:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PORT="$(HOST="$HOST" python3 <<'PY'
import os
import socket

host = os.environ["HOST"]
for candidate in range(5002, 5102):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate))
        except OSError:
            continue
        print(candidate)
        break
else:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        print(sock.getsockname()[1])
PY
    )"
  else
    PORT=5002
  fi
fi

export HOST
export PORT
export FLASK_DEBUG

echo "Starting TEB local controller at http://$HOST:$PORT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  exec "$REPO_ROOT/.venv/bin/python" my/teb_local_controller/app.py
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to run this app when .venv is not available." >&2
  exit 1
fi

exec uv run \
  --python 3.11 \
  --with flask \
  --with numpy \
  --with casadi \
  python my/teb_local_controller/app.py