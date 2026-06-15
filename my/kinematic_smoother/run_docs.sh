#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command uv
require_command "$python_bin"

docs_host="${CS_DOCS_HOST:-127.0.0.1}"
docs_port="${CS_DOCS_PORT:-8000}"
docs_port_explicit="${CS_DOCS_PORT+x}"

resolve_docs_port() {
  "$python_bin" - "$docs_host" "$docs_port" "$docs_port_explicit" <<'PY'
import socket
import sys

host = sys.argv[1]
start_port = int(sys.argv[2])
port_is_explicit = len(sys.argv) > 3 and bool(sys.argv[3])


def is_port_available(port: int) -> bool:
  family = socket.AF_INET6 if ":" in host else socket.AF_INET
  with socket.socket(family, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
      sock.bind((host, port))
    except OSError:
      return False
  return True


if port_is_explicit:
  if not is_port_available(start_port):
    print(
      f"Requested docs port {start_port} on {host} is already in use. "
      "Set CS_DOCS_PORT to another port and retry.",
      file=sys.stderr,
    )
    sys.exit(1)
  print(start_port)
  sys.exit(0)

for port in range(start_port, start_port + 100):
  if is_port_available(port):
    print(port)
    sys.exit(0)

print(
  f"Could not find a free docs port on {host} in range {start_port}-{start_port + 99}.",
  file=sys.stderr,
)
sys.exit(1)
PY
}

resolved_docs_port="$(resolve_docs_port)"

if [[ "$resolved_docs_port" != "$docs_port" ]]; then
  echo "Docs port $docs_port is busy; using $resolved_docs_port instead." >&2
fi

cd "$script_dir"
exec uvx --with mkdocs-material mkdocs serve -f mkdocs.yml -a "${docs_host}:${resolved_docs_port}"