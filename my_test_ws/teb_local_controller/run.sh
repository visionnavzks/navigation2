#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_PATH="my_test_ws/teb_local_controller/app.py"
ACTION="${1:-run}"

HOST="${HOST:-127.0.0.1}"
FLASK_DEBUG="${FLASK_DEBUG:-0}"
DEFAULT_PORT=5002
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}/teb_local_controller"
RUN_CMD=()

usage() {
  cat <<'EOF'
Usage: run.sh [run|start|stop|restart|status]

Commands:
  run      Run the service in the foreground. This is the default.
  start    Start the service in the background and write a pid file.
  stop     Stop the background service tracked by the pid file.
  restart  Stop the tracked background service, then start it again.
  status   Show the tracked background service status.

Environment:
  HOST         Bind host. Default: 127.0.0.1
  PORT         Bind port. Default for managed commands: 5002
  FLASK_DEBUG  Flask debug flag. Default: 0
EOF
}

choose_free_port() {
  local utility_python
  if utility_python="$(python_for_utils)"; then
    HOST="$HOST" "$utility_python" <<'PY'
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
    return
  else
    echo "$DEFAULT_PORT"
  fi
}

python_for_utils() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "$REPO_ROOT/.venv/bin/python"
    return
  fi

  return 1
}

resolve_port() {
  if [[ -n "${PORT:-}" ]]; then
    return
  fi

  if [[ "$ACTION" == "run" ]]; then
    PORT="$(choose_free_port)"
  else
    PORT="$DEFAULT_PORT"
  fi
}

build_run_cmd() {
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    RUN_CMD=("$REPO_ROOT/.venv/bin/python" "$APP_PATH")
    return
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to run this app when .venv is not available." >&2
    exit 1
  fi

  RUN_CMD=(
    uv run
    --python 3.11
    --with flask
    --with numpy
    --with casadi
    python "$APP_PATH"
  )
}

port_available() {
  local utility_python
  if ! utility_python="$(python_for_utils)"; then
    return 0
  fi

  HOST="$HOST" PORT="$PORT" "$utility_python" <<'PY'
import os
import socket
import sys

host = os.environ["HOST"]
port = int(os.environ["PORT"])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sys.exit(1)
PY
}

pid_is_running() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

tracked_pid() {
  if [[ -f "$PID_FILE" ]]; then
    tr -d '[:space:]' < "$PID_FILE"
  fi
}

service_status() {
  local pid
  pid="$(tracked_pid || true)"
  if pid_is_running "$pid"; then
    echo "TEB local controller is running: pid=$pid url=http://$HOST:$PORT"
    echo "Log: $LOG_FILE"
    return 0
  fi

  if [[ -f "$PID_FILE" ]]; then
    echo "TEB local controller is not running. Removing stale pid file: $PID_FILE"
    rm -f "$PID_FILE"
  else
    echo "TEB local controller is not running."
  fi
}

stop_service() {
  local pid
  pid="$(tracked_pid || true)"
  if ! pid_is_running "$pid"; then
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
    echo "No tracked TEB local controller service is running."
    return 0
  fi

  echo "Stopping TEB local controller pid=$pid"
  kill "$pid"

  for _ in {1..50}; do
    if ! pid_is_running "$pid"; then
      rm -f "$PID_FILE"
      echo "Stopped."
      return 0
    fi
    sleep 0.1
  done

  echo "Service did not stop after 5s; sending SIGKILL to pid=$pid"
  kill -9 "$pid"
  rm -f "$PID_FILE"
  echo "Stopped."
}

start_service() {
  local pid
  pid="$(tracked_pid || true)"
  if pid_is_running "$pid"; then
    echo "TEB local controller is already running: pid=$pid url=http://$HOST:$PORT"
    return 0
  fi
  [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"

  if ! port_available; then
    echo "Port $HOST:$PORT is already in use, and no tracked pid file can stop it." >&2
    echo "Use PORT=... $0 start, or stop the process that owns this port." >&2
    exit 1
  fi

  mkdir -p "$RUNTIME_DIR"
  echo "Starting TEB local controller at http://$HOST:$PORT"
  echo "Log: $LOG_FILE"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -c 'cd "$1"; shift; exec "$@"' bash "$REPO_ROOT" \
      env HOST="$HOST" PORT="$PORT" FLASK_DEBUG="$FLASK_DEBUG" "${RUN_CMD[@]}" \
      </dev/null >> "$LOG_FILE" 2>&1 &
  else
    nohup bash -c 'cd "$1"; shift; exec "$@"' bash "$REPO_ROOT" \
      env HOST="$HOST" PORT="$PORT" FLASK_DEBUG="$FLASK_DEBUG" "${RUN_CMD[@]}" \
      </dev/null >> "$LOG_FILE" 2>&1 &
  fi
  pid="$!"
  echo "$pid" > "$PID_FILE"
  echo "Started pid=$pid"
}

case "$ACTION" in
  run|start|stop|restart|status)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

resolve_port
HOST_ID="${HOST//[^[:alnum:]._-]/_}"
PID_FILE="${PID_FILE:-$RUNTIME_DIR/teb_local_controller_${HOST_ID}_${PORT}.pid}"
LOG_FILE="${LOG_FILE:-$RUNTIME_DIR/teb_local_controller_${HOST_ID}_${PORT}.log}"

case "$ACTION" in
  run|start|restart)
    build_run_cmd
    ;;
esac

case "$ACTION" in
  run)
    cd "$REPO_ROOT"
    export HOST PORT FLASK_DEBUG
    echo "Starting TEB local controller at http://$HOST:$PORT"
    exec "${RUN_CMD[@]}"
    ;;
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_service
    ;;
  status)
    service_status
    ;;
esac
