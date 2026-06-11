#!/usr/bin/env bash
# Run the full test suite for ceres_smoother_2d.
# Usage:  ./run_tests.sh          # build C++, run C++, Python, and (if up) Web API tests
#         ./run_tests.sh --no-cpp # skip C++ build & tests (e.g. when nothing changed)
#         ./run_tests.sh --web URL   # explicitly set the web URL for Web API tests

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Resolve repo root (parent of this script). The map lives at <repo>/maps/occupancy_map.png.
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export MAP_PATH="${MAP_PATH:-$REPO_ROOT/maps/occupancy_map.png}"

WEB_URL=""
for arg in "$@"; do
  case "$arg" in
    --no-cpp) SKIP_CPP=1 ;;
    --web=*)  WEB_URL="${arg#--web=}" ;;
    --web)    shift; WEB_URL="${1:-}" ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

run_cpp() {
  echo
  echo "================  C++ unit tests  ================"
  if [ ! -d build ]; then
    echo "[1/2] Configuring..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
  fi
  echo "[2/2] Building ceres_smoother_2d_tests..."
  cmake --build build --target ceres_smoother_2d_tests -- -j"$(nproc)"
  echo "Running ceres_smoother_2d_tests..."
  ./build/ceres_smoother_2d_tests
}

run_python() {
  echo
  echo "================  Python unit tests  ================"
  if command -v uv >/dev/null 2>&1; then
    PYTEST=(uv run --project "$HERE" pytest)
  else
    PYTEST=(python3 -m pytest)
  fi
  "${PYTEST[@]}" tests/test_python.py
}

run_web() {
  echo
  echo "================  Web API tests  ================"
  # Probe for a running server unless user supplied one.
  if [ -z "$WEB_URL" ]; then
    if curl -fsS --max-time 1 http://127.0.0.1:5000/api/costmap >/dev/null 2>&1; then
      WEB_URL="http://127.0.0.1:5000"
    else
      echo "[skip] No web server detected on :5000. Start with: ./run_web.sh"
      return 0
    fi
  fi
  echo "Testing $WEB_URL ..."
  WEB_BASE_URL="$WEB_URL" python3 -m pytest tests/test_web_api.py
}

if [ "${SKIP_CPP:-}" = "1" ]; then
  echo "[skip] C++ tests (--no-cpp)"
else
  run_cpp
fi
run_python
run_web

echo
echo "================  ALL DONE  ================"