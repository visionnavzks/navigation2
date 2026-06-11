#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-web"
PORT="${KINEMATIC_SMOOTHER_WEB_PORT:-5055}"
PYTHON_BIN="${KINEMATIC_SMOOTHER_PYTHON:-/home/zks/.venv/bin/python3}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DBUILD_PYTHON=ON -DBUILD_TESTS=OFF -DPython_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Serving kinematic_path_smoother web demo at http://127.0.0.1:${PORT}"
cd "${ROOT_DIR}"
KINEMATIC_SMOOTHER_WEB_PORT="${PORT}" "${PYTHON_BIN}" web/app.py
