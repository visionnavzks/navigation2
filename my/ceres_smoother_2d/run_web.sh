#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Ensure nanobind module is up-to-date (cmake增量编译，未改动文件不会重编)
BUILD_DIR="$SCRIPT_DIR/build"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"

PORT="${1:-5000}"
# 默认开启自动重载(便于调参与前后端联调)。设置 CERES_WEB_RELOAD=0 可关闭。
AUTO_RELOAD="${CERES_WEB_RELOAD:-1}"
AUTO_DEBUG="${CERES_WEB_DEBUG:-1}"
PYTHON="/home/zks/.venv/bin/python3"
if [[ "$AUTO_RELOAD" != "0" ]]; then
  export CERES_WEB_RELOAD=1
fi
if [[ "$AUTO_DEBUG" != "0" ]]; then
  export CERES_WEB_DEBUG=1
fi
"$PYTHON" "$SCRIPT_DIR/python/app.py" "$PORT"
