#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== Ceres 2D Smoother — Build & Run (Python) ==="

# Build
if [ ! -f "$BUILD_DIR/ceres_smoother_2d.cpython-"*.so ]; then
  echo "[1/2] Building..."
  cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" -j"$(nproc)"
else
  echo "[1/2] Using existing build..."
fi

# Run Python demo
echo "[2/2] Running Python demo..."
PYTHON="${PYTHON:-/home/zks/.venv/bin/python3}"
"$PYTHON" "$SCRIPT_DIR/python/demo.py" "$@"

echo ""
echo "Output: $BUILD_DIR/smooth_result.png"
