#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== Ceres 2D Smoother — Build & Run ==="

# Build
echo "[1/3] Configuring..."
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "[2/3] Building..."
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Run C++ demo
echo "[3/3] Running demo..."
"$BUILD_DIR/ceres_smoother_2d_demo" "$@"

echo ""
echo "Output: $BUILD_DIR/smoothed_result.png"
