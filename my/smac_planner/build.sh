#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "==> Configuring..."
cmake -B "$BUILD_DIR" -S . -DCMAKE_BUILD_TYPE=Release

echo "==> Building (${CORES} cores)..."
cmake --build "$BUILD_DIR" -j"$CORES"

echo ""
echo "==> Running tests..."
ctest --test-dir "$BUILD_DIR" --output-on-failure -j"$CORES"
