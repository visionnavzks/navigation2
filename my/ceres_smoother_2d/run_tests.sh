#!/usr/bin/env bash
# 运行 ceres_smoother_2d 的完整测试套件。
# 用法：./run_tests.sh          # 构建 C++，运行 C++、Python 和（若已启动）Web API 测试
#      ./run_tests.sh --no-cpp # 跳过 C++ 构建和测试（例如未改 C++ 时）
#      ./run_tests.sh --web URL # 显式指定 Web API 测试使用的 URL

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 解析仓库根目录（该脚本的父目录）。地图位于 <repo>/maps/occupancy_map.png。
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
  # 若用户未指定 URL，则探测本机是否已有运行中的服务。
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
