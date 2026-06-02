#!/usr/bin/env bash
# Build pybind11 modules and launch the Hybrid A* debugger web app.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${VENV_DIR:-${script_dir}/.venv}"
python_bin="${PYTHON_BIN:-python3}"
parallel_jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command uv
require_command cmake
require_command "$python_bin"

venv_python_is_healthy() {
  [[ -x "$venv_dir/bin/python" ]] || return 1
  "$venv_dir/bin/python" - <<'PY' >/dev/null 2>&1
import encodings
PY
}

if [[ -z "${parallel_jobs}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    parallel_jobs="$(nproc)"
  else
    parallel_jobs="4"
  fi
fi

if ! venv_python_is_healthy; then
  rm -rf "$venv_dir"
  uv venv --python "$python_bin" "$venv_dir"
fi

uv pip install --python "${venv_dir}/bin/python" flask numpy pybind11

pybind11_dir="$("${venv_dir}/bin/python" -m pybind11 --cmakedir)"
python_build_tag="$("${venv_dir}/bin/python" - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
)"
build_dir="${BUILD_DIR:-${script_dir}/build-${python_build_tag}}"

# ---- Build hybrid_astar pybind module ----
cmake \
  -S "$script_dir" \
  -B "$build_dir" \
  -DBUILD_PYTHON=ON \
  -DBUILD_TESTS=OFF \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="${venv_dir}/bin/python" \
  -Dpybind11_DIR="$pybind11_dir"

cmake --build "$build_dir" --parallel "$parallel_jobs"

export PYTHONPATH="${build_dir}:${script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export HA_WEBAPP_DEBUG="${HA_WEBAPP_DEBUG:-1}"
export HA_WEBAPP_RELOADER="${HA_WEBAPP_RELOADER:-0}"
export HA_WEBAPP_HOST="${HA_WEBAPP_HOST:-127.0.0.1}"
export HA_WEBAPP_PORT="${HA_WEBAPP_PORT:-5006}"
exec "${venv_dir}/bin/python" "$script_dir/web/app.py"
