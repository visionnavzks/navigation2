#!/usr/bin/env bash

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

uv pip install --python "${venv_dir}/bin/python" flask numpy nanobind Pillow

nanobind_dir="$("${venv_dir}/bin/python" -c "import nanobind; print(nanobind.cmake_dir())" 2>/dev/null || echo "")"
python_build_tag="$("${venv_dir}/bin/python" - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
)"
build_dir="${BUILD_DIR:-${script_dir}/build-${python_build_tag}}"

cmake_args=(
  -S "$script_dir"
  -B "$build_dir"
  -DBUILD_PYTHON=OFF
  -DBUILD_TESTS=OFF
  -DUSE_NANOBIND=ON
  -DPython_EXECUTABLE="${venv_dir}/bin/python"
)

if [[ -n "$nanobind_dir" ]]; then
  cmake_args+=(-Dnanobind_DIR="$nanobind_dir")
fi

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --parallel "$parallel_jobs"

export PYTHONPATH="${build_dir}:${script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export SC_WEBAPP_DEBUG="${SC_WEBAPP_DEBUG:-0}"
export SC_WEBAPP_RELOADER="${SC_WEBAPP_RELOADER:-0}"
export SC_WEBAPP_HOST="${SC_WEBAPP_HOST:-127.0.0.1}"
export SC_WEBAPP_PORT="${SC_WEBAPP_PORT:-5007}"
exec "${venv_dir}/bin/python" "$script_dir/web/app.py"
