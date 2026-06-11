#!/usr/bin/env bash
# Run the full smoother_clothoid test suite (Python + C++).
#
# Usage:
#   ./run_tests.sh           # python only (default)
#   ./run_tests.sh --all     # python + ctest
#   ./run_tests.sh --cpp     # ctest only
#
# ROS's PYTHONPATH can leak into the venv and break pytest collection;
# we clear it before invoking uv.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

run_python() {
  echo "==> Python tests"
  PYTHONPATH= uv run --no-sync python -m pytest tests/ smoother_clothoid_py/tests/ \
      --timeout=60 "$@"
}

run_cpp() {
  echo "==> C++ tests"
  if [[ ! -d build ]]; then
    echo "(no build/ directory; skipping C++ tests)"
    return 0
  fi
  ( cd build && ctest --output-on-failure )
}

case "${1:-python}" in
  --all)    run_python; echo; run_cpp ;;
  --cpp)    run_cpp ;;
  python|"") run_python ;;
  *) echo "unknown argument: $1" >&2; exit 1 ;;
esac
