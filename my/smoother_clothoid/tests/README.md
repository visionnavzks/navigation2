# Tests

This directory contains the comprehensive Python test suite for `smoother_clothoid`.
The C++ tests live in `../test/test_smoother.cpp` and are built into `../build/test_smoother`.

## Layout

| File | What it covers |
| --- | --- |
| `conftest.py` | Shared fixtures (native module, costmaps, paths). Adds `build-py313` and `build` to `sys.path` so the nanobind extension is importable. |
| `test_costmap2d.py` | `Costmap2D` dataclass, set/get roundtrip, overflow clamping. |
| `test_utils.py` | `normalize_angle`, `angle_diff`, grid/world conversion, goal-frame heading. |
| `test_exceptions.py` | Error codes, failure reasons, message formatting, `throw_or_store_smoothing_failure`. |
| `test_costs.py` | `transition_residuals`, `boundary_residuals`, `reference_residuals`, `obstacle_residuals` (all cost functors). |
| `test_esdf.py` | `compute_esdf` / `compute_approximate_esdf`, ESDF algorithm enum. |
| `test_options.py` | `SmootherParams` / `OptimizerParams` defaults and isolation. |
| `test_problem_builder.py` | Path processing (cusp insertion, gear assignment), bounds, unpack / upsample. |
| `test_smoother_request.py` | `SmootherRequest` / `SmootherResult` dataclasses. |
| `test_astar.py` | A* planner + `downsample_path` (used by the web demo). |
| `test_native_smoother.py` | End-to-end smoke tests against the compiled C++ extension. |
| `test_web_helpers.py` | Flask app helpers: `_path_length`, `_reconstruct_path_with_yaw`, `_occupancy_to_costmap`, `_inflate_costmap`, `_coerce_bool`, plus `/api/costmap` and `/api/smooth` integration. |

## Running

```bash
# from this directory
./run_tests.sh            # python only
./run_tests.sh --all      # python + ctest
./run_tests.sh --cpp      # ctest only
```

The script clears `PYTHONPATH` so ROS's site-packages (which point at Python 3.10
and a missing `yaml` module) do not break pytest collection.

To run a single file:

```bash
PYTHONPATH= uv run --no-sync python -m pytest tests/test_costs.py -v
```

The `native_module` fixture is automatically skipped if the C++ extension has not
been built (i.e. neither `build-py313/` nor `build/` contains the .so).
