# kinematic_path_smoother

Header-only C++17 kinematic path smoother built on Ceres and `esdf_core`.

The public API is intentionally small:

- `KinematicPathSmoother::initialize()` stores solver-level settings.
- `KinematicPathSmoother::smooth()` accepts a read-only `SmoothingRequest`.
- `SmoothingResult` returns both the direct optimized knots and the final upsampled path.

Input path points use `(x, y, direction_sign)`, where a negative sign means reverse motion.
Output path points use `(x, y, yaw)`.

```bash
cmake -S my/kinematic_path_smoother -B /tmp/kinematic_path_smoother_build
cmake --build /tmp/kinematic_path_smoother_build
ctest --test-dir /tmp/kinematic_path_smoother_build --output-on-failure
```

Web demo:

```bash
cd my/kinematic_path_smoother
./run_web_app.sh
```

`BUILD_PYTHON=ON` builds a nanobind module named `py_kinematic_path_smoother`.
The run script defaults to `/home/zks/.venv/bin/python3` because that environment
contains nanobind and Flask; override it with `KINEMATIC_SMOOTHER_PYTHON=/path/to/python`.
