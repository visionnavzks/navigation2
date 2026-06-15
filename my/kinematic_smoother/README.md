# Constrained Smoother (Standalone)

`my/kinematic_smoother` is a standalone extraction of the Navigation2 constrained smoother experiment package.

The current standalone package keeps one smoothing backend only: `KinematicSmoother`.

## What Is In This Package

- A C++ kinematic path smoother built on Ceres and Eigen.
- A lightweight C++ A* planner plus ESDF utilities.
- A Flask-based Web Lab for interactive planning, smoothing, and validation inspection.

## Current Behavior Contracts

1. Input paths use `(x, y, direction_sign)`, not `(x, y, yaw)`.
2. Output paths are returned explicitly; their third component is `yaw` in radians.
3. `KinematicSmoother` is the only retained C++ / pybind smoothing class.
4. `cost_check_points` is consumed directly as `(x_local, y_local, weight)` triples.
5. `reversing_enabled=false` forces the kinematic backend to treat all segments as forward motion.
6. `max_curvature` is curvature in `1 / m`, not minimum turning radius.

`KinematicSmoother::smooth(...)` returns a `SmootherResult` carrying `candidate_path`,
`smoothed_path`, `optimized_knot_count`, and `target_spacing`.

## Core Code Layout

- `include/kinematic_smoother/kinematic_smoother.hpp`
	- Public smoother entrypoint and per-call execution lifecycle.
- `include/kinematic_smoother/kinematic_smoother_problem_builder.hpp`
	- ESDF preparation, state expansion, residual assembly, bounds, and unpacking.
- `include/kinematic_smoother/kinematic_smoother_costs.hpp`
	- Kinematic residual functors.
- `include/kinematic_smoother/smoother_request.hpp`
	- Shared request object for one smoothing call.
- `include/kinematic_smoother/smoother_validator.hpp`
	- Post-solve hard validation.
- `web/app.py`
	- Flask API, Web Lab scene state, and response shaping.

## Runtime Flow

### Native smoothing flow

```text
smooth()
	-> Run::prepare()
		-> KinematicSmootherProblemBuilder::initializeEsdfValues()
		-> KinematicSmootherProblemBuilder::buildProcessedPath()
		-> KinematicSmootherProblemBuilder::buildProblem()
		-> KinematicSmootherProblemBuilder::applyBounds()
	-> Run::solve()
	-> Run::finalize()
		-> KinematicSmootherProblemBuilder::unpackPath()
		-> SmootherValidator::validateKinematicSolution()
```

### Web `/api/plan` flow

```text
request JSON
	-> PlanRequestConfig.from_payload()
	-> build footprint / solver params
	-> planner stage (manual reference or A*)
	-> KinematicSmoother.try_*()
	-> rectangle validation
	-> pipeline payload + frontend response JSON
```

## Failure Model

- Invalid inputs fail immediately with structured exceptions such as `InvalidPath`, `InvalidCostmap`, or `PrecomputedEsdfSizeMismatch`.
- Solver and post-validation failures flow through `throwOrStoreSmoothingFailure(...)`.
- Web API failures return a structured `error` object.
- Web smoothing failures may still return a candidate path for visualization through `smooth_error`, `candidate_rectangle_validation`, and `final_rectangle_validation`.

See `docs/error-codes.md` for the stable code catalog.

## Build

### C++

```bash
cmake -S . -B build
cmake --build build --parallel
```

### Python bindings

```bash
cmake -S . -B build-py \
	-DBUILD_TESTS=OFF \
	-DBUILD_PYTHON=ON \
	-Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build-py --target py_kinematic_smoother --parallel
```

### Tests

```bash
cmake -S . -B build-test -DBUILD_TESTS=ON -DBUILD_PYTHON=OFF
cmake --build build-test --target test_smoother --parallel
ctest --test-dir build-test -R test_smoother
```

## Web Lab

```bash
./run_web_app.sh
```

The Web Lab is kinematic-only. It no longer exposes any backend switch.

### Key endpoints

- `GET /api/costmap`
	- Returns the current costmap grid, optional ESDF grid, and scene metadata.
- `POST /api/obstacles`
	- Rebuilds the obstacle layout from `obstacle_rects_cells`.
- `POST /api/plan`
	- Runs A* (or a provided manual reference), then runs the kinematic smoother, then applies final rectangle validation.

## Docs

- `docs/index.md`
- `docs/package-guide.md`
- `docs/KINEMATIC_SMOOTHER_DESIGN.md`
- `docs/error-codes.md`

### Local preview

```bash
./run_docs.sh
```

## Original Source

This standalone package was extracted from the Navigation2 repository
(<https://github.com/ros-navigation/navigation2>). The smoothing backend
descends from the upstream package `nav2_constrained_smoother`, which
exposes two backends: a `ConstrainedSmoother` (geometric, retained only as
a historical reference in the upstream source) and the kinematic smoother
originally implemented as `nav2_smoother` / `Smoother` and now kept here
under the class name `KinematicSmoother`. The bundled A* planner and
ESDF helpers share lineage with `nav2_smoother` and `nav2_costmap_2d`
respectively. This package drops the ROS 2 build glue, the
`nav2_constrained_smoother` ROS `package.xml`, and the geometric
smoother backend; only the kinematic smoother is shipped.

## License

Apache License 2.0
