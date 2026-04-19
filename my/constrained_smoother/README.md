# Constrained Smoother (Standalone)

A standalone, ROS-independent extraction of `nav2_constrained_smoother` with three pieces bundled together:

- A C++ constrained path smoother built on Ceres and Eigen.
- A lightweight C++ A* planner plus ESDF utilities.
- A Flask-based Web Lab for inspecting costmaps, planner output, and smoother behavior.

The smoother solves a nonlinear least-squares problem over 2D path geometry with smoothness, curvature, distance-to-reference, and obstacle-clearance terms.

## Key API Conventions

These are the most important behavior contracts in the current standalone implementation.

1. Input paths use `(x, y, direction_sign)`, not `(x, y, yaw)`.
	 - `direction_sign` should typically be `+1` for forward and `-1` for reverse.
2. Output paths overwrite the third component with `yaw` in radians.
	 - After `smooth()` returns, `path[i][2]` is no longer a direction sign.
3. `SmootherParams` expects square-root weights.
	 - Set `smooth_weight_sqrt = sqrt(weight)`, `costmap_weight_sqrt = sqrt(weight)`, and so on.
4. `cost_check_points` is used as-is.
	 - The standalone build does not preprocess footprint sample weights the way the ROS plugin layer does.
	 - Pass triples of `(x_local, y_local, weight)` in the robot local frame.
5. `reversing_enabled` is kept for compatibility but is not currently read by the standalone smoother.
6. `max_curvature` is curvature in `1 / m`, not minimum turning radius.

## Dependencies

- Ceres Solver
- Eigen3
- Google Test (optional, for tests)
- pybind11 (optional, for Python bindings)
- Flask and NumPy (optional, for the Web Lab)

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --parallel
```

### Build with Python bindings

```bash
mkdir -p build
cd build
cmake .. -DBUILD_PYTHON=ON -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build . --parallel
```

This produces `py_constrained_smoother.*.so` in `build/`.

## Run Tests

```bash
cd build
ctest --output-on-failure
```

## C++ Usage

```cpp
#include <cmath>
#include <vector>

#include "Eigen/Core"
#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/smoother.hpp"

constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);
// Fill costmap with obstacle data here.

constrained_smoother::SmootherParams params;
params.smooth_weight_sqrt = std::sqrt(50.0);
params.costmap_weight_sqrt = std::sqrt(0.015);
params.distance_weight_sqrt = std::sqrt(0.0);
params.curvature_weight_sqrt = std::sqrt(30.0);
params.curvature_rate_weight_sqrt = std::sqrt(5.0);  // optional D3 curvature-rate proxy
params.max_curvature = 1.0 / 0.4;  // minimum turning radius = 0.4 m
params.keep_start_orientation = true;
params.keep_goal_orientation = true;

constrained_smoother::OptimizerParams opt_params;
opt_params.max_iterations = 50;

constrained_smoother::Smoother smoother;
smoother.initialize(opt_params);

// On input, z is direction_sign (+1 / -1).
std::vector<Eigen::Vector3d> path = {
	{0.5, 0.5, 1.0},
	{1.0, 0.7, 1.0},
	{1.6, 1.0, 1.0},
	{2.2, 1.4, 1.0},
};

Eigen::Vector2d start_dir(1.0, 0.0);
Eigen::Vector2d end_dir(1.0, 0.0);

smoother.smooth(path, start_dir, end_dir, &costmap, params);

// After smoothing, z has been rewritten to yaw in radians.
for (const auto & pose : path) {
	const double yaw = pose.z();
	(void)yaw;
}
```

## Python Usage

```python
import math
import numpy as np
import py_constrained_smoother as pcs

costmap = pcs.Costmap2D(100, 100, 0.05, 0.0, 0.0)

params = pcs.SmootherParams()
params.smooth_weight_sqrt = math.sqrt(50.0)
params.costmap_weight_sqrt = math.sqrt(0.015)
params.distance_weight_sqrt = math.sqrt(0.0)
params.curvature_weight_sqrt = math.sqrt(30.0)
params.curvature_rate_weight_sqrt = math.sqrt(5.0)  # optional D3 curvature-rate proxy
params.max_curvature = 2.5
params.keep_start_orientation = True
params.keep_goal_orientation = True

opt_params = pcs.OptimizerParams()
opt_params.max_iterations = 50

smoother = pcs.Smoother()
smoother.initialize(opt_params)

# On input, the third component is direction_sign, not yaw.
path = [
		np.array([0.5, 0.5, 1.0]),
		np.array([1.0, 0.7, 1.0]),
		np.array([1.6, 1.0, 1.0]),
		np.array([2.2, 1.4, 1.0]),
]
start_dir = np.array([1.0, 0.0])
end_dir = np.array([1.0, 0.0])

smoothed = smoother.smooth(path, start_dir, end_dir, costmap, params)
optimized_knot_count = smoother.get_last_optimized_knot_count()

# On output, smoothed[i][2] is yaw in radians.
print(optimized_knot_count, float(smoothed[0][2]))
```

## Web Lab

The Web Lab is an interactive scene editor and visualizer around the C++ A* planner and smoother.

Current behavior:

- The map is a synthetic 20 m x 20 m costmap with draggable rectangular lethal obstacles plus inflated safety cells.
- Start and goal markers are draggable.
- Obstacle rectangles are draggable.
- Left-drag on empty space pans the camera.
- Double-click or use Reset View to restore full-map framing.
- Slider and toggle changes trigger automatic replanning.
- The toolbar can switch between costmap and ESDF views.
- The sidebar exposes heading constraints, planner penalty settings, footprint mode, solver controls, and live metrics.
- The frontend shows a cursor inspector, optimized-point inspector, and a curvature chart for the current smoothed path.

### Run the Web Lab

```bash
# 1. Build the Python bindings first.
# 2. Activate your environment if needed.
cd my/constrained_smoother
python3 web/app.py
```

Open `http://localhost:5002` in your browser.

When launched from `my/constrained_smoother`, `web/app.py` adds both the package root and `build/` directory to `sys.path`, so an extra `PYTHONPATH` export is usually not required once the pybind module has been built.

### Web API Summary

- `GET /api/costmap`
	- Returns the current costmap grid, optional ESDF grid, and map metadata.
- `POST /api/obstacles`
	- Accepts `obstacle_rects_cells` and rebuilds the scene costmap.
- `POST /api/plan`
	- Runs A* and then the constrained smoother.
	- Accepts start and goal positions, start and goal yaw constraints, footprint mode, planner penalty settings, and solver parameters.
	- Returns raw A* points, downsampled reference points, smoothed points, `opt_theta`, timing, lengths, and optimized knot counts.

The smoother route currently derives its planner safe distance from the shared hinge-loss threshold, and in point-robot mode it adds the point-robot radius on top of that shared threshold.
The standalone A* now also performs hard footprint feasibility checks: point-robot mode rejects cells whose ESDF clearance is smaller than the configured radius, and rectangle mode rejects any axis-aligned pose whose box footprint overlaps lethal cells. Rectangle A* checking intentionally ignores yaw.

## Original Source

Extracted from [navigation2/nav2_constrained_smoother](https://github.com/ros-navigation/navigation2/tree/main/nav2_constrained_smoother).

## License

Apache License 2.0
