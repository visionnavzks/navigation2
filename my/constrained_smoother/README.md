# Constrained Smoother (Standalone)

A standalone, ROS-independent path smoother extracted from `nav2_constrained_smoother`.

This library uses **Ceres Solver** for nonlinear optimization and **Eigen** for linear algebra to smooth robot paths with constraints on curvature, smoothness, distance from original path, and costmap obstacle avoidance.

## Dependencies

- **Ceres Solver** (>= 2.0 recommended)
- **Eigen3**
- **Google Test** (for tests, optional)
- **pybind11** (for Python bindings, optional)

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build with Python bindings

```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j$(nproc)
```

This produces `py_constrained_smoother.*.so` in `build/`.

## Run Tests

```bash
cd build
ctest --output-on-failure
```

## C++ Usage

```cpp
#include "constrained_smoother/smoother.hpp"
#include "constrained_smoother/costmap2d.hpp"

// 1. Create a costmap
constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);
// ... fill costmap with obstacle data ...

// 2. Configure parameters
constrained_smoother::SmootherParams params;
params.smooth_weight_sqrt = std::sqrt(2000000.0);
params.costmap_weight_sqrt = std::sqrt(0.015);
params.max_curvature = 1.0 / 0.4;  // minimum turning radius = 0.4m
// ... set other params ...

constrained_smoother::OptimizerParams opt_params;

// 3. Create and initialize smoother
constrained_smoother::Smoother smoother;
smoother.initialize(opt_params);

// 4. Smooth a path
// path is vector of (x, y, direction_sign) where direction_sign is +1 (forward) or -1 (reverse)
std::vector<Eigen::Vector3d> path = ...;
Eigen::Vector2d start_dir(1.0, 0.0);
Eigen::Vector2d end_dir(1.0, 0.0);

smoother.smooth(path, start_dir, end_dir, &costmap, params);
```

## Python Usage

```python
import py_constrained_smoother as pcs
import numpy as np, math

# Create costmap
costmap = pcs.Costmap2D(100, 100, 0.05, 0.0, 0.0)

# Configure smoother
params = pcs.SmootherParams()
params.smooth_weight_sqrt = math.sqrt(2000000.0)
params.costmap_weight_sqrt = math.sqrt(0.015)
params.max_curvature = 2.5

opt_params = pcs.OptimizerParams()
smoother = pcs.Smoother()
smoother.initialize(opt_params)

# Smooth a path  —  list of (x, y, direction_sign) numpy arrays
path = [np.array([0.5 + i*0.1, 2.5, 1.0]) for i in range(10)]
start_dir = np.array([1.0, 0.0])
end_dir   = np.array([1.0, 0.0])

smoothed = smoother.smooth(path, start_dir, end_dir, costmap, params)
```

## Web Demo (A* + Constrained Smoother)

An interactive web demo that lets you click start/goal points on a 2D costmap,
runs A\* to generate a reference path, and optimizes it with the constrained smoother.

### Run the demo

```bash
# 1. Build Python bindings (see above)
# 2. Start the Flask server
cd my/constrained_smoother
PYTHONPATH="$(pwd)/build:$(pwd)/web:$PYTHONPATH" python3 web/app.py
# 3. Open http://localhost:5002 in your browser
```

- **Click** on the map to set **Start** (green) then **Goal** (red).
- A\* finds a grid path (blue), which is downsampled (orange) and fed to the constrained smoother (pink).
- Adjust smoother weights in the sidebar and re-run.

## Original Source

Extracted from [navigation2/nav2_constrained_smoother](https://github.com/ros-navigation/navigation2/tree/main/nav2_constrained_smoother).

## License

Apache License 2.0
