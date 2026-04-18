# Constrained Smoother (Standalone)

A standalone, ROS-independent path smoother extracted from `nav2_constrained_smoother`.

This library uses **Ceres Solver** for nonlinear optimization and **Eigen** for linear algebra to smooth robot paths with constraints on curvature, smoothness, distance from original path, and costmap obstacle avoidance.

## Dependencies

- **Ceres Solver** (>= 2.0 recommended)
- **Eigen3**
- **Google Test** (for tests, optional)

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run Tests

```bash
cd build
ctest --output-on-failure
```

## Usage

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

## Original Source

Extracted from [navigation2/nav2_constrained_smoother](https://github.com/ros-navigation/navigation2/tree/main/nav2_constrained_smoother).

## License

Apache License 2.0
