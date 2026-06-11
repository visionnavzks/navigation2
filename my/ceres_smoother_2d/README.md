# Ceres 2D Path Smoother with ESDF

Standalone C++ 2D path smoother using Ceres Solver and Euclidean Signed Distance Field (ESDF) for obstacle avoidance. No ROS dependency.

## Features

- **ESDF Map**: Exact Euclidean Signed Distance Transform (Felzenszwalb & Huttenlocher, O(n) per row/col)
- **Ceres Optimization**: AutoDiffCostFunction with Jet-compatible bi-linear ESDF interpolation
- **Cost Terms** (all auto-diffed via Jet; obstacle cost uses bounded
  bilinear ESDF interpolation to avoid overshoot at obstacle boundaries):
  - **Smoothness** — jerk penalty (second-order finite difference)
  - **Curvature** — hinge on local turning angle ≤ `min_turning_radius`
  - **Reference** — spring toward the A* reference path (`w_reference`)
  - **Length** — elastic-band squared segment length `Σ‖Δs‖²` (rubber-band force)
  - **Obstacle** — ESDF hinge, pushes path away from walls
- **Optional arc-length resampling** before and/or after the optimization:
  input resampling is on by default, output resampling is off by default.
  `target_spacing` only drives resampling, never the cost.
- **nanobind Python bindings**: Use the smoother from Python
- **Matplotlib visualization**: 3-panel output (occupancy, ESDF heatmap, clearance profile)

## Dependencies

- CMake >= 3.14
- Eigen3 >= 3.4
- Ceres Solver >= 2.0
- stb_image (bundled in `thirdparty/stb/`)
- nanobind (Python bindings)
- matplotlib, numpy (Python demo)

## Build

```bash
cd my/ceres_smoother_2d
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### C++ Demo
```bash
./ceres_smoother_2d_demo [path_to_occupancy_map.png]
```

### Python Demo
```bash
python3 python/demo.py [path_to_occupancy_map.png]
```

Output: `build/smooth_result.png` (3-panel visualization)

## Python API

```python
import ceres_smoother_2d as cs2d

# Load map
esdf_map = cs2d.ESDFMap("map.png", resolution=0.05)

# Configure
params = cs2d.SmootherParams()
params.w_smooth = 100.0
params.safety_margin = 0.3

# Smooth
smoother = cs2d.PathSmoother2D(params)
result = smoother.smooth(x_list, y_list, esdf_map)

print(result.x, result.y)  # smoothed coordinates
```

## Architecture

```
include/
├── esdf_map.hpp              # ESDF computation and querying
└── ceres_smoother_2d.hpp     # Ceres-based 2D path smoother
src/
├── main.cpp                  # C++ demo
├── nanobind_module.cpp       # nanobind Python bindings
└── stb_image_impl.cpp        # stb_image implementation
python/
└── demo.py                   # Python demo with matplotlib visualization
thirdparty/stb/
├── stb_image.h               # PNG loading
└── stb_image_write.h         # PNG output
```

## Key Design Decisions

1. **Per-node 2D parameter blocks** (`std::array<double,2>`) for sparsity preservation
2. **Jet-compatible bi-linear interpolation** — gradient flows through interpolation weights via Ceres AutoDiff
3. **Signed distance field**: positive = free space (distance to nearest obstacle), negative = inside obstacle
4. **Hinge obstacle cost**: penalizes when `distance < safety_margin + robot_radius`; the two are summed so adding a non-zero `robot_radius` automatically inflates the soft-repulsion zone for the cost function (it is also used by A* to inflate the grid via `ESDF < robot_radius` → obstacle).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_smooth` | 100.0 | Smoothness weight |
| `w_max_curvature` | 50.0 | Curvature constraint weight |
| `w_reference` | 10.0 | Reference tracking weight |
| `w_length` | 10.0 | Rubber-band inter-point distance weight |
| `w_obstacle` | 200.0 | ESDF obstacle avoidance weight (hinge) |
| `min_turning_radius` | 0.5 m | Minimum turning radius |
| `safety_margin` | 1.0 m | Desired minimum clearance from robot edge to obstacles |
| `robot_radius` | 0.0 m | Robot inscribed radius. Effective clearance = `safety_margin + robot_radius` |
| `target_spacing` | 0.3 m | Desired inter-point spacing for optional pre/post resampling; it is not part of the optimization cost |
| `resample_after_smooth` | **`false`** | When `true`, uniformly resample the smoothed path along arc length so consecutive output points are ~`target_spacing` meters apart. Start/goal preserved exactly. |
| `resample_before_smooth` | **`true`** | When `true`, uniformly resample the **input** reference path to ~`target_spacing` before optimization. Recommended when the upstream path (e.g. A*) has uneven point density. |
| `max_iterations` | 200 | Ceres solver max iterations |
