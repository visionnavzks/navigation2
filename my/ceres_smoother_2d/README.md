# Ceres 2D Path Smoother with ESDF

**Standalone C++ 2D path smoother using Ceres Solver and Euclidean Signed Distance Field (ESDF) for obstacle avoidance. No ROS dependency.**

<p align="center">
  <img src="doc/smooth_result.png" width="800" alt="3-panel visualization: Occupancy Map + Paths, ESDF + Smoothed Path, Obstacle Clearance Profile" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Optimization Pipeline](#optimization-pipeline)
  - [Cost Functions](#cost-functions)
  - [ESDF Map](#esdf-map)
  - [A\* Path Planning](#a-path-planning)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Build](#build)
- [Usage](#usage)
  - [C++ Demo](#c-demo)
  - [Python Demo](#python-demo)
  - [Web Interface](#web-interface)
- [Python API Reference](#python-api-reference)
  - [ESDFMap](#esdfmap)
  - [SmootherParams](#smootherparams)
  - [PathSmoother2D](#pathsmoother2d)
  - [SmootherResult](#smootherresult)
  - [A\* Search](#a-search)
  - [Resampling Utility](#resampling-utility)
- [Parameters](#parameters)
- [Performance](#performance)
- [Key Design Decisions](#key-design-decisions)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

`ceres_smoother_2d` is a high-performance, standalone C++ library (with Python bindings) that transforms rough reference paths — such as those produced by A\* or grid-based planners — into smooth, kinematically feasible, and obstacle-free trajectories.

It solves a multi-objective nonlinear least-squares problem using [Ceres Solver](http://ceres-solver.org/), with five distinct cost terms that jointly enforce smoothness, curvature limits, obstacle clearance, path length minimization, and reference tracking. The obstacle representation uses an exact **Euclidean Signed Distance Field (ESDF)** with Jet-compatible bilinear interpolation, enabling seamless integration with Ceres' automatic differentiation.

### Typical Workflow

```
Occupancy Map (PNG)
        │
        ▼
   ┌─────────┐
   │  ESDF   │  ← Exact distance transform (Felzenszwalb & Huttenlocher)
   │   Map   │
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │  A*     │  ← Fast C++ 8-connected search with robot-radius inflation
   │ Search  │
   └────┬────┘
        │  rough reference path (x, y)
        ▼
   ┌─────────────────┐
   │  Ceres Smoother │  ← Multi-stage nonlinear optimization
   │  (5 cost terms) │
   └────┬────────────┘
        │  smoothed path
        ▼
   ┌─────────────┐
   │  Resample   │  ← Optional uniform arc-length resampling
   └────┬────────┘
        │
        ▼
   Smooth, safe trajectory
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Exact ESDF** | Felzenszwalb & Huttenlocher O(n) signed distance transform; positive = free, negative = inside obstacle |
| **Ceres AutoDiff** | All gradients via Ceres Jet — no manual Jacobians, no finite differences |
| **5 Cost Terms** | Smoothness, curvature, reference, elastic-band length, ESDF obstacle (hinge + penetration) |
| **Multi-Stage Solving** | Progressive ramp of obstacle weight to avoid local minima |
| **Bilinear ESDF Interpolation** | Jet-compatible, bounded (no overshoot at obstacle boundaries unlike BiCubic) |
| **Sparse Solver** | `SPARSE_NORMAL_CHOLESKY` with SuiteSparse for fast solve on 1000+ point paths |
| **Robot-Radius Inflation** | A\* grid is inflated by `robot_radius` for circular footprint planning |
| **Arc-Length Resampling** | Optional pre/post uniform resampling; start/goal preserved exactly |
| **nanobind Python Bindings** | Full Python API with identical functionality to C++ |
| **Web Demo** | Flask + Plotly interactive interface for real-time path planning |
| **No ROS Dependency** | Pure C++17, standalone library usable in any project |

---

## How It Works

### Optimization Pipeline

The smoother minimizes a weighted sum of five cost terms over a discretized path $P = \{p_0, p_1, \ldots, p_{N-1}\}$ where each $p_i = (x_i, y_i) \in \mathbb{R}^2$:

$$
\min_{P} \; J_{\text{smooth}} + J_{\text{curvature}} + J_{\text{reference}} + J_{\text{length}} + J_{\text{obstacle}}
$$

**Start and goal are fixed** ($p_0$ and $p_{N-1}$ are held constant). All intermediate points are optimized jointly.

The solver uses a **multi-stage** approach for the obstacle weight: it starts with a low $w_{\text{obstacle}}$ (allowing the path to find a globally smooth shape) and progressively increases it to the target value. This prevents the optimizer from getting trapped in poor local minima near walls.

### Cost Functions

#### 1. Smoothness ($J_{\text{smooth}}$)

Penalizes the **second-order finite difference** (discrete acceleration):

$$
r_i = \sqrt{w_{\text{smooth}}} \cdot (p_{i+1} - 2p_i + p_{i-1})
$$

This produces a tridiagonal Hessian structure for efficient sparse solving. It directly minimizes path "jerk" — the rate of curvature change — yielding visually smooth trajectories.

#### 2. Curvature ($J_{\text{curvature}}$)

A **turning-angle hinge loss** — penalizes when the actual angle exceeds the allowed limit:

$$
\theta = \text{atan2}\!\big(\sqrt{(v_1 \times v_2)^2 + \epsilon},\; v_1 \cdot v_2\big)
$$

$$
r_i = \sqrt{w_{\text{curv}}} \cdot \max\!(0,\; \theta - \kappa_{\max} \cdot d_s)
$$

where $v_1 = p_i - p_{i-1}$, $v_2 = p_{i+1} - p_i$, $d_s = \frac{\|v_1\| + \|v_2\|}{2}$, and $\kappa_{\max} = \frac{1}{r_{\min}}$.

Uses $\sqrt{(\text{cross})^2 + \epsilon}$ instead of $|\text{cross}|$ for smoothness at 0 under Ceres AutoDiff. The unsigned turning angle $\theta \in [0, \pi]$ is directly compared against the geometric limit $\kappa_{\max} \cdot d_s$.

#### 3. Reference ($J_{\text{reference}}$)

A **spring-like anchor** to the original A\* reference path:

$$
r_i = \sqrt{w_{\text{ref}}} \cdot (p_i - p_i^{\text{ref}})
$$

This prevents the optimizer from deviating too far from the planner's intended route when obstacle/length weights are strong.

#### 4. Elastic-Band Length ($J_{\text{length}}$)

Minimizes the **sum of squared inter-point distances** (rubber-band force):

$$
r_i = \sqrt{w_{\text{length}}} \cdot (p_{i+1} - p_i)
$$

Ceres reports $0.5 \sum \|r_i\|^2$, so this contributes $0.5 \cdot w_{\text{length}} \cdot \sum \|p_{i+1} - p_i\|^2$. The key advantages over a target-spacing spring:
- **Constant Jacobian** → Ceres converges in very few iterations
- **No rest-length conflict** with fixed start/goal points
- Encourages **uniform spacing** as a side effect

> **Note**: `resample_spacing` is used **only** by the optional resample stages, never by the optimization cost itself.

#### 5. Obstacle ($J_{\text{obstacle}}$)

Two complementary terms:

**Soft hinge** (pushes away from safety boundary):
$$
r_0 = \sqrt{w_{\text{obs}}} \cdot \max\!\big(0,\; d_{\text{safe}} - d_{\text{esdf}}(p)\big)
$$

**Penetration penalty** (pulls out of walls):
$$
r_1 = \sqrt{w_{\text{pen}}} \cdot \max\!\big(0,\; -d_{\text{esdf}}(p)\big)
$$

where $d_{\text{safe}} = \texttt{safety\_margin} + \texttt{robot\_radius}$ and $d_{\text{esdf}}(p)$ is the bilinear-interpolated ESDF distance at point $p$.

The penetration term provides **independent weight tuning** for deep-wall penalties: $w_{\text{obs}}$ controls the soft repulsion near the safety boundary, while $w_{\text{pen}}$ (typically 100× larger) controls the strong pull-out force deep inside walls. Both are quadratic in penetration depth, but the 100× weight ratio means the penetration term dominates at depth, ensuring the optimizer cannot settle deep inside obstacles.

### ESDF Map

The ESDF is computed using the **Felzenszwalb & Huttenlocher** algorithm (2012):

> *"Distance Transforms of Sampled Functions"*, IEEE TPAMI

This gives **exact** Euclidean distances in $O(n)$ per row/column — no approximation, no iterative diffusion.

**Key properties:**
- **Signed**: positive in free space (distance to nearest obstacle), negative inside obstacles
- **Resolution-aware**: distances are in world coordinates (meters)
- **Bilinear interpolation**: Jet-compatible for seamless Ceres AutoDiff; bounded by the min/max of the 4 nearest cells (avoids overshoot at sharp obstacle boundaries)
- **Convention**: PNG row 0 maps to world $y_{\max}$ (ROS `map_server` compatible)

### A\* Path Planning

The built-in A\* implementation provides a fast reference path for the smoother:

- **8-connected grid** with Euclidean step cost (1 cardinal, $\sqrt{2}$ diagonal)
- **Robot-radius inflation**: cells with $\text{ESDF} < r_{\text{robot}}$ are treated as obstacles
- **Binary heap + lazy deletion** for efficient frontier management
- **Flat arrays** for g-score / came-from / closed (cache-friendly)
- **~100x faster** than a Python fallback on typical occupancy maps

---

## Project Structure

```
ceres_smoother_2d/
├── CMakeLists.txt              # Build system
├── README.md                   # This file
├── include/
│   ├── ceres_smoother_2d.hpp   # Core: SmootherParams, cost structs, PathSmoother2D
│   ├── esdf_map.hpp            # ESDF computation and bilinear querying
│   └── astar.hpp               # C++ A* with robot-radius inflation
├── src/
│   ├── main.cpp                # C++ demo executable
│   ├── nanobind_module.cpp     # Python bindings (nanobind)
│   └── stb_image_impl.cpp      # stb_image implementation unit
├── python/
│   ├── demo.py                 # Matplotlib visualization demo
│   ├── app.py                  # Flask web server + REST API
│   └── templates/
│       └── index.html          # Plotly-based interactive web frontend
├── tests/
│   ├── test_cpp.cpp            # C++ unit tests (ESDF, cost functions, end-to-end)
│   ├── test_python.py          # Python integration tests
│   └── test_web_api.py         # Flask API endpoint tests
├── thirdparty/
│   └── stb/
│       ├── stb_image.h         # PNG loading (bundled, no external dependency)
│       └── stb_image_write.h   # PNG output
├── run.sh                      # C++ demo runner
├── run_python.sh               # Python demo runner
├── run_tests.sh                # Test suite runner
└── run_web.sh                  # Web interface launcher
```

---

## Dependencies

| Dependency | Version | Notes |
|------------|---------|-------|
| **CMake** | >= 3.14 | Build system |
| **Eigen3** | >= 3.4 | Linear algebra |
| **Ceres Solver** | >= 2.0 | Nonlinear least-squares optimizer |
| **SuiteSparse** | (bundled with Ceres) | Sparse Cholesky factorization |
| **stb_image** | bundled in `thirdparty/stb/` | PNG loading |
| **nanobind** | any recent | Python bindings |
| **Python** | >= 3.8 | For Python API, demo, and web interface |
| **NumPy** | any recent | Array operations in Python |
| **Matplotlib** | any recent | Visualization (demo only) |
| **Flask** | any recent | Web server (web demo only) |

---

## Build

### Quick Build

```bash
cd my/ceres_smoother_2d
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces:
- `build/ceres_smoother_2d_demo` — C++ demo executable
- `build/ceres_smoother_2d_tests` — C++ test suite
- `build/ceres_smoother_2d.*.so` — nanobind Python module

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (`Release` enables `-O3 -march=native`) |

---

## Usage

### C++ Demo

```bash
# With default map (if maps/occupancy_map.png exists in parent directory)
./build/ceres_smoother_2d_demo

# With custom map
./build/ceres_smoother_2d_demo /path/to/occupancy_map.png
```

**Output**: `build/smooth_result.png` (3-panel visualization)

### Python Demo

```bash
# Using the run script (handles PYTHONPATH setup)
./run_python.sh [path_to_occupancy_map.png]

# Or directly
PYTHONPATH=build python3 python/demo.py [path_to_occupancy_map.png]
```

**Output**: `build/smooth_result.png`

### Web Interface

```bash
# Start the web server
./run_web.sh

# Or manually
MAP_PATH=/path/to/occupancy_map.png RESOLUTION=0.05 \
  PYTHONPATH=build python3 python/app.py
```

Then open **http://127.0.0.1:5000/** in your browser.

The web interface provides:
- **Interactive map visualization** with Plotly (pan, zoom, click-to-plan)
- **Click-to-set start/goal** on the occupancy map
- **Real-time A\* + smoothing** with configurable parameters
- **ESDF heatmap overlays** with multiple color schemes
- **Clearance profile plot** along the smoothed path

#### Web API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/costmap` | Returns map metadata + base64-encoded PNG images |
| `POST` | `/api/plan` | Plans a path from start to goal |

**`POST /api/plan` request body:**
```json
{
  "start": [x, y],
  "goal": [x, y],
  "robot_radius": 0.5,
  "safety_margin": 1.0,
  "downsample": 3
}
```

**Response fields**: `found`, `success`, `raw_x`, `raw_y`, `smooth_x`, `smooth_y`, `raw_points`, `smooth_points`, `smooth_length`, `min_clearance`, `plan_ms`, `smooth_ms`, `start_ok`, `goal_ok`, `reason`.

---

## Python API Reference

```python
import ceres_smoother_2d as cs2d
```

### ESDFMap

```python
# Load from PNG file
esdf_map = cs2d.ESDFMap(
    path="map.png",           # Grayscale PNG (0=obstacle, 255=free)
    resolution=0.05,          # Meters per pixel
    origin_x=0.0,             # World x of grid pixel (0,0)
    origin_y=0.0,             # World y of grid pixel (0,0)
    obstacle_thresh=127       # Pixels <= this are obstacles
)

# Construct from raw data
esdf_map = cs2d.ESDFMap(
    occupancy=[0, 0, 1, ...], # Flat array (0=free, 1=obstacle)
    width=100, height=100,
    resolution=0.05
)

# Query distance
dist = esdf_map.get_distance(wx, wy)  # Signed; positive=free, negative=obstacle

# Check bounds
in_bounds = esdf_map.in_bounds(wx, wy, margin=0.0)

# Properties
esdf_map.width          # Grid width (pixels)
esdf_map.height         # Grid height (pixels)
esdf_map.resolution     # Meters per pixel
esdf_map.origin_x       # World x origin
esdf_map.origin_y       # World y origin
esdf_map.world_width    # Width in meters
esdf_map.world_height   # Height in meters

# Export grids
esdf_array = esdf_map.get_esdf_array()      # Flat list (row-major)
occ_array = esdf_map.get_occupancy_array()   # Flat list (row-major)
```

### SmootherParams

```python
params = cs2d.SmootherParams()

# Solver settings
params.max_iterations = 100
params.max_time_seconds = 0.5
params.verbose = False

# Cost weights
params.w_smooth = 100.0           # Smoothness (jerk penalty)
params.w_max_curvature = 1000.0   # Curvature constraint
params.w_reference = 0.0          # Reference path tracking
params.w_length = 1.0             # Elastic-band length
params.w_obstacle = 1.0           # Obstacle avoidance (hinge)
params.w_penetration = 1000.0     # Inside-obstacle penalty

# Geometric constraints
params.min_turning_radius = 0.2   # meters
params.safety_margin = 1.0        # meters (clearance from robot edge)
params.robot_radius = 0.5         # meters (inscribed radius)

# Resampling
params.resample_before_smooth = True
params.resample_after_smooth = False
params.resample_spacing = 0.3
```

### PathSmoother2D

```python
smoother = cs2d.PathSmoother2D(params)

result = smoother.smooth(
    x=[1.0, 2.0, 3.0, ...],   # Input x coordinates (meters)
    y=[5.0, 5.2, 5.5, ...],   # Input y coordinates (meters)
    map=esdf_map               # ESDFMap instance
)
```

### SmootherResult

```python
result.success        # bool: whether optimization converged
result.x              # list[float]: smoothed x coordinates
result.y              # list[float]: smoothed y coordinates
result.final_cost     # float: final objective value
result.solve_time_ms  # float: total solve time in milliseconds
result.iterations     # int: total iterations across all stages
result.report         # str: detailed Ceres solver report
```

### A\* Search

```python
# Run A* on the ESDFMap's occupancy grid
astar_result = cs2d.astar_solve(
    map=esdf_map,
    sx=1.0, sy=5.0,       # Start (world coordinates, meters)
    gx=10.0, gy=8.0,      # Goal
    robot_radius=0.5       # Inflates obstacles by this radius
)

astar_result.success     # bool
astar_result.x           # list[float]: path x coordinates
astar_result.y           # list[float]: path y coordinates
astar_result.expansions  # int: nodes expanded
astar_result.time_ms     # float: search time in ms
```

### Resampling Utility

```python
# Uniform arc-length resampling
rx, ry = cs2d.resample_path_by_arc_length(
    x=[1.0, 2.0, 5.0, ...],
    y=[3.0, 3.1, 4.0, ...],
    target_spacing=0.3    # meters
)
# Returns (list[float], list[float])
# Endpoints are preserved exactly; intermediate points interpolated.
```

### Complete Example

```python
import ceres_smoother_2d as cs2d

# 1. Load map
esdf_map = cs2d.ESDFMap("maps/occupancy_map.png", resolution=0.05)

# 2. Plan with A*
astar = cs2d.astar_solve(esdf_map, sx=8.0, sy=28.0, gx=65.0, gy=28.0,
                         robot_radius=0.5)
if not astar.success:
    print("No path found!")
    exit(1)

# 3. Configure smoother
params = cs2d.SmootherParams()
params.w_smooth = 100.0
params.w_obstacle = 200.0
params.safety_margin = 1.0
params.robot_radius = 0.5
params.resample_before_smooth = True
params.resample_after_smooth = False

# 4. Smooth
smoother = cs2d.PathSmoother2D(params)
result = smoother.smooth(astar.x, astar.y, esdf_map)

if result.success:
    print(f"Smoothed {len(result.x)} points in {result.solve_time_ms:.1f} ms")
    print(f"Min clearance: {min(esdf_map.get_distance(x, y) for x, y in zip(result.x, result.y)):.3f} m")
```

---

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `w_smooth` | 10.0 | 0 – ∞ | Smoothness weight. Higher = smoother path (less jerk). |
| `w_max_curvature` | 1000.0 | 0 – ∞ | Curvature constraint weight. Enforces turning radius ≥ `min_turning_radius`. |
| `min_turning_radius` | 0.2 m | > 0 | Minimum turning radius for the robot. |
| `w_reference` | 0.0 | 0 – ∞ | Reference tracking weight. Higher = path stays closer to A\* route. Set 0 to disable. |
| `w_length` | 1.0 | 0 – ∞ | Elastic-band length weight. Minimizes Σ‖Δp‖² for uniform spacing. |
| `w_obstacle` | 1.0 | 0 – ∞ | Obstacle avoidance weight (soft hinge outside safety boundary). |
| `w_penetration` | 1000.0 | 0 – ∞ | Inside-obstacle penalty. Prevents stalling in walls. Set 0 to disable. |
| `safety_margin` | 1.0 m | ≥ 0 | Desired clearance from robot **edge** to nearest obstacle. |
| `robot_radius` | 0.5 m | ≥ 0 | Robot inscribed radius. Effective clearance = `safety_margin + robot_radius`. |
| `resample_before_smooth` | `true` | bool | Resample input path to uniform spacing before optimization. |
| `resample_after_smooth` | `false` | bool | Resample output path to uniform spacing after optimization. |
| `resample_spacing` | 0.3 m | > 0 | Desired inter-point spacing for enabled resampling stages (not used by optimization). |
| `max_iterations` | 100 | ≥ 0 | Ceres solver max iterations per stage. |
| `max_time_seconds` | 0.5 s | ≥ 0 | Ceres solver wall-clock time limit. |
| `verbose` | `false` | bool | Print per-iteration Ceres output. |

### Tuning Guide

| Scenario | Suggestion |
|----------|------------|
| Path too jagged | Increase `w_smooth` (e.g. 200–500) |
| Path cuts corners | Decrease `min_turning_radius` or increase `w_max_curvature` |
| Path too far from A\* | Increase `w_reference` (e.g. 20–50) |
| Path too close to walls | Increase `w_obstacle` (e.g. 100–500) or `safety_margin` |
| Path stuck inside a wall | Increase `w_penetration` (e.g. 5000) |
| Path too short / points too close | Decrease `w_length` (e.g. 0.5–1.0) |
| Uneven point spacing on input | Enable `resample_before_smooth` (default on) |
| Need uniform output spacing | Enable `resample_after_smooth` |

---

## Performance

Typical solve times on a **1436 × 847** occupancy map (5 cm resolution):

| Path Length | Points | Solve Time | Notes |
|-------------|--------|------------|-------|
| ~20 m | ~65 | ~2 ms | Single stage, sparse solver |
| ~50 m | ~170 | ~5 ms | Multi-stage (3 stages) |
| ~100 m | ~340 | ~10 ms | Multi-stage |
| ~300 m | ~1000 | ~15 ms | Multi-stage, SuiteSparse |

- **A\***: ~5–15 ms for typical maps (100x faster than Python fallback)
- **ESDF computation**: ~50 ms on initial map load (one-time cost)
- **Sparse solver** (`SPARSE_NORMAL_CHOLESKY`): ~5–15x faster than dense for 1000+ points
- **Threading**: Single-threaded by default (`num_threads = 1`) — overhead exceeds speedup for sub-2k variable problems

---

## Key Design Decisions

1. **Per-node 2D parameter blocks** (`std::array<double,2>`) — preserves sparsity in the Hessian, enabling the sparse solver to be effective

2. **Bilinear (not BiCubic) ESDF interpolation** — BiCubic overshoots across the sharp discontinuity at obstacle boundaries, producing wildly wrong distances and gradients that point *into* walls. Bilinear is $C^0$ only but always bounded by the min/max of the 4 nearest cells.

3. **Dual obstacle cost** (hinge + penetration) — The hinge alone has a flat plateau inside obstacles where the gradient is constant but small. A point stuck deep inside may never escape. The penetration term adds a cost that grows with depth, making wall-interior states strictly suboptimal.

4. **Elastic-band length** (no rest length) — Pure linear residual with constant Jacobian → fast convergence. No conflict with fixed start/goal points (a target-spacing spring would require `spacing × (N-1) ≈ total_length`, which is rarely true).

5. **Multi-stage obstacle ramp** — Progressively increases $w_{\text{obstacle}}$ from `min(w_obstacle, 2.0)` up to the target value. This lets the path find a globally smooth shape first, then tighten clearance constraints.

6. **Pre-resampling** — A\* paths have uneven point density (dense near walls, sparse in open space). Resampling to uniform spacing before optimization gives the optimizer a better initial guess and prevents `w_length` from fighting the uneven distribution.

7. **PNG vertical flip** — Corrects the convention mismatch where PNG row 0 is the visual top, but in world coordinates $y = 0$ is the bottom. After the flip, grid row $r$ corresponds to PNG row $(H - 1 - r)$.

---

## Testing

### C++ Tests

```bash
cd build
./ceres_smoother_2d_tests
```

Covers:
- ESDFMap construction, sign convention, resolution scaling, bounds checking
- Cost function correctness (SmoothnessCost, CurvatureCost, ReferenceCost, ObstacleCostCeres)
- End-to-end smoothing (straight line, zigzag, obstacle avoidance)
- Edge cases (N=1, N=2, all-in-obstacle)

### Python Tests

```bash
cd my/ceres_smoother_2d
PYTHONPATH=build python -m pytest tests/test_python.py -v
```

Covers:
- Python API surface (ESDFMap, SmootherParams, PathSmoother2D)
- Resampling logic
- Integration with A\* search

### Web API Tests

```bash
# Start the web server first
./run_web.sh &

# Run tests
PYTHONPATH=build python -m pytest tests/test_web_api.py -v
```

Covers:
- `/api/costmap` endpoint (structure, dimensions, PNG validity)
- `/api/plan` endpoint (input validation, free paths, collision cases, parameter overrides)
- Start/goal in obstacle detection
- Clearance reporting

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **CMake can't find Ceres** | Install: `sudo apt install libceres-dev` or build from source. Ensure `find_package(Ceres REQUIRED)` succeeds. |
| **CMake can't find Eigen3** | Install: `sudo apt install libeigen3-dev` |
| **Python `import ceres_smoother_2d` fails** | Ensure `PYTHONPATH` includes the `build/` directory: `export PYTHONPATH=$PWD/build` |
| **NaN in output** | Check that the map is loaded correctly (white=free, black=obstacle). Ensure start/goal are in free space. |
| **Path goes through obstacles** | Increase `w_obstacle` or `w_penetration`. Verify `safety_margin + robot_radius` is reasonable. |
| **Path is too jagged** | Increase `w_smooth`. Decrease `w_obstacle` slightly if obstacle weight is overpowering smoothness. |
| **Solver is slow** | Ensure Ceres was built with SuiteSparse. Check that `SPARSE_NORMAL_CHOLESKY` is used (default). Reduce `max_iterations` if needed. |
| **Map orientation is wrong** | The library assumes PNG row 0 = world top ($y_{\max}$). If your map uses a different convention, adjust `origin_y` accordingly. |

---

## License

See [LICENSE](../../LICENSE) for project-level licensing.

### Third-Party

- **stb_image / stb_image_write** — Public domain (Sean Barrett)
- **Ceres Solver** — BSD 3-Clause (Google)
- **nanobind** — BSD 3-Clause (Wenzel Jakob)
