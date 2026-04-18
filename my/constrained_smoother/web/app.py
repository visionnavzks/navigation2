"""Flask web application: A* planning + constrained smoother visualization.

Usage
-----
    cd my/constrained_smoother
    # Build the pybind11 module first (see CMakeLists.txt, BUILD_PYTHON=ON)
    python3 web/app.py
"""

import os
import sys
import math
import time
import traceback

import numpy as np
from flask import Flask, request, jsonify, render_template

# Allow importing the built pybind11 module and the astar module
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)

# Add parent directory (constrained_smoother/) to path so py_constrained_smoother can be found
sys.path.insert(0, _parent_dir)
# Also check in build/ directory
_build_dir = os.path.join(_parent_dir, "build")
if os.path.isdir(_build_dir):
    sys.path.insert(0, _build_dir)

import py_constrained_smoother as pcs  # noqa: E402
from astar import AStarPlanner, downsample_path  # noqa: E402

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Default costmap: 200×200 cells, 0.1 m resolution → 20 m × 20 m world area
# ---------------------------------------------------------------------------
DEFAULT_SIZE_X = 200
DEFAULT_SIZE_Y = 200
DEFAULT_RESOLUTION = 0.1
DEFAULT_ORIGIN_X = 0.0
DEFAULT_ORIGIN_Y = 0.0


def _build_default_costmap():
    """Create a costmap with some sample obstacles."""
    grid = np.zeros((DEFAULT_SIZE_Y, DEFAULT_SIZE_X), dtype=np.uint8)

    # Add rectangular obstacles
    obstacles = [
        # (x_start, y_start, x_end, y_end)  in cell coords
        (60, 40, 80, 100),
        (120, 60, 140, 160),
        (30, 130, 90, 150),
        (150, 20, 170, 80),
    ]
    for x0, y0, x1, y1 in obstacles:
        grid[y0:y1, x0:x1] = 254  # LETHAL

    # Inflate obstacles (simple dilation)
    inflate_radius = 5  # cells
    inflated = grid.copy()
    lethal_cells = np.argwhere(grid == 254)
    for cy, cx in lethal_cells:
        for dy in range(-inflate_radius, inflate_radius + 1):
            for dx in range(-inflate_radius, inflate_radius + 1):
                ny, nx_cell = cy + dy, cx + dx
                if 0 <= ny < DEFAULT_SIZE_Y and 0 <= nx_cell < DEFAULT_SIZE_X:
                    dist = math.hypot(dx, dy)
                    if dist <= inflate_radius and inflated[ny, nx_cell] < 254:
                        cost = int(253 * max(0, 1 - dist / inflate_radius))
                        inflated[ny, nx_cell] = max(inflated[ny, nx_cell], cost)
    return inflated


COSTMAP_GRID = _build_default_costmap()


def _grid_to_pcs_costmap(grid):
    """Convert numpy grid to pcs.Costmap2D for the smoother."""
    size_y, size_x = grid.shape
    costmap = pcs.Costmap2D(size_x, size_y, DEFAULT_RESOLUTION, DEFAULT_ORIGIN_X, DEFAULT_ORIGIN_Y)
    for my in range(size_y):
        for mx in range(size_x):
            costmap.setCost(mx, my, int(grid[my, mx]))
    return costmap


PCS_COSTMAP = _grid_to_pcs_costmap(COSTMAP_GRID)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/costmap", methods=["GET"])
def get_costmap():
    """Return the costmap grid as a flat list for the frontend to render."""
    return jsonify({
        "size_x": DEFAULT_SIZE_X,
        "size_y": DEFAULT_SIZE_Y,
        "resolution": DEFAULT_RESOLUTION,
        "origin_x": DEFAULT_ORIGIN_X,
        "origin_y": DEFAULT_ORIGIN_Y,
        "data": COSTMAP_GRID.flatten().tolist(),
    })


@app.route("/api/plan", methods=["POST"])
def plan_and_smooth():
    """Run A* to find a reference path, then smooth it with the constrained smoother."""
    try:
        req = request.get_json(silent=True) or {}
        start_x = float(req.get("start_x", 1.0))
        start_y = float(req.get("start_y", 1.0))
        goal_x = float(req.get("goal_x", 18.0))
        goal_y = float(req.get("goal_y", 18.0))

        # Smoother tuning knobs from the frontend
        smooth_weight = float(req.get("smooth_weight", 2000000.0))
        costmap_weight = float(req.get("costmap_weight", 0.015))
        distance_weight = float(req.get("distance_weight", 0.0))
        curvature_weight = float(req.get("curvature_weight", 30.0))
        max_curvature = float(req.get("max_curvature", 2.5))
        path_downsample = max(1, int(req.get("path_downsampling_factor", 1)))
        path_upsample = max(1, int(req.get("path_upsampling_factor", 1)))
        max_iterations = max(1, int(req.get("max_iterations", 50)))

        # 1) A* path planning
        planner = AStarPlanner(
            COSTMAP_GRID, DEFAULT_SIZE_X, DEFAULT_SIZE_Y,
            DEFAULT_RESOLUTION, DEFAULT_ORIGIN_X, DEFAULT_ORIGIN_Y,
        )
        t0 = time.time()
        raw_path = planner.plan(start_x, start_y, goal_x, goal_y)
        astar_time = (time.time() - t0) * 1000.0

        if raw_path is None:
            return jsonify({"success": False, "message": "A* could not find a path."})

        # Downsample dense grid path
        ds_target = DEFAULT_RESOLUTION * 3
        sparse_path = downsample_path(raw_path, ds_target)

        # Build Eigen-compatible path: (x, y, direction_sign=1.0)
        eigen_path = [np.array([p[0], p[1], 1.0]) for p in sparse_path]

        # Direction vectors for start/end
        if len(eigen_path) >= 2:
            s_dir = np.array([eigen_path[1][0] - eigen_path[0][0],
                              eigen_path[1][1] - eigen_path[0][1]])
            norm = np.linalg.norm(s_dir)
            if norm > 1e-9:
                s_dir /= norm
            else:
                s_dir = np.array([1.0, 0.0])

            e_dir = np.array([eigen_path[-1][0] - eigen_path[-2][0],
                              eigen_path[-1][1] - eigen_path[-2][1]])
            norm = np.linalg.norm(e_dir)
            if norm > 1e-9:
                e_dir /= norm
            else:
                e_dir = np.array([1.0, 0.0])
        else:
            s_dir = np.array([1.0, 0.0])
            e_dir = np.array([1.0, 0.0])

        # 2) Constrained smoother
        smoother_params = pcs.SmootherParams()
        smoother_params.smooth_weight_sqrt = math.sqrt(smooth_weight)
        smoother_params.costmap_weight_sqrt = math.sqrt(costmap_weight)
        smoother_params.cusp_costmap_weight_sqrt = smoother_params.costmap_weight_sqrt * math.sqrt(3.0)
        smoother_params.cusp_zone_length = 2.5
        smoother_params.distance_weight_sqrt = math.sqrt(distance_weight)
        smoother_params.curvature_weight_sqrt = math.sqrt(curvature_weight)
        smoother_params.max_curvature = max_curvature
        smoother_params.max_time = 10.0
        smoother_params.path_downsampling_factor = path_downsample
        smoother_params.path_upsampling_factor = path_upsample

        opt_params = pcs.OptimizerParams()
        opt_params.max_iterations = max_iterations

        smoother = pcs.Smoother()
        smoother.initialize(opt_params)

        t1 = time.time()
        try:
            smoothed = smoother.smooth(eigen_path, s_dir, e_dir, PCS_COSTMAP, smoother_params)
            smooth_time = (time.time() - t1) * 1000.0
            smooth_success = True
        except Exception as e:
            smooth_time = (time.time() - t1) * 1000.0
            smoothed = eigen_path  # fall back to unsmoothed
            smooth_success = False

        # Format response
        astar_x = [p[0] for p in raw_path]
        astar_y = [p[1] for p in raw_path]
        ref_x = [p[0] for p in sparse_path]
        ref_y = [p[1] for p in sparse_path]
        opt_x = [p[0] for p in smoothed]
        opt_y = [p[1] for p in smoothed]
        opt_theta = [p[2] for p in smoothed]

        return jsonify({
            "success": True,
            "smooth_success": smooth_success,
            "astar_time_ms": round(astar_time, 2),
            "smooth_time_ms": round(smooth_time, 2),
            "astar_x": astar_x,
            "astar_y": astar_y,
            "ref_x": ref_x,
            "ref_y": ref_y,
            "opt_x": opt_x,
            "opt_y": opt_y,
            "opt_theta": opt_theta,
            "num_astar_pts": len(raw_path),
            "num_ref_pts": len(sparse_path),
            "num_opt_pts": len(smoothed),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
