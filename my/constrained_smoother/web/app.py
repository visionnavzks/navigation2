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
from threading import Lock

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
INFLATION_RADIUS_CELLS = 5
DEFAULT_OBSTACLE_RECTS = [
    # (x_start, y_start, x_end, y_end) in cell coordinates
    (60, 40, 80, 100),
    (120, 60, 140, 160),
    (30, 130, 90, 150),
    (150, 20, 170, 80),
]
CURRENT_OBSTACLE_RECTS = [tuple(rect) for rect in DEFAULT_OBSTACLE_RECTS]
STATE_LOCK = Lock()


def _build_costmap(obstacle_rects):
    """Create a costmap with the provided obstacle rectangles."""
    grid = np.zeros((DEFAULT_SIZE_Y, DEFAULT_SIZE_X), dtype=np.uint8)

    # Add rectangular obstacles
    for x0, y0, x1, y1 in obstacle_rects:
        grid[y0:y1, x0:x1] = 254  # LETHAL

    # Inflate obstacles (simple dilation)
    inflated = grid.copy()
    lethal_cells = np.argwhere(grid == 254)
    for cy, cx in lethal_cells:
        for dy in range(-INFLATION_RADIUS_CELLS, INFLATION_RADIUS_CELLS + 1):
            for dx in range(-INFLATION_RADIUS_CELLS, INFLATION_RADIUS_CELLS + 1):
                ny, nx_cell = cy + dy, cx + dx
                if 0 <= ny < DEFAULT_SIZE_Y and 0 <= nx_cell < DEFAULT_SIZE_X:
                    dist = math.hypot(dx, dy)
                    if dist <= INFLATION_RADIUS_CELLS and inflated[ny, nx_cell] < 254:
                        cost = int(253 * max(0, 1 - dist / INFLATION_RADIUS_CELLS))
                        inflated[ny, nx_cell] = max(inflated[ny, nx_cell], cost)
    return inflated


def _summarize_costmap(grid, obstacle_rects):
    """Return metadata used by the frontend to explain the map semantics."""
    total_cells = int(grid.size)
    lethal_cells = int(np.count_nonzero(grid >= 254))
    inflated_cells = int(np.count_nonzero((grid > 0) & (grid < 254)))
    free_cells = total_cells - lethal_cells - inflated_cells
    return {
        "name": "Synthetic obstacle field",
        "description": (
            "A draggable 20m x 20m costmap with rectangular lethal obstacles and "
            f"a {INFLATION_RADIUS_CELLS}-cell inflated safety buffer rendered around them."
        ),
        "world_width_m": DEFAULT_SIZE_X * DEFAULT_RESOLUTION,
        "world_height_m": DEFAULT_SIZE_Y * DEFAULT_RESOLUTION,
        "origin": {
            "x": DEFAULT_ORIGIN_X,
            "y": DEFAULT_ORIGIN_Y,
        },
        "obstacle_count": len(obstacle_rects),
        "obstacle_rects_cells": [
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            for x0, y0, x1, y1 in obstacle_rects
        ],
        "default_obstacle_rects_cells": [
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            for x0, y0, x1, y1 in DEFAULT_OBSTACLE_RECTS
        ],
        "inflation_radius_cells": INFLATION_RADIUS_CELLS,
        "inflation_radius_m": INFLATION_RADIUS_CELLS * DEFAULT_RESOLUTION,
        "free_cells": free_cells,
        "inflated_cells": inflated_cells,
        "lethal_cells": lethal_cells,
        "cell_count": total_cells,
        "cost_value_meanings": {
            "free": "0",
            "inflated": "1-253",
            "lethal": "254",
        },
    }


def _path_length(points):
    """Compute Euclidean polyline length in meters."""
    if len(points) < 2:
        return 0.0

    total = 0.0
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        curr = points[idx]
        total += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
    return total


COSTMAP_GRID = None
COSTMAP_METADATA = None


def _grid_to_pcs_costmap(grid):
    """Convert numpy grid to pcs.Costmap2D for the smoother."""
    size_y, size_x = grid.shape
    costmap = pcs.Costmap2D(size_x, size_y, DEFAULT_RESOLUTION, DEFAULT_ORIGIN_X, DEFAULT_ORIGIN_Y)
    for my in range(size_y):
        for mx in range(size_x):
            costmap.setCost(mx, my, int(grid[my, mx]))
    return costmap


PCS_COSTMAP = None


def _normalize_obstacle_rects(rect_payloads):
    """Validate and clamp incoming obstacle rectangles in cell coordinates."""
    normalized = []
    for payload in rect_payloads:
        x0 = int(payload["x0"])
        y0 = int(payload["y0"])
        x1 = int(payload["x1"])
        y1 = int(payload["y1"])

        if x1 <= x0 or y1 <= y0:
            raise ValueError("Obstacle rectangles must have positive width and height.")

        x0 = max(0, min(DEFAULT_SIZE_X - 1, x0))
        y0 = max(0, min(DEFAULT_SIZE_Y - 1, y0))
        x1 = max(x0 + 1, min(DEFAULT_SIZE_X, x1))
        y1 = max(y0 + 1, min(DEFAULT_SIZE_Y, y1))
        normalized.append((x0, y0, x1, y1))
    return normalized


def _rebuild_costmap_state(obstacle_rects):
    """Regenerate all costmap-derived globals from the obstacle list."""
    global CURRENT_OBSTACLE_RECTS, COSTMAP_GRID, COSTMAP_METADATA, PCS_COSTMAP

    CURRENT_OBSTACLE_RECTS = [tuple(rect) for rect in obstacle_rects]
    COSTMAP_GRID = _build_costmap(CURRENT_OBSTACLE_RECTS)
    COSTMAP_METADATA = _summarize_costmap(COSTMAP_GRID, CURRENT_OBSTACLE_RECTS)
    PCS_COSTMAP = _grid_to_pcs_costmap(COSTMAP_GRID)


def _serialize_costmap_state():
    """Return the current costmap payload used by the frontend."""
    return {
        "size_x": DEFAULT_SIZE_X,
        "size_y": DEFAULT_SIZE_Y,
        "resolution": DEFAULT_RESOLUTION,
        "origin_x": DEFAULT_ORIGIN_X,
        "origin_y": DEFAULT_ORIGIN_Y,
        "data": COSTMAP_GRID.flatten().tolist(),
        "metadata": COSTMAP_METADATA,
    }


with STATE_LOCK:
    _rebuild_costmap_state(DEFAULT_OBSTACLE_RECTS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/costmap", methods=["GET"])
def get_costmap():
    """Return the costmap grid as a flat list for the frontend to render."""
    with STATE_LOCK:
        return jsonify(_serialize_costmap_state())


@app.route("/api/obstacles", methods=["POST"])
def update_obstacles():
    """Update the draggable obstacle rectangles and rebuild the costmap."""
    try:
        req = request.get_json(silent=True) or {}
        rect_payloads = req.get("obstacle_rects_cells")
        if not isinstance(rect_payloads, list) or not rect_payloads:
            return jsonify({"success": False, "message": "No obstacle rectangles were provided."}), 400

        normalized_rects = _normalize_obstacle_rects(rect_payloads)
        with STATE_LOCK:
            _rebuild_costmap_state(normalized_rects)
            payload = _serialize_costmap_state()

        payload["success"] = True
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 400


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

        with STATE_LOCK:
            planner_grid = COSTMAP_GRID.copy()
            planner_costmap = _grid_to_pcs_costmap(planner_grid)

        # 1) A* path planning
        planner = AStarPlanner(
            planner_grid, DEFAULT_SIZE_X, DEFAULT_SIZE_Y,
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
        smooth_message = ""
        try:
            smoothed = smoother.smooth(eigen_path, s_dir, e_dir, planner_costmap, smoother_params)
            smooth_time = (time.time() - t1) * 1000.0
            smooth_success = True
        except Exception as e:
            smooth_time = (time.time() - t1) * 1000.0
            smoothed = eigen_path  # fall back to unsmoothed
            smooth_success = False
            smooth_message = str(e)

        # Format response
        astar_x = [p[0] for p in raw_path]
        astar_y = [p[1] for p in raw_path]
        ref_x = [p[0] for p in sparse_path]
        ref_y = [p[1] for p in sparse_path]
        opt_x = [p[0] for p in smoothed]
        opt_y = [p[1] for p in smoothed]
        opt_theta = [p[2] for p in smoothed]
        raw_length = _path_length(raw_path)
        ref_length = _path_length(sparse_path)
        opt_length = _path_length(smoothed)

        return jsonify({
            "success": True,
            "smooth_success": smooth_success,
            "astar_time_ms": round(astar_time, 2),
            "smooth_time_ms": round(smooth_time, 2),
            "smooth_message": smooth_message,
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
            "raw_path_length_m": round(raw_length, 3),
            "ref_path_length_m": round(ref_length, 3),
            "opt_path_length_m": round(opt_length, 3),
            "opt_vs_ref_delta_m": round(opt_length - ref_length, 3),
            "reference_spacing_target_m": round(ds_target, 3),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
