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
from astar import downsample_path  # noqa: E402


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


app = Flask(__name__)

# ---------------------------------------------------------------------------
# Default costmap: 200×200 cells, 0.1 m resolution → 20 m × 20 m world area
# ---------------------------------------------------------------------------
DEFAULT_SIZE_X = 200
DEFAULT_SIZE_Y = 200
DEFAULT_RESOLUTION = 0.1
DEFAULT_ORIGIN_X = 0.0
DEFAULT_ORIGIN_Y = 0.0
DEFAULT_REFERENCE_SPACING_TARGET_M = DEFAULT_RESOLUTION * 3
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
ESDF_GRID = None
HAS_COMPUTE_ESDF = hasattr(pcs, "compute_esdf")


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
            "A draggable 20m x 20m obstacle map with rectangular lethal obstacles and "
            f"a {INFLATION_RADIUS_CELLS}-cell inflated safety buffer for visualization. "
            "The C++ A* planner and constrained smoother both optimize ESDF-derived obstacle penalties."
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


def _build_robot_cost_check_points(footprint_mode, robot_length_m, robot_width_m):
    """Build local-frame footprint samples for the smoother obstacle term."""
    if footprint_mode != "rectangle":
        return []

    half_length = max(robot_length_m * 0.5, DEFAULT_RESOLUTION * 0.5)
    half_width = max(robot_width_m * 0.5, DEFAULT_RESOLUTION * 0.5)
    return [
        0.0, 0.0, 0.6,
        half_length, half_width, 1.0,
        half_length, -half_width, 1.0,
        -half_length, half_width, 1.0,
        -half_length, -half_width, 1.0,
        half_length, 0.0, 0.8,
        -half_length, 0.0, 0.8,
        0.0, half_width, 0.8,
        0.0, -half_width, 0.8,
    ]


COSTMAP_GRID = None
COSTMAP_METADATA = None
ESDF_GRID = None


def _grid_to_pcs_costmap(grid):
    """Convert numpy grid to pcs.Costmap2D for the smoother."""
    size_y, size_x = grid.shape
    costmap = pcs.Costmap2D(size_x, size_y, DEFAULT_RESOLUTION, DEFAULT_ORIGIN_X, DEFAULT_ORIGIN_Y)
    for my in range(size_y):
        for mx in range(size_x):
            costmap.setCost(mx, my, int(grid[my, mx]))
    return costmap


def _compute_esdf_grid(costmap):
    """Compute an ESDF grid in meters from the obstacle map."""
    if not HAS_COMPUTE_ESDF:
        return None

    outside_esdf = np.asarray(
        pcs.compute_esdf(costmap, pcs.Costmap2D.LETHAL_OBSTACLE),
        dtype=np.float64,
    ).reshape((DEFAULT_SIZE_Y, DEFAULT_SIZE_X))

    # Build an inverted occupancy map so occupied cells measure distance to the nearest free cell.
    inside_costmap = pcs.Costmap2D(
        DEFAULT_SIZE_X,
        DEFAULT_SIZE_Y,
        DEFAULT_RESOLUTION,
        DEFAULT_ORIGIN_X,
        DEFAULT_ORIGIN_Y,
    )
    for my in range(DEFAULT_SIZE_Y):
        for mx in range(DEFAULT_SIZE_X):
            is_obstacle = costmap.getCost(mx, my) >= pcs.Costmap2D.LETHAL_OBSTACLE
            inside_costmap.setCost(
                mx,
                my,
                pcs.Costmap2D.FREE_SPACE if is_obstacle else pcs.Costmap2D.LETHAL_OBSTACLE,
            )

    inside_esdf = np.asarray(
        pcs.compute_esdf(inside_costmap, pcs.Costmap2D.LETHAL_OBSTACLE),
        dtype=np.float64,
    ).reshape((DEFAULT_SIZE_Y, DEFAULT_SIZE_X))

    signed_esdf = outside_esdf.copy()
    obstacle_mask = np.asarray(COSTMAP_GRID >= pcs.Costmap2D.LETHAL_OBSTACLE)
    signed_esdf[obstacle_mask] = -inside_esdf[obstacle_mask]
    return signed_esdf


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
    global CURRENT_OBSTACLE_RECTS, COSTMAP_GRID, COSTMAP_METADATA, PCS_COSTMAP, ESDF_GRID

    CURRENT_OBSTACLE_RECTS = [tuple(rect) for rect in obstacle_rects]
    COSTMAP_GRID = _build_costmap(CURRENT_OBSTACLE_RECTS)
    COSTMAP_METADATA = _summarize_costmap(COSTMAP_GRID, CURRENT_OBSTACLE_RECTS)
    PCS_COSTMAP = _grid_to_pcs_costmap(COSTMAP_GRID)
    ESDF_GRID = _compute_esdf_grid(PCS_COSTMAP)


def _serialize_costmap_state():
    """Return the current costmap payload used by the frontend."""
    return {
        "size_x": DEFAULT_SIZE_X,
        "size_y": DEFAULT_SIZE_Y,
        "resolution": DEFAULT_RESOLUTION,
        "origin_x": DEFAULT_ORIGIN_X,
        "origin_y": DEFAULT_ORIGIN_Y,
        "data": COSTMAP_GRID.flatten().tolist(),
        "esdf": ESDF_GRID.flatten().tolist() if ESDF_GRID is not None else None,
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
        start_yaw_deg = float(req.get("start_yaw_deg", 45.0))
        goal_yaw_deg = float(req.get("goal_yaw_deg", 45.0))
        keep_start_orientation = bool(req.get("keep_start_orientation", True))
        keep_goal_orientation = bool(req.get("keep_goal_orientation", True))
        footprint_mode = str(req.get("footprint_mode", "point")).strip().lower()
        if footprint_mode not in {"point", "rectangle"}:
            footprint_mode = "point"
        hinge_loss_threshold_m = max(0.05, float(req.get("hinge_loss_threshold_m", 0.5)))
        point_robot_radius_m = max(0.0, float(req.get("point_robot_radius_m", 1.0)))
        robot_length_m = max(DEFAULT_RESOLUTION, float(req.get("robot_length_m", 0.8)))
        robot_width_m = max(DEFAULT_RESOLUTION, float(req.get("robot_width_m", 0.5)))

        # Smoother tuning knobs from the frontend
        smooth_weight = float(req.get("smooth_weight", 20.0))
        costmap_weight = float(req.get("costmap_weight", 1.0))
        distance_weight = float(req.get("distance_weight", 0.0))
        curvature_weight = float(req.get("curvature_weight", 30.0))
        curvature_rate_weight = float(req.get("curvature_rate_weight", 5.0))
        max_curvature = float(req.get("max_curvature", 2.5))
        reference_spacing_target_m = min(
            2.0,
            max(DEFAULT_RESOLUTION, float(req.get("reference_spacing_target_m", DEFAULT_REFERENCE_SPACING_TARGET_M))),
        )
        path_downsample = max(1, int(req.get("path_downsampling_factor", 1)))
        path_upsample = max(1, int(req.get("path_upsampling_factor", 1)))
        max_iterations = max(1, int(req.get("max_iterations", 50)))
        planner_penalty_weight = max(0.0, float(req.get("planner_penalty_weight", 1.0)))

        with STATE_LOCK:
            planner_costmap = _grid_to_pcs_costmap(COSTMAP_GRID.copy())

        # 1) A* path planning
        planner = pcs.AStarPlanner()
        planner_params = pcs.AStarPlannerParams()
        planner_params.safe_distance = hinge_loss_threshold_m + (
            point_robot_radius_m if footprint_mode == "point" else 0.0
        )
        planner_params.cost_penalty_weight = planner_penalty_weight
        planner_params.point_radius = point_robot_radius_m if footprint_mode == "point" else 0.0
        planner_params.use_rectangular_footprint = footprint_mode == "rectangle"
        planner_params.rectangular_length = robot_length_m if footprint_mode == "rectangle" else 0.0
        planner_params.rectangular_width = robot_width_m if footprint_mode == "rectangle" else 0.0
        t0 = time.time()
        raw_path = planner.plan(
            planner_costmap,
            start_x,
            start_y,
            goal_x,
            goal_y,
            planner_params,
        )
        astar_time = (time.time() - t0) * 1000.0

        if not raw_path:
            return jsonify({"success": False, "message": "A* could not find a path."})

        raw_path = [(float(point[0]), float(point[1])) for point in raw_path]

        # Downsample dense grid path
        sparse_path = downsample_path(raw_path, reference_spacing_target_m)

        # Build Eigen-compatible path: (x, y, direction_sign=1.0)
        eigen_path = [np.array([p[0], p[1], 1.0]) for p in sparse_path]

        start_yaw_rad = math.radians(start_yaw_deg)
        goal_yaw_rad = math.radians(goal_yaw_deg)
        s_dir = np.array([math.cos(start_yaw_rad), math.sin(start_yaw_rad)])
        e_dir = np.array([math.cos(goal_yaw_rad), math.sin(goal_yaw_rad)])

        # 2) Constrained smoother
        smoother_params = pcs.SmootherParams()
        smoother_params.smooth_weight_sqrt = math.sqrt(smooth_weight)
        smoother_params.costmap_weight_sqrt = math.sqrt(costmap_weight)
        smoother_params.cusp_costmap_weight_sqrt = smoother_params.costmap_weight_sqrt * math.sqrt(3.0)
        smoother_params.cusp_zone_length = 2.5
        smoother_params.obstacle_safe_distance = planner_params.safe_distance
        smoother_params.distance_weight_sqrt = math.sqrt(distance_weight)
        smoother_params.curvature_weight_sqrt = math.sqrt(curvature_weight)
        smoother_params.curvature_rate_weight_sqrt = math.sqrt(curvature_rate_weight)
        smoother_params.max_curvature = max_curvature
        smoother_params.max_time = 10.0
        smoother_params.keep_start_orientation = keep_start_orientation
        smoother_params.keep_goal_orientation = keep_goal_orientation
        smoother_params.cost_check_points = _build_robot_cost_check_points(
            footprint_mode,
            robot_length_m,
            robot_width_m,
        )
        smoother_params.path_downsampling_factor = path_downsample
        smoother_params.path_upsampling_factor = path_upsample

        opt_params = pcs.OptimizerParams()
        opt_params.max_iterations = max_iterations

        smoother = pcs.Smoother()
        smoother.initialize(opt_params)

        t1 = time.time()
        smooth_message = ""
        optimized_knot_count = 0
        try:
            smoothed = smoother.smooth(eigen_path, s_dir, e_dir, planner_costmap, smoother_params)
            smooth_time = (time.time() - t1) * 1000.0
            smooth_success = True
        except Exception as e:
            smooth_time = (time.time() - t1) * 1000.0
            smoothed = eigen_path  # fall back to unsmoothed
            smooth_success = False
            smooth_message = str(e)
        optimized_knot_count = int(smoother.get_last_optimized_knot_count())

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
            "num_opt_knots": optimized_knot_count,
            "num_opt_pts": len(smoothed),
            "num_returned_pts": len(smoothed),
            "raw_path_length_m": round(raw_length, 3),
            "ref_path_length_m": round(ref_length, 3),
            "opt_path_length_m": round(opt_length, 3),
            "opt_vs_ref_delta_m": round(opt_length - ref_length, 3),
            "reference_spacing_target_m": round(reference_spacing_target_m, 3),
            "planner_penalty_weight": round(planner_penalty_weight, 3),
            "curvature_rate_weight": round(curvature_rate_weight, 3),
            "start_yaw_deg": round(start_yaw_deg, 2),
            "goal_yaw_deg": round(goal_yaw_deg, 2),
            "keep_start_orientation": keep_start_orientation,
            "keep_goal_orientation": keep_goal_orientation,
            "hinge_loss_threshold_m": round(hinge_loss_threshold_m, 3),
            "point_robot_radius_m": round(point_robot_radius_m, 3),
            "effective_safe_distance_m": round(planner_params.safe_distance, 3),
            "footprint_mode": footprint_mode,
            "robot_length_m": round(robot_length_m, 3),
            "robot_width_m": round(robot_width_m, 3),
            "robot_check_points": len(smoother_params.cost_check_points) // 3,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("CS_WEBAPP_HOST", "127.0.0.1"),
        port=int(os.environ.get("CS_WEBAPP_PORT", "5002")),
        debug=_env_flag("CS_WEBAPP_DEBUG", True),
        use_reloader=_env_flag("CS_WEBAPP_RELOADER", False),
    )
