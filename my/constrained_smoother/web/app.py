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


def _coerce_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


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


def _build_capsule_center_offsets(limit_x, radius, tolerance):
    """Distribute circle centers so the union approximates a continuous capsule band."""
    if limit_x <= 1e-6:
        return [0.0]

    max_gap_depth = min(max(tolerance, 1e-3), max(radius * 0.5, 1e-3))
    min_val = radius * radius - max(radius - max_gap_depth, 0.0) ** 2
    max_spacing = 2.0 * math.sqrt(max(min_val, 1e-9))
    max_spacing = max(max_spacing, DEFAULT_RESOLUTION * 0.5)
    interval_count = max(1, int(math.ceil((2.0 * limit_x) / max_spacing)))
    return np.linspace(-limit_x, limit_x, interval_count + 1).tolist()


def _build_robot_footprint_model(
    footprint_mode,
    hinge_loss_threshold_m,
    point_robot_radius_m,
    robot_length_m,
    robot_width_m,
):
    """Build the unified checkpoint + radius geometry used by planning and smoothing."""
    mode = footprint_mode if footprint_mode in {"point", "capsule"} else "capsule"
    half_length = max(robot_length_m * 0.5, DEFAULT_RESOLUTION * 0.5)
    half_width = max(robot_width_m * 0.5, DEFAULT_RESOLUTION * 0.5)

    if mode == "point":
        check_radius = max(point_robot_radius_m, DEFAULT_RESOLUTION * 0.5)
        local_points = [(0.0, 0.0)]
    else:
        check_radius = half_width
        local_points = [(offset_x, 0.0) for offset_x in _build_capsule_center_offsets(
            half_length,
            check_radius,
            max(DEFAULT_RESOLUTION * 0.35, 0.02),
        )]

    planner_points = []
    smoother_points = []
    serialized_points = []
    for point_x, point_y in local_points:
        planner_points.extend((float(point_x), float(point_y)))
        smoother_points.extend((float(point_x), float(point_y), 1.0))
        serialized_points.append({
            "x": round(float(point_x), 4),
            "y": round(float(point_y), 4),
        })

    safe_distance = hinge_loss_threshold_m
    return {
        "mode": mode,
        "safe_distance": safe_distance,
        "check_radius": check_radius,
        "planner_points": planner_points,
        "smoother_points": smoother_points,
        "serialized_points": serialized_points,
        "robot_length_m": robot_length_m,
        "robot_width_m": robot_width_m,
    }


def _collides_oriented_rectangle(grid, center_x, center_y, yaw, length_m, width_m):
    """Check whether an oriented rectangle overlaps any lethal cells."""
    half_length = max(length_m * 0.5, DEFAULT_RESOLUTION * 0.5)
    half_width = max(width_m * 0.5, DEFAULT_RESOLUTION * 0.5)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    corners = []
    for local_x, local_y in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, half_width),
        (-half_length, -half_width),
    ):
        corners.append((
            center_x + cos_yaw * local_x - sin_yaw * local_y,
            center_y + sin_yaw * local_x + cos_yaw * local_y,
        ))

    min_x = min(point[0] for point in corners)
    max_x = max(point[0] for point in corners)
    min_y = min(point[1] for point in corners)
    max_y = max(point[1] for point in corners)

    min_mx = int(math.floor((min_x - DEFAULT_ORIGIN_X) / DEFAULT_RESOLUTION))
    max_mx = int(math.ceil((max_x - DEFAULT_ORIGIN_X) / DEFAULT_RESOLUTION)) - 1
    min_my = int(math.floor((min_y - DEFAULT_ORIGIN_Y) / DEFAULT_RESOLUTION))
    max_my = int(math.ceil((max_y - DEFAULT_ORIGIN_Y) / DEFAULT_RESOLUTION)) - 1
    if min_mx < 0 or min_my < 0 or max_mx >= DEFAULT_SIZE_X or max_my >= DEFAULT_SIZE_Y:
        return True

    for my in range(min_my, max_my + 1):
        for mx in range(min_mx, max_mx + 1):
            if int(grid[my, mx]) < 254:
                continue

            cell_x = DEFAULT_ORIGIN_X + (mx + 0.5) * DEFAULT_RESOLUTION
            cell_y = DEFAULT_ORIGIN_Y + (my + 0.5) * DEFAULT_RESOLUTION
            dx = cell_x - center_x
            dy = cell_y - center_y
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            if abs(local_x) <= half_length and abs(local_y) <= half_width:
                return True

    return False


def _validate_smoothed_path_rectangles(grid, xs, ys, thetas, robot_length_m, robot_width_m):
    """Final collision validation using the actual rectangular footprint."""
    colliding_indices = []
    for index, (world_x, world_y, theta) in enumerate(zip(xs, ys, thetas)):
        if _collides_oriented_rectangle(grid, world_x, world_y, theta, robot_length_m, robot_width_m):
            colliding_indices.append(index)

    return {
        "collision_free": not colliding_indices,
        "collision_count": len(colliding_indices),
        "colliding_indices": colliding_indices[:20],
    }


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

    return np.asarray(
        pcs.compute_esdf(costmap, pcs.Costmap2D.LETHAL_OBSTACLE),
        dtype=np.float64,
    ).reshape((DEFAULT_SIZE_Y, DEFAULT_SIZE_X))


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
        keep_start_orientation = _coerce_bool(req.get("keep_start_orientation"), True)
        keep_goal_orientation = _coerce_bool(req.get("keep_goal_orientation"), True)
        footprint_mode = str(req.get("footprint_mode", "capsule")).strip().lower()
        if footprint_mode not in {"point", "capsule"}:
            footprint_mode = "capsule"
        hinge_loss_threshold_m = max(0.05, float(req.get("hinge_loss_threshold_m", 0.5)))
        point_robot_radius_m = max(0.0, float(req.get("point_robot_radius_m", 1.0)))
        robot_length_m = max(DEFAULT_RESOLUTION, float(req.get("robot_length_m", 0.8)))
        robot_width_m = max(DEFAULT_RESOLUTION, float(req.get("robot_width_m", 0.5)))

        # Smoother tuning knobs from the frontend
        smooth_weight = float(req.get("smooth_weight", 20.0))
        costmap_weight = float(req.get("costmap_weight", 1.0))
        cusp_costmap_weight = max(0.0, float(req.get("cusp_costmap_weight", costmap_weight * 3.0)))
        cusp_zone_length = max(0.0, float(req.get("cusp_zone_length", 2.5)))
        distance_weight = float(req.get("distance_weight", 0.0))
        curvature_weight = float(req.get("curvature_weight", 30.0))
        curvature_rate_weight = float(req.get("curvature_rate_weight", 5.0))
        max_curvature = float(req.get("max_curvature", 2.5))
        max_time = max(0.01, float(req.get("max_time", 10.0)))
        reference_spacing_target_m = min(
            2.0,
            max(DEFAULT_RESOLUTION, float(req.get("reference_spacing_target_m", DEFAULT_REFERENCE_SPACING_TARGET_M))),
        )
        path_downsample = max(1, int(req.get("path_downsampling_factor", 1)))
        path_upsample = max(1, int(req.get("path_upsampling_factor", 1)))
        max_iterations = max(1, int(req.get("max_iterations", 50)))
        optimizer_type = str(req.get("optimizer_type", "constrained_smoother")).strip().lower()
        if optimizer_type not in {"constrained_smoother", "kinematic_simple"}:
            optimizer_type = "constrained_smoother"
        linear_solver_type = str(req.get("linear_solver_type", "SPARSE_NORMAL_CHOLESKY")).strip().upper()
        if linear_solver_type not in {"DENSE_QR", "SPARSE_NORMAL_CHOLESKY"}:
            linear_solver_type = "SPARSE_NORMAL_CHOLESKY"
        param_tol = max(0.0, float(req.get("param_tol", 1e-8)))
        fn_tol = max(0.0, float(req.get("fn_tol", 1e-6)))
        gradient_tol = max(0.0, float(req.get("gradient_tol", 1e-10)))
        optimizer_debug = _coerce_bool(req.get("optimizer_debug"), False)
        planner_penalty_weight = max(0.0, float(req.get("planner_penalty_weight", 1.0)))

        footprint_model = _build_robot_footprint_model(
            footprint_mode,
            hinge_loss_threshold_m,
            point_robot_radius_m,
            robot_length_m,
            robot_width_m,
        )

        with STATE_LOCK:
            costmap_grid = COSTMAP_GRID.copy()
            planner_costmap = _grid_to_pcs_costmap(costmap_grid)

        # 1) A* path planning
        planner = pcs.AStarPlanner()
        planner_params = pcs.AStarPlannerParams()
        planner_params.safe_distance = footprint_model["safe_distance"]
        planner_params.cost_penalty_weight = planner_penalty_weight
        planner_params.point_radius = 0.0
        planner_params.collision_check_radius = footprint_model["check_radius"]
        planner_params.collision_check_points = footprint_model["planner_points"]
        planner_params.use_rectangular_footprint = False
        planner_params.rectangular_length = 0.0
        planner_params.rectangular_width = 0.0
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
        smoother_params.cusp_costmap_weight_sqrt = math.sqrt(cusp_costmap_weight)
        smoother_params.cusp_zone_length = cusp_zone_length
        smoother_params.obstacle_safe_distance = footprint_model["safe_distance"]
        smoother_params.cost_check_radius = footprint_model["check_radius"]
        smoother_params.distance_weight_sqrt = math.sqrt(distance_weight)
        smoother_params.curvature_weight_sqrt = math.sqrt(curvature_weight)
        smoother_params.curvature_rate_weight_sqrt = math.sqrt(curvature_rate_weight)
        smoother_params.max_curvature = max_curvature
        smoother_params.max_time = max_time
        smoother_params.keep_start_orientation = keep_start_orientation
        smoother_params.keep_goal_orientation = keep_goal_orientation
        smoother_params.cost_check_points = footprint_model["smoother_points"]
        smoother_params.path_downsampling_factor = path_downsample
        smoother_params.path_upsampling_factor = path_upsample

        opt_params = pcs.OptimizerParams()
        opt_params.debug = optimizer_debug
        opt_params.linear_solver_type = linear_solver_type
        opt_params.max_iterations = max_iterations
        opt_params.param_tol = param_tol
        opt_params.fn_tol = fn_tol
        opt_params.gradient_tol = gradient_tol

        optimizer_label = (
            "Kinematic Simple"
            if optimizer_type == "kinematic_simple"
            else "Constrained Smoother"
        )
        if optimizer_type == "kinematic_simple":
            smoother = pcs.SimpleKinematicSmoother()
        else:
            smoother = pcs.Smoother()
        smoother.initialize(opt_params)

        t1 = time.time()
        smooth_message = ""
        optimized_knot_count = 0
        try:
            smoothed = smoother.smooth_with_planner_esdf(
                eigen_path,
                s_dir,
                e_dir,
                planner_costmap,
                smoother_params,
                planner,
            )
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
        final_rectangle_validation = _validate_smoothed_path_rectangles(
            costmap_grid,
            opt_x,
            opt_y,
            opt_theta,
            robot_length_m,
            robot_width_m,
        )

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
            "optimizer_type": optimizer_type,
            "optimizer_label": optimizer_label,
            "curvature_rate_weight": round(curvature_rate_weight, 3),
            "start_yaw_deg": round(start_yaw_deg, 2),
            "goal_yaw_deg": round(goal_yaw_deg, 2),
            "keep_start_orientation": keep_start_orientation,
            "keep_goal_orientation": keep_goal_orientation,
            "hinge_loss_threshold_m": round(hinge_loss_threshold_m, 3),
            "point_robot_radius_m": round(point_robot_radius_m, 3),
            "effective_safe_distance_m": round(footprint_model["safe_distance"], 3),
            "footprint_mode": footprint_model["mode"],
            "robot_length_m": round(robot_length_m, 3),
            "robot_width_m": round(robot_width_m, 3),
            "robot_check_points": len(footprint_model["serialized_points"]),
            "collision_check_radius_m": round(footprint_model["check_radius"], 3),
            "collision_check_points_local": footprint_model["serialized_points"],
            "final_rectangle_validation": final_rectangle_validation,
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
