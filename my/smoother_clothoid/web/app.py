"""Flask web application: A* planning + clothoid smoother visualization.

Usage
-----
    cd my/smoother_clothoid
    python3 web/app.py
"""

import os
import sys
import math
import time
import traceback
from threading import Lock

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, Response

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, _this_dir)
_build_dir = os.path.join(_parent_dir, "build")
if os.path.isdir(_build_dir):
    sys.path.insert(0, _build_dir)

import nb_smoother_clothoid as sc
from astar import AStarPlanner, downsample_path

MAP_PATH = os.path.join(_parent_dir, "..", "maps", "occupancy_map.png")

ERROR_INVALID_REQUEST = "SC_INVALID_REQUEST"
ERROR_ASTAR_NO_PATH = "SC_ASTAR_NO_PATH"
ERROR_INTERNAL = "SC_INTERNAL_ERROR"


class ApiError(Exception):
    def __init__(self, code, message, status_code=400, source="server", details=None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.source = str(source)
        self.details = details or {}

    def to_payload(self):
        error_payload = {"code": self.code, "message": self.message, "source": self.source}
        if self.details:
            error_payload["details"] = self.details
        return {"success": False, "message": self.message, "error": error_payload}


def _error_response(exc, *, default_status=400, default_source="server"):
    if isinstance(exc, ApiError):
        return jsonify(exc.to_payload()), exc.status_code
    message = str(exc) or "Unknown error"
    api_error = ApiError(
        code=ERROR_INTERNAL, message=message, status_code=default_status, source=default_source,
        details={"exception_type": type(exc).__name__},
    )
    return jsonify(api_error.to_payload()), api_error.status_code


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


def _load_occupancy_map():
    if not os.path.exists(MAP_PATH):
        return None, None
    img = Image.open(MAP_PATH).convert("L")
    grid = np.array(img, dtype=np.uint8)
    size_y, size_x = grid.shape
    resolution = 0.05
    origin_x = 0.0
    origin_y = 0.0
    costmap_grid = _occupancy_to_costmap(grid)
    return costmap_grid, {
        "size_x": size_x, "size_y": size_y,
        "resolution": resolution,
        "origin_x": origin_x, "origin_y": origin_y,
        "image": grid,
    }


def _occupancy_to_costmap(occ_grid):
    costmap = np.zeros_like(occ_grid)
    unknown = occ_grid == 205
    # Lethal: any value close to fully-occupied (255 or 254)
    lethal = (occ_grid >= 250) & ~unknown
    # Free: zero (or near zero) is free
    free = occ_grid <= 5
    # Inflated: anything in between
    inflated = (occ_grid > 5) & (occ_grid < 250) & ~unknown
    costmap[lethal] = 254
    costmap[free] = 0
    if inflated.any():
        costmap[inflated] = np.clip(253 * (1.0 - occ_grid[inflated].astype(np.float64) / 255.0), 1, 253).astype(np.uint8)
    costmap[unknown] = 255
    return costmap


def _inflate_costmap(grid, radius_cells=3):
    """Vectorized obstacle inflation using scipy's Euclidean distance transform.

    For every cell inside ``radius_cells`` of any lethal cell, raise its cost
    with a linear falloff. Falls back to a slow Python loop if scipy is unavailable.
    """
    inflated = grid.copy()
    if radius_cells <= 0 or not (grid == 254).any():
        return inflated
    try:
        from scipy.ndimage import distance_transform_edt
        free = (grid != 254)
        dist = distance_transform_edt(free).astype(np.float32)
        inside = dist <= radius_cells
        new_cost = np.where(inside, 254.0 * np.maximum(0.0, 1.0 - dist / radius_cells), 0.0)
        update = (new_cost > 0) & (inflated < 254)
        inflated = np.where(update, np.maximum(inflated, new_cost.astype(np.int32)), inflated)
        return inflated
    except ImportError:
        pass
    # Slow fallback: pure Python loop
    lethal_cells = np.argwhere(grid == 254)
    sy, sx = grid.shape
    for cy, cx in lethal_cells:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < sy and 0 <= nx < sx:
                    dist = math.hypot(dx, dy)
                    if dist <= radius_cells and inflated[ny, nx] < 254:
                        cost = int(254 * max(0, 1 - dist / radius_cells))
                        inflated[ny, nx] = max(inflated[ny, nx], cost)
    return inflated


def _grid_to_costmap(grid, res, ox, oy):
    sy, sx = grid.shape
    costmap = sc.Costmap2D(sx, sy, res, ox, oy)
    for my in range(sy):
        for mx in range(sx):
            costmap.setCost(mx, my, int(grid[my, mx]))
    return costmap


def _path_length(points):
    if len(points) < 2:
        return 0.0
    return sum(math.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
               for i in range(1, len(points)))


def _reconstruct_path_with_yaw(path, start_yaw=0.0, goal_yaw=0.0):
    if not path:
        return []
    xs = [float(p[0]) for p in path]
    ys = [float(p[1]) for p in path]
    yaws = []
    for i in range(len(path)):
        if len(path) == 1:
            yaw = start_yaw
        else:
            pi = max(0, i - 1)
            ni = min(len(path) - 1, i + 1)
            if pi == ni:
                yaw = start_yaw
            else:
                dx = xs[ni] - xs[pi]
                dy = ys[ni] - ys[pi]
                yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else start_yaw
        yaws.append(yaw)
    if len(path) > 0:
        yaws[-1] = goal_yaw
    return [(xs[i], ys[i], yaws[i]) for i in range(len(path))]


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


app = Flask(__name__)

STATE_LOCK = Lock()
COSTMAP_GRID = None
MAP_META = None
ESDF_GRID = None

HAS_COMPUTE_ESDF = hasattr(sc, "compute_esdf")


def _rebuild_state():
    global COSTMAP_GRID, MAP_META, ESDF_GRID
    grid, meta = _load_occupancy_map()
    if grid is None:
        grid = np.zeros((200, 200), dtype=np.uint8)
        meta = {
            "size_x": 200, "size_y": 200, "resolution": 0.05,
            "origin_x": 0.0, "origin_y": 0.0,
            "image": np.zeros((200, 200), dtype=np.uint8),
        }
    COSTMAP_GRID = _inflate_costmap(grid, radius_cells=3)
    MAP_META = meta
    if HAS_COMPUTE_ESDF:
        try:
            costmap_obj = _grid_to_costmap(COSTMAP_GRID, meta["resolution"], meta["origin_x"], meta["origin_y"])
            ESDF_GRID = np.asarray(
                sc.compute_esdf(costmap_obj, sc.Costmap2D.LETHAL_OBSTACLE),
                dtype=np.float64,
            ).reshape((meta["size_y"], meta["size_x"]))
        except Exception:
            ESDF_GRID = None
    else:
        ESDF_GRID = None


def _serialize_costmap():
    with STATE_LOCK:
        sy, sx = COSTMAP_GRID.shape
        return {
            "size_x": int(sx), "size_y": int(sy),
            "resolution": MAP_META["resolution"],
            "origin_x": MAP_META["origin_x"], "origin_y": MAP_META["origin_y"],
            "data": COSTMAP_GRID.flatten().tolist(),
            "esdf": ESDF_GRID.flatten().tolist() if ESDF_GRID is not None else None,
            "metadata": {
                "world_width_m": round(sx * MAP_META["resolution"], 2),
                "world_height_m": round(sy * MAP_META["resolution"], 2),
            },
        }


with STATE_LOCK:
    _rebuild_state()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/api/costmap", methods=["GET"])
def get_costmap():
    return jsonify(_serialize_costmap())


@app.route("/api/smooth", methods=["POST"])
def smooth():
    try:
        req = request.get_json(silent=True) or {}
        start_x = float(req.get("start_x", 10.0))
        start_y = float(req.get("start_y", 10.0))
        goal_x = float(req.get("goal_x", 50.0))
        goal_y = float(req.get("goal_y", 30.0))
        start_yaw_deg = float(req.get("start_yaw_deg", 0.0))
        goal_yaw_deg = float(req.get("goal_yaw_deg", 0.0))
        keep_start = _coerce_bool(req.get("keep_start_orientation"), True)
        keep_goal = _coerce_bool(req.get("keep_goal_orientation"), True)

        reference_spacing = max(0.1, float(req.get("reference_spacing_target_m", 0.5)))
        max_curvature = float(req.get("max_curvature", 3.0))
        max_time = float(req.get("max_time", 10.0))
        max_iterations = int(req.get("max_iterations", 50))
        costmap_weight = float(req.get("costmap_weight", 1.0))
        model_weight = float(req.get("model_weight", 20.0))
        fix_weight = float(req.get("fix_weight", 100.0))
        kinematic_curvature_weight = float(req.get("kinematic_curvature_weight", 1.0))
        kinematic_curvature_rate_weight = float(req.get("kinematic_curvature_rate_weight", 5.0))
        kinematic_spacing_weight = float(req.get("kinematic_spacing_weight", 1.0))
        path_length_weight = float(req.get("path_length_weight", 0.1))
        reference_path_weight = float(req.get("reference_path_weight", 0.0))
        obstacle_safe_distance = float(req.get("obstacle_safe_distance", 0.3))
        debug = _coerce_bool(req.get("debug"), False)

        start_yaw_rad = math.radians(start_yaw_deg)
        goal_yaw_rad = math.radians(goal_yaw_deg)
        start_dir = [math.cos(start_yaw_rad), math.sin(start_yaw_rad)]
        end_dir = [math.cos(goal_yaw_rad), math.sin(goal_yaw_rad)]

        with STATE_LOCK:
            costmap_grid = COSTMAP_GRID.copy()
            esdf_grid = ESDF_GRID.copy() if ESDF_GRID is not None else None
            meta = dict(MAP_META)

        t0 = time.time()
        planner = AStarPlanner(costmap_grid, meta["size_x"], meta["size_y"],
                               meta["resolution"], meta["origin_x"], meta["origin_y"])
        raw_path = planner.plan(start_x, start_y, goal_x, goal_y)
        astar_time_ms = (time.time() - t0) * 1000.0

        if not raw_path:
            raise ApiError(ERROR_ASTAR_NO_PATH, "A* could not find a path.", status_code=409)

        sparse_path = downsample_path(raw_path, reference_spacing)
        eigen_path = [[p[0], p[1], 1.0] for p in sparse_path]
        ref_with_yaw = _reconstruct_path_with_yaw(eigen_path, start_yaw_rad, goal_yaw_rad)

        costmap_obj = _grid_to_costmap(costmap_grid, meta["resolution"], meta["origin_x"], meta["origin_y"])

        smoother_params = sc.SmootherParams()
        smoother_params.model_weight_sqrt = math.sqrt(model_weight)
        smoother_params.costmap_weight_sqrt = math.sqrt(costmap_weight)
        smoother_params.cusp_costmap_weight_sqrt = math.sqrt(costmap_weight * 3.0)
        smoother_params.cusp_zone_length = 2.5
        smoother_params.reference_path_weight_sqrt = math.sqrt(reference_path_weight)
        smoother_params.kinematic_curvature_weight_sqrt = math.sqrt(kinematic_curvature_weight)
        smoother_params.kinematic_curvature_rate_weight_sqrt = math.sqrt(kinematic_curvature_rate_weight)
        smoother_params.kinematic_spacing_weight_sqrt = math.sqrt(kinematic_spacing_weight)
        smoother_params.path_length_weight_sqrt = math.sqrt(path_length_weight)
        smoother_params.fix_weight = fix_weight
        smoother_params.max_curvature = max_curvature
        smoother_params.max_time = max_time
        smoother_params.keep_start_orientation = keep_start
        smoother_params.keep_goal_orientation = keep_goal
        smoother_params.obstacle_safe_distance = obstacle_safe_distance

        optimizer_params = sc.OptimizerParams()
        optimizer_params.debug = debug
        optimizer_params.max_iterations = max_iterations

        smoother = sc.ClothoidSmoother()
        smoother.initialize(optimizer_params)

        t1 = time.time()
        smooth_result = smoother.try_smooth(eigen_path, start_dir, end_dir, costmap_obj, smoother_params)
        smooth_time_ms = (time.time() - t1) * 1000.0

        smooth_success = bool(smooth_result["ok"])
        smoothed_path = smooth_result.get("smoothed_path") or smooth_result.get("candidate_path") or ref_with_yaw

        opt_x = [p[0] for p in smoothed_path]
        opt_y = [p[1] for p in smoothed_path]
        opt_theta = [p[2] if len(p) > 2 else 0.0 for p in smoothed_path]

        return jsonify({
            "success": True,
            "smooth_success": smooth_success,
            "astar_time_ms": round(astar_time_ms, 2),
            "smooth_time_ms": round(smooth_time_ms, 2),
            "smooth_message": smooth_result.get("error_message"),
            "smooth_error": {
                "code": smooth_result.get("error_code"),
                "reason": smooth_result.get("error_reason"),
                "details": smooth_result.get("error_details"),
            } if not smooth_success else None,

            "astar_x": [p[0] for p in raw_path],
            "astar_y": [p[1] for p in raw_path],
            "ref_x": [p[0] for p in sparse_path],
            "ref_y": [p[1] for p in sparse_path],
            "opt_x": opt_x, "opt_y": opt_y, "opt_theta": opt_theta,

            "num_astar_pts": len(raw_path),
            "num_ref_pts": len(sparse_path),
            "num_opt_knots": int(smooth_result.get("optimized_knot_count") or 0),
            "num_opt_pts": len(smoothed_path),
            "raw_path_length_m": round(_path_length(raw_path), 3),
            "ref_path_length_m": round(_path_length(sparse_path), 3),
            "opt_path_length_m": round(_path_length(smoothed_path), 3),

            "reference_spacing_target_m": round(reference_spacing, 3),
            "start_yaw_deg": round(start_yaw_deg, 2),
            "goal_yaw_deg": round(goal_yaw_deg, 2),
            "max_curvature": round(max_curvature, 4),
            "optimized_knot_count": int(smooth_result.get("optimized_knot_count") or 0),
            "target_spacing_m": float(smooth_result.get("target_spacing_m") or 0.0),
        })

    except ApiError:
        raise
    except Exception as e:
        traceback.print_exc()
        return _error_response(e, default_status=400)


@app.errorhandler(ApiError)
def _handle_api_error(exc: ApiError):
    return _error_response(exc)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("SC_WEBAPP_HOST", "127.0.0.1"),
        port=int(os.environ.get("SC_WEBAPP_PORT", "5005")),
        debug=os.environ.get("SC_WEBAPP_DEBUG", "1") in {"1", "true", "yes"},
        use_reloader=os.environ.get("SC_WEBAPP_RELOADER", "0") in {"1", "true", "yes"},
    )
