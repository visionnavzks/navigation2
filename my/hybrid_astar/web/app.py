"""Flask web application: Hybrid A* planner debugging visualization.

Usage
-----
    cd my/hybrid_astar
    # Build the pybind11 module first (see CMakeLists.txt, BUILD_PYTHON=ON)
    python3 web/app.py
"""

import os
import sys
import math
import time
import traceback

from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# Path setup – allow importing the built pybind11 module
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)

sys.path.insert(0, _parent_dir)
_build_dir = os.path.join(_parent_dir, "build")
if os.path.isdir(_build_dir):
    sys.path.insert(0, _build_dir)

import py_hybrid_astar as pha  # noqa: E402

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Costmap defaults
# ---------------------------------------------------------------------------
DEFAULT_SIZE_X = 200
DEFAULT_SIZE_Y = 200
DEFAULT_RESOLUTION = 0.1  # m/cell
DEFAULT_ORIGIN_X = 0.0
DEFAULT_ORIGIN_Y = 0.0
INFLATION_RADIUS_CELLS = 5

DEFAULT_OBSTACLE_RECTS = [
    (60, 40, 80, 100),
    (120, 60, 140, 160),
    (30, 130, 90, 150),
    (150, 20, 170, 80),
]

DEFAULT_START = (10, 10, 0.0)   # (cell_x, cell_y, theta_rad)
DEFAULT_GOAL = (190, 190, 0.0)

# ---------------------------------------------------------------------------
# In-memory costmap state
# ---------------------------------------------------------------------------
_costmap_grid = None
_obstacle_rects = list(DEFAULT_OBSTACLE_RECTS)


def _build_costmap(obstacle_rects):
    """Build a 200x200 costmap with inflated rectangular obstacles."""
    grid = [[0] * DEFAULT_SIZE_X for _ in range(DEFAULT_SIZE_Y)]

    # Fill obstacles with lethal cost
    for (x0, y0, x1, y1) in obstacle_rects:
        for y in range(max(0, y0), min(DEFAULT_SIZE_Y, y1)):
            for x in range(max(0, x0), min(DEFAULT_SIZE_X, x1)):
                grid[y][x] = 254  # OCCUPIED_COST

    # Simple inflation
    inflated = [row[:] for row in grid]
    r = INFLATION_RADIUS_CELLS
    for y in range(DEFAULT_SIZE_Y):
        for x in range(DEFAULT_SIZE_X):
            if grid[y][x] == 254:
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < DEFAULT_SIZE_Y and 0 <= nx < DEFAULT_SIZE_X:
                            dist = math.hypot(dx, dy)
                            if dist <= r and inflated[ny][nx] != 254:
                                cost = int(253 * (1.0 - dist / r))
                                inflated[ny][nx] = max(inflated[ny][nx], cost)

    return inflated


def _get_costmap():
    global _costmap_grid
    if _costmap_grid is None:
        _costmap_grid = _build_costmap(_obstacle_rects)
    return _costmap_grid


def _cell_to_world(cx, cy):
    return (
        cx * DEFAULT_RESOLUTION + DEFAULT_ORIGIN_X,
        cy * DEFAULT_RESOLUTION + DEFAULT_ORIGIN_Y,
    )


def _world_to_cell(wx, wy):
    cx = int((wx - DEFAULT_ORIGIN_X) / DEFAULT_RESOLUTION)
    cy = int((wy - DEFAULT_ORIGIN_Y) / DEFAULT_RESOLUTION)
    return max(0, min(cx, DEFAULT_SIZE_X - 1)), max(0, min(cy, DEFAULT_SIZE_Y - 1))


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/costmap", methods=["GET"])
def api_costmap():
    grid = _get_costmap()
    flat = [cell for row in grid for cell in row]
    return jsonify({
        "size_x": DEFAULT_SIZE_X,
        "size_y": DEFAULT_SIZE_Y,
        "resolution": DEFAULT_RESOLUTION,
        "origin_x": DEFAULT_ORIGIN_X,
        "origin_y": DEFAULT_ORIGIN_Y,
        "data": flat,
        "obstacle_rects": _obstacle_rects,
        "start": DEFAULT_START,
        "goal": DEFAULT_GOAL,
    })


@app.route("/api/obstacles", methods=["POST"])
def api_obstacles():
    global _costmap_grid, _obstacle_rects
    body = request.get_json(force=True)
    rects = body.get("obstacle_rects")
    if rects is None:
        return jsonify({"error": "missing obstacle_rects"}), 400

    _obstacle_rects = [tuple(r) for r in rects]
    _costmap_grid = None  # invalidate cache
    _ = _get_costmap()
    return jsonify({"ok": True, "obstacle_rects": _obstacle_rects})


@app.route("/api/plan", methods=["POST"])
def api_plan():
    body = request.get_json(force=True)

    # Extract start / goal (world coords from frontend)
    start_x = float(body.get("start_x", DEFAULT_START[0] * DEFAULT_RESOLUTION))
    start_y = float(body.get("start_y", DEFAULT_START[1] * DEFAULT_RESOLUTION))
    start_theta = float(body.get("start_theta", DEFAULT_START[2]))
    goal_x = float(body.get("goal_x", DEFAULT_GOAL[0] * DEFAULT_RESOLUTION))
    goal_y = float(body.get("goal_y", DEFAULT_GOAL[1] * DEFAULT_RESOLUTION))
    goal_theta = float(body.get("goal_theta", DEFAULT_GOAL[2]))

    # Planner parameters
    motion_model = body.get("motion_model", "DUBIN")
    tolerance = float(body.get("tolerance", 0.25))
    angle_bins = int(body.get("angle_bins", 72))
    max_planning_time = float(body.get("max_planning_time", 5.0))
    allow_unknown = bool(body.get("allow_unknown", True))
    smooth_path = bool(body.get("smooth_path", True))
    min_turning_radius = float(body.get("minimum_turning_radius", 1.5))
    reverse_penalty = float(body.get("reverse_penalty", 2.0))
    cost_penalty = float(body.get("cost_penalty", 2.0))
    max_iterations = int(body.get("max_iterations", 1000000))

    # Build costmap for hybrid A*
    grid = _get_costmap()
    flat_costs = [cell for row in grid for cell in row]
    ha_costmap = pha.make_costmap(
        DEFAULT_SIZE_X, DEFAULT_SIZE_Y, DEFAULT_RESOLUTION,
        DEFAULT_ORIGIN_X, DEFAULT_ORIGIN_Y, flat_costs,
    )

    # Configure planner
    config = pha.SmacPlannerHybridConfig()
    config.motion_model_for_search = motion_model
    config.tolerance = tolerance
    config.angle_quantization_bins = angle_bins
    config.max_planning_time = max_planning_time
    config.allow_unknown = allow_unknown
    config.smooth_path = smooth_path
    config.max_iterations = max_iterations
    config.goal_heading_mode = "DEFAULT"
    config.use_radius = True
    config.circumscribed_radius = 0.1
    config.inflation_radius = 0.1

    config.search_info.minimum_turning_radius = min_turning_radius
    config.search_info.reverse_penalty = reverse_penalty
    config.search_info.cost_penalty = cost_penalty

    planner = pha.SmacPlannerHybrid()
    planner.configure(ha_costmap, config)

    # Run planner
    t0 = time.time()
    result = planner.create_plan(
        pha.Pose(start_x, start_y, start_theta),
        pha.Pose(goal_x, goal_y, goal_theta),
    )
    elapsed_ms = (time.time() - t0) * 1000.0

    path = result.get("path", [])
    ok = result.get("ok", False)
    message = result.get("message", "")

    return jsonify({
        "ok": ok,
        "message": message,
        "path": path,
        "elapsed_ms": round(elapsed_ms, 2),
        "path_length": len(path),
        "config": {
            "motion_model": motion_model,
            "tolerance": tolerance,
            "angle_bins": angle_bins,
            "max_planning_time": max_planning_time,
            "minimum_turning_radius": min_turning_radius,
            "reverse_penalty": reverse_penalty,
            "cost_penalty": cost_penalty,
            "smooth_path": smooth_path,
        },
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HA_WEBAPP_HOST", "127.0.0.1"),
        port=int(os.environ.get("HA_WEBAPP_PORT", "5006")),
        debug=_env_flag("HA_WEBAPP_DEBUG", True),
        use_reloader=_env_flag("HA_WEBAPP_RELOADER", False),
    )
