"""Interactive web demo for kinematic_path_smoother.

Run from the package root:

    ./run_web_app.sh

The backend tries to import the optional pybind11 module first. If it is not built,
the page still runs with a deterministic Python-only smoothing fallback so the UI
remains usable while dependencies are being configured.
"""

from __future__ import annotations

import heapq
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for candidate in (ROOT, os.path.join(ROOT, "build"), os.path.join(ROOT, "build-web")):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import py_kinematic_path_smoother as native_smoother
except Exception:  # pragma: no cover - intentionally reported in API payload
    native_smoother = None

app = Flask(__name__)

LETHAL = 254
FREE = 0


@dataclass(frozen=True)
class GridSpec:
    width: int = 92
    height: int = 62
    resolution: float = 0.1
    origin_x: float = 0.0
    origin_y: float = 0.0


DEFAULT_SPEC = GridSpec()
DEFAULT_START = {"x": 0.8, "y": 0.8, "yaw_deg": 25.0}
DEFAULT_GOAL = {"x": 8.2, "y": 5.0, "yaw_deg": 0.0}
DEFAULT_OBSTACLES = [
    {"x": 22, "y": 0, "w": 8, "h": 38},
    {"x": 44, "y": 24, "w": 8, "h": 38},
    {"x": 64, "y": 0, "w": 8, "h": 34},
    {"x": 12, "y": 46, "w": 18, "h": 5},
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def world_to_cell(spec: GridSpec, x: float, y: float) -> tuple[int, int]:
    return (
        int(math.floor((x - spec.origin_x) / spec.resolution)),
        int(math.floor((y - spec.origin_y) / spec.resolution)),
    )


def cell_to_world(spec: GridSpec, mx: int, my: int) -> tuple[float, float]:
    return (
        spec.origin_x + (mx + 0.5) * spec.resolution,
        spec.origin_y + (my + 0.5) * spec.resolution,
    )


def direction_from_yaw(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


def build_costmap(spec: GridSpec, obstacles: Iterable[dict], inflate_radius_m: float) -> list[int]:
    costs = [FREE] * (spec.width * spec.height)
    for obstacle in obstacles:
        ox = int(obstacle.get("x", 0))
        oy = int(obstacle.get("y", 0))
        ow = max(1, int(obstacle.get("w", 1)))
        oh = max(1, int(obstacle.get("h", 1)))
        for y in range(max(0, oy), min(spec.height, oy + oh)):
            for x in range(max(0, ox), min(spec.width, ox + ow)):
                costs[y * spec.width + x] = LETHAL

    inflate_cells = int(math.ceil(max(0.0, inflate_radius_m) / spec.resolution))
    if inflate_cells <= 0:
        return costs

    inflated = costs[:]
    lethal_cells = [
        (index % spec.width, index // spec.width)
        for index, value in enumerate(costs)
        if value >= LETHAL
    ]
    for ox, oy in lethal_cells:
        for dy in range(-inflate_cells, inflate_cells + 1):
            for dx in range(-inflate_cells, inflate_cells + 1):
                if dx * dx + dy * dy > inflate_cells * inflate_cells:
                    continue
                x = ox + dx
                y = oy + dy
                if 0 <= x < spec.width and 0 <= y < spec.height:
                    inflated[y * spec.width + x] = LETHAL
    return inflated


def astar(spec: GridSpec, costs: list[int], start_world: dict, goal_world: dict) -> list[tuple[int, int]]:
    start = world_to_cell(spec, float(start_world["x"]), float(start_world["y"]))
    goal = world_to_cell(spec, float(goal_world["x"]), float(goal_world["y"]))

    def free(cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < spec.width and 0 <= y < spec.height and costs[y * spec.width + x] < LETHAL

    if not free(start) or not free(goal):
        raise ValueError("Start or goal is inside an obstacle or outside the map")

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    open_heap = [(heuristic(start), 0.0, start)]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    best = {start: 0.0}

    while open_heap:
        _, cost_so_far, cell = heapq.heappop(open_heap)
        if cell == goal:
            break
        if cost_so_far > best.get(cell, math.inf):
            continue
        for dx, dy, step_cost in neighbors:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not free(nxt):
                continue
            new_cost = cost_so_far + step_cost
            if new_cost >= best.get(nxt, math.inf):
                continue
            best[nxt] = new_cost
            parent[nxt] = cell
            heapq.heappush(open_heap, (new_cost + heuristic(nxt), new_cost, nxt))

    if goal not in best:
        raise ValueError("A* failed to find a collision-free path")

    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def simplify_cells(cells: list[tuple[int, int]], stride: int) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells
    stride = max(1, int(stride))
    simplified = [cells[0]]
    last_direction = None
    for index in range(1, len(cells) - 1):
        prev_cell = cells[index - 1]
        cell = cells[index]
        next_cell = cells[index + 1]
        direction = (next_cell[0] - prev_cell[0], next_cell[1] - prev_cell[1])
        turning = last_direction is not None and direction != last_direction
        if turning or index % stride == 0:
            simplified.append(cell)
        last_direction = direction
    simplified.append(cells[-1])
    return simplified


def cells_to_reference(spec: GridSpec, cells: list[tuple[int, int]]) -> list[list[float]]:
    return [[*cell_to_world(spec, x, y), 1.0] for x, y in cells]


def python_fallback_smooth(path: list[list[float]], iterations: int, upsample: int) -> list[list[float]]:
    points = [[p[0], p[1]] for p in path]
    for _ in range(max(0, iterations)):
      if len(points) <= 2:
          break
      refined = [points[0]]
      for i in range(1, len(points) - 1):
          px, py = points[i - 1]
          x, y = points[i]
          nx, ny = points[i + 1]
          refined.append([0.25 * px + 0.5 * x + 0.25 * nx, 0.25 * py + 0.5 * y + 0.25 * ny])
      refined.append(points[-1])
      points = refined

    dense: list[list[float]] = []
    factor = max(1, upsample)
    for i in range(len(points) - 1):
      x0, y0 = points[i]
      x1, y1 = points[i + 1]
      yaw = math.atan2(y1 - y0, x1 - x0)
      for j in range(factor):
          t = j / factor
          dense.append([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, yaw])
    if points:
      if len(points) >= 2:
          yaw = math.atan2(points[-1][1] - points[-2][1], points[-1][0] - points[-2][0])
      else:
          yaw = 0.0
      dense.append([points[-1][0], points[-1][1], yaw])
    return dense


def parse_payload(data: dict) -> tuple[GridSpec, dict, dict, list[dict], dict]:
    spec = GridSpec(
        width=int(data.get("width", DEFAULT_SPEC.width)),
        height=int(data.get("height", DEFAULT_SPEC.height)),
        resolution=float(data.get("resolution", DEFAULT_SPEC.resolution)),
        origin_x=float(data.get("origin_x", DEFAULT_SPEC.origin_x)),
        origin_y=float(data.get("origin_y", DEFAULT_SPEC.origin_y)),
    )
    start = data.get("start") or DEFAULT_START
    goal = data.get("goal") or DEFAULT_GOAL
    obstacles = data.get("obstacles") or DEFAULT_OBSTACLES
    params = data.get("params") or {}
    return spec, start, goal, obstacles, params


@app.get("/")
def index():
    return render_template(
        "index.html",
        native_available=native_smoother is not None,
        spec=DEFAULT_SPEC,
        start=DEFAULT_START,
        goal=DEFAULT_GOAL,
        obstacles=DEFAULT_OBSTACLES,
    )


@app.get("/api/default_scene")
def default_scene():
    return jsonify(
        {
            "width": DEFAULT_SPEC.width,
            "height": DEFAULT_SPEC.height,
            "resolution": DEFAULT_SPEC.resolution,
            "origin_x": DEFAULT_SPEC.origin_x,
            "origin_y": DEFAULT_SPEC.origin_y,
            "start": DEFAULT_START,
            "goal": DEFAULT_GOAL,
            "obstacles": DEFAULT_OBSTACLES,
            "native_available": native_smoother is not None,
        }
    )


@app.post("/api/solve")
def solve():
    started = time.perf_counter()
    try:
        spec, start, goal, obstacles, params = parse_payload(request.get_json(force=True) or {})
        footprint_radius = float(params.get("footprint_radius", 0.2))
        costs = build_costmap(spec, obstacles, footprint_radius)
        raw_cells = astar(spec, costs, start, goal)
        reference_cells = simplify_cells(raw_cells, int(params.get("reference_stride", 4)))
        reference_path = cells_to_reference(spec, reference_cells)

        smoother_params = {
            "model_weight": float(params.get("model_weight", 8.0)),
            "reference_weight": float(params.get("reference_weight", 0.5)),
            "obstacle_weight": float(params.get("obstacle_weight", 0.0)),
            "cusp_obstacle_weight": float(params.get("cusp_obstacle_weight", 0.0)),
            "curvature_weight": float(params.get("curvature_weight", 0.2)),
            "curvature_rate_weight": float(params.get("curvature_rate_weight", 1.0)),
            "spacing_weight": float(params.get("spacing_weight", 1.0)),
            "length_weight": float(params.get("length_weight", 0.01)),
            "fix_weight": float(params.get("fix_weight", 100.0)),
            "max_curvature": float(params.get("max_curvature", 2.5)),
            "max_segment_length": float(params.get("max_segment_length", 1.0)),
            "max_reference_deviation": float(params.get("max_reference_deviation", 0.6)),
            "max_time": float(params.get("max_time", 2.0)),
            "use_exact_esdf": bool(params.get("use_exact_esdf", True)),
            "obstacle_safe_distance": float(params.get("obstacle_safe_distance", 0.25)),
            "footprint_radius": footprint_radius,
            "path_downsampling_factor": 1,
            "path_upsampling_factor": int(params.get("path_upsampling_factor", 4)),
            "reversing_enabled": True,
            "keep_start_orientation": bool(params.get("keep_start_orientation", True)),
            "keep_goal_orientation": bool(params.get("keep_goal_orientation", True)),
            "goal_longitudinal_tolerance": float(params.get("goal_longitudinal_tolerance", 0.0)),
            "goal_lateral_tolerance": float(params.get("goal_lateral_tolerance", 0.0)),
            "goal_orientation_tolerance": math.radians(float(params.get("goal_orientation_tolerance_deg", 5.0))),
        }
        optimizer_params = {
            "linear_solver": str(params.get("linear_solver", "DENSE_QR")),
            "max_iterations": int(params.get("max_iterations", 60)),
            "function_tolerance": 1e-6,
            "gradient_tolerance": 1e-10,
            "parameter_tolerance": 1e-8,
            "debug": bool(params.get("debug", False)),
        }

        backend = "python-fallback"
        smooth_result = None
        if native_smoother is not None:
            try:
                smooth_result = native_smoother.smooth_path(
                    reference_path,
                    direction_from_yaw(float(start.get("yaw_deg", 0.0))),
                    direction_from_yaw(float(goal.get("yaw_deg", 0.0))),
                    spec.width,
                    spec.height,
                    spec.resolution,
                    spec.origin_x,
                    spec.origin_y,
                    costs,
                    smoother_params,
                    optimizer_params,
                )
                backend = "cpp"
            except Exception as exc:
                smooth_result = {
                    "success": False,
                    "path": [],
                    "optimized_path": python_fallback_smooth(
                        reference_path, 3, int(smoother_params["path_upsampling_factor"])
                    ),
                    "failure": {"reason": "native_exception", "message": str(exc), "index": -1},
                    "optimized_knot_count": len(reference_path),
                    "target_spacing": 0.0,
                }
                backend = "python-fallback-after-cpp-error"

        if smooth_result is None:
            smoothed = python_fallback_smooth(
                reference_path, 3, int(smoother_params["path_upsampling_factor"])
            )
            smooth_result = {
                "success": True,
                "path": smoothed,
                "optimized_path": smoothed,
                "failure": None,
                "optimized_knot_count": len(reference_path),
                "target_spacing": 0.0,
            }

        raw_path = [[*cell_to_world(spec, x, y), 1.0] for x, y in raw_cells]
        return jsonify(
            {
                "success": bool(smooth_result.get("success")),
                "backend": backend,
                "native_available": native_smoother is not None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "costmap": costs,
                "raw_path": raw_path,
                "reference_path": reference_path,
                "smoothed_path": smooth_result.get("path") or smooth_result.get("optimized_path") or [],
                "optimized_path": smooth_result.get("optimized_path") or [],
                "failure": smooth_result.get("failure"),
                "stats": {
                    "raw_points": len(raw_path),
                    "reference_points": len(reference_path),
                    "optimized_knot_count": int(smooth_result.get("optimized_knot_count", 0)),
                    "target_spacing": float(smooth_result.get("target_spacing", 0.0)),
                },
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc), "error": type(exc).__name__}), 400


if __name__ == "__main__":
    port = int(os.environ.get("KINEMATIC_SMOOTHER_WEB_PORT", "5055"))
    app.run(host="127.0.0.1", port=port, debug=False)
