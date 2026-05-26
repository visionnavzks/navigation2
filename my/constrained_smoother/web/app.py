"""Flask web application: A* planning + kinematic smoother visualization.

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
from dataclasses import dataclass
from threading import Lock

import numpy as np
from flask import Flask, request, jsonify, render_template, Response

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


def _normalize_capsule_mode(value):
    mode = str(value or "conservative").strip().lower()
    return mode if mode in {"exact", "conservative"} else "conservative"


ERROR_INVALID_REQUEST = "CS_INVALID_REQUEST"
ERROR_ASTAR_NO_PATH = "CS_ASTAR_NO_PATH"
ERROR_INTERNAL = "CS_INTERNAL_ERROR"
ERROR_FINAL_PATH_NONFINITE = "CS_FINAL_PATH_NONFINITE"
ERROR_FINAL_PATH_OUT_OF_BOUNDS = "CS_FINAL_PATH_OUT_OF_BOUNDS"
ERROR_FINAL_PATH_COLLISION = "CS_FINAL_PATH_COLLISION"
PCS_ERROR_CODE_BY_TYPE = {
    "InvalidPathError": getattr(pcs, "ERROR_INVALID_PATH", "CS_INVALID_PATH"),
    "FailedToSmoothPathError": getattr(pcs, "ERROR_FAILED_TO_SMOOTH_PATH", "CS_SMOOTHING_FAILED"),
    "InvalidCostmapError": getattr(pcs, "ERROR_INVALID_COSTMAP", "CS_INVALID_COSTMAP"),
    "PrecomputedEsdfSizeMismatchError": getattr(
        pcs,
        "ERROR_PRECOMPUTED_ESDF_SIZE_MISMATCH",
        "CS_PRECOMPUTED_ESDF_SIZE_MISMATCH",
    ),
}


class ApiError(Exception):
    def __init__(self, code, message, status_code=400, source="server", details=None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.source = str(source)
        self.details = details or {}

    def to_payload(self):
        error_payload = {
            "code": self.code,
            "message": self.message,
            "source": self.source,
        }
        if self.details:
            error_payload["details"] = self.details
        return {
            "success": False,
            "message": self.message,
            "error": error_payload,
        }


def _extract_prefixed_error_code(message):
    if not isinstance(message, str) or not message.startswith("CS_"):
        return None, message
    prefix, separator, remainder = message.partition(": ")
    if not separator:
        return None, message
    return prefix, remainder


def _exception_to_api_error(exc, *, default_status=400, default_source="server"):
    if isinstance(exc, ApiError):
        return exc

    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message:
        message = str(exc) or "Unknown error"

    prefixed_code, prefixed_message = _extract_prefixed_error_code(message)
    if prefixed_code:
        return ApiError(
            code=prefixed_code,
            message=prefixed_message,
            status_code=default_status,
            source=default_source,
            details={"exception_type": type(exc).__name__},
        )

    code = getattr(exc, "code", None)

    # ---- API error normalization helpers ----

    numeric_code = getattr(exc, "numeric_code", None)
    if code:
        details = {"exception_type": type(exc).__name__}
        if numeric_code is not None:
            details["numeric_code"] = int(numeric_code)
        return ApiError(
            code=code,
            message=message,
            status_code=default_status,
            source=default_source,
            details=details,
        )

    exception_type = type(exc).__name__
    mapped_code = PCS_ERROR_CODE_BY_TYPE.get(exception_type)
    if mapped_code:
        return ApiError(
            code=mapped_code,
            message=message,
            status_code=default_status,
            source=default_source,
            details={"exception_type": exception_type},
        )

    if isinstance(exc, ValueError):
        return ApiError(
            code=ERROR_INVALID_REQUEST,
            message=message,
            status_code=400,
            source="request",
            details={"exception_type": type(exc).__name__},
        )

    return ApiError(
        code=ERROR_INTERNAL,
        message=message,
        status_code=500,
        source=default_source,
        details={"exception_type": type(exc).__name__},
    )


def _error_response(exc, *, default_status=400, default_source="server"):
    api_error = _exception_to_api_error(
        exc,
        default_status=default_status,
        default_source=default_source,
    )
    return jsonify(api_error.to_payload()), api_error.status_code


def _error_payload(exc, *, default_status=400, default_source="server"):
    return _exception_to_api_error(
        exc,
        default_status=default_status,
        default_source=default_source,
    ).to_payload()["error"]


def _build_smoother_error_payload(result):
    error_payload = {
        "code": str(result["error_code"]),
        "message": str(result["error_message"]),
        "source": "smoother",
    }

    details = {}
    failure_reason = result.get("error_reason")
    if failure_reason is not None:
        details["failure_reason"] = str(failure_reason)

    error_details = result.get("error_details")
    if isinstance(error_details, dict):
        details.update(error_details)

    if details:
        error_payload["details"] = details

    return error_payload


_PRE_FINALIZE_FAILURE_REASONS = {
    "solver_rejected_solution",
    "no_cost_improvement",
}


def _failure_has_displayable_candidate_path(result):
    candidate_path = result.get("path")
    if candidate_path is None:
        return False

    failure_reason = result.get("error_reason")
    return failure_reason not in _PRE_FINALIZE_FAILURE_REASONS


# ---- Pipeline stage reporting helpers ----

def _make_pipeline_stage(
    stage_key,
    label,
    status,
    message,
    *,
    elapsed_ms=None,
    error_code=None,
    path_key=None,
    details=None,
):
    stage = {
        "key": str(stage_key),
        "label": str(label),
        "status": str(status),
        "message": str(message),
        "error_code": str(error_code) if error_code else None,
    }
    if elapsed_ms is not None:
        stage["elapsed_ms"] = round(float(elapsed_ms), 2)
    if path_key is not None:
        stage["path"] = str(path_key)
    if details:
        stage["details"] = details
    return stage


def _build_pipeline_payload(stages):
    active_stages = [stage for stage in stages if stage is not None]
    overall_status = "ok"
    for stage in active_stages:
        if stage["status"] == "error":
            overall_status = "error"
            break
        if stage["status"] == "fallback" and overall_status == "ok":
            overall_status = "fallback"

    summary_parts = []
    for stage in active_stages:
        summary = f"{stage['label']}: {stage['status']}"
        if stage.get("path"):
            summary += f" ({stage['path']})"
        if stage.get("error_code"):
            summary += f" [{stage['error_code']}]"
        summary_parts.append(summary)

    return {
        "overall_status": overall_status,
        "summary": " -> ".join(summary_parts),
        "stages": active_stages,
    }


# ---- Planner / smoother / validation pipeline stages ----

def _run_astar_stage(
    costmap_grid,
    esdf_grid,
    planner_costmap,
    footprint_model,
    planner_penalty_weight,
    start_x,
    start_y,
    goal_x,
    goal_y,
    reference_spacing_target_m,
    start_yaw_rad,
    goal_yaw_rad,
    keep_start_orientation,
    keep_goal_orientation,
):
    """Run the planner stage and normalize its result into a stable dict.

    Return contract:
        planner: Planner instance whose ESDF may be reused by the smoother stage.
        raw_path: Raw A* path as world-coordinate tuples.
        sparse_path: Downsampled reference path used as the smoother input chain.
        eigen_path: Reference path encoded as `[x, y, direction_sign]` triples.
        reference_with_yaw: Display-ready reference path with reconstructed yaw.
        astar_time_ms: Planner runtime in milliseconds.
        stage: Pipeline-stage payload for frontend status rendering.
    """
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
    astar_time_ms = (time.time() - t0) * 1000.0

    if not raw_path:
        failure = _diagnose_astar_no_path(
            costmap_grid,
            esdf_grid,
            footprint_model,
            start_x,
            start_y,
            goal_x,
            goal_y,
        )
        raise ApiError(
            ERROR_ASTAR_NO_PATH,
            failure["message"],
            status_code=409,
            source="planner",
            details=failure,
        )

    raw_path = [(float(point[0]), float(point[1])) for point in raw_path]
    sparse_path = downsample_path(raw_path, reference_spacing_target_m)
    eigen_path = [[point[0], point[1], 1.0] for point in sparse_path]
    reference_with_yaw = _reconstruct_path_with_yaw(
        eigen_path,
        start_yaw=start_yaw_rad,
        goal_yaw=goal_yaw_rad,
        keep_start_orientation=keep_start_orientation,
        keep_goal_orientation=keep_goal_orientation,
    )

    stage = _make_pipeline_stage(
        "planner",
        "A*",
        "ok",
        f"A* produced {len(raw_path)} raw pose(s) and {len(sparse_path)} reference pose(s).",
        elapsed_ms=astar_time_ms,
        path_key="reference_path",
    )

    return {
        "planner": planner,
        "raw_path": raw_path,
        "sparse_path": sparse_path,
        "eigen_path": eigen_path,
        "reference_with_yaw": reference_with_yaw,
        "astar_time_ms": astar_time_ms,
        "stage": stage,
    }


def _parse_manual_reference_path(raw_path):
    if raw_path is None:
        return None
    if not isinstance(raw_path, list):
        raise ApiError(
            ERROR_INVALID_REQUEST,
            "manual_reference_path must be a list of poses.",
            status_code=400,
            source="request",
        )

    parsed_path = []
    for index, pose in enumerate(raw_path):
        if isinstance(pose, dict):
            x_value = pose.get("x")
            y_value = pose.get("y")
            direction_value = pose.get("direction_sign", 1.0)
        elif isinstance(pose, (list, tuple)) and len(pose) >= 2:
            x_value = pose[0]
            y_value = pose[1]
            direction_value = pose[2] if len(pose) >= 3 else 1.0
        else:
            raise ApiError(
                ERROR_INVALID_REQUEST,
                f"manual_reference_path[{index}] must contain x/y and optional direction_sign.",
                status_code=400,
                source="request",
            )

        x = float(x_value)
        y = float(y_value)
        direction_sign = -1.0 if float(direction_value) < 0.0 else 1.0
        parsed_path.append((x, y, direction_sign))

    if len(parsed_path) < 2:
        raise ApiError(
            ERROR_INVALID_REQUEST,
            "manual_reference_path must contain at least 2 poses.",
            status_code=400,
            source="request",
        )

    return parsed_path


def _run_manual_reference_stage(
    manual_reference_path,
    start_yaw_rad,
    goal_yaw_rad,
    keep_start_orientation,
    keep_goal_orientation,
):
    raw_path = [(float(point[0]), float(point[1])) for point in manual_reference_path]
    eigen_path = [
        [float(point[0]), float(point[1]), -1.0 if float(point[2]) < 0.0 else 1.0]
        for point in manual_reference_path
    ]
    reference_with_yaw = _reconstruct_path_with_yaw(
        eigen_path,
        start_yaw=start_yaw_rad,
        goal_yaw=goal_yaw_rad,
        keep_start_orientation=keep_start_orientation,
        keep_goal_orientation=keep_goal_orientation,
    )
    stage = _make_pipeline_stage(
        "planner",
        "Manual Reference",
        "ok",
        f"Manual reference provided {len(eigen_path)} pose(s), including signed direction metadata.",
        elapsed_ms=0.0,
        path_key="reference_path",
        details={"source": "manual_reference_path"},
    )
    return {
        "planner": None,
        "raw_path": raw_path,
        "sparse_path": raw_path,
        "eigen_path": eigen_path,
        "reference_with_yaw": reference_with_yaw,
        "astar_time_ms": 0.0,
        "stage": stage,
    }


app = Flask(__name__)

# ---- Shared in-memory scene state ----

# ---------------------------------------------------------------------------
# Default costmap: 200×200 cells, 0.1 m resolution → 20 m × 20 m world area
# ---------------------------------------------------------------------------
DEFAULT_SIZE_X = 200
DEFAULT_SIZE_Y = 200
DEFAULT_RESOLUTION = 0.1
DEFAULT_ORIGIN_X = 0.0
DEFAULT_ORIGIN_Y = 0.0
DEFAULT_REFERENCE_SPACING_TARGET_M = DEFAULT_RESOLUTION * 3
DEFAULT_CAPSULE_SAMPLING_TOLERANCE_M = max(DEFAULT_RESOLUTION * 0.35, 0.02)
INFLATION_RADIUS_CELLS = 5
KINEMATIC_GOAL_ORIENTATION_TOLERANCE_RAD = 0.1
DEFAULT_OBSTACLE_RECTS = [
    # (x_start, y_start, x_end, y_end) in cell coordinates
    (60, 40, 80, 100),
    (120, 60, 140, 160),
    (30, 130, 90, 150),
    (150, 20, 170, 80),
]
CURRENT_OBSTACLE_RECTS = [tuple(rect) for rect in DEFAULT_OBSTACLE_RECTS]
STATE_LOCK = Lock()
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
            "The C++ A* planner and kinematic smoother both optimize ESDF-derived obstacle penalties."
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


def _split_path_xyz(path):
    """Split a (x, y, yaw) path list into x/y/yaw arrays for downstream consumers."""
    xs = [pose[0] for pose in path]
    ys = [pose[1] for pose in path]
    yaws = [pose[2] for pose in path]
    return xs, ys, yaws


def _normalize_angle_rad(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _reconstruct_path_with_yaw(
    path,
    start_yaw=None,
    goal_yaw=None,
    keep_start_orientation=True,
    keep_goal_orientation=True,
):
    """Convert a path using direction-sign z into a path using yaw z."""
    if not path:
        return []

    xs = [float(pose[0]) for pose in path]
    ys = [float(pose[1]) for pose in path]
    direction_signs = []
    for pose in path:
        direction_sign = float(pose[2]) if len(pose) >= 3 else 1.0
        direction_signs.append(-1.0 if direction_sign < 0.0 else 1.0)

    pose_count = len(path)
    fallback_yaw = _normalize_angle_rad(float(start_yaw)) if start_yaw is not None else 0.0
    yaws = []
    for index in range(pose_count):
        if pose_count == 1:
            yaw = fallback_yaw
        else:
            prev_index = max(0, index - 1)
            next_index = min(pose_count - 1, index + 1)
            if prev_index == next_index:
                yaw = fallback_yaw
            else:
                delta_x = xs[next_index] - xs[prev_index]
                delta_y = ys[next_index] - ys[prev_index]
                if math.hypot(delta_x, delta_y) <= 1e-6:
                    yaw = fallback_yaw
                else:
                    yaw = math.atan2(delta_y, delta_x)
                    if direction_signs[index] < 0.0:
                        yaw += math.pi
                    yaw = _normalize_angle_rad(yaw)
            fallback_yaw = yaw
        yaws.append(yaw)

    if keep_start_orientation and start_yaw is not None:
        yaws[0] = _normalize_angle_rad(float(start_yaw))
    if keep_goal_orientation and goal_yaw is not None:
        yaws[-1] = _normalize_angle_rad(float(goal_yaw))

    return [
        (xs[index], ys[index], yaws[index])
        for index in range(pose_count)
    ]


def _serialize_validation_scalar(value, digits=4):
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return round(numeric_value, digits)


def _serialize_angle_pair(radians, digits_rad=4, digits_deg=2):
    numeric_value = float(radians)
    if not math.isfinite(numeric_value):
        return {"rad": None, "deg": None}
    return {
        "rad": round(numeric_value, digits_rad),
        "deg": round(math.degrees(numeric_value), digits_deg),
    }


def _build_goal_orientation_diagnostics(path, goal_yaw_rad, tolerance_rad):
    if not path:
        return None

    terminal_pose_yaw = float(path[-1][2])
    terminal_pose_yaw_error = _normalize_angle_rad(terminal_pose_yaw - goal_yaw_rad)

    terminal_segment_heading = terminal_pose_yaw
    terminal_segment_error = terminal_pose_yaw_error
    if len(path) >= 2:
        dx = float(path[-1][0]) - float(path[-2][0])
        dy = float(path[-1][1]) - float(path[-2][1])
        if math.hypot(dx, dy) > 1e-9:
            terminal_segment_heading = math.atan2(dy, dx)
            terminal_segment_error = _normalize_angle_rad(terminal_segment_heading - goal_yaw_rad)

    return {
        "expected_goal_heading": _serialize_angle_pair(goal_yaw_rad),
        "terminal_segment_heading": _serialize_angle_pair(terminal_segment_heading),
        "terminal_segment_error": _serialize_angle_pair(terminal_segment_error),
        "terminal_pose_heading": _serialize_angle_pair(terminal_pose_yaw),
        "terminal_pose_error": _serialize_angle_pair(terminal_pose_yaw_error),
        "tolerance": _serialize_angle_pair(tolerance_rad),
    }


def _world_to_costmap_cell(world_x, world_y):
    return (
        int((world_x - DEFAULT_ORIGIN_X) / DEFAULT_RESOLUTION),
        int((world_y - DEFAULT_ORIGIN_Y) / DEFAULT_RESOLUTION),
    )


def _costmap_cell_in_bounds(grid, mx, my):
    size_y, size_x = grid.shape
    return 0 <= mx < size_x and 0 <= my < size_y


def _build_astar_endpoint_payload(endpoint, world_x, world_y, mx, my):
    return {
        "endpoint": endpoint,
        "world_x": _serialize_validation_scalar(world_x),
        "world_y": _serialize_validation_scalar(world_y),
        "mx": int(mx),
        "my": int(my),
    }


def _diagnose_astar_endpoint(
    grid,
    esdf_grid,
    footprint_model,
    endpoint,
    world_x,
    world_y,
    yaw,
):
    mx, my = _world_to_costmap_cell(world_x, world_y)
    endpoint_payload = _build_astar_endpoint_payload(endpoint, world_x, world_y, mx, my)

    if not _costmap_cell_in_bounds(grid, mx, my):
        return {
            "reason": f"{endpoint}_out_of_bounds",
            "message": f"A* could not find a path because the {endpoint} pose lies outside the costmap bounds.",
            endpoint: endpoint_payload,
        }

    endpoint_payload["cell_cost"] = int(grid[my, mx])
    if endpoint_payload["cell_cost"] >= int(pcs.Costmap2D.LETHAL_OBSTACLE):
        return {
            "reason": f"{endpoint}_in_lethal_obstacle",
            "message": f"A* could not find a path because the {endpoint} pose lies inside a lethal obstacle cell.",
            endpoint: endpoint_payload,
        }

    radius = max(float(footprint_model["check_radius"]), 0.0)
    planner_points = footprint_model["planner_points"]
    if not planner_points or radius <= 1e-9:
        return None

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    min_clearance = math.inf
    min_clearance_sample = None
    for offset in range(0, len(planner_points), 2):
        local_x = float(planner_points[offset])
        local_y = float(planner_points[offset + 1])
        checkpoint_world_x = world_x + cos_yaw * local_x - sin_yaw * local_y
        checkpoint_world_y = world_y + sin_yaw * local_x + cos_yaw * local_y
        checkpoint_mx, checkpoint_my = _world_to_costmap_cell(checkpoint_world_x, checkpoint_world_y)

        if not _costmap_cell_in_bounds(grid, checkpoint_mx, checkpoint_my):
            return {
                "reason": f"{endpoint}_footprint_out_of_bounds",
                "message": f"A* could not find a path because the {endpoint} footprint leaves the costmap bounds.",
                endpoint: endpoint_payload,
                "checkpoint": {
                    "index": offset // 2,
                    "local_x": _serialize_validation_scalar(local_x),
                    "local_y": _serialize_validation_scalar(local_y),
                    "world_x": _serialize_validation_scalar(checkpoint_world_x),
                    "world_y": _serialize_validation_scalar(checkpoint_world_y),
                    "mx": int(checkpoint_mx),
                    "my": int(checkpoint_my),
                },
            }

        if esdf_grid is None:
            continue
        clearance = float(esdf_grid[checkpoint_my, checkpoint_mx])
        if clearance < min_clearance:
            min_clearance = clearance
            min_clearance_sample = {
                "mx": int(checkpoint_mx),
                "my": int(checkpoint_my),
                "world_x": _serialize_validation_scalar(checkpoint_world_x),
                "world_y": _serialize_validation_scalar(checkpoint_world_y),
            }

    if esdf_grid is not None and min_clearance < radius:
        return {
            "reason": f"{endpoint}_footprint_collision",
            "message": (
                f"A* could not find a path because the {endpoint} footprint is in collision or too close "
                f"to obstacles (clearance {min_clearance:.3f} m, required {radius:.3f} m)."
            ),
            endpoint: endpoint_payload,
            "required_clearance_m": _serialize_validation_scalar(radius),
            "clearance_m": _serialize_validation_scalar(min_clearance),
            "closest_observed_cell": min_clearance_sample,
        }

    return None


def _diagnose_astar_no_path(
    grid,
    esdf_grid,
    footprint_model,
    start_x,
    start_y,
    goal_x,
    goal_y,
):
    nominal_yaw = math.atan2(goal_y - start_y, goal_x - start_x)

    start_failure = _diagnose_astar_endpoint(
        grid,
        esdf_grid,
        footprint_model,
        "start",
        start_x,
        start_y,
        nominal_yaw,
    )
    if start_failure is not None:
        start_failure["goal"] = _build_astar_endpoint_payload(
            "goal",
            goal_x,
            goal_y,
            *_world_to_costmap_cell(goal_x, goal_y),
        )
        return start_failure

    goal_failure = _diagnose_astar_endpoint(
        grid,
        esdf_grid,
        footprint_model,
        "goal",
        goal_x,
        goal_y,
        nominal_yaw,
    )
    if goal_failure is not None:
        goal_failure["start"] = _build_astar_endpoint_payload(
            "start",
            start_x,
            start_y,
            *_world_to_costmap_cell(start_x, start_y),
        )
        return goal_failure

    return {
        "reason": "disconnected_free_space",
        "message": (
            "A* could not find a path because the start and goal are individually traversable, "
            "but no connected corridor satisfies the current obstacle layout and footprint clearance constraints."
        ),
        "start": _build_astar_endpoint_payload(
            "start",
            start_x,
            start_y,
            *_world_to_costmap_cell(start_x, start_y),
        ),
        "goal": _build_astar_endpoint_payload(
            "goal",
            goal_x,
            goal_y,
            *_world_to_costmap_cell(goal_x, goal_y),
        ),
        "required_clearance_m": _serialize_validation_scalar(float(footprint_model["check_radius"])),
        "safe_distance_m": _serialize_validation_scalar(float(footprint_model["safe_distance"])),
    }


def _build_validation_error_payload(validation):
    return {
        "code": validation["error_code"],
        "message": validation["message"],
        "source": "post_validation",
        "details": {
            "failure_count": validation["failure_count"],
            "collision_count": validation["collision_count"],
            "out_of_bounds_count": validation["out_of_bounds_count"],
            "nonfinite_count": validation["nonfinite_count"],
            "first_failure": validation["first_failure"],
        },
    }


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


def _resolve_capsule_center_limit(half_length, radius, capsule_mode):
    if _normalize_capsule_mode(capsule_mode) == "exact":
        return max(half_length - radius, 0.0)
    return half_length


def _build_robot_footprint_model(
    footprint_mode,
    capsule_mode,
    surface_clearance_margin_m,
    point_robot_radius_m,
    robot_length_m,
    robot_width_m,
    capsule_sampling_tolerance_m=None,
):
    """Build the unified checkpoint + radius geometry used by planning and smoothing."""
    mode = footprint_mode if footprint_mode in {"point", "capsule"} else "capsule"
    normalized_capsule_mode = _normalize_capsule_mode(capsule_mode)
    sampling_tolerance = max(
        0.0,
        DEFAULT_CAPSULE_SAMPLING_TOLERANCE_M
        if capsule_sampling_tolerance_m is None else float(capsule_sampling_tolerance_m),
    )
    half_length = max(robot_length_m * 0.5, DEFAULT_RESOLUTION * 0.5)
    half_width = max(robot_width_m * 0.5, DEFAULT_RESOLUTION * 0.5)

    if mode == "point":
        check_radius = max(point_robot_radius_m, DEFAULT_RESOLUTION * 0.5)
        local_points = [(0.0, 0.0)]
    else:
        check_radius = half_width
        center_limit = _resolve_capsule_center_limit(half_length, check_radius, normalized_capsule_mode)
        local_points = [(offset_x, 0.0) for offset_x in _build_capsule_center_offsets(
            center_limit,
            check_radius,
            sampling_tolerance,
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

    safe_distance = surface_clearance_margin_m
    return {
        "mode": mode,
        "capsule_mode": normalized_capsule_mode if mode == "capsule" else None,
        "capsule_sampling_tolerance_m": sampling_tolerance,
        "safe_distance": safe_distance,
        "check_radius": check_radius,
        "planner_points": planner_points,
        "smoother_points": smoother_points,
        "serialized_points": serialized_points,
        "robot_length_m": robot_length_m,
        "robot_width_m": robot_width_m,
    }


@dataclass(frozen=True)
class PlanRequestConfig:
    """Normalized frontend inputs for a single `/api/plan` request."""

    manual_reference_path: list | None
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float
    start_yaw_deg: float
    goal_yaw_deg: float
    keep_start_orientation: bool
    keep_goal_orientation: bool
    goal_longitudinal_tolerance_m: float
    goal_lateral_tolerance_m: float
    goal_orientation_tolerance_deg: float
    footprint_mode: str
    capsule_mode: str
    capsule_sampling_tolerance_m: float
    surface_clearance_margin_m: float
    point_robot_radius_m: float
    robot_length_m: float
    robot_width_m: float
    model_weight: float
    costmap_weight: float
    cusp_costmap_weight: float
    cusp_zone_length: float
    reference_path_weight: float
    enable_reference_point_max_deviation: bool
    reference_point_deviation_limit_m: float
    kinematic_curvature_weight: float
    kinematic_curvature_rate_weight: float
    kinematic_spacing_weight: float
    kinematic_max_spacing_m: float
    path_length_weight: float
    max_curvature: float
    max_time: float
    reference_spacing_target_m: float
    path_downsample: int
    path_upsample: int
    max_iterations: int
    optimizer_type: str
    linear_solver_type: str
    parameter_tolerance: float
    function_tolerance: float
    gradient_tolerance: float
    optimizer_debug: bool
    planner_penalty_weight: float

    @classmethod
    def from_payload(cls, req):
        """Parse request values once so later stages can read a stable config."""
        footprint_mode = str(req.get("footprint_mode", "capsule")).strip().lower()
        if footprint_mode not in {"point", "capsule"}:
            footprint_mode = "capsule"
        capsule_mode = _normalize_capsule_mode(req.get("capsule_mode", "conservative"))

        linear_solver_type = str(req.get("linear_solver_type", "SPARSE_NORMAL_CHOLESKY")).strip().upper()
        if linear_solver_type not in {"DENSE_QR", "SPARSE_NORMAL_CHOLESKY"}:
            linear_solver_type = "SPARSE_NORMAL_CHOLESKY"

        costmap_weight = float(req.get("costmap_weight", 1.0))

        return cls(
            manual_reference_path=_parse_manual_reference_path(req.get("manual_reference_path")),
            start_x=float(req.get("start_x", 1.0)),
            start_y=float(req.get("start_y", 1.0)),
            goal_x=float(req.get("goal_x", 18.0)),
            goal_y=float(req.get("goal_y", 18.0)),
            start_yaw_deg=float(req.get("start_yaw_deg", 45.0)),
            goal_yaw_deg=float(req.get("goal_yaw_deg", 45.0)),
            keep_start_orientation=_coerce_bool(req.get("keep_start_orientation"), True),
            keep_goal_orientation=_coerce_bool(req.get("keep_goal_orientation"), True),
            goal_longitudinal_tolerance_m=max(0.0, float(req.get("goal_longitudinal_tolerance_m", 0.0))),
            goal_lateral_tolerance_m=max(0.0, float(req.get("goal_lateral_tolerance_m", 0.0))),
            goal_orientation_tolerance_deg=max(0.0, float(req.get("goal_orientation_tolerance_deg", 0.0))),
            footprint_mode=footprint_mode,
            capsule_mode=capsule_mode,
            capsule_sampling_tolerance_m=max(
                0.0,
                float(req.get("capsule_sampling_tolerance_m", DEFAULT_CAPSULE_SAMPLING_TOLERANCE_M)),
            ),
            surface_clearance_margin_m=max(
                0.05,
                float(req.get("surface_clearance_margin_m", req.get("hinge_loss_threshold_m", 0.5))),
            ),
            point_robot_radius_m=max(0.0, float(req.get("point_robot_radius_m", 1.0))),
            robot_length_m=max(DEFAULT_RESOLUTION, float(req.get("robot_length_m", 0.8))),
            robot_width_m=max(DEFAULT_RESOLUTION, float(req.get("robot_width_m", 0.5))),
            model_weight=float(req.get("model_weight", 20.0)),
            costmap_weight=costmap_weight,
            cusp_costmap_weight=max(0.0, float(req.get("cusp_costmap_weight", costmap_weight * 3.0))),
            cusp_zone_length=max(0.0, float(req.get("cusp_zone_length", 2.5))),
            reference_path_weight=float(req.get("reference_path_weight", 0.0)),
            enable_reference_point_max_deviation=_coerce_bool(
                req.get("enable_reference_point_max_deviation"),
                False,
            ),
            reference_point_deviation_limit_m=max(
                0.0,
                float(req.get("reference_point_max_deviation_m", 0.25)),
            ),
            kinematic_curvature_weight=float(req.get("kinematic_curvature_weight", 1.0)),
            kinematic_curvature_rate_weight=float(req.get("kinematic_curvature_rate_weight", 5.0)),
            kinematic_spacing_weight=max(0.0, float(req.get("kinematic_spacing_weight", 1.0))),
            kinematic_max_spacing_m=max(0.0, float(req.get("kinematic_max_spacing_m", 0.0))),
            path_length_weight=float(req.get("path_length_weight", 0.1)),
            max_curvature=float(req.get("max_curvature", 2.5)),
            max_time=max(0.01, float(req.get("max_time", 10.0))),
            reference_spacing_target_m=min(
                2.0,
                max(
                    DEFAULT_RESOLUTION,
                    float(req.get("reference_spacing_target_m", DEFAULT_REFERENCE_SPACING_TARGET_M)),
                ),
            ),
            path_downsample=max(1, int(req.get("path_downsampling_factor", 1))),
            path_upsample=max(1, int(req.get("path_upsampling_factor", 1))),
            max_iterations=max(1, int(req.get("max_iterations", 50))),
            optimizer_type="kinematic_smoother",
            linear_solver_type=linear_solver_type,
            parameter_tolerance=max(0.0, float(req.get("param_tol", 1e-8))),
            function_tolerance=max(0.0, float(req.get("fn_tol", 1e-6))),
            gradient_tolerance=max(0.0, float(req.get("gradient_tol", 1e-10))),
            optimizer_debug=_coerce_bool(req.get("optimizer_debug"), False),
            planner_penalty_weight=max(0.0, float(req.get("planner_penalty_weight", 1.0))),
        )

    @property
    def start_yaw_rad(self):
        return math.radians(self.start_yaw_deg)

    @property
    def goal_yaw_rad(self):
        return math.radians(self.goal_yaw_deg)

    @property
    def reference_point_max_deviation_m(self):
        if not self.enable_reference_point_max_deviation:
            return 0.0
        return self.reference_point_deviation_limit_m

    def heading_vectors(self):
        """Return the unit direction vectors expected by the pybind smoother API."""
        return [math.cos(self.start_yaw_rad), math.sin(self.start_yaw_rad)], [
            math.cos(self.goal_yaw_rad),
            math.sin(self.goal_yaw_rad),
        ]

    def build_footprint_model(self):
        return _build_robot_footprint_model(
            self.footprint_mode,
            self.capsule_mode,
            self.surface_clearance_margin_m,
            self.point_robot_radius_m,
            self.robot_length_m,
            self.robot_width_m,
            capsule_sampling_tolerance_m=self.capsule_sampling_tolerance_m,
        )

    def build_smoother_params(self, footprint_model):
        """Translate request-level tuning knobs into native smoother params."""
        smoother_params = pcs.SmootherParams()
        smoother_params.model_weight_sqrt = math.sqrt(self.model_weight)
        smoother_params.costmap_weight_sqrt = math.sqrt(self.costmap_weight)
        smoother_params.cusp_costmap_weight_sqrt = math.sqrt(self.cusp_costmap_weight)
        smoother_params.cusp_zone_length = self.cusp_zone_length
        smoother_params.obstacle_safe_distance = footprint_model["safe_distance"]
        smoother_params.cost_check_radius = footprint_model["check_radius"]
        smoother_params.reference_path_weight_sqrt = math.sqrt(self.reference_path_weight)
        smoother_params.reference_point_max_deviation_m = self.reference_point_max_deviation_m
        smoother_params.kinematic_curvature_weight_sqrt = math.sqrt(self.kinematic_curvature_weight)
        smoother_params.kinematic_curvature_rate_weight_sqrt = math.sqrt(
            self.kinematic_curvature_rate_weight
        )
        smoother_params.kinematic_spacing_weight_sqrt = math.sqrt(self.kinematic_spacing_weight)
        smoother_params.kinematic_max_spacing = self.kinematic_max_spacing_m
        smoother_params.path_length_weight_sqrt = math.sqrt(self.path_length_weight)
        smoother_params.max_curvature = self.max_curvature
        smoother_params.max_time = self.max_time
        smoother_params.keep_start_orientation = self.keep_start_orientation
        smoother_params.keep_goal_orientation = self.keep_goal_orientation
        smoother_params.goal_longitudinal_tolerance = self.goal_longitudinal_tolerance_m
        smoother_params.goal_lateral_tolerance = self.goal_lateral_tolerance_m
        smoother_params.goal_orientation_tolerance = math.radians(self.goal_orientation_tolerance_deg)
        smoother_params.cost_check_points = footprint_model["smoother_points"]
        smoother_params.path_downsampling_factor = self.path_downsample
        smoother_params.path_upsampling_factor = self.path_upsample
        return smoother_params

    def build_optimizer_params(self):
        optimizer_params = pcs.OptimizerParams()
        optimizer_params.debug = self.optimizer_debug
        optimizer_params.linear_solver_type = self.linear_solver_type
        optimizer_params.max_iterations = self.max_iterations
        optimizer_params.parameter_tolerance = self.parameter_tolerance
        optimizer_params.function_tolerance = self.function_tolerance
        optimizer_params.gradient_tolerance = self.gradient_tolerance
        return optimizer_params


def _evaluate_oriented_rectangle_pose(grid, center_x, center_y, yaw, length_m, width_m, index):
    """Classify a final rectangle-validation result for a single pose."""
    pose_payload = {
        "index": int(index),
        "x": _serialize_validation_scalar(center_x),
        "y": _serialize_validation_scalar(center_y),
        "yaw": _serialize_validation_scalar(yaw),
    }
    if not all(math.isfinite(float(value)) for value in (center_x, center_y, yaw)):
        return {
            "valid": False,
            "code": ERROR_FINAL_PATH_NONFINITE,
            "message": f"Final validation failed because pose {index} contains non-finite x/y/yaw values.",
            "reason": "nonfinite_pose",
            "pose": pose_payload,
        }

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
        return {
            "valid": False,
            "code": ERROR_FINAL_PATH_OUT_OF_BOUNDS,
            "message": f"Final validation failed because pose {index} leaves the map bounds.",
            "reason": "out_of_bounds",
            "pose": pose_payload,
            "bounding_box_cells": {
                "min_mx": min_mx,
                "max_mx": max_mx,
                "min_my": min_my,
                "max_my": max_my,
            },
        }

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
                return {
                    "valid": False,
                    "code": ERROR_FINAL_PATH_COLLISION,
                    "message": f"Final validation failed because pose {index} overlaps a lethal obstacle cell.",
                    "reason": "lethal_overlap",
                    "pose": pose_payload,
                    "collision_cell": {
                        "mx": int(mx),
                        "my": int(my),
                        "world_x": _serialize_validation_scalar(cell_x),
                        "world_y": _serialize_validation_scalar(cell_y),
                    },
                }

    return {
        "valid": True,
        "pose": pose_payload,
    }


def _validate_smoothed_path_rectangles(grid, xs, ys, thetas, robot_length_m, robot_width_m):
    """Final collision validation using the actual rectangular footprint."""
    colliding_indices = []
    out_of_bounds_indices = []
    nonfinite_indices = []
    failures = []
    pose_count = 0
    for index, (world_x, world_y, theta) in enumerate(zip(xs, ys, thetas)):
        pose_count += 1
        validation = _evaluate_oriented_rectangle_pose(
            grid,
            world_x,
            world_y,
            theta,
            robot_length_m,
            robot_width_m,
            index,
        )
        if not validation["valid"]:
            failures.append(validation)
            if validation["code"] == ERROR_FINAL_PATH_COLLISION:
                colliding_indices.append(index)
            elif validation["code"] == ERROR_FINAL_PATH_OUT_OF_BOUNDS:
                out_of_bounds_indices.append(index)
            elif validation["code"] == ERROR_FINAL_PATH_NONFINITE:
                nonfinite_indices.append(index)

    first_failure = failures[0] if failures else None
    failure_count = len(failures)
    if first_failure is None:
        message = f"Rectangle validation passed on all {pose_count} pose(s)."
        error_code = None
    else:
        message = first_failure["message"]
        error_code = first_failure["code"]

    return {
        "collision_free": failure_count == 0,
        "valid": failure_count == 0,
        "message": message,
        "error_code": error_code,
        "failure_count": failure_count,
        "collision_count": len(colliding_indices),
        "colliding_indices": colliding_indices[:20],
        "out_of_bounds_count": len(out_of_bounds_indices),
        "out_of_bounds_indices": out_of_bounds_indices[:20],
        "nonfinite_count": len(nonfinite_indices),
        "nonfinite_indices": nonfinite_indices[:20],
        "first_failure": first_failure,
    }


def _validate_path_rectangles(grid, path, robot_length_m, robot_width_m):
    """Wrapper that validates a path expressed as (x, y, yaw) tuples."""
    xs, ys, thetas = _split_path_xyz(path)
    return _validate_smoothed_path_rectangles(
        grid,
        xs,
        ys,
        thetas,
        robot_length_m,
        robot_width_m,
    )


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
    global CURRENT_OBSTACLE_RECTS, COSTMAP_GRID, COSTMAP_METADATA, ESDF_GRID

    CURRENT_OBSTACLE_RECTS = [tuple(rect) for rect in obstacle_rects]
    COSTMAP_GRID = _build_costmap(CURRENT_OBSTACLE_RECTS)
    COSTMAP_METADATA = _summarize_costmap(COSTMAP_GRID, CURRENT_OBSTACLE_RECTS)
    ESDF_GRID = _compute_esdf_grid(_grid_to_pcs_costmap(COSTMAP_GRID))


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


def _run_planner_stage(config, costmap_grid, esdf_grid, planner_costmap, footprint_model):
    """Build the reference path stage using either manual input or A*."""
    start_yaw_rad = config.start_yaw_rad
    goal_yaw_rad = config.goal_yaw_rad
    start_dir, end_dir = config.heading_vectors()

    if config.manual_reference_path is not None:
        planner_stage_result = _run_manual_reference_stage(
            config.manual_reference_path,
            start_yaw_rad,
            goal_yaw_rad,
            config.keep_start_orientation,
            config.keep_goal_orientation,
        )
    else:
        planner_stage_result = _run_astar_stage(
            costmap_grid,
            esdf_grid,
            planner_costmap,
            footprint_model,
            config.planner_penalty_weight,
            config.start_x,
            config.start_y,
            config.goal_x,
            config.goal_y,
            config.reference_spacing_target_m,
            start_yaw_rad,
            goal_yaw_rad,
            config.keep_start_orientation,
            config.keep_goal_orientation,
        )

    return {
        "planner_stage_result": planner_stage_result,
        "start_yaw_rad": start_yaw_rad,
        "goal_yaw_rad": goal_yaw_rad,
        "start_dir": start_dir,
        "end_dir": end_dir,
    }


def _run_smoother_stage(
    config,
    planner_stage_result,
    planner_costmap,
    smoother_params,
    optimizer_params,
    start_dir,
    end_dir,
):
    """Run the kinematic smoother and normalize the fallback metadata."""
    optimizer_label = "Kinematic Smoother"
    smoother = pcs.KinematicSmoother()
    smoother.initialize(optimizer_params)

    smooth_t0 = time.time()
    smooth_message = ""
    smooth_error = None
    candidate_smoothed = None

    try:
        if planner_stage_result["planner"] is None:
            smooth_result = smoother.try_smooth(
                planner_stage_result["eigen_path"],
                start_dir,
                end_dir,
                planner_costmap,
                smoother_params,
            )
        else:
            smooth_result = smoother.try_smooth_with_planner_esdf(
                planner_stage_result["eigen_path"],
                start_dir,
                end_dir,
                planner_costmap,
                smoother_params,
                planner_stage_result["planner"],
            )

        smooth_time = (time.time() - smooth_t0) * 1000.0
        if bool(smooth_result["ok"]):
            candidate_smoothed = smooth_result["path"]
        elif _failure_has_displayable_candidate_path(smooth_result):
            candidate_smoothed = smooth_result["path"]

        smooth_success = bool(smooth_result["ok"])
        if smooth_success:
            smoother_stage = _make_pipeline_stage(
                "smoother",
                optimizer_label,
                "ok",
                f"Optimizer produced a smoothed candidate with {len(candidate_smoothed)} pose(s).",
                elapsed_ms=smooth_time,
                path_key="smoothed_candidate",
            )
        else:
            smooth_error = _build_smoother_error_payload(smooth_result)
            smooth_message = smooth_error["message"]
            smoother_stage = _make_pipeline_stage(
                "smoother",
                optimizer_label,
                "fallback",
                smooth_message or "Optimizer failed; using the reference path.",
                elapsed_ms=smooth_time,
                error_code=smooth_error.get("code"),
                path_key="smoothed_candidate" if candidate_smoothed is not None else "reference_fallback",
            )
    except Exception as exc:
        smooth_time = (time.time() - smooth_t0) * 1000.0
        smooth_success = False
        smooth_error = _error_payload(exc, default_status=422, default_source="smoother")
        smooth_message = smooth_error["message"]
        smoother_stage = _make_pipeline_stage(
            "smoother",
            optimizer_label,
            "fallback",
            smooth_message or "Optimizer raised an exception; using the reference path.",
            elapsed_ms=smooth_time,
            error_code=smooth_error.get("code"),
            path_key="reference_fallback",
        )

    return {
        "optimizer_label": optimizer_label,
        "smooth_time_ms": smooth_time,
        "smooth_success": smooth_success,
        "smooth_message": smooth_message,
        "smooth_error": smooth_error,
        "candidate_smoothed": candidate_smoothed,
        "smoother_stage": smoother_stage,
        "optimized_knot_count": int(smoother.get_last_optimized_knot_count()),
    }


def _run_validation_stage(
    config,
    costmap_grid,
    reference_with_yaw,
    candidate_smoothed,
    smooth_success,
    smooth_error,
    smooth_message,
):
    """Validate candidate/fallback paths and produce frontend pipeline stages."""
    candidate_rectangle_validation = None
    smoothed = candidate_smoothed if candidate_smoothed is not None else reference_with_yaw

    if candidate_smoothed is not None:
        candidate_rectangle_validation = _validate_path_rectangles(
            costmap_grid,
            candidate_smoothed,
            config.robot_length_m,
            config.robot_width_m,
        )
        candidate_rectangle_validation["validated_path"] = "smoothed_candidate"
        if not candidate_rectangle_validation["valid"]:
            smooth_success = False
            smooth_error = _build_validation_error_payload(candidate_rectangle_validation)
            smooth_message = smooth_error["message"]

    final_rectangle_validation = _validate_path_rectangles(
        costmap_grid,
        smoothed,
        config.robot_length_m,
        config.robot_width_m,
    )
    final_rectangle_validation["validated_path"] = (
        "smoothed_path" if smoothed is candidate_smoothed else "reference_fallback"
    )

    if not final_rectangle_validation["valid"]:
        validation_stage = _make_pipeline_stage(
            "validate",
            "Rectangle Validate",
            "error",
            final_rectangle_validation["message"],
            error_code=final_rectangle_validation["error_code"],
            path_key=final_rectangle_validation["validated_path"],
        )
        response_stage = _make_pipeline_stage(
            "web",
            "Web",
            "error",
            "Returned path failed final rectangle validation on the web.",
            error_code=final_rectangle_validation["error_code"],
            path_key=final_rectangle_validation["validated_path"],
        )
    elif candidate_rectangle_validation and not candidate_rectangle_validation["valid"]:
        validation_stage = _make_pipeline_stage(
            "validate",
            "Rectangle Validate",
            "error",
            candidate_rectangle_validation["message"],
            error_code=candidate_rectangle_validation["error_code"],
            path_key="smoothed_path",
        )
        response_stage = _make_pipeline_stage(
            "web",
            "Web",
            "error",
            "Showing the smoothed candidate on the web even though candidate validation failed.",
            error_code=(smooth_error or {}).get("code"),
            path_key="smoothed_path",
        )
    else:
        validation_stage = _make_pipeline_stage(
            "validate",
            "Rectangle Validate",
            "ok",
            final_rectangle_validation["message"],
            path_key=final_rectangle_validation["validated_path"],
        )
        response_stage = _make_pipeline_stage(
            "web",
            "Web",
            "ok" if smooth_success else "fallback",
            "Showing the smoothed path on the web."
            if smooth_success
            else "Showing the reference fallback path on the web.",
            error_code=None if smooth_success else (smooth_error or {}).get("code"),
            path_key=final_rectangle_validation["validated_path"],
        )

    return {
        "smoothed_path": smoothed,
        "smooth_success": smooth_success,
        "smooth_message": smooth_message,
        "smooth_error": smooth_error,
        "candidate_rectangle_validation": candidate_rectangle_validation,
        "final_rectangle_validation": final_rectangle_validation,
        "validation_stage": validation_stage,
        "response_stage": response_stage,
    }


def _build_plan_response_payload(
    config,
    footprint_model,
    raw_path,
    sparse_path,
    smoothed,
    pipeline,
    optimizer_label,
    optimized_knot_count,
    astar_time,
    smooth_time,
    smooth_success,
    smooth_message,
    smooth_error,
    candidate_rectangle_validation,
    goal_orientation_diagnostics,
    final_rectangle_validation,
):
    """Build the stable JSON payload returned by /api/plan."""
    optimizer_config = {
        "optimizer_type": config.optimizer_type,
        "model_weight": round(config.model_weight, 3),
        "costmap_weight": round(config.costmap_weight, 3),
        "cusp_costmap_weight": round(config.cusp_costmap_weight, 3),
        "cusp_zone_length_m": round(config.cusp_zone_length, 3),
        "reference_path_weight": round(config.reference_path_weight, 3),
        "reference_point_max_deviation_m": round(config.reference_point_max_deviation_m, 3),
        "kinematic_curvature_weight": round(config.kinematic_curvature_weight, 3),
        "kinematic_curvature_rate_weight": round(config.kinematic_curvature_rate_weight, 3),
        "kinematic_spacing_weight": round(config.kinematic_spacing_weight, 3),
        "kinematic_max_spacing_m": round(config.kinematic_max_spacing_m, 3),
        "path_length_weight": round(config.path_length_weight, 3),
        "max_curvature": round(config.max_curvature, 4),
        "max_time_s": round(config.max_time, 3),
        "max_iterations": int(config.max_iterations),
        "parameter_tolerance": config.parameter_tolerance,
        "function_tolerance": config.function_tolerance,
        "gradient_tolerance": config.gradient_tolerance,
        "path_downsampling_factor": int(config.path_downsample),
        "path_upsampling_factor": int(config.path_upsample),
        "keep_start_orientation": bool(config.keep_start_orientation),
        "keep_goal_orientation": bool(config.keep_goal_orientation),
    }

    astar_x = [point[0] for point in raw_path]
    astar_y = [point[1] for point in raw_path]
    ref_x = [point[0] for point in sparse_path]
    ref_y = [point[1] for point in sparse_path]
    opt_x, opt_y, opt_theta = _split_path_xyz(smoothed)
    raw_length = _path_length(raw_path)
    ref_length = _path_length(sparse_path)
    opt_length = _path_length(smoothed)

    return {
        # High-level pipeline and fallback status.
        "success": True,
        "pipeline": pipeline,
        "smooth_success": smooth_success,
        "astar_time_ms": round(astar_time, 2),
        "smooth_time_ms": round(smooth_time, 2),
        "smooth_message": smooth_message,
        "smooth_error": smooth_error,
        "candidate_rectangle_validation": candidate_rectangle_validation,
        "goal_orientation_diagnostics": goal_orientation_diagnostics,
        "optimizer_config": optimizer_config,

        # Raw planner / reference / returned path geometry.
        "astar_x": astar_x,
        "astar_y": astar_y,
        "ref_x": ref_x,
        "ref_y": ref_y,
        "opt_x": opt_x,
        "opt_y": opt_y,
        "opt_theta": opt_theta,

        # Path cardinality and length metrics.
        "num_astar_pts": len(raw_path),
        "num_ref_pts": len(sparse_path),
        "num_opt_knots": optimized_knot_count,
        "num_opt_pts": len(smoothed),
        "num_returned_pts": len(smoothed),
        "raw_path_length_m": round(raw_length, 3),
        "ref_path_length_m": round(ref_length, 3),
        "opt_path_length_m": round(opt_length, 3),
        "opt_vs_ref_delta_m": round(opt_length - ref_length, 3),

        # Planner / optimizer configuration echoed back to the frontend.
        "reference_spacing_target_m": round(config.reference_spacing_target_m, 3),
        "planner_penalty_weight": round(config.planner_penalty_weight, 3),
        "optimizer_type": config.optimizer_type,
        "optimizer_label": optimizer_label,
        "start_yaw_deg": round(config.start_yaw_deg, 2),
        "goal_yaw_deg": round(config.goal_yaw_deg, 2),
        "keep_start_orientation": config.keep_start_orientation,
        "keep_goal_orientation": config.keep_goal_orientation,
        "hinge_loss_threshold_m": round(config.surface_clearance_margin_m, 3),
        "surface_clearance_margin_m": round(config.surface_clearance_margin_m, 3),
        "point_robot_radius_m": round(config.point_robot_radius_m, 3),
        "effective_safe_distance_m": round(footprint_model["safe_distance"], 3),
        "footprint_mode": footprint_model["mode"],
        "footprint_capsule_mode": footprint_model["capsule_mode"],
        "capsule_sampling_tolerance_m": round(footprint_model["capsule_sampling_tolerance_m"], 3),
        "robot_length_m": round(config.robot_length_m, 3),
        "robot_width_m": round(config.robot_width_m, 3),
        "robot_check_points": len(footprint_model["serialized_points"]),
        "collision_check_radius_m": round(footprint_model["check_radius"], 3),
        "collision_check_points_local": footprint_model["serialized_points"],

        # Final post-validation status for the path actually returned.
        "final_rectangle_validation": final_rectangle_validation,
    }


with STATE_LOCK:
    _rebuild_costmap_state(DEFAULT_OBSTACLE_RECTS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Return an empty favicon response so browser console logs stay clean."""
    return Response(status=204)


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
            raise ApiError(
                ERROR_INVALID_REQUEST,
                "No obstacle rectangles were provided.",
                status_code=400,
                source="request",
            )

        normalized_rects = _normalize_obstacle_rects(rect_payloads)
        with STATE_LOCK:
            _rebuild_costmap_state(normalized_rects)
            payload = _serialize_costmap_state()

        payload["success"] = True
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        return _error_response(e, default_status=400, default_source="costmap")


@app.route("/api/plan", methods=["POST"])
def plan_and_smooth():
    """Run A* to find a reference path, then smooth it with the kinematic smoother."""
    try:
        config = PlanRequestConfig.from_payload(request.get_json(silent=True) or {})
        footprint_model = config.build_footprint_model()

        with STATE_LOCK:
            costmap_grid = COSTMAP_GRID.copy()
            esdf_grid = ESDF_GRID.copy() if ESDF_GRID is not None else None
            planner_costmap = _grid_to_pcs_costmap(costmap_grid)

        planner_stage = _run_planner_stage(
            config,
            costmap_grid,
            esdf_grid,
            planner_costmap,
            footprint_model,
        )
        planner_stage_result = planner_stage["planner_stage_result"]
        raw_path = planner_stage_result["raw_path"]
        sparse_path = planner_stage_result["sparse_path"]
        reference_with_yaw = planner_stage_result["reference_with_yaw"]
        astar_time = planner_stage_result["astar_time_ms"]

        smoother_params = config.build_smoother_params(footprint_model)
        opt_params = config.build_optimizer_params()

        smoother_stage = _run_smoother_stage(
            config,
            planner_stage_result,
            planner_costmap,
            smoother_params,
            opt_params,
            planner_stage["start_dir"],
            planner_stage["end_dir"],
        )

        validation_stage = _run_validation_stage(
            config,
            costmap_grid,
            reference_with_yaw,
            smoother_stage["candidate_smoothed"],
            smoother_stage["smooth_success"],
            smoother_stage["smooth_error"],
            smoother_stage["smooth_message"],
        )

        optimizer_label = smoother_stage["optimizer_label"]
        smooth_time = smoother_stage["smooth_time_ms"]
        optimized_knot_count = smoother_stage["optimized_knot_count"]
        smoothed = validation_stage["smoothed_path"]
        smooth_success = validation_stage["smooth_success"]
        smooth_message = validation_stage["smooth_message"]
        smooth_error = validation_stage["smooth_error"]
        candidate_rectangle_validation = validation_stage["candidate_rectangle_validation"]
        final_rectangle_validation = validation_stage["final_rectangle_validation"]

        pipeline = _build_pipeline_payload([
            planner_stage_result["stage"],
            smoother_stage["smoother_stage"],
            validation_stage["validation_stage"],
            validation_stage["response_stage"],
        ])
        goal_orientation_diagnostics = _build_goal_orientation_diagnostics(
            smoothed,
            planner_stage["goal_yaw_rad"],
            KINEMATIC_GOAL_ORIENTATION_TOLERANCE_RAD,
        )
        if (
            smooth_error
            and smooth_error.get("details", {}).get("failure_reason") == "goal_orientation_constraint"
            and goal_orientation_diagnostics is not None
        ):
            smooth_error.setdefault("details", {}).update({
                "goal_orientation": goal_orientation_diagnostics,
            })

        return jsonify(_build_plan_response_payload(
            config,
            footprint_model,
            raw_path,
            sparse_path,
            smoothed,
            pipeline,
            optimizer_label,
            optimized_knot_count,
            astar_time,
            smooth_time,
            smooth_success,
            smooth_message,
            smooth_error,
            candidate_rectangle_validation,
            goal_orientation_diagnostics,
            final_rectangle_validation,
        ))

    except Exception as e:
        traceback.print_exc()
        return _error_response(e, default_status=400, default_source="planner")


if __name__ == "__main__":
    app.run(
        host=os.environ.get("CS_WEBAPP_HOST", "127.0.0.1"),
        port=int(os.environ.get("CS_WEBAPP_PORT", "5002")),
        debug=_env_flag("CS_WEBAPP_DEBUG", True),
        use_reloader=_env_flag("CS_WEBAPP_RELOADER", False),
    )
