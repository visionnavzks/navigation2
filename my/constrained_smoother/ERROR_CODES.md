# Error Codes

This file is the reference catalog for the standalone constrained smoother project's stable error codes.

## Design Rules

- Codes are stable identifiers; callers should branch on `code`, not on free-form `message` text.
- Messages may be refined over time for clarity.
- High-level APIs should prefer structured return payloads over uncaught native exceptions.

## Core Smoother Codes

| Code | Layer | Meaning | Typical Trigger | Recommended Handling |
| --- | --- | --- | --- | --- |
| `CS_INVALID_PATH` | C++, pybind `try_*` | Input path is too short or malformed for smoothing. | Fewer than 2 knots passed to smoother backends. | Validate path length before smoothing; keep the reference path. |
| `CS_SMOOTHING_FAILED` | C++, pybind `try_*`, web `smooth_error` | Optimizer ran but did not produce a usable solution. | Non-usable solution, no cost decrease, or backend convergence failure. | Fall back to the reference path and surface the failure to the operator. |
| `CS_INVALID_COSTMAP` | C++, pybind `try_*` | Planner or smoother received no valid costmap. | Null or otherwise invalid costmap object. | Rebuild or reinitialize the costmap before retrying. |
| `CS_PRECOMPUTED_ESDF_SIZE_MISMATCH` | C++, pybind `try_*` | The supplied ESDF does not match costmap dimensions. | Reusing planner ESDF with mismatched map dimensions. | Discard cached ESDF and recompute from the active map. |

## Web API Codes

| Code | Endpoint | Meaning | Typical Trigger | Recommended Handling |
| --- | --- | --- | --- | --- |
| `CS_INVALID_REQUEST` | `/api/obstacles`, `/api/plan` | Request payload failed validation. | Missing obstacle list, malformed numeric values, invalid shapes. | Fix request payload and retry. |
| `CS_ASTAR_NO_PATH` | `/api/plan` | A* could not find a feasible route. | Start/goal outside map, blocked start/goal, fully obstructed corridor. | Adjust endpoints or obstacle layout. |
| `CS_FINAL_PATH_NONFINITE` | `/api/plan` `smooth_error` | Final post-smoothing validation found non-finite pose values. | Smoothed candidate contains `NaN` or `Inf` in `x`, `y`, or `yaw`. | Reject the candidate path and inspect the optimizer output. |
| `CS_FINAL_PATH_OUT_OF_BOUNDS` | `/api/plan` `smooth_error` | Final post-smoothing validation found the robot footprint outside the map. | Smoothed candidate leaves the costmap extent once the full rectangle footprint is applied. | Reject the candidate path; reduce deformation or adjust constraints. |
| `CS_FINAL_PATH_COLLISION` | `/api/plan` `smooth_error` | Final post-smoothing validation found a footprint collision. | Smoothed candidate overlaps lethal cells after rectangle-footprint validation. | Reject the candidate path and fall back to the reference path. |
| `CS_INTERNAL_ERROR` | Any web endpoint | Unexpected server-side error. | Unhandled Python exception or runtime fault. | Inspect logs and server state before retrying. |

## Pure Python SciPy Helper Codes

These codes come from `include/constrained_smoother/kinematic_smoother.py`.

| Code | API | Meaning | Typical Trigger | Recommended Handling |
| --- | --- | --- | --- | --- |
| `CS_INVALID_RAW_PATH` | `try_optimize(...)` | `raw_path` shape is not `(N, 2)` or `(N, 3)`. | Passing a flat array, ragged data, or non-pose columns. | Normalize input to an `N x 2` or `N x 3` array. |
| `CS_EMPTY_RAW_PATH` | `try_optimize(...)` | No poses were supplied. | Empty input array or list. | Ensure at least one pose exists before calling the helper. |
| `CS_INVALID_GEAR_DIRECTIONS` | `try_optimize(...)` | `gear_directions` length does not match segment count. | Shape differs from `(N - 1,)`. | Recompute directions so there is one entry per segment. |
| `CS_KINEMATIC_OPTIMIZATION_FAILED` | `try_optimize(...)` | SciPy least-squares did not converge to a successful solution. | Iteration budget exhausted or model/constraints inconsistent. | Inspect `optimizer_result`, tune weights/iterations, or fall back to the reference path. |

## Safe Python API Summary

### pybind smoother wrappers

- `Smoother.try_smooth(...)`
- `Smoother.try_smooth_with_planner_esdf(...)`
- `KinematicSmoother.try_smooth(...)`
- `KinematicSmoother.try_smooth_with_planner_esdf(...)`

Return shape:

```python
{
    "ok": bool,
    "path": list | None,
    "error_code": str | None,
    "error_message": str | None,
}
```

### `/api/plan` smoother-validation fields

When smoothing runs, the web API may return these additional fields:

```python
{
    "smooth_success": bool,
    "smooth_error": {
        "code": str,
        "message": str,
        "source": "smoother" | "post_validation",
        "details": dict | None,
    } | None,
    "candidate_rectangle_validation": {
        "valid": bool,
        "error_code": str | None,
        "message": str,
        "first_failure": dict | None,
        "validated_path": "smoothed_candidate",
    } | None,
    "final_rectangle_validation": {
        "valid": bool,
        "error_code": str | None,
        "message": str,
        "validated_path": "smoothed_path" | "reference_fallback",
    },
}
```

If `candidate_rectangle_validation.valid` is `false`, the smoothed candidate was rejected after optimization and the response falls back to the reference path.

### pure Python SciPy helper

- `KinematicSmoother.try_optimize(...)`

Return shape:

```python
{
    "ok": bool,
    "states": np.ndarray | None,
    "optimizer_result": OptimizeResult | None,
    "error_code": str | None,
    "error_message": str | None,
}
```