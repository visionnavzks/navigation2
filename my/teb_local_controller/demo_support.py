from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from my.teb_local_controller.teb_mpc import (
    ArcSegment,
    LineSegment,
    ReferenceTrajectory,
    TEBMPCController,
    VehicleState,
    build_reference_trajectory,
)


DEMO_REFERENCE_DS = 0.25
DEMO_CRUISE_SPEED = 1.0


DEMO_REFERENCE_DEFAULTS: Dict[str, float] = {
    "ds": DEMO_REFERENCE_DS,
    "cruise_speed": DEMO_CRUISE_SPEED,
    "selection_length": 0.0,
    "near_terminal_s_tol": 0.0,
    "extra_points": 0,
    "line_1_length": 1.8,
    "arc_1_radius": 1.8,
    "arc_1_angle": math.pi / 4.0,
    "line_2_length": 1.1,
    "arc_2_radius": 1.5,
    "arc_2_angle": -math.pi / 6.0,
    "line_3_length": 0.9,
}


DEMO_SAMPLING_DEFAULTS: Dict[str, float] = {
    "x_offset_range": 1.5,
    "y_offset_range": 2.0,
    "theta_offset_range": 0.7,
    "speed_min": 0.1,
    "speed_max": 1.2,
    "accel_min": -0.5,
    "accel_max": 0.5,
    "kappa_offset_range": 0.08,
    "kappa_min": -0.2,
    "kappa_max": 0.2,
}


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _merged_reference_config(reference_config: Dict[str, float] | None = None) -> Dict[str, float]:
    return {**DEMO_REFERENCE_DEFAULTS, **(reference_config or {})}


def _merged_sampling_config(sampling_config: Dict[str, float] | None = None) -> Dict[str, float]:
    return {**DEMO_SAMPLING_DEFAULTS, **(sampling_config or {})}


def _resolve_extra_points(extra_points: float | int | None) -> int:
    if extra_points is None:
        return 0

    resolved_extra_points = float(extra_points)
    if not resolved_extra_points.is_integer():
        raise ValueError("extra_points must be an integer")

    resolved_extra_points_int = int(resolved_extra_points)
    return resolved_extra_points_int


def _resolve_selection_length(selection_length: float | None) -> float | None:
    if selection_length is None:
        return None

    resolved_selection_length = float(selection_length)
    if resolved_selection_length <= 0.0:
        return None
    return resolved_selection_length


def _resolve_near_terminal_s_tol(reference: ReferenceTrajectory, near_terminal_s_tol: float | None) -> float:
    if near_terminal_s_tol is not None:
        resolved_near_terminal_s_tol = float(near_terminal_s_tol)
        if resolved_near_terminal_s_tol > 0.0:
            return resolved_near_terminal_s_tol

    if reference.size < 2:
        return 0.0

    positive_spacings = np.diff(np.array(reference.s, dtype=float))
    positive_spacings = positive_spacings[positive_spacings > 1e-9]
    if positive_spacings.size == 0:
        return 0.0
    return float(np.median(positive_spacings))


def default_demo_segments(reference_config: Dict[str, float] | None = None) -> List[LineSegment | ArcSegment]:
    config = _merged_reference_config(reference_config)
    return [
        LineSegment(length=float(config["line_1_length"])),
        ArcSegment(radius=float(config["arc_1_radius"]), angle=float(config["arc_1_angle"])),
        LineSegment(length=float(config["line_2_length"])),
        ArcSegment(radius=float(config["arc_2_radius"]), angle=float(config["arc_2_angle"])),
        LineSegment(length=float(config["line_3_length"])),
    ]


def default_demo_reference(reference_config: Dict[str, float] | None = None) -> ReferenceTrajectory:
    config = _merged_reference_config(reference_config)
    dt_ref = config.get("dt_ref")
    return build_reference_trajectory(
        start=VehicleState(x=0.0, y=0.0, theta=0.0, v=0.6, a=0.0, kappa=0.0),
        segments=default_demo_segments(reference_config=config),
        ds=float(config["ds"]),
        cruise_speed=float(config["cruise_speed"]),
        dt_ref=float(dt_ref) if dt_ref is not None else None,
    )


def _reference_terminal_state(reference: ReferenceTrajectory) -> VehicleState:
    return VehicleState(
        x=float(reference.x[-1]),
        y=float(reference.y[-1]),
        theta=float(reference.theta[-1]),
        v=float(reference.v[-1]),
        a=float(reference.a[-1]),
        kappa=float(reference.kappa[-1]),
    )


def describe_demo_configuration(
    params: Dict[str, float | str] | None = None,
    reference_config: Dict[str, float] | None = None,
    sampling_config: Dict[str, float] | None = None,
) -> Dict[str, object]:
    controller = TEBMPCController(params=params)
    merged_reference = _merged_reference_config(reference_config)
    merged_sampling = _merged_sampling_config(sampling_config)
    segments = default_demo_segments(reference_config=merged_reference)
    reference = default_demo_reference(reference_config=merged_reference)
    segment_descriptions = []
    for segment in segments:
        if isinstance(segment, LineSegment):
            segment_descriptions.append(f"L({segment.length:.2f}m)")
        else:
            segment_descriptions.append(
                f"A(r={segment.radius:.2f}m, ang={segment.angle:.2f}rad, len={segment.length:.2f}m)"
            )

    return {
        "reference": {
            "ds": float(merged_reference["ds"]),
            "cruise_speed": float(merged_reference["cruise_speed"]),
            "dt_ref": float(reference.dt_ref),
            "segment_descriptions": segment_descriptions,
            "segment_count": len(segments),
            "target_length": float(sum(segment.length for segment in segments)),
            "params": merged_reference,
        },
        "sampling": {
            **merged_sampling,
        },
        "limits": {
            "dt_min": controller.dt_min,
            "dt_max": controller.dt_max,
            "max_speed": controller.max_speed,
            "max_accel": controller.max_accel,
            "max_lat_accel": controller.max_lat_accel,
            "max_jerk": controller.max_jerk,
            "max_kappa": controller.max_kappa,
            "max_dkappa": controller.max_dkappa,
        },
        "weights": {
            "w_pos": controller.w_pos,
            "terminal_cost_mode": controller.terminal_cost_mode,
            "w_pos_terminal": controller.w_pos_terminal,
            "w_pos_terminal_lateral": controller.w_pos_terminal_lateral,
            "w_pos_terminal_longitudinal": controller.w_pos_terminal_longitudinal,
            "w_theta": controller.w_theta,
            "w_speed": controller.w_speed,
            "w_time": controller.w_time,
            "w_speed_terminal": controller.w_speed_terminal,
            "w_pos_terminal_real": controller.w_pos_terminal_real,
            "w_pos_terminal_real_lateral": controller.w_pos_terminal_real_lateral,
            "w_pos_terminal_real_longitudinal": controller.w_pos_terminal_real_longitudinal,
            "w_theta_terminal_real": controller.w_theta_terminal_real,
            "w_speed_terminal_real": controller.w_speed_terminal_real,
            "w_dt_smooth": controller.w_dt_smooth,
            "w_jerk": controller.w_jerk,
            "w_dkappa": controller.w_dkappa,
        },
        "solver": {
            "ipopt_max_iter": controller.ipopt_max_iter,
            "ipopt_tol": controller.ipopt_tol,
            "ipopt_print_level": controller.ipopt_print_level,
        },
    }


def sample_random_initial_state(
    rng: np.random.Generator | None = None,
    reference: ReferenceTrajectory | None = None,
    sampling_config: Dict[str, float] | None = None,
) -> VehicleState:
    rng = rng or np.random.default_rng()
    config = _merged_sampling_config(sampling_config)
    reference = reference or default_demo_reference()
    base_x = float(reference.x[0])
    base_y = float(reference.y[0])
    base_theta = float(reference.theta[0])
    base_kappa = float(reference.kappa[0])

    return VehicleState(
        x=base_x + float(rng.uniform(-config["x_offset_range"], config["x_offset_range"])),
        y=base_y + float(rng.uniform(-config["y_offset_range"], config["y_offset_range"])),
        theta=_normalize_angle(base_theta + float(rng.uniform(-config["theta_offset_range"], config["theta_offset_range"]))),
        v=float(rng.uniform(config["speed_min"], config["speed_max"])),
        a=float(rng.uniform(config["accel_min"], config["accel_max"])),
        kappa=float(
            np.clip(
                base_kappa + rng.uniform(-config["kappa_offset_range"], config["kappa_offset_range"]),
                config["kappa_min"],
                config["kappa_max"],
            )
        ),
    )


def _projection_ratio_for_segment(index: int, segment_count: int, raw_ratio: float) -> float:
    if segment_count == 1:
        return raw_ratio
    if index == 0:
        return min(raw_ratio, 1.0)
    if index == segment_count - 1:
        return max(raw_ratio, 0.0)
    return float(np.clip(raw_ratio, 0.0, 1.0))


def _smoothstep_quintic(alpha: np.ndarray) -> np.ndarray:
    return 6.0 * np.power(alpha, 5) - 15.0 * np.power(alpha, 4) + 10.0 * np.power(alpha, 3)


def _derive_heading_from_positions(positions: np.ndarray, theta_start: float, theta_end: float) -> np.ndarray:
    if positions.shape[0] == 1:
        return np.array([theta_end], dtype=float)

    deltas = np.diff(positions, axis=0)
    segment_theta = np.array([math.atan2(delta[1], delta[0]) for delta in deltas], dtype=float)
    theta = np.empty(positions.shape[0], dtype=float)
    theta[0] = theta_start
    theta[1:] = np.unwrap(segment_theta, discont=math.pi)
    theta = np.unwrap(theta, discont=math.pi)
    theta[0] = theta_start
    theta[-1] = theta_end
    return theta


def _shape_curvature_profile(
    theta: np.ndarray,
    s: np.ndarray,
    state: VehicleState,
    v: np.ndarray,
    dt_ref: float,
    stop_constraints: Dict[str, float] | None,
) -> np.ndarray:
    sample_count = theta.shape[0]
    if sample_count <= 1:
        return np.array([float(state.kappa)], dtype=float)

    target_kappa = np.zeros(sample_count, dtype=float)
    ds = np.diff(s)
    dtheta = np.diff(theta)
    valid = ds > 1e-9
    target_kappa[1:][valid] = dtheta[valid] / ds[valid]
    target_kappa[0] = float(state.kappa)

    max_kappa = float(stop_constraints.get("max_kappa", 2.0)) if stop_constraints else 2.0
    max_dkappa = float(stop_constraints.get("max_dkappa", 1.5)) if stop_constraints else 1.5
    max_lat_accel = float(stop_constraints.get("max_lat_accel", float("inf"))) if stop_constraints else float("inf")

    for index in range(sample_count):
        lat_limit = max_kappa
        if math.isfinite(max_lat_accel) and max_lat_accel > 0.0:
            speed_sq = max(float(v[index]) ** 2, 1e-6)
            lat_limit = min(lat_limit, max_lat_accel / speed_sq)
        target_kappa[index] = float(np.clip(target_kappa[index], -lat_limit, lat_limit))

    delta_limit = max_dkappa * dt_ref
    forward = np.empty(sample_count, dtype=float)
    forward[0] = float(np.clip(state.kappa, -max_kappa, max_kappa))
    for index in range(1, sample_count):
        forward[index] = float(np.clip(target_kappa[index], forward[index - 1] - delta_limit, forward[index - 1] + delta_limit))

    shaped = np.empty(sample_count, dtype=float)
    shaped[-1] = 0.0
    for index in range(sample_count - 2, -1, -1):
        shaped[index] = float(np.clip(forward[index], shaped[index + 1] - delta_limit, shaped[index + 1] + delta_limit))

    return shaped


def _build_stopping_reference(
    state: VehicleState,
    reference: ReferenceTrajectory,
    sample_count: int,
    dt_ref: float,
    stop_constraints: Dict[str, float] | None = None,
) -> ReferenceTrajectory:
    sample_count = max(int(sample_count), 2)
    dt_ref = float(dt_ref)
    times = np.arange(sample_count, dtype=float) * dt_ref
    speed0 = max(float(state.v), 0.0)
    horizon = max(times[-1], dt_ref)
    decel = -speed0 / horizon if speed0 > 1e-9 else 0.0
    travel = speed0 * times + 0.5 * decel * np.square(times)
    travel = np.maximum(travel, 0.0)

    end_theta = float(reference.theta[-1])
    tangent = np.array([math.cos(end_theta), math.sin(end_theta)], dtype=float)
    end_point = np.array([float(reference.x[-1]), float(reference.y[-1])], dtype=float)
    state_point = np.array([float(state.x), float(state.y)], dtype=float)
    relative_to_end = state_point - end_point
    longitudinal_offset = float(np.dot(relative_to_end, tangent))
    projected_point = end_point + longitudinal_offset * tangent
    lateral_offset = state_point - projected_point
    alpha = np.linspace(0.0, 1.0, sample_count, dtype=float)
    blend = 1.0 - _smoothstep_quintic(alpha)
    positions = projected_point + np.outer(travel, tangent) + np.outer(blend, lateral_offset)

    v = np.maximum(speed0 + decel * times, 0.0)
    a = np.full(sample_count, decel, dtype=float)
    a[0] = float(state.a)
    a[-1] = 0.0
    s = np.zeros(sample_count, dtype=float)
    if sample_count > 1:
        s[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    theta = _derive_heading_from_positions(positions, float(state.theta), end_theta)
    kappa = _shape_curvature_profile(theta, s, state, v, dt_ref, stop_constraints)

    return ReferenceTrajectory(
        x=np.array(positions[:, 0], dtype=float),
        y=np.array(positions[:, 1], dtype=float),
        theta=theta,
        v=np.array(v, dtype=float),
        a=a,
        kappa=kappa,
        s=s,
        dt_ref=dt_ref,
    )


def _sample_reference_at_s(reference: ReferenceTrajectory, query_s: np.ndarray) -> Dict[str, np.ndarray]:
    original_s = np.array(reference.s, dtype=float)
    start_s = float(original_s[0])
    end_s = float(original_s[-1])
    clipped_s = np.clip(query_s, start_s, end_s)

    samples = {
        "x": np.interp(clipped_s, original_s, reference.x),
        "y": np.interp(clipped_s, original_s, reference.y),
        "theta": np.interp(clipped_s, original_s, reference.theta),
        "v": np.interp(clipped_s, original_s, reference.v),
        "a": np.interp(clipped_s, original_s, reference.a),
        "kappa": np.interp(clipped_s, original_s, reference.kappa),
    }

    before_start_mask = query_s < start_s
    if np.any(before_start_mask):
        delta_s = query_s[before_start_mask] - start_s
        theta_start = float(reference.theta[0])
        samples["x"][before_start_mask] = float(reference.x[0]) + delta_s * math.cos(theta_start)
        samples["y"][before_start_mask] = float(reference.y[0]) + delta_s * math.sin(theta_start)
        samples["theta"][before_start_mask] = theta_start
        samples["v"][before_start_mask] = float(reference.v[0])
        samples["a"][before_start_mask] = float(reference.a[0])
        samples["kappa"][before_start_mask] = 0.0

    return samples


def _resample_reference(reference: ReferenceTrajectory, sample_count: int) -> ReferenceTrajectory:
    sample_count = int(sample_count)
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if sample_count == reference.size:
        return reference

    original_s = np.array(reference.s, dtype=float)
    total_length = float(original_s[-1] - original_s[0])
    if total_length > 1e-9:
        query_s = np.linspace(float(original_s[0]), float(original_s[-1]), sample_count, dtype=float)
        samples = _sample_reference_at_s(reference, query_s)
        s = query_s - query_s[0]
    else:
        base_axis = np.linspace(0.0, 1.0, reference.size, dtype=float)
        query_axis = np.linspace(0.0, 1.0, sample_count, dtype=float)
        samples = {
            "x": np.interp(query_axis, base_axis, reference.x),
            "y": np.interp(query_axis, base_axis, reference.y),
            "theta": np.interp(query_axis, base_axis, reference.theta),
            "v": np.interp(query_axis, base_axis, reference.v),
            "a": np.interp(query_axis, base_axis, reference.a),
            "kappa": np.interp(query_axis, base_axis, reference.kappa),
        }
        s = np.zeros(sample_count, dtype=float)

    return ReferenceTrajectory(
        x=np.array(samples["x"], dtype=float),
        y=np.array(samples["y"], dtype=float),
        theta=np.array(samples["theta"], dtype=float),
        v=np.array(samples["v"], dtype=float),
        a=np.array(samples["a"], dtype=float),
        kappa=np.array(samples["kappa"], dtype=float),
        s=np.array(s, dtype=float),
        dt_ref=reference.dt_ref,
    )


def _build_aligned_query_s(
    original_s: np.ndarray,
    projection_s: float,
    selection_length: float | None = None,
) -> np.ndarray:
    shifted_s = projection_s + original_s
    end_s = float(original_s[-1])
    selection_end_s = end_s if selection_length is None else min(end_s, projection_s + float(selection_length))

    if projection_s > end_s:
        return shifted_s

    tol = 1e-9
    query_s = shifted_s[shifted_s <= selection_end_s + tol]
    if query_s.size == 0:
        query_s = np.array([projection_s], dtype=float)

    if projection_s >= 0.0 and query_s[-1] < selection_end_s - tol:
        query_s = np.concatenate((query_s, np.array([selection_end_s], dtype=float)))

    if query_s.size == 1:
        query_s = np.concatenate((query_s, np.array([query_s[0]], dtype=float)))

    return query_s


def project_state_onto_reference(reference: ReferenceTrajectory, state: VehicleState) -> Dict[str, float]:
    if reference.size == 1:
        return {
            "x": float(reference.x[0]),
            "y": float(reference.y[0]),
            "theta": float(reference.theta[0]),
            "v": float(reference.v[0]),
            "a": float(reference.a[0]),
            "kappa": float(reference.kappa[0]),
            "s": float(reference.s[0]),
        }

    query = np.array([state.x, state.y], dtype=float)
    best_projection = None
    best_distance_sq = float("inf")
    segment_count = reference.size - 1

    for index in range(segment_count):
        start = np.array([reference.x[index], reference.y[index]], dtype=float)
        end = np.array([reference.x[index + 1], reference.y[index + 1]], dtype=float)
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))

        if segment_len_sq <= 1e-12:
            ratio = 0.0
            projection = start
        else:
            raw_ratio = float(np.dot(query - start, segment) / segment_len_sq)
            ratio = _projection_ratio_for_segment(index=index, segment_count=segment_count, raw_ratio=raw_ratio)
            projection = start + ratio * segment

        distance_sq = float(np.dot(query - projection, query - projection))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_projection = {
                "x": float(projection[0]),
                "y": float(projection[1]),
                "theta": float(reference.theta[index] + ratio * (reference.theta[index + 1] - reference.theta[index])),
                "v": float(reference.v[index] + ratio * (reference.v[index + 1] - reference.v[index])),
                "a": float(reference.a[index] + ratio * (reference.a[index + 1] - reference.a[index])),
                "kappa": float(reference.kappa[index] + ratio * (reference.kappa[index + 1] - reference.kappa[index])),
                "s": float(reference.s[index] + ratio * (reference.s[index + 1] - reference.s[index])),
            }

    return best_projection


def align_reference_to_projection(
    reference: ReferenceTrajectory,
    state: VehicleState,
    extra_points: int = 0,
    selection_length: float | None = None,
    near_terminal_s_tol: float | None = None,
) -> ReferenceTrajectory:
    aligned_reference, _ = align_reference_to_projection_with_constraints(
        reference,
        state,
        extra_points=extra_points,
        selection_length=selection_length,
        near_terminal_s_tol=near_terminal_s_tol,
    )
    return aligned_reference


def align_reference_to_projection_with_constraints(
    reference: ReferenceTrajectory,
    state: VehicleState,
    stop_constraints: Dict[str, float] | None = None,
    extra_points: int = 0,
    selection_length: float | None = None,
    near_terminal_s_tol: float | None = None,
) -> Tuple[ReferenceTrajectory, Dict[str, object]]:
    resolved_extra_points = _resolve_extra_points(extra_points)
    resolved_selection_length = _resolve_selection_length(selection_length)
    projection = project_state_onto_reference(reference, state)
    projection_s = float(projection["s"])
    end_s = float(reference.s[-1])
    resolved_near_terminal_s_tol = _resolve_near_terminal_s_tol(reference, near_terminal_s_tol)
    remaining_s = end_s - projection_s
    if projection_s > end_s or remaining_s <= resolved_near_terminal_s_tol:
        stop_mode = "beyond_end_stop" if projection_s > end_s else "near_end_stop"
        return (
            _build_stopping_reference(
                state=state,
                reference=reference,
                sample_count=reference.size + resolved_extra_points,
                dt_ref=reference.dt_ref,
                stop_constraints=stop_constraints,
            ),
            {
                "mode": stop_mode,
                "is_stopping_reference": True,
                "remaining_s": float(remaining_s),
                "near_terminal_s_tol": float(resolved_near_terminal_s_tol),
                "end_extension_line": {
                    "x": float(reference.x[-1]),
                    "y": float(reference.y[-1]),
                    "theta": float(reference.theta[-1]),
                },
            },
        )

    original_s = np.array(reference.s, dtype=float)
    query_s = _build_aligned_query_s(
        original_s,
        float(projection["s"]),
        selection_length=resolved_selection_length,
    )
    aligned_s = query_s - query_s[0]
    aligned_samples = _sample_reference_at_s(reference, query_s)

    aligned_reference = ReferenceTrajectory(
            x=aligned_samples["x"],
            y=aligned_samples["y"],
            theta=aligned_samples["theta"],
            v=aligned_samples["v"],
            a=aligned_samples["a"],
            kappa=aligned_samples["kappa"],
            s=aligned_s,
            dt_ref=reference.dt_ref,
        )
    if resolved_extra_points != 0:
        target_sample_count = max(aligned_reference.size + resolved_extra_points, 2)
        aligned_reference = _resample_reference(aligned_reference, target_sample_count)

    return (
        aligned_reference,
        {
            "mode": "aligned_projection",
            "is_stopping_reference": False,
        },
    )


def run_random_demo(
    seed: int | None = None,
    params: Dict[str, float | str] | None = None,
    reference_config: Dict[str, float] | None = None,
    sampling_config: Dict[str, float] | None = None,
) -> Tuple[VehicleState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    merged_reference = _merged_reference_config(reference_config)
    extra_points = _resolve_extra_points(merged_reference.get("extra_points"))
    selection_length = _resolve_selection_length(merged_reference.get("selection_length"))
    near_terminal_s_tol = float(merged_reference.get("near_terminal_s_tol", 0.0))
    base_reference = default_demo_reference(reference_config=merged_reference)
    real_terminal_state = _reference_terminal_state(base_reference)
    initial_state = sample_random_initial_state(rng=rng, reference=base_reference, sampling_config=sampling_config)
    controller = TEBMPCController(params=params)
    stop_constraints = {
        "max_lat_accel": controller.max_lat_accel,
        "max_kappa": controller.max_kappa,
        "max_dkappa": controller.max_dkappa,
    }
    reference, reference_meta = align_reference_to_projection_with_constraints(
        base_reference,
        initial_state,
        stop_constraints=stop_constraints,
        extra_points=extra_points,
        selection_length=selection_length,
        near_terminal_s_tol=near_terminal_s_tol,
    )
    solution = controller.solve(initial_state=initial_state, reference=reference, real_terminal_state=real_terminal_state)
    solution["reference_meta"] = reference_meta
    return initial_state, reference, solution


def solve_demo(
    initial_state: VehicleState,
    params: Dict[str, float | str] | None = None,
    reference_config: Dict[str, float] | None = None,
) -> Tuple[VehicleState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    merged_reference = _merged_reference_config(reference_config)
    extra_points = _resolve_extra_points(merged_reference.get("extra_points"))
    selection_length = _resolve_selection_length(merged_reference.get("selection_length"))
    near_terminal_s_tol = float(merged_reference.get("near_terminal_s_tol", 0.0))
    base_reference = default_demo_reference(reference_config=merged_reference)
    real_terminal_state = _reference_terminal_state(base_reference)
    controller = TEBMPCController(params=params)
    stop_constraints = {
        "max_lat_accel": controller.max_lat_accel,
        "max_kappa": controller.max_kappa,
        "max_dkappa": controller.max_dkappa,
    }
    reference, reference_meta = align_reference_to_projection_with_constraints(
        base_reference,
        initial_state,
        stop_constraints=stop_constraints,
        extra_points=extra_points,
        selection_length=selection_length,
        near_terminal_s_tol=near_terminal_s_tol,
    )
    solution = controller.solve(initial_state=initial_state, reference=reference, real_terminal_state=real_terminal_state)
    solution["reference_meta"] = reference_meta
    return initial_state, reference, solution


def demo_problem(
    params: Dict[str, float | str] | None = None,
    reference_config: Dict[str, float] | None = None,
) -> Tuple[VehicleState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    initial_state = VehicleState(x=0.0, y=-0.3, theta=0.05, v=0.5, a=0.0, kappa=0.0)
    controller = TEBMPCController(params=params)
    merged_reference = _merged_reference_config(reference_config)
    extra_points = _resolve_extra_points(merged_reference.get("extra_points"))
    selection_length = _resolve_selection_length(merged_reference.get("selection_length"))
    near_terminal_s_tol = float(merged_reference.get("near_terminal_s_tol", 0.0))
    real_terminal_state = _reference_terminal_state(default_demo_reference(reference_config=merged_reference))
    stop_constraints = {
        "max_lat_accel": controller.max_lat_accel,
        "max_kappa": controller.max_kappa,
        "max_dkappa": controller.max_dkappa,
    }
    reference, reference_meta = align_reference_to_projection_with_constraints(
        default_demo_reference(reference_config=merged_reference),
        initial_state,
        stop_constraints=stop_constraints,
        extra_points=extra_points,
        selection_length=selection_length,
        near_terminal_s_tol=near_terminal_s_tol,
    )
    solution = controller.solve(initial_state=initial_state, reference=reference, real_terminal_state=real_terminal_state)
    solution["reference_meta"] = reference_meta
    return initial_state, reference, solution


if __name__ == "__main__":
    _, reference_traj, solution_dict = demo_problem()
    total_time = float(solution_dict["time"][-1])
    mean_dt = float(np.mean(solution_dict["dt"]))
    print(f"Reference points: {reference_traj.size}")
    print(f"Solve time: {solution_dict['solve_time_ms']:.2f} ms")
    print(f"Optimized horizon time: {total_time:.2f} s")
    print(f"Average dt: {mean_dt:.3f} s")