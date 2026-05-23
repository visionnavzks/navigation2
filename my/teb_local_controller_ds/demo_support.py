from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from my.teb_local_controller_ds.teb_mpc import (
    ArcSegment,
    DSMPCController,
    LineSegment,
    ReferenceTrajectory,
    SpatialState,
    build_reference_trajectory,
)


DEMO_REFERENCE_DS = 0.25


DEMO_REFERENCE_DEFAULTS: Dict[str, float] = {
    "ds": DEMO_REFERENCE_DS,
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
    return build_reference_trajectory(
        start=SpatialState(x=0.0, y=0.0, theta=0.0, kappa=0.0),
        segments=default_demo_segments(reference_config=config),
        ds=float(config["ds"]),
    )


def describe_demo_configuration(
    params: Dict[str, float] | None = None,
    reference_config: Dict[str, float] | None = None,
    sampling_config: Dict[str, float] | None = None,
) -> Dict[str, object]:
    controller = DSMPCController(params=params)
    merged_reference = _merged_reference_config(reference_config)
    merged_sampling = _merged_sampling_config(sampling_config)
    segments = default_demo_segments(reference_config=merged_reference)
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
            "segment_descriptions": segment_descriptions,
            "segment_count": len(segments),
            "target_length": float(sum(segment.length for segment in segments)),
            "params": merged_reference,
        },
        "sampling": {
            **merged_sampling,
        },
        "limits": {
            "ds_min": controller.ds_min,
            "ds_max": controller.ds_max,
            "max_kappa": controller.max_kappa,
            "max_dkappa": controller.max_dkappa,
        },
        "weights": {
            "w_pos": controller.w_pos,
            "w_theta": controller.w_theta,
            "w_kappa": controller.w_kappa,
            "w_ds": controller.w_ds,
            "w_dkappa": controller.w_dkappa,
            "w_terminal": controller.w_terminal,
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
) -> SpatialState:
    rng = rng or np.random.default_rng()
    config = _merged_sampling_config(sampling_config)
    reference = reference or default_demo_reference()
    base_x = float(reference.x[0])
    base_y = float(reference.y[0])
    base_theta = float(reference.theta[0])
    base_kappa = float(reference.kappa[0])

    return SpatialState(
        x=base_x + float(rng.uniform(-config["x_offset_range"], config["x_offset_range"])),
        y=base_y + float(rng.uniform(-config["y_offset_range"], config["y_offset_range"])),
        theta=_normalize_angle(base_theta + float(rng.uniform(-config["theta_offset_range"], config["theta_offset_range"]))),
        kappa=float(
            np.clip(
                base_kappa + rng.uniform(-config["kappa_offset_range"], config["kappa_offset_range"]),
                config["kappa_min"],
                config["kappa_max"],
            )
        ),
    )


def project_state_onto_reference(reference: ReferenceTrajectory, state: SpatialState) -> Dict[str, float]:
    if reference.size == 1:
        return {
            "x": float(reference.x[0]),
            "y": float(reference.y[0]),
            "theta": float(reference.theta[0]),
            "kappa": float(reference.kappa[0]),
            "s": float(reference.s[0]),
        }

    query = np.array([state.x, state.y], dtype=float)
    best_projection = None
    best_distance_sq = float("inf")

    for index in range(reference.size - 1):
        start = np.array([reference.x[index], reference.y[index]], dtype=float)
        end = np.array([reference.x[index + 1], reference.y[index + 1]], dtype=float)
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))

        if segment_len_sq <= 1e-12:
            ratio = 0.0
            projection = start
        else:
            ratio = float(np.clip(np.dot(query - start, segment) / segment_len_sq, 0.0, 1.0))
            projection = start + ratio * segment

        distance_sq = float(np.dot(query - projection, query - projection))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_projection = {
                "x": float(projection[0]),
                "y": float(projection[1]),
                "theta": float(reference.theta[index] + ratio * (reference.theta[index + 1] - reference.theta[index])),
                "kappa": float(reference.kappa[index] + ratio * (reference.kappa[index + 1] - reference.kappa[index])),
                "s": float(reference.s[index] + ratio * (reference.s[index + 1] - reference.s[index])),
            }

    return best_projection


def align_reference_to_projection(reference: ReferenceTrajectory, state: SpatialState) -> ReferenceTrajectory:
    projection = project_state_onto_reference(reference, state)
    original_s = np.array(reference.s, dtype=float)
    query_s = np.clip(projection["s"] + original_s, projection["s"], original_s[-1])
    aligned_s = query_s - query_s[0]

    return ReferenceTrajectory(
        x=np.interp(query_s, original_s, reference.x),
        y=np.interp(query_s, original_s, reference.y),
        theta=np.interp(query_s, original_s, reference.theta),
        kappa=np.interp(query_s, original_s, reference.kappa),
        s=aligned_s,
    )


def run_random_demo(
    seed: int | None = None,
    params: Dict[str, float] | None = None,
    reference_config: Dict[str, float] | None = None,
    sampling_config: Dict[str, float] | None = None,
) -> Tuple[SpatialState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    base_reference = default_demo_reference(reference_config=reference_config)
    initial_state = sample_random_initial_state(rng=rng, reference=base_reference, sampling_config=sampling_config)
    reference = align_reference_to_projection(base_reference, initial_state)
    controller = DSMPCController(params=params)
    solution = controller.solve(initial_state=initial_state, reference=reference)
    return initial_state, reference, solution


def solve_demo(
    initial_state: SpatialState,
    params: Dict[str, float] | None = None,
    reference_config: Dict[str, float] | None = None,
) -> Tuple[SpatialState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    base_reference = default_demo_reference(reference_config=reference_config)
    reference = align_reference_to_projection(base_reference, initial_state)
    controller = DSMPCController(params=params)
    solution = controller.solve(initial_state=initial_state, reference=reference)
    return initial_state, reference, solution


def demo_problem(
    params: Dict[str, float] | None = None,
    reference_config: Dict[str, float] | None = None,
) -> Tuple[SpatialState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    initial_state = SpatialState(x=0.0, y=-0.3, theta=0.05, kappa=0.0)
    reference = align_reference_to_projection(default_demo_reference(reference_config=reference_config), initial_state)
    controller = DSMPCController(params=params)
    solution = controller.solve(initial_state=initial_state, reference=reference)
    return initial_state, reference, solution


if __name__ == "__main__":
    _, reference_traj, solution_dict = demo_problem()
    total_distance = float(solution_dict["s"][-1])
    mean_ds = float(np.mean(solution_dict["ds"]))
    print(f"Reference points: {reference_traj.size}")
    print(f"Solve time: {solution_dict['solve_time_ms']:.2f} ms")
    print(f"Optimized horizon distance: {total_distance:.2f} m")
    print(f"Average ds: {mean_ds:.3f} m")