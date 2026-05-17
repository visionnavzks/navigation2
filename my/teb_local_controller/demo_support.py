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
DEMO_DT_REF = 0.1


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def default_demo_segments() -> List[LineSegment | ArcSegment]:
    return [
        LineSegment(length=1.8),
        ArcSegment(radius=1.8, angle=math.pi / 4.0),
        LineSegment(length=1.1),
        ArcSegment(radius=1.5, angle=-math.pi / 6.0),
        LineSegment(length=0.9),
    ]


def default_demo_reference() -> ReferenceTrajectory:
    return build_reference_trajectory(
        start=VehicleState(x=0.0, y=0.0, theta=0.0, v=0.6, a=0.0, kappa=0.0),
        segments=default_demo_segments(),
        ds=DEMO_REFERENCE_DS,
        cruise_speed=DEMO_CRUISE_SPEED,
        dt_ref=DEMO_DT_REF,
    )


def describe_demo_configuration(params: Dict[str, float] | None = None) -> Dict[str, object]:
    controller = TEBMPCController(params=params)
    segments = default_demo_segments()
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
            "ds": DEMO_REFERENCE_DS,
            "cruise_speed": DEMO_CRUISE_SPEED,
            "dt_ref": DEMO_DT_REF,
            "segment_descriptions": segment_descriptions,
            "segment_count": len(segments),
            "target_length": float(sum(segment.length for segment in segments)),
        },
        "limits": {
            "dt_min": controller.dt_min,
            "dt_max": controller.dt_max,
            "max_speed": controller.max_speed,
            "max_accel": controller.max_accel,
            "max_jerk": controller.max_jerk,
            "max_kappa": controller.max_kappa,
            "max_dkappa": controller.max_dkappa,
        },
        "weights": {
            "w_pos": controller.w_pos,
            "w_theta": controller.w_theta,
            "w_speed": controller.w_speed,
            "w_accel": controller.w_accel,
            "w_kappa": controller.w_kappa,
            "w_dt": controller.w_dt,
            "w_jerk": controller.w_jerk,
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
) -> VehicleState:
    rng = rng or np.random.default_rng()
    reference = reference or default_demo_reference()
    base_x = float(reference.x[0])
    base_y = float(reference.y[0])
    base_theta = float(reference.theta[0])
    base_kappa = float(reference.kappa[0])

    return VehicleState(
        x=base_x + float(rng.uniform(-1.5, 1.5)),
        y=base_y + float(rng.uniform(-2.0, 2.0)),
        theta=_normalize_angle(base_theta + float(rng.uniform(-0.7, 0.7))),
        v=float(rng.uniform(0.1, 1.2)),
        a=float(rng.uniform(-0.5, 0.5)),
        kappa=float(np.clip(base_kappa + rng.uniform(-0.08, 0.08), -0.2, 0.2)),
    )


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
                "v": float(reference.v[index] + ratio * (reference.v[index + 1] - reference.v[index])),
                "a": float(reference.a[index] + ratio * (reference.a[index + 1] - reference.a[index])),
                "kappa": float(reference.kappa[index] + ratio * (reference.kappa[index + 1] - reference.kappa[index])),
                "s": float(reference.s[index] + ratio * (reference.s[index + 1] - reference.s[index])),
            }

    return best_projection


def align_reference_to_projection(reference: ReferenceTrajectory, state: VehicleState) -> ReferenceTrajectory:
    projection = project_state_onto_reference(reference, state)
    original_s = np.array(reference.s, dtype=float)
    query_s = np.clip(projection["s"] + original_s, projection["s"], original_s[-1])
    aligned_s = query_s - query_s[0]

    aligned_x = np.interp(query_s, original_s, reference.x)
    aligned_y = np.interp(query_s, original_s, reference.y)
    aligned_theta = np.interp(query_s, original_s, reference.theta)
    aligned_v = np.interp(query_s, original_s, reference.v)
    aligned_a = np.interp(query_s, original_s, reference.a)
    aligned_kappa = np.interp(query_s, original_s, reference.kappa)

    return ReferenceTrajectory(
        x=aligned_x,
        y=aligned_y,
        theta=aligned_theta,
        v=aligned_v,
        a=aligned_a,
        kappa=aligned_kappa,
        s=aligned_s,
        dt_ref=reference.dt_ref,
    )


def run_random_demo(
    seed: int | None = None,
    params: Dict[str, float] | None = None,
) -> Tuple[VehicleState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    base_reference = default_demo_reference()
    initial_state = sample_random_initial_state(rng=rng, reference=base_reference)
    reference = align_reference_to_projection(base_reference, initial_state)
    controller = TEBMPCController(params=params)
    solution = controller.solve(initial_state=initial_state, reference=reference)
    return initial_state, reference, solution


def demo_problem() -> Tuple[VehicleState, ReferenceTrajectory, Dict[str, np.ndarray | float | Dict[str, float]]]:
    initial_state = VehicleState(x=0.0, y=-0.3, theta=0.05, v=0.5, a=0.0, kappa=0.0)
    reference = align_reference_to_projection(default_demo_reference(), initial_state)
    controller = TEBMPCController()
    solution = controller.solve(initial_state=initial_state, reference=reference)
    return initial_state, reference, solution


if __name__ == "__main__":
    _, reference_traj, solution_dict = demo_problem()
    total_time = float(solution_dict["time"][-1])
    mean_dt = float(np.mean(solution_dict["dt"]))
    print(f"Reference points: {reference_traj.size}")
    print(f"Solve time: {solution_dict['solve_time_ms']:.2f} ms")
    print(f"Optimized horizon time: {total_time:.2f} s")
    print(f"Average dt: {mean_dt:.3f} s")