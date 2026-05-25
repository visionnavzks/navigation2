from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from my.teb_local_controller.teb_mpc import ReferenceTrajectory, VehicleState


@dataclass(frozen=True)
class StoppingReferenceDecision:
    mode: str
    remaining_s: float
    near_terminal_s_tol: float


class StoppingReferenceBuilder:
    def __init__(self, stop_constraints: Dict[str, float] | None = None):
        constraints = stop_constraints or {}
        self.max_kappa = float(constraints.get("max_kappa", 2.0))
        self.max_dkappa = float(constraints.get("max_dkappa", 1.5))
        self.max_lat_accel = float(constraints.get("max_lat_accel", float("inf")))
        self.max_stop_decel = max(float(constraints.get("max_accel", 1.0)), 1e-3)

    def evaluate(
        self,
        reference: ReferenceTrajectory,
        state: VehicleState,
        projection_s: float,
        near_terminal_s_tol: float | None,
    ) -> StoppingReferenceDecision | None:
        end_s = float(reference.s[-1])
        remaining_s = end_s - float(projection_s)
        resolved_near_terminal_s_tol = self.resolve_near_terminal_s_tol(reference, state, near_terminal_s_tol)

        if projection_s > end_s:
            return StoppingReferenceDecision(
                mode="beyond_end_stop",
                remaining_s=float(remaining_s),
                near_terminal_s_tol=float(resolved_near_terminal_s_tol),
            )

        if remaining_s <= resolved_near_terminal_s_tol:
            return StoppingReferenceDecision(
                mode="near_end_stop",
                remaining_s=float(remaining_s),
                near_terminal_s_tol=float(resolved_near_terminal_s_tol),
            )

        return None

    def resolve_near_terminal_s_tol(
        self,
        reference: ReferenceTrajectory,
        state: VehicleState,
        near_terminal_s_tol: float | None,
    ) -> float:
        if near_terminal_s_tol is not None:
            resolved_near_terminal_s_tol = float(near_terminal_s_tol)
            if resolved_near_terminal_s_tol > 0.0:
                return resolved_near_terminal_s_tol

        return max(self._reference_sample_spacing(reference), self._nominal_stop_distance(float(state.v)))

    def build(
        self,
        state: VehicleState,
        reference: ReferenceTrajectory,
        sample_count: int,
        dt_ref: float,
        mode: str,
    ) -> ReferenceTrajectory:
        sample_count = max(int(sample_count), 2)
        dt_ref = float(dt_ref)
        speed0 = max(float(state.v), 0.0)

        end_theta = float(reference.theta[-1])
        tangent = np.array([math.cos(end_theta), math.sin(end_theta)], dtype=float)
        end_point = np.array([float(reference.x[-1]), float(reference.y[-1])], dtype=float)
        state_point = np.array([float(state.x), float(state.y)], dtype=float)
        relative_to_end = state_point - end_point
        longitudinal_offset = float(np.dot(relative_to_end, tangent))
        projected_point = end_point + longitudinal_offset * tangent
        lateral_offset = state_point - projected_point

        alpha = np.linspace(0.0, 1.0, sample_count, dtype=float)
        smooth_alpha = self._smoothstep_quintic(alpha)

        if mode == "near_end_stop" and longitudinal_offset <= 0.0:
            positions = end_point + np.outer((1.0 - smooth_alpha) * longitudinal_offset, tangent) + np.outer(
                1.0 - smooth_alpha, lateral_offset
            )
            positions[-1] = end_point
        else:
            stopping_distance = self._nominal_stop_distance(speed0)
            positions = projected_point + np.outer(stopping_distance * smooth_alpha, tangent) + np.outer(
                1.0 - smooth_alpha, lateral_offset
            )

        s = np.zeros(sample_count, dtype=float)
        if sample_count > 1:
            s[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

        v, a = self._build_speed_profile(s, speed0, float(state.a))
        theta = self._derive_heading_from_positions(positions, float(state.theta), end_theta)
        kappa = self._shape_curvature_profile(theta, s, state, v, dt_ref)

        return ReferenceTrajectory(
            x=np.array(positions[:, 0], dtype=float),
            y=np.array(positions[:, 1], dtype=float),
            theta=theta,
            v=v,
            a=a,
            kappa=kappa,
            s=s,
            dt_ref=dt_ref,
        )

    def _reference_sample_spacing(self, reference: ReferenceTrajectory) -> float:
        if reference.size < 2:
            return 0.0

        positive_spacings = np.diff(np.array(reference.s, dtype=float))
        positive_spacings = positive_spacings[positive_spacings > 1e-9]
        if positive_spacings.size == 0:
            return 0.0
        return float(np.median(positive_spacings))

    def _nominal_stop_distance(self, speed: float) -> float:
        speed = max(float(speed), 0.0)
        if speed <= 1e-9:
            return 0.0
        return speed * speed / (2.0 * self.max_stop_decel)

    def _build_speed_profile(self, s: np.ndarray, speed0: float, initial_accel: float) -> tuple[np.ndarray, np.ndarray]:
        sample_count = s.shape[0]
        v = np.zeros(sample_count, dtype=float)
        a = np.zeros(sample_count, dtype=float)
        speed0 = max(float(speed0), 0.0)

        if sample_count == 0:
            return v, a

        if speed0 <= 1e-9 or sample_count == 1:
            a[0] = float(initial_accel)
            return v, a

        total_length = max(float(s[-1]), 1e-6)
        decel = -min(speed0 * speed0 / (2.0 * total_length), self.max_stop_decel)
        v = np.sqrt(np.maximum(speed0 * speed0 + 2.0 * decel * s, 0.0))
        a.fill(decel)
        a[0] = float(initial_accel)
        a[-1] = 0.0
        return v, a

    def _smoothstep_quintic(self, alpha: np.ndarray) -> np.ndarray:
        return 6.0 * np.power(alpha, 5) - 15.0 * np.power(alpha, 4) + 10.0 * np.power(alpha, 3)

    def _derive_heading_from_positions(
        self,
        positions: np.ndarray,
        theta_start: float,
        theta_end: float,
    ) -> np.ndarray:
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
        self,
        theta: np.ndarray,
        s: np.ndarray,
        state: VehicleState,
        v: np.ndarray,
        dt_ref: float,
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

        for index in range(sample_count):
            lat_limit = self.max_kappa
            if math.isfinite(self.max_lat_accel) and self.max_lat_accel > 0.0:
                speed_sq = max(float(v[index]) ** 2, 1e-6)
                lat_limit = min(lat_limit, self.max_lat_accel / speed_sq)
            target_kappa[index] = float(np.clip(target_kappa[index], -lat_limit, lat_limit))

        delta_limit = self.max_dkappa * dt_ref
        forward = np.empty(sample_count, dtype=float)
        forward[0] = float(np.clip(state.kappa, -self.max_kappa, self.max_kappa))
        for index in range(1, sample_count):
            forward[index] = float(
                np.clip(target_kappa[index], forward[index - 1] - delta_limit, forward[index - 1] + delta_limit)
            )

        shaped = np.empty(sample_count, dtype=float)
        shaped[-1] = 0.0
        for index in range(sample_count - 2, -1, -1):
            shaped[index] = float(np.clip(forward[index], shaped[index + 1] - delta_limit, shaped[index + 1] + delta_limit))

        return shaped