from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class SpatialState:
    x: float
    y: float
    theta: float
    kappa: float = 0.0


@dataclass(frozen=True)
class LineSegment:
    length: float


@dataclass(frozen=True)
class ArcSegment:
    radius: float
    angle: float

    @property
    def curvature(self) -> float:
        if abs(self.radius) < 1e-9:
            raise ValueError("Arc radius must be non-zero")
        return math.copysign(1.0 / abs(self.radius), self.angle)

    @property
    def length(self) -> float:
        return abs(self.radius * self.angle)


PathSegment = Union[LineSegment, ArcSegment]


@dataclass(frozen=True)
class ReferenceTrajectory:
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    s: np.ndarray

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


def _append_line_samples(
    samples: List[Tuple[float, float, float, float]],
    state: SpatialState,
    segment: LineSegment,
    ds: float,
) -> SpatialState:
    steps = max(int(math.ceil(abs(segment.length) / ds)), 1)
    step_length = segment.length / steps
    x = state.x
    y = state.y
    theta = state.theta
    s = samples[-1][3]

    for _ in range(steps):
        x += step_length * math.cos(theta)
        y += step_length * math.sin(theta)
        s += abs(step_length)
        samples.append((x, y, theta, s))

    return SpatialState(x=x, y=y, theta=theta, kappa=0.0)


def _append_arc_samples(
    samples: List[Tuple[float, float, float, float]],
    state: SpatialState,
    segment: ArcSegment,
    ds: float,
) -> SpatialState:
    steps = max(int(math.ceil(segment.length / ds)), 1)
    step_angle = segment.angle / steps
    curvature = segment.curvature
    arc_step = segment.length / steps
    x = state.x
    y = state.y
    theta = state.theta
    s = samples[-1][3]

    for _ in range(steps):
        theta_next = theta + step_angle
        if abs(curvature) < 1e-9:
            x += arc_step * math.cos(theta)
            y += arc_step * math.sin(theta)
        else:
            x += (math.sin(theta_next) - math.sin(theta)) / curvature
            y += (math.cos(theta) - math.cos(theta_next)) / curvature
        theta = theta_next
        s += arc_step
        samples.append((x, y, theta, s))

    return SpatialState(x=x, y=y, theta=theta, kappa=curvature)


def build_reference_trajectory(
    start: SpatialState,
    segments: Sequence[PathSegment],
    ds: float = 0.2,
) -> ReferenceTrajectory:
    if ds <= 0.0:
        raise ValueError("ds must be positive")

    samples: List[Tuple[float, float, float, float]] = [(start.x, start.y, start.theta, 0.0)]
    curvatures = [start.kappa]
    state = start

    for segment in segments:
        if isinstance(segment, LineSegment):
            state = _append_line_samples(samples, state, segment, ds)
            curvatures.extend([0.0] * (len(samples) - len(curvatures)))
        elif isinstance(segment, ArcSegment):
            prev_len = len(samples)
            state = _append_arc_samples(samples, state, segment, ds)
            curvatures.extend([segment.curvature] * (len(samples) - prev_len))
        else:
            raise TypeError(f"Unsupported segment type: {type(segment)!r}")

    x = np.array([item[0] for item in samples], dtype=float)
    y = np.array([item[1] for item in samples], dtype=float)
    theta = np.unwrap(np.array([item[2] for item in samples], dtype=float))
    s = np.array([item[3] for item in samples], dtype=float)
    kappa = np.array(curvatures, dtype=float)
    return ReferenceTrajectory(x=x, y=y, theta=theta, kappa=kappa, s=s)


class DSMPCController:
    def __init__(self, params: Dict[str, float] | None = None):
        self.params = params or {}
        self._load_parameters()

    def _load_parameters(self) -> None:
        self.ds_min = float(self.params.get("ds_min", 0.08))
        self.ds_max = float(self.params.get("ds_max", 0.45))
        self.max_kappa = float(self.params.get("max_kappa", 0.4))
        self.max_dkappa = float(self.params.get("max_dkappa", 1.2))
        self.w_pos = float(self.params.get("w_pos", 30.0))
        self.w_theta = float(self.params.get("w_theta", 8.0))
        self.w_kappa = float(self.params.get("w_kappa", 4.0))
        self.w_ds = float(self.params.get("w_ds", 8.0))
        self.w_dkappa = float(self.params.get("w_dkappa", 0.8))
        self.w_terminal = float(self.params.get("w_terminal", 60.0))
        self.ipopt_max_iter = int(self.params.get("ipopt_max_iter", 500))
        self.ipopt_tol = float(self.params.get("ipopt_tol", 1e-6))
        self.ipopt_print_level = int(self.params.get("ipopt_print_level", 0))

    def solve(self, initial_state: SpatialState, reference: ReferenceTrajectory) -> Dict[str, np.ndarray | float | Dict[str, float]]:
        if reference.size < 2:
            raise ValueError("Reference trajectory must contain at least 2 samples")

        n = reference.size
        reference_ds = np.diff(reference.s)
        opti = ca.Opti()

        x = opti.variable(n)
        y = opti.variable(n)
        theta = opti.variable(n)
        kappa = opti.variable(n)

        ds = opti.variable(n - 1)
        dkappa = opti.variable(n - 1)

        cost_track = 0
        cost_control = 0
        for i in range(n):
            cost_track += self.w_pos * ((x[i] - reference.x[i]) ** 2 + (y[i] - reference.y[i]) ** 2)
            cost_track += self.w_theta * (1.0 - ca.cos(theta[i] - reference.theta[i]))
            cost_track += self.w_kappa * (kappa[i] - reference.kappa[i]) ** 2

        for i in range(n - 1):
            cost_control += self.w_ds * (ds[i] - reference_ds[i]) ** 2
            cost_control += self.w_dkappa * dkappa[i] ** 2

        terminal_cost = self.w_terminal * (
            (x[-1] - reference.x[-1]) ** 2
            + (y[-1] - reference.y[-1]) ** 2
            + (1.0 - ca.cos(theta[-1] - reference.theta[-1]))
            + (kappa[-1] - reference.kappa[-1]) ** 2
        )
        opti.minimize(cost_track + cost_control + terminal_cost)

        for i in range(n - 1):
            kappa_next = kappa[i] + ds[i] * dkappa[i]
            kappa_mid = kappa[i] + 0.5 * ds[i] * dkappa[i]
            theta_next = theta[i] + ds[i] * kappa_mid
            theta_mid = theta[i] + 0.5 * ds[i] * kappa_mid
            x_next = x[i] + ds[i] * ca.cos(theta_mid)
            y_next = y[i] + ds[i] * ca.sin(theta_mid)

            opti.subject_to(kappa[i + 1] == kappa_next)
            opti.subject_to(theta[i + 1] == theta_next)
            opti.subject_to(x[i + 1] == x_next)
            opti.subject_to(y[i + 1] == y_next)

        opti.subject_to(x[0] == initial_state.x)
        opti.subject_to(y[0] == initial_state.y)
        opti.subject_to(theta[0] == initial_state.theta)
        opti.subject_to(kappa[0] == initial_state.kappa)

        opti.subject_to(opti.bounded(self.ds_min, ds, self.ds_max))
        opti.subject_to(opti.bounded(-self.max_kappa, kappa, self.max_kappa))
        opti.subject_to(opti.bounded(-self.max_dkappa, dkappa, self.max_dkappa))

        opti.set_initial(x, reference.x)
        opti.set_initial(y, reference.y)
        opti.set_initial(theta, reference.theta)
        opti.set_initial(kappa, np.clip(reference.kappa, -self.max_kappa, self.max_kappa))
        opti.set_initial(ds, np.clip(reference_ds, self.ds_min, self.ds_max))
        opti.set_initial(dkappa, np.zeros(n - 1, dtype=float))

        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": self.ipopt_max_iter,
            "tol": self.ipopt_tol,
            "print_level": self.ipopt_print_level,
        }
        opti.solver("ipopt", p_opts, s_opts)

        start_time = time.time()
        try:
            sol = opti.solve()
        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000.0
            raise RuntimeError(f"DS MPC solve failed after {elapsed_ms:.2f} ms: {exc}") from exc

        elapsed_ms = (time.time() - start_time) * 1000.0
        optimized_ds = np.array(sol.value(ds), dtype=float)
        result = {
            "x": np.array(sol.value(x), dtype=float),
            "y": np.array(sol.value(y), dtype=float),
            "theta": np.array(sol.value(theta), dtype=float),
            "kappa": np.array(sol.value(kappa), dtype=float),
            "ds": optimized_ds,
            "dkappa": np.array(sol.value(dkappa), dtype=float),
            "solve_time_ms": float(elapsed_ms),
            "costs": {
                "track": float(sol.value(cost_track)),
                "control": float(sol.value(cost_control)),
                "terminal": float(sol.value(terminal_cost)),
                "total": float(sol.value(cost_track + cost_control + terminal_cost)),
            },
        }
        result["s"] = np.concatenate(([0.0], np.cumsum(optimized_ds)))
        return result
