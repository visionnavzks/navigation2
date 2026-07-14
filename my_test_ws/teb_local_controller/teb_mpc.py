from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    theta: float
    v: float = 0.0
    a: float = 0.0
    kappa: float = 0.0


@dataclass(frozen=True)
class GoalPoint:
    x: float
    y: float
    theta: float | None = None
    v: float = 0.0
    a: float = 0.0
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
GoalInput = Union[GoalPoint, VehicleState, Sequence[float], Dict[str, float]]


@dataclass(frozen=True)
class ReferenceTrajectory:
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    a: np.ndarray
    kappa: np.ndarray
    s: np.ndarray
    dt_ref: float

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


def _append_line_samples(
    samples: List[Tuple[float, float, float, float]],
    state: VehicleState,
    segment: LineSegment,
    ds: float,
) -> VehicleState:
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

    return VehicleState(x=x, y=y, theta=theta, v=state.v, a=state.a, kappa=0.0)


def _append_arc_samples(
    samples: List[Tuple[float, float, float, float]],
    state: VehicleState,
    segment: ArcSegment,
    ds: float,
) -> VehicleState:
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

    return VehicleState(x=x, y=y, theta=theta, v=state.v, a=state.a, kappa=curvature)


def build_reference_trajectory(
    start: VehicleState,
    segments: Sequence[PathSegment],
    ds: float = 0.2,
    cruise_speed: float = 1.0,
    dt_ref: float = 0.1,
    terminal_speed: float = 0.0,
) -> ReferenceTrajectory:
    if ds <= 0.0:
        raise ValueError("ds must be positive")
    if dt_ref <= 0.0:
        raise ValueError("dt_ref must be positive")

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

    v = np.full_like(x, float(cruise_speed))
    if x.shape[0] > 0:
        v[0] = start.v
        v[-1] = float(terminal_speed)
    a = np.zeros_like(x)
    if x.shape[0] > 0:
        a[0] = start.a

    return ReferenceTrajectory(x=x, y=y, theta=theta, v=v, a=a, kappa=kappa, s=s, dt_ref=float(dt_ref))


def _coerce_goal_point(goal: GoalInput, goal_theta: float | None = None) -> GoalPoint:
    if isinstance(goal, GoalPoint):
        theta = goal.theta if goal_theta is None else goal_theta
        return GoalPoint(x=goal.x, y=goal.y, theta=theta, v=goal.v, a=goal.a, kappa=goal.kappa)

    if isinstance(goal, VehicleState):
        theta = goal.theta if goal_theta is None else goal_theta
        return GoalPoint(x=goal.x, y=goal.y, theta=theta, v=goal.v, a=goal.a, kappa=goal.kappa)

    if isinstance(goal, dict):
        theta_value = goal.get("theta", goal_theta)
        return GoalPoint(
            x=float(goal["x"]),
            y=float(goal["y"]),
            theta=None if theta_value is None else float(theta_value),
            v=float(goal.get("v", 0.0)),
            a=float(goal.get("a", 0.0)),
            kappa=float(goal.get("kappa", 0.0)),
        )

    values = list(goal)
    if len(values) not in {2, 3}:
        raise ValueError("Goal sequence must be (x, y) or (x, y, theta)")
    theta = goal_theta if goal_theta is not None else (float(values[2]) if len(values) == 3 else None)
    return GoalPoint(x=float(values[0]), y=float(values[1]), theta=theta)


def _smoothstep_quintic(alpha: np.ndarray) -> np.ndarray:
    return 6.0 * np.power(alpha, 5) - 15.0 * np.power(alpha, 4) + 10.0 * np.power(alpha, 3)


def _derive_heading_from_positions(
    positions: np.ndarray,
    theta_start: float,
    theta_end: float,
) -> np.ndarray:
    if positions.shape[0] == 1:
        return np.array([theta_end], dtype=float)

    deltas = np.diff(positions, axis=0)
    segment_theta = np.array(
        [
            theta_start if float(np.linalg.norm(delta)) <= 1e-9 else math.atan2(float(delta[1]), float(delta[0]))
            for delta in deltas
        ],
        dtype=float,
    )
    theta = np.empty(positions.shape[0], dtype=float)
    theta[0] = theta_start
    theta[1:] = np.unwrap(segment_theta, discont=math.pi)
    theta = np.unwrap(theta, discont=math.pi)
    theta[0] = theta_start
    theta[-1] = theta_end
    return theta


def _curvature_from_heading(theta: np.ndarray, s: np.ndarray, initial_kappa: float, terminal_kappa: float) -> np.ndarray:
    if theta.shape[0] <= 1:
        return np.array([float(terminal_kappa)], dtype=float)

    kappa = np.zeros(theta.shape[0], dtype=float)
    ds = np.diff(s)
    dtheta = np.diff(theta)
    valid = ds > 1e-9
    kappa[1:][valid] = dtheta[valid] / ds[valid]
    kappa[0] = float(initial_kappa)
    kappa[-1] = float(terminal_kappa)
    return kappa


def build_goal_reference(
    start: VehicleState,
    goal: GoalInput,
    ds: float = 0.2,
    cruise_speed: float = 1.0,
    dt_ref: float = 0.1,
    sample_count: int | None = None,
    min_samples: int = 8,
    max_samples: int | None = 50,
    goal_theta: float | None = None,
) -> ReferenceTrajectory:
    if ds <= 0.0:
        raise ValueError("ds must be positive")
    if dt_ref <= 0.0:
        raise ValueError("dt_ref must be positive")

    target = _coerce_goal_point(goal, goal_theta=goal_theta)
    start_xy = np.array([float(start.x), float(start.y)], dtype=float)
    goal_xy = np.array([float(target.x), float(target.y)], dtype=float)
    delta = goal_xy - start_xy
    distance = float(np.linalg.norm(delta))

    if target.theta is None:
        terminal_theta = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-9 else float(start.theta)
    else:
        terminal_theta = float(target.theta)

    if sample_count is None:
        sample_count = max(int(math.ceil(distance / ds)) + 1, int(min_samples), 2)
    else:
        sample_count = max(int(sample_count), 2)
    if max_samples is not None:
        sample_count = min(sample_count, max(int(max_samples), 2))

    alpha = np.linspace(0.0, 1.0, sample_count, dtype=float)
    if distance <= 1e-9:
        positions = np.repeat(start_xy[np.newaxis, :], sample_count, axis=0)
    elif target.theta is None:
        blend = _smoothstep_quintic(alpha)
        positions = start_xy + np.outer(blend, delta)
    else:
        tangent_scale = max(distance, ds * (sample_count - 1)) * 0.45
        start_tangent = tangent_scale * np.array([math.cos(start.theta), math.sin(start.theta)], dtype=float)
        goal_tangent = tangent_scale * np.array([math.cos(terminal_theta), math.sin(terminal_theta)], dtype=float)
        h00 = 2.0 * alpha**3 - 3.0 * alpha**2 + 1.0
        h10 = alpha**3 - 2.0 * alpha**2 + alpha
        h01 = -2.0 * alpha**3 + 3.0 * alpha**2
        h11 = alpha**3 - alpha**2
        positions = (
            np.outer(h00, start_xy)
            + np.outer(h10, start_tangent)
            + np.outer(h01, goal_xy)
            + np.outer(h11, goal_tangent)
        )

    positions[0] = start_xy
    positions[-1] = goal_xy

    s = np.zeros(sample_count, dtype=float)
    if sample_count > 1:
        s[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

    theta = _derive_heading_from_positions(positions, float(start.theta), terminal_theta)
    kappa = _curvature_from_heading(theta, s, float(start.kappa), float(target.kappa))

    v = np.full(sample_count, float(cruise_speed), dtype=float)
    v[0] = float(start.v)
    v[-1] = float(target.v)
    a = np.zeros(sample_count, dtype=float)
    a[0] = float(start.a)
    a[-1] = float(target.a)

    return ReferenceTrajectory(
        x=np.array(positions[:, 0], dtype=float),
        y=np.array(positions[:, 1], dtype=float),
        theta=theta,
        v=v,
        a=a,
        kappa=kappa,
        s=s,
        dt_ref=float(dt_ref),
    )


def resample_reference(
    reference: ReferenceTrajectory,
    sample_count: int,
    time_values: np.ndarray | None = None,
) -> ReferenceTrajectory:
    """把参考轨迹重采样为 sample_count 个点。

    首尾(含终点目标)保持不变,因此重采样只改变时间/空间离散密度,不改变
    起点约束和终端代价所对应的目标位姿。用于外层 autoResize 循环:当优化得到的
    dt 系统性偏离 dt_ref 时,通过增/减采样点把 dt 拉回期望值附近。

    ``time_values`` 是上一轮求解得到的、跟 reference 各点一一对应的累计时间
    (即 solution["time"]);给定时按"预估时间"等距重采样——用上一轮的速度画像把
    新知识点在弧长上重新分布(慢的地方点更密、快的地方点更疏),让下一轮逐段 dt
    更接近 dt_ref,而不只是平均值接近。不给或退化失败时按弧长 s 等距重采样。
    """
    sample_count = max(int(sample_count), 2)
    if sample_count == reference.size:
        return reference

    original_s = np.asarray(reference.s, dtype=float)
    total_s = float(original_s[-1])
    if total_s <= 1e-9:
        return reference

    query_s = np.linspace(0.0, total_s, sample_count)
    if time_values is not None:
        old_t = np.asarray(time_values, dtype=float)
        total_t = float(old_t[-1]) if old_t.size else 0.0
        if old_t.shape == original_s.shape and total_t > 1e-9:
            query_t = np.linspace(0.0, total_t, sample_count)
            query_s = np.interp(query_t, old_t, original_s)

    return ReferenceTrajectory(
        x=np.interp(query_s, original_s, reference.x),
        y=np.interp(query_s, original_s, reference.y),
        theta=np.interp(query_s, original_s, reference.theta),
        v=np.interp(query_s, original_s, reference.v),
        a=np.interp(query_s, original_s, reference.a),
        kappa=np.interp(query_s, original_s, reference.kappa),
        s=query_s,
        dt_ref=float(reference.dt_ref),
    )


class TEBMPCController:
    def __init__(self, params: Dict[str, float] | None = None):
        self.params = params or {}
        self._load_parameters()

    def _load_parameters(self) -> None:
        self.dt_ref = float(self.params.get("dt_ref", 0.1))
        self.dt_min = float(self.params.get("dt_min", 0.03))
        self.dt_max = float(self.params.get("dt_max", 0.35))
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        # 外层重采样(autoResize):dt_ref 是期望的时间步长,dt_hysteresis 是判断
        # "dt 太大/太小"的死区半径,max_outer_iterations 是外层最多迭代几轮。
        self.dt_hysteresis = float(self.params.get("dt_hysteresis", 0.1 * self.dt_ref))
        self.max_outer_iterations = int(self.params.get("max_outer_iterations", 3))
        self.max_speed = float(self.params.get("max_speed", 2.5))
        self.max_accel = float(self.params.get("max_accel", 1.0))
        self.max_lat_accel = float(self.params.get("max_lat_accel", 1.5))
        self.max_jerk = float(self.params.get("max_jerk", 3.0))
        self.max_kappa = float(self.params.get("max_kappa", 2.0))
        self.max_dkappa = float(self.params.get("max_dkappa", 1.5))
        self.w_lat_goal = float(self.params.get("w_lat_goal", self.params.get("w_terminal", 30.0)))
        self.w_lon_goal = float(self.params.get("w_lon_goal", self.params.get("w_terminal", 10.0)))
        self.w_theta_goal = float(self.params.get("w_theta_goal", 60.0))
        self.w_speed_goal = float(self.params.get("w_speed_goal", 10.0))
        self.w_accel_goal = float(self.params.get("w_accel_goal", 2.0))
        self.w_time = float(self.params.get("w_time", 2.0))
        self.w_length = float(self.params.get("w_length", 0.0))
        self.w_dt_uniform = float(self.params.get("w_dt_uniform", 10000.0))
        self.w_jerk = float(self.params.get("w_jerk", 0.5))
        self.w_dkappa = float(self.params.get("w_dkappa", 0.5))
        self.w_accel = float(self.params.get("w_accel", 0.0))
        self.w_kappa = float(self.params.get("w_kappa", 0.0))
        self.ipopt_max_iter = int(self.params.get("ipopt_max_iter", 500))
        self.ipopt_tol = float(self.params.get("ipopt_tol", 1e-6))
        self.ipopt_print_level = int(self.params.get("ipopt_print_level", 0))

    def solve_to_goal(
        self,
        initial_state: VehicleState,
        goal: GoalInput,
        ds: float = 0.2,
        cruise_speed: float | None = None,
        dt_ref: float | None = None,
        sample_count: int | None = None,
        goal_theta: float | None = None,
    ) -> Dict[str, object]:
        reference = build_goal_reference(
            start=initial_state,
            goal=goal,
            ds=ds,
            cruise_speed=self.max_speed * 0.4 if cruise_speed is None else cruise_speed,
            dt_ref=self.dt_ref if dt_ref is None else dt_ref,
            sample_count=sample_count,
            goal_theta=goal_theta,
        )
        solution = self.solve_with_resize(initial_state=initial_state, reference=reference)
        solution["reference_meta"] = {
            "mode": "point_goal",
            "is_stopping_reference": False,
            "goal": {
                "x": float(reference.x[-1]),
                "y": float(reference.y[-1]),
                "theta": float(reference.theta[-1]),
                "v": float(reference.v[-1]),
                "a": float(reference.a[-1]),
                "kappa": float(reference.kappa[-1]),
            },
            "reference_size": int(np.asarray(solution["x"]).shape[0]),
            "reference_length": float(reference.s[-1]),
        }
        return solution

    def solve_with_resize(
        self,
        initial_state: VehicleState,
        reference: ReferenceTrajectory,
        max_outer_iterations: int | None = None,
        dt_hysteresis: float | None = None,
        min_samples: int = 3,
        max_samples: int = 300,
        record: bool = False,
    ) -> Dict[str, object]:
        """外层 autoResize 循环:反复 solve,并在 dt 系统性偏离 dt_ref 时重采样参考。

        单轮 ``solve`` 固定采样点数 n,只优化每段 dt;由于 dt_i ≈ Δs_i / v_i,dt 的
        大小其实由采样密度 n 和速度画像共同决定,天然不会等于 dt_ref。这里在外层
        根据优化得到的总时长把 n 调成 ``round(T / dt_ref) + 1``,并用本轮的 dt/速度
        画像按预估时间重新分布新采样点(慢的地方点密、快的地方点疏,见
        ``resample_reference``),从而把平均 dt 和逐段 dt 都拉回 dt_ref 附近:
        dt 太大就加点,dt 太小就删点。用 ``dt_hysteresis`` 作死区避免抖动,最多迭代
        ``max_outer_iterations`` 轮。每轮的诊断记录放在 ``solution["resize_log"]``。

        ``record=True`` 时会额外把内层(每次 IPOPT 迭代)与外层(每轮重采样)的完整
        轨迹录进 ``solution["playback"]``,供前端逐帧回放。
        """
        max_outer = self.max_outer_iterations if max_outer_iterations is None else int(max_outer_iterations)
        max_outer = max(max_outer, 1)
        hysteresis = self.dt_hysteresis if dt_hysteresis is None else float(dt_hysteresis)
        hysteresis = max(hysteresis, 0.0)
        dt_ref = float(np.clip(reference.dt_ref, self.dt_min, self.dt_max))

        resize_log: List[Dict[str, object]] = []
        playback_rounds: List[Dict[str, object]] = []
        solution: Dict[str, object] | None = None

        for iteration in range(max_outer):
            solution = self.solve(initial_state=initial_state, reference=reference, record=record)
            dt_values = np.atleast_1d(np.asarray(solution["dt"], dtype=float))
            total_time = float(np.sum(dt_values))
            n_current = reference.size
            mean_dt = total_time / max(n_current - 1, 1)

            n_desired = (int(round(total_time / dt_ref)) + 1) if dt_ref > 0.0 else n_current
            n_new = int(np.clip(n_desired, min_samples, max_samples))

            within_band = abs(mean_dt - dt_ref) <= hysteresis
            will_resize = iteration < max_outer - 1 and not within_band and n_new != n_current

            resize_log.append(
                {
                    "iteration": int(iteration),
                    "reference_size": int(n_current),
                    "mean_dt": float(mean_dt),
                    "min_dt": float(np.min(dt_values)),
                    "max_dt": float(np.max(dt_values)),
                    "total_time": float(total_time),
                    "dt_ref": float(dt_ref),
                    "desired_size": int(n_desired),
                    "resized_to": int(n_new) if will_resize else int(n_current),
                    "resized": bool(will_resize),
                }
            )

            if record:
                playback_rounds.append(
                    {
                        "outer_iteration": int(iteration),
                        "reference_size": int(n_current),
                        "mean_dt": float(mean_dt),
                        "dt_ref": float(dt_ref),
                        "resized": bool(will_resize),
                        "resized_to": int(n_new) if will_resize else int(n_current),
                        # 本轮使用的参考线(点数会随重采样变化),供回放叠加显示。
                        "reference": {
                            "x": np.asarray(reference.x, dtype=float).tolist(),
                            "y": np.asarray(reference.y, dtype=float).tolist(),
                            "theta": np.asarray(reference.theta, dtype=float).tolist(),
                            "v": np.asarray(reference.v, dtype=float).tolist(),
                            "a": np.asarray(reference.a, dtype=float).tolist(),
                            "kappa": np.asarray(reference.kappa, dtype=float).tolist(),
                            "s": np.asarray(reference.s, dtype=float).tolist(),
                            "dt_ref": float(reference.dt_ref),
                        },
                        "frames": list(solution.get("inner_frames", [])),
                    }
                )
                solution.pop("inner_frames", None)

            if not will_resize:
                break
            reference = resample_reference(reference, n_new, time_values=solution["time"])

        assert solution is not None
        solution["resize_log"] = resize_log
        solution["resize_iterations"] = len(resize_log)
        # 最终(可能已重采样)的参考轨迹,便于调用方让展示用的参考线与优化结果点数一致。
        solution["resampled_reference"] = reference
        if record:
            solution["playback"] = {
                "recorded": True,
                "dt_ref": float(dt_ref),
                "rounds": playback_rounds,
            }
        return solution

    def solve(
        self,
        initial_state: VehicleState,
        reference: ReferenceTrajectory,
        record: bool = False,
    ) -> Dict[str, object]:
        if reference.size < 2:
            raise ValueError("Reference trajectory must contain at least 2 samples")

        n = reference.size
        dt_ref_raw = float(reference.dt_ref)
        dt_ref_used = float(np.clip(dt_ref_raw, self.dt_min, self.dt_max))
        opti = ca.Opti()

        x = opti.variable(n)
        y = opti.variable(n)
        theta = opti.variable(n)
        v = opti.variable(n)
        a = opti.variable(n)
        kappa = opti.variable(n)

        dt = opti.variable(n - 1)
        jerk = opti.variable(n - 1)
        dkappa = opti.variable(n - 1)

        cost_jerk = 0
        cost_dkappa = 0
        cost_dt_uniform = 0
        cost_time = 0

        for i in range(n - 1):
            if i > 0:
                dt_delta = dt[i] - dt[i - 1]
                cost_dt_uniform += self.w_dt_uniform * dt_delta ** 2
            cost_jerk += self.w_jerk * jerk[i] ** 2
            cost_dkappa += self.w_dkappa * dkappa[i] ** 2
            cost_time += dt[i]

        cost_accel = self.w_accel * ca.sumsqr(a)
        cost_kappa = self.w_kappa * ca.sumsqr(kappa)

        terminal_theta_ref = float(reference.theta[-1])
        terminal_dx = x[-1] - float(reference.x[-1])
        terminal_dy = y[-1] - float(reference.y[-1])
        terminal_lon_error = math.cos(terminal_theta_ref) * terminal_dx + math.sin(terminal_theta_ref) * terminal_dy
        terminal_lat_error = -math.sin(terminal_theta_ref) * terminal_dx + math.cos(terminal_theta_ref) * terminal_dy
        terminal_theta_error = ca.atan2(
            ca.sin(theta[-1] - terminal_theta_ref),
            ca.cos(theta[-1] - terminal_theta_ref),
        )
        terminal_speed_error = v[-1] - float(reference.v[-1])
        terminal_accel_error = a[-1] - float(reference.a[-1])
        terminal_lat_cost = self.w_lat_goal * terminal_lat_error ** 2
        terminal_lon_cost = self.w_lon_goal * terminal_lon_error ** 2
        terminal_theta_cost = self.w_theta_goal * terminal_theta_error ** 2
        terminal_speed_cost = self.w_speed_goal * terminal_speed_error ** 2
        terminal_accel_cost = self.w_accel_goal * terminal_accel_error ** 2
        terminal_cost = (
            terminal_lat_cost
            + terminal_lon_cost
            + terminal_theta_cost
            + terminal_speed_cost
            + terminal_accel_cost
        )
        time_cost = self.w_time * cost_time
        cost_control = cost_dt_uniform + cost_jerk + cost_dkappa + cost_accel + cost_kappa

        # cost_length 累加每段实际推进弧长 ds,与 cost_time 累加 dt 是同样的写法;
        # 需要 dynamics 循环里算出的 ds,所以 opti.minimize(total_cost) 挪到该循环之后。
        cost_length = 0

        for i in range(n - 1):
            a_next = a[i] + dt[i] * jerk[i]
            kappa_next = kappa[i] + dt[i] * dkappa[i]
            v_next = v[i] + dt[i] * a[i] + 0.5 * dt[i] ** 2 * jerk[i]

            v_mid = v[i] + 0.5 * dt[i] * a[i] + 0.125 * dt[i] ** 2 * jerk[i]
            kappa_mid = kappa[i] + 0.5 * dt[i] * dkappa[i]
            lat_accel_mid = v_mid ** 2 * kappa_mid
            theta_next = theta[i] + dt[i] * v_mid * kappa_mid
            ds = dt[i] * v_mid
            near_straight = ca.fabs(kappa_mid) < 1e-6
            x_arc = x[i] + (ca.sin(theta_next) - ca.sin(theta[i])) / kappa_mid
            y_arc = y[i] + (ca.cos(theta[i]) - ca.cos(theta_next)) / kappa_mid
            x_straight = x[i] + ds * ca.cos(theta[i])
            y_straight = y[i] + ds * ca.sin(theta[i])
            x_next = ca.if_else(near_straight, x_straight, x_arc)
            y_next = ca.if_else(near_straight, y_straight, y_arc)
            cost_length += ds

            opti.subject_to(a[i + 1] == a_next)
            opti.subject_to(kappa[i + 1] == kappa_next)
            opti.subject_to(v[i + 1] == v_next)
            opti.subject_to(theta[i + 1] == theta_next)
            opti.subject_to(x[i + 1] == x_next)
            opti.subject_to(y[i + 1] == y_next)
            opti.subject_to(opti.bounded(-self.max_lat_accel, lat_accel_mid, self.max_lat_accel))

        length_cost = self.w_length * cost_length
        total_cost = terminal_cost + cost_control + time_cost + length_cost
        opti.minimize(total_cost)

        opti.subject_to(x[0] == initial_state.x)
        opti.subject_to(y[0] == initial_state.y)
        opti.subject_to(theta[0] == initial_state.theta)
        opti.subject_to(v[0] == initial_state.v)
        opti.subject_to(a[0] == initial_state.a)
        opti.subject_to(kappa[0] == initial_state.kappa)

        opti.subject_to(opti.bounded(self.dt_min, dt, self.dt_max))
        opti.subject_to(opti.bounded(0.0, v, self.max_speed))
        opti.subject_to(opti.bounded(-self.max_accel, a, self.max_accel))
        opti.subject_to(opti.bounded(-self.max_lat_accel, v ** 2 * kappa, self.max_lat_accel))
        opti.subject_to(opti.bounded(-self.max_jerk, jerk, self.max_jerk))
        opti.subject_to(opti.bounded(-self.max_kappa, kappa, self.max_kappa))
        opti.subject_to(opti.bounded(-self.max_dkappa, dkappa, self.max_dkappa))

        opti.set_initial(x, reference.x)
        opti.set_initial(y, reference.y)
        opti.set_initial(theta, reference.theta)
        opti.set_initial(v, np.clip(reference.v, 0.0, self.max_speed))
        opti.set_initial(a, np.clip(reference.a, -self.max_accel, self.max_accel))
        opti.set_initial(kappa, np.clip(reference.kappa, -self.max_kappa, self.max_kappa))
        opti.set_initial(dt, np.full(n - 1, dt_ref_used, dtype=float))
        opti.set_initial(jerk, np.zeros(n - 1, dtype=float))
        opti.set_initial(dkappa, np.zeros(n - 1, dtype=float))

        inner_frames: List[Dict[str, list]] = []
        if record:
            # 通过 opti.callback 抓每次 IPOPT 迭代的中间迭代值,用 opti.debug.value 读当前解。
            def _capture_iterate(_iteration: int) -> None:
                try:
                    inner_frames.append(
                        {
                            "x": np.atleast_1d(np.array(opti.debug.value(x), dtype=float)).tolist(),
                            "y": np.atleast_1d(np.array(opti.debug.value(y), dtype=float)).tolist(),
                            "theta": np.atleast_1d(np.array(opti.debug.value(theta), dtype=float)).tolist(),
                            "v": np.atleast_1d(np.array(opti.debug.value(v), dtype=float)).tolist(),
                            "a": np.atleast_1d(np.array(opti.debug.value(a), dtype=float)).tolist(),
                            "kappa": np.atleast_1d(np.array(opti.debug.value(kappa), dtype=float)).tolist(),
                            "dt": np.atleast_1d(np.array(opti.debug.value(dt), dtype=float)).tolist(),
                            "jerk": np.atleast_1d(np.array(opti.debug.value(jerk), dtype=float)).tolist(),
                            "dkappa": np.atleast_1d(np.array(opti.debug.value(dkappa), dtype=float)).tolist(),
                        }
                    )
                except Exception:
                    # 求解早期某些迭代取值可能失败,跳过该帧即可,不影响最终解。
                    pass

            opti.callback(_capture_iterate)

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
            raise RuntimeError(f"TEB MPC solve failed after {elapsed_ms:.2f} ms: {exc}") from exc

        elapsed_ms = (time.time() - start_time) * 1000.0
        solver_stats = opti.stats()
        # sol.value(...) 对长度为 1 的变量(n == 2)返回标量,np.array 会得到 0 维数组;
        # 用 atleast_1d 保证后续 np.diff / rms 等运算始终作用在至少 1 维数组上。
        dt_values = np.atleast_1d(np.array(sol.value(dt), dtype=float))
        jerk_values = np.atleast_1d(np.array(sol.value(jerk), dtype=float))
        dkappa_values = np.atleast_1d(np.array(sol.value(dkappa), dtype=float))
        a_values = np.atleast_1d(np.array(sol.value(a), dtype=float))
        kappa_values = np.atleast_1d(np.array(sol.value(kappa), dtype=float))
        dt_delta_values = np.diff(dt_values)
        dt_ref_error_values = dt_values - dt_ref_used

        def rms(values: np.ndarray) -> float:
            if values.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(values))))

        cost_items = [
            {
                "key": "terminal_lat",
                "label": "terminal lateral error",
                "residual": float(sol.value(terminal_lat_error)),
                "unit": "m",
                "weight": self.w_lat_goal,
                "cost": float(sol.value(terminal_lat_cost)),
            },
            {
                "key": "terminal_lon",
                "label": "terminal longitudinal error",
                "residual": float(sol.value(terminal_lon_error)),
                "unit": "m",
                "weight": self.w_lon_goal,
                "cost": float(sol.value(terminal_lon_cost)),
            },
            {
                "key": "terminal_theta",
                "label": "terminal heading error",
                "residual": float(sol.value(terminal_theta_error)),
                "unit": "rad",
                "weight": self.w_theta_goal,
                "cost": float(sol.value(terminal_theta_cost)),
            },
            {
                "key": "terminal_speed",
                "label": "terminal speed error",
                "residual": float(sol.value(terminal_speed_error)),
                "unit": "m/s",
                "weight": self.w_speed_goal,
                "cost": float(sol.value(terminal_speed_cost)),
            },
            {
                "key": "terminal_accel",
                "label": "terminal accel error",
                "residual": float(sol.value(terminal_accel_error)),
                "unit": "m/s^2",
                "weight": self.w_accel_goal,
                "cost": float(sol.value(terminal_accel_cost)),
            },
            {
                "key": "dt_uniform",
                "label": "neighbor dt jump",
                "residual": rms(dt_delta_values),
                "unit": "s RMS",
                "weight": self.w_dt_uniform,
                "cost": float(sol.value(cost_dt_uniform)),
            },
            {
                "key": "jerk",
                "label": "jerk smoothness",
                "residual": rms(jerk_values),
                "unit": "m/s^3 RMS",
                "weight": self.w_jerk,
                "cost": float(sol.value(cost_jerk)),
            },
            {
                "key": "dkappa",
                "label": "dkappa smoothness",
                "residual": rms(dkappa_values),
                "unit": "1/(m*s) RMS",
                "weight": self.w_dkappa,
                "cost": float(sol.value(cost_dkappa)),
            },
            {
                "key": "accel",
                "label": "acceleration magnitude",
                "residual": rms(a_values),
                "unit": "m/s^2 RMS",
                "weight": self.w_accel,
                "cost": float(sol.value(cost_accel)),
            },
            {
                "key": "kappa",
                "label": "curvature magnitude",
                "residual": rms(kappa_values),
                "unit": "1/m RMS",
                "weight": self.w_kappa,
                "cost": float(sol.value(cost_kappa)),
            },
            {
                "key": "time",
                "label": "total time",
                "residual": float(np.sum(dt_values)),
                "unit": "s",
                "weight": self.w_time,
                "cost": float(sol.value(time_cost)),
            },
            {
                "key": "length",
                "label": "total path length",
                "residual": float(sol.value(cost_length)),
                "unit": "m",
                "weight": self.w_length,
                "cost": float(sol.value(length_cost)),
            },
        ]

        result = {
            "x": np.array(sol.value(x), dtype=float),
            "y": np.array(sol.value(y), dtype=float),
            "theta": np.array(sol.value(theta), dtype=float),
            "v": np.array(sol.value(v), dtype=float),
            "a": np.array(sol.value(a), dtype=float),
            "kappa": np.array(sol.value(kappa), dtype=float),
            "dt": dt_values,
            "jerk": jerk_values,
            "dkappa": dkappa_values,
            "solve_time_ms": float(elapsed_ms),
            "solver_status": str(solver_stats.get("return_status", "Solve_Succeeded")),
            "costs": {
                "control": float(sol.value(cost_control)),
                "dt_uniform": float(sol.value(cost_dt_uniform)),
                "jerk": float(sol.value(cost_jerk)),
                "dkappa": float(sol.value(cost_dkappa)),
                "accel": float(sol.value(cost_accel)),
                "kappa": float(sol.value(cost_kappa)),
                "time": float(sol.value(time_cost)),
                "length": float(sol.value(length_cost)),
                "terminal": float(sol.value(terminal_cost)),
                "terminal_lat": float(sol.value(terminal_lat_cost)),
                "terminal_lon": float(sol.value(terminal_lon_cost)),
                "terminal_theta": float(sol.value(terminal_theta_cost)),
                "terminal_speed": float(sol.value(terminal_speed_cost)),
                "terminal_accel": float(sol.value(terminal_accel_cost)),
                "total": float(sol.value(total_cost)),
                "terminal_lat_error": float(sol.value(terminal_lat_error)),
                "terminal_lon_error": float(sol.value(terminal_lon_error)),
                "terminal_theta_error": float(sol.value(terminal_theta_error)),
                "terminal_speed_error": float(sol.value(terminal_speed_error)),
                "terminal_accel_error": float(sol.value(terminal_accel_error)),
                "dt_ref_error": rms(dt_ref_error_values),
                "dt_ref_raw": dt_ref_raw,
                "dt_ref_used": dt_ref_used,
            },
            "cost_items": cost_items,
        }
        result["time"] = np.concatenate(([0.0], np.cumsum(result["dt"])))
        if record:
            result["inner_frames"] = inner_frames
        return result
