"""Problem builder for the clothoid smoother."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from smoother_clothoid_py.options import SmootherParams
from smoother_clothoid_py.costmap2d import Costmap2D
from smoother_clothoid_py.exceptions import PrecomputedEsdfSizeMismatch
from smoother_clothoid_py.esdf import compute_esdf
from smoother_clothoid_py.costs import (
    transition_residuals, boundary_residuals, reference_residuals, obstacle_residuals,
)
from smoother_clothoid_py.utils import normalize_angle, angle_diff, goal_position_frame_heading, EPSILON, PI


@dataclass
class ProcessedPath:
    reference_points: list[tuple[float, float]] = field(default_factory=list)
    gears: list[float] = field(default_factory=list)
    is_cusp_segment: list[bool] = field(default_factory=list)
    initial_variables: list[float] = field(default_factory=list)
    state_count: int = 0
    start_theta: float = 0.0
    end_theta: float = 0.0
    target_spacing: float = 0.2


class ProblemBuilder:
    def __init__(self, esdf_values: list[float]) -> None:
        self._esdf = esdf_values
        self._costmap: Optional[Costmap2D] = None
        self._params: Optional[SmootherParams] = None

    def initialize_esdf_values(self, costmap: Optional[Costmap2D], params: SmootherParams,
                               precomputed: Optional[list[float]] = None) -> None:
        if not params.obstacle_terms_enabled():
            self._esdf.clear()
            return
        if costmap is None:
            return
        expected = costmap.size_x * costmap.size_y
        if precomputed is not None:
            if len(precomputed) != expected:
                raise PrecomputedEsdfSizeMismatch("ESDF size mismatch")
            self._esdf = list(precomputed)
        else:
            self._esdf = compute_esdf(costmap, Costmap2D.LETHAL_OBSTACLE)
        self._costmap = costmap
        self._params = params

    @staticmethod
    def build_processed_path(
        path: list[np.ndarray], start_dir: np.ndarray, end_dir: np.ndarray,
        params: SmootherParams, costmap: Optional[Costmap2D] = None,
    ) -> ProcessedPath:
        p = ProcessedPath()
        p.start_theta = math.atan2(start_dir[1], start_dir[0])
        p.end_theta = math.atan2(end_dir[1], end_dir[0])
        sampled = _downsample(path, params)

        gears = [(-1.0 if sampled[i][2] < 0 else 1.0) if params.reversing_enabled else 1.0
                 for i in range(len(sampled) - 1)]

        p.reference_points.append((sampled[0][0], sampled[0][1]))
        for i in range(len(sampled) - 1):
            cg = gears[i]
            ng = gears[i + 1] if i + 1 < len(gears) else cg
            p.gears.append(cg)
            p.is_cusp_segment.append(False)
            p.reference_points.append((sampled[i + 1][0], sampled[i + 1][1]))
            if i + 2 < len(sampled) and cg != ng:
                p.gears.append(0.0)
                p.is_cusp_segment.append(True)
                p.reference_points.append((sampled[i + 1][0], sampled[i + 1][1]))

        p.state_count = len(p.reference_points)
        theta = [0.0] * p.state_count
        ds = [0.0] * p.state_count
        sp_sum, sp_cnt = 0.0, 0

        for i in range(p.state_count - 1):
            rx = p.reference_points[i + 1][0] - p.reference_points[i][0]
            ry = p.reference_points[i + 1][1] - p.reference_points[i][1]
            norm = math.hypot(rx, ry)
            if p.is_cusp_segment[i]:
                theta[i] = theta[i - 1] if i > 0 else p.start_theta
                continue
            if norm > 1e-6:
                h = math.atan2(ry, rx)
                if p.gears[i] < 0: h += PI
                theta[i] = normalize_angle(h)
                ds[i] = norm
                sp_sum += norm; sp_cnt += 1
            else:
                theta[i] = theta[i - 1] if i > 0 else p.start_theta

        theta[-1] = theta[-2] if len(theta) > 1 else p.start_theta
        if params.keep_start_orientation: theta[0] = p.start_theta
        if params.keep_goal_orientation: theta[-1] = p.end_theta
        p.target_spacing = sp_sum / sp_cnt if sp_cnt > 0 else (max(costmap.resolution, 1e-3) if costmap else 0.2)

        p.initial_variables = []
        for i in range(p.state_count):
            p.initial_variables.extend([p.reference_points[i][0], p.reference_points[i][1],
                                        theta[i], 0.0, ds[i]])
        return p

    def build_residual_fn(self, proc: ProcessedPath, costmap: Optional[Costmap2D],
                          params: SmootherParams) -> tuple[callable, int]:
        n = proc.state_count
        mw = max(params.model_weight_sqrt, 0.0)
        cw = max(params.kinematic_curvature_weight_sqrt, 0.0)
        crw = max(params.kinematic_curvature_rate_weight_sqrt, 0.0)
        sw = max(params.kinematic_spacing_weight_sqrt, 0.0)
        lw = max(params.path_length_weight_sqrt, 0.0)
        fw = max(params.fix_weight, 0.0)
        rw = max(params.reference_path_weight_sqrt, 0.0)
        has_obs = params.obstacle_terms_enabled()
        ow = max(params.costmap_weight_sqrt, 0.0)
        cusp_w = max(params.cusp_costmap_weight_sqrt, params.costmap_weight_sqrt)
        gth = goal_position_frame_heading(proc.reference_points, proc.end_theta, params.keep_goal_orientation)

        def fn(variables: np.ndarray) -> np.ndarray:
            res = []
            for i in range(n - 1):
                res.extend(transition_residuals(
                    variables[5*i:5*i+5], variables[5*(i+1):5*(i+1)+5],
                    proc.gears[i], proc.is_cusp_segment[i],
                    mw, cw, crw, sw, lw, fw, proc.target_spacing))
            res.extend(boundary_residuals(variables[0:5], np.array(proc.reference_points[0]),
                proc.start_theta, params.keep_start_orientation, 0, 0, 0, fw, False))
            res.extend(boundary_residuals(variables[5*(n-1):5*(n-1)+5], np.array(proc.reference_points[-1]),
                gth, params.keep_goal_orientation, params.goal_longitudinal_tolerance,
                params.goal_lateral_tolerance, params.goal_orientation_tolerance, fw, True))
            if rw > 1e-9:
                for i in range(n):
                    res.extend(reference_residuals(variables[5*i:5*i+5], np.array(proc.reference_points[i]), rw))
            if has_obs and costmap:
                for i in range(n):
                    is_cusp = (i < len(proc.is_cusp_segment) and proc.is_cusp_segment[i]) or (i > 0 and proc.is_cusp_segment[i-1])
                    res.extend(obstacle_residuals(variables[5*i:5*i+5], self._esdf,
                        costmap.size_x, costmap.size_y, costmap.origin_x, costmap.origin_y,
                        costmap.resolution, params.obstacle_safe_distance, params.cost_check_radius,
                        ow, cusp_w, is_cusp, params.cost_check_points or None))
            return np.array(res, dtype=np.float64)

        return fn, n * 5

    @staticmethod
    def apply_bounds(lower: np.ndarray, upper: np.ndarray, refs: list, n: int,
                     mc: float, ms: float, md: float) -> None:
        mc = max(mc, 1e-6)
        for i in range(n):
            b = 5 * i
            if md > 1e-9:
                lower[b] = refs[i][0] - md; upper[b] = refs[i][0] + md
                lower[b+1] = refs[i][1] - md; upper[b+1] = refs[i][1] + md
            lower[b+3] = -mc; upper[b+3] = mc
            lower[b+4] = 0.0
            if ms > 1e-9: upper[b+4] = ms

    @staticmethod
    def unpack_path(vars: np.ndarray, n: int) -> list[np.ndarray]:
        return [np.array([vars[5*i], vars[5*i+1], normalize_angle(vars[5*i+2])]) for i in range(n)]

    @staticmethod
    def upsample_path(vars: np.ndarray, proc: ProcessedPath, params: SmootherParams) -> list[np.ndarray]:
        f = max(params.path_upsampling_factor, 1)
        path = ProblemBuilder.unpack_path(vars, proc.state_count)
        if f <= 1 or proc.state_count < 2: return path

        up = [path[0]]
        for i in range(proc.state_count - 1):
            cusp = i < len(proc.is_cusp_segment) and proc.is_cusp_segment[i]
            gear = proc.gears[i] if i < len(proc.gears) else 1.0
            x, y, th = vars[5*i], vars[5*i+1], normalize_angle(vars[5*i+2])
            kap, ds = vars[5*i+3], max(vars[5*i+4], 0.0)
            nk = vars[5*(i+1)+3]
            nxt = path[i + 1]
            if cusp or abs(gear) < 1e-9 or ds <= 1e-6:
                up.append(nxt); continue
            d = 1.0 if gear >= 0 else -1.0
            step = ds / f
            ix, iy, it = x, y, th
            seg = []
            for j in range(1, f):
                t0, t1 = (j-1)/f, j/f
                k0, k1 = kap + (nk-kap)*t0, kap + (nk-kap)*t1
                tm = it + d*step*0.5*k0
                ix += d*step*math.cos(tm); iy += d*step*math.sin(tm)
                it = normalize_angle(it + d*step*0.5*(k0+k1))
                seg.append(np.array([ix, iy, it]))
            ft0 = (f-1)/f; fk0 = kap + (nk-kap)*ft0
            ftm = it + d*step*0.5*fk0
            px, py = ix + d*step*math.cos(ftm), iy + d*step*math.sin(ftm)
            pt = normalize_angle(it + d*step*0.5*(fk0+nk))
            cx, cy, ct = nxt[0]-px, nxt[1]-py, normalize_angle(nxt[2]-pt)
            for j in range(1, f):
                t = j/f; s = seg[j-1]
                up.append(np.array([s[0]+t*cx, s[1]+t*cy, normalize_angle(s[2]+t*ct)]))
            up.append(nxt)
        return up


def _downsample(path: list[np.ndarray], params: SmootherParams) -> list[np.ndarray]:
    f = max(params.path_downsampling_factor, 1)
    if f <= 1 or len(path) <= 2: return list(path)
    s = [path[0]]
    last = 0
    def ds(i): return (-1.0 if path[i][2] < 0 else 1.0) if params.reversing_enabled else 1.0
    for i in range(1, len(path) - 1):
        cusp = ds(i) != ds(i-1) or ds(i) != ds(i+1)
        if cusp or (i - last) >= f:
            s.append(path[i]); last = i
    if not np.allclose(s[-1], path[-1], atol=1e-9): s.append(path[-1])
    if len(s) < 2: s = [path[0], path[-1]]
    return s
