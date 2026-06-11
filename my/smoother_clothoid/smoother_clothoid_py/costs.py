"""Cost residual functions."""

from __future__ import annotations
import math
from typing import Optional

import numpy as np

from smoother_clothoid_py.utils import normalize_angle, angle_diff, EPSILON, PI


def transition_residuals(
    current: np.ndarray, next_state: np.ndarray,
    gear: float, is_cusp: bool,
    mw: float, cw: float, crw: float, sw: float, lw: float, fw: float,
    target_spacing: float,
) -> np.ndarray:
    r = np.zeros(7)
    if is_cusp:
        r[0] = fw * (next_state[0] - current[0])
        r[1] = fw * (next_state[1] - current[1])
        r[2] = fw * angle_diff(next_state[2], current[2])
        r[5] = sw * 10.0 * current[4]
        r[6] = lw * current[4]
        return r

    x, y, theta, kappa, ds = current
    nx, ny, nt, nk = next_state[:4]
    d = 1.0 if gear >= 0.0 else -1.0
    tp = theta + d * ds * (kappa + nk) * 0.5
    tm = theta + d * ds * kappa * 0.5
    xp = x + d * ds * math.cos(tm)
    yp = y + d * ds * math.sin(tm)
    denom = math.sqrt(ds) if ds > 1e-3 else 0.03

    r[0] = mw * (nx - xp)
    r[1] = mw * (ny - yp)
    r[2] = mw * angle_diff(nt, tp)
    r[3] = cw * (kappa + nk) * 0.5
    r[4] = crw * (nk - kappa) / denom
    sr = max(target_spacing, 1e-3)
    r[5] = sw * (ds - sr) / sr
    r[6] = lw * ds
    return r


def boundary_residuals(
    state: np.ndarray, ref: np.ndarray, target_theta: float,
    keep_ori: bool, lon_tol: float, lat_tol: float, ori_tol: float,
    fw: float, constrain_stop: bool,
) -> np.ndarray:
    dx, dy = state[0] - ref[0], state[1] - ref[1]
    ct, st = math.cos(target_theta), math.sin(target_theta)
    lon = ct * dx + st * dy
    lat = -st * dx + ct * dy
    lv = abs(lon) - max(lon_tol, 0.0)
    bv = abs(lat) - max(lat_tol, 0.0)
    r = np.zeros(4)
    r[0] = fw * lv if lv > 0 else 0.0
    r[1] = fw * bv if bv > 0 else 0.0
    if keep_ori:
        he = abs(angle_diff(state[2], target_theta))
        hv = he - max(ori_tol, 0.0)
        r[2] = fw * hv if hv > 0 else 0.0
    r[3] = fw * state[4] if constrain_stop else 0.0
    return r


def reference_residuals(state: np.ndarray, ref: np.ndarray, w: float) -> np.ndarray:
    return np.array([w * (state[0] - ref[0]), w * (state[1] - ref[1])])


def obstacle_residuals(
    state: np.ndarray, esdf: list[float],
    sx: int, sy: int, ox: float, oy: float, res: float,
    safe_dist: float, check_r: float,
    obs_w: float, cusp_w: float, is_cusp: bool,
    pts: Optional[list[float]] = None,
) -> np.ndarray:
    x, y, theta = state[0], state[1], state[2]
    pw = cusp_w if is_cusp else obs_w

    def penalty(wx: float, wy: float) -> float:
        gx = (wx - ox) / res
        gy = (wy - oy) / res
        if gx < 1.5 or gy < 1.5 or gx >= sx - 1.5 or gy >= sy - 1.5:
            return 1.0
        gxp, gyp = gx - 0.5, gy - 0.5
        ix, iy = int(math.floor(gxp)), int(math.floor(gyp))
        fx, fy = gxp - ix, gyp - iy
        def v(r, c):
            if 0 <= c < sx and 0 <= r < sy:
                idx = r * sx + c
                return esdf[idx] if 0 <= idx < len(esdf) else float("inf")
            return float("inf")
        dist = v(iy,ix)*(1-fx)*(1-fy) + v(iy+1,ix)*fx*(1-fy) + v(iy,ix+1)*(1-fx)*fy + v(iy+1,ix+1)*fx*fy
        sd = dist - check_r
        sd_safe = max(safe_dist, 1e-6)
        if sd >= sd_safe: return 0.0
        g = (sd_safe - sd) / sd_safe
        return g * g

    if not pts:
        return np.array([pw * penalty(x, y)])
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        pw * pts[i+2] * penalty(x + ct*pts[i] - st*pts[i+1], y + st*pts[i] + ct*pts[i+1])
        for i in range(0, len(pts) - 2, 3)
    ])
