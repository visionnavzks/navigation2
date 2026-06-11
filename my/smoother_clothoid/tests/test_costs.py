"""Tests for cost residual functions."""

import math
import numpy as np
import pytest

from smoother_clothoid_py.costs import (
    transition_residuals, boundary_residuals, reference_residuals, obstacle_residuals,
)
from smoother_clothoid_py.utils import PI


# ----- transition_residuals -----

def test_transition_residuals_cusp_path():
    """Cusp: only x, y, theta, ds and sw*ds residuals active."""
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    nxt = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
    r = transition_residuals(current, nxt, 0.0, True, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.2)
    assert r.shape == (7,)
    assert r[0] == pytest.approx(10.0 * 0.1)
    assert r[1] == pytest.approx(0.0)
    assert r[2] == pytest.approx(0.0)
    assert r[3] == pytest.approx(0.0)
    assert r[4] == pytest.approx(0.0)
    assert r[5] == pytest.approx(10.0 * 0.1)
    assert r[6] == pytest.approx(0.0)


def test_transition_residuals_cusp_yaw_diff():
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    nxt = np.array([0.0, 0.0, PI, 0.0, 0.0])
    r = transition_residuals(current, nxt, 0.0, True, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.2)
    assert r[2] == pytest.approx(5.0 * PI)


def test_transition_residuals_straight_line():
    current = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    nxt = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    mw = 1.0
    r = transition_residuals(current, nxt, 1.0, False, mw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2)
    assert r[0] == pytest.approx(0.0, abs=1e-9)
    assert r[1] == pytest.approx(0.0, abs=1e-9)
    assert r[2] == pytest.approx(0.0, abs=1e-9)


def test_transition_residuals_reversing():
    current = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    nxt = np.array([-1.0, 0.0, PI, 0.0, 0.0])
    r = transition_residuals(current, nxt, -1.0, False, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2)
    # driving backwards 1m, the position residual should be near zero
    assert r[0] == pytest.approx(0.0, abs=1e-9)


def test_transition_residuals_curvature_cost():
    current = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    nxt = np.array([1.0, 0.5, 0.5, 1.0, 0.0])
    cw = 2.0
    r = transition_residuals(current, nxt, 1.0, False, 0.0, cw, 0.0, 0.0, 0.0, 0.0, 0.2)
    assert r[3] == pytest.approx(cw * (1.0 + 1.0) * 0.5)


def test_transition_residuals_curvature_rate_cost():
    current = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    nxt = np.array([0.1, 0.0, 0.0, 1.5, 0.0])
    crw = 4.0
    r = transition_residuals(current, nxt, 1.0, False, 0.0, 0.0, crw, 0.0, 0.0, 0.0, 0.2)
    assert r[4] == pytest.approx(crw * (1.5 - 0.5) / math.sqrt(0.1))


def test_transition_residuals_spacing_cost():
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
    nxt = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    sw = 2.0
    target = 0.2
    r = transition_residuals(current, nxt, 1.0, False, 0.0, 0.0, 0.0, sw, 0.0, 0.0, target)
    assert r[5] == pytest.approx(sw * (0.5 - target) / target)


def test_transition_residuals_path_length_cost():
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.7])
    nxt = np.array([0.7, 0.0, 0.0, 0.0, 0.0])
    lw = 0.5
    r = transition_residuals(current, nxt, 1.0, False, 0.0, 0.0, 0.0, 0.0, lw, 0.0, 0.2)
    assert r[6] == pytest.approx(lw * 0.7)


def test_transition_residuals_small_ds_handled():
    """With ds < 1e-3, the sqrt denominator falls back to 0.03."""
    current = np.array([0.0, 0.0, 0.0, 0.0, 1e-5])
    nxt = np.array([1e-5, 0.0, 0.0, 1.0, 0.0])
    r = transition_residuals(current, nxt, 1.0, False, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2)
    assert math.isfinite(r[4])


# ----- boundary_residuals -----

def test_boundary_residuals_inside_tolerance():
    state = np.array([0.05, 0.05, PI / 2, 0.0, 0.1])
    ref = np.array([0.0, 0.0])
    r = boundary_residuals(state, ref, PI / 2, True, 0.2, 0.2, 0.1, 10.0, False)
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.0)
    assert r[2] == pytest.approx(0.0)
    assert r[3] == pytest.approx(0.0)


def test_boundary_residuals_outside_tolerance():
    state = np.array([0.3, 0.0, 0.0, 0.0, 0.1])
    ref = np.array([0.0, 0.0])
    r = boundary_residuals(state, ref, 0.0, False, 0.1, 0.1, 0.1, 10.0, True)
    assert r[0] == pytest.approx(10.0 * 0.2)
    assert r[1] == pytest.approx(0.0)
    assert r[3] == pytest.approx(10.0 * 0.1)


def test_boundary_residuals_lateral_uses_orthogonal_frame():
    state = np.array([0.0, 0.3, 0.0, 0.0, 0.1])
    ref = np.array([0.0, 0.0])
    r = boundary_residuals(state, ref, 0.0, False, 0.1, 0.1, 0.1, 10.0, True)
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(10.0 * 0.2)


def test_boundary_residuals_orientation_off():
    state = np.array([0.0, 0.0, 0.5, 0.0, 0.1])
    ref = np.array([0.0, 0.0])
    r = boundary_residuals(state, ref, 0.0, True, 0.1, 0.1, 0.1, 5.0, False)
    assert r[2] == pytest.approx(5.0 * (0.5 - 0.1))


def test_boundary_residuals_no_orientation_when_not_kept():
    state = np.array([0.0, 0.0, 0.5, 0.0, 0.1])
    ref = np.array([0.0, 0.0])
    r = boundary_residuals(state, ref, 0.0, False, 0.1, 0.1, 0.1, 5.0, False)
    assert r[2] == pytest.approx(0.0)


def test_boundary_residuals_constrain_stop_zero_ds():
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
    ref = np.array([0.0, 0.0])
    r = boundary_residuals(state, ref, 0.0, False, 0.1, 0.1, 0.1, 5.0, True)
    assert r[3] == pytest.approx(5.0 * 0.5)


def test_boundary_residuals_no_constrain_stop_keeps_zero():
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
    ref = np.array([0.0, 0.0])
    r = boundary_residuals(state, ref, 0.0, False, 0.1, 0.1, 0.1, 5.0, False)
    assert r[3] == pytest.approx(0.0)


# ----- reference_residuals -----

def test_reference_residuals_zero_when_aligned():
    state = np.array([0.5, 0.5, 0.0, 0.0, 0.1])
    ref = np.array([0.5, 0.5])
    r = reference_residuals(state, ref, 1.0)
    assert r.shape == (2,)
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.0)


def test_reference_residuals_weighted():
    state = np.array([0.6, 0.4, 0.0, 0.0, 0.1])
    ref = np.array([0.5, 0.5])
    r = reference_residuals(state, ref, 3.0)
    assert r[0] == pytest.approx(0.3)
    assert r[1] == pytest.approx(-0.3)


# ----- obstacle_residuals -----

def test_obstacle_residuals_far_from_obstacle():
    state = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    esdf = [1.0] * 100  # far
    r = obstacle_residuals(state, esdf, 10, 10, 0.0, 0.0, 1.0, 0.3, 0.0, 1.0, 1.0, False)
    assert r.shape == (1,)
    assert r[0] == pytest.approx(0.0)


def test_obstacle_residuals_inside_obstacle():
    state = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    esdf = [-0.5] * 100
    r = obstacle_residuals(state, esdf, 10, 10, 0.0, 0.0, 1.0, 0.3, 0.0, 1.0, 1.0, False)
    assert r[0] > 0.0


def test_obstacle_residuals_cusp_weight():
    state = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    esdf = [-0.5] * 100
    r_norm = obstacle_residuals(state, esdf, 10, 10, 0.0, 0.0, 1.0, 0.3, 0.0, 1.0, 1.0, False)
    r_cusp = obstacle_residuals(state, esdf, 10, 10, 0.0, 0.0, 1.0, 0.3, 0.0, 1.0, 3.0, True)
    assert r_cusp[0] == pytest.approx(3.0 * r_norm[0])


def test_obstacle_residuals_out_of_bounds():
    state = np.array([-5.0, 5.0, 0.0, 0.0, 0.0])
    esdf = [0.0] * 100
    r = obstacle_residuals(state, esdf, 10, 10, 0.0, 0.0, 1.0, 0.3, 0.0, 1.0, 1.0, False)
    assert r[0] == pytest.approx(1.0)


def test_obstacle_residuals_with_check_points():
    state = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    esdf = [1.0] * 100
    # 2 footprint points, both well clear
    pts = [0.0, 0.0, 1.0, 0.5, 0.0, 0.5]
    r = obstacle_residuals(state, esdf, 10, 10, 0.0, 0.0, 1.0, 0.3, 0.0, 1.0, 1.0, False, pts)
    assert r.shape == (2,)
    assert all(v == pytest.approx(0.0) for v in r)
