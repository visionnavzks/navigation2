"""Tests for the ESDF computation."""

import math
import numpy as np
import pytest

from smoother_clothoid_py.esdf import compute_esdf, compute_approximate_esdf, ESDFAlgorithm
from smoother_clothoid_py.exceptions import InvalidCostmap
from smoother_clothoid_py.costmap2d import Costmap2D


def test_algorithm_enum_values():
    assert ESDFAlgorithm.Exact.value == "exact"
    assert ESDFAlgorithm.Approximate.value == "approximate"


def test_compute_esdf_rejects_null():
    with pytest.raises(InvalidCostmap):
        compute_esdf(None)


def test_empty_costmap_all_inf():
    cm = Costmap2D(5, 5, 0.1, 0.0, 0.0)
    esdf = compute_esdf(cm)
    assert len(esdf) == 25
    assert all(math.isinf(v) for v in esdf)


def test_single_obstacle_signed_distance_at_source():
    """For a single lethal cell, the signed distance is -resolution (you are
    0.1m into the obstacle from the nearest free cell), not 0."""
    cm = Costmap2D(5, 5, 0.1, 0.0, 0.0)
    cm.set_cost(2, 2, Costmap2D.LETHAL_OBSTACLE)
    esdf = compute_esdf(cm)
    # The obstacle cell itself has a negative signed distance (= -distance to nearest free cell)
    assert esdf[2 * 5 + 2] == pytest.approx(-0.1, abs=1e-9)
    # Free cells have positive distance to nearest obstacle
    assert esdf[2 * 5 + 1] == pytest.approx(0.1, abs=1e-9)
    assert esdf[2 * 5 + 0] == pytest.approx(0.2, abs=1e-9)
    assert esdf[0] == pytest.approx(math.hypot(0.2, 0.2), abs=1e-9)


def test_diagonal_neighbors_use_sqrt2():
    cm = Costmap2D(5, 5, 0.1, 0.0, 0.0)
    cm.set_cost(2, 2, Costmap2D.LETHAL_OBSTACLE)
    esdf = compute_esdf(cm)
    assert esdf[1 * 5 + 1] == pytest.approx(0.1 * math.sqrt(2), abs=1e-9)


def test_inside_obstacle_is_negative():
    cm = Costmap2D(5, 5, 0.1, 0.0, 0.0)
    cm.set_cost(2, 2, Costmap2D.LETHAL_OBSTACLE)
    esdf = compute_esdf(cm)
    assert esdf[2 * 5 + 2] < 0.0


def test_blocked_2x2_picks_max():
    cm = Costmap2D(5, 5, 0.1, 0.0, 0.0)
    cm.set_cost(2, 2, Costmap2D.LETHAL_OBSTACLE)
    cm.set_cost(2, 3, Costmap2D.LETHAL_OBSTACLE)
    esdf = compute_esdf(cm)
    # a cell adjacent to the 2-cell cluster
    assert esdf[1 * 5 + 2] == pytest.approx(0.1, abs=1e-9)


def test_resolution_scales_esdf():
    cm1 = Costmap2D(5, 5, 0.05, 0.0, 0.0)
    cm2 = Costmap2D(5, 5, 0.10, 0.0, 0.0)
    cm1.set_cost(2, 2, Costmap2D.LETHAL_OBSTACLE)
    cm2.set_cost(2, 2, Costmap2D.LETHAL_OBSTACLE)
    e1 = compute_esdf(cm1)
    e2 = compute_esdf(cm2)
    # Cell 1 step to the right should be exactly 0.05 vs 0.10
    assert e1[2 * 5 + 3] == pytest.approx(0.05, abs=1e-9)
    assert e2[2 * 5 + 3] == pytest.approx(0.10, abs=1e-9)


def test_compute_approximate_matches_compute():
    cm = Costmap2D(7, 7, 0.1)
    cm.set_cost(3, 3, Costmap2D.LETHAL_OBSTACLE)
    e1 = compute_esdf(cm)
    e2 = compute_approximate_esdf(cm)
    assert len(e1) == len(e2)
    for a, b in zip(e1, e2):
        assert a == pytest.approx(b, abs=1e-9)


def test_lethal_threshold_is_inclusive():
    cm = Costmap2D(5, 5, 0.1, 0.0, 0.0)
    cm.set_cost(2, 2, 254)
    cm.set_cost(2, 3, 253)
    esdf = compute_esdf(cm, lethal=253)
    # Both 254 and 253 are treated as obstacles when threshold is 253
    # (signed distance is negative inside, with magnitude equal to distance to nearest free cell)
    assert esdf[2 * 5 + 2] < 0
    assert esdf[3 * 5 + 2] < 0
