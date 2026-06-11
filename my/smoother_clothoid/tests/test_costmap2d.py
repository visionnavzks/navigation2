"""Tests for Costmap2D wrapper."""

import math
import numpy as np
import pytest

from smoother_clothoid_py.costmap2d import Costmap2D


def test_constants():
    assert Costmap2D.NO_INFORMATION == 255
    assert Costmap2D.LETHAL_OBSTACLE == 254
    assert Costmap2D.INSCRIBED_INFLATED_OBSTACLE == 253
    assert Costmap2D.FREE_SPACE == 0


def test_default_zeros():
    cm = Costmap2D(5, 7, 0.1, 1.0, 2.0)
    assert cm.size_x == 5
    assert cm.size_y == 7
    assert cm.resolution == pytest.approx(0.1)
    assert cm.origin_x == pytest.approx(1.0)
    assert cm.origin_y == pytest.approx(2.0)
    assert cm.data.shape == (7, 5)
    assert np.all(cm.data == 0)


def test_set_get_roundtrip():
    cm = Costmap2D(20, 20, 0.05)
    for mx in range(20):
        for my in range(20):
            cm.set_cost(mx, my, 200)
    for mx in range(20):
        for my in range(20):
            assert cm.get_cost(mx, my) == 200


def test_set_cost_clipped_to_uint8():
    cm = Costmap2D(5, 5, 0.1)
    cm.set_cost(0, 0, 300)
    assert cm.get_cost(0, 0) == 255


def test_set_cost_negative_clipped_to_zero():
    cm = Costmap2D(5, 5, 0.1)
    cm.set_cost(0, 0, -1)
    assert cm.get_cost(0, 0) == 0


def test_get_char_map_returns_data():
    cm = Costmap2D(3, 3, 0.1)
    cm.set_cost(1, 1, 50)
    arr = cm.get_char_map()
    assert isinstance(arr, np.ndarray)
    assert arr[1, 1] == 50
