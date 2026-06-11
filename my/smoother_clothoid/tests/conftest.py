"""Shared pytest fixtures for smoother_clothoid tests."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_BUILD_PY313 = os.path.join(_REPO_ROOT, "build-py313")
_BUILD = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_PY313, _BUILD):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def native_module():
    try:
        import nb_smoother_clothoid as sc
    except ImportError as e:
        pytest.skip(f"Native smoother module not built: {e}")
    return sc


@pytest.fixture
def free_costmap(native_module):
    cm = native_module.Costmap2D(100, 100, 0.05, 0.0, 0.0)
    return cm


@pytest.fixture
def costmap_with_wall(native_module):
    cm = native_module.Costmap2D(100, 100, 0.05, 0.0, 0.0)
    for y in range(35, 65):
        cm.setCost(50, y, native_module.Costmap2D.LETHAL_OBSTACLE)
    return cm


@pytest.fixture
def straight_path():
    return [np.array([0.5 + i * 0.2, 2.5, 1.0], dtype=np.float64) for i in range(15)]


@pytest.fixture
def curved_path():
    pts = []
    for i in range(20):
        t = i * 0.2
        pts.append(np.array([2.0 + t, 2.0 + 0.3 * np.sin(t * 1.5), 1.0], dtype=np.float64))
    return pts


@pytest.fixture
def cusp_path():
    pts = []
    sp = 0.2
    for x in np.arange(1.0, 6.0 + 1e-9, sp):
        pts.append(np.array([x, 2.0, 1.0], dtype=np.float64))
    for x in np.arange(6.0 - sp, 1.4 - 1e-9, -sp):
        pts.append(np.array([x, 2.0, -1.0], dtype=np.float64))
    return pts
