"""Tests for SmootherRequest / SmootherResult dataclasses."""

import numpy as np
import pytest

from smoother_clothoid_py.smoother_request import SmootherRequest, SmootherResult


def test_smoother_result_defaults():
    r = SmootherResult()
    assert r.candidate_path == []
    assert r.smoothed_path == []
    assert r.optimized_knot_count == 0
    assert r.target_spacing == 0.0
    assert r.success is False


def test_smoother_result_mutable():
    r = SmootherResult()
    r.success = True
    r.smoothed_path = [np.array([0.0, 0.0, 0.0])]
    r.optimized_knot_count = 1
    assert r.success is True
    assert len(r.smoothed_path) == 1


def test_smoother_request_basic():
    req = SmootherRequest(
        path=[np.array([0.0, 0.0, 1.0])],
        start_dir=np.array([1.0, 0.0]),
        end_dir=np.array([1.0, 0.0]),
        costmap=None,
        params=None,
    )
    assert len(req.path) == 1
    assert req.precomputed_esdf is None
    assert req.failure is None
    assert req.costmap is None
    assert req.params is None


def test_smoother_request_with_optional_fields():
    from smoother_clothoid_py.options import SmootherParams
    from smoother_clothoid_py.exceptions import SmoothingFailureInfo

    params = SmootherParams()
    failure = SmoothingFailureInfo()
    req = SmootherRequest(
        path=[],
        start_dir=np.array([1.0, 0.0]),
        end_dir=np.array([1.0, 0.0]),
        costmap=None,
        params=params,
        precomputed_esdf=[1.0, 2.0, 3.0],
        failure=failure,
    )
    assert req.precomputed_esdf == [1.0, 2.0, 3.0]
    assert req.failure is failure
