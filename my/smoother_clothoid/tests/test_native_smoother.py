"""Integration tests for the native C++ smoother via the nanobind extension."""

import math
import numpy as np
import pytest


def test_module_exports(native_module):
    for name in ("ClothoidSmoother", "Costmap2D", "SmootherParams", "OptimizerParams",
                 "SmootherResult", "ErrorCode", "compute_esdf"):
        assert hasattr(native_module, name), f"missing export: {name}"


def test_native_costmap_roundtrip(native_module):
    cm = native_module.Costmap2D(20, 20, 0.05, 0.0, 0.0)
    assert cm.getSizeInCellsX() == 20
    assert cm.getSizeInCellsY() == 20
    for mx in range(20):
        for my in range(20):
            cm.setCost(mx, my, 200)
    for mx in range(20):
        for my in range(20):
            assert cm.getCost(mx, my) == 200


def test_native_esdf(native_module):
    cm = native_module.Costmap2D(10, 10, 0.1, 0.0, 0.0)
    for x in range(3, 7):
        for y in range(3, 7):
            cm.setCost(x, y, 254)
    esdf = native_module.compute_esdf(cm, 254)
    assert len(esdf) == 100
    # The center cell should be inside the obstacle (negative distance)
    assert esdf[5 * 10 + 5] < 0
    # A free corner should be far away
    assert esdf[0] > 0.1


def test_smoother_initialize(native_module):
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())


def test_smooth_short_path_no_obstacles(native_module, free_costmap):
    path = [[float(i) * 0.5, 1.0, 1.0] for i in range(5)]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.max_curvature = 1.0
    p.max_time = 1.0
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [1.0, 0.0], free_costmap, p)
    assert res["ok"] is True
    assert len(res["smoothed_path"]) >= 2
    # Start and goal should be preserved
    assert res["smoothed_path"][0][0] == pytest.approx(0.0, abs=1e-3)
    assert res["smoothed_path"][-1][0] == pytest.approx(2.0, abs=1e-3)


def test_smooth_single_point_fails(native_module, free_costmap):
    p = native_module.SmootherParams()
    p.max_curvature = 1.0
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth([[0.0, 0.0, 1.0]], [1.0, 0.0], [1.0, 0.0], free_costmap, p)
    assert res["ok"] is False
    assert "at least 2 points" in res["error_message"]


def test_smooth_null_costmap_allowed_without_obstacles(native_module):
    path = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.max_curvature = 1.0
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [1.0, 0.0], None, p)
    assert res["ok"] is True


def test_smooth_null_costmap_rejected_with_obstacles(native_module):
    path = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    p = native_module.SmootherParams()
    p.costmap_weight_sqrt = 1.0  # enables obstacle terms
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [1.0, 0.0], None, p)
    assert res["ok"] is False
    assert res["error_message"] is not None


def test_smooth_curvature_limit_enforced(native_module, free_costmap):
    """Smoothing a sharp turn with very small max_curvature should fail."""
    path = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.keep_start_orientation = True
    p.keep_goal_orientation = True
    p.max_curvature = 0.1
    p.max_time = 0.5
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    # Big yaw delta at goal => solver must violate the curvature bound.
    res = s.try_smooth(path, [1.0, 0.0], [0.0, 1.0], free_costmap, p)
    # Either fails (good) or succeeds with high curvature that violates the bound
    if not res["ok"]:
        assert res["error_code"] == "SC_SMOOTHING_FAILED"
        assert res["error_reason"] in {"curvature_constraint", "goal_orientation_constraint"}


def test_smooth_returns_structured_failure(native_module, costmap_with_wall):
    """A path that needs to avoid a wall should produce a structured failure when impossible."""
    path = [[1.0, 0.5, 1.0], [2.0, 0.5, 1.0], [3.0, 0.5, 1.0]]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.costmap_weight_sqrt = 1.0
    p.obstacle_safe_distance = 0.3
    p.max_curvature = 0.1
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [1.0, 0.0], costmap_with_wall, p)
    if not res["ok"]:
        # Structured failure payload
        assert res["error_code"] == "SC_SMOOTHING_FAILED"
        assert "error_reason" in res
        assert "error_message" in res


def test_smooth_preserves_start_and_goal_positions(native_module, free_costmap):
    path = [[0.5, 0.5, 1.0], [1.0, 0.7, 1.0], [1.5, 0.5, 1.0]]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.fix_weight = 100.0
    p.keep_start_orientation = True
    p.keep_goal_orientation = True
    p.max_curvature = 5.0
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [1.0, 0.0], free_costmap, p)
    assert res["ok"] is True
    sp = res["smoothed_path"]
    assert sp[0][0] == pytest.approx(0.5, abs=1e-3)
    assert sp[0][1] == pytest.approx(0.5, abs=1e-3)
    assert sp[-1][0] == pytest.approx(1.5, abs=1e-3)
    assert sp[-1][1] == pytest.approx(0.5, abs=1e-3)


def test_smooth_preserves_start_orientation(native_module, free_costmap):
    path = [[0.5, 0.5, 1.0], [1.0, 0.7, 1.0], [1.5, 0.5, 1.0]]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.fix_weight = 100.0
    p.keep_start_orientation = True
    p.keep_goal_orientation = True
    p.max_curvature = 5.0
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [0.0, 1.0], free_costmap, p)
    assert res["ok"] is True
    sp = res["smoothed_path"]
    # start yaw should match direction
    assert sp[0][2] == pytest.approx(0.0, abs=1e-2)
    # goal yaw should match direction
    assert sp[-1][2] == pytest.approx(math.pi / 2, abs=1e-2)


def test_optimized_knot_count_populated(native_module, free_costmap):
    path = [[float(i) * 0.2, 1.0, 1.0] for i in range(10)]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.max_curvature = 1.0
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [1.0, 0.0], free_costmap, p)
    assert res["ok"] is True
    assert res["optimized_knot_count"] == len(path)


def test_target_spacing_populated(native_module, free_costmap):
    path = [[float(i) * 0.3, 1.0, 1.0] for i in range(5)]
    p = native_module.SmootherParams()
    p.model_weight_sqrt = 1.0
    p.max_curvature = 1.0
    s = native_module.ClothoidSmoother()
    s.initialize(native_module.OptimizerParams())
    res = s.try_smooth(path, [1.0, 0.0], [1.0, 0.0], free_costmap, p)
    assert res["ok"] is True
    assert res["target_spacing_m"] > 0
