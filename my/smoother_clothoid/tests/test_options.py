"""Tests for options / dataclasses."""

import math
import pytest

from smoother_clothoid_py.options import SmootherParams, OptimizerParams, LinearSolver


def test_smoother_params_defaults():
    p = SmootherParams()
    assert p.model_weight_sqrt == 0.0
    assert p.costmap_weight_sqrt == 0.0
    assert p.cusp_costmap_weight_sqrt == 0.0
    assert p.cusp_zone_length == 0.0
    assert p.reference_path_weight_sqrt == 0.0
    assert p.reference_point_max_deviation_m == 0.0
    assert p.kinematic_curvature_weight_sqrt == 0.0
    assert p.kinematic_curvature_rate_weight_sqrt == 0.0
    assert p.kinematic_spacing_weight_sqrt == 1.0
    assert p.kinematic_max_spacing == 0.0
    assert p.path_length_weight_sqrt == 0.0
    assert p.fix_weight == 100.0
    assert p.max_curvature == 0.0
    assert p.max_time == 10.0
    assert p.use_exact_esdf is True
    assert p.obstacle_safe_distance == 0.5
    assert p.cost_check_radius == 0.0
    assert p.cost_check_points == []
    assert p.path_downsampling_factor == 1
    assert p.path_upsampling_factor == 1
    assert p.reversing_enabled is True
    assert p.goal_longitudinal_tolerance == 0.0
    assert p.goal_lateral_tolerance == 0.0
    assert p.goal_orientation_tolerance == 0.0
    assert p.keep_goal_orientation is True
    assert p.keep_start_orientation is True


def test_smoother_params_obstacle_terms_disabled_by_default():
    assert SmootherParams().obstacle_terms_enabled() is False


def test_smoother_params_obstacle_terms_via_costmap_weight():
    assert SmootherParams(costmap_weight_sqrt=1.0).obstacle_terms_enabled() is True


def test_smoother_params_obstacle_terms_via_cusp_weight():
    assert SmootherParams(cusp_costmap_weight_sqrt=0.1).obstacle_terms_enabled() is True


def test_smoother_params_obstacle_terms_threshold():
    # Threshold is 1e-9; weights below are treated as zero
    assert SmootherParams(costmap_weight_sqrt=1e-12).obstacle_terms_enabled() is False
    assert SmootherParams(costmap_weight_sqrt=0.0).obstacle_terms_enabled() is False
    assert SmootherParams(costmap_weight_sqrt=1e-6).obstacle_terms_enabled() is True


def test_optimizer_params_defaults():
    p = OptimizerParams()
    assert p.debug is False
    assert p.linear_solver == LinearSolver.SparseNormalCholesky
    assert p.max_iterations == 50
    assert p.parameter_tolerance == 1e-8
    assert p.function_tolerance == 1e-6
    assert p.gradient_tolerance == 1e-10


def test_linear_solver_enum_values():
    assert LinearSolver.DenseQr.value == "DENSE_QR"
    assert LinearSolver.SparseNormalCholesky.value == "SPARSE_NORMAL_CHOLESKY"


def test_params_are_independent():
    a = SmootherParams()
    b = SmootherParams()
    a.max_curvature = 5.0
    assert b.max_curvature == 0.0


def test_cost_check_points_isolated_per_instance():
    a = SmootherParams()
    b = SmootherParams()
    a.cost_check_points.append(0.1)
    assert b.cost_check_points == []
