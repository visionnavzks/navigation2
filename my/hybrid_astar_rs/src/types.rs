// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Core data structures used throughout the planner.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Basic geometry
// ---------------------------------------------------------------------------

/// A 2-D pose with heading in radians.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }
}

/// Ordered sequence of poses representing a path.
pub type Path = Vec<Pose>;

/// Simple 2-D point.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Robot footprint (polygon vertices in the body frame).
pub type Footprint = Vec<Point2D>;

// ---------------------------------------------------------------------------
// Search-space coordinate helpers
// ---------------------------------------------------------------------------

/// Continuous-valued SE(2) coordinate used during the search.
///
/// * `x`, `y` — cell coordinates (may be fractional).
/// * `theta` — angular-bin index (may be fractional for interpolation).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Coordinates {
    pub x: f32,
    pub y: f32,
    pub theta: f32,
}

impl Coordinates {
    pub fn new(x: f32, y: f32, theta: f32) -> Self {
        Self { x, y, theta }
    }
}

/// (cos θ, sin θ) pair for a given angular bin.
pub type TrigValues = (f64, f64);

/// A single motion-primitive projection.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionPose {
    pub x: f32,
    pub y: f32,
    pub theta: f32,
    pub turn_dir: TurnDirection,
}

impl MotionPose {
    pub fn new(x: f32, y: f32, theta: f32, turn_dir: TurnDirection) -> Self {
        Self {
            x,
            y,
            theta,
            turn_dir,
        }
    }
}

/// Vector of motion-primitive projections.
pub type MotionPoses = Vec<MotionPose>;

// ---------------------------------------------------------------------------
// Turn direction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TurnDirection {
    #[default]
    Unknown = 0,
    Forward = 1,
    Left = 2,
    Right = 3,
    Reverse = 4,
    RevLeft = 5,
    RevRight = 6,
}

// ---------------------------------------------------------------------------
// Heuristic helpers
// ---------------------------------------------------------------------------

/// (heuristic-cost, linear-index) pair used in the priority queue.
pub type NodeHeuristicPair = (f32, u64);

/// Lookup table (flat vector of floats).
pub type LookupTable = Vec<f32>;

/// Min-heap comparator for [`NodeHeuristicPair`].
pub struct NodeHeuristicComparator;

impl NodeHeuristicComparator {
    /// Returns `true` when `a` should be ordered *after* `b` (min-heap).
    pub fn less(a: &NodeHeuristicPair, b: &NodeHeuristicPair) -> bool {
        a.0 > b.0
    }
}

// ---------------------------------------------------------------------------
// Search configuration
// ---------------------------------------------------------------------------

/// Parameters that control the A* search behaviour.
#[derive(Debug, Clone)]
pub struct SearchInfo {
    pub minimum_turning_radius: f32,
    pub non_straight_penalty: f32,
    pub change_penalty: f32,
    pub reverse_penalty: f32,
    pub cost_penalty: f32,
    pub retrospective_penalty: f32,
    pub rotation_penalty: f32,
    pub analytic_expansion_ratio: f32,
    pub analytic_expansion_max_length: f32,
    pub analytic_expansion_max_cost: f32,
    pub analytic_expansion_max_cost_override: bool,
    pub cache_obstacle_heuristic: bool,
    pub allow_reverse_expansion: bool,
    pub allow_primitive_interpolation: bool,
    pub downsample_obstacle_heuristic: bool,
    pub use_quadratic_cost_penalty: bool,
}

impl Default for SearchInfo {
    fn default() -> Self {
        Self {
            minimum_turning_radius: 8.0,
            non_straight_penalty: 1.05,
            change_penalty: 0.0,
            reverse_penalty: 2.0,
            cost_penalty: 2.0,
            retrospective_penalty: 0.015,
            rotation_penalty: 5.0,
            analytic_expansion_ratio: 3.5,
            analytic_expansion_max_length: 60.0,
            analytic_expansion_max_cost: 200.0,
            analytic_expansion_max_cost_override: false,
            cache_obstacle_heuristic: false,
            allow_reverse_expansion: false,
            allow_primitive_interpolation: false,
            downsample_obstacle_heuristic: true,
            use_quadratic_cost_penalty: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Smoother parameters
// ---------------------------------------------------------------------------

/// Parameters for the iterative path smoother.
#[derive(Debug, Clone)]
pub struct SmootherParams {
    pub tolerance: f64,
    pub max_its: usize,
    pub w_data: f64,
    pub w_smooth: f64,
    pub holonomic: bool,
    pub do_refinement: bool,
    pub refinement_num: usize,
}

impl Default for SmootherParams {
    fn default() -> Self {
        Self {
            tolerance: 1e-3,
            max_its: 1000,
            w_data: 0.32,
            w_smooth: 0.25,
            holonomic: false,
            do_refinement: true,
            refinement_num: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Goal state
// ---------------------------------------------------------------------------

/// A candidate goal in the search graph.
#[derive(Debug, Clone)]
pub struct GoalState {
    pub index: u64,
    pub is_valid: bool,
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Wrap an angular-bin index into `[0, num_bins)`.
#[inline]
pub fn wrap_bin_index(bin: i32, num_bins: u32) -> u32 {
    let n = num_bins as i32;
    let mut b = bin % n;
    if b < 0 {
        b += n;
    }
    b as u32
}

/// Wrap an angle in radians into `[0, 2π)`.
#[inline]
pub fn wrap_angle(angle: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut a = angle % two_pi;
    if a < 0.0 {
        a += two_pi;
    }
    a
}
