// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! # hybrid_astar
//!
//! A standalone Hybrid A* path planner supporting Ackermann vehicle kinematics
//! (Dubin / Reeds-Shepp models).  This is a Rust port of the C++ `hybrid_astar`
//! crate from the navigation2 ecosystem.
//!
//! ## Crate organisation
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`constants`] | Cost thresholds, enums (`MotionModel`, `GoalHeadingMode`) |
//! | [`types`] | Core data structures (`Pose`, `Path`, `Coordinates`, `SearchInfo`, …) |
//! | [`costmap`] | 2-D cost grid (`Costmap2D`) |
//! | [`collision`] | `GridCollisionChecker` (legacy polygon + ESDF capsule backends) |
//! | [`esdf`] | Cached Euclidean Signed Distance Field |
//! | [`steering`] | Dubin / Reeds-Shepp state-space wrapper |
//! | [`node`] | `NodeHybrid`, `HybridMotionTable`, `NodeContext` |
//! | [`a_star`] | `AStarAlgorithm` — the main search loop |
//! | [`analytic`] | Analytic expansion (steering-function direct connections) |
//! | [`obstacle_heuristic`] | D\*-style 2-D obstacle distance heuristic |
//! | [`distance_heuristic`] | Pre-computed kinematic distance lookup table |
//! | [`goal_manager`] | Multi-goal management with heading modes |
//! | [`smoother`] | Path smoother with boundary-condition enforcement |
//! | [`costmap_downsampler`] | Cost-map down-sampling utility |
//! | [`planner`] | `SmacPlannerHybrid` — high-level planner façade |

pub mod constants;
pub mod types;
pub mod costmap;
pub mod collision;
pub mod esdf;
pub mod steering;
pub mod node;
pub mod a_star;
pub mod analytic;
pub mod obstacle_heuristic;
pub mod distance_heuristic;
pub mod goal_manager;
pub mod smoother;
pub mod costmap_downsampler;
pub mod planner;
pub mod utils;

// Re-export the most commonly used types at crate root for convenience.
pub use constants::{GoalHeadingMode, MotionModel};
pub use costmap::Costmap2D;
pub use planner::{SmacPlannerHybrid, SmacPlannerHybridConfig};
pub use types::{Path, Pose};
