// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Cost thresholds and motion-model enums.

use std::fmt;

// ---------------------------------------------------------------------------
// Motion model
// ---------------------------------------------------------------------------

/// Ackermann kinematic model used during the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotionModel {
    Unknown = 0,
    Dubin = 2,
    ReedsShepp = 3,
}

impl MotionModel {
    pub fn from_str(s: &str) -> Self {
        match s {
            "DUBIN" | "Dubin" | "dubin" => Self::Dubin,
            "REEDS_SHEPP" | "Reeds-Shepp" | "reeds_shepp" => Self::ReedsShepp,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for MotionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dubin => write!(f, "Dubin"),
            Self::ReedsShepp => write!(f, "Reeds-Shepp"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Goal heading mode
// ---------------------------------------------------------------------------

/// Controls which heading orientations are accepted as valid goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GoalHeadingMode {
    Unknown = 0,
    Default = 1,
    Bidirectional = 2,
    AllDirection = 3,
}

impl GoalHeadingMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "DEFAULT" => Self::Default,
            "BIDIRECTIONAL" => Self::Bidirectional,
            "ALL_DIRECTION" => Self::AllDirection,
            _ => Self::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// Cost constants
// ---------------------------------------------------------------------------

/// Value assigned to unknown cells.
pub const UNKNOWN_COST: f32 = 255.0;
/// Lethal / occupied cost.
pub const OCCUPIED_COST: f32 = 254.0;
/// Inscribed cost.
pub const INSCRIBED_COST: f32 = 253.0;
/// Maximum cost treated as non-obstacle.
pub const MAX_NON_OBSTACLE_COST: f32 = 252.0;
/// Pre-computed square of [`MAX_NON_OBSTACLE_COST`].
pub const MAX_NON_OBSTACLE_COST_SQ: f32 = MAX_NON_OBSTACLE_COST * MAX_NON_OBSTACLE_COST;
/// Free-cell cost.
pub const FREE_COST: f32 = 0.0;
