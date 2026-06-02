// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Utility functions.

use crate::constants::*;
use crate::costmap::Costmap2D;
use crate::types::*;

/// Convert continuous cell coordinates to world coordinates.
pub fn get_world_coords(mx: f32, my: f32, costmap: &Costmap2D) -> Pose {
    let (wx, wy) = costmap.map_cell_to_world(mx, my);
    Pose::new(wx, wy, 0.0)
}

/// Compute the cost at the circumscribed radius.
pub fn find_circumscribed_cost(
    costmap: &Costmap2D,
    circumscribed_radius: f64,
    inflation_radius: f64,
) -> f64 {
    if inflation_radius < circumscribed_radius {
        return 0.0;
    }
    let resolution = costmap.resolution();
    let distance_cells = circumscribed_radius / resolution;
    let inflation_cells = inflation_radius / resolution;
    let cost = INSCRIBED_COST as f64 * (1.0 - distance_cells / inflation_cells);
    cost.max(0.0)
}
