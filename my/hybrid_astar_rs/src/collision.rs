// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Collision checking for the Hybrid A* search.
//!
//! Supports two backends:
//!
//! 1. **Legacy polygon / radius** — pre-rotate footprint vertices and check
//!    raw cell costs against OCCUPIED / INSCRIBED thresholds.
//! 2. **ESDF + capsule** — rotate cost-check-points into the world frame and
//!    query the cached ESDF for clearance.

use std::f64::consts::PI;

use crate::constants::*;
use crate::costmap::Costmap2D;
use crate::esdf::EsdfHolder;
use crate::types::Footprint;

/// Hybrid A* collision checker.
pub struct GridCollisionChecker {
    costmap: *const Costmap2D,
    angles: Vec<f32>,
    oriented_footprints: Vec<Footprint>,
    unoriented_footprint: Footprint,
    center_cost: f32,
    footprint_is_radius: bool,
    possible_collision_cost: f32,

    // ESDF capsule path
    use_esdf_footprint: bool,
    cost_check_points: Vec<f64>,
    robot_radius: f64,
    safe_distance: f64,
    esdf_holder: Option<*mut EsdfHolder>,
}

// Safety: GridCollisionChecker is used single-threaded.
unsafe impl Send for GridCollisionChecker {}

impl GridCollisionChecker {
    pub fn new(costmap: &Costmap2D, num_quantizations: u32) -> Self {
        let bin_size = (2.0 * PI) as f32 / num_quantizations as f32;
        let angles: Vec<f32> = (0..num_quantizations).map(|i| bin_size * i as f32).collect();
        Self {
            costmap,
            angles,
            oriented_footprints: Vec::new(),
            unoriented_footprint: Vec::new(),
            center_cost: 0.0,
            footprint_is_radius: false,
            possible_collision_cost: -1.0,
            use_esdf_footprint: false,
            cost_check_points: Vec::new(),
            robot_radius: 0.0,
            safe_distance: 0.0,
            esdf_holder: None,
        }
    }

    pub fn set_costmap(&mut self, costmap: &Costmap2D) {
        self.costmap = costmap;
    }

    pub fn costmap(&self) -> &Costmap2D {
        unsafe { &*self.costmap }
    }

    pub fn uses_esdf_footprint(&self) -> bool {
        self.use_esdf_footprint
    }

    pub fn get_cost(&self) -> f32 {
        self.center_cost
    }

    pub fn precomputed_angles(&self) -> &[f32] {
        &self.angles
    }

    // -- Footprint configuration --------------------------------------------

    /// Configure the legacy polygon / single-radius path.
    pub fn set_footprint(
        &mut self,
        footprint: &Footprint,
        radius: bool,
        possible_collision_cost: f64,
    ) {
        self.possible_collision_cost = possible_collision_cost as f32;
        self.footprint_is_radius = radius;

        // Clear ESDF state
        self.use_esdf_footprint = false;
        self.cost_check_points.clear();
        self.robot_radius = 0.0;
        self.safe_distance = 0.0;
        self.esdf_holder = None;

        if radius {
            return;
        }
        if footprint == &self.unoriented_footprint {
            return;
        }

        self.oriented_footprints.clear();
        self.oriented_footprints.reserve(self.angles.len());

        for &angle in &self.angles {
            let sin_t = angle.sin() as f64;
            let cos_t = angle.cos() as f64;
            let oriented: Footprint = footprint
                .iter()
                .map(|pt| {
                    crate::types::Point2D::new(
                        pt.x * cos_t - pt.y * sin_t,
                        pt.x * sin_t + pt.y * cos_t,
                    )
                })
                .collect();
            self.oriented_footprints.push(oriented);
        }
        self.unoriented_footprint = footprint.clone();
    }

    /// Configure the ESDF + capsule path.
    pub fn set_esdf_footprint(
        &mut self,
        cost_check_points: Vec<f64>,
        robot_radius: f64,
        safe_distance: f64,
        esdf_holder: &mut EsdfHolder,
    ) {
        self.oriented_footprints.clear();
        self.unoriented_footprint.clear();
        self.footprint_is_radius = false;
        self.possible_collision_cost = -1.0;

        self.use_esdf_footprint = true;
        self.cost_check_points = cost_check_points;
        self.robot_radius = robot_radius.max(0.0);
        self.safe_distance = safe_distance.max(0.0);
        self.esdf_holder = Some(esdf_holder);
    }

    // -- Collision checks ---------------------------------------------------

    /// Legacy in-collision check at continuous cell coordinates.
    pub fn in_collision(&mut self, x: f32, y: f32, angle_bin: f32, traverse_unknown: bool) -> bool {
        if self.outside_range(self.costmap().size_x(), x) || self.outside_range(self.costmap().size_y(), y) {
            return true;
        }

        let center_cost = self.costmap().get_cost_float(x, y) as f32;
        self.center_cost = center_cost;

        if !self.footprint_is_radius {
            if self.center_cost < self.possible_collision_cost && self.possible_collision_cost > 0.0 {
                return false;
            }
            if self.center_cost == UNKNOWN_COST && !traverse_unknown {
                return true;
            }
            if self.center_cost == INSCRIBED_COST || self.center_cost == OCCUPIED_COST {
                return true;
            }

            let (wx, wy) = self.costmap().map_cell_to_world(x, y);
            let angle_idx = angle_bin as usize;
            if angle_idx >= self.oriented_footprints.len() {
                return true;
            }
            let oriented = &self.oriented_footprints[angle_idx];

            for pt in oriented {
                let px = wx + pt.x;
                let py = wy + pt.y;
                if let Some((mx, my)) = self.costmap().world_to_map(px, py) {
                    let cell_cost = self.costmap().get_cost(mx, my) as f32;
                    if cell_cost >= OCCUPIED_COST {
                        return true;
                    }
                    if cell_cost == UNKNOWN_COST && !traverse_unknown {
                        return true;
                    }
                } else {
                    return true;
                }
            }
            false
        } else {
            if self.center_cost == UNKNOWN_COST && traverse_unknown {
                return false;
            }
            self.center_cost >= INSCRIBED_COST
        }
    }

    /// ESDF/capsule in-collision check at a world pose.
    pub fn in_collision_esdf(
        &self,
        wx: f64,
        wy: f64,
        theta: f64,
        _traverse_unknown: bool,
    ) -> bool {
        if !self.use_esdf_footprint {
            return false;
        }
        let min_clearance = self.get_min_clearance(wx, wy, theta);
        if !min_clearance.is_finite() {
            return true;
        }
        min_clearance < self.robot_radius
    }

    /// Minimum ESDF clearance over the footprint, in meters.
    pub fn get_min_clearance(&self, wx: f64, wy: f64, theta: f64) -> f64 {
        let holder = match self.esdf_holder {
            Some(h) => unsafe { &*h },
            None => return f64::INFINITY,
        };
        if !holder.valid() {
            return f64::INFINITY;
        }

        if self.cost_check_points.is_empty() {
            return holder.clearance_at_world(wx, wy);
        }

        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let mut min_clearance = f64::INFINITY;

        let mut offset = 0;
        while offset + 2 < self.cost_check_points.len() {
            let lx = self.cost_check_points[offset];
            let ly = self.cost_check_points[offset + 1];
            let world_x = wx + cos_t * lx - sin_t * ly;
            let world_y = wy + sin_t * lx + cos_t * ly;
            let d = holder.clearance_at_world(world_x, world_y);
            if d < min_clearance {
                min_clearance = d;
            }
            offset += 3;
        }
        min_clearance
    }

    /// Quadratic soft penalty in [0, 1] driven by `safe_distance`.
    pub fn get_soft_penalty(&self, wx: f64, wy: f64, theta: f64) -> f64 {
        if !self.use_esdf_footprint || self.safe_distance <= 1e-9 {
            return 0.0;
        }
        let min_clearance = self.get_min_clearance(wx, wy, theta);
        if !min_clearance.is_finite() {
            return 1.0;
        }
        let surface_distance = min_clearance - self.robot_radius;
        if surface_distance >= self.safe_distance {
            return 0.0;
        }
        let clamped = surface_distance.clamp(0.0, self.safe_distance);
        let normalized_gap = (self.safe_distance - clamped) / self.safe_distance;
        normalized_gap * normalized_gap
    }

    // -- Helpers ------------------------------------------------------------

    fn outside_range(&self, max: u32, value: f32) -> bool {
        value < 0.0 || value >= max as f32
    }
}
