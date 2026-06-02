// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Cached Euclidean Signed Distance Field.
//!
//! Provides a per-cell clearance value used by the ESDF-capsule collision
//! path and the obstacle heuristic.

use crate::costmap::Costmap2D;

/// Cached ESDF for the active [`Costmap2D`].
///
/// On each rebuild the ESDF is recomputed for the given costmap.  Subsequent
/// queries use bilinear-style nearest-cell lookups.
pub struct EsdfHolder {
    costmap: Option<*const Costmap2D>,
    esdf: Vec<f64>,
    valid: bool,
    origin_x: f64,
    origin_y: f64,
    resolution: f64,
    inv_resolution: f64,
    size_x: u32,
    size_y: u32,
}

// Safety: EsdfHolder is used single-threaded within the planner context.
unsafe impl Send for EsdfHolder {}

impl Default for EsdfHolder {
    fn default() -> Self {
        Self {
            costmap: None,
            esdf: Vec::new(),
            valid: false,
            origin_x: 0.0,
            origin_y: 0.0,
            resolution: 1.0,
            inv_resolution: 1.0,
            size_x: 0,
            size_y: 0,
        }
    }
}

impl EsdfHolder {
    /// Rebuild the ESDF for the given costmap.  Does nothing if the pointer
    /// is the same as the last call.
    pub fn rebuild(&mut self, costmap: &Costmap2D, _use_exact: bool) {
        let ptr = costmap as *const Costmap2D;
        if self.valid && self.costmap == Some(ptr) {
            return;
        }

        self.costmap = Some(ptr);
        self.origin_x = costmap.origin_x();
        self.origin_y = costmap.origin_y();
        self.resolution = costmap.resolution();
        self.inv_resolution = 1.0 / self.resolution;
        self.size_x = costmap.size_x();
        self.size_y = costmap.size_y();

        let total = (self.size_x * self.size_y) as usize;
        self.esdf.resize(total, 0.0);

        // Simple 2-pass approximate distance transform (Meijster / Felzenszwalb).
        // 1. Forward pass: propagate distance from top-left.
        let map = costmap.char_map();
        let lethal = 253u8; // INSCRIBED_COST threshold for lethal

        // Initialize: 0 for lethal cells, INF otherwise.
        let mut dist = vec![f64::INFINITY; total];
        for y in 0..self.size_y {
            for x in 0..self.size_x {
                let idx = (y * self.size_x + x) as usize;
                if map[idx] >= lethal {
                    dist[idx] = 0.0;
                }
            }
        }
        drop(map);

        // Horizontal pass
        for y in 0..self.size_y {
            for x in 1..self.size_x {
                let idx = (y * self.size_x + x) as usize;
                let prev = (y * self.size_x + (x - 1)) as usize;
                if dist[prev] + self.resolution < dist[idx] {
                    dist[idx] = dist[prev] + self.resolution;
                }
            }
            for x in (0..self.size_x).rev() {
                let idx = (y * self.size_x + x) as usize;
                if x + 1 < self.size_x {
                    let next = (y * self.size_x + (x + 1)) as usize;
                    if dist[next] + self.resolution < dist[idx] {
                        dist[idx] = dist[next] + self.resolution;
                    }
                }
            }
        }

        // Vertical pass
        for x in 0..self.size_x {
            for y in 1..self.size_y {
                let idx = (y * self.size_x + x) as usize;
                let prev = ((y - 1) * self.size_x + x) as usize;
                if dist[prev] + self.resolution < dist[idx] {
                    dist[idx] = dist[prev] + self.resolution;
                }
            }
            for y in (0..self.size_y).rev() {
                let idx = (y * self.size_x + x) as usize;
                if y + 1 < self.size_y {
                    let next = ((y + 1) * self.size_x + x) as usize;
                    if dist[next] + self.resolution < dist[idx] {
                        dist[idx] = dist[next] + self.resolution;
                    }
                }
            }
        }

        self.esdf = dist;
        self.valid = true;
    }

    pub fn valid(&self) -> bool {
        self.valid
    }

    pub fn values(&self) -> &[f64] {
        &self.esdf
    }

    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Raw ESDF value at a cell.
    pub fn clearance_at_cell(&self, mx: i32, my: i32) -> f64 {
        if !self.in_bounds(mx, my) {
            return f64::NEG_INFINITY;
        }
        let idx = (my as u32 * self.size_x + mx as u32) as usize;
        self.esdf[idx]
    }

    /// Signed clearance at a continuous world point (meters).
    pub fn clearance_at_world(&self, wx: f64, wy: f64) -> f64 {
        if !self.valid {
            return f64::NEG_INFINITY;
        }
        let mx = ((wx - self.origin_x) * self.inv_resolution).floor() as i32;
        let my = ((wy - self.origin_y) * self.inv_resolution).floor() as i32;
        if !self.in_bounds(mx, my) {
            return f64::NEG_INFINITY;
        }
        let idx = (my as u32 * self.size_x + mx as u32) as usize;

        // We need the costmap to decide lethal vs free.  Since we don't store
        // the costmap reference, we approximate using the ESDF value itself:
        // a clearance of 0 means we are at/touching an obstacle.
        let d_center = self.esdf[idx];
        let d_boundary = d_center - 0.5 * self.resolution;
        if d_boundary > 0.0 {
            d_boundary
        } else {
            0.0
        }
    }

    fn in_bounds(&self, mx: i32, my: i32) -> bool {
        mx >= 0 && my >= 0 && (mx as u32) < self.size_x && (my as u32) < self.size_y
    }
}
