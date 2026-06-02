// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! 2-D cost grid used as the planner's world representation.

use std::sync::Mutex;

/// A 2-D occupancy / inflation cost grid.
///
/// Internally the data is stored as a flat `Vec<u8>` in row-major order.
/// The map is wrapped in a `Mutex` so that concurrent read/write from the
/// smoother or collision checker is safe.
pub struct Costmap2D {
    size_x: u32,
    size_y: u32,
    resolution: f64,
    origin_x: f64,
    origin_y: f64,
    cost_map: Mutex<Vec<u8>>,
}

impl Costmap2D {
    pub fn new(
        size_x: u32,
        size_y: u32,
        resolution: f64,
        origin_x: f64,
        origin_y: f64,
        default_cost: u8,
    ) -> Self {
        Self {
            size_x,
            size_y,
            resolution,
            origin_x,
            origin_y,
            cost_map: Mutex::new(vec![default_cost; (size_x * size_y) as usize]),
        }
    }

    // -- accessors ----------------------------------------------------------

    pub fn size_x(&self) -> u32 {
        self.size_x
    }
    pub fn size_y(&self) -> u32 {
        self.size_y
    }
    pub fn resolution(&self) -> f64 {
        self.resolution
    }
    pub fn origin_x(&self) -> f64 {
        self.origin_x
    }
    pub fn origin_y(&self) -> f64 {
        self.origin_y
    }

    // -- cost access --------------------------------------------------------

    /// Cost at integer cell `(mx, my)`.
    pub fn get_cost(&self, mx: u32, my: u32) -> u8 {
        let map = self.cost_map.lock().unwrap();
        map[(my * self.size_x + mx) as usize]
    }

    /// Cost at a flat index.
    pub fn get_cost_idx(&self, idx: u32) -> u8 {
        let map = self.cost_map.lock().unwrap();
        map[idx as usize]
    }

    /// Cost at continuous cell coordinates (nearest-cell, clamped).
    pub fn get_cost_float(&self, fx: f32, fy: f32) -> u8 {
        let ix = (fx + 0.5).floor() as i32;
        let iy = (fy + 0.5).floor() as i32;
        let ix = ix.clamp(0, self.size_x as i32 - 1) as u32;
        let iy = iy.clamp(0, self.size_y as i32 - 1) as u32;
        self.get_cost(ix, iy)
    }

    /// Set cost at integer cell `(mx, my)`.
    pub fn set_cost(&self, mx: u32, my: u32, cost: u8) {
        let mut map = self.cost_map.lock().unwrap();
        map[(my * self.size_x + mx) as usize] = cost;
    }

    /// Direct mutable access to the underlying byte slice (for ESDF, etc.).
    pub fn char_map(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.cost_map.lock().unwrap()
    }

    // -- coordinate conversion ----------------------------------------------

    /// World → continuous map coordinates.  Returns `false` if outside bounds.
    pub fn world_to_map_continuous(&self, wx: f64, wy: f64) -> Option<(f32, f32)> {
        let mx = ((wx - self.origin_x) / self.resolution) as f32;
        let my = ((wy - self.origin_y) / self.resolution) as f32;
        if mx >= 0.0
            && mx < self.size_x as f32
            && my >= 0.0
            && my < self.size_y as f32
        {
            Some((mx, my))
        } else {
            None
        }
    }

    /// Continuous cell coordinates → world coordinates.
    pub fn map_cell_to_world(&self, mx: f32, my: f32) -> (f64, f64) {
        let wx = self.origin_x + mx as f64 * self.resolution;
        let wy = self.origin_y + my as f64 * self.resolution;
        (wx, wy)
    }

    /// World → integer map coordinates.  Returns `false` if outside bounds.
    pub fn world_to_map(&self, wx: f64, wy: f64) -> Option<(u32, u32)> {
        let map_x = (wx - self.origin_x) / self.resolution;
        let map_y = (wy - self.origin_y) / self.resolution;
        if map_x < 0.0 || map_y < 0.0 {
            return None;
        }
        let mx = map_x as u32;
        let my = map_y as u32;
        if mx < self.size_x && my < self.size_y {
            Some((mx, my))
        } else {
            None
        }
    }

    /// Integer cell index → world coordinates (cell centre).
    pub fn map_to_world(&self, mx: u32, my: u32) -> (f64, f64) {
        let wx = self.origin_x + (mx as f64 + 0.5) * self.resolution;
        let wy = self.origin_y + (my as f64 + 0.5) * self.resolution;
        (wx, wy)
    }

    /// Resize the costmap (re-allocates and fills with zeros).
    pub fn resize(
        &mut self,
        size_x: u32,
        size_y: u32,
        resolution: f64,
        origin_x: f64,
        origin_y: f64,
    ) {
        self.size_x = size_x;
        self.size_y = size_y;
        self.resolution = resolution;
        self.origin_x = origin_x;
        self.origin_y = origin_y;
        *self.cost_map.lock().unwrap() = vec![0; (size_x * size_y) as usize];
    }

    /// Set all cells to zero.
    pub fn reset(&self) {
        let mut map = self.cost_map.lock().unwrap();
        map.iter_mut().for_each(|b| *b = 0);
    }
}
