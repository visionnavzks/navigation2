// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Cost-map down-sampling utility.

use crate::constants::UNKNOWN_COST;
use crate::costmap::Costmap2D;

/// Down-samples a [`Costmap2D`] by a given factor.
pub struct CostmapDownsampler {
    costmap: *const Costmap2D,
    downsampled_costmap: Option<Costmap2D>,
    size_x: u32,
    size_y: u32,
    downsampled_size_x: u32,
    downsampled_size_y: u32,
    downsampling_factor: u32,
    use_min_cost_neighbor: bool,
    downsampled_resolution: f64,
}

unsafe impl Send for CostmapDownsampler {}

impl CostmapDownsampler {
    pub fn new() -> Self {
        Self {
            costmap: std::ptr::null(),
            downsampled_costmap: None,
            size_x: 0,
            size_y: 0,
            downsampled_size_x: 0,
            downsampled_size_y: 0,
            downsampling_factor: 1,
            use_min_cost_neighbor: false,
            downsampled_resolution: 0.0,
        }
    }

    pub fn on_configure(&mut self, costmap: &Costmap2D, downsampling_factor: u32, use_min_cost_neighbor: bool) {
        self.costmap = costmap;
        self.downsampling_factor = downsampling_factor;
        self.use_min_cost_neighbor = use_min_cost_neighbor;
        self.update_costmap_size();

        self.downsampled_costmap = Some(Costmap2D::new(
            self.downsampled_size_x,
            self.downsampled_size_y,
            self.downsampled_resolution,
            costmap.origin_x(),
            costmap.origin_y(),
            UNKNOWN_COST as u8,
        ));
    }

    pub fn downsample(&mut self, downsampling_factor: u32) -> &mut Costmap2D {
        self.downsampling_factor = downsampling_factor;
        self.update_costmap_size();

        let need_resize = if let Some(ref dc) = self.downsampled_costmap {
            dc.size_x() != self.downsampled_size_x
                || dc.size_y() != self.downsampled_size_y
                || (dc.resolution() - self.downsampled_resolution).abs() > 1e-9
        } else {
            true
        };

        if need_resize {
            self.resize_costmap();
        }

        for i in 0..self.downsampled_size_x {
            for j in 0..self.downsampled_size_y {
                self.set_cost_of_cell(i, j);
            }
        }

        self.downsampled_costmap.as_mut().unwrap()
    }

    fn update_costmap_size(&mut self) {
        let cm = unsafe { &*self.costmap };
        self.size_x = cm.size_x();
        self.size_y = cm.size_y();
        self.downsampled_size_x =
            ((self.size_x as f64) / (self.downsampling_factor as f64)).ceil() as u32;
        self.downsampled_size_y =
            ((self.size_y as f64) / (self.downsampling_factor as f64)).ceil() as u32;
        self.downsampled_resolution =
            self.downsampling_factor as f64 * cm.resolution();
    }

    fn resize_costmap(&mut self) {
        let cm = unsafe { &*self.costmap };
        if let Some(ref mut dc) = self.downsampled_costmap {
            dc.resize(
                self.downsampled_size_x,
                self.downsampled_size_y,
                self.downsampled_resolution,
                cm.origin_x(),
                cm.origin_y(),
            );
        }
    }

    fn set_cost_of_cell(&mut self, new_mx: u32, new_my: u32) {
        let cm = unsafe { &*self.costmap };
        let mut cost: u8 = if self.use_min_cost_neighbor { 255 } else { 0 };
        let x_offset = new_mx * self.downsampling_factor;
        let y_offset = new_my * self.downsampling_factor;

        for i in 0..self.downsampling_factor {
            let mx = x_offset + i;
            if mx >= self.size_x {
                continue;
            }
            for j in 0..self.downsampling_factor {
                let my = y_offset + j;
                if my >= self.size_y {
                    continue;
                }
                let cell_cost = cm.get_cost(mx, my);
                cost = if self.use_min_cost_neighbor {
                    cost.min(cell_cost)
                } else {
                    cost.max(cell_cost)
                };
            }
        }

        if let Some(ref dc) = self.downsampled_costmap {
            dc.set_cost(new_mx, new_my, cost);
        }
    }
}
