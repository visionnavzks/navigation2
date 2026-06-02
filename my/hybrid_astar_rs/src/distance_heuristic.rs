// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Pre-computed kinematic distance lookup table.

use crate::node::HybridMotionTable;
use crate::steering::{self, State};
use crate::types::*;

/// Pre-computed motion-model-aware distance heuristic.
pub struct DistanceHeuristic {
    lookup_table: LookupTable,
    size_lookup: f32,
}

impl DistanceHeuristic {
    pub fn new() -> Self {
        Self {
            lookup_table: Vec::new(),
            size_lookup: 0.0,
        }
    }

    /// Build the lookup table.
    pub fn precompute_distance_heuristic(
        &mut self,
        lookup_table_dim: f32,
        motion_model: crate::constants::MotionModel,
        dim_3_size: u32,
        search_info: &SearchInfo,
        motion_table: &mut HybridMotionTable,
    ) {
        motion_table.state_space = Some(steering::create_steering_state_space(
            motion_model,
            search_info.minimum_turning_radius as f64,
        ));

        let ss = motion_table.state_space.as_ref().unwrap();
        let to = State::new(0.0, 0.0, 0.0);
        self.size_lookup = lookup_table_dim;
        let dim_3_size_int = dim_3_size as i32;
        let angular_bin_size = 2.0 * std::f64::consts::PI / dim_3_size as f64;

        let floored_half = (self.size_lookup / 2.0).floor() as i32;
        let ceiled_neg_half = (-self.size_lookup / 2.0).ceil() as i32;

        let table_size = ((floored_half - ceiled_neg_half + 1) as usize)
            * ((floored_half + 1) as usize)
            * dim_3_size as usize;
        self.lookup_table.resize(table_size, 0.0);

        let mut index = 0;
        for x in ceiled_neg_half..=floored_half {
            for y in 0..=floored_half {
                for heading in 0..dim_3_size_int {
                    let from = State::new(x as f64, y as f64, heading as f64 * angular_bin_size);
                    let motion_heuristic = ss.distance(from, to);
                    self.lookup_table[index] = motion_heuristic as f32;
                    index += 1;
                }
            }
        }
    }

    /// Get the pre-computed kinematic distance from `node_coords` to `goal_coords`.
    pub fn get_distance_heuristic(
        &self,
        node_coords: Coordinates,
        goal_coords: Coordinates,
        obstacle_heuristic: f32,
        motion_table: &HybridMotionTable,
    ) -> f32 {
        let (cos_th, sin_th_raw) = motion_table.trig_values[goal_coords.theta as usize];
        let cos_th = cos_th;
        let sin_th = -sin_th_raw;

        let dx = node_coords.x - goal_coords.x;
        let dy = node_coords.y - goal_coords.y;

        let dtheta_bin = wrap_bin_index(
            (node_coords.theta - goal_coords.theta) as i32,
            motion_table.num_angle_quantization,
        );

        let node_coords_relative = Coordinates::new(
            (dx as f64 * cos_th - dy as f64 * sin_th).round() as f32,
            (dx as f64 * sin_th + dy as f64 * cos_th).round() as f32,
            dtheta_bin as f32,
        );

        let floored_size = (self.size_lookup / 2.0).floor() as i32;
        let y_size = floored_size + 1;
        let mirrored_relative_y = node_coords_relative.y.abs() as i32;

        if node_coords_relative.x.abs() as i32 <= floored_size
            && mirrored_relative_y <= floored_size
        {
            let theta_pos = if node_coords_relative.y < 0.0 {
                let t = motion_table.num_angle_quantization as i32 - node_coords_relative.theta as i32;
                t % motion_table.num_angle_quantization as i32
            } else {
                node_coords_relative.theta as i32
            };

            let x_pos = node_coords_relative.x as i32 + floored_size;
            let index = x_pos * y_size * motion_table.num_angle_quantization as i32
                + mirrored_relative_y * motion_table.num_angle_quantization as i32
                + theta_pos;
            self.lookup_table[index as usize]
        } else if obstacle_heuristic <= 0.0 {
            // Fallback: compute on the fly
            if let Some(ref ss) = motion_table.state_space {
                let from = State::new(
                    node_coords.x as f64,
                    node_coords.y as f64,
                    motion_table.get_angle_from_bin(node_coords.theta) as f64,
                );
                let to = State::new(
                    goal_coords.x as f64,
                    goal_coords.y as f64,
                    motion_table.get_angle_from_bin(goal_coords.theta) as f64,
                );
                ss.distance(from, to) as f32
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}
