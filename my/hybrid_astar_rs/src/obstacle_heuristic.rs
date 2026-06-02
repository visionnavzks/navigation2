// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! D*-style 2-D obstacle distance heuristic.

use std::cell::RefCell;

use crate::constants::*;
use crate::costmap::Costmap2D;
use crate::esdf::EsdfHolder;
use crate::types::*;

/// D*-style obstacle heuristic that pre-computes a 2-D distance field from
/// the goal cell.
pub struct ObstacleHeuristic {
    lookup_table: RefCell<Vec<f32>>,
    queue: RefCell<Vec<NodeHeuristicPair>>,
    costmap: Option<*const Costmap2D>,
    cached_size_x: u32,
    cached_size_y: u32,

    // Optional ESDF path
    esdf_holder: Option<*const EsdfHolder>,
    cost_check_points: Vec<f64>,
    robot_radius: f64,
    safe_distance: f64,
}

// Safety: used single-threaded.
unsafe impl Send for ObstacleHeuristic {}

impl ObstacleHeuristic {
    pub fn new() -> Self {
        Self {
            lookup_table: RefCell::new(Vec::new()),
            queue: RefCell::new(Vec::new()),
            costmap: None,
            cached_size_x: 0,
            cached_size_y: 0,
            esdf_holder: None,
            cost_check_points: Vec::new(),
            robot_radius: 0.0,
            safe_distance: 0.0,
        }
    }

    pub fn set_esdf_holder(&mut self, holder: *const EsdfHolder) {
        self.esdf_holder = Some(holder);
    }

    pub fn set_esdf_footprint_params(
        &mut self,
        cost_check_points: Vec<f64>,
        robot_radius: f64,
        safe_distance: f64,
    ) {
        self.cost_check_points = cost_check_points;
        self.robot_radius = robot_radius;
        self.safe_distance = safe_distance;
    }

    pub fn reset_obstacle_heuristic(
        &mut self,
        costmap: &Costmap2D,
        start_x: f32,
        start_y: f32,
        goal_x: f32,
        goal_y: f32,
        downsample_obstacle_heuristic: bool,
    ) {
        self.costmap = Some(costmap);

        let (size_x, size_y) = if downsample_obstacle_heuristic {
            (
                ((costmap.size_x() as f32) / 2.0).ceil() as u32,
                ((costmap.size_y() as f32) / 2.0).ceil() as u32,
            )
        } else {
            (costmap.size_x(), costmap.size_y())
        };
        self.cached_size_x = size_x;
        self.cached_size_y = size_y;
        let size = (size_x * size_y) as usize;

        let mut lt = self.lookup_table.borrow_mut();
        lt.resize(size, 0.0);
        lt.iter_mut().for_each(|v| *v = 0.0);

        let mut q = self.queue.borrow_mut();
        q.clear();
        q.reserve(size);

        let goal_x_floor = goal_x.floor().max(0.0) as u32;
        let goal_y_floor = goal_y.floor().max(0.0) as u32;
        let gx = goal_x_floor.min(size_x - 1);
        let gy = goal_y_floor.min(size_y - 1);

        let goal_index = if downsample_obstacle_heuristic {
            (gy / 2) * size_x + (gx / 2)
        } else {
            gy * size_x + gx
        };

        let inv = if downsample_obstacle_heuristic { 2.0 } else { 1.0 };
        let start_x_floor = (start_x / inv).floor();
        let start_y_floor = (start_y / inv).floor();

        q.push((
            self.distance_heuristic_2d(goal_index, size_x, start_x_floor as u32, start_y_floor as u32),
            goal_index as u64,
        ));
        lt[goal_index as usize] = -0.00001;
    }

    pub fn get_obstacle_heuristic(
        &self,
        node_coords: Coordinates,
        cost_penalty: f32,
        use_quadratic_cost_penalty: bool,
        downsample_obstacle_heuristic: bool,
    ) -> f32 {
        let size_x = self.cached_size_x;
        let size_y = self.cached_size_y;

        let (start_x_f, start_y_f) = if downsample_obstacle_heuristic {
            (node_coords.x.floor() / 2.0, node_coords.y.floor() / 2.0)
        } else {
            (node_coords.x.floor(), node_coords.y.floor())
        };
        let start_x = (start_x_f.max(0.0)) as u32;
        let start_y = (start_y_f.max(0.0)) as u32;
        let start_index = start_y * size_x + start_x;

        let lt = self.lookup_table.borrow();
        let requested_node_cost = lt[start_index as usize];
        drop(lt);

        if requested_node_cost > 0.0 {
            return if downsample_obstacle_heuristic {
                2.0 * requested_node_cost
            } else {
                requested_node_cost
            };
        }

        let size_x_int = size_x as i32;
        let sqrt2 = 2.0_f32.sqrt();
        let neighborhood: [i32; 8] = [
            1, -1, size_x_int, -size_x_int, size_x_int + 1, size_x_int - 1,
            -size_x_int + 1, -size_x_int - 1,
        ];

        let mut lt = self.lookup_table.borrow_mut();
        let mut q = self.queue.borrow_mut();

        while !q.is_empty() {
            let mut min_idx = 0;
            for i in 1..q.len() {
                if q[i].0 < q[min_idx].0 {
                    min_idx = i;
                }
            }
            let node_idx = q[min_idx].1;
            q.swap_remove(min_idx);

            let c_cost = lt[node_idx as usize];
            if c_cost > 0.0 {
                continue;
            }
            let c_cost = -c_cost;
            lt[node_idx as usize] = c_cost;

            for (i, &nbr) in neighborhood.iter().enumerate() {
                let new_idx_int = node_idx as i32 + nbr;
                if new_idx_int < 0 || new_idx_int >= (size_x * size_y) as i32 {
                    continue;
                }
                let new_idx = new_idx_int as u32;

                let new_my = new_idx / size_x;
                let new_mx = new_idx - new_my * size_x;
                if new_mx >= size_x || new_my >= size_y {
                    continue;
                }

                let cost = self.cell_cost_for_heuristic(new_mx, new_my);
                if cost >= INSCRIBED_COST {
                    continue;
                }

                if size_x <= 3 || size_y <= 3 {
                    continue;
                }
                if new_mx <= 3 || new_mx >= size_x - 3 {
                    continue;
                }
                if new_my <= 3 || new_my >= size_y - 3 {
                    continue;
                }

                let existing_cost = lt[new_idx as usize];
                if existing_cost <= 0.0 {
                    let travel_cost = if use_quadratic_cost_penalty {
                        (if i <= 3 { 1.0 } else { sqrt2 })
                            * (1.0 + cost_penalty * cost * cost / MAX_NON_OBSTACLE_COST_SQ)
                    } else {
                        (if i <= 3 { 1.0 } else { sqrt2 })
                            * (1.0 + cost_penalty * cost / MAX_NON_OBSTACLE_COST)
                    };

                    let new_cost = c_cost + travel_cost;
                    if existing_cost == 0.0 || -existing_cost > new_cost {
                        lt[new_idx as usize] = -new_cost;
                        q.push((
                            new_cost + self.distance_heuristic_2d(new_idx, size_x, start_x, start_y),
                            new_idx as u64,
                        ));
                    }
                }
            }

            if node_idx == start_index as u64 {
                break;
            }
        }

        let cost = lt[start_index as usize];
        if downsample_obstacle_heuristic {
            2.0 * cost
        } else {
            cost
        }
    }

    fn cell_cost_for_heuristic(&self, mx: u32, my: u32) -> f32 {
        if self.esdf_holder.is_none() {
            let cm = unsafe { &*self.costmap.unwrap() };
            return cm.get_cost(mx, my) as f32;
        }

        let holder = unsafe { &*self.esdf_holder.unwrap() };
        if !holder.valid() {
            let cm = unsafe { &*self.costmap.unwrap() };
            return cm.get_cost(mx, my) as f32;
        }

        let cm = unsafe { &*self.costmap.unwrap() };
        let resolution = cm.resolution();
        let wx = cm.origin_x() + (mx as f64 + 0.5) * resolution;
        let wy = cm.origin_y() + (my as f64 + 0.5) * resolution;

        let mut min_clearance = holder.clearance_at_world(wx, wy);
        if !self.cost_check_points.is_empty() {
            min_clearance = f64::INFINITY;
            let mut offset = 0;
            while offset + 2 < self.cost_check_points.len() {
                let lx = self.cost_check_points[offset];
                let ly = self.cost_check_points[offset + 1];
                let d = holder.clearance_at_world(wx + lx, wy + ly);
                if d < min_clearance {
                    min_clearance = d;
                }
                offset += 3;
            }
        }

        if !min_clearance.is_finite() {
            return MAX_NON_OBSTACLE_COST;
        }
        let surface_distance = min_clearance - self.robot_radius;
        if surface_distance >= self.safe_distance {
            return 0.0;
        }
        let normalized_gap = (self.safe_distance - surface_distance) / self.safe_distance;
        let penalty = normalized_gap * normalized_gap;
        (penalty * MAX_NON_OBSTACLE_COST as f64) as f32
    }

    #[inline]
    fn distance_heuristic_2d(&self, idx: u32, size_x: u32, target_x: u32, target_y: u32) -> f32 {
        let dx = (idx % size_x) as i32 - target_x as i32;
        let dy = (idx / size_x) as i32 - target_y as i32;
        ((dx * dx + dy * dy) as f32).sqrt()
    }
}
