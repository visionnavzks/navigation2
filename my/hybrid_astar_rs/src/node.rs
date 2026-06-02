// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Search-node types and motion-table precomputation.
//!
//! [`NodeHybrid`] is the SE(2) node used by the A* search, and
//! [`HybridMotionTable`] precomputes the motion-primitive projections,
//! trigonometric values, and travel costs for each (model, resolution) pair.

use std::f32;

use crate::collision::GridCollisionChecker;
use crate::constants::*;
use crate::distance_heuristic::DistanceHeuristic;
use crate::obstacle_heuristic::ObstacleHeuristic;
use crate::steering::{self, SteeringStateSpacePtr};
use crate::types::*;

// ---------------------------------------------------------------------------
// HybridMotionTable
// ---------------------------------------------------------------------------

/// Precomputed motion primitives for a given grid resolution and motion model.
#[derive(Clone)]
pub struct HybridMotionTable {
    pub motion_model: MotionModel,
    pub projections: MotionPoses,
    pub size_x: u32,
    pub size_y: u32,
    pub num_angle_quantization: u32,
    pub num_angle_quantization_f: f32,
    pub min_turning_radius: f32,
    pub bin_size: f32,
    pub change_penalty: f32,
    pub non_straight_penalty: f32,
    pub cost_penalty: f32,
    pub reverse_penalty: f32,
    pub travel_distance_reward: f32,
    pub downsample_obstacle_heuristic: bool,
    pub use_quadratic_cost_penalty: bool,
    pub allow_primitive_interpolation: bool,
    pub state_space: Option<SteeringStateSpacePtr>,
    pub delta_xs: Vec<Vec<f64>>,
    pub delta_ys: Vec<Vec<f64>>,
    pub trig_values: Vec<TrigValues>,
    pub travel_costs: Vec<f32>,
}

impl Default for HybridMotionTable {
    fn default() -> Self {
        Self {
            motion_model: MotionModel::Unknown,
            projections: Vec::new(),
            size_x: 0,
            size_y: 0,
            num_angle_quantization: 0,
            num_angle_quantization_f: 0.0,
            min_turning_radius: 0.0,
            bin_size: 0.0,
            change_penalty: 0.0,
            non_straight_penalty: 0.0,
            cost_penalty: 0.0,
            reverse_penalty: 0.0,
            travel_distance_reward: 0.0,
            downsample_obstacle_heuristic: false,
            use_quadratic_cost_penalty: false,
            allow_primitive_interpolation: false,
            state_space: None,
            delta_xs: Vec::new(),
            delta_ys: Vec::new(),
            trig_values: Vec::new(),
            travel_costs: Vec::new(),
        }
    }
}

impl HybridMotionTable {
    pub fn init_dubin(
        &mut self,
        size_x: u32,
        size_y: u32,
        num_angle_quantization: u32,
        search_info: &SearchInfo,
    ) {
        self.init_common(
            size_x,
            size_y,
            num_angle_quantization,
            search_info,
            MotionModel::Dubin,
        );
    }

    pub fn init_reeds_shepp(
        &mut self,
        size_x: u32,
        size_y: u32,
        num_angle_quantization: u32,
        search_info: &SearchInfo,
    ) {
        self.init_common(
            size_x,
            size_y,
            num_angle_quantization,
            search_info,
            MotionModel::ReedsShepp,
        );
    }

    fn init_common(
        &mut self,
        size_x_in: u32,
        size_y_in: u32,
        num_angle_quantization_in: u32,
        search_info: &SearchInfo,
        model: MotionModel,
    ) {
        self.change_penalty = search_info.change_penalty;
        self.non_straight_penalty = search_info.non_straight_penalty;
        self.cost_penalty = search_info.cost_penalty;
        self.reverse_penalty = search_info.reverse_penalty;
        self.travel_distance_reward = 1.0 - search_info.retrospective_penalty;
        self.downsample_obstacle_heuristic = search_info.downsample_obstacle_heuristic;
        self.use_quadratic_cost_penalty = search_info.use_quadratic_cost_penalty;

        // Skip re-init if nothing changed
        if num_angle_quantization_in == self.num_angle_quantization
            && (self.min_turning_radius - search_info.minimum_turning_radius).abs() < 1e-6
            && self.allow_primitive_interpolation == search_info.allow_primitive_interpolation
            && self.motion_model == model
        {
            return;
        }

        self.size_x = size_x_in;
        self.size_y = size_y_in;
        self.num_angle_quantization = num_angle_quantization_in;
        self.num_angle_quantization_f = num_angle_quantization_in as f32;
        self.min_turning_radius = search_info.minimum_turning_radius;
        self.allow_primitive_interpolation = search_info.allow_primitive_interpolation;
        self.motion_model = model;

        let asin_arg = (std::f64::consts::SQRT_2 / (2.0 * self.min_turning_radius as f64)).min(1.0);
        let angle_rad = 2.0 * asin_arg;
        self.bin_size = std::f32::consts::TAU / self.num_angle_quantization as f32;
        let increments = if (angle_rad as f32) < self.bin_size {
            1.0f32
        } else {
            (angle_rad as f32 / self.bin_size).ceil()
        };
        let angle = increments * self.bin_size;

        let delta_x = self.min_turning_radius * angle.sin();
        let delta_y = self.min_turning_radius - self.min_turning_radius * angle.cos();
        let delta_dist = (delta_x * delta_x + delta_y * delta_y).sqrt();

        self.projections.clear();

        // Forward + Left + Right (shared by both models)
        self.projections
            .push(MotionPose::new(delta_dist, 0.0, 0.0, TurnDirection::Forward));
        self.projections
            .push(MotionPose::new(delta_x, delta_y, increments, TurnDirection::Left));
        self.projections
            .push(MotionPose::new(delta_x, -delta_y, -increments, TurnDirection::Right));

        if model == MotionModel::ReedsShepp {
            self.projections
                .push(MotionPose::new(-delta_dist, 0.0, 0.0, TurnDirection::Reverse));
            self.projections
                .push(MotionPose::new(-delta_x, delta_y, -increments, TurnDirection::RevLeft));
            self.projections
                .push(MotionPose::new(-delta_x, -delta_y, increments, TurnDirection::RevRight));
        }

        let base_count: u32 = if model == MotionModel::ReedsShepp { 6 } else { 3 };

        // Primitive interpolation
        if search_info.allow_primitive_interpolation && increments > 1.0 {
            let mut proj = std::mem::take(&mut self.projections);
            proj.reserve(
                (base_count + 2 * base_count * (increments as u32 - 1)) as usize,
            );
            for i in 1..increments as u32 {
                let angle_n = i as f32 * self.bin_size;
                let turning_rad_n = delta_dist / (2.0 * (angle_n / 2.0).sin());
                let dx_n = turning_rad_n * angle_n.sin();
                let dy_n = turning_rad_n - turning_rad_n * angle_n.cos();

                proj.push(MotionPose::new(
                    dx_n,
                    dy_n,
                    i as f32,
                    TurnDirection::Left,
                ));
                proj.push(MotionPose::new(
                    dx_n,
                    -dy_n,
                    -(i as f32),
                    TurnDirection::Right,
                ));
                if model == MotionModel::ReedsShepp {
                    proj.push(MotionPose::new(
                        -dx_n,
                        dy_n,
                        -(i as f32),
                        TurnDirection::RevLeft,
                    ));
                    proj.push(MotionPose::new(
                        -dx_n,
                        -dy_n,
                        i as f32,
                        TurnDirection::RevRight,
                    ));
                }
            }
            self.projections = proj;
        }

        // Create steering state space
        self.state_space = Some(steering::create_steering_state_space(
            model,
            self.min_turning_radius as f64,
        ));

        // Pre-compute delta_xs, delta_ys, trig_values
        let n_prim = self.projections.len();
        let n_ang = self.num_angle_quantization as usize;
        self.delta_xs = vec![vec![0.0; n_ang]; n_prim];
        self.delta_ys = vec![vec![0.0; n_ang]; n_prim];
        self.trig_values = vec![(0.0, 0.0); n_ang];

        for i in 0..n_prim {
            for j in 0..n_ang {
                let theta = self.bin_size * j as f32;
                let cos_t = theta.cos() as f64;
                let sin_t = theta.sin() as f64;
                if i == 0 {
                    self.trig_values[j] = (cos_t, sin_t);
                }
                self.delta_xs[i][j] =
                    self.projections[i].x as f64 * cos_t - self.projections[i].y as f64 * sin_t;
                self.delta_ys[i][j] =
                    self.projections[i].x as f64 * sin_t + self.projections[i].y as f64 * cos_t;
            }
        }

        // Travel costs
        self.travel_costs = vec![0.0; n_prim];
        for i in 0..n_prim {
            let turn_dir = self.projections[i].turn_dir;
            if turn_dir != TurnDirection::Forward && turn_dir != TurnDirection::Reverse {
                let arc_angle = self.projections[i].theta * self.bin_size;
                let turning_rad = delta_dist / (2.0 * (arc_angle / 2.0).sin());
                self.travel_costs[i] = turning_rad * arc_angle;
            } else {
                self.travel_costs[i] = delta_dist;
            }
        }
    }

    /// Project all motion primitives from the given node.
    pub fn get_projections(&self, node_x: f32, node_y: f32, node_heading: f32) -> MotionPoses {
        let mut result = MotionPoses::with_capacity(self.projections.len());
        for (i, proj) in self.projections.iter().enumerate() {
            let new_heading =
                wrap_bin_index((node_heading + proj.theta) as i32, self.num_angle_quantization);
            result.push(MotionPose::new(
                self.delta_xs[i][node_heading as usize] as f32 + node_x,
                self.delta_ys[i][node_heading as usize] as f32 + node_y,
                new_heading as f32,
                proj.turn_dir,
            ));
        }
        result
    }

    pub fn get_closest_angular_bin(&self, theta: f64) -> u32 {
        let bin = (wrap_angle(theta) / self.bin_size as f64).round() as u32;
        if bin < self.num_angle_quantization {
            bin
        } else {
            0
        }
    }

    pub fn get_angle_from_bin(&self, bin_idx: f32) -> f32 {
        bin_idx * self.bin_size
    }

    pub fn get_angle(&self, theta: f64) -> f32 {
        (theta / self.bin_size as f64) as f32
    }
}

// ---------------------------------------------------------------------------
// NodeContext — shared state for all nodes in a search
// ---------------------------------------------------------------------------

/// Shared context bound to a single search instance.
pub struct NodeContext {
    pub motion_table: HybridMotionTable,
    pub obstacle_heuristic: ObstacleHeuristic,
    pub distance_heuristic: DistanceHeuristic,
}

impl NodeContext {
    pub fn new() -> Self {
        Self {
            motion_table: HybridMotionTable::default(),
            obstacle_heuristic: ObstacleHeuristic::new(),
            distance_heuristic: DistanceHeuristic::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// NodeHybrid — SE(2) search node
// ---------------------------------------------------------------------------

/// A node in the SE(2) search graph.
#[derive(Clone)]
pub struct NodeHybrid {
    pub parent: Option<u64>, // index of parent in the graph
    pub pose: Coordinates,

    pub cell_cost: f32,
    pub accumulated_cost: f32,
    pub index: u64,
    pub was_visited: bool,
    pub motion_primitive_index: u32,
    pub turn_dir: TurnDirection,
    pub is_node_valid: bool,
}

impl NodeHybrid {
    pub fn new(index: u64) -> Self {
        Self {
            parent: None,
            pose: Coordinates::default(),
            cell_cost: f32::NAN,
            accumulated_cost: f32::MAX,
            index,
            was_visited: false,
            motion_primitive_index: u32::MAX,
            turn_dir: TurnDirection::Unknown,
            is_node_valid: false,
        }
    }

    /// Linear index from discrete (x, y, angle) coordinates.
    #[inline]
    pub fn get_index(x: u32, y: u32, angle: u32, width: u32, angle_quantization: u32) -> u64 {
        angle as u64 + x as u64 * angle_quantization as u64
            + y as u64 * width as u64 * angle_quantization as u64
    }

    /// Recover coordinates from a linear index.
    #[inline]
    pub fn get_coords(index: u64, width: u32, angle_quantization: u32) -> Coordinates {
        let aq = angle_quantization as u64;
        let w = width as u64;
        Coordinates::new(
            ((index / aq) % w) as f32,
            (index / (aq * w)) as f32,
            (index % aq) as f32,
        )
    }

    pub fn set_pose(&mut self, pose: Coordinates) {
        self.pose = pose;
    }

    pub fn reset(&mut self) {
        self.parent = None;
        self.cell_cost = f32::NAN;
        self.accumulated_cost = f32::MAX;
        self.was_visited = false;
        self.motion_primitive_index = u32::MAX;
        self.pose = Coordinates::default();
        self.is_node_valid = false;
    }

    pub fn set_motion_primitive_index(&mut self, idx: u32, turn_dir: TurnDirection) {
        self.motion_primitive_index = idx;
        self.turn_dir = turn_dir;
    }

    /// Validate this node against the collision checker.
    pub fn is_node_valid(
        &mut self,
        traverse_unknown: bool,
        collision_checker: &mut GridCollisionChecker,
        ctx: &NodeContext,
    ) -> bool {
        if !self.cell_cost.is_nan() {
            return self.is_node_valid;
        }

        if collision_checker.uses_esdf_footprint() {
            let cm = collision_checker.costmap();
            let wx = cm.origin_x()
                + (self.pose.x as f64 + 0.5) * cm.resolution();
            let wy = cm.origin_y()
                + (self.pose.y as f64 + 0.5) * cm.resolution();
            let theta =
                ctx.motion_table.get_angle_from_bin(self.pose.theta) as f64;
            self.is_node_valid =
                !collision_checker.in_collision_esdf(wx, wy, theta, traverse_unknown);
            let penalty = collision_checker.get_soft_penalty(wx, wy, theta);
            self.cell_cost = (penalty * MAX_NON_OBSTACLE_COST as f64) as f32;
            return self.is_node_valid;
        }

        self.is_node_valid =
            !collision_checker.in_collision(self.pose.x, self.pose.y, self.pose.theta, traverse_unknown);
        self.cell_cost = collision_checker.get_cost();
        self.is_node_valid
    }

    /// Compute the traversal cost from this node to `child`.
    pub fn get_traversal_cost(&self, child: &NodeHybrid, ctx: &NodeContext) -> f32 {
        let normalized_cost = child.cell_cost / MAX_NON_OBSTACLE_COST;
        if normalized_cost.is_nan() {
            panic!("Node attempted to get traversal cost without a known SE2 collision cost!");
        }

        let mt = &ctx.motion_table;
        let mut travel_cost_raw = mt.travel_costs[child.motion_primitive_index as usize];

        if mt.use_quadratic_cost_penalty {
            travel_cost_raw *= mt.travel_distance_reward
                + mt.cost_penalty * normalized_cost * normalized_cost;
        } else {
            travel_cost_raw *= mt.travel_distance_reward + mt.cost_penalty * normalized_cost;
        }

        let travel_cost = if child.turn_dir == TurnDirection::Forward
            || child.turn_dir == TurnDirection::Reverse
            || self.motion_primitive_index == u32::MAX
        {
            travel_cost_raw
        } else if self.turn_dir == child.turn_dir {
            travel_cost_raw * mt.non_straight_penalty
        } else {
            travel_cost_raw * (mt.non_straight_penalty + mt.change_penalty)
        };

        if child.turn_dir == TurnDirection::RevRight
            || child.turn_dir == TurnDirection::RevLeft
            || child.turn_dir == TurnDirection::Reverse
        {
            travel_cost * mt.reverse_penalty
        } else {
            travel_cost
        }
    }

    /// Initialize the motion model for the shared context.
    pub fn init_motion_model(ctx: &mut NodeContext, motion_model: MotionModel, size_x: u32, size_y: u32, num_angle_quantization: u32, search_info: &SearchInfo) {
        match motion_model {
            MotionModel::Dubin => {
                ctx.motion_table
                    .init_dubin(size_x, size_y, num_angle_quantization, search_info);
            }
            MotionModel::ReedsShepp => {
                ctx.motion_table
                    .init_reeds_shepp(size_x, size_y, num_angle_quantization, search_info);
            }
            _ => panic!("Invalid motion model for Hybrid A*."),
        }
    }

    /// Get heuristic cost from obstacle heuristic + distance heuristic.
    pub fn get_heuristic_cost(
        &self,
        node_coords: Coordinates,
        goals_coords: &[Coordinates],
        ctx: &NodeContext,
    ) -> f32 {
        let obstacle_heuristic = ctx.obstacle_heuristic.get_obstacle_heuristic(
            node_coords,
            ctx.motion_table.cost_penalty,
            ctx.motion_table.use_quadratic_cost_penalty,
            ctx.motion_table.downsample_obstacle_heuristic,
        );
        let mut distance_heuristic = f32::MAX;
        for goal in goals_coords {
            distance_heuristic = distance_heuristic.min(
                ctx.distance_heuristic
                    .get_distance_heuristic(node_coords, *goal, obstacle_heuristic, &ctx.motion_table),
            );
        }
        obstacle_heuristic.max(distance_heuristic)
    }

    /// Expand neighbors.
    pub fn get_neighbors(
        &self,
        _collision_checker: &mut GridCollisionChecker,
        _traverse_unknown: bool,
        ctx: &NodeContext,
    ) -> Vec<(u64, MotionPose)> {
        let motion_projections = ctx.motion_table.get_projections(self.pose.x, self.pose.y, self.pose.theta);
        let mut neighbors = Vec::new();

        for (_i, proj) in motion_projections.iter().enumerate() {
            let px = proj.x as i32;
            let py = proj.y as i32;
            if px < 0 || py < 0 || px >= ctx.motion_table.size_x as i32 || py >= ctx.motion_table.size_y as i32 {
                continue;
            }
            let index = Self::get_index(
                px as u32,
                py as u32,
                proj.theta as u32,
                ctx.motion_table.size_x,
                ctx.motion_table.num_angle_quantization,
            );
            neighbors.push((index, MotionPose::new(proj.x, proj.y, proj.theta, proj.turn_dir)));
        }
        neighbors
    }

    /// Backtrace the path from this node to the start.
    pub fn backtrace_path(
        &self,
        graph: &hashbrown::HashMap<u64, NodeHybrid>,
        ctx: &NodeContext,
    ) -> Option<Vec<Coordinates>> {
        let mut path = Vec::new();
        let mut current = self;

        loop {
            path.push(Coordinates::new(
                current.pose.x,
                current.pose.y,
                ctx.motion_table.get_angle_from_bin(current.pose.theta),
            ));
            match current.parent {
                Some(parent_idx) => {
                    current = graph.get(&parent_idx)?;
                }
                None => break,
            }
        }
        Some(path)
    }
}
