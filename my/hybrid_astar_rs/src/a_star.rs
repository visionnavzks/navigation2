// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! A* search algorithm for Hybrid A*.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::time::Instant;

use crate::analytic::AnalyticExpansion;
use crate::collision::GridCollisionChecker;
use crate::constants::*;
use crate::esdf::EsdfHolder;
use crate::goal_manager::{GoalManager, CoordinateVector};
use crate::node::{NodeContext, NodeHybrid};
use crate::types::*;

// ---------------------------------------------------------------------------
// Priority-queue element
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct QueueElement {
    cost: f32,
    index: u64,
}

impl Eq for QueueElement {}

impl PartialEq for QueueElement {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl PartialOrd for QueueElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// AStarAlgorithm
// ---------------------------------------------------------------------------

pub struct AStarAlgorithm {
    traverse_unknown: bool,
    is_initialized: bool,
    max_iterations: i32,
    max_on_approach_iterations: i32,
    terminal_checking_interval: i32,
    max_planning_time: f64,
    tolerance: f32,
    x_size: u32,
    y_size: u32,
    dim3_size: u32,
    coarse_search_resolution: usize,
    search_info: SearchInfo,

    start_index: Option<u64>,
    goal_manager: GoalManager,
    graph: hashbrown::HashMap<u64, NodeHybrid>,
    queue: BinaryHeap<QueueElement>,

    motion_model: MotionModel,
    best_heuristic_node: (f32, u64),

    collision_checker: Option<*mut GridCollisionChecker>,
    costmap: Option<*const crate::costmap::Costmap2D>,
    shared_ctx: Option<Box<NodeContext>>,
}

// Safety: single-threaded usage
unsafe impl Send for AStarAlgorithm {}

impl AStarAlgorithm {
    pub fn new(motion_model: MotionModel, search_info: SearchInfo) -> Self {
        let mut graph = hashbrown::HashMap::new();
        graph.reserve(100_000);
        Self {
            traverse_unknown: true,
            is_initialized: false,
            max_iterations: 0,
            max_on_approach_iterations: 1000,
            terminal_checking_interval: 5000,
            max_planning_time: 0.0,
            tolerance: 0.0,
            x_size: 0,
            y_size: 0,
            dim3_size: 0,
            coarse_search_resolution: 1,
            search_info,
            start_index: None,
            goal_manager: GoalManager::new(),
            graph,
            queue: BinaryHeap::new(),
            motion_model,
            best_heuristic_node: (f32::MAX, 0),
            collision_checker: None,
            costmap: None,
            shared_ctx: None,
        }
    }

    pub fn initialize(
        &mut self,
        allow_unknown: bool,
        max_iterations: i32,
        max_on_approach_iterations: i32,
        terminal_checking_interval: i32,
        max_planning_time: f64,
        lookup_table_size: f32,
        dim_3_size: u32,
    ) {
        self.traverse_unknown = allow_unknown;
        self.max_iterations = max_iterations;
        self.max_on_approach_iterations = max_on_approach_iterations;
        self.terminal_checking_interval = terminal_checking_interval;
        self.max_planning_time = max_planning_time;

        if !self.is_initialized {
            let mut ctx = NodeContext::new();
            ctx.distance_heuristic.precompute_distance_heuristic(
                lookup_table_size,
                self.motion_model,
                dim_3_size,
                &self.search_info,
                &mut ctx.motion_table,
            );
            self.shared_ctx = Some(Box::new(ctx));
        }
        self.is_initialized = true;
        self.dim3_size = dim_3_size;
    }

    pub fn set_collision_checker(&mut self, collision_checker: &mut GridCollisionChecker) {
        self.collision_checker = Some(collision_checker);
        let cm = collision_checker.costmap();
        self.costmap = Some(cm);
        let x_size = cm.size_x();
        let y_size = cm.size_y();

        self.clear_graph();

        if self.x_size != x_size || self.y_size != y_size {
            self.x_size = x_size;
            self.y_size = y_size;
        }

        if let Some(ctx) = &mut self.shared_ctx {
            NodeHybrid::init_motion_model(
                ctx,
                self.motion_model,
                self.x_size,
                self.y_size,
                self.dim3_size,
                &self.search_info,
            );
        }
    }

    pub fn set_esdf_resources(
        &mut self,
        holder: *const EsdfHolder,
        cost_check_points: Vec<f64>,
        robot_radius: f64,
        safe_distance: f64,
    ) {
        if let Some(ctx) = &mut self.shared_ctx {
            ctx.obstacle_heuristic.set_esdf_holder(holder);
            ctx.obstacle_heuristic.set_esdf_footprint_params(
                cost_check_points,
                robot_radius,
                safe_distance,
            );
        }
    }

    pub fn set_start(&mut self, mx: f32, my: f32, dim_3: u32) {
        let index = NodeHybrid::get_index(
            mx as u32,
            my as u32,
            dim_3,
            self.x_size,
            self.dim3_size,
        );
        self.add_to_graph(index);
        if let Some(node) = self.graph.get_mut(&index) {
            node.set_pose(Coordinates::new(mx, my, dim_3 as f32));
        }
        self.start_index = Some(index);
    }

    pub fn set_goal(
        &mut self,
        mx: f32,
        my: f32,
        dim_3: u32,
        goal_heading_mode: GoalHeadingMode,
        coarse_search_resolution: usize,
    ) {
        self.coarse_search_resolution = coarse_search_resolution;
        self.goal_manager.clear();
        let ref_goal_coord = Coordinates::new(mx, my, dim_3 as f32);

        if !self.search_info.cache_obstacle_heuristic || self.goal_manager.has_goal_changed(ref_goal_coord)
        {
            if self.start_index.is_none() {
                panic!("Start must be set before goal.");
            }
            let start = self.graph.get(&self.start_index.unwrap()).unwrap();
            if let Some(ctx) = &mut self.shared_ctx {
                let cm = unsafe { &*self.costmap.unwrap() };
                ctx.obstacle_heuristic.reset_obstacle_heuristic(
                    cm,
                    start.pose.x,
                    start.pose.y,
                    mx,
                    my,
                    ctx.motion_table.downsample_obstacle_heuristic,
                );
            }
        }

        self.goal_manager.set_ref_goal_coordinates(ref_goal_coord);

        let num_bins = self.shared_ctx.as_ref().unwrap().motion_table.num_angle_quantization;

        match goal_heading_mode {
            GoalHeadingMode::Default => {
                let index = NodeHybrid::get_index(
                    mx as u32, my as u32, dim_3, self.x_size, self.dim3_size,
                );
                self.add_to_graph(index);
                if let Some(node) = self.graph.get_mut(&index) {
                    node.set_pose(Coordinates::new(mx, my, dim_3 as f32));
                }
                self.goal_manager.add_goal(index);
            }
            GoalHeadingMode::Bidirectional => {
                let index = NodeHybrid::get_index(
                    mx as u32, my as u32, dim_3, self.x_size, self.dim3_size,
                );
                self.add_to_graph(index);
                if let Some(node) = self.graph.get_mut(&index) {
                    node.set_pose(Coordinates::new(mx, my, dim_3 as f32));
                }
                self.goal_manager.add_goal(index);

                let opposite = (dim_3 + num_bins / 2) % num_bins;
                let opp_index = NodeHybrid::get_index(
                    mx as u32, my as u32, opposite, self.x_size, self.dim3_size,
                );
                self.add_to_graph(opp_index);
                if let Some(node) = self.graph.get_mut(&opp_index) {
                    node.set_pose(Coordinates::new(mx, my, opposite as f32));
                }
                self.goal_manager.add_goal(opp_index);
            }
            GoalHeadingMode::AllDirection => {
                for i in 0..num_bins {
                    let index = NodeHybrid::get_index(
                        mx as u32, my as u32, i, self.x_size, self.dim3_size,
                    );
                    self.add_to_graph(index);
                    if let Some(node) = self.graph.get_mut(&index) {
                        node.set_pose(Coordinates::new(mx, my, i as f32));
                    }
                    self.goal_manager.add_goal(index);
                }
            }
            GoalHeadingMode::Unknown => {
                panic!("Goal heading is UNKNOWN.");
            }
        }
    }

    pub fn create_path(
        &mut self,
        path: &mut CoordinateVector,
        num_iterations: &mut i32,
        tolerance: f32,
        cancel_checker: impl Fn() -> bool,
        mut expansions_log: Option<&mut Vec<(f32, f32, f32)>>,
    ) -> bool {
        let start_time = Instant::now();
        self.tolerance = tolerance;
        self.best_heuristic_node = (f32::MAX, 0);
        self.clear_queue();

        if !self.are_inputs_valid() {
            return false;
        }

        let (coarse_check_goals, fine_check_goals) =
            self.goal_manager
                .prepare_goals_for_analytic_expansion(self.coarse_search_resolution);

        // Add start node
        if let Some(start_idx) = self.start_index {
            if let Some(node) = self.graph.get_mut(&start_idx) {
                node.accumulated_cost = 0.0;
            }
            self.queue.push(QueueElement {
                cost: 0.0,
                index: start_idx,
            });
        }

        let max_index = self.x_size as u64 * self.y_size as u64 * self.dim3_size as u64;
        let mut approach_iterations = 0i32;
        let _analytic_iterations = 0i32;
        let mut closest_distance = i32::MAX;

        let expander = AnalyticExpansion::new(
            self.motion_model,
            &self.search_info,
            self.traverse_unknown,
            self.dim3_size,
        );

        let mut iterations = 0i32;

        while iterations < self.max_iterations && !self.queue.is_empty() {
            if iterations % self.terminal_checking_interval == 0 {
                if cancel_checker() {
                    panic!("Planner was cancelled");
                }
                let planning_duration = start_time.elapsed();
                if planning_duration.as_secs_f64() >= self.max_planning_time {
                    return self.get_closest_path_within_tolerance(path);
                }
            }

            let element = self.queue.pop().unwrap();
            let current_index = element.index;

            // Log expansion
            if let Some(log) = &mut expansions_log {
                if let Some(node) = self.graph.get(&current_index) {
                    let cm = unsafe { &*self.costmap.unwrap() };
                    let coords = node.pose;
                    log.push((
                        (cm.origin_x() + (coords.x as f64 + 0.5) * cm.resolution()) as f32,
                        (cm.origin_y() + (coords.y as f64 + 0.5) * cm.resolution()) as f32,
                        self.shared_ctx.as_ref().unwrap().motion_table.get_angle_from_bin(coords.theta),
                    ));
                }
            }

            if let Some(node) = self.graph.get(&current_index) {
                if node.was_visited {
                    continue;
                }
            }

            iterations += 1;

            if let Some(node) = self.graph.get_mut(&current_index) {
                node.was_visited = true;
            }

            // Try analytic expansion
            let expansion_result = {
                let current_node = self.graph.get(&current_index).unwrap();
                expander.try_analytic_expansion(
                    current_node,
                    &coarse_check_goals,
                    &fine_check_goals,
                    self.goal_manager.goals_coordinates(),
                    &self.graph,
                    unsafe { &*self.collision_checker.unwrap() },
                    self.shared_ctx.as_ref().unwrap(),
                    &mut closest_distance,
                )
            };

            if let Some(expansion_nodes) = expansion_result {
                if !expansion_nodes.nodes.is_empty() {
                    // Set analytic path
                    if let Some(last) = expansion_nodes.nodes.last() {
                        if let Some(goal_node) = self.graph.get(&last.node_index) {
                            if self.goal_manager.is_goal(last.node_index) {
                                // Backtrace from goal
                                if let Some(ctx) = &self.shared_ctx {
                                    if let Some(p) = goal_node.backtrace_path(&self.graph, ctx) {
                                        *path = p;
                                        *num_iterations = iterations;
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Check if current node is goal
            if self.goal_manager.is_goal(current_index) {
                if let Some(node) = self.graph.get(&current_index) {
                    if let Some(ctx) = &self.shared_ctx {
                        if let Some(p) = node.backtrace_path(&self.graph, ctx) {
                            *path = p;
                            *num_iterations = iterations;
                            return true;
                        }
                    }
                }
            } else if self.best_heuristic_node.0 < self.tolerance {
                approach_iterations += 1;
                if approach_iterations >= self.max_on_approach_iterations {
                    let best_idx = self.best_heuristic_node.1;
                    if let Some(node) = self.graph.get(&best_idx) {
                        if let Some(ctx) = &self.shared_ctx {
                            if let Some(p) = node.backtrace_path(&self.graph, ctx) {
                                *path = p;
                                *num_iterations = iterations;
                                return true;
                            }
                        }
                    }
                }
            }

            // Expand neighbors
            let neighbors: Vec<(u64, MotionPose)> = {
                let current_node = self.graph.get(&current_index).unwrap();
                let cc = unsafe { &mut *self.collision_checker.unwrap() };
                current_node.get_neighbors(cc, self.traverse_unknown, self.shared_ctx.as_ref().unwrap())
            };

            for (nbr_index, proj) in &neighbors {
                if *nbr_index >= max_index {
                    continue;
                }

                self.add_to_graph(*nbr_index);

                let g_cost = {
                    let current = self.graph.get(&current_index).unwrap();
                    let traversal = current.get_traversal_cost(
                        self.graph.get(nbr_index).unwrap(),
                        self.shared_ctx.as_ref().unwrap(),
                    );
                    current.accumulated_cost + traversal
                };

                let should_update = {
                    if let Some(neighbor) = self.graph.get(nbr_index) {
                        !neighbor.was_visited && g_cost < neighbor.accumulated_cost
                    } else {
                        false
                    }
                };

                if should_update {
                    let cc = unsafe { &mut *self.collision_checker.unwrap() };
                    let traverse = self.traverse_unknown;
                    let ctx_ref = self.shared_ctx.as_ref().unwrap();

                    let mut should_add = false;
                    if let Some(neighbor) = self.graph.get_mut(nbr_index) {
                        // Set pose and validate
                        let initial_coords = neighbor.pose;
                        neighbor.set_pose(Coordinates::new(proj.x, proj.y, proj.theta));

                        let valid = neighbor.is_node_valid(traverse, cc, ctx_ref);
                        if valid {
                            neighbor.accumulated_cost = g_cost;
                            neighbor.parent = Some(current_index);
                            neighbor.set_motion_primitive_index(proj.turn_dir as u32, proj.turn_dir);
                            should_add = true;
                        } else {
                            neighbor.set_pose(initial_coords);
                        }
                    }

                    if should_add {
                        let heuristic = {
                            let neighbor = self.graph.get(nbr_index).unwrap();
                            let cm = unsafe { &*self.costmap.unwrap() };
                            let node_coords = NodeHybrid::get_coords(
                                neighbor.index,
                                cm.size_x(),
                                self.dim3_size,
                            );
                            let ctx = self.shared_ctx.as_ref().unwrap();
                            let goals = self.goal_manager.goals_coordinates().to_vec();
                            neighbor.get_heuristic_cost(
                                node_coords,
                                &goals,
                                ctx,
                            )
                        };

                        if heuristic < self.best_heuristic_node.0 {
                            self.best_heuristic_node = (heuristic, *nbr_index);
                        }

                        self.queue.push(QueueElement {
                            cost: g_cost + heuristic,
                            index: *nbr_index,
                        });
                    }
                }
            }
        }

        *num_iterations = iterations;
        self.get_closest_path_within_tolerance(path)
    }

    // -- Internal helpers ---------------------------------------------------

    fn add_to_graph(&mut self, index: u64) {
        self.graph.entry(index).or_insert_with(|| NodeHybrid::new(index));
    }

    fn get_closest_path_within_tolerance(&self, path: &mut CoordinateVector) -> bool {
        if self.best_heuristic_node.0 < self.tolerance {
            if let Some(node) = self.graph.get(&self.best_heuristic_node.1) {
                if let Some(ctx) = &self.shared_ctx {
                    if let Some(p) = node.backtrace_path(&self.graph, ctx) {
                        *path = p;
                        return true;
                    }
                }
            }
        }
        false
    }

    fn are_inputs_valid(&self) -> bool {
        if self.graph.is_empty() {
            panic!("Failed to compute path, no costmap given.");
        }
        if self.start_index.is_none() || self.goal_manager.goals_is_empty() {
            panic!("Failed to compute path, no valid start or goal given.");
        }
        // Note: remove_invalid_goals requires mutable access to collision checker
        // which we handle separately in the C++ version
        true
    }

    fn clear_queue(&mut self) {
        self.queue = BinaryHeap::new();
    }

    fn clear_graph(&mut self) {
        self.graph.clear();
        self.graph.reserve(100_000);
    }

    pub fn max_iterations(&self) -> i32 {
        self.max_iterations
    }

    pub fn context(&self) -> Option<&NodeContext> {
        self.shared_ctx.as_ref().map(|c| c.as_ref())
    }
}
