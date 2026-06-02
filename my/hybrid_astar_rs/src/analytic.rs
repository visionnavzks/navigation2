// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Analytic expansion — direct steering-function connections from the current
//! node to each candidate goal.

use crate::collision::GridCollisionChecker;
use crate::constants::*;
use crate::node::{NodeContext, NodeHybrid};
use crate::steering::State;
use crate::types::*;

/// Result of a single analytic expansion attempt.
#[derive(Clone)]
pub struct AnalyticExpansionNode {
    pub node_index: u64,
    pub initial_coords: Coordinates,
    pub proposed_coords: Coordinates,
}

/// A collection of analytic expansion nodes plus direction-change count.
#[derive(Clone, Default)]
pub struct AnalyticExpansionNodes {
    pub nodes: Vec<AnalyticExpansionNode>,
    pub direction_changes: i32,
}

impl AnalyticExpansionNodes {
    pub fn add(&mut self, node_index: u64, initial: Coordinates, proposed: Coordinates) {
        self.nodes.push(AnalyticExpansionNode {
            node_index,
            initial_coords: initial,
            proposed_coords: proposed,
        });
    }

    pub fn set_direction_changes(&mut self, changes: i32) {
        self.direction_changes = changes;
    }
}

/// Tries analytic expansions from the current node to all goal nodes.
pub struct AnalyticExpansion {
    motion_model: MotionModel,
    search_info: SearchInfo,
    _traverse_unknown: bool,
    dim_3_size: u32,
}

impl AnalyticExpansion {
    pub fn new(
        motion_model: MotionModel,
        search_info: &SearchInfo,
        traverse_unknown: bool,
        dim_3_size: u32,
    ) -> Self {
        Self {
            motion_model,
            search_info: search_info.clone(),
            _traverse_unknown: traverse_unknown,
            dim_3_size,
        }
    }

    pub fn try_analytic_expansion(
        &self,
        current_node: &NodeHybrid,
        coarse_check_goals: &[u64],
        fine_check_goals: &[u64],
        goals_coords: &[Coordinates],
        graph: &hashbrown::HashMap<u64, NodeHybrid>,
        collision_checker: &GridCollisionChecker,
        ctx: &NodeContext,
        closest_distance: &mut i32,
    ) -> Option<AnalyticExpansionNodes> {
        if self.motion_model != MotionModel::Dubin && self.motion_model != MotionModel::ReedsShepp {
            return None;
        }

        let cm = collision_checker.costmap();
        let node_coords =
            NodeHybrid::get_coords(current_node.index, cm.size_x(), self.dim_3_size);

        // Compute closest distance
        let h = current_node.get_heuristic_cost(node_coords, goals_coords, ctx);
        *closest_distance = (*closest_distance).min(h as i32);

        let _desired_iterations = (*closest_distance as f32 / self.search_info.analytic_expansion_ratio)
            .max(self.search_info.analytic_expansion_ratio.ceil()) as i32;

        let mut current_best_nodes: Option<AnalyticExpansionNodes> = None;
        let mut current_best_score = f32::MAX;

        // Try coarse goals first
        for &goal_idx in coarse_check_goals {
            if let Some(goal_node) = graph.get(&goal_idx) {
                if let Some(analytic_nodes) =
                    self.get_analytic_path(current_node, goal_node, graph, collision_checker, ctx)
                {
                    if !analytic_nodes.nodes.is_empty() {
                        let score = self.score_path(&analytic_nodes, graph, ctx);
                        if score < current_best_score {
                            current_best_nodes = Some(analytic_nodes);
                            current_best_score = score;
                        }
                    }
                }
            }
        }

        // Try fine goals if coarse succeeded
        if current_best_nodes.is_some() {
            for &goal_idx in fine_check_goals {
                if let Some(goal_node) = graph.get(&goal_idx) {
                    if let Some(analytic_nodes) =
                        self.get_analytic_path(current_node, goal_node, graph, collision_checker, ctx)
                    {
                        if !analytic_nodes.nodes.is_empty() {
                            let score = self.score_path(&analytic_nodes, graph, ctx);
                            if score < current_best_score {
                                current_best_nodes = Some(analytic_nodes);
                                current_best_score = score;
                            }
                        }
                    }
                }
            }
        }

        current_best_nodes
    }

    fn get_analytic_path(
        &self,
        node: &NodeHybrid,
        goal: &NodeHybrid,
        graph: &hashbrown::HashMap<u64, NodeHybrid>,
        collision_checker: &GridCollisionChecker,
        ctx: &NodeContext,
    ) -> Option<AnalyticExpansionNodes> {
        let ss = ctx.motion_table.state_space.as_ref()?;
        let cm = collision_checker.costmap();

        let from = State::new(
            node.pose.x as f64,
            node.pose.y as f64,
            ctx.motion_table.get_angle_from_bin(node.pose.theta) as f64,
        );
        let to = State::new(
            goal.pose.x as f64,
            goal.pose.y as f64,
            ctx.motion_table.get_angle_from_bin(goal.pose.theta) as f64,
        );

        let d = ss.distance(from, to) as f32;

        // Count direction changes
        let direction_changes = if self.motion_model == MotionModel::ReedsShepp {
            let controls = ss.get_controls(from, to);
            count_direction_changes(&controls)
        } else {
            0
        };

        let sqrt_2 = std::f32::consts::SQRT_2;

        if d > self.search_info.analytic_expansion_max_length || d < sqrt_2 {
            return None;
        }

        let num_intervals = (d / sqrt_2).floor() as u32;
        if num_intervals == 0 {
            return None;
        }

        let mut possible_nodes = AnalyticExpansionNodes::default();
        possible_nodes.nodes.reserve(num_intervals as usize);
        let mut node_costs: Vec<f32> = Vec::with_capacity(num_intervals as usize);

        let mut prev_index = node.index;
        let mut failure = false;

        for i in 1..=num_intervals {
            let s = ss.interpolate(from, to, i as f64 / num_intervals as f64);
            let theta = wrap_angle(s.theta);
            let angle = ctx.motion_table.get_angle(theta);

            let cell_x = s.x as u32;
            let cell_y = s.y as u32;
            let angle_bin = angle as u32;

            if cell_x >= cm.size_x() || cell_y >= cm.size_y() {
                failure = true;
                break;
            }

            let index = NodeHybrid::get_index(
                cell_x,
                cell_y,
                angle_bin.min(ctx.motion_table.num_angle_quantization - 1),
                ctx.motion_table.size_x,
                ctx.motion_table.num_angle_quantization,
            );

            if let Some(next) = graph.get(&index) {
                let proposed = Coordinates::new(s.x as f32, s.y as f32, angle);
                // Check collision (simplified: check costmap directly)
                let world_x = cm.origin_x() + (s.x + 0.5) * cm.resolution();
                let world_y = cm.origin_y() + (s.y + 0.5) * cm.resolution();
                let in_collision = if collision_checker.uses_esdf_footprint() {
                    collision_checker.in_collision_esdf(world_x, world_y, theta, self._traverse_unknown)
                } else {
                    if let Some((mx, my)) = cm.world_to_map(world_x, world_y) {
                        let cost = cm.get_cost(mx, my) as f32;
                        cost >= INSCRIBED_COST
                    } else {
                        true
                    }
                };

                if !in_collision && index != prev_index {
                    possible_nodes.add(index, next.pose, proposed);
                    node_costs.push(cm.get_cost_float(cell_x as f32, cell_y as f32) as f32);
                    prev_index = index;
                } else {
                    failure = true;
                    break;
                }
            } else {
                failure = true;
                break;
            }
        }

        if !failure {
            // Check max cost constraint
            let max_cost = self.search_info.analytic_expansion_max_cost;
            if let Some(max_node_cost) = node_costs.iter().cloned().reduce(f32::max) {
                if max_node_cost > max_cost {
                    let mut cost_exit_high_cost_region = false;
                    for &curr_cost in node_costs.iter().rev() {
                        if curr_cost <= max_cost {
                            cost_exit_high_cost_region = true;
                        } else if curr_cost > max_cost && cost_exit_high_cost_region {
                            failure = true;
                            break;
                        }
                    }

                    if failure
                        && d < (2.0 * std::f64::consts::PI * ctx.motion_table.min_turning_radius as f64) as f32
                        && self.search_info.analytic_expansion_max_cost_override
                    {
                        failure = false;
                    }
                }
            }
        }

        if failure {
            return None;
        }

        possible_nodes.set_direction_changes(direction_changes);
        Some(possible_nodes)
    }

    fn score_path(
        &self,
        expansion: &AnalyticExpansionNodes,
        graph: &hashbrown::HashMap<u64, NodeHybrid>,
        ctx: &NodeContext,
    ) -> f32 {
        if expansion.nodes.len() < 2 {
            return f32::MAX;
        }

        let weight = ctx.motion_table.cost_penalty;
        let mut score = 0.0f32;

        for node_pose in &expansion.nodes {
            let distance = ((expansion.nodes[1].proposed_coords.x - expansion.nodes[0].proposed_coords.x).powi(2)
                + (expansion.nodes[1].proposed_coords.y - expansion.nodes[0].proposed_coords.y).powi(2))
                .sqrt();
            let normalized_cost = graph
                .get(&node_pose.node_index)
                .map(|n| n.cell_cost / MAX_NON_OBSTACLE_COST)
                .unwrap_or(0.0);
            score += distance * (1.0 + weight * normalized_cost);
        }
        score
    }
}

fn count_direction_changes(controls: &[crate::steering::Control]) -> i32 {
    let mut changes = 0;
    let mut last_dir = 0i32;
    for ctrl in controls {
        if ctrl.delta_s.abs() < 1e-9 {
            continue;
        }
        let current_dir = if ctrl.delta_s > 0.0 { 1 } else { -1 };
        if last_dir != 0 && current_dir != last_dir {
            changes += 1;
        }
        last_dir = current_dir;
    }
    changes
}
