// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Multi-goal management with heading modes.

use std::collections::HashSet;

use crate::collision::GridCollisionChecker;
use crate::node::{HybridMotionTable, NodeContext, NodeHybrid};
use crate::types::*;

pub type CoordinateVector = Vec<Coordinates>;

/// Manages multiple goal states for the A* search.
pub struct GoalManager {
    goals_set: HashSet<u64>,
    goals_state: Vec<GoalState>,
    goals_coordinate: CoordinateVector,
    ref_goal_coord: Coordinates,
}

impl GoalManager {
    pub fn new() -> Self {
        Self {
            goals_set: HashSet::new(),
            goals_state: Vec::new(),
            goals_coordinate: Vec::new(),
            ref_goal_coord: Coordinates::default(),
        }
    }

    pub fn goals_is_empty(&self) -> bool {
        self.goals_state.is_empty()
    }

    pub fn add_goal(&mut self, goal_index: u64) {
        self.goals_state.push(GoalState {
            index: goal_index,
            is_valid: true,
        });
    }

    pub fn clear(&mut self) {
        self.goals_set.clear();
        self.goals_state.clear();
        self.goals_coordinate.clear();
    }

    pub fn set_ref_goal_coordinates(&mut self, coord: Coordinates) {
        self.ref_goal_coord = coord;
    }

    pub fn has_goal_changed(&self, coord: Coordinates) -> bool {
        self.ref_goal_coord != coord
    }

    pub fn prepare_goals_for_analytic_expansion(
        &self,
        coarse_search_resolution: usize,
    ) -> (Vec<u64>, Vec<u64>) {
        let mut coarse_check_goals = Vec::new();
        let mut fine_check_goals = Vec::new();
        for (i, gs) in self.goals_state.iter().enumerate() {
            if gs.is_valid {
                if i % coarse_search_resolution == 0 {
                    coarse_check_goals.push(gs.index);
                } else {
                    fine_check_goals.push(gs.index);
                }
            }
        }
        (coarse_check_goals, fine_check_goals)
    }

    pub fn remove_invalid_goals(
        &mut self,
        tolerance: f32,
        collision_checker: &mut GridCollisionChecker,
        traverse_unknown: bool,
        graph: &hashbrown::HashMap<u64, NodeHybrid>,
        motion_table: &HybridMotionTable,
    ) {
        assert!(
            self.goals_set.is_empty() && self.goals_coordinate.is_empty(),
            "Goal set should be cleared before calling remove_invalid_goals"
        );

        let mut results: Vec<(u64, bool, Coordinates)> = Vec::new();
        for gs in &self.goals_state {
            if let Some(node) = graph.get(&gs.index) {
                let zone_valid = is_zone_valid(
                    node, tolerance, collision_checker, traverse_unknown, motion_table,
                );
                if zone_valid || node.is_node_valid {
                    results.push((gs.index, true, node.pose));
                } else {
                    results.push((gs.index, false, Coordinates::default()));
                }
            } else {
                results.push((gs.index, false, Coordinates::default()));
            }
        }

        for (i, gs) in self.goals_state.iter_mut().enumerate() {
            let (idx, valid, coords) = results[i];
            gs.is_valid = valid;
            if valid {
                self.goals_set.insert(idx);
                self.goals_coordinate.push(coords);
            }
        }
    }

    pub fn is_goal(&self, index: u64) -> bool {
        self.goals_set.contains(&index)
    }

    pub fn goals_set(&self) -> &HashSet<u64> {
        &self.goals_set
    }

    pub fn goals_state(&self) -> &[GoalState] {
        &self.goals_state
    }

    pub fn goals_coordinates(&self) -> &[Coordinates] {
        &self.goals_coordinate
    }
}

fn is_zone_valid(
    node: &NodeHybrid,
    radius: f32,
    collision_checker: &mut GridCollisionChecker,
    traverse_unknown: bool,
    motion_table: &HybridMotionTable,
) -> bool {
    if radius < 1.0 {
        return false;
    }

    let cm = collision_checker.costmap();
    let size_x = cm.size_x();
    let size_y = cm.size_y();
    let center = &node.pose;

    let min_x = (center.x - radius).floor().max(0.0) as u32;
    let min_y = (center.y - radius).floor().max(0.0) as u32;
    let max_x = (center.x + radius).ceil().min((size_x - 1) as f32) as u32;
    let max_y = (center.y + radius).ceil().min((size_y - 1) as f32) as u32;
    let radius_sq = radius * radius;

    let mut m = Coordinates::default();
    for mx in min_x..=max_x {
        for my in min_y..=max_y {
            m.x = mx as f32;
            m.y = my as f32;
            let dx = m.x - center.x;
            let dy = m.y - center.y;
            if dx * dx + dy * dy > radius_sq {
                continue;
            }
            let angle = (m.theta as u32).min(motion_table.num_angle_quantization - 1);
            let idx = NodeHybrid::get_index(
                mx, my, angle, motion_table.size_x, motion_table.num_angle_quantization,
            );
            let mut test_node = NodeHybrid::new(idx);
            test_node.set_pose(m);
            if test_node.is_node_valid(
                traverse_unknown,
                collision_checker,
                &NodeContext {
                    motion_table: motion_table.clone(),
                    obstacle_heuristic: crate::obstacle_heuristic::ObstacleHeuristic::new(),
                    distance_heuristic: crate::distance_heuristic::DistanceHeuristic::new(),
                },
            ) {
                return true;
            }
        }
    }
    false
}
