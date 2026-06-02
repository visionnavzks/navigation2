// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Path smoother with iterative weighted-averaging and boundary-condition
//! enforcement using Dubin/Reeds-Shepp expansions.

use std::f64::consts::PI;

use crate::costmap::Costmap2D;
use crate::steering::{self, State};
use crate::types::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct PathSegment {
    start: usize,
    end: usize,
}

fn find_directional_path_segments(path: &[Pose], is_holonomic: bool) -> Vec<PathSegment> {
    let mut segments = Vec::new();
    if path.is_empty() {
        return segments;
    }
    let mut seg_start = 0;
    for i in 1..path.len() {
        let dy = path[i].y - path[i - 1].y;
        let dx = path[i].x - path[i - 1].x;
        if !is_holonomic && (dx.abs() > 1e-6 || dy.abs() > 1e-6) {
            let angle = dy.atan2(dx);
            let prev_angle = if i >= 2 {
                let pdy = path[i - 1].y - path[i - 2].y;
                let pdx = path[i - 1].x - path[i - 2].x;
                pdy.atan2(pdx)
            } else {
                angle
            };
            let mut angle_diff = (angle - prev_angle).abs() % (2.0 * PI);
            if angle_diff > PI {
                angle_diff = 2.0 * PI - angle_diff;
            }
            if angle_diff > PI / 2.0 {
                segments.push(PathSegment {
                    start: seg_start,
                    end: i - 1,
                });
                seg_start = i - 1;
            }
        }
    }
    segments.push(PathSegment {
        start: seg_start,
        end: path.len() - 1,
    });
    segments
}

fn update_approximate_path_orientations(path: &mut [Pose], _is_holonomic: bool) {
    if path.len() < 2 {
        return;
    }
    for i in 0..path.len() - 1 {
        let dx = path[i + 1].x - path[i].x;
        let dy = path[i + 1].y - path[i].y;
        if dx.abs() > 1e-6 || dy.abs() > 1e-6 {
            path[i].theta = dy.atan2(dx);
        }
    }
    if path.len() > 1 {
        let last = path.len() - 1;
        path[last].theta = path[last - 1].theta;
    }
}

fn get_field_by_dim(pose: &Pose, dim: usize) -> f64 {
    match dim {
        0 => pose.x,
        1 => pose.y,
        _ => 0.0,
    }
}

fn set_field_by_dim(pose: &mut Pose, dim: usize, value: f64) {
    match dim {
        0 => pose.x = value,
        1 => pose.y = value,
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Boundary expansion types
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct BoundaryPoint {
    x: f64,
    y: f64,
    theta: f64,
}

#[derive(Clone, Default)]
struct BoundaryExpansion {
    path_end_idx: usize,
    expansion_path_length: f64,
    original_path_length: f64,
    pts: Vec<BoundaryPoint>,
    in_collision: bool,
}

// ---------------------------------------------------------------------------
// Smoother
// ---------------------------------------------------------------------------

pub struct Smoother {
    min_turning_rad: f64,
    tolerance: f64,
    data_w: f64,
    smooth_w: f64,
    max_its: usize,
    refinement_ctr: usize,
    refinement_num: usize,
    is_holonomic: bool,
    do_refinement: bool,
    state_space: steering::SteeringStateSpacePtr,
}

impl Smoother {
    pub fn new(params: &crate::types::SmootherParams) -> Self {
        Self {
            min_turning_rad: 0.0,
            tolerance: params.tolerance,
            max_its: params.max_its,
            data_w: params.w_data,
            smooth_w: params.w_smooth,
            refinement_ctr: 0,
            refinement_num: params.refinement_num,
            is_holonomic: params.holonomic,
            do_refinement: params.do_refinement,
            state_space: steering::create_steering_state_space(crate::constants::MotionModel::Dubin, 1.0),
        }
    }

    pub fn initialize(&mut self, min_turning_radius: f64) {
        self.min_turning_rad = min_turning_radius;
        self.state_space =
            steering::create_steering_state_space(crate::constants::MotionModel::Dubin, min_turning_radius);
    }

    pub fn smooth(&mut self, path: &mut Path, costmap: &Costmap2D, max_time: f64) -> bool {
        if self.max_its == 0 {
            return false;
        }

        let start = std::time::Instant::now();
        let mut _time_remaining = max_time;
        let mut success = true;
        let mut _reversing_segment = false;

        let path_segments = find_directional_path_segments(path, self.is_holonomic);

        for seg in &path_segments {
            if seg.end - seg.start > 10 {
                let mut curr_path_segment: Path = path[seg.start..=seg.end].to_vec();

                let elapsed = start.elapsed().as_secs_f64();
                _time_remaining = (max_time - elapsed).max(0.0);
                self.refinement_ctr = 0;

                let start_pose = curr_path_segment[0];
                let goal_pose = *curr_path_segment.last().unwrap();

                let local_success =
                    self.smooth_impl(&mut curr_path_segment, &_reversing_segment, costmap, _time_remaining);
                success = success && local_success;

                if !self.is_holonomic && local_success {
                    self.enforce_start_boundary_conditions(
                        start_pose,
                        &mut curr_path_segment,
                        costmap,
                        _reversing_segment,
                    );
                    self.enforce_end_boundary_conditions(
                        goal_pose,
                        &mut curr_path_segment,
                        costmap,
                        _reversing_segment,
                    );
                }

                path[seg.start..=seg.end]
                    .copy_from_slice(&curr_path_segment);
            }
        }

        success
    }

    fn smooth_impl(
        &mut self,
        path: &mut Path,
        reversing_segment: &mut bool,
        costmap: &Costmap2D,
        max_time: f64,
    ) -> bool {
        let start = std::time::Instant::now();
        let mut its = 0usize;
        let mut change = self.tolerance;
        let path_size = path.len();
        let mut new_path = path.clone();
        let mut last_path = path.clone();

        while change >= self.tolerance {
            its += 1;
            change = 0.0;

            if its >= self.max_its {
                *path = last_path;
                update_approximate_path_orientations(path, self.is_holonomic);
                return false;
            }

            if start.elapsed().as_secs_f64() > max_time {
                *path = last_path;
                update_approximate_path_orientations(path, self.is_holonomic);
                return false;
            }

            for i in 1..path_size - 1 {
                for j in 0..2 {
                    let x_i = get_field_by_dim(&path[i], j);
                    let y_i = get_field_by_dim(&new_path[i], j);
                    let y_m1 = get_field_by_dim(&new_path[i - 1], j);
                    let y_ip1 = get_field_by_dim(&new_path[i + 1], j);

                    let new_y =
                        y_i + self.data_w * (x_i - y_i) + self.smooth_w * (y_ip1 + y_m1 - 2.0 * y_i);
                    set_field_by_dim(&mut new_path[i], j, new_y);
                    change += (new_y - y_i).abs();
                }

                // Check cost
                let wx = new_path[i].x;
                let wy = new_path[i].y;
                if let Some((mx, my)) = costmap.world_to_map(wx, wy) {
                    let cost = costmap.get_cost(mx, my) as f32;
                    if cost > crate::constants::MAX_NON_OBSTACLE_COST
                        && cost != crate::constants::UNKNOWN_COST
                    {
                        *path = last_path;
                        update_approximate_path_orientations(path, self.is_holonomic);
                        return false;
                    }
                }
            }

            last_path = new_path.clone();
        }

        if self.do_refinement && self.refinement_ctr < self.refinement_num {
            self.refinement_ctr += 1;
            let elapsed = start.elapsed().as_secs_f64();
            let remaining = (max_time - elapsed).max(0.0);
            if !self.smooth_impl(&mut new_path, reversing_segment, costmap, remaining) {
                return false;
            }
        }

        update_approximate_path_orientations(&mut new_path, self.is_holonomic);
        *path = new_path;
        true
    }

    fn enforce_start_boundary_conditions(
        &self,
        start_pose: Pose,
        path: &mut Path,
        costmap: &Costmap2D,
        reversing_segment: bool,
    ) {
        let expansions =
            self.generate_boundary_expansion_points(path.iter(), path.len());

        let mut expansions = expansions;
        for expansion in &mut expansions {
            if expansion.path_end_idx == 0 {
                continue;
            }
            if !reversing_segment {
                self.find_boundary_expansion(
                    start_pose,
                    path[expansion.path_end_idx],
                    expansion,
                    costmap,
                );
            } else {
                self.find_boundary_expansion(
                    path[expansion.path_end_idx],
                    start_pose,
                    expansion,
                    costmap,
                );
            }
        }

        let best_idx = self.find_shortest_boundary_expansion_idx(&expansions);
        if best_idx >= expansions.len() {
            return;
        }

        let best = &expansions[best_idx];
        let mut pts = best.pts.clone();
        if reversing_segment {
            pts.reverse();
        }
        for (i, pt) in pts.iter().enumerate() {
            path[i].x = pt.x;
            path[i].y = pt.y;
            path[i].theta = pt.theta;
        }
    }

    fn enforce_end_boundary_conditions(
        &self,
        end_pose: Pose,
        path: &mut Path,
        costmap: &Costmap2D,
        reversing_segment: bool,
    ) {
        let reversed: Vec<Pose> = path.iter().rev().cloned().collect();
        let expansions =
            self.generate_boundary_expansion_points(reversed.iter(), reversed.len());

        let mut expansions = expansions;
        let mut _best_starting_idx = 0;
        for expansion in &mut expansions {
            if expansion.path_end_idx == 0 {
                continue;
            }
            let starting_idx = reversed.len() - expansion.path_end_idx - 1;
            _best_starting_idx = starting_idx;
            if !reversing_segment {
                self.find_boundary_expansion(
                    reversed[starting_idx],
                    end_pose,
                    expansion,
                    costmap,
                );
            } else {
                self.find_boundary_expansion(
                    end_pose,
                    reversed[starting_idx],
                    expansion,
                    costmap,
                );
            }
        }

        let best_idx = self.find_shortest_boundary_expansion_idx(&expansions);
        if best_idx >= expansions.len() {
            return;
        }

        let best = &expansions[best_idx];
        let mut pts = best.pts.clone();
        if reversing_segment {
            pts.reverse();
        }
        let expansion_start = path.len() - best.path_end_idx - 1;
        for (i, pt) in pts.iter().enumerate() {
            path[expansion_start + i].x = pt.x;
            path[expansion_start + i].y = pt.y;
            path[expansion_start + i].theta = pt.theta;
        }
    }

    fn find_boundary_expansion(
        &self,
        start: Pose,
        end: Pose,
        expansion: &mut BoundaryExpansion,
        costmap: &Costmap2D,
    ) {
        let from = State::new(start.x, start.y, start.theta);
        let to = State::new(end.x, end.y, end.theta);

        let d = self.state_space.distance(from, to);
        if d > 2.0 * expansion.original_path_length {
            return;
        }

        let mut x_m = start.x;
        let mut y_m = start.y;

        for i in 0..=expansion.path_end_idx {
            let s = self.state_space.interpolate(
                from,
                to,
                i as f64 / expansion.path_end_idx as f64,
            );
            let theta = wrap_angle(s.theta);

            if let Some((mx, my)) = costmap.world_to_map(s.x, s.y) {
                let cost = costmap.get_cost(mx, my) as f32;
                if cost >= crate::constants::INSCRIBED_COST {
                    expansion.in_collision = true;
                }
            }

            expansion.expansion_path_length += ((s.x - x_m).powi(2) + (s.y - y_m).powi(2)).sqrt();
            x_m = s.x;
            y_m = s.y;

            expansion.pts.push(BoundaryPoint {
                x: s.x,
                y: s.y,
                theta,
            });
        }
    }

    fn generate_boundary_expansion_points<'a, I>(
        &self,
        start: I,
        _len: usize,
    ) -> Vec<BoundaryExpansion>
    where
        I: Iterator<Item = &'a Pose>,
    {
        let distances = [
            self.min_turning_rad,
            2.0 * self.min_turning_rad,
            PI * self.min_turning_rad,
            2.0 * PI * self.min_turning_rad,
        ];

        let mut expansions: Vec<BoundaryExpansion> = distances
            .iter()
            .map(|_| BoundaryExpansion::default())
            .collect();

        let mut curr_dist = 0.0;
        let mut x_last = 0.0;
        let mut y_last = 0.0;
        let mut curr_dist_idx = 0;
        let mut count = 0usize;

        for pt in start {
            if count > 0 {
                curr_dist += ((pt.x - x_last).powi(2) + (pt.y - y_last).powi(2)).sqrt();
            }
            x_last = pt.x;
            y_last = pt.y;

            if curr_dist_idx < expansions.len() && curr_dist >= distances[curr_dist_idx] {
                expansions[curr_dist_idx].path_end_idx = count;
                expansions[curr_dist_idx].original_path_length = curr_dist;
                curr_dist_idx += 1;
            }

            if curr_dist_idx == expansions.len() {
                break;
            }
            count += 1;
        }

        expansions
    }

    fn find_shortest_boundary_expansion_idx(&self, expansions: &[BoundaryExpansion]) -> usize {
        let mut min_length = f64::INFINITY;
        let mut best_idx = expansions.len(); // sentinel = not found

        for (idx, exp) in expansions.iter().enumerate() {
            if exp.expansion_path_length < min_length
                && !exp.in_collision
                && exp.path_end_idx > 0
                && exp.expansion_path_length > 0.0
            {
                min_length = exp.expansion_path_length;
                best_idx = idx;
            }
        }

        best_idx
    }
}
