// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! High-level planner façade — `SmacPlannerHybrid`.
//!
//! This module mirrors the C++ `SmacPlannerHybrid` class that ties together
//! all the sub-systems (A*, collision checker, smoother, costmap downsampler)
//! into a single `create_plan` entry-point.

use std::f64::consts::PI;

use crate::a_star::AStarAlgorithm;
use crate::collision::GridCollisionChecker;
use crate::constants::*;
use crate::costmap::Costmap2D;
use crate::costmap_downsampler::CostmapDownsampler;
use crate::esdf::EsdfHolder;
use crate::smoother::Smoother;
use crate::types::*;
use crate::utils::{find_circumscribed_cost, get_world_coords};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration struct for [`SmacPlannerHybrid`].
#[derive(Clone)]
pub struct SmacPlannerHybridConfig {
    pub downsample_costmap: bool,
    pub downsampling_factor: u32,
    pub angle_quantization_bins: u32,
    pub tolerance: f32,
    pub allow_unknown: bool,
    pub max_iterations: i32,
    pub max_on_approach_iterations: i32,
    pub terminal_checking_interval: i32,
    pub smooth_path: bool,
    pub max_planning_time: f64,
    pub lookup_table_size: f64,
    pub debug_visualizations: bool,
    pub motion_model_for_search: String,
    pub goal_heading_mode: String,
    pub coarse_search_resolution: usize,

    pub search_info: SearchInfo,
    pub smoother_params: SmootherParams,
    pub robot_footprint: Footprint,
    pub use_radius: bool,
    pub circumscribed_cost: f64,
    pub inflation_radius: f64,
    pub circumscribed_radius: f64,

    // ESDF capsule
    pub use_esdf_footprint: bool,
    pub use_exact_esdf: bool,
    pub cost_check_points: Vec<f64>,
    pub robot_radius: f64,
    pub safe_distance: f64,
}

impl Default for SmacPlannerHybridConfig {
    fn default() -> Self {
        Self {
            downsample_costmap: false,
            downsampling_factor: 1,
            angle_quantization_bins: 72,
            tolerance: 0.25,
            allow_unknown: true,
            max_iterations: 1_000_000,
            max_on_approach_iterations: 1000,
            terminal_checking_interval: 5000,
            smooth_path: true,
            max_planning_time: 5.0,
            lookup_table_size: 20.0,
            debug_visualizations: false,
            motion_model_for_search: "DUBIN".to_string(),
            goal_heading_mode: "DEFAULT".to_string(),
            coarse_search_resolution: 1,
            search_info: SearchInfo::default(),
            smoother_params: SmootherParams::default(),
            robot_footprint: Vec::new(),
            use_radius: false,
            circumscribed_cost: -1.0,
            inflation_radius: 0.5,
            circumscribed_radius: 0.5,
            use_esdf_footprint: false,
            use_exact_esdf: true,
            cost_check_points: Vec::new(),
            robot_radius: 0.0,
            safe_distance: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// SmacPlannerHybrid
// ---------------------------------------------------------------------------

/// The top-level Hybrid A* planner.
pub struct SmacPlannerHybrid {
    a_star: Option<AStarAlgorithm>,
    collision_checker: Option<GridCollisionChecker>,
    smoother: Option<Smoother>,
    costmap: Option<*mut Costmap2D>,
    costmap_downsampler: CostmapDownsampler,
    esdf_holder: EsdfHolder,

    config: SmacPlannerHybridConfig,
    angle_bin_size: f32,
    angle_quantizations: u32,
    lookup_table_dim: f32,
    motion_model: MotionModel,
    goal_heading_mode: GoalHeadingMode,
}

unsafe impl Send for SmacPlannerHybrid {}

impl SmacPlannerHybrid {
    pub fn new() -> Self {
        Self {
            a_star: None,
            collision_checker: None,
            smoother: None,
            costmap: None,
            costmap_downsampler: CostmapDownsampler::new(),
            esdf_holder: EsdfHolder::default(),
            config: SmacPlannerHybridConfig::default(),
            angle_bin_size: 0.0,
            angle_quantizations: 0,
            lookup_table_dim: 0.0,
            motion_model: MotionModel::Unknown,
            goal_heading_mode: GoalHeadingMode::Unknown,
        }
    }

    pub fn configure(&mut self, costmap: &mut Costmap2D, config: SmacPlannerHybridConfig) {
        self.costmap = Some(costmap);
        self.config = config;

        self.angle_bin_size =
            2.0 * PI as f32 / self.config.angle_quantization_bins as f32;
        self.angle_quantizations = self.config.angle_quantization_bins;

        let mut max_iterations = self.config.max_iterations;
        let mut max_on_approach_iterations = self.config.max_on_approach_iterations;

        if max_on_approach_iterations <= 0 {
            max_on_approach_iterations = i32::MAX;
        }
        if max_iterations <= 0 {
            max_iterations = i32::MAX;
        }
        if self.config.coarse_search_resolution == 0 {
            self.config.coarse_search_resolution = 1;
        }

        self.motion_model = MotionModel::from_str(&self.config.motion_model_for_search);
        self.goal_heading_mode = GoalHeadingMode::from_str(&self.config.goal_heading_mode);

        let cm = unsafe { &*self.costmap.unwrap() };
        self.lookup_table_dim =
            (self.config.lookup_table_size / cm.resolution()).floor() as f32;

        let circumscribed_cost = if self.config.circumscribed_cost < 0.0 {
            find_circumscribed_cost(
                cm,
                self.config.circumscribed_radius,
                self.config.inflation_radius,
            )
        } else {
            self.config.circumscribed_cost
        };

        self.collision_checker = Some(GridCollisionChecker::new(cm, self.angle_quantizations));

        let want_esdf = self.config.use_esdf_footprint || !self.config.cost_check_points.is_empty();

        if want_esdf {
            self.esdf_holder.rebuild(cm, self.config.use_exact_esdf);
            if let Some(cc) = &mut self.collision_checker {
                cc.set_esdf_footprint(
                    self.config.cost_check_points.clone(),
                    self.config.robot_radius,
                    self.config.safe_distance,
                    &mut self.esdf_holder,
                );
            }
        } else {
            if let Some(cc) = &mut self.collision_checker {
                cc.set_footprint(
                    &self.config.robot_footprint,
                    self.config.use_radius,
                    circumscribed_cost,
                );
            }
        }

        let mut a_star = AStarAlgorithm::new(self.motion_model, self.config.search_info.clone());
        a_star.initialize(
            self.config.allow_unknown,
            max_iterations,
            max_on_approach_iterations,
            self.config.terminal_checking_interval,
            self.config.max_planning_time,
            self.lookup_table_dim,
            self.angle_quantizations,
        );

        if want_esdf {
            a_star.set_esdf_resources(
                &self.esdf_holder as *const EsdfHolder,
                self.config.cost_check_points.clone(),
                self.config.robot_radius,
                self.config.safe_distance,
            );
        }

        self.a_star = Some(a_star);

        if self.config.smooth_path {
            let mut sm = Smoother::new(&self.config.smoother_params);
            let min_turning_radius =
                self.config.search_info.minimum_turning_radius as f64 * cm.resolution();
            sm.initialize(min_turning_radius);
            self.smoother = Some(sm);
        }

        self.costmap_downsampler.on_configure(
            cm,
            self.config.downsampling_factor,
            false,
        );
    }

    pub fn set_footprint(
        &mut self,
        footprint: Footprint,
        use_radius: bool,
        circumscribed_cost: f64,
    ) {
        self.config.robot_footprint = footprint;
        self.config.use_radius = use_radius;

        let cm = unsafe { &*self.costmap.unwrap() };
        let cc = if circumscribed_cost < 0.0 {
            find_circumscribed_cost(cm, self.config.circumscribed_radius, self.config.inflation_radius)
        } else {
            circumscribed_cost
        };
        self.config.circumscribed_cost = cc;

        if let Some(checker) = &mut self.collision_checker {
            checker.set_footprint(&self.config.robot_footprint, use_radius, cc);
        }
    }

    pub fn create_plan(
        &mut self,
        start: &Pose,
        goal: &Pose,
        cancel_checker: impl Fn() -> bool,
    ) -> Path {
        let a = std::time::Instant::now();

        let cm = unsafe { &mut *self.costmap.unwrap() };

        // Downsample costmap if configured
        if self.config.downsample_costmap && self.config.downsampling_factor > 1 {
            let downsampled = self.costmap_downsampler.downsample(self.config.downsampling_factor);
            if let Some(cc) = &mut self.collision_checker {
                cc.set_costmap(downsampled);
            }
        }

        let active_costmap = if self.config.downsample_costmap && self.config.downsampling_factor > 1 {
            self.costmap_downsampler.downsample(self.config.downsampling_factor)
        } else {
            cm
        };

        // Rebuild ESDF if needed
        let want_esdf = self.config.use_esdf_footprint || !self.config.cost_check_points.is_empty();
        if want_esdf {
            self.esdf_holder.rebuild(active_costmap, self.config.use_exact_esdf);
        }

        if let Some(cc) = &mut self.collision_checker {
            cc.set_costmap(active_costmap);
        }

        if let Some(a_star) = &mut self.a_star {
            if let Some(cc) = &mut self.collision_checker {
                a_star.set_collision_checker(cc);
            }
        }

        // Convert start/goal to map coordinates
        let (mx_start, my_start) = active_costmap
            .world_to_map_continuous(start.x, start.y)
            .expect("Start coordinates outside bounds");

        let start_orientation_bin_int =
            wrap_bin_index((start.theta / self.angle_bin_size as f64).round() as i32, self.angle_quantizations);

        if let Some(a_star) = &mut self.a_star {
            a_star.set_start(mx_start, my_start, start_orientation_bin_int);
        }

        let (mx_goal, my_goal) = active_costmap
            .world_to_map_continuous(goal.x, goal.y)
            .expect("Goal coordinates outside bounds");

        let goal_orientation_bin_int =
            wrap_bin_index((goal.theta / self.angle_bin_size as f64).round() as i32, self.angle_quantizations);

        if let Some(a_star) = &mut self.a_star {
            a_star.set_goal(
                mx_goal,
                my_goal,
                goal_orientation_bin_int,
                self.goal_heading_mode,
                self.config.coarse_search_resolution,
            );
        }

        // Check trivial case
        if mx_start as i32 == mx_goal as i32
            && my_start as i32 == my_goal as i32
            && start_orientation_bin_int == goal_orientation_bin_int
        {
            return vec![*goal];
        }

        let mut path: Vec<Coordinates> = Vec::new();
        let mut num_iterations = 0i32;
        let mut expansions = if self.config.debug_visualizations {
            Some(Vec::new())
        } else {
            None
        };

        let tolerance_cells =
            self.config.tolerance / active_costmap.resolution() as f32;

        let success = if let Some(a_star) = &mut self.a_star {
            a_star.create_path(
                &mut path,
                &mut num_iterations,
                tolerance_cells,
                cancel_checker,
                expansions.as_mut(),
            )
        } else {
            false
        };

        if !success {
            if num_iterations == 1 {
                panic!("Start occupied");
            }
            if let Some(a_star) = &self.a_star {
                if num_iterations < a_star.max_iterations() {
                    panic!("No valid path found");
                } else {
                    panic!("Exceeded maximum iterations");
                }
            }
        }

        let mut plan: Path = Vec::with_capacity(path.len());
        for coords in path.iter().rev() {
            let mut pose = get_world_coords(coords.x, coords.y, active_costmap);
            pose.theta = coords.theta as f64;
            plan.push(pose);
        }

        let b = std::time::Instant::now();
        let time_span = (b - a).as_secs_f64();
        let time_remaining = self.config.max_planning_time - time_span;

        if let Some(sm) = &mut self.smoother {
            if num_iterations > 1 {
                sm.smooth(&mut plan, active_costmap, time_remaining);
            }
        }

        plan
    }

    pub fn a_star(&self) -> Option<&AStarAlgorithm> {
        self.a_star.as_ref()
    }
}
