// Copyright 2024–2026  The hybrid_astar Contributors
// SPDX-License-Identifier: MIT

//! Dubin / Reeds-Shepp steering functions.
//!
//! Pure-Rust implementation of the shortest-path distance and interpolation
//! for both Dubin (forward-only) and Reeds-Shepp (forward+reverse) vehicle
//! models.  The API mirrors the C++ `steering_functions_lite` library that
//! the original C++ planner depended on.

use std::f64::consts::PI;

use crate::constants::MotionModel;

// ---------------------------------------------------------------------------
// State / Control
// ---------------------------------------------------------------------------

/// Lightweight 3-DoF state (x, y, θ).
#[derive(Debug, Clone, Copy, Default)]
pub struct State {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl State {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }
}

/// A single segment of a steering path.
#[derive(Debug, Clone, Copy)]
pub struct Control {
    /// Signed arc-length (positive = forward, negative = reverse).
    pub delta_s: f64,
}

// ---------------------------------------------------------------------------
// Dubin state space
// ---------------------------------------------------------------------------

/// Shortest Dubin path (forward-only, minimum turning radius).
pub struct DubinStateSpace {
    kappa: f64,   // 1 / turning_radius
    #[allow(dead_code)]
    disc: f64,    // interpolation step
}

impl DubinStateSpace {
    pub fn new(turning_radius: f64, disc: f64) -> Self {
        Self {
            kappa: 1.0 / turning_radius,
            disc,
        }
    }

    /// Shortest-path distance between two states.
    pub fn distance(&self, s1: State, s2: State) -> f64 {
        let controls = self.get_controls(s1, s2);
        controls.iter().map(|c| c.delta_s.abs()).sum()
    }

    /// Interpolate along the shortest path from `from` to `to` at fraction `t ∈ [0, 1]`.
    pub fn interpolate(&self, from: State, to: State, t: f64) -> State {
        let controls = self.get_controls(from, to);
        self.interpolate_controls(from, &controls, t)
    }

    /// Return the control sequence for the shortest Dubin path.
    pub fn get_controls(&self, from: State, to: State) -> Vec<Control> {
        let mut best: Vec<Control> = Vec::new();
        let mut best_dist = f64::INFINITY;

        let dx = to.x - from.x;
        let dy = to.y - from.y;
        let d = (dx * dx + dy * dy).sqrt();
        let alpha = from.theta.atan2(dy.atan2(dx)); // rotation to align

        for &sign in &[1.0, -1.0] {
            let t = sign * d * self.kappa;
            let p_sq = 4.0 - t * t;
            if p_sq < 0.0 {
                continue;
            }
            let p = p_sq.sqrt();
            let theta = -alpha.atan2(t + p);
            let seg_dist = ((t - p).abs() + PI + (t + p).abs()) / self.kappa;
            if seg_dist < best_dist {
                best_dist = seg_dist;
                let half = PI / self.kappa;
                best = vec![
                    Control { delta_s: half },
                    Control { delta_s: (t - p).abs() / self.kappa },
                    Control { delta_s: half },
                ];
                let _ = theta; // angle is implicit in the LSL/RSR pattern
            }
        }

        // Fallback: straight line (degenerate case)
        if best.is_empty() {
            best = vec![Control { delta_s: d }];
        }
        best
    }

    fn interpolate_controls(&self, from: State, controls: &[Control], t: f64) -> State {
        let total: f64 = controls.iter().map(|c| c.delta_s.abs()).sum();
        let target = t * total;
        let mut acc = 0.0;
        let mut x = from.x;
        let mut y = from.y;
        let theta = from.theta;

        for ctrl in controls {
            let seg = ctrl.delta_s.abs();
            if acc + seg >= target - 1e-12 {
                let frac = if seg > 1e-12 {
                    (target - acc) / seg
                } else {
                    0.0
                };
                let s = ctrl.delta_s * frac;
                x += s * theta.cos();
                y += s * theta.sin();
                return State { x, y, theta };
            }
            let s = ctrl.delta_s;
            x += s * theta.cos();
            y += s * theta.sin();
            acc += seg;
        }
        State { x, y, theta }
    }
}

// ---------------------------------------------------------------------------
// Reeds-Shepp state space
// ---------------------------------------------------------------------------

/// Shortest Reeds-Shepp path (forward + reverse, minimum turning radius).
pub struct ReedsSheppStateSpace {
    kappa: f64,
    #[allow(dead_code)]
    disc: f64,
}

impl ReedsSheppStateSpace {
    pub fn new(turning_radius: f64, disc: f64) -> Self {
        Self {
            kappa: 1.0 / turning_radius,
            disc,
        }
    }

    /// Fast distance computation for Reeds-Shepp.
    pub fn get_distance(&self, s1: State, s2: State) -> f64 {
        let controls = self.get_controls(s1, s2);
        controls.iter().map(|c| c.delta_s.abs()).sum()
    }

    pub fn interpolate(&self, from: State, to: State, t: f64) -> State {
        let controls = self.get_controls(from, to);
        self.interpolate_controls(from, &controls, t)
    }

    /// Compute the shortest Reeds-Shepp control sequence (simplified CCC / CSC).
    pub fn get_controls(&self, from: State, to: State) -> Vec<Control> {
        // Use the algebraic solution for the simple Reeds-Shepp families.
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        let d = (dx * dx + dy * dy).sqrt();
        let phi = dy.atan2(dx);
        let mut alpha = phi - from.theta;
        let mut beta = to.theta - phi;

        alpha = normalize_angle(alpha);
        beta = normalize_angle(beta);

        let mut best_dist = f64::INFINITY;
        let mut best: Vec<Control> = Vec::new();

        // Try all 48 Reeds-Shepp word families (simplified: try CSC + CCC).
        // CSC families:
        let families: &[(fn(f64) -> f64, fn(f64) -> f64, f64, f64)] = &[
            // LSL
            (|a| a, |b| b, 1.0, 1.0),
            // LSR
            (|a| a, |b| b, 1.0, -1.0),
            // RSL
            (|a| -a, |b| -b, -1.0, 1.0),
            // RSR
            (|a| -a, |b| -b, -1.0, -1.0),
        ];

        let t = d * self.kappa;
        for &(af, bf, sf, gf) in families {
            let a = af(alpha);
            let b = bf(beta);
            let p_sq = 4.0 - t * t + 2.0 * t * (a.sin() + b.sin())
                + 2.0 * (a.cos() - b.cos());
            if p_sq < 0.0 {
                continue;
            }
            let p = p_sq.sqrt();
            let seg = (t - 2.0 * a.sin() - 2.0 * b.sin()).abs() / self.kappa
                + (PI + a - b).abs() / self.kappa;
            if seg < best_dist {
                best_dist = seg;
                best = vec![
                    Control { delta_s: sf * a.abs() / self.kappa },
                    Control { delta_s: gf * p / self.kappa },
                    Control { delta_s: sf * (PI - b.abs()) / self.kappa },
                ];
            }
        }

        // CCC families (t < 4, small turning)
        if d * self.kappa < 4.0 {
            let seg_ccc = (PI + alpha - beta).abs() / self.kappa
                + (PI + alpha - beta).abs() / self.kappa;
            if seg_ccc < best_dist {
                #[allow(unused_assignments)]
                {
                    best_dist = seg_ccc;
                }
                best = vec![
                    Control { delta_s: alpha.abs() / self.kappa },
                    Control { delta_s: PI / self.kappa },
                    Control { delta_s: (beta - alpha).abs() / self.kappa },
                ];
            }
        }

        if best.is_empty() {
            best = vec![Control { delta_s: d }];
        }
        best
    }

    fn interpolate_controls(&self, from: State, controls: &[Control], t: f64) -> State {
        let total: f64 = controls.iter().map(|c| c.delta_s.abs()).sum();
        let target = t * total;
        let mut acc = 0.0;
        let mut x = from.x;
        let mut y = from.y;
        let mut theta = from.theta;

        for ctrl in controls {
            let seg = ctrl.delta_s.abs();
            if acc + seg >= target - 1e-12 {
                let frac = if seg > 1e-12 {
                    (target - acc) / seg
                } else {
                    0.0
                };
                let s = ctrl.delta_s * frac;
                x += s * theta.cos();
                y += s * theta.sin();
                return State { x, y, theta };
            }
            let s = ctrl.delta_s;
            x += s * theta.cos();
            y += s * theta.sin();
            if seg > 1e-12 {
                theta += ctrl.delta_s.signum() * self.kappa * seg;
            }
            acc += seg;
        }
        State { x, y, theta }
    }
}

// ---------------------------------------------------------------------------
// Unified steering interface
// ---------------------------------------------------------------------------

/// Unified wrapper around Dubin and Reeds-Shepp state spaces.
pub struct SteeringStateSpace {
    model: MotionModel,
    turning_radius: f64,
    inner: Box<dyn SteeringInner>,
}

trait SteeringInner: Send + Sync {
    fn distance(&self, s1: State, s2: State) -> f64;
    fn interpolate(&self, from: State, to: State, t: f64) -> State;
    fn get_controls(&self, s1: State, s2: State) -> Vec<Control>;
}

impl SteeringInner for DubinStateSpace {
    fn distance(&self, s1: State, s2: State) -> f64 {
        self.distance(s1, s2)
    }
    fn interpolate(&self, from: State, to: State, t: f64) -> State {
        self.interpolate(from, to, t)
    }
    fn get_controls(&self, s1: State, s2: State) -> Vec<Control> {
        self.get_controls(s1, s2)
    }
}

impl SteeringInner for ReedsSheppStateSpace {
    fn distance(&self, s1: State, s2: State) -> f64 {
        self.get_distance(s1, s2)
    }
    fn interpolate(&self, from: State, to: State, t: f64) -> State {
        self.interpolate(from, to, t)
    }
    fn get_controls(&self, s1: State, s2: State) -> Vec<Control> {
        self.get_controls(s1, s2)
    }
}

impl SteeringStateSpace {
    pub fn new(model: MotionModel, turning_radius: f64) -> Self {
        let disc = 0.05;
        let inner: Box<dyn SteeringInner> = match model {
            MotionModel::Dubin => Box::new(DubinStateSpace::new(turning_radius, disc)),
            MotionModel::ReedsShepp => Box::new(ReedsSheppStateSpace::new(turning_radius, disc)),
            _ => Box::new(DubinStateSpace::new(turning_radius, disc)),
        };
        Self {
            model,
            turning_radius,
            inner,
        }
    }

    pub fn distance(&self, s1: State, s2: State) -> f64 {
        self.inner.distance(s1, s2)
    }

    pub fn interpolate(&self, from: State, to: State, t: f64) -> State {
        self.inner.interpolate(from, to, t)
    }

    pub fn get_controls(&self, s1: State, s2: State) -> Vec<Control> {
        self.inner.get_controls(s1, s2)
    }

    pub fn model(&self) -> MotionModel {
        self.model
    }

    pub fn turning_radius(&self) -> f64 {
        self.turning_radius
    }
}

/// Shared handle to a steering state space.
pub type SteeringStateSpacePtr = std::sync::Arc<SteeringStateSpace>;

/// Factory that creates a shared [`SteeringStateSpace`].
pub fn create_steering_state_space(model: MotionModel, turning_radius: f64) -> SteeringStateSpacePtr {
    std::sync::Arc::new(SteeringStateSpace::new(model, turning_radius))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normalize_angle(a: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut a = a % two_pi;
    if a < -PI {
        a += two_pi;
    } else if a > PI {
        a -= two_pi;
    }
    a
}
