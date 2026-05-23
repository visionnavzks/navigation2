//! Convex hull shape implementation.

use crate::quat::Quat;
use crate::shapes::Shape;
use crate::vec3::Vec3;

/// Convex hull represented by a set of points in local coordinates.
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Vertices in the hull's local frame.
    pub points: Vec<Vec3>,
    /// Cached local centroid used as the default center.
    pub local_center: Vec3,
    /// World-space translation.
    pub pos: Vec3,
    /// Optional world rotation.
    pub rot: Option<Quat>,
    rot_inv: Option<Quat>,
}

impl ConvexHull {
    /// Create a new convex hull from local-space vertices.
    pub fn new(points: Vec<Vec3>) -> Self {
        assert!(!points.is_empty(), "ConvexHull requires at least one point");

        let mut sum = Vec3::ZERO;
        for point in &points {
            sum = sum + *point;
        }
        let local_center = sum * (1.0 / points.len() as f32);

        Self {
            points,
            local_center,
            pos: Vec3::ZERO,
            rot: None,
            rot_inv: None,
        }
    }

    /// Set the hull's world-space position.
    pub fn with_pos(mut self, pos: Vec3) -> Self {
        self.pos = pos;
        self
    }

    /// Set the hull's world-space rotation.
    pub fn with_rot(mut self, rot: Quat) -> Self {
        self.rot_inv = rot.invert();
        self.rot = Some(rot);
        self
    }
}

impl Shape for ConvexHull {
    fn support(&self, dir: Vec3) -> Vec3 {
        let local_dir = match self.rot_inv {
            Some(inv) => inv.rotate_vec3(dir),
            None => dir,
        };

        let mut best_point = self.points[0];
        let mut best_dot = best_point.dot(local_dir);
        for point in self.points.iter().skip(1) {
            let dot = point.dot(local_dir);
            if dot > best_dot {
                best_dot = dot;
                best_point = *point;
            }
        }

        let world_support = match self.rot {
            Some(rot) => rot.rotate_vec3(best_point),
            None => best_point,
        };

        world_support + self.pos
    }

    fn center(&self) -> Vec3 {
        let world_center = match self.rot {
            Some(rot) => rot.rotate_vec3(self.local_center),
            None => self.local_center,
        };
        world_center + self.pos
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
