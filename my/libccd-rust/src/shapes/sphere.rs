//! Sphere shape implementation.

use crate::shapes::Shape;
use crate::vec3::Vec3;

/// Sphere shape.
#[derive(Debug, Clone)]
pub struct SphereShape {
    /// Radius of the sphere.
    pub radius: f32,
    /// Center position.
    pub pos: Vec3,
}

impl SphereShape {
    /// Create a new sphere centered at origin.
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            pos: Vec3::ZERO,
        }
    }

    /// Set the center position.
    pub fn with_pos(mut self, pos: Vec3) -> Self {
        self.pos = pos;
        self
    }
}

impl Shape for SphereShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        let dir_normalized = dir.normalize();
        dir_normalized * self.radius + self.pos
    }

    fn center(&self) -> Vec3 {
        self.pos
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
