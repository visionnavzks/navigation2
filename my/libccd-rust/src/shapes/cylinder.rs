//! Cylinder shape implementation.

use crate::quat::Quat;
use crate::shapes::Shape;
use crate::vec3::Vec3;

/// Cylinder shape aligned along the Z axis in local coordinates.
#[derive(Debug, Clone)]
pub struct CylinderShape {
    /// Radius of the cylinder.
    pub radius: f32,
    /// Half-height along the Z axis.
    pub height_half: f32,
    /// Center position.
    pub pos: Vec3,
    /// Optional rotation quaternion.
    pub rot: Option<Quat>,
    rot_inv: Option<Quat>,
}

impl CylinderShape {
    /// Create a new cylinder centered at origin, aligned along Z.
    pub fn new(radius: f32, height: f32) -> Self {
        Self {
            radius,
            height_half: height / 2.0,
            pos: Vec3::ZERO,
            rot: None,
            rot_inv: None,
        }
    }

    /// Set the center position.
    pub fn with_pos(mut self, pos: Vec3) -> Self {
        self.pos = pos;
        self
    }

    /// Set the rotation.
    pub fn with_rot(mut self, rot: Quat) -> Self {
        self.rot_inv = rot.invert();
        self.rot = Some(rot);
        self
    }
}

impl Shape for CylinderShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        let local_dir = match self.rot_inv {
            Some(ref inv) => inv.rotate_vec3(dir),
            None => dir,
        };

        // Cylinder support: radial component + Z cap
        let radial = Vec3::new(local_dir.x(), local_dir.y(), 0.0);
        let radial_len = radial.length();

        let sx = if radial_len > 1e-8 {
            self.radius * local_dir.x() / radial_len
        } else {
            0.0
        };
        let sy = if radial_len > 1e-8 {
            self.radius * local_dir.y() / radial_len
        } else {
            0.0
        };
        let sz = local_dir.z().signum() * self.height_half;

        let local_support = Vec3::new(sx, sy, sz);

        let world_support = match self.rot {
            Some(ref rot) => rot.rotate_vec3(local_support),
            None => local_support,
        };

        world_support + self.pos
    }

    fn center(&self) -> Vec3 {
        self.pos
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}