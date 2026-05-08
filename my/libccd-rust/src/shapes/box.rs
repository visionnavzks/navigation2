//! Box shape implementation.

use crate::quat::Quat;
use crate::shapes::Shape;
use crate::vec3::Vec3;

/// Axis-aligned or oriented box shape.
///
/// The box is centered at `pos` with half-extents given by `radii`.
/// If `rot` is `None`, the box is axis-aligned.
#[derive(Debug, Clone)]
pub struct BoxShape {
    /// Half-extents along each axis.
    pub radii: Vec3,
    /// Center position.
    pub pos: Vec3,
    /// Optional rotation quaternion.
    pub rot: Option<Quat>,
    /// Inverse rotation quaternion (cached).
    rot_inv: Option<Quat>,
}

impl BoxShape {
    /// Create a new axis-aligned box centered at origin with given half-extents.
    pub fn new(radii: Vec3) -> Self {
        Self {
            radii,
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

impl Shape for BoxShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        // Transform direction to local frame
        let local_dir = match self.rot_inv {
            Some(ref inv) => inv.rotate_vec3(dir),
            None => dir,
        };

        // Box support: sign * half-extent for each axis
        let sx = local_dir.x().signum() * self.radii.x();
        let sy = local_dir.y().signum() * self.radii.y();
        let sz = local_dir.z().signum() * self.radii.z();
        let local_support = Vec3::new(sx, sy, sz);

        // Transform back to world frame
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