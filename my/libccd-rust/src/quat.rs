//! Quaternion type wrapping `glam::Quat`.

use crate::vec3::Vec3;

/// Quaternion newtype over `glam::Quat`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat(pub glam::Quat);

impl Quat {
    /// Identity quaternion (no rotation).
    pub const IDENTITY: Self = Self(glam::Quat::IDENTITY);

    /// Create from axis-angle.
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        Self(glam::Quat::from_axis_angle(
            axis.0.normalize().into(),
            angle,
        ))
    }

    /// Invert the quaternion. Returns `None` if nearly zero length.
    #[inline]
    pub fn invert(self) -> Option<Self> {
        let len2 = self.0.length_squared();
        if len2 < crate::vec3::EPSILON {
            return None;
        }
        Some(Self(self.0.conjugate() * (1.0 / len2)))
    }

    /// Rotate a vector by this quaternion.
    #[inline]
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        Vec3(self.0 * v.0)
    }

    /// Multiply two quaternions: `self * other`.
    #[inline]
    pub fn multiply(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Normalize the quaternion. Returns `None` if nearly zero length.
    #[inline]
    pub fn normalize(self) -> Option<Self> {
        let len = self.0.length();
        if len < crate::vec3::EPSILON {
            return None;
        }
        Some(Self(self.0 / len))
    }
}
