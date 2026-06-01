//! Support point for Minkowski difference computation.

use crate::shapes::Shape;
use crate::vec3::Vec3;

/// A support point in the Minkowski difference of two shapes.
///
/// - `v` is the Minkowski difference support point (`v1 - v2`)
/// - `v1` is the support point on object 1
/// - `v2` is the support point on object 2
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SupportPoint {
    pub v: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
}

impl SupportPoint {
    /// Compute the Minkowski difference support point for two shapes.
    ///
    /// The support function returns the farthest point on the Minkowski
    /// difference `obj1 ⊖ obj2` in direction `dir`.
    #[inline]
    pub fn compute(obj1: &dyn Shape, obj2: &dyn Shape, dir: Vec3) -> Self {
        let v1 = obj1.support(dir);
        let v2 = obj2.support(-dir);
        let v = v1 - v2;
        Self { v, v1, v2 }
    }
}
