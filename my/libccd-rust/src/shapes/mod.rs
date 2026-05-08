//! Shape trait and built-in shape implementations.

mod r#box;
mod convex_hull;
mod sphere;
mod cylinder;

use std::any::Any;

use crate::vec3::Vec3;

pub use r#box::BoxShape;
pub use convex_hull::ConvexHull;
pub use sphere::SphereShape;
pub use cylinder::CylinderShape;

/// Trait for convex shapes that can participate in collision detection.
///
/// Every shape must provide:
/// - `support(dir)`: the farthest point on the shape in the given direction
/// - `center()`: the geometric center of the shape
pub trait Shape: Any {
    /// Returns the farthest point on the shape in the given direction.
    fn support(&self, dir: Vec3) -> Vec3;

    /// Returns the geometric center of the shape.
    fn center(&self) -> Vec3;

    /// Downcast hook for shape-specific exact paths.
    fn as_any(&self) -> &dyn Any;
}