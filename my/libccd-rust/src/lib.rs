//! libccd-rust: Idiomatic Rust port of [libccd](https://github.com/danfis/libccd)
//!
//! A collision detection library for convex shapes using GJK, EPA, and MPR algorithms.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use libccd_rust::{Ccd, shapes::BoxShape, Vec3};
//!
//! let ccd = Ccd::builder().build();
//! let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
//! let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(0.5, 0.0, 0.0));
//!
//! assert!(ccd.gjk_intersect(&box1, &box2));
//! ```

pub mod vec3;
pub mod quat;
pub mod support;
pub mod simplex;
pub mod shapes;
pub mod polytope;
pub mod gjk;
pub mod epa;
pub mod mpr;
pub mod ccd;

// Public re-exports
pub use vec3::Vec3;
pub use quat::Quat;
pub use support::SupportPoint;
pub use simplex::Simplex;
pub use shapes::Shape;
pub use ccd::{Ccd, CcdBuilder, CcdConfig, Penetration};
