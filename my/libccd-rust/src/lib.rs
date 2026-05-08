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

pub mod ccd;
pub mod epa;
pub mod gjk;
pub mod mpr;
pub mod polytope;
pub mod quat;
pub mod shapes;
pub mod simplex;
pub mod support;
pub mod vec3;

// Public re-exports
pub use ccd::{Ccd, CcdBuilder, CcdConfig, Penetration};
pub use quat::Quat;
pub use shapes::Shape;
pub use simplex::Simplex;
pub use support::SupportPoint;
pub use vec3::Vec3;
