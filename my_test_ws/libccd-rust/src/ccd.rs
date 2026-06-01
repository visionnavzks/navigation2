//! Main CCD (Collision Detection) API.

use crate::epa;
use crate::gjk;
use crate::gjk::FirstDirFn;
use crate::mpr;
use crate::shapes::Shape;
use crate::shapes::SphereShape;
use crate::vec3::Vec3;

/// Penetration result returned by collision detection algorithms.
#[derive(Debug, Clone, Copy)]
pub struct Penetration {
    /// Depth of penetration.
    pub depth: f32,
    /// Normalized direction of penetration.
    pub dir: Vec3,
    /// Approximate contact position.
    pub pos: Vec3,
}

/// Configuration for collision detection algorithms.
#[derive(Debug, Clone)]
pub struct CcdConfig {
    /// Maximum number of iterations (default: u64::MAX).
    pub max_iterations: u64,
    /// EPA tolerance (default: 1e-4).
    pub epa_tolerance: f32,
    /// MPR tolerance (default: 1e-4).
    pub mpr_tolerance: f32,
    /// Distance tolerance used by GJK simplex contact checks.
    pub dist_tolerance: f32,
    /// Function used to choose the initial GJK search direction.
    pub first_dir: FirstDirFn,
}

impl Default for CcdConfig {
    fn default() -> Self {
        Self {
            max_iterations: u64::MAX,
            epa_tolerance: 1e-4,
            mpr_tolerance: 1e-4,
            dist_tolerance: 1e-6,
            first_dir: gjk::default_first_dir,
        }
    }
}

/// Builder for `Ccd`.
#[derive(Debug, Clone)]
pub struct CcdBuilder {
    config: CcdConfig,
}

impl CcdBuilder {
    pub fn max_iterations(mut self, v: u64) -> Self {
        self.config.max_iterations = v;
        self
    }
    pub fn epa_tolerance(mut self, v: f32) -> Self {
        self.config.epa_tolerance = v;
        self
    }
    pub fn mpr_tolerance(mut self, v: f32) -> Self {
        self.config.mpr_tolerance = v;
        self
    }
    pub fn dist_tolerance(mut self, v: f32) -> Self {
        self.config.dist_tolerance = v;
        self
    }
    pub fn first_dir(mut self, first_dir: FirstDirFn) -> Self {
        self.config.first_dir = first_dir;
        self
    }
    pub fn build(self) -> Ccd {
        Ccd {
            config: self.config,
        }
    }
}

/// Main collision detection struct.
///
/// Use the builder pattern to configure:
/// ```rust,ignore
/// let ccd = Ccd::builder().max_iterations(100).build();
/// ```
pub struct Ccd {
    config: CcdConfig,
}

impl Ccd {
    /// Create a CCD instance with default configuration.
    pub fn new() -> Self {
        Self {
            config: CcdConfig::default(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> CcdBuilder {
        CcdBuilder {
            config: CcdConfig::default(),
        }
    }

    /// Create a CCD instance from a config.
    pub fn from_config(config: CcdConfig) -> Self {
        Self { config }
    }

    /// Test if two shapes intersect using GJK algorithm.
    pub fn gjk_intersect(&self, obj1: &dyn Shape, obj2: &dyn Shape) -> bool {
        gjk::gjk_intersect(
            obj1,
            obj2,
            self.config.max_iterations,
            self.config.first_dir,
            self.config.dist_tolerance,
        )
    }

    /// Compute penetration using GJK + EPA algorithms.
    ///
    /// Returns `Some(Penetration)` if shapes intersect, `None` otherwise.
    pub fn gjk_penetration(&self, obj1: &dyn Shape, obj2: &dyn Shape) -> Option<Penetration> {
        if let Some(penetration) = exact_sphere_penetration(obj1, obj2) {
            return Some(penetration);
        }

        epa::gjk_epa(
            obj1,
            obj2,
            self.config.max_iterations,
            self.config.epa_tolerance,
            self.config.first_dir,
            self.config.dist_tolerance,
        )
        .map(|r| Penetration {
            depth: r.depth,
            dir: r.dir,
            pos: r.pos,
        })
    }

    /// Compute separation vector using GJK + EPA.
    ///
    /// Returns `Some(Vec3)` separation vector if shapes intersect, `None` otherwise.
    pub fn gjk_separate(&self, obj1: &dyn Shape, obj2: &dyn Shape) -> Option<Vec3> {
        epa::gjk_epa(
            obj1,
            obj2,
            self.config.max_iterations,
            self.config.epa_tolerance,
            self.config.first_dir,
            self.config.dist_tolerance,
        )
        .map(|r| r.dir * r.depth)
    }

    /// Test if two shapes intersect using MPR algorithm.
    pub fn mpr_intersect(&self, obj1: &dyn Shape, obj2: &dyn Shape) -> bool {
        mpr::mpr_intersect(obj1, obj2, self.config.max_iterations)
    }

    /// Compute penetration using MPR algorithm.
    ///
    /// Returns `Some(Penetration)` if shapes intersect, `None` otherwise.
    pub fn mpr_penetration(&self, obj1: &dyn Shape, obj2: &dyn Shape) -> Option<Penetration> {
        mpr::mpr_penetration(
            obj1,
            obj2,
            self.config.max_iterations,
            self.config.mpr_tolerance,
        )
        .map(|r| Penetration {
            depth: r.depth,
            dir: r.dir,
            pos: r.pos,
        })
    }
}

impl Default for Ccd {
    fn default() -> Self {
        Self::new()
    }
}

fn exact_sphere_penetration(obj1: &dyn Shape, obj2: &dyn Shape) -> Option<Penetration> {
    let sphere1 = obj1.as_any().downcast_ref::<SphereShape>()?;
    let sphere2 = obj2.as_any().downcast_ref::<SphereShape>()?;

    let delta = sphere1.pos - sphere2.pos;
    let distance = delta.length();
    let combined_radius = sphere1.radius + sphere2.radius;
    if distance >= combined_radius {
        return None;
    }

    let dir = if distance < 1e-6 {
        Vec3::X_AXIS
    } else {
        delta * (1.0 / distance)
    };
    let pos1 = sphere1.pos - dir * sphere1.radius;
    let pos2 = sphere2.pos + dir * sphere2.radius;

    Some(Penetration {
        depth: combined_radius - distance,
        dir,
        pos: (pos1 + pos2) * 0.5,
    })
}
