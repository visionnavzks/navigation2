//! 3D vector type wrapping `glam::Vec3A` for SIMD-accelerated math.

use std::ops::{Add, Index, Mul, Neg, Sub};

/// Epsilon for floating-point comparisons.
pub const EPSILON: f32 = 1e-6;

/// 3D vector newtype over `glam::Vec3A`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3(pub glam::Vec3A);

impl Vec3 {
    // ---- constants ----
    pub const ZERO: Self = Self(glam::Vec3A::ZERO);
    pub const X_AXIS: Self = Self(glam::Vec3A::X);
    pub const Y_AXIS: Self = Self(glam::Vec3A::Y);
    pub const Z_AXIS: Self = Self(glam::Vec3A::Z);

    // ---- constructors ----
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(glam::Vec3A::new(x, y, z))
    }

    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }
    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }
    #[inline]
    pub fn z(self) -> f32 {
        self.0.z
    }

    // ---- basic ops ----
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.0.dot(other.0)
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self(self.0.cross(other.0))
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.0.length_squared()
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.0.length()
    }

    #[inline]
    pub fn distance_squared(self, other: Self) -> f32 {
        (self - other).length_squared()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < EPSILON {
            Self::ZERO
        } else {
            self * (1.0 / len)
        }
    }

    // ---- helpers ----
    /// Returns sign of a scalar: -1, 0, or 1.
    #[inline]
    pub fn sign(val: f32) -> i32 {
        if val.abs() < EPSILON {
            0
        } else if val < 0.0 {
            -1
        } else {
            1
        }
    }

    /// Returns true if the value is effectively zero (within EPSILON).
    #[inline]
    pub fn is_zero(val: f32) -> bool {
        val.abs() < EPSILON
    }

    /// Returns true if this vector is effectively zero.
    #[inline]
    pub fn is_zero_vec(self) -> bool {
        self.length_squared() < EPSILON * EPSILON
    }

    /// Approximate equality check (relative + absolute).
    #[inline]
    pub fn approx_eq(a: f32, b: f32) -> bool {
        let diff = (a - b).abs();
        if diff < EPSILON {
            return true;
        }
        diff < EPSILON * a.abs().max(b.abs())
    }

    /// Approximate vector equality.
    #[inline]
    pub fn vec_approx_eq(self, other: Self) -> bool {
        Self::approx_eq(self.x(), other.x())
            && Self::approx_eq(self.y(), other.y())
            && Self::approx_eq(self.z(), other.z())
    }

    // ---- distance functions (ported from libccd vec3.c) ----

    /// Returns squared distance from point `p` to segment `x0-b`,
    /// and optionally the witness (closest point on segment).
    pub fn point_segment_dist2(p: Self, x0: Self, b: Self) -> (f32, Option<Self>) {
        let d = b - x0;
        let a = x0 - p;
        let d_len2 = d.length_squared();

        if d_len2 < EPSILON * EPSILON {
            // Degenerate segment
            let dist = x0.distance_squared(p);
            return (dist, Some(x0));
        }

        let t = -a.dot(d) / d_len2;

        if t <= 0.0 {
            let dist = x0.distance_squared(p);
            (dist, Some(x0))
        } else if t >= 1.0 {
            let dist = b.distance_squared(p);
            (dist, Some(b))
        } else {
            let witness = x0 + d * t;
            let dist = witness.distance_squared(p);
            (dist, Some(witness))
        }
    }

    /// Returns squared distance from point `p` to triangle `x0-b-c`,
    /// and optionally the witness (closest point on triangle).
    pub fn point_tri_dist2(p: Self, x0: Self, b: Self, c: Self) -> (f32, Option<Self>) {
        let d1 = b - x0;
        let d2 = c - x0;
        let a = x0 - p;

        let _u = a.dot(a);
        let v = d1.dot(d1);
        let w = d2.dot(d2);
        let pp = a.dot(d1);
        let q = a.dot(d2);
        let r = d1.dot(d2);

        let det = w * v - r * r;

        let (s, t) = if det.abs() < EPSILON {
            (-1.0, -1.0) // degenerate triangle
        } else {
            let s = (q * r - w * pp) / det;
            let t = (-s * r - q) / w;
            (s, t)
        };

        let in_triangle = (s >= 0.0 || Self::is_zero(s))
            && (s <= 1.0 || Self::approx_eq(s, 1.0))
            && (t >= 0.0 || Self::is_zero(t))
            && (t <= 1.0 || Self::approx_eq(t, 1.0))
            && (s + t <= 1.0 || Self::approx_eq(s + t, 1.0));

        if in_triangle {
            let witness = x0 + d1 * s + d2 * t;
            let dist = witness.distance_squared(p);
            (dist, Some(witness))
        } else {
            // Check all three edges
            let (d01, w01) = Self::point_segment_dist2(p, x0, b);
            let (d02, w02) = Self::point_segment_dist2(p, x0, c);
            let (d12, w12) = Self::point_segment_dist2(p, b, c);

            let (dist, witness) = if d02 < d01 {
                if d12 < d02 { (d12, w12) } else { (d02, w02) }
            } else {
                if d12 < d01 { (d12, w12) } else { (d01, w01) }
            };
            (dist, witness)
        }
    }
}

// ---- operator overloads ----

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.0.x,
            1 => &self.0.y,
            2 => &self.0.z,
            _ => panic!("Vec3 index out of bounds: {index}"),
        }
    }
}

// Convenience: f32 * Vec3
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3(self * rhs.0)
    }
}
