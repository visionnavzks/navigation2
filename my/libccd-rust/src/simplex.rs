//! Simplex for GJK algorithm — up to 4 support points.

use crate::support::SupportPoint;

/// GJK simplex containing up to 4 support points (0D → 1D → 2D → 3D).
#[derive(Clone, Debug, PartialEq)]
pub struct Simplex {
    points: [SupportPoint; 4],
    count: usize,
}

impl Simplex {
    /// Create an empty simplex.
    #[inline]
    pub fn new() -> Self {
        Self {
            points: [
                SupportPoint { v: crate::vec3::Vec3::ZERO, v1: crate::vec3::Vec3::ZERO, v2: crate::vec3::Vec3::ZERO },
                SupportPoint { v: crate::vec3::Vec3::ZERO, v1: crate::vec3::Vec3::ZERO, v2: crate::vec3::Vec3::ZERO },
                SupportPoint { v: crate::vec3::Vec3::ZERO, v1: crate::vec3::Vec3::ZERO, v2: crate::vec3::Vec3::ZERO },
                SupportPoint { v: crate::vec3::Vec3::ZERO, v1: crate::vec3::Vec3::ZERO, v2: crate::vec3::Vec3::ZERO },
            ],
            count: 0,
        }
    }

    /// Number of points in the simplex.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Is the simplex empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Push a support point onto the simplex.
    /// Panics if the simplex already has 4 points.
    #[inline]
    pub fn push(&mut self, point: SupportPoint) {
        assert!(self.count < 4, "Simplex overflow: cannot push more than 4 points");
        self.points[self.count] = point;
        self.count += 1;
    }

    /// Get a point by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&SupportPoint> {
        if index < self.count {
            Some(&self.points[index])
        } else {
            None
        }
    }

    /// Get a mutable reference to a point by index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut SupportPoint> {
        if index < self.count {
            Some(&mut self.points[index])
        } else {
            None
        }
    }

    /// Set a point at a given index (within current count).
    #[inline]
    pub fn set(&mut self, index: usize, point: SupportPoint) {
        debug_assert!(index < self.count, "Simplex set index out of bounds");
        self.points[index] = point;
    }

    /// Get the last (most recently added) point.
    #[inline]
    pub fn last(&self) -> Option<&SupportPoint> {
        if self.count > 0 {
            Some(&self.points[self.count - 1])
        } else {
            None
        }
    }

    /// Swap two points by index.
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.count && b < self.count, "Simplex swap index out of bounds");
        self.points.swap(a, b);
    }

    /// Set the size of the simplex, effectively truncating or defining the count.
    #[inline]
    pub fn set_count(&mut self, count: usize) {
        debug_assert!(count <= 4, "Simplex count cannot exceed 4");
        self.count = count;
    }

    /// Iterate over the points in the simplex.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &SupportPoint> {
        self.points[..self.count].iter()
    }

    /// Direct indexed access (no bounds check — used in hot loops).
    /// # Safety
    /// Caller must ensure index < count.
    #[inline]
    /// # Safety
    /// Caller must ensure index < count.
    pub unsafe fn get_unchecked(&self, index: usize) -> &SupportPoint {
        debug_assert!(index < self.count);
        // SAFETY: caller guarantees index < count
        unsafe { self.points.get_unchecked(index) }
    }
}

impl Default for Simplex {
    fn default() -> Self {
        Self::new()
    }
}
