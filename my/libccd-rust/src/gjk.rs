//! Gilbert-Johnson-Keerthi (GJK) algorithm for collision detection.

use crate::shapes::Shape;
use crate::simplex::Simplex;
use crate::support::SupportPoint;
use crate::vec3::Vec3;

pub type FirstDirFn = fn(&dyn Shape, &dyn Shape) -> Vec3;

/// Result of GJK intersection test.
#[derive(Debug, Clone)]
pub enum GjkResult {
    /// Objects intersect. The simplex contains the witness points.
    Intersection(Simplex),
    /// Objects do not intersect.
    NoIntersection,
}

/// Triple cross product: (a × b) × c
#[inline]
fn triple_cross(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a.cross(b).cross(c)
}

/// Run the GJK algorithm to determine if two convex shapes intersect.
///
/// Returns `GjkResult::Intersection(simplex)` if they intersect (with the terminal simplex),
/// or `GjkResult::NoIntersection` otherwise.
pub fn gjk(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    max_iterations: u64,
    first_dir: FirstDirFn,
    dist_tolerance: f32,
) -> GjkResult {
    let mut simplex = Simplex::new();

    let dir = first_dir(obj1, obj2);

    // Get first support point
    let last = SupportPoint::compute(obj1, obj2, dir);
    simplex.push(last);

    // Direction toward origin from last support point
    let mut dir = -last.v;

    for _ in 0..max_iterations {
        let last = SupportPoint::compute(obj1, obj2, dir);

        // If the farthest point in Minkowski difference is behind the origin,
        // the objects don't intersect.
        if last.v.dot(dir) < 0.0 {
            return GjkResult::NoIntersection;
        }

        simplex.push(last);

        match do_simplex(&mut simplex, &mut dir, dist_tolerance) {
            SimplexResult::Intersection => return GjkResult::Intersection(simplex),
            SimplexResult::NoIntersection => return GjkResult::NoIntersection,
            SimplexResult::Continue => {}
        }

        if dir.is_zero_vec() {
            return GjkResult::NoIntersection;
        }
    }

    GjkResult::NoIntersection
}

/// Simple test: do two shapes intersect using GJK?
pub fn gjk_intersect(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    max_iterations: u64,
    first_dir: FirstDirFn,
    dist_tolerance: f32,
) -> bool {
    matches!(gjk(obj1, obj2, max_iterations, first_dir, dist_tolerance), GjkResult::Intersection(_))
}

/// Default first direction: from center of obj1 toward center of obj2.
pub fn default_first_dir(obj1: &dyn Shape, obj2: &dyn Shape) -> Vec3 {
    let c1 = obj1.center();
    let c2 = obj2.center();
    let dir = c2 - c1;
    if dir.is_zero_vec() {
        Vec3::X_AXIS
    } else {
        dir.normalize()
    }
}

/// Internal result from do_simplex processing.
enum SimplexResult {
    Intersection,
    NoIntersection,
    Continue,
}

/// Dispatch to the appropriate do_simplex function based on simplex size.
fn do_simplex(simplex: &mut Simplex, dir: &mut Vec3, dist_tolerance: f32) -> SimplexResult {
    match simplex.len() {
        2 => do_simplex2(simplex, dir),
        3 => do_simplex3(simplex, dir, dist_tolerance),
        4 => do_simplex4(simplex, dir, dist_tolerance),
        _ => SimplexResult::Continue,
    }
}

/// Process a 2-point simplex (line segment).
fn do_simplex2(simplex: &mut Simplex, dir: &mut Vec3) -> SimplexResult {
    let a = *simplex.get(1).unwrap(); // last added
    let b = *simplex.get(0).unwrap(); // the other point

    let ab = b.v - a.v;
    let ao = -a.v;

    // Check if origin lies on AB segment
    let cross = ab.cross(ao);
    if cross.is_zero_vec() && ab.dot(ao) > 0.0 {
        return SimplexResult::Intersection;
    }

    if ao.dot(ab) <= 0.0 {
        // Origin is in region of A alone
        simplex.set(0, a);
        simplex.set_count(1);
        *dir = ao;
    } else {
        // Origin is in region of AB segment
        *dir = triple_cross(ab, ao, ab);
    }

    SimplexResult::Continue
}

/// Process a 3-point simplex (triangle).
fn do_simplex3(simplex: &mut Simplex, dir: &mut Vec3, dist_tolerance: f32) -> SimplexResult {
    let a = *simplex.get(2).unwrap(); // last added
    let b = *simplex.get(1).unwrap();
    let c = *simplex.get(0).unwrap();

    // Check touching contact: origin lies on triangle plane
    let (tri_dist, _) = Vec3::point_tri_dist2(Vec3::ZERO, a.v, b.v, c.v);
    if tri_dist <= dist_tolerance * dist_tolerance {
        return SimplexResult::Intersection;
    }

    // Check if triangle has area
    if a.v.vec_approx_eq(b.v) || a.v.vec_approx_eq(c.v) {
        return SimplexResult::NoIntersection;
    }

    let ao = -a.v;
    let ab = b.v - a.v;
    let ac = c.v - a.v;
    let abc = ab.cross(ac);

    let tmp = abc.cross(ac);
    let dot = tmp.dot(ao);

    if dot >= 0.0 {
        let dot_ac = ac.dot(ao);
        if dot_ac >= 0.0 {
            // Origin is in region of AC
            simplex.set(1, a);
            simplex.set_count(2);
            *dir = triple_cross(ac, ao, ac);
        } else {
            handle_ab_region(simplex, a, b, ao, ab, dir);
        }
    } else {
        let tmp = ab.cross(abc);
        let dot = tmp.dot(ao);
        if dot >= 0.0 {
            handle_ab_region(simplex, a, b, ao, ab, dir);
        } else {
            let dot_abc = abc.dot(ao);
            if dot_abc >= 0.0 {
                *dir = abc;
            } else {
                // Swap B and C, flip direction
                let c_tmp = c;
                simplex.set(0, b);
                simplex.set(1, c_tmp);
                *dir = -abc;
            }
        }
    }

    SimplexResult::Continue
}

/// Handle the case where origin is in the AB region of the triangle.
fn handle_ab_region(
    simplex: &mut Simplex,
    a: SupportPoint,
    b: SupportPoint,
    ao: Vec3,
    ab: Vec3,
    dir: &mut Vec3,
) {
    let dot_ab = ab.dot(ao);
    if dot_ab >= 0.0 {
        simplex.set(0, b);
        simplex.set(1, a);
        simplex.set_count(2);
        *dir = triple_cross(ab, ao, ab);
    } else {
        simplex.set(0, a);
        simplex.set_count(1);
        *dir = ao;
    }
}

/// Process a 4-point simplex (tetrahedron).
fn do_simplex4(simplex: &mut Simplex, dir: &mut Vec3, dist_tolerance: f32) -> SimplexResult {
    let a = *simplex.get(3).unwrap(); // last added
    let b = *simplex.get(2).unwrap();
    let c = *simplex.get(1).unwrap();
    let d = *simplex.get(0).unwrap();

    // Check if tetrahedron has volume
    let (vol_dist, _) = Vec3::point_tri_dist2(a.v, b.v, c.v, d.v);
    if vol_dist <= dist_tolerance * dist_tolerance {
        return SimplexResult::NoIntersection;
    }

    // Check if origin lies on any face → intersection
    let (dist_abc, _) = Vec3::point_tri_dist2(Vec3::ZERO, a.v, b.v, c.v);
    if dist_abc <= dist_tolerance * dist_tolerance {
        return SimplexResult::Intersection;
    }
    let (dist_acd, _) = Vec3::point_tri_dist2(Vec3::ZERO, a.v, c.v, d.v);
    if dist_acd <= dist_tolerance * dist_tolerance {
        return SimplexResult::Intersection;
    }
    let (dist_abd, _) = Vec3::point_tri_dist2(Vec3::ZERO, a.v, b.v, d.v);
    if dist_abd <= dist_tolerance * dist_tolerance {
        return SimplexResult::Intersection;
    }
    let (dist_bcd, _) = Vec3::point_tri_dist2(Vec3::ZERO, b.v, c.v, d.v);
    if dist_bcd <= dist_tolerance * dist_tolerance {
        return SimplexResult::Intersection;
    }

    let ao = -a.v;
    let ab = b.v - a.v;
    let ac = c.v - a.v;
    let ad = d.v - a.v;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    // Which side of each face is the opposing vertex on?
    let b_on_acd = Vec3::sign(acd.dot(ab));
    let c_on_adb = Vec3::sign(adb.dot(ac));
    let d_on_abc = Vec3::sign(abc.dot(ad));

    // Is origin on the same side as the opposing vertex?
    let ab_o = Vec3::sign(acd.dot(ao)) == b_on_acd;
    let ac_o = Vec3::sign(adb.dot(ao)) == c_on_adb;
    let ad_o = Vec3::sign(abc.dot(ao)) == d_on_abc;

    if ab_o && ac_o && ad_o {
        // Origin is inside the tetrahedron!
        return SimplexResult::Intersection;
    }

    // Reduce to triangle by removing the farthest vertex
    if !ab_o {
        // B is farthest — remove it
        // D and C stay, replace position 2 with A
        simplex.set(2, a);
        simplex.set_count(3);
    } else if !ac_o {
        // C is farthest
        simplex.set(1, d);
        simplex.set(0, b);
        simplex.set(2, a);
        simplex.set_count(3);
    } else {
        // D is farthest
        simplex.set(0, c);
        simplex.set(1, b);
        simplex.set(2, a);
        simplex.set_count(3);
    }

    do_simplex3(simplex, dir, dist_tolerance)
}
