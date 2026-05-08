//! Minkowski Portal Refinement (MPR) algorithm for collision detection.

use crate::shapes::Shape;
use crate::simplex::Simplex;
use crate::support::SupportPoint;
use crate::vec3::{Vec3, EPSILON};

const DEFAULT_MPR_MAX_ITER: u64 = 1000;
const DEFAULT_MPR_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone, Copy)]
pub struct MprResult {
    pub depth: f32,
    pub dir: Vec3,
    pub pos: Vec3,
}

pub fn mpr_intersect(obj1: &dyn Shape, obj2: &dyn Shape, max_iterations: u64) -> bool {
    let mut portal = Simplex::new();
    let res = discover_portal(obj1, obj2, &mut portal);
    if res < 0 {
        return false;
    }
    if res > 0 {
        return true;
    }

    refine_portal(
        obj1,
        obj2,
        &mut portal,
        cap_iterations(max_iterations),
        DEFAULT_MPR_TOLERANCE,
    )
}

pub fn mpr_penetration(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    max_iterations: u64,
    mpr_tolerance: f32,
) -> Option<MprResult> {
    let mut portal = Simplex::new();
    let res = discover_portal(obj1, obj2, &mut portal);

    if res < 0 {
        return None;
    }
    if res == 1 {
        let v1 = portal.get(1)?;
        return Some(MprResult {
            depth: 0.0,
            dir: Vec3::ZERO,
            pos: (v1.v1 + v1.v2) * 0.5,
        });
    }
    if res == 2 {
        let v1 = portal.get(1)?;
        let depth = v1.v.length();
        return Some(MprResult {
            depth,
            dir: v1.v.normalize(),
            pos: (v1.v1 + v1.v2) * 0.5,
        });
    }

    let max_iter = cap_iterations(max_iterations);
    if !refine_portal(obj1, obj2, &mut portal, max_iter, mpr_tolerance) {
        return None;
    }

    find_penetration(obj1, obj2, &mut portal, max_iter, mpr_tolerance)
}

fn cap_iterations(max_iterations: u64) -> u64 {
    if max_iterations == u64::MAX {
        DEFAULT_MPR_MAX_ITER
    } else {
        max_iterations
    }
}

fn find_origin(obj1: &dyn Shape, obj2: &dyn Shape) -> SupportPoint {
    let v1 = obj1.center();
    let v2 = obj2.center();
    SupportPoint { v: v1 - v2, v1, v2 }
}

fn discover_portal(obj1: &dyn Shape, obj2: &dyn Shape, portal: &mut Simplex) -> i32 {
    let mut v0 = find_origin(obj1, obj2);
    if v0.v.is_zero_vec() {
        v0.v = v0.v + Vec3::new(EPSILON * 10.0, 0.0, 0.0);
    }
    portal.push(v0);

    let dir = (-v0.v).normalize();
    let v1 = SupportPoint::compute(obj1, obj2, dir);
    let dot = v1.v.dot(dir);
    if Vec3::is_zero(dot) || dot < 0.0 {
        return -1;
    }
    portal.push(v1);

    let dir = v0.v.cross(v1.v);
    if dir.is_zero_vec() {
        return if v1.v.is_zero_vec() { 1 } else { 2 };
    }

    let dir = dir.normalize();
    let v2 = SupportPoint::compute(obj1, obj2, dir);
    let dot = v2.v.dot(dir);
    if Vec3::is_zero(dot) || dot < 0.0 {
        return -1;
    }
    portal.push(v2);

    let va = portal.get(1).unwrap().v - v0.v;
    let vb = portal.get(2).unwrap().v - v0.v;
    let mut dir = va.cross(vb).normalize();
    if dir.dot(v0.v) > 0.0 {
        portal.swap(1, 2);
        dir = -dir;
    }

    while portal.len() < 4 {
        let v3 = SupportPoint::compute(obj1, obj2, dir);
        let dot = v3.v.dot(dir);
        if Vec3::is_zero(dot) || dot < 0.0 {
            return -1;
        }

        let va = portal.get(1).unwrap().v.cross(v3.v);
        let dot13 = va.dot(v0.v);
        if dot13 < 0.0 && !Vec3::is_zero(dot13) {
            portal.set(2, v3);
            let va = portal.get(1).unwrap().v - v0.v;
            let vb = portal.get(2).unwrap().v - v0.v;
            dir = va.cross(vb).normalize();
            continue;
        }

        let va = v3.v.cross(portal.get(2).unwrap().v);
        let dot32 = va.dot(v0.v);
        if dot32 < 0.0 && !Vec3::is_zero(dot32) {
            portal.set(1, v3);
            let va = portal.get(1).unwrap().v - v0.v;
            let vb = portal.get(2).unwrap().v - v0.v;
            dir = va.cross(vb).normalize();
            continue;
        }

        portal.push(v3);
    }

    0
}

fn refine_portal(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    portal: &mut Simplex,
    max_iterations: u64,
    mpr_tolerance: f32,
) -> bool {
    for _ in 0..max_iterations {
        let dir = portal_dir(portal);
        if portal_encapsules_origin(portal, dir) {
            return true;
        }

        let v4 = SupportPoint::compute(obj1, obj2, dir);
        if !portal_can_encapsule_origin(&v4, dir)
            || portal_reach_tolerance(portal, &v4, dir, mpr_tolerance)
        {
            return false;
        }

        expand_portal(portal, v4);
    }

    false
}

fn find_penetration(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    portal: &mut Simplex,
    max_iterations: u64,
    mpr_tolerance: f32,
) -> Option<MprResult> {
    for _ in 0..=max_iterations {
        let dir = portal_dir(portal);
        let v4 = SupportPoint::compute(obj1, obj2, dir);

        if portal_reach_tolerance(portal, &v4, dir, mpr_tolerance) {
            let v1 = portal.get(1)?.v;
            let v2 = portal.get(2)?.v;
            let v3 = portal.get(3)?.v;
            let (depth_sq, witness) = Vec3::point_tri_dist2(Vec3::ZERO, v1, v2, v3);
            let depth = depth_sq.sqrt();
            let dir = if depth < EPSILON {
                Vec3::ZERO
            } else {
                witness.unwrap_or(dir).normalize()
            };
            return Some(MprResult {
                depth,
                dir,
                pos: find_pos(portal),
            });
        }

        expand_portal(portal, v4);
    }

    None
}

fn find_pos(portal: &Simplex) -> Vec3 {
    let dir = portal_dir(portal);
    let v = [
        portal.get(0).unwrap(),
        portal.get(1).unwrap(),
        portal.get(2).unwrap(),
        portal.get(3).unwrap(),
    ];

    let mut b = [0.0; 4];
    b[0] = v[1].v.cross(v[2].v).dot(v[3].v);
    b[1] = v[3].v.cross(v[2].v).dot(v[0].v);
    b[2] = v[0].v.cross(v[1].v).dot(v[3].v);
    b[3] = v[2].v.cross(v[1].v).dot(v[0].v);

    let mut sum = b.iter().sum::<f32>();
    if sum.abs() < EPSILON || sum < 0.0 {
        b[0] = 0.0;
        b[1] = v[2].v.cross(v[3].v).dot(dir);
        b[2] = v[3].v.cross(v[1].v).dot(dir);
        b[3] = v[1].v.cross(v[2].v).dot(dir);
        sum = b[1] + b[2] + b[3];
    }

    let inv = 1.0 / sum.max(EPSILON);
    let mut p1 = Vec3::ZERO;
    let mut p2 = Vec3::ZERO;
    for (weight, point) in b.into_iter().zip(v.into_iter()) {
        p1 = p1 + point.v1 * weight;
        p2 = p2 + point.v2 * weight;
    }

    (p1 * inv + p2 * inv) * 0.5
}

fn portal_dir(portal: &Simplex) -> Vec3 {
    let v2v1 = portal.get(2).unwrap().v - portal.get(1).unwrap().v;
    let v3v1 = portal.get(3).unwrap().v - portal.get(1).unwrap().v;
    v2v1.cross(v3v1).normalize()
}

fn portal_encapsules_origin(portal: &Simplex, dir: Vec3) -> bool {
    let dot = dir.dot(portal.get(1).unwrap().v);
    Vec3::is_zero(dot) || dot > 0.0
}

fn portal_reach_tolerance(portal: &Simplex, v4: &SupportPoint, dir: Vec3, mpr_tolerance: f32) -> bool {
    let dv1 = portal.get(1).unwrap().v.dot(dir);
    let dv2 = portal.get(2).unwrap().v.dot(dir);
    let dv3 = portal.get(3).unwrap().v.dot(dir);
    let dv4 = v4.v.dot(dir);
    let delta = (dv4 - dv1).min(dv4 - dv2).min(dv4 - dv3);
    Vec3::approx_eq(delta, mpr_tolerance) || delta < mpr_tolerance
}

fn portal_can_encapsule_origin(v4: &SupportPoint, dir: Vec3) -> bool {
    let dot = v4.v.dot(dir);
    Vec3::is_zero(dot) || dot > 0.0
}

fn expand_portal(portal: &mut Simplex, v4: SupportPoint) {
    let v4v0 = v4.v.cross(portal.get(0).unwrap().v);
    let dot1 = portal.get(1).unwrap().v.dot(v4v0);

    if dot1 > 0.0 {
        let dot2 = portal.get(2).unwrap().v.dot(v4v0);
        if dot2 > 0.0 {
            portal.set(1, v4);
        } else {
            portal.set(3, v4);
        }
    } else {
        let dot3 = portal.get(3).unwrap().v.dot(v4v0);
        if dot3 > 0.0 {
            portal.set(2, v4);
        } else {
            portal.set(1, v4);
        }
    }
}
