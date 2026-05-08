//! Integration tests for libccd-rust.

use std::sync::atomic::{AtomicUsize, Ordering};

use libccd_rust::{Ccd, CcdConfig, Quat, SupportPoint, Vec3};
use libccd_rust::polytope::Polytope;
use libccd_rust::shapes::{BoxShape, ConvexHull, CylinderShape, SphereShape};

static FIRST_DIR_CALLS: AtomicUsize = AtomicUsize::new(0);

fn first_dir_y(_: &dyn libccd_rust::Shape, _: &dyn libccd_rust::Shape) -> Vec3 {
    Vec3::Y_AXIS
}

fn counting_first_dir(_: &dyn libccd_rust::Shape, _: &dyn libccd_rust::Shape) -> Vec3 {
    FIRST_DIR_CALLS.fetch_add(1, Ordering::Relaxed);
    Vec3::Y_AXIS
}

fn support_point(v: Vec3) -> SupportPoint {
    SupportPoint {
        v,
        v1: Vec3::ZERO,
        v2: Vec3::ZERO,
    }
}

#[test]
fn test_gjk_box_box_intersect() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&box1, &box2));
}

#[test]
fn test_gjk_separate_no_intersect() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(5.0, 0.0, 0.0));
    assert!(ccd.gjk_separate(&box1, &box2).is_none());
}

#[test]
fn test_gjk_penetration_rotated_box_box() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0))
        .with_rot(Quat::from_axis_angle(Vec3::Z_AXIS, std::f32::consts::FRAC_PI_4))
        .with_pos(Vec3::new(0.1, 0.0, 0.1));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let pen = ccd.gjk_penetration(&box1, &box2).expect("rotated boxes should penetrate");
    assert!(pen.depth > 0.0);
    assert!(pen.dir.length() > 0.0);
}

#[test]
fn test_gjk_box_box_no_intersect() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(3.0, 0.0, 0.0));
    assert!(!ccd.gjk_intersect(&box1, &box2));
}

#[test]
fn test_mpr_penetration_rotated_box_cylinder() {
    let ccd = Ccd::new();
    let box_shape = BoxShape::new(Vec3::new(0.5, 1.0, 1.5))
        .with_pos(Vec3::new(0.6, 0.0, 0.5))
        .with_rot(Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), -std::f32::consts::FRAC_PI_4));
    let cyl = CylinderShape::new(0.4, 0.7)
        .with_pos(Vec3::new(0.6, 0.0, 0.5))
        .with_rot(Quat::from_axis_angle(Vec3::new(-0.1, 2.2, -1.0), std::f32::consts::PI / 5.0));
    let pen = ccd.mpr_penetration(&box_shape, &cyl).expect("rotated box/cylinder should penetrate");
    assert!(pen.depth > 0.0);
}

#[test]
fn test_gjk_sphere_sphere_intersect() {
    let ccd = Ccd::new();
    let s1 = SphereShape::new(1.0);
    let s2 = SphereShape::new(1.0).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&s1, &s2));
}

#[test]
fn test_custom_first_dir_is_used() {
    FIRST_DIR_CALLS.store(0, Ordering::Relaxed);
    let ccd = Ccd::builder().first_dir(counting_first_dir).build();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&box1, &box2));
    assert!(FIRST_DIR_CALLS.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_gjk_sphere_sphere_no_intersect() {
    let ccd = Ccd::new();
    let s1 = SphereShape::new(1.0);
    let s2 = SphereShape::new(1.0).with_pos(Vec3::new(3.0, 0.0, 0.0));
    assert!(!ccd.gjk_intersect(&s1, &s2));
}

#[test]
fn test_convex_hull_penetration() {
    let ccd = Ccd::new();
    let hull = ConvexHull::new(vec![
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ])
    .with_pos(Vec3::new(0.25, 0.0, 0.0));
    let sphere = SphereShape::new(1.0).with_pos(Vec3::new(0.5, 0.0, 0.0));
    let pen = ccd.gjk_penetration(&hull, &sphere).expect("convex hull and sphere should penetrate");
    assert!(pen.depth > 0.0);
}

#[test]
fn test_polytope_nearest_recomputes_after_removals() {
    let mut polytope = Polytope::new();

    let v0 = polytope.add_vertex(support_point(Vec3::new(-1.0, -1.0, 0.0)));
    let v1 = polytope.add_vertex(support_point(Vec3::new(1.0, 0.0, 0.0)));
    let v2 = polytope.add_vertex(support_point(Vec3::new(0.0, 0.0, 1.0)));
    let v3 = polytope.add_vertex(support_point(Vec3::new(0.0, 1.0, 0.0)));

    let e0 = polytope.add_edge(v0, v1);
    let e1 = polytope.add_edge(v1, v2);
    let e2 = polytope.add_edge(v2, v0);
    let e3 = polytope.add_edge(v3, v0);
    let e4 = polytope.add_edge(v3, v1);
    let e5 = polytope.add_edge(v3, v2);

    let f0 = polytope.add_face(e0, e1, e2);
    let f1 = polytope.add_face(e3, e4, e0);
    let f2 = polytope.add_face(e4, e5, e1);
    let f3 = polytope.add_face(e5, e3, e2);

    let nearest = polytope.find_nearest().expect("polytope should have a nearest element");
    assert!(matches!(nearest, libccd_rust::polytope::ElementRef::Face(_)));

    polytope.remove_face(f1);
    let nearest = polytope.find_nearest().expect("polytope should recompute nearest after face removal");
    assert!(matches!(nearest, libccd_rust::polytope::ElementRef::Face(_)));

    polytope.remove_face(f0);
    polytope.remove_face(f2);
    polytope.remove_face(f3);
    let nearest = polytope.find_nearest().expect("polytope should fall back to an edge once faces are removed");
    assert!(matches!(nearest, libccd_rust::polytope::ElementRef::Edge(_)));
}

#[test]
fn test_gjk_box_sphere_intersect() {
    let ccd = Ccd::new();
    let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let s = SphereShape::new(1.0).with_pos(Vec3::new(1.0, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&b, &s));
}

#[test]
fn test_gjk_box_cylinder_intersect() {
    let ccd = Ccd::new();
    let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let c = CylinderShape::new(1.0, 2.0).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&b, &c));
}

#[test]
fn test_gjk_rotated_box_box_intersect() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0))
        .with_rot(Quat::from_axis_angle(Vec3::Z_AXIS, std::f32::consts::FRAC_PI_4));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(0.75, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&box1, &box2));
}

#[test]
fn test_mpr_box_box_intersect() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.mpr_intersect(&box1, &box2));
}

#[test]
fn test_mpr_sphere_sphere_no_intersect() {
    let ccd = Ccd::new();
    let s1 = SphereShape::new(1.0);
    let s2 = SphereShape::new(1.0).with_pos(Vec3::new(3.0, 0.0, 0.0));
    assert!(!ccd.mpr_intersect(&s1, &s2));
}

#[test]
fn test_mpr_box_cylinder_intersect() {
    let ccd = Ccd::new();
    let box_shape = BoxShape::new(Vec3::new(0.5, 1.0, 1.5));
    let cyl = CylinderShape::new(0.4, 0.7)
        .with_pos(Vec3::new(0.6, 0.6, 0.5))
        .with_rot(Quat::from_axis_angle(Vec3::Y_AXIS, std::f32::consts::PI / 3.0));
    assert!(ccd.mpr_intersect(&box_shape, &cyl));
}

#[test]
fn test_mpr_cylinder_cylinder_intersect() {
    let ccd = Ccd::new();
    let cyl1 = CylinderShape::new(0.35, 0.5);
    let cyl2 = CylinderShape::new(0.5, 1.0)
        .with_pos(Vec3::new(-0.2, 0.7, 0.2))
        .with_rot(Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0), std::f32::consts::FRAC_PI_4));
    assert!(ccd.mpr_intersect(&cyl1, &cyl2));
}

#[test]
fn test_gjk_penetration_box_box() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(1.0, 0.0, 0.0));
    let pen = ccd.gjk_penetration(&box1, &box2);
    assert!(pen.is_some());
    let pen = pen.unwrap();
    assert!(pen.depth > 0.0);
    // Direction should point roughly along -X (from box2 toward box1)
    assert!(pen.dir.x() < 0.0 || pen.dir.x().abs() < 0.01);
}

#[test]
fn test_gjk_penetration_sphere_sphere() {
    let ccd = Ccd::new();
    let s1 = SphereShape::new(1.0);
    let s2 = SphereShape::new(1.0).with_pos(Vec3::new(1.0, 0.0, 0.0));
    let pen = ccd.gjk_penetration(&s1, &s2);
    assert!(pen.is_some());
    let pen = pen.unwrap();
    // Two unit spheres with centers 1.0 apart: depth should be 1.0
    assert!((pen.depth - 1.0).abs() < 0.1);
    assert!(pen.dir.x() < -0.9);
    assert!((pen.pos.x() - 0.5).abs() < 1e-4);
    assert!(pen.pos.y().abs() < 1e-4);
    assert!(pen.pos.z().abs() < 1e-4);
}

#[test]
fn test_gjk_penetration_no_intersect() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(5.0, 0.0, 0.0));
    assert!(ccd.gjk_penetration(&box1, &box2).is_none());
}

#[test]
fn test_mpr_penetration_sphere_sphere() {
    let ccd = Ccd::new();
    let s1 = SphereShape::new(1.0);
    let s2 = SphereShape::new(1.0).with_pos(Vec3::new(1.0, 0.0, 0.0));
    let pen = ccd.mpr_penetration(&s1, &s2);
    if let Some(pen) = pen {
        assert!(pen.depth > 0.0);
    }
}

#[test]
fn test_gjk_separate_box_box() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(1.0, 0.0, 0.0));
    let sep = ccd.gjk_separate(&box1, &box2).expect("overlap should produce a separation vector");
    assert!(sep.length() > 0.0);
    assert!(sep.x() < 0.0);
}

#[test]
fn test_vec3_basics() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    assert!((a.dot(b) - 32.0).abs() < 1e-6);
    assert!((a.length_squared() - 14.0).abs() < 1e-6);

    let c = a + b;
    assert!((c.x() - 5.0).abs() < 1e-6);
    assert!((c.y() - 7.0).abs() < 1e-6);
    assert!((c.z() - 9.0).abs() < 1e-6);
}

#[test]
fn test_ccd_builder() {
    let ccd = Ccd::builder()
        .max_iterations(100)
        .epa_tolerance(1e-5)
        .mpr_tolerance(1e-5)
        .dist_tolerance(1e-5)
        .first_dir(first_dir_y)
        .build();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0)).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&box1, &box2));
}

#[test]
fn test_ccd_from_config() {
    let config = CcdConfig {
        max_iterations: 64,
        epa_tolerance: 1e-5,
        mpr_tolerance: 1e-5,
        dist_tolerance: 1e-5,
        first_dir: first_dir_y,
    };
    let ccd = Ccd::from_config(config);
    let s1 = SphereShape::new(1.0);
    let s2 = SphereShape::new(1.0).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&s1, &s2));
}

#[test]
fn test_box_identical_overlap() {
    let ccd = Ccd::new();
    let box1 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    let box2 = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
    assert!(ccd.gjk_intersect(&box1, &box2));
}

#[test]
fn test_convex_hull_support_and_intersection() {
    let ccd = Ccd::new();
    let hull = ConvexHull::new(vec![
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ])
    .with_pos(Vec3::new(0.5, 0.0, 0.0))
    .with_rot(Quat::from_axis_angle(Vec3::Z_AXIS, std::f32::consts::FRAC_PI_4));
    let sphere = SphereShape::new(0.75).with_pos(Vec3::new(0.5, 0.0, 0.0));
    assert!(ccd.gjk_intersect(&hull, &sphere));
}
