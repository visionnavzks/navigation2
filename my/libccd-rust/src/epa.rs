//! Expanding Polytope Algorithm (EPA) for computing penetration depth.

use std::time::{Duration, Instant};

use crate::gjk::{gjk, GjkResult};
use crate::polytope::{ElementRef, Polytope};
use crate::shapes::Shape;
use crate::simplex::Simplex;
use crate::support::SupportPoint;
use crate::vec3::Vec3;

/// Penetration result from EPA.
#[derive(Debug, Clone, Copy)]
pub struct EpaResult {
    pub depth: f32,
    pub dir: Vec3,
    pub pos: Vec3,
}

#[derive(Default)]
struct EpaTraceStats {
    iterations: u64,
    nearest_calls: u64,
    nearest_time: Duration,
    feasible_time: Duration,
    expand_time: Duration,
    face_expands: u64,
    edge_expands: u64,
    edge_fallbacks: u64,
}

fn emit_trace_summary(
    trace_enabled: bool,
    outcome: &str,
    stats: &EpaTraceStats,
    polytope: Option<&Polytope>,
) {
    if !trace_enabled {
        return;
    }

    let (active_vertices, total_vertices, active_edges, total_edges, active_faces, total_faces) =
        if let Some(polytope) = polytope {
            let active_vertices = polytope
                .vertices
                .iter()
                .filter(|vertex| vertex.dist < f32::MAX / 2.0)
                .count();
            let active_edges = polytope
                .edges
                .iter()
                .filter(|edge| edge.dist < f32::MAX / 2.0)
                .count();
            let active_faces = polytope
                .faces
                .iter()
                .filter(|face| face.dist < f32::MAX / 2.0)
                .count();
            (
                active_vertices,
                polytope.vertices.len(),
                active_edges,
                polytope.edges.len(),
                active_faces,
                polytope.faces.len(),
            )
        } else {
            (0, 0, 0, 0, 0, 0)
        };

    eprintln!(
        "epa_trace outcome={} iterations={} nearest_calls={} nearest_ns={} feasible_ns={} expand_ns={} face_expands={} edge_expands={} edge_fallbacks={} vertices={}/{} edges={}/{} faces={}/{}",
        outcome,
        stats.iterations,
        stats.nearest_calls,
        stats.nearest_time.as_nanos(),
        stats.feasible_time.as_nanos(),
        stats.expand_time.as_nanos(),
        stats.face_expands,
        stats.edge_expands,
        stats.edge_fallbacks,
        active_vertices,
        total_vertices,
        active_edges,
        total_edges,
        active_faces,
        total_faces,
    );
}

/// Run GJK+EPA to find penetration info.
///
/// Returns `Some(EpaResult)` if objects intersect, `None` otherwise.
pub fn gjk_epa(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    max_iterations: u64,
    epa_tolerance: f32,
    first_dir: crate::gjk::FirstDirFn,
    dist_tolerance: f32,
) -> Option<EpaResult> {
    let trace_enabled = std::env::var_os("LIBCCD_EPA_TRACE").is_some();
    let mut trace_stats = EpaTraceStats::default();

    // Run GJK to get terminal simplex
    let simplex = match gjk(obj1, obj2, max_iterations, first_dir, dist_tolerance) {
        GjkResult::Intersection(s) => s,
        GjkResult::NoIntersection => {
            emit_trace_summary(trace_enabled, "gjk_no_intersection", &trace_stats, None);
            return None;
        }
    };

    // Convert simplex to polytope
    let mut polytope = Polytope::new();
    let Some(_nearest_el) = simplex_to_polytope(obj1, obj2, &simplex, &mut polytope, epa_tolerance) else {
        emit_trace_summary(trace_enabled, "simplex_to_polytope_none", &trace_stats, None);
        return None;
    };

    // EPA expansion loop with iteration limit
    let epa_max = if max_iterations == u64::MAX { 1000u64 } else { max_iterations };
    for _ in 0..epa_max {
        trace_stats.iterations += 1;

        let nearest_start = trace_enabled.then(Instant::now);
        let nearest_el = polytope.find_nearest()?;
        let expand_el = preferred_expand_element(&polytope, nearest_el);
        if let Some(start) = nearest_start {
            trace_stats.nearest_calls += 1;
            trace_stats.nearest_time += start.elapsed();
        }

        if matches!(expand_el, ElementRef::Vertex(_)) {
            break;
        }

        // Get next support point in nearest element's direction
        let witness = match expand_el {
            ElementRef::Vertex(i) => polytope.vertices[i].witness,
            ElementRef::Edge(i) => polytope.edges[i].witness,
            ElementRef::Face(i) => polytope.faces[i].witness,
        };

        let new_support = SupportPoint::compute(obj1, obj2, witness);

        // Check convergence
        let dist_along = new_support.v.dot(witness);
        let nearest_dist = match expand_el {
            ElementRef::Vertex(i) => polytope.vertices[i].dist,
            ElementRef::Edge(i) => polytope.edges[i].dist,
            ElementRef::Face(i) => polytope.faces[i].dist,
        };

        if dist_along - nearest_dist < epa_tolerance {
            break;
        }

        // Check if new support can significantly expand polytope
        let feasible_start = trace_enabled.then(Instant::now);
        if !next_support_feasible(&polytope, expand_el, &new_support, epa_tolerance) {
            if let Some(start) = feasible_start {
                trace_stats.feasible_time += start.elapsed();
            }
            break;
        }
        if let Some(start) = feasible_start {
            trace_stats.feasible_time += start.elapsed();
        }

        // Expand polytope
        let expand_start = trace_enabled.then(Instant::now);
        if !expand_polytope(&mut polytope, expand_el, new_support, &mut trace_stats) {
            emit_trace_summary(trace_enabled, "expand_failed", &trace_stats, Some(&polytope));
            return None; // allocation failure
        }
        if let Some(start) = expand_start {
            trace_stats.expand_time += start.elapsed();
        }
    }

    // Extract penetration info from nearest element
    let nearest_start = trace_enabled.then(Instant::now);
    let final_nearest = polytope.find_nearest()?;
    if let Some(start) = nearest_start {
        trace_stats.nearest_calls += 1;
        trace_stats.nearest_time += start.elapsed();
    }

    emit_trace_summary(trace_enabled, "ok", &trace_stats, Some(&polytope));

    extract_penetration(&mut polytope, final_nearest, obj1, obj2)
}

/// Convert GJK terminal simplex to EPA polytope.
fn simplex_to_polytope(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    simplex: &Simplex,
    polytope: &mut Polytope,
    epa_tolerance: f32,
) -> Option<ElementRef> {
    match simplex.len() {
        4 => simplex_to_polytope4(obj1, obj2, simplex, polytope),
        3 => simplex_to_polytope3(obj1, obj2, simplex, polytope),
        2 => simplex_to_polytope2(obj1, obj2, simplex, polytope, epa_tolerance),
        _ => None,
    }
}

/// Convert 4-point simplex (tetrahedron) to polytope.
fn simplex_to_polytope4(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    simplex: &Simplex,
    polytope: &mut Polytope,
) -> Option<ElementRef> {
    let a = *simplex.get(0)?;
    let b = *simplex.get(1)?;
    let c = *simplex.get(2)?;
    let d = *simplex.get(3)?;

    // Check if origin lies on any face → degenerate, use 3-point case
    let (dist_abc, _) = Vec3::point_tri_dist2(Vec3::ZERO, a.v, b.v, c.v);
    let (dist_acd, _) = Vec3::point_tri_dist2(Vec3::ZERO, a.v, c.v, d.v);
    let (dist_abd, _) = Vec3::point_tri_dist2(Vec3::ZERO, a.v, b.v, d.v);
    let (dist_bcd, _) = Vec3::point_tri_dist2(Vec3::ZERO, b.v, c.v, d.v);

    if Vec3::is_zero(dist_abc) || Vec3::is_zero(dist_acd) || Vec3::is_zero(dist_abd) || Vec3::is_zero(dist_bcd) {
        let mut degenerate = Simplex::new();
        degenerate.push(a);
        degenerate.push(b);
        degenerate.push(c);
        degenerate.set_count(3);
        return simplex_to_polytope3(obj1, obj2, &degenerate, polytope);
    }

    // Create tetrahedron
    let v0 = polytope.add_vertex(a);
    let v1 = polytope.add_vertex(b);
    let v2 = polytope.add_vertex(c);
    let v3 = polytope.add_vertex(d);

    let e0 = polytope.add_edge(v0, v1);
    let e1 = polytope.add_edge(v1, v2);
    let e2 = polytope.add_edge(v2, v0);
    let e3 = polytope.add_edge(v3, v0);
    let e4 = polytope.add_edge(v3, v1);
    let e5 = polytope.add_edge(v3, v2);

    polytope.add_face(e0, e1, e2);
    polytope.add_face(e3, e4, e0);
    polytope.add_face(e4, e5, e1);
    polytope.add_face(e5, e3, e2);

    polytope.find_nearest()
}

/// Convert 3-point simplex (triangle) to bipyramid polytope.
fn simplex_to_polytope3(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    simplex: &Simplex,
    polytope: &mut Polytope,
) -> Option<ElementRef> {
    let a = *simplex.get(0)?;
    let b = *simplex.get(1)?;
    let c = *simplex.get(2)?;

    let ab = b.v - a.v;
    let ac = c.v - a.v;
    let normal = ab.cross(ac);

    let d = SupportPoint::compute(obj1, obj2, normal);
    let d2 = SupportPoint::compute(obj1, obj2, -normal);

    // Check for touching contact
    let (dist_d, _) = Vec3::point_tri_dist2(d.v, a.v, b.v, c.v);
    let (dist_d2, _) = Vec3::point_tri_dist2(d2.v, a.v, b.v, c.v);

    if Vec3::is_zero(dist_d) || Vec3::is_zero(dist_d2) {
        // Touching contact
        let v0 = polytope.add_vertex(a);
        let v1 = polytope.add_vertex(b);
        let v2 = polytope.add_vertex(c);
        let e0 = polytope.add_edge(v0, v1);
        let e1 = polytope.add_edge(v1, v2);
        let e2 = polytope.add_edge(v2, v0);
        let face_idx = polytope.add_face(e0, e1, e2);
        return Some(ElementRef::Face(face_idx));
    }

    // Create bipyramid (5 vertices, 6 faces)
    let v0 = polytope.add_vertex(a);
    let v1 = polytope.add_vertex(b);
    let v2 = polytope.add_vertex(c);
    let v3 = polytope.add_vertex(d);
    let v4 = polytope.add_vertex(d2);

    let e0 = polytope.add_edge(v0, v1);
    let e1 = polytope.add_edge(v1, v2);
    let e2 = polytope.add_edge(v2, v0);
    let e3 = polytope.add_edge(v3, v0);
    let e4 = polytope.add_edge(v3, v1);
    let e5 = polytope.add_edge(v3, v2);
    let e6 = polytope.add_edge(v4, v0);
    let e7 = polytope.add_edge(v4, v1);
    let e8 = polytope.add_edge(v4, v2);

    polytope.add_face(e3, e4, e0);
    polytope.add_face(e4, e5, e1);
    polytope.add_face(e5, e3, e2);
    polytope.add_face(e6, e7, e0);
    polytope.add_face(e7, e8, e1);
    polytope.add_face(e8, e6, e2);

    polytope.find_nearest()
}

/// Convert 2-point simplex (segment) to polytope using search directions.
fn simplex_to_polytope2(
    obj1: &dyn Shape,
    obj2: &dyn Shape,
    simplex: &Simplex,
    polytope: &mut Polytope,
    _epa_tolerance: f32,
) -> Option<ElementRef> {
    let a = *simplex.get(0)?;
    let b = *simplex.get(1)?;

    // Use points_on_sphere to find additional support points
    let mut supps = Vec::new();
    for dir in POINTS_ON_SPHERE.iter() {
        let s = SupportPoint::compute(obj1, obj2, *dir);
        if !s.v.vec_approx_eq(a.v) && !s.v.vec_approx_eq(b.v) {
            supps.push(s);
            if supps.len() >= 4 {
                break;
            }
        }
    }

    if supps.is_empty() {
        // Touching contact — degenerate
        let v0 = polytope.add_vertex(a);
        let v1 = polytope.add_vertex(b);
        let e = polytope.add_edge(v0, v1);
        return Some(ElementRef::Edge(e));
    }

    // Find more support points to form a full polyhedron
    let dir_neg = -supps[0].v;
    let s1 = SupportPoint::compute(obj1, obj2, dir_neg);
    if s1.v.vec_approx_eq(a.v) || s1.v.vec_approx_eq(b.v) {
        let v0 = polytope.add_vertex(a);
        let v1 = polytope.add_vertex(b);
        let e = polytope.add_edge(v0, v1);
        return Some(ElementRef::Edge(e));
    }

    let ab = supps[0].v - a.v;
    let ac = s1.v - a.v;
    let normal = ab.cross(ac);
    let s2 = SupportPoint::compute(obj1, obj2, normal);
    if s2.v.vec_approx_eq(a.v) || s2.v.vec_approx_eq(b.v) {
        let v0 = polytope.add_vertex(a);
        let v1 = polytope.add_vertex(b);
        let e = polytope.add_edge(v0, v1);
        return Some(ElementRef::Edge(e));
    }

    let s3 = SupportPoint::compute(obj1, obj2, -normal);
    if s3.v.vec_approx_eq(a.v) || s3.v.vec_approx_eq(b.v) {
        let v0 = polytope.add_vertex(a);
        let v1 = polytope.add_vertex(b);
        let e = polytope.add_edge(v0, v1);
        return Some(ElementRef::Edge(e));
    }

    // Build octahedron-like polytope
    let v0 = polytope.add_vertex(a);
    let v1 = polytope.add_vertex(supps[0]);
    let v2 = polytope.add_vertex(b);
    let v3 = polytope.add_vertex(s1);
    let v4 = polytope.add_vertex(s2);
    let v5 = polytope.add_vertex(s3);

    let e0 = polytope.add_edge(v0, v1);
    let e1 = polytope.add_edge(v1, v2);
    let e2 = polytope.add_edge(v2, v3);
    let e3 = polytope.add_edge(v3, v0);
    let e4 = polytope.add_edge(v4, v0);
    let e5 = polytope.add_edge(v4, v1);
    let e6 = polytope.add_edge(v4, v2);
    let e7 = polytope.add_edge(v4, v3);
    let e8 = polytope.add_edge(v5, v0);
    let e9 = polytope.add_edge(v5, v1);
    let e10 = polytope.add_edge(v5, v2);
    let e11 = polytope.add_edge(v5, v3);

    polytope.add_face(e4, e5, e0);
    polytope.add_face(e5, e6, e1);
    polytope.add_face(e6, e7, e2);
    polytope.add_face(e7, e4, e3);
    polytope.add_face(e8, e9, e0);
    polytope.add_face(e9, e10, e1);
    polytope.add_face(e10, e11, e2);
    polytope.add_face(e11, e8, e3);

    polytope.find_nearest()
}

/// Check if a new support point can significantly expand the polytope.
fn next_support_feasible(
    polytope: &Polytope,
    el: ElementRef,
    new_support: &SupportPoint,
    epa_tolerance: f32,
) -> bool {
    let (witness, dist, el_type) = match el {
        ElementRef::Vertex(i) => (polytope.vertices[i].witness, polytope.vertices[i].dist, 1),
        ElementRef::Edge(i) => (polytope.edges[i].witness, polytope.edges[i].dist, 2),
        ElementRef::Face(i) => (polytope.faces[i].witness, polytope.faces[i].dist, 3),
    };

    // Touching contact — can't expand further
    if Vec3::is_zero(dist) {
        return false;
    }

    // Check if new support point is far enough from nearest element
    let dist_along = new_support.v.dot(witness);
    if dist_along - dist < epa_tolerance {
        return false;
    }

    // Check distance from new support point to element
    let point_dist = match el_type {
        2 => {
            // Edge: check distance to segment
            let edge = &polytope.edges[match el {
                ElementRef::Edge(i) => i,
                _ => 0,
            }];
            let a = polytope.vertices[edge.vertices[0]].support.v;
            let b = polytope.vertices[edge.vertices[1]].support.v;
            Vec3::point_segment_dist2(new_support.v, a, b).0
        }
        3 => {
            // Face: check distance to triangle
            let _face = &polytope.faces[match el {
                ElementRef::Face(i) => i,
                _ => 0,
            }];
            let (va, vb, vc) = polytope.face_positions(match el {
                ElementRef::Face(i) => i,
                _ => 0,
            });
            Vec3::point_tri_dist2(new_support.v, va, vb, vc).0
        }
        _ => return false, // Vertex terminates EPA in the original libccd flow
    };

    point_dist >= epa_tolerance
}

fn preferred_expand_element(polytope: &Polytope, el: ElementRef) -> ElementRef {
    match el {
        ElementRef::Edge(edge_idx) => {
            let edge = &polytope.edges[edge_idx];
            match (edge.faces[0], edge.faces[1]) {
                (Some(face0), Some(face1)) => {
                    if polytope.faces[face0].dist <= polytope.faces[face1].dist {
                        ElementRef::Face(face0)
                    } else {
                        ElementRef::Face(face1)
                    }
                }
                (Some(face_idx), None) | (None, Some(face_idx)) => ElementRef::Face(face_idx),
                (None, None) => el,
            }
        }
        _ => el,
    }
}

/// Expand polytope with a new vertex, replacing the nearest element.
fn expand_polytope(
    polytope: &mut Polytope,
    el: ElementRef,
    new_support: SupportPoint,
    trace_stats: &mut EpaTraceStats,
) -> bool {
    match el {
        ElementRef::Face(face_idx) => {
            trace_stats.face_expands += 1;
            expand_polytope_face(polytope, face_idx, new_support)
        }
        ElementRef::Edge(edge_idx) => {
            trace_stats.edge_expands += 1;
            expand_polytope_edge(polytope, edge_idx, new_support, trace_stats)
        }
        ElementRef::Vertex(_) => false, // Can't expand from a vertex
    }
}

/// Expand polytope from a face by adding a new vertex (tetrahedron split).
fn expand_polytope_face(polytope: &mut Polytope, face_idx: usize, new_support: SupportPoint) -> bool {
    let face = &polytope.faces[face_idx];
    let e0_idx = face.edges[0];
    let e1_idx = face.edges[1];
    let e2_idx = face.edges[2];

    let (v0, v1, v2) = polytope.face_vertices_from_edges(e0_idx, e1_idx, e2_idx);

    // Remove old face
    polytope.remove_face(face_idx);

    // Add new vertex and three edges connecting it to face vertices
    let v3 = polytope.add_vertex(new_support);
    let e3 = polytope.add_edge(v3, v0);
    let e4 = polytope.add_edge(v3, v1);
    let e5 = polytope.add_edge(v3, v2);

    // Add three new faces
    polytope.add_face(e3, e4, e0_idx);
    polytope.add_face(e4, e5, e1_idx);
    polytope.add_face(e5, e3, e2_idx);

    true
}

fn expand_polytope_edge_fallback(
    polytope: &mut Polytope,
    edge_idx: usize,
    new_support: SupportPoint,
    trace_stats: &mut EpaTraceStats,
) -> bool {
    trace_stats.edge_fallbacks += 1;
    let edge = &polytope.edges[edge_idx];
    let face0 = edge.faces[0];
    let face1 = edge.faces[1];

    if face0.is_none() {
        let v0 = edge.vertices[0];
        let v2 = edge.vertices[1];
        let v4 = polytope.add_vertex(new_support);
        let e4 = polytope.add_edge(v4, v2);
        let e5 = polytope.add_edge(v4, v0);
        polytope.add_face(edge_idx, e4, e5);
        return true;
    }

    if let Some(face_idx) = face0 {
        trace_stats.face_expands += 1;
        expand_polytope_face(polytope, face_idx, new_support);
    }
    if let Some(face_idx) = face1 {
        trace_stats.face_expands += 1;
        expand_polytope_face(polytope, face_idx, new_support);
    }

    true
}

fn ordered_face_edges(
    polytope: &Polytope,
    face_idx: usize,
    excluded_edge_idx: usize,
    start_vertex: usize,
    end_vertex: usize,
) -> Option<(usize, usize, usize)> {
    let face = polytope.faces.get(face_idx)?;
    let mut remaining = face
        .edges
        .into_iter()
        .filter(|edge_idx| *edge_idx != excluded_edge_idx);
    let edge_a = remaining.next()?;
    let edge_b = remaining.next()?;

    let classify = |edge_idx: usize| {
        let edge = &polytope.edges[edge_idx];
        let [v0, v1] = edge.vertices;
        if v0 == start_vertex {
            Some((edge_idx, v1))
        } else if v1 == start_vertex {
            Some((edge_idx, v0))
        } else {
            None
        }
    };

    let (edge_from_start, middle_vertex, edge_to_end) = if let Some((edge_idx, middle)) = classify(edge_a) {
        (edge_idx, middle, edge_b)
    } else if let Some((edge_idx, middle)) = classify(edge_b) {
        (edge_idx, middle, edge_a)
    } else {
        return None;
    };

    let end_edge = &polytope.edges[edge_to_end];
    if !end_edge.vertices.contains(&middle_vertex) || !end_edge.vertices.contains(&end_vertex) {
        return None;
    }

    Some((edge_from_start, edge_to_end, middle_vertex))
}

/// Expand polytope from an edge by adding a new vertex.
fn expand_polytope_edge(
    polytope: &mut Polytope,
    edge_idx: usize,
    new_support: SupportPoint,
    trace_stats: &mut EpaTraceStats,
) -> bool {
    let edge = &polytope.edges[edge_idx];
    let v0 = edge.vertices[0];
    let v2 = edge.vertices[1];
    let face0 = edge.faces[0];
    let face1 = edge.faces[1];

    if let Some(face0_idx) = face0 {
        let Some((edge0, edge1, v1)) = ordered_face_edges(polytope, face0_idx, edge_idx, v0, v2) else {
            return expand_polytope_edge_fallback(polytope, edge_idx, new_support, trace_stats);
        };

        let face1_data = if let Some(face1_idx) = face1 {
            let Some((edge2, edge3, v3)) = ordered_face_edges(polytope, face1_idx, edge_idx, v2, v0) else {
                return expand_polytope_edge_fallback(polytope, edge_idx, new_support, trace_stats);
            };
            Some((face1_idx, edge2, edge3, v3))
        } else {
            None
        };

        polytope.remove_face(face0_idx);
        if let Some((face1_idx, ..)) = face1_data {
            polytope.remove_face(face1_idx);
            polytope.remove_edge(edge_idx);
        }

        let v4 = polytope.add_vertex(new_support);
        let edge4 = polytope.add_edge(v4, v2);
        let edge5 = polytope.add_edge(v4, v0);
        let edge6 = polytope.add_edge(v4, v1);

        if polytope.add_face(edge1, edge4, edge6) >= polytope.faces.len() {
            return false;
        }
        if polytope.add_face(edge0, edge6, edge5) >= polytope.faces.len() {
            return false;
        }

        if let Some((_, edge2, edge3, v3)) = face1_data {
            let edge7 = polytope.add_edge(v4, v3);
            polytope.add_face(edge3, edge5, edge7);
            polytope.add_face(edge4, edge7, edge2);
        } else {
            polytope.add_face(edge4, edge5, edge_idx);
        }

        true
    } else {
        let v4 = polytope.add_vertex(new_support);
        let edge4 = polytope.add_edge(v4, v2);
        let edge5 = polytope.add_edge(v4, v0);
        polytope.add_face(edge_idx, edge4, edge5);
        true
    }
}

/// Extract penetration info from the polytope's nearest element.
fn extract_penetration(
    polytope: &mut Polytope,
    _nearest: ElementRef,
    _obj1: &dyn Shape,
    _obj2: &dyn Shape,
) -> Option<EpaResult> {
    let nearest = polytope.find_nearest()?;

    let (depth, dir) = match nearest {
        ElementRef::Vertex(i) => {
            let v = &polytope.vertices[i];
            (v.dist.sqrt(), v.witness.normalize())
        }
        ElementRef::Edge(i) => {
            let e = &polytope.edges[i];
            (e.dist.sqrt(), e.witness.normalize())
        }
        ElementRef::Face(i) => {
            let f = &polytope.faces[i];
            (f.dist.sqrt(), f.witness.normalize())
        }
    };

    // Compute position using median of closest vertices
    let pos = compute_position(polytope, nearest);

    Some(EpaResult {
        depth,
        dir,
        pos,
    })
}

/// Compute approximate contact position from polytope vertices.
fn compute_position(polytope: &Polytope, _nearest: ElementRef) -> Vec3 {
    // Sort vertices by distance, take median of closest half
    let mut indices: Vec<usize> = (0..polytope.vertices.len()).collect();
    indices.sort_by(|a, b| {
        polytope.vertices[*a]
            .dist
            .partial_cmp(&polytope.vertices[*b].dist)
            .unwrap()
    });

    let half = (indices.len() + 1) / 2;
    let mut sum = Vec3::ZERO;
    let mut count = 0.0;
    for i in 0..half {
        sum = sum + polytope.vertices[indices[i]].support.v1;
        sum = sum + polytope.vertices[indices[i]].support.v2;
        count += 2.0;
    }

    sum * (1.0 / count)
}

/// 42 uniformly distributed points on the unit sphere.
/// Used by simplexToPolytope2 to find additional support directions.
static POINTS_ON_SPHERE: [Vec3; 42] = [
    Vec3::new(0.000000, -0.000000, -1.000000),
    Vec3::new(0.723608, -0.525725, -0.447219),
    Vec3::new(-0.276388, -0.850649, -0.447219),
    Vec3::new(-0.894426, -0.000000, -0.447216),
    Vec3::new(-0.276388, 0.850649, -0.447220),
    Vec3::new(0.723608, 0.525725, -0.447219),
    Vec3::new(0.276388, -0.850649, 0.447220),
    Vec3::new(-0.723608, -0.525725, 0.447219),
    Vec3::new(-0.723608, 0.525725, 0.447219),
    Vec3::new(0.276388, 0.850649, 0.447219),
    Vec3::new(0.894426, 0.000000, 0.447216),
    Vec3::new(-0.000000, 0.000000, 1.000000),
    Vec3::new(0.425323, -0.309011, -0.850654),
    Vec3::new(-0.162456, -0.499995, -0.850654),
    Vec3::new(0.262869, -0.809012, -0.525738),
    Vec3::new(0.425323, 0.309011, -0.850654),
    Vec3::new(0.850648, -0.000000, -0.525736),
    Vec3::new(-0.525730, -0.000000, -0.850652),
    Vec3::new(-0.688190, -0.499997, -0.525736),
    Vec3::new(-0.162456, 0.499995, -0.850654),
    Vec3::new(-0.688190, 0.499997, -0.525736),
    Vec3::new(0.262869, 0.809012, -0.525738),
    Vec3::new(0.951058, 0.309013, 0.000000),
    Vec3::new(0.951058, -0.309013, 0.000000),
    Vec3::new(0.587786, -0.809017, 0.000000),
    Vec3::new(0.000000, -1.000000, 0.000000),
    Vec3::new(-0.587786, -0.809017, 0.000000),
    Vec3::new(-0.951058, -0.309013, -0.000000),
    Vec3::new(-0.951058, 0.309013, -0.000000),
    Vec3::new(-0.587786, 0.809017, -0.000000),
    Vec3::new(-0.000000, 1.000000, -0.000000),
    Vec3::new(0.587786, 0.809017, -0.000000),
    Vec3::new(0.688190, -0.499997, 0.525736),
    Vec3::new(-0.262869, -0.809012, 0.525738),
    Vec3::new(-0.850648, 0.000000, 0.525736),
    Vec3::new(-0.262869, 0.809012, 0.525738),
    Vec3::new(0.688190, 0.499997, 0.525736),
    Vec3::new(0.525730, 0.000000, 0.850652),
    Vec3::new(0.162456, -0.499995, 0.850654),
    Vec3::new(-0.425323, -0.309011, 0.850654),
    Vec3::new(-0.425323, 0.309011, 0.850654),
    Vec3::new(0.162456, 0.499995, 0.850654),
];