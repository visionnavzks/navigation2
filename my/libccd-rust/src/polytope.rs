//! Polytope for EPA algorithm (index-based topology).

use crate::support::SupportPoint;
use crate::vec3::Vec3;

/// Index-based reference to a polytope element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElementRef {
    Vertex(usize),
    Edge(usize),
    Face(usize),
}

/// A vertex in the polytope.
#[derive(Clone, Debug)]
pub struct Vertex {
    pub support: SupportPoint,
    pub dist: f32,
    pub witness: Vec3,
    /// Indices of edges connected to this vertex.
    pub edges: Vec<usize>,
}

/// An edge in the polytope.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Indices of the two vertices.
    pub vertices: [usize; 2],
    /// Indices of the two adjacent faces (if any).
    pub faces: [Option<usize>; 2],
    pub dist: f32,
    pub witness: Vec3,
}

/// A triangular face in the polytope.
#[derive(Clone, Debug)]
pub struct Face {
    /// Indices of the three surrounding edges.
    pub edges: [usize; 3],
    pub dist: f32,
    pub witness: Vec3,
}

/// The polytope used by EPA to expand and find penetration.
///
/// Uses index-based topology instead of C's intrusive linked lists.
#[derive(Clone, Debug)]
pub struct Polytope {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,
    /// Cached nearest element to origin.
    pub nearest: Option<ElementRef>,
    pub nearest_dist: f32,
    pub nearest_type: u8, // 1=vertex, 2=edge, 3=face (lower = higher priority)
}

impl Polytope {
    fn edge_other_vertex(&self, edge_idx: usize, vertex_idx: usize) -> Option<usize> {
        let [v0, v1] = self.edges.get(edge_idx)?.vertices;
        if v0 == vertex_idx {
            Some(v1)
        } else if v1 == vertex_idx {
            Some(v0)
        } else {
            None
        }
    }

    fn triangular_face_topology(&self, edges: [usize; 3]) -> Option<([usize; 3], [usize; 3])> {
        const PERMUTATIONS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];

        if edges[0] == edges[1] || edges[1] == edges[2] || edges[0] == edges[2] {
            return None;
        }

        for permutation in PERMUTATIONS {
            let ordered_edges = [
                edges[permutation[0]],
                edges[permutation[1]],
                edges[permutation[2]],
            ];
            let [a, b] = self.edges.get(ordered_edges[0])?.vertices;

            for (start, middle) in [(a, b), (b, a)] {
                let Some(end) = self.edge_other_vertex(ordered_edges[1], middle) else {
                    continue;
                };
                if end == start || end == middle {
                    continue;
                }
                if self.edge_other_vertex(ordered_edges[2], end) == Some(start) {
                    return Some((ordered_edges, [start, middle, end]));
                }
            }
        }

        None
    }

    /// Create an empty polytope.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            nearest: None,
            nearest_dist: f32::MAX,
            nearest_type: 3,
        }
    }

    /// Add a vertex (support point) to the polytope. Returns its index.
    pub fn add_vertex(&mut self, support: SupportPoint) -> usize {
        let dist = support.v.length_squared();
        let witness = support.v;
        let idx = self.vertices.len();
        self.vertices.push(Vertex {
            support,
            dist,
            witness,
            edges: Vec::new(),
        });
        self.update_nearest(ElementRef::Vertex(idx), dist);
        idx
    }

    /// Add an edge between two vertices. Returns its index.
    pub fn add_edge(&mut self, v1: usize, v2: usize) -> usize {
        let a = self.vertices[v1].support.v;
        let b = self.vertices[v2].support.v;
        let (dist, witness) = Vec3::point_segment_dist2(Vec3::ZERO, a, b);
        let idx = self.edges.len();

        self.vertices[v1].edges.push(idx);
        self.vertices[v2].edges.push(idx);

        self.edges.push(Edge {
            vertices: [v1, v2],
            faces: [None, None],
            dist,
            witness: witness.unwrap_or(Vec3::ZERO),
        });

        self.update_nearest(ElementRef::Edge(idx), dist);
        idx
    }

    /// Add a triangular face from three edges. Returns its index.
    pub fn add_face(&mut self, e1: usize, e2: usize, e3: usize) -> usize {
        let idx = self.faces.len();
        let (edges, vertices) = self
            .triangular_face_topology([e1, e2, e3])
            .expect("face edges should form a closed triangle");

        let [a, b, c] = vertices;
        let va = self.vertices[a].support.v;
        let vb = self.vertices[b].support.v;
        let vc = self.vertices[c].support.v;
        let (dist, witness) = Vec3::point_tri_dist2(Vec3::ZERO, va, vb, vc);

        // Register face in edges
        for &ei in &edges {
            if self.edges[ei].faces[0].is_none() {
                self.edges[ei].faces[0] = Some(idx);
            } else {
                self.edges[ei].faces[1] = Some(idx);
            }
        }

        self.faces.push(Face {
            edges,
            dist,
            witness: witness.unwrap_or(Vec3::ZERO),
        });

        self.update_nearest(ElementRef::Face(idx), dist);
        idx
    }

    /// Remove a face by index.
    pub fn remove_face(&mut self, face_idx: usize) {
        let face = &self.faces[face_idx];

        // Unlink from edges
        for &ei in &face.edges {
            if let Some(ref mut edge) = self.edges.get_mut(ei) {
                if edge.faces[0] == Some(face_idx) {
                    edge.faces[0] = edge.faces[1];
                }
                edge.faces[1] = None;
            }
        }

        // Invalidate the face (mark as removed)
        // We use Option<Face> internally? No — we keep Vec<Face> and just
        // set dist to f32::MAX and clear edges. But this is messy.
        // Better approach: use a swap-remove or mark invalid.
        // For simplicity, just set dist to MAX so it's never "nearest".
        self.faces[face_idx].dist = f32::MAX;
        self.faces[face_idx].edges = [0; 3]; // invalidate

        if self.nearest == Some(ElementRef::Face(face_idx)) {
            self.nearest = None;
        }
    }

    /// Remove an edge by index (only if it has no faces).
    pub fn remove_edge(&mut self, edge_idx: usize) {
        let edge = &self.edges[edge_idx];
        if edge.faces[0].is_some() {
            return; // still has faces
        }

        // Unlink from vertices
        for &vi in &edge.vertices {
            if let Some(ref mut v) = self.vertices.get_mut(vi) {
                v.edges.retain(|&e| e != edge_idx);
            }
        }

        self.edges[edge_idx].dist = f32::MAX; // invalidate

        if self.nearest == Some(ElementRef::Edge(edge_idx)) {
            self.nearest = None;
        }
    }

    /// Find the element nearest to the origin.
    pub fn find_nearest(&mut self) -> Option<ElementRef> {
        if self.nearest.is_none() {
            self.recompute_nearest();
        }
        self.nearest
    }

    fn recompute_nearest(&mut self) {
        self.nearest_dist = f32::MAX;
        self.nearest_type = 3;
        self.nearest = None;

        for index in 0..self.vertices.len() {
            let dist = self.vertices[index].dist;
            if dist < f32::MAX / 2.0 {
                self.update_nearest(ElementRef::Vertex(index), dist);
            }
        }

        for index in 0..self.edges.len() {
            let dist = self.edges[index].dist;
            if dist < f32::MAX / 2.0 {
                self.update_nearest(ElementRef::Edge(index), dist);
            }
        }

        for index in 0..self.faces.len() {
            let dist = self.faces[index].dist;
            if dist < f32::MAX / 2.0 {
                self.update_nearest(ElementRef::Face(index), dist);
            }
        }
    }

    fn update_nearest(&mut self, el: ElementRef, dist: f32) {
        let el_type = match el {
            ElementRef::Vertex(_) => 1,
            ElementRef::Edge(_) => 2,
            ElementRef::Face(_) => 3,
        };

        let better = if Vec3::approx_eq(dist, self.nearest_dist) {
            el_type < self.nearest_type
        } else {
            dist < self.nearest_dist
        };

        if better {
            self.nearest = Some(el);
            self.nearest_dist = dist;
            self.nearest_type = el_type;
        }
    }

    /// Get the three unique vertex indices referenced by a face's edges.
    pub fn face_vertices_from_edges(
        &self,
        e1: usize,
        e2: usize,
        e3: usize,
    ) -> (usize, usize, usize) {
        let (_, [a, b, c]) = self
            .triangular_face_topology([e1, e2, e3])
            .expect("face edges should form a closed triangle");
        (a, b, c)
    }

    /// Get the three vertex positions of a face.
    pub fn face_positions(&self, face_idx: usize) -> (Vec3, Vec3, Vec3) {
        let face = &self.faces[face_idx];
        let (a, b, c) = self.face_vertices_from_edges(face.edges[0], face.edges[1], face.edges[2]);
        (
            self.vertices[a].support.v,
            self.vertices[b].support.v,
            self.vertices[c].support.v,
        )
    }

    /// Get the element type for an ElementRef.
    pub fn element_type(&self, el: ElementRef) -> u8 {
        match el {
            ElementRef::Vertex(_) => 1,
            ElementRef::Edge(_) => 2,
            ElementRef::Face(_) => 3,
        }
    }
}

impl Default for Polytope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support_point(v: Vec3) -> SupportPoint {
        SupportPoint {
            v,
            v1: Vec3::ZERO,
            v2: Vec3::ZERO,
        }
    }

    fn edge_matches_vertices(polytope: &Polytope, edge_idx: usize, a: usize, b: usize) -> bool {
        let [v0, v1] = polytope.edges[edge_idx].vertices;
        (v0 == a && v1 == b) || (v0 == b && v1 == a)
    }

    #[test]
    fn test_face_vertices_follow_face_edge_cycle() {
        let mut polytope = Polytope::new();

        let v0 = polytope.add_vertex(support_point(Vec3::new(-1.0, 0.0, 0.0)));
        let v1 = polytope.add_vertex(support_point(Vec3::new(1.0, 0.0, 0.0)));
        let v2 = polytope.add_vertex(support_point(Vec3::new(0.0, 1.0, 0.0)));

        let base = polytope.add_edge(v0, v1);
        let spoke_a = polytope.add_edge(v2, v0);
        let spoke_b = polytope.add_edge(v2, v1);
        let face_idx = polytope.add_face(spoke_a, spoke_b, base);

        let face = &polytope.faces[face_idx];
        let (a, b, c) =
            polytope.face_vertices_from_edges(face.edges[0], face.edges[1], face.edges[2]);

        assert!(edge_matches_vertices(&polytope, face.edges[0], a, b));
        assert!(edge_matches_vertices(&polytope, face.edges[1], b, c));
        assert!(edge_matches_vertices(&polytope, face.edges[2], c, a));
    }
}
