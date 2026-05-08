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

        // Get the three vertices of the triangle from the edges
        let (a, b, c) = self.face_vertices_from_edges(e1, e2, e3);
        let va = self.vertices[a].support.v;
        let vb = self.vertices[b].support.v;
        let vc = self.vertices[c].support.v;
        let (dist, witness) = Vec3::point_tri_dist2(Vec3::ZERO, va, vb, vc);

        // Register face in edges
        for &ei in &[e1, e2, e3] {
            if self.edges[ei].faces[0].is_none() {
                self.edges[ei].faces[0] = Some(idx);
            } else {
                self.edges[ei].faces[1] = Some(idx);
            }
        }

        self.faces.push(Face {
            edges: [e1, e2, e3],
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
    pub fn face_vertices_from_edges(&self, e1: usize, e2: usize, e3: usize) -> (usize, usize, usize) {
        let mut vertices = [usize::MAX; 3];
        let mut count = 0usize;

        for edge_idx in [e1, e2, e3] {
            for vertex_idx in self.edges[edge_idx].vertices {
                if vertices[..count].contains(&vertex_idx) {
                    continue;
                }
                debug_assert!(count < 3, "face edges should reference exactly three unique vertices");
                if count < 3 {
                    vertices[count] = vertex_idx;
                    count += 1;
                }
            }
        }

        debug_assert!(count == 3, "face edges should reference exactly three unique vertices");
        (vertices[0], vertices[1], vertices[2])
    }

    /// Get the three vertex positions of a face.
    pub fn face_positions(&self, face_idx: usize) -> (Vec3, Vec3, Vec3) {
        let face = &self.faces[face_idx];
        let (a, b, c) = self.face_vertices_from_edges(face.edges[0], face.edges[1], face.edges[2]);
        (self.vertices[a].support.v, self.vertices[b].support.v, self.vertices[c].support.v)
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
