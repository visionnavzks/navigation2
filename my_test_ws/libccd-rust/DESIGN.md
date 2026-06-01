# libccd-rust: Design Document

Port of [libccd](https://github.com/danfis/libccd) — a collision detection library for convex shapes — to idiomatic Rust.

## Core Design Principles

1. **Don't transliterate — redesign for Rust**: traits, Result/Option, ownership, no unsafe, no raw pointers
2. **Use `glam::Vec3A`** (f32 SIMD) as the math backbone instead of raw arrays
3. **Trait-based polymorphism** instead of `void*` + function pointers
4. **Index-based data structures** instead of intrusive linked lists

## Module Layout

```
src/
├── lib.rs           # Re-exports, crate-level docs
├── vec3.rs          # Vec3 newtype wrapping glam::Vec3A
├── quat.rs          # Quat newtype wrapping glam::Quat
├── support.rs       # SupportPoint (Minkowski difference support)
├── simplex.rs       # Simplex (up to 4 support points + do_simplex logic)
├── polytope.rs      # Polytope with index-based vertex/edge/face topology
├── gjk.rs           # GJK algorithm
├── epa.rs           # EPA algorithm
├── mpr.rs           # MPR algorithm
├── shapes/
│   ├── mod.rs       # Shape trait + re-exports
│   ├── box.rs       # BoxShape
│   ├── sphere.rs    # SphereShape
│   ├── cylinder.rs  # CylinderShape
│   └── convex_hull.rs  # ConvexHull
└── ccd.rs           # Ccd struct (main API), CcdConfig, Penetration
```

## Key Type Mappings

| C Concept | Rust Equivalent |
|-----------|----------------|
| `ccd_real_t` (double) | `f32` (with glam::Vec3A) |
| `ccd_vec3_t` | `Vec3` newtype wrapping `glam::Vec3A` |
| `ccd_quat_t` | `Quat` newtype wrapping `glam::Quat` |
| `void*` + function pointers | `&dyn Shape` trait object |
| `ccd_t` config struct | `Ccd::builder().max_iterations(100).build()` |
| `int` error codes (0/-1/-2) | `bool`, `Option<T>`, or `Result<T, CcdError>` |
| Intrusive linked list (polytope) | `Vec<Vertex/Edge/Face>` with `usize` indices |
| `CCD_INIT` macro | `Ccd::new()` (Default impl) |
| `#define CCD_EPS` | `const EPSILON: f32 = 1e-6;` |
| Global `ccd_points_on_sphere[]` | `LazyLock<Vec<Vec3>>` |
| `__ccdSupport()` | Method on `Ccd` |

## Shape Trait

```rust
pub trait Shape {
    /// Returns the farthest point on the shape in the given direction.
    fn support(&self, dir: Vec3) -> Vec3;

    /// Returns the geometric center of the shape.
    fn center(&self) -> Vec3;
}
```

## Public API

```rust
let ccd = Ccd::builder()
    .max_iterations(100)
    .epa_tolerance(1e-4)
    .mpr_tolerance(1e-4)
    .build();

// Intersection test (GJK)
let intersects = ccd.gjk_intersect(&box1, &box2);

// Penetration info (GJK + EPA)
if let Some(penetration) = ccd.gjk_penetration(&box1, &box2) {
    println!("depth={}, dir={:?}, pos={:?}", penetration.depth, penetration.dir, penetration.pos);
}

// MPR intersection
let intersects = ccd.mpr_intersect(&box1, &box2);

// MPR penetration
if let Some(penetration) = ccd.mpr_penetration(&box1, &box2) {
    // ...
}
```

## Penetration Result

```rust
pub struct Penetration {
    pub depth: f32,
    pub dir: Vec3,   // normalized direction
    pub pos: Vec3,   // contact position
}
```

## Polytope Design (EPA)

Replace intrusive linked lists with arena-style index-based access:

```rust
pub struct Polytope {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
    nearest: Option<ElementRef>,  // index + type
    nearest_dist: f32,
}

pub struct Vertex {
    pub support: SupportPoint,
    pub edges: Vec<usize>,  // indices into edges
}

pub struct Edge {
    pub vertices: [usize; 2],
    pub faces: [Option<usize>; 2],
}

pub struct Face {
    pub edges: [usize; 3],
    pub dist: f32,
    pub witness: Vec3,
}

pub enum ElementRef {
    Vertex(usize),
    Edge(usize),
    Face(usize),
}
```
