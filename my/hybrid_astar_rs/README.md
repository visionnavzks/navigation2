# hybrid_astar (Rust)

A standalone Hybrid A\* path planner supporting Ackermann vehicle kinematics (Dubin / Reeds-Shepp models), ported from the C++ `hybrid_astar` crate.

## Building

```bash
cd my/hybrid_astar_rs
cargo build
```

## Module mapping (C++ → Rust)

| C++ module | Rust module | Description |
|---|---|---|
| `constants.hpp` | `constants.rs` | Cost thresholds, `MotionModel`, `GoalHeadingMode` |
| `types.hpp` | `types.rs` | `Pose`, `Path`, `Coordinates`, `SearchInfo`, `SmootherParams` |
| `costmap_2d.hpp` | `costmap.rs` | `Costmap2D` (2-D cost grid) |
| `collision_checker.hpp/cpp` | `collision.rs` | `GridCollisionChecker` (legacy + ESDF capsule) |
| `esdf_holder.hpp` | `esdf.rs` | `EsdfHolder` (cached distance field) |
| `steering_state_space.hpp` | `steering.rs` | Dubin / Reeds-Shepp state space |
| `node_hybrid.hpp/cpp` | `node.rs` | `NodeHybrid`, `HybridMotionTable`, `NodeContext` |
| `node_basic.hpp/cpp` | (inlined into `a_star.rs`) | Priority-queue element wrapper |
| `a_star.hpp/cpp` | `a_star.rs` | `AStarAlgorithm` — main search loop |
| `analytic_expansion.hpp/cpp` | `analytic.rs` | Analytic expansion (steering-function direct connections) |
| `obstacle_heuristic.hpp/cpp` | `obstacle_heuristic.rs` | D\*-style 2-D obstacle distance heuristic |
| `distance_heuristic.hpp/cpp` | `distance_heuristic.rs` | Pre-computed kinematic distance lookup table |
| `goal_manager.hpp` | `goal_manager.rs` | Multi-goal management with heading modes |
| `smoother.hpp/cpp` | `smoother.rs` | Path smoother with boundary-condition enforcement |
| `costmap_downsampler.hpp/cpp` | `costmap_downsampler.rs` | Cost-map down-sampling utility |
| `smac_planner_hybrid.hpp/cpp` | `planner.rs` | `SmacPlannerHybrid` — high-level planner façade |
| `utils.hpp` | `utils.rs` | Coordinate conversion, circumscribed cost |

## Key design differences from the C++ version

- **No raw pointers** — the graph is a `HashMap<u64, NodeHybrid>` and parent links are stored as `Option<u64>` indices rather than raw pointers.
- **No robin_hood hash map** — uses `hashbrown` (Rust port of the same algorithm).
- **Interior mutability** — the obstacle heuristic uses `RefCell` for its mutable Dijkstra state, allowing `&self` access from the heuristic cost computation.
- **No ESDF crate dependency** — the ESDF is computed internally via a simple 2-pass distance transform (Meijster/Felzenszwalb style).
- **Steering functions** — Dubin and Reeds-Shepp are implemented in pure Rust instead of depending on the `cpp-dubins-rs` C++ library.
