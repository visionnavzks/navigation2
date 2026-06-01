# SMAC Planner Bug Report

## 1. Memory Bugs

### BUG-1: Null Pointer Dereference in `NodeLattice::getTraversalCost`
- **Severity:** CRITICAL
- **File:** `src/node_lattice.cpp:314-319`
- **Problem:** `transition_prim` (child's motion primitive) is dereferenced before any null check. The null check on line 318 is for `prim` (the current node's primitive), not `transition_prim`.
```cpp
const MotionPrimitive * transition_prim = child->getMotionPrimitive();
const float prim_length =
  transition_prim->trajectory_length / ...;  // DEREFERENCE before check
if (prim == nullptr) {   // null check on WRONG pointer
  return prim_length;
}
```
- **Fix:** Check `transition_prim` for null before dereferencing.

### BUG-2: Out-of-Bounds via Float-Used-as-Index in `HybridMotionTable::getProjections`
- **Severity:** CRITICAL
- **File:** `src/node_hybrid.cpp:315-318`
- **Problem:** `node_heading` is a `float` (`node->pose.theta`) used directly as a vector index into `delta_xs[i]` and `delta_ys[i]`. Implicit float-to-size_t conversion yields out-of-bounds index if not exact integer.
```cpp
projection_list.emplace_back(
  delta_xs[i][node_heading] + node->pose.x,   // node_heading is float
  delta_ys[i][node_heading] + node->pose.y,
  ...);
```
- **Fix:** Round `node_heading` to nearest integer and validate bounds before indexing.

### BUG-3: Out-of-Bounds via Float-Used-as-Index in `LatticeMotionTable::getMotionPrimitives`
- **Severity:** CRITICAL
- **File:** `src/node_lattice.cpp:127`
- **Problem:** `node->pose.theta` (float) used directly as index into `motion_primitives`.
```cpp
MotionPrimitives & prims_at_heading = motion_primitives[node->pose.theta];
```
- **Fix:** Same as BUG-2.

### BUG-4: No Bounds Check in `LatticeMotionTable::getAngleFromBin`
- **Severity:** CRITICAL
- **File:** `src/node_lattice.cpp:184-187`
- **Problem:** No bounds checking on `bin_idx`. Returns non-const reference to internal state.
```cpp
float & LatticeMotionTable::getAngleFromBin(const unsigned int & bin_idx)
{
  return lattice_metadata.heading_angles[bin_idx];  // no bounds check
}
```
- **Fix:** Add bounds check before accessing `heading_angles`.

### BUG-5: Out-of-Bounds in `ObstacleHeuristic::getObstacleHeuristic`
- **Severity:** HIGH
- **File:** `src/obstacle_heuristic.cpp:105`
- **Problem:** `start_index` computed from node coordinates can exceed `obstacle_heuristic_lookup_table_.size()` if coordinates are slightly outside bounds due to float rounding.
- **Fix:** Add bounds check before array access.

### BUG-6: Uninitialized `NodeBasic` Members
- **Severity:** HIGH
- **File:** `include/my/smac_planner/node_basic.hpp:29-33`
- **Problem:** `prim_ptr`, `motion_index`, `backward`, `turn_dir` are never initialized in constructor. For `NodeBasic<Node2D>`, `processSearchNode()` is a no-op, so these remain uninitialized garbage.
- **Fix:** Add default member initializers.

### BUG-7: Uninitialized `HybridMotionTable` Members
- **Severity:** HIGH
- **File:** `include/my/smac_planner/node_hybrid.hpp:22-66`
- **Problem:** Empty default constructor. All members uninitialized until `initDubin()`/`initReedsShepp()` is called. Short-circuit comparisons in init functions read uninitialized values (UB).
- **Fix:** Add default member initializers or initialize in constructor.

### BUG-8: Uninitialized `LatticeMotionTable` Members
- **Severity:** HIGH
- **File:** `include/my/smac_planner/node_lattice.hpp:24-62`
- **Problem:** Same pattern as BUG-7.
- **Fix:** Add default member initializers or initialize in constructor.

### BUG-9: Smoother Boundary Expansion Out-of-Bounds Write
- **Severity:** HIGH
- **File:** `src/smoother.cpp:384-388`
- **Problem:** `best_expansion.pts` can have more elements than `path.size()`. Loop writes to `path[i]` without checking `i < path.size()`.
```cpp
for (unsigned int i = 0; i != best_expansion.pts.size(); i++) {
    path[i].x = best_expansion.pts[i].x;  // may exceed path.size()
```
- **Fix:** Clamp loop to `min(best_expansion.pts.size(), path.size())`.

### BUG-10: Smoother End Boundary Expansion Out-of-Bounds + Unsigned Underflow
- **Severity:** HIGH
- **File:** `src/smoother.cpp:423-428`
- **Problem:** `expansion_starting_idx + best_expansion.pts.size() > path.size()` possible. Also unsigned underflow if `path_end_idx >= path.size()`.
```cpp
expansion_starting_idx = path.size() - best_expansion.path_end_idx - 1;
for (unsigned int i = 0; i != best_expansion.pts.size(); i++) {
    path[expansion_starting_idx + i].x = ...;
```
- **Fix:** Add bounds check. Use signed arithmetic for subtraction.

### BUG-11: Negative Float Cast to `unsigned int` in Collision Checker
- **Severity:** HIGH
- **File:** `src/collision_checker.cpp:91`
- **Problem:** `angle_bin` is float. `static_cast<unsigned int>` of negative float is UB. No bounds check.
```cpp
const Footprint & oriented_footprint =
  oriented_footprints_[static_cast<unsigned int>(angle_bin)];
```
- **Fix:** Clamp angle_bin to valid range before casting.

### BUG-12: `LatticeMotionTable::initMotionModel` Does Not Clear State on Re-init
- **Severity:** MEDIUM
- **File:** `src/node_lattice.cpp:66-121`
- **Problem:** If `initMotionModel()` called with different filepath, old `motion_primitives` not cleared before new data appended via `push_back`.
- **Fix:** Clear vectors at start of initialization.

### BUG-13: Negative Double Cast to `unsigned int` in Analytic Expansion
- **Severity:** MEDIUM
- **File:** `src/analytic_expansion.cpp:236-241`
- **Problem:** `reals[0]`, `reals[1]` from OMPL interpolation can be negative at map edges. `static_cast<unsigned int>` of negative double is UB.
- **Fix:** Add bounds validation before cast.

---

## 2. Logic/Algorithm Bugs

### BUG-14: Analytic Expansion Stale Node Validity Cache
- **Severity:** HIGH
- **File:** `src/analytic_expansion.cpp:243-302`, `src/node_hybrid.cpp:371-384`
- **Problem:** Analytic expansion sets node pose to proposed coordinates and calls `isNodeValid()`, caching result in `_cell_cost`. After expansion, pose is restored (line 301) but `_cell_cost` is NOT reset to NaN. Subsequent A* search via `getNeighbors` may hit stale cache with wrong collision result.
- **Fix:** Reset `_cell_cost = NaN` after restoring pose in `getAnalyticPath`.

### BUG-15: Distance Heuristic Lookup Table theta=0 Mirroring Off-by-One
- **Severity:** MEDIUM
- **File:** `src/distance_heuristic.cpp:177-179`
- **Problem:** When `theta == 0` and `y < 0`: `theta_pos = N - 0 = N`, which is out of `[0, N-1]` range. Reads wrong bin.
```cpp
theta_pos = motion_table.num_angle_quantization - node_coords_relative.theta;
```
- **Fix:** Use modulo: `theta_pos = (N - theta) % N`.

### BUG-16: Smoother Ignores Reversing Segment Flag
- **Severity:** HIGH
- **File:** `src/smoother.cpp:56-70`
- **Problem:** `updateApproximatePathOrientations` always sets heading to `atan2(dy, dx)` (forward direction). `reversing_segment` parameter is explicitly discarded via `(void)reversing_segment`. For Reeds-Shepp paths, reverse segments should have heading opposite to travel direction.
```cpp
(void)reversing_segment;  // ignored!
path[i].theta = std::atan2(dy, dx);  // always forward
```
- **Fix:** When `reversing_segment` is true, set `path[i].theta = atan2(dy, dx) + PI`.

### BUG-17: Angle Wraparound Not Handled in Path Segment Splitting
- **Severity:** MEDIUM
- **File:** `src/smoother.cpp:46`
- **Problem:** `atan2` returns `[-PI, PI]`. Raw difference `angle - prev_angle` can be ~6.0 when crossing ±PI boundary, triggering false segment break.
```cpp
if (std::abs(angle - prev_angle) > M_PI_2) {
```
- **Fix:** Normalize difference: `diff = angle - prev_angle; if (diff > PI) diff -= 2*PI; if (diff < -PI) diff += 2*PI;`

### BUG-18: Smoother Recursive Refinement Exceeds Time Budget
- **Severity:** MEDIUM
- **File:** `src/smoother.cpp:214-216`
- **Problem:** Recursive `smoothImpl` call passes original `max_time` instead of remaining time. Each recursive call starts fresh timer against full budget. Total time can reach `(refinement_num + 1) * max_time`.
```cpp
smoothImpl(new_path, reversing_segment, costmap, max_time);  // should be remaining time
```
- **Fix:** Compute elapsed time and pass `max_time - elapsed`.

### BUG-19: `isZoneValid` Only Checks Angle Bin 0
- **Severity:** MEDIUM
- **File:** `include/my/smac_planner/goal_manager.hpp:116-133`
- **Problem:** Loop only varies `m.x` and `m.y`, `m.theta` stays 0.0. For NodeHybrid, different heading bins have different collision footprints. Valid goal from heading bin 5 may be rejected because only bin 0 is checked.
```cpp
current_node.setPose(m);  // m.theta always 0.0
```
- **Fix:** Also iterate over relevant heading bins, or use radius-based collision check.

### BUG-20: Start/Goal Orientation Normalization Uses `if` Instead of `while` for Upper Bound
- **Severity:** MEDIUM
- **File:** `src/smac_planner_hybrid.cpp:149-172`
- **Problem:** Lower bound uses `while` loop (handles multiple wraps), upper bound uses single `if` (handles at most one wrap). With angles > 4π, result may remain out of range.
```cpp
while (start_orientation_bin < 0.0) {           // while loop
    start_orientation_bin += _angle_quantizations;
}
if (start_orientation_bin >= _angle_quantizations) {  // single if
    start_orientation_bin -= _angle_quantizations;
}
```
- **Fix:** Use `while` for both bounds, or use `fmod` for proper modular arithmetic.

### BUG-21: `getHeuristicCost` Uses Grid-Snapped Coordinates Instead of Continuous
- **Severity:** LOW
- **File:** `src/a_star.cpp:506-507`
- **Problem:** Heuristic computed from `getCoords(node->getIndex())` (truncated integer coords) rather than `node->pose` (continuous coords). Inconsistency with g-cost which uses continuous coordinates.
- **Fix:** Use `node->pose` for heuristic computation.

### BUG-22: `reversing_segment` Uninitialized in Smoother
- **Severity:** MEDIUM
- **File:** `src/smoother.cpp:100`
- **Problem:** `reversing_segment` declared but never initialized. Passed by reference to `smoothImpl` which never sets it. Then passed to `enforceStartBoundaryConditions` / `enforceEndBoundaryConditions` where it controls Dubins direction. Undefined behavior.
```cpp
bool success = true, reversing_segment;  // uninitialized
```
- **Fix:** Initialize `reversing_segment = false` or derive it from path analysis.

---

## 3. Thread Safety Issues

### BUG-23: Costmap Accessed Without Locking
- **Severity:** MEDIUM
- **File:** `include/my/smac_planner/costmap_2d.hpp:83`
- **Problem:** `Costmap2D` has a `recursive_mutex` exposed via `getMutex()`, but the SMAC planner never acquires it. Costmap can be updated by sensor thread while planner reads it.
- **Fix:** Lock mutex during costmap reads in the planner.

---

## Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| CRITICAL | 4 | Null deref, float-as-index OOB (×3) |
| HIGH | 8 | Stale cache, OOB writes, uninitialized members, reversing heading |
| MEDIUM | 9 | Angle normalization, time budget, zone check, segment splitting |
| LOW | 2 | Heuristic precision, unsigned wrapping |
