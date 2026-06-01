# Hybrid A* Bug Report

Generated: 2026-05-31 (updated 2026-06-01)

---

## Bug 1 (Critical): Distance heuristic fallback path angle conversion wrong

**File**: `src/distance_heuristic.cpp:107-108`

`goal_coords.theta` and `node_coords.theta` are already angular bin indices (integers), but they are multiplied by `num_angle_quantization`, producing `bin_index * num_bins` — which is **not radians**. OMPL state space expects radians.

Compare with correct usage in `analytic_expansion.cpp:153`:
```cpp
from[2] = _ctx->motion_table.getAngleFromBin(node->pose.theta); // bin_idx * 2PI / num_bins
```

**Fix**: Replace with `getAngleFromBin()` or equivalent `theta * 2*PI / num_angle_quantization`.

---

## Bug 2 (Critical): Lookup table boundary check off-by-one

**File**: `src/distance_heuristic.cpp:87,95`

The precompute loop generates entries for `abs(x) <= floored_size` and `y <= floored_size`, but the lookup uses strict `<`, causing boundary entries to never be used. Nodes at the boundary fall through to the OMPL fallback or return 0.

**Fix**: Change `<` to `<=` in the bounds check.

---

## Bug 3 (Moderate): Angle wrapping off-by-one

**File**: `src/distance_heuristic.cpp:74`

When `dtheta_bin == num_angle_quantization`, it represents a zero heading difference (full rotation). The condition `>` misses this case, leaving the bin index out of range [0, num_angle_quantization-1].

**Fix**: Change `>` to `>=`.

---

## Bug 4 (Moderate): Uninitialized `reversing_segment`

**File**: `src/smoother.cpp:86`

```cpp
bool success = true, reversing_segment;
```

Only `success` is initialized. `reversing_segment` is passed by reference to `smoothImpl` and used in boundary condition enforcement, causing random behavior.

**Fix**: Initialize to `false`.

---

## Bug 5 (Minor): `floor()` on unsigned int

**File**: `src/obstacle_heuristic.cpp:44`

Parameters `goal_x` and `goal_y` are `unsigned int`, making `floor()` a no-op. Indicates the parameters should likely be `float`.

---

## Bug 6 (Minor): Negative double to unsigned int cast

**File**: `include/my/hybrid_astar/costmap_2d.hpp:51-52`

Casting a negative `double` to `unsigned int` is implementation-defined before C++20. Should check for negative before casting.

---

## Bug 7 (Minor): `static const` in template function

**File**: `src/analytic_expansion.cpp:166`

`static const float sqrt_2` in a template function creates per-instantiation storage. Should use `constexpr`.

---

## Bug 8 (Moderate): `abs()` on double may truncate to int

**File**: `src/smoother.cpp:175`

```cpp
change += abs(y_i - y_i_org);
```

`y_i` and `y_i_org` are `double`. Unqualified `abs()` may resolve to C's `abs(int)` depending on the platform, silently truncating the floating-point difference. Should use `std::fabs()` or `std::abs()`.

---

## Bug 9 (Minor): Off-by-one in boundary expansion index validation

**File**: `src/smoother.cpp:362,401`

```cpp
if (best_expansion_idx > boundary_expansions.size()) {
```

Should be `>=` since valid indices are `[0, size()-1]`. Currently latent because the sentinel value (`1e9`) is always much larger than `size()`, but would cause an out-of-bounds access if the sentinel were ever changed to `size()`.

---

## Bug 10 (Minor): Namespace-scope `const` in header creates duplicate symbols

**File**: `include/my/hybrid_astar/constants.hpp:73-77`

```cpp
const float UNKNOWN_COST = 255.0;
const float OCCUPIED_COST = 254.0;
// ...
```

`const` at namespace scope has internal linkage in C++, so each translation unit gets its own copy. Should use `inline constexpr` (C++17).

---

## Bug 11 (Moderate): Obstacle heuristic neighborhood expansion has row-wrapping risk

**File**: `src/obstacle_heuristic.cpp:116-119`

```cpp
new_idx = static_cast<unsigned int>(static_cast<int>(idx) + neighborhood[i]);
```

The `±1` offsets in the 8-connected neighborhood don't check row boundaries. A cell at `mx=0` with offset `-1` wraps to the previous row's last cell. Currently mitigated by the 3-cell margin check at lines 151-156, but the protection is incidental — if the margin were ever reduced, row-wrapping would cause incorrect heuristic values.

---

## Bug 12 (Moderate): Collision checker `mapToWorld` adds spurious half-cell offset

**File**: `src/collision_checker.cpp:88-89`

```cpp
costmap_->mapToWorld(static_cast<double>(x), static_cast<double>(y), wx, wy);
```

`mapToWorld` computes `origin + (mx + 0.5) * resolution`, but `x`/`y` are already floating-point map coordinates (not integer cell indices). This offsets the footprint check by half a cell. Meanwhile, `getWorldCoords()` in `utils.hpp:21` does NOT add 0.5, creating an inconsistency between where the planner thinks the robot is and where the collision checker checks.

---

## Bug 13 (Moderate): `setGoal` ignores `coarse_search_resolution` parameter

**File**: `src/a_star.cpp:138`

```cpp
_coarse_search_resolution = 1;
```

`setGoal` receives a `coarse_search_resolution` parameter but discards it, always hardcoding `_coarse_search_resolution` to 1. This makes `SmacPlannerHybridConfig::coarse_search_resolution` completely ineffective — the coarse search resolution in `ALL_DIRECTION` mode cannot be adjusted.

**Fix**: `_coarse_search_resolution = coarse_search_resolution;`

---

## Bug 14 (Moderate): Distance heuristic mirrored `theta_pos` out-of-bounds

**File**: `src/distance_heuristic.cpp:90`

```cpp
theta_pos = motion_table.num_angle_quantization - node_coords_relative.theta;
```

When `node_coords_relative.y < 0` and `node_coords_relative.theta == 0`, `theta_pos = num_angle_quantization`, which is out of the valid lookup table index range `[0, num_angle_quantization-1]`, causing an out-of-bounds access.

**Fix**: Apply modulo: `theta_pos %= motion_table.num_angle_quantization;`

---

## Bug 15 (Moderate): Direction change detection doesn't handle angle wrapping

**File**: `src/smoother.cpp:32`

```cpp
if (std::abs(angle - prev_angle) > M_PI_2) {
```

`atan2` returns values in `[-π, π]`. When `angle ≈ -π + ε` and `prev_angle ≈ π - ε`, the actual angular difference is tiny, but `abs(angle - prev_angle) ≈ 2π`, falsely triggering a direction change detection and incorrectly splitting the path into segments.

**Fix**: Normalize the angle difference:
```cpp
double diff = std::fmod(std::abs(angle - prev_angle), 2.0 * M_PI);
if (diff > M_PI) diff = 2.0 * M_PI - diff;
if (diff > M_PI_2) { ... }
```

---

## Bug 16 (Moderate): `asin` domain error for small turning radius

**File**: `src/node_hybrid.cpp:50` and `src/node_hybrid.cpp:148`

```cpp
float angle = 2.0 * asin(sqrt(2.0) / (2 * min_turning_radius));
```

When `min_turning_radius < √2/2 ≈ 0.707`, the argument `sqrt(2.0) / (2 * min_turning_radius)` exceeds 1.0, causing `asin` to return NaN. This propagates NaN through all motion primitive projections and travel costs, completely breaking the search.

**Fix**: Clamp the argument: `asin(std::min(1.0, sqrt(2.0) / (2 * min_turning_radius)))` or validate `min_turning_radius` lower bound.

---

## Bug 17 (Minor): `getClosestAngularBin` undefined behavior on negative theta

**File**: `src/node_hybrid.cpp:258-261`

```cpp
auto bin = static_cast<unsigned int>(round(static_cast<float>(theta) / bin_size));
```

When `theta` is negative, `round(theta / bin_size)` is negative, and `static_cast<unsigned int>` on a negative floating-point value is undefined behavior before C++20.

**Fix**: Normalize theta to `[0, 2π)` before computing the bin index.

---

## Bug 18 (Minor): `setFootprint` not thread-safe with `createPlan`

**File**: `src/smac_planner_hybrid.cpp:113`

`createPlan` acquires `_mutex`, but `setFootprint` does not. If both are called concurrently (e.g., dynamic footprint updates), there is a data race on `_collision_checker` and `_config`.

**Fix**: Add `std::lock_guard<std::mutex> lock(_mutex);` at the top of `setFootprint`.

---

## Bug 19 (Critical): Buffer overflow when obstacle heuristic lookup table shrinks

**File**: `src/obstacle_heuristic.cpp:31-34`

```cpp
unsigned int obstacle_size = obstacle_heuristic_lookup_table_.size();
obstacle_heuristic_lookup_table_.resize(size, 0.0f);
std::fill_n(
    obstacle_heuristic_lookup_table_.begin(), obstacle_size, 0.0f);
```

When `size < obstacle_heuristic_lookup_table_.size()`, `resize` shrinks the vector to `size` elements, but `fill_n` writes `obstacle_size` (the OLD, larger size) elements from `begin()`. This writes `obstacle_size - size` elements past the end of the vector — a heap buffer overflow.

**Trigger**: The `downsample_obstacle_heuristic` parameter changes between `resetObstacleHeuristic` calls (e.g., planner reconfigured), or the costmap dimensions change.

**Fix**: Swap the two lines so `fill_n` runs before `resize`, or use `std::fill(begin, begin + size, 0.0f)`:
```cpp
obstacle_heuristic_lookup_table_.resize(size, 0.0f);
std::fill(obstacle_heuristic_lookup_table_.begin(),
          obstacle_heuristic_lookup_table_.end(), 0.0f);
```

---

## Bug 20 (Critical): Out-of-bounds array access in collision checker center cost check

**File**: `src/collision_checker.cpp:71-72`

```cpp
center_cost_ = static_cast<float>(costmap_->getCost(
    static_cast<unsigned int>(x + 0.5f), static_cast<unsigned int>(y + 0.5f)));
```

The `outsideRange` check at lines 65-66 allows `x` up to `size_x - epsilon` (e.g., 9.999 in a 10-cell map). Then `x + 0.5f` rounds up to `size_x` (10.499 → truncates to 10), and `getCost(10, ...)` accesses `cost_map_[my * size_x_ + 10]` — one element past the row boundary.

**Trigger**: Any node whose continuous x coordinate is in `[size_x - 0.5, size_x)`.

**Fix**: Clamp the rounded value before indexing:
```cpp
unsigned int cx = std::min(static_cast<unsigned int>(x + 0.5f), costmap_->getSizeInCellsX() - 1);
unsigned int cy = std::min(static_cast<unsigned int>(y + 0.5f), costmap_->getSizeInCellsY() - 1);
center_cost_ = static_cast<float>(costmap_->getCost(cx, cy));
```

---

## Bug 21 (Moderate): Unchecked `worldToMap` return value in smoother boundary expansion

**File**: `src/smoother.cpp:287-288`

```cpp
unsigned int mx, my;
costmap->worldToMap(x, y, mx, my);
if (static_cast<float>(costmap->getCost(mx, my)) >= INSCRIBED_COST) {
```

`worldToMap` returns `false` when the point is outside the map, and in the negative case, `mx`/`my` are **never assigned** (they remain uninitialized). Even when `map_x >= 0` but `mx >= size_x_`, the returned `mx` is out of range. In both cases, `getCost(mx, my)` accesses invalid memory.

**Trigger**: An interpolated Dubins/Reeds-Shepp boundary expansion point falls outside the costmap bounds.

**Fix**: Check the return value:
```cpp
if (!costmap->worldToMap(x, y, mx, my)) {
    expansion.in_collision = true;
    continue;
}
```

---

## Bug 22 (Moderate): Recursive smooth refinement ignores elapsed time

**File**: `src/smoother.cpp:202-205`

```cpp
if (do_refinement_ && refinement_ctr_ < refinement_num_) {
    refinement_ctr_++;
    smoothImpl(new_path, reversing_segment, costmap, max_time);
}
```

`max_time` is the time budget passed into `smoothImpl`, which was the remaining time from the outer `smooth()` call. After the first `smoothImpl` iteration completes (potentially consuming significant time), the recursive call receives the **original** `max_time` rather than the remaining time. This allows refinement iterations to exceed the planner's time budget.

**Fix**: Pass the remaining time:
```cpp
steady_clock::time_point now = steady_clock::now();
double remaining = max_time - duration_cast<duration<double>>(now - a).count();
smoothImpl(new_path, reversing_segment, costmap, std::max(0.0, remaining));
```

---

## Bug 23 (Moderate): Negative float to unsigned int conversion for obstacle heuristic start coordinates

**File**: `src/obstacle_heuristic.cpp:69-76`

```cpp
unsigned int start_y, start_x;
if (downsample_obstacle_heuristic) {
    start_y = floor(node_coords.y / 2.0f);
    start_x = floor(node_coords.x / 2.0f);
} else {
    start_y = floor(node_coords.y);
    start_x = floor(node_coords.x);
}
```

`node_coords.x` and `node_coords.y` are floats that can be negative (e.g., for nodes near the map origin or due to motion primitive projections slightly outside the map). `floor()` of a negative float returns a negative double, and assigning that to `unsigned int` is undefined behavior (pre-C++20) or wraps to a huge value. This produces a bogus `start_index` at line 78, leading to an out-of-bounds vector access at line 79.

**Fix**: Clamp to zero before conversion:
```cpp
start_x = static_cast<unsigned int>(std::max(0.0f, std::floor(node_coords.x)));
start_y = static_cast<unsigned int>(std::max(0.0f, std::floor(node_coords.y)));
```

---

## Bug 24 (Moderate): Unsigned underflow in obstacle heuristic boundary check for small costmaps

**File**: `src/obstacle_heuristic.cpp:155-159`

```cpp
if (mx >= size_x - 3 || mx <= 3) {
    continue;
}
if (my >= size_y - 3 || my <= 3) {
    continue;
}
```

When `size_x` or `size_y` is less than 3, `size_x - 3` (unsigned subtraction) wraps to a very large value (~4 billion). The condition `mx >= (huge value)` is always false, but `mx <= 3` is always true for small maps. This means **every cell** is skipped by the boundary check, making the obstacle heuristic unable to expand any neighbors — the Dijkstra search never progresses, always returning 0 cost.

**Fix**: Use signed comparison:
```cpp
if (mx <= 3u || mx >= (size_x <= 3u ? size_x : size_x - 3u)) {
    continue;
}
```

---

## Bug 25 (Moderate): `getNeighbors` casts negative floats to unsigned int for index computation

**File**: `src/node_hybrid.cpp:428-432`

```cpp
index = NodeHybrid::getIndex(
    static_cast<unsigned int>(motion_projections[i]._x),
    static_cast<unsigned int>(motion_projections[i]._y),
    static_cast<unsigned int>(motion_projections[i]._theta),
    _ctx->motion_table.size_x, _ctx->motion_table.num_angle_quantization);
```

`motion_projections[i]._x` and `._y` are absolute world coordinates (floats). If a motion primitive projects the robot outside the map (negative coordinates), `static_cast<unsigned int>` on a negative float is UB (pre-C++20) or wraps to a large value. The `NeighborGetter` checks `index >= max_index`, but the wrapped value could coincidentally be < max_index, producing a wrong index.

**Fix**: Cast to int first and check bounds before computing the index:
```cpp
int px = static_cast<int>(motion_projections[i]._x);
int py = static_cast<int>(motion_projections[i]._y);
if (px < 0 || py < 0 ||
    static_cast<unsigned int>(px) >= _ctx->motion_table.size_x ||
    static_cast<unsigned int>(py) >= _ctx->motion_table.size_y) {
    continue;
}
```

---

## Bug 26 (Minor): `node_basic.hpp` leaves `motion_index` and `turn_dir` uninitialized

**File**: `include/my/hybrid_astar/node_basic.hpp:16-20`

```cpp
explicit NodeBasic(const uint64_t new_index)
: graph_node_ptr(nullptr),
    index(new_index)
{
}
```

`motion_index` (unsigned int) and `turn_dir` (TurnDirection enum) are never initialized in the constructor. They are only set by `populateSearchNode()`. While the current code flow always calls `populateSearchNode()` before `processSearchNode()`, this is fragile — any future use of NodeBasic that reads these fields before population would invoke UB.

**Fix**: Initialize in the constructor:
```cpp
: graph_node_ptr(nullptr),
    index(new_index),
    motion_index(0),
    turn_dir(TurnDirection::UNKNOWN)
{}
```

---

## Bug 27 (Minor): `smoother.cpp:138` — `size_t` to `unsigned int` narrowing via const reference

**File**: `src/smoother.cpp:138`

```cpp
const unsigned int & path_size = path.size();
```

`path.size()` returns `size_t` (64-bit on most platforms). Binding to `const unsigned int &` creates a temporary `unsigned int`, narrowing the value. While paths won't realistically exceed 2^32 elements, this is a type-safety issue and could mask truncation bugs.

**Fix**: Use the correct type:
```cpp
const size_t path_size = path.size();
```

---

## Bug 28 (Critical): Distance heuristic lookup table undersized for precompute loop

**File**: `src/distance_heuristic.cpp:31-43`

The `resize` call allocates `size_lookup_ * ceil(size_lookup_/2) * dim_3_size` entries, but the precompute loop iterates over `floor(size_lookup_/2) - ceil(-size_lookup_/2) + 1` x-positions (e.g., 21 for `size_lookup_=20`) and `floor(size_lookup_/2) + 1` y-positions (e.g., 11). For `size_lookup_=20`, the table holds 20·10·72=14400 entries but the loop writes 21·11·72=16632 — a heap buffer overflow of 2232 elements on every planner initialization.

**Trigger**: Any call that triggers `precomputeDistanceHeuristic` (i.e. any `setGoal`).

**Fix**: Resize with the same formula the precompute loop uses:
```cpp
dist_heuristic_lookup_table_.resize(
  (static_cast<int>(floor(size_lookup_/2)) -
   static_cast<int>(ceil(-size_lookup_/2)) + 1) *
  (static_cast<int>(floor(size_lookup_/2)) + 1) * dim_3_size_int);
```

---

## Summary

| Severity | Count | Bugs |
|----------|-------|------|
| Critical | 5 | #1, #2, #19, #20, #28 |
| Moderate | 13 | #3, #4, #8, #11, #12, #13, #14, #15, #16, #21, #22, #23, #24, #25 |
| Minor | 9 | #5, #6, #7, #9, #10, #17, #18, #26, #27 |
| **Total** | **27 (28 incl. new)** | |

> Bug #6 was re-evaluated: the actual code (`costmap_2d.hpp:88-93`) already
> guards against negative `map_x`/`map_y` before the unsigned cast, so it does
> not exhibit the described behavior. Marked closed.
> Bug #28 was discovered during re-verification; see BUGFIX_LOG.md.
