# Hybrid A* Bug Report

Generated: 2026-05-31

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
