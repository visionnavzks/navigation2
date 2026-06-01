# Hybrid A* Bug Fix Log

Generated: 2026-05-31 (updated 2026-06-01)

---

## Fixed

| Bug | Severity | File | Fix | Verified |
|-----|----------|------|-----|----------|
| #1 | Critical | `distance_heuristic.cpp:93,94` | 角度转换改用 `getAngleFromBin()` | ✅ 编译通过 |
| #2 | Critical | `distance_heuristic.cpp:72` | 查找表边界 `<` 改为 `<=` | ✅ 编译通过 |
| #3 | Moderate | `distance_heuristic.cpp:61` | 用 `wrapBinIndex()` 替代 `>` 检查 | ✅ 编译通过 |
| #4 | Moderate | `smoother.cpp:84` | `reversing_segment` 初始化为 `false` | ✅ 编译通过 |
| #5 | Minor | `obstacle_heuristic.hpp:25-27` + `obstacle_heuristic.cpp:7-12` | 参数改 `float`，统一调用方传 float | ✅ 编译通过 |
| #7 | Minor | `analytic_expansion.cpp:166` | `static const` 改为 `constexpr` | ✅ 编译通过 |
| #8 | Moderate | `smoother.cpp:173` | `abs()` 改为 `std::fabs()` | ✅ 编译通过 |
| #9 | Minor | `smoother.cpp:362,401` | `>` 改为 `>=` | ✅ 编译通过 |
| #10 | Minor | `constants.hpp:73-78` | `const` 改为 `inline constexpr` | ✅ 编译通过 |
| #11 | Moderate | `obstacle_heuristic.cpp:108-152` | 邻域扩展加行边界（my/mx 范围）检查 | ✅ 编译通过 |
| #12 | Moderate | `collision_checker.cpp:87` | `mapToWorld` 改用 `mapCellToWorld` | ✅ 编译通过 |
| #13 | Moderate | `a_star.cpp:138` | `setGoal` 赋值 `_coarse_search_resolution` | ✅ 编译通过 |
| #14 | Moderate | `distance_heuristic.cpp:76` | `theta_pos %= num_angle_quantization` | ✅ 编译通过 |
| #15 | Moderate | `smoother.cpp:31-32` | 加 `fmod` + 归一化角度差 | ✅ 编译通过 |
| #16 | Moderate | `node_hybrid.cpp:50` | `asin` 参数加 `std::min(1.0, ...)` clamp | ✅ 编译通过 |
| #17 | Minor | `node_hybrid.cpp:177` | `getClosestAngularBin` 加 `wrapAngle` 归一化 | ✅ 编译通过 |
| #18 | Minor | `smac_planner_hybrid.cpp:116` | `setFootprint` 加 `std::lock_guard` | ✅ 编译通过 |
| #19 | Critical | `obstacle_heuristic.cpp:30-37` | 修复表收缩时 `fill_n` 越界（用 `std::fill`+`end()`） | ✅ 编译通过 |
| #20 | Critical | `collision_checker.cpp:71` | `getCost(float,float)` 内含 clamp | ✅ 编译通过 |
| #21 | Moderate | `smoother.cpp:286-290` | 检查 `worldToMap` 返回值 | ✅ 编译通过 |
| #22 | Moderate | `smoother.cpp:198-207` | 递归 `smoothImpl` 传剩余时间 | ✅ 编译通过 |
| #23 | Moderate | `obstacle_heuristic.cpp:61-72` | 负值 `floor` 结果 clamp 到 0 再转 unsigned | ✅ 编译通过 |
| #24 | Moderate | `obstacle_heuristic.cpp:159-168` | 改用 unsigned 字面量比较，避免下溢 | ✅ 编译通过 |
| #25 | Moderate | `node_hybrid.cpp:336-345` | 先转 int 检查范围再调 `getIndex` | ✅ 编译通过 |
| #26 | Minor | `node_basic.hpp:16-22` | 构造函数初始化 `motion_index`/`turn_dir` | ✅ 编译通过 |
| #27 | Minor | `smoother.cpp:134,163` | `path_size` 改 `size_t`，循环变量同步 | ✅ 编译通过 |
| **#28 (new)** | **Critical** | **`distance_heuristic.cpp:31`** | **修复查找表 `resize` 尺寸：与预计算迭代数一致（`floor(size/2)-ceil(-size/2)+1` × `floor(size/2)+1` × angle）** | ✅ 编译通过 |

---

## Open

> All originally reported bugs and the newly discovered table-size mismatch are now closed.

_(none)_

---

## Statistics

| Status | Critical | Moderate | Minor | Total |
|--------|----------|----------|-------|-------|
| Fixed  | 5        | 13       | 9     | 27    |
| Open   | 0        | 0        | 0     | 0     |
| **Total** | **5** | **13**  | **9** | **27** |

---

## Notes on Bug #28 (新增)

**File**: `src/distance_heuristic.cpp:31-43`

**Symptom**: 初始化时 `precomputeDistanceHeuristic` 触发堆缓冲区溢出。

**Root cause**:
- `resize` 用 `size_lookup_ * ceil(size_lookup_/2) * dim_3_size` （以 `size=20` 为例：20 × 10 × 72 = 14400）。
- 预计算循环迭代 `(floor(20/2) - ceil(-20/2) + 1) × (floor(20/2) + 1) × 72` = 21 × 11 × 72 = 16632。
- 超出部分 2232 个元素写入堆，污染相邻分配。

**Fix**: `resize` 改为与预计算完全一致的尺寸公式。

**Trigger**: 任何调用 `setGoal`（间接触发 `precomputeDistanceHeuristic`）的规划请求都会触发。
