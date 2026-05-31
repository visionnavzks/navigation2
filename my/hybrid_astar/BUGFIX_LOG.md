# Hybrid A* Bug Fix Log

Generated: 2026-05-31

---

## Fixed

| Bug | Severity | File | Fix | Verified |
|-----|----------|------|-----|----------|
| #1 | Critical | `distance_heuristic.cpp:107-108` | 角度转换改用 `getAngleFromBin()` | ✅ 编译通过 |
| #3 | Moderate | `distance_heuristic.cpp:74` | `>` 改为 `>=` | ✅ 编译通过 |
| #4 | Moderate | `smoother.cpp:86` | `reversing_segment` 初始化为 `false` | ✅ 编译通过 |
| #7 | Minor | `analytic_expansion.cpp:166` | `static const` 改为 `constexpr` | ✅ 编译通过 |
| #8 | Moderate | `smoother.cpp:175` | `abs()` 改为 `std::fabs()` | ✅ 编译通过 |
| #9 | Minor | `smoother.cpp:362,401` | `>` 改为 `>=` | ✅ 编译通过 |
| #10 | Minor | `constants.hpp:73-77` | `const` 改为 `inline constexpr` | ✅ 编译通过 |
| #12 | Moderate | `collision_checker.cpp:87-89` | `mapToWorld` 改用 `mapToCenter()` | ✅ 编译通过 |
| #15 | Moderate | `smoother.cpp:32` | 加 `fmod` + 归一化角度差 | ✅ 编译通过 |
| #16 | Moderate | `node_hybrid.cpp:50,148` | `asin` 参数加 `std::min(1.0, ...)` clamp | ✅ 编译通过 |
| #17 | Minor | `node_hybrid.cpp:258-261` | `getClosestAngularBin` 加归一化到 `[0, 2π)` | ✅ 编译通过 |
| #18 | Minor | `smac_planner_hybrid.cpp:113` | `setFootprint` 加 `std::lock_guard` | ✅ 编译通过 |
| #13 | Moderate | `a_star.cpp:138` | `setGoal` 赋值 `_coarse_search_resolution` | ✅ 编译通过 |
| #20 | Critical | `collision_checker.cpp:71-72` | `getCost(float,float)` 内含 clamp，替代手动转 `unsigned int` | ✅ 编译通过 |

---

## Open — Critical

| Bug | File | Issue | Fix Needed |
|-----|------|-------|------------|
| #2 | `distance_heuristic.cpp:87,95` | 查找表边界 `<` vs 预计算 `<=` 不匹配 | 改为 `<=` |
| #19 | `obstacle_heuristic.cpp:31-34` | 查找表收缩时 `fill_n` 越界写 | 先 `fill` 再 `resize` |

---

## Open — Moderate

| Bug | File | Issue | Fix Needed |
|-----|------|-------|------------|
| #11 | `obstacle_heuristic.cpp:116-119` | 邻域扩展行环绕 | 加行边界检查 |
| #14 | `distance_heuristic.cpp:90` | `theta_pos` 越界 | `%= num_angle_quantization` (已部分修复需验证) |
| #21 | `smoother.cpp:287-288` | `worldToMap` 返回值未检查，`mx`/`my` 未初始化 | 检查返回值 |
| #22 | `smoother.cpp:202-205` | 递归精化忽略已用时间 | 传剩余时间 |
| #23 | `obstacle_heuristic.cpp:69-76` | 负 float 转 unsigned int UB | clamp 到 0 |
| #24 | `obstacle_heuristic.cpp:155-159` | `size_x - 3` 无符号下溢 | 改用 signed 比较 |
| #25 | `node_hybrid.cpp:428-432` | 负 float 转 unsigned int 计算索引 | 先转 int 检查边界 |

---

## Open — Minor

| Bug | File | Issue | Fix Needed |
|-----|------|-------|------------|
| #5 | `obstacle_heuristic.cpp:44` | `floor()` on `unsigned int` 参数 | 参数类型改 `float` |
| #6 | `costmap_2d.hpp:85-86` | 负 double 转 unsigned int | 加负值检查 |
| #26 | `node_basic.hpp:16-20` | `motion_index`/`turn_dir` 未初始化 | 构造函数初始化 |
| #27 | `smoother.cpp:138` | `size_t` → `unsigned int` 窄化 | 改为 `size_t` |

---

## Statistics

| Status | Critical | Moderate | Minor | Total |
|--------|----------|----------|-------|-------|
| Fixed | 2 | 6 | 6 | 14 |
| Open | 2 | 7 | 4 | 13 |
| **Total** | **4** | **13** | **10** | **27** |
