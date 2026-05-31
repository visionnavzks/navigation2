# Hybrid A* Code Redundancy Report

Generated: 2026-05-31

---

## 1. `initDubin` / `initReedsShepp` ~80% 重复 (High) — ✅ Fixed

**Files**: `src/node_hybrid.cpp:21-118` vs `src/node_hybrid.cpp:120-230`

提取 `initCommon(MotionModel)` 私有方法，`initDubin` 和 `initReedsShepp` 各缩减为一行调用。

---

## 2. Start/Goal 朝向 bin 归一化重复 (Medium) — ✅ Fixed

**File**: `src/smac_planner_hybrid.cpp`

提取 `wrapBinIndex()` 到 `types.hpp`，替代所有手动 while/if 归一化。

---

## 3. `setGoal` 中 `_coarse_search_resolution` 重复赋值 (Low) — ✅ Fixed

**File**: `src/a_star.cpp:192`

删除 ALL_DIRECTION case 中的重复赋值。

---

## 4. 重复类型定义 (Medium) — ✅ Fixed

**Files**: `include/my/hybrid_astar/types.hpp` vs `include/my/hybrid_astar/obstacle_heuristic.hpp`

统一使用 `NodeHeuristicPair`，删除 `ObstacleHeuristicElement`。

---

## 5. 重复比较器结构体 (Medium) — ✅ Fixed

**Files**: `include/my/hybrid_astar/types.hpp`

统一使用 `NodeHeuristicComparator`，删除 `ObstacleHeuristicComparator`。

---

## 6. 五种不同的 Theta 归一化模式 (Medium) — ✅ Fixed

**Files**: 多个文件

添加 `wrapBinIndex(int, unsigned int)` 和 `wrapAngle(double)` 到 `types.hpp`，所有调用点统一使用。

---

## 7. 魔数 `252.0f` 应使用命名常量 (Medium) — ✅ Fixed

**Files**: `node_hybrid.cpp`, `analytic_expansion.cpp`, `obstacle_heuristic.cpp`

添加 `MAX_NON_OBSTACLE_COST_SQ`，所有 `252.0f` 和 `63504.0f` 替换为命名常量。

---

## 8. `getWorldCoords()` ≡ `mapToCenter()` (Medium) — ✅ Fixed

**Files**: `utils.hpp`, `costmap_2d.hpp`

`getWorldCoords` 内部调用 `mapToCenter`，添加 double 重载。

---

## 9. Start/End 边界条件逻辑重复 (Medium) — ⏭️ Skipped

两个方法的迭代方向和索引计算差异较大，强行合并会增加复杂度。

---

## 10. OMPL StateSpace 创建分散在 5 处 (Medium) — ✅ Fixed

**Files**: 4 个源文件

提取 `createStateSpace(MotionModel, double)` 到 `types.hpp`，5 处调用统一。

---

## 11. Downsampled size 重复计算 (Low) — ✅ Fixed

**File**: `src/obstacle_heuristic.cpp`

缓存为 `cached_size_x_`/`cached_size_y_` 成员变量。

---

## 12. 无意义的 `onVisitationCheckNode` 包装 (Low) — ✅ Fixed

**File**: `src/a_star.cpp`

内联为 `current_node->wasVisited()`，删除函数定义和声明。

---

## 13. 未使用的 `const getCostmap()` 重载 (Low) — ✅ Fixed

**File**: `include/my/hybrid_astar/collision_checker.hpp`

删除未使用的 `const Costmap2D * getCostmap() const`。

---

## Summary

| Status | Count |
|--------|-------|
| Fixed | 12 |
| Skipped | 1 (#9) |
| **Total** | **13** |
