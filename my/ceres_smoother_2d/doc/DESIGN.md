# Ceres 2D Path Smoother — Design Specification

> 精简设计文档，供其他大模型复现本项目使用。
> 包含完整的算法描述、数据结构、代价函数公式和关键实现细节。

---

## 1. 项目概述

一个无 ROS 依赖的 C++17 2D 路径平滑库。输入：occupancy grid (PNG) + 粗糙参考路径（如 A* 输出）。输出：光滑、运动学可行、避障的轨迹。

**核心思路**：将路径建模为 $N$ 个 2D 点 $\{p_0, \ldots, p_{N-1}\}$，用 Ceres Solver 最小化 5 个代价项的加权和。起点 $p_0$ 和终点 $p_{N-1}$ 固定不动。

---

## 2. 文件结构

```
include/
  esdf_map.hpp            # ESDF 地图：加载 PNG → 计算符号距离场 → 双线性查询
  ceres_smoother_2d.hpp   # 核心：SmootherParams + 5 个 cost struct + PathSmoother2D 类
  astar.hpp               # C++ A*（8-connected, robot-radius 膨胀）
src/
  main.cpp                # C++ demo 入口
  nanobind_module.cpp     # Python 绑定
  stb_image_impl.cpp      # stb_image 实现单元（#define STB_IMAGE_IMPLEMENTATION）
python/
  app.py                  # Flask web 服务
  demo.py                 # matplotlib 可视化 demo
  templates/index.html    # Plotly 交互前端
tests/
  test_cpp.cpp            # C++ 单元测试
  test_python.py          # Python 集成测试
  test_web_api.py         # Web API 端到端测试
thirdparty/stb/
  stb_image.h, stb_image_write.h  # 内置 PNG 读写
```

---

## 3. ESDF 地图 (`esdf_map.hpp`)

### 3.1 数据结构

```
class ESDFMap:
  int width_, height_           # 像素尺寸
  double resolution_            # m/pixel
  double origin_x_, origin_y_   # 世界坐标原点
  vector<uint8_t> grid_         # 占据栅格: 1=obstacle, 0=free
  vector<double> esdf_          # 符号距离场 (row-major, 行对应 world-y)
```

### 3.2 加载与预处理

1. 用 stb_image 加载灰度 PNG（自动转单通道）
2. **垂直翻转**：PNG row 0 → grid row `(H-1)`，使 grid 行号与 world-y 正相关（匹配 ROS map_server 约定）
3. 阈值化：`pixel <= obstacle_thresh` → obstacle(1), 否则 free(0)

### 3.3 ESDF 计算算法

使用 **Felzenszwalb & Huttenlocher** (2012) 的精确欧氏距离变换：

```
算法: edt2d(binary_grid, w, h) → squared_distance_grid

1. 初始化: obstacle → 0, free → +∞
2. 对每一列做 1D 变换（列有 h 个元素）
3. 对每一行做 1D 变换（行有 w 个元素）
```

**1D 变换** (`distTransform1D`):

```
输入: f[q], q=0..n-1  (cost function)
输出: d[q] = min_q' (q-q')² + f[q']

使用抛物线包络算法:
  - 维护抛物线位置数组 v[] 和边界数组 z[]
  - O(n) 时间复杂度
```

**签名距离合成**:

```
d_free  = edt2d(grid, w, h)       # free→obstacle 的平方距离
d_obst  = edt2d(1-grid, w, h)     # obstacle→free 的平方距离

for each pixel i:
  if obstacle: esdf_[i] = -sqrt(d_obst[i]) * resolution
  if free:     esdf_[i] = +sqrt(d_free[i])  * resolution
```

**符号约定**：正 = 自由空间（到最近障碍物的距离），负 = 障碍物内部。

### 3.4 双线性插值（Jet 兼容）

```cpp
template<typename T>
T bilinearJet(T wx, T wy) const
{
  // world → 归一化网格坐标
  col = (wx - origin_x) / resolution
  row = (wy - origin_y) / resolution

  // clamp 到 [0, H-1] × [0, W-1]
  r = clamp(row, 0, H-1)
  c = clamp(col, 0, W-1)

  // 整数索引（从 Jet 的 .a 成员提取标量）
  r0 = floor(r_scalar), c0 = floor(c_scalar)
  r1 = r0+1, c1 = c0+1

  // 小数部分
  fr = r - r0, fc = c - c0

  // 4 个角点值（从 esdf_ 数组读取）
  v00, v10, v01, v11

  // 双线性插值
  return v00*(1-fr)*(1-fc) + v10*fc*(1-fr) + v01*(1-fc)*fr + v11*fc*fr
}
```

**为什么用双线性而不是 BiCubic**：ESDF 在障碍物边界处有尖锐不连续，BiCubic 会过冲（overshoot），产生错误的距离值和指向障碍物内部的梯度。双线性始终被 4 个邻居的 min/max 约束，梯度方向永远正确。

---

## 4. 代价函数 (`ceres_smoother_2d.hpp`)

路径表示：$P = \{p_0, p_1, \ldots, p_{N-1}\}$，$p_i = (x_i, y_i) \in \mathbb{R}^2$。

所有代价项的形式为 `residual = sqrt(w) * ...`，Ceres 内部计算 $0.5 \sum r^2$。

### 4.1 SmoothnessCost

**目标**：惩罚二阶差分（离散加速度/jerk）。

```
输入: p_prev, p_curr, p_next  (各 2D)
残差 (2 组件):
  r[0] = sqrt_w * (p_next.x - 2*p_curr.x + p_prev.x)
  r[1] = sqrt_w * (p_next.y - 2*p_curr.y + p_prev.y)
```

**效果**：产生三对角 Hessian，稀疏求解高效。直接最小化路径的"抖动"。

### 4.2 CurvatureCost

**目标**：约束局部转弯半径 ≥ `min_turning_radius`（角度超限版本）。

```
输入: p_prev, p_curr, p_next
v1 = p_curr - p_prev
v2 = p_next - p_curr
n1 = ||v1|| + ε,  n2 = ||v2|| + ε    (ε=1e-12 防除零)
ds = 0.5 * (n1 + n2)                  # 局部步长
dot = v1 · v2
cross = v1 × v2

θ = atan2(sqrt(cross² + ε), dot)      # 无符号转角 ∈ [0, π]
θ_limit = κ_max * ds                  # κ_max = 1/r_min
violation = θ - θ_limit

r[0] = violation > 0 ? sqrt_w * violation : 0
```

**关键设计**：
- 用 `atan2(sqrt(cross² + ε), dot)` 直接算角度，比点积 deficit 版更直观
- `sqrt(cross² + ε)` 替代 `abs(cross)`，避免 `abs()` 在 0 附近不可导
- 无分母，对近共线三角形免疫（无 NaN/Inf 风险）
- 对 Ceres Jet 友好，梯度平滑通过 `atan2` 和 `sqrt`

### 4.3 ReferenceCost

**目标**：弹簧式锚定到 A* 参考路径。

```
输入: p (当前点), p_ref (对应参考点)
残差 (2 组件):
  r[0] = sqrt_w * (p.x - p_ref.x)
  r[1] = sqrt_w * (p.y - p_ref.y)
```

**注意**：参考点是重采样后的路径点索引对应点，不是原始 A* 点。

### 4.4 PathLengthSquareCost

**目标**：弹性带力 — 最小化 $\sum \|p_{i+1} - p_i\|^2$。

```
输入: p_curr, p_next
残差 (2 组件):
  r[0] = sqrt_w * (p_next.x - p_curr.x)
  r[1] = sqrt_w * (p_next.y - p_curr.y)
```

**为什么不用 target-spacing 弹簧**：
- 线性残差 → 常数 Jacobian → Ceres 1-3 次迭代收敛
- 无固定休息长度 → 不与锁定的起点/终点冲突
- 均匀间距是副产品（$\sum \|\Delta s\|^2$ 在段等长时最小）

### 4.5 ObstacleCostCeres

**目标**：双项障碍物代价 — 软铰链 + 穿透惩罚。

```
输入: p (当前点)
dist = map.bilinearJet(p.x, p.y)     # ESDF 双线性插值（Jet 自动微分）
safe_dist = safety_margin + robot_radius

# 项 1: 软铰链（安全边界外侧惩罚）
diff = safe_dist - dist
r[0] = diff > 0 ? sqrt_w_obstacle * diff : 0

# 项 2: 穿透惩罚（障碍物内部惩罚）
pen = -dist
r[1] = pen > 0 ? sqrt_w_penetration * pen : 0
```

**为什么需要双项**：两项都是二次函数，但权重解耦。$w_{\text{obs}}=10$ 控制靠近安全边界的软推力，$w_{\text{pen}}=1000$ 控制深入墙壁的强拉力（斜率差 100 倍）。纯铰链也能推出，但权重调大会影响边界外的平滑度；双项设计让边界附近的避障力度和深处的惩罚力度可以独立调节。

---

## 5. 多阶段求解策略

```
obstacle_weight_stages = []
stage_weight = min(w_obstacle, 2.0)
obstacle_weight_stages.append(stage_weight)
while stage_weight * 10 < w_obstacle:
    stage_weight *= 10
    obstacle_weight_stages.append(stage_weight)
if stages[-1] < w_obstacle:
    stages.append(w_obstacle)
```

例如 `w_obstacle=200` → stages = `[2, 20, 200]`。

**每个阶段**构建一个新的 Ceres Problem，设置当前阶段的 `w_obstacle`：
- 阶段 0：低障碍权重，路径找到全局光滑形状
- 阶段 1..K：逐步收紧避障约束

`w_penetration` 在所有阶段保持全强度（不随阶段缩放）。

---

## 6. 优化器配置

```cpp
ceres::Solver::Options options;
options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // SuiteSparse
options.max_num_iterations = params.max_iterations;           // 默认 100
options.max_solver_time_in_seconds = params.max_time_seconds; // 默认 0.5s
options.num_threads = 1;  // 稀疏 Hessian 不受益于多线程
```

**稀疏性来源**：每个节点 $p_i$ 是一个 2D parameter block。Smoothness 和 Curvature 连接 $p_{i-1}, p_i, p_{i+1}$（三对角），Length 连接 $p_i, p_{i+1}$（双对角），Reference 和 Obstacle 只作用于单点。Hessian 天然稀疏。

---

## 7. A* 搜索 (`astar.hpp`)

### 7.1 规格

- 8-connected 网格
- 步代价：cardinal = 1, diagonal = √2
- 启发函数：$(max - min) + \sqrt{2} \cdot min$（精确 8-connected 最短距离，比 Euclidean 更紧但依然 admissible）
- 数据结构：binary heap + lazy deletion
- 存储：flat arrays（`g_score`, `came_from`, `closed`）— cache-friendly

### 7.2 Robot-radius 膨胀

```
if robot_radius > 0:
  for each cell i:
    inflated_occ[i] = (esdf[i] < robot_radius) ? 1 : 0
else:
  inflated_occ = original_occupancy
```

A* 在膨胀后的 occupancy 上搜索，返回的路径对圆形机器人可行。

### 7.3 坐标转换

```
world_to_grid:  cell_x = int((wx - origin_x) / resolution)  // truncation toward zero
                cell_y = int((wy - origin_y) / resolution)
grid_to_world:  wx = (cell_x + 0.5) * resolution + origin_x
                wy = (cell_y + 0.5) * resolution + origin_y
```

---

## 8. 弧长重采样 (`resamplePathByArcLength`)

```
输入: xs_in, ys_in (N ≥ 2), target_spacing
输出: xs_out, ys_out

1. 计算累积弧长 cum[i]
2. 总弧长 L = cum[N-1]
3. 输出点数 M = max(2, round(L / target_spacing) + 1)
4. 端点精确保留: out[0] = in[0], out[M-1] = in[N-1]
5. 中间点: 对每个 j=1..M-2:
   s = j * L / (M-1)           # 目标弧长位置
   找到 cum[i-1] ≤ s < cum[i]  # 所在段
   t = (s - cum[i-1]) / seg_len # 段内比例
   out[j] = in[i-1] + t * (in[i] - in[i-1])
```

两个开关：
- `resample_before_smooth`（默认 true）：优化前重采样输入路径
- `resample_after_smooth`（默认 false）：优化后重采样输出路径

---

## 9. Smooth 主流程

```
smooth(x_in, y_in, map) → SmootherResult:

1. 前处理: if resample_before_smooth → 重采样输入到均匀间距
2. 构建 path_optim[N] = {{x0,y0}, ..., {x_{N-1},y_{N-1}}}
3. 锁定 path_optim[0] 和 path_optim[N-1]
4. 对每个 obstacle_weight in stages:
   a. 构建 Ceres Problem
   b. 对每个中间节点 i=1..N-2:
      - 添加 ReferenceCost (if w_reference > 0)
      - 添加 ObstacleCostCeres (if w_obstacle > 0 或 w_penetration > 0)
   c. 对每个相邻对 i=0..N-2:
      - 添加 PathLengthSquareCost (if w_length > 0)
   d. 对每个三元组 i=1..N-2:
      - 添加 SmoothnessCost (if w_smooth > 0)
      - 添加 CurvatureCost (if w_max_curvature > 0)
   e. Solve → 更新 path_optim
   f. 检查: IsSolutionUsable? min_dist / min_margin?
5. 提取 result.x, result.y
6. 后处理: if resample_after_smooth → 重采样输出
7. 返回 SmootherResult
```

---

## 10. 默认参数

```cpp
SmootherParams defaults = {
  .max_iterations      = 100,
  .max_time_seconds    = 0.5,
  .verbose             = false,
  .w_smooth            = 10.0,
  .w_max_curvature     = 1000.0,
  .min_turning_radius  = 0.2,     // meters
  .w_reference         = 5.0,
  .w_length            = 2.0,
  .target_spacing      = 0.3,     // meters (resample only)
  .w_obstacle          = 10.0,
  .w_penetration       = 1000.0,
  .safety_margin       = 1.0,     // meters
  .robot_radius        = 0.5,     // meters
  .resample_after_smooth  = false,
  .resample_before_smooth = true,
};
```

---

## 11. Ceres AutoDiff 的 Jet 机制

所有代价函数模板化为 `operator()(const T* ..., T* residual)`。Ceres 在运行时将 `T` 替换为 `Jet<double, N>`，其中 N = 该 residual block 涉及的参数维度。

**关键实现细节**：
- `bilinearJet<T>()` 在 Jet 空间完成插值，梯度通过插值权重自动传播
- 整数索引从 `Jet.a` 成员提取（`if constexpr (is_same<T, double>)` 分支处理标量和 Jet 两种情况）
- `GetScalar<T>` 辅助结构通过 `.a` 成员提取标量（Ceres 2.0 无 `JetOps`）

---

## 12. Python 绑定 (nanobind)

暴露的类型：
- `ESDFMap` — 构造函数 (path + resolution 或 raw data)，属性 (width, height, resolution, origin_x, origin_y, world_width, world_height)，方法 (get_distance, in_bounds, get_esdf_array, get_occupancy_array)
- `SmootherParams` — 所有字段可读写
- `SmootherResult` — 只读属性 (success, x, y, final_cost, solve_time_ms, iterations, report)
- `PathSmoother2D` — 构造函数 (params)，方法 (smooth(x, y, map))
- `AStarResult` — 只读属性 (success, x, y, expansions, time_ms)
- `astar_solve(map, sx, sy, gx, gy, robot_radius=0.0)` — 自由函数
- `resample_path_by_arc_length(x, y, target_spacing)` → `(xs, ys)` — 自由函数

---

## 13. Web API (Flask)

### 端点

| Method | Path | 描述 |
|--------|------|------|
| GET | `/api/costmap` | 返回地图元数据 + base64 PNG |
| POST | `/api/plan` | 规划路径 |

### POST /api/plan

请求体:
```json
{
  "start": [sx, sy],
  "goal": [gx, gy],
  "robot_radius": 0.5,
  "safety_margin": 1.0,
  "downsample": 3
}
```

响应:
```json
{
  "found": true,
  "success": true,
  "start_ok": true,
  "goal_ok": true,
  "raw_x": [...], "raw_y": [...],
  "smooth_x": [...], "smooth_y": [...],
  "raw_points": N, "smooth_points": M,
  "smooth_length": 45.2,
  "min_clearance": 0.31,
  "clearances": [...],
  "plan_ms": 2.5,
  "smooth_ms": 5.1,
  "reason": ""
}
```

---

## 14. 构建依赖

- CMake >= 3.14
- Eigen3 >= 3.4
- Ceres Solver >= 2.0 (含 SuiteSparse)
- stb_image (内置)
- nanobind (Python 绑定)
- Python >= 3.8 + numpy, matplotlib, flask (可选)

---

## 15. 已知设计约束与陷阱

1. **PNG 翻转**：必须垂直翻转 PNG 行，否则 world-y 与视觉上下颠倒
2. **双线性 vs BiCubic**：ESDF 边界处 BiCubic 过冲导致梯度指向障碍物内部
3. **穿透项必要性**：纯铰链在障碍物内部梯度恒定但小，优化器可卡住
4. **多阶段求解**：直接用高 w_obstacle 会陷入局部最优（路径贴墙走）
5. **num_threads=1**：稀疏 Hessian 的多线程开销 > 加速（<2k 变量时）
6. **resample_before_smooth**：A* 路径点密度不均（墙边密、空旷稀疏），不重采样则 w_length 无法插入新点
7. **start/goal 锁定**：`SetParameterBlockConstant` 固定首尾，优化只动中间点
8. **Jet 标量提取**：`if constexpr (is_same<T, double>)` 分支是必须的，Ceres Jet 无隐式转换
