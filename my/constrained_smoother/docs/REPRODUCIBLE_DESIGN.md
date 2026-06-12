# Kinematic Smoother — 可复现设计文档

> **目的**：本文档足够自包含，可以让另一个 AI 模型或工程师从零实现 `constrained_smoother` 包的全部功能，无需阅读任何源码。
>
> **阅读约定**：数学符号中，向量用粗体，标量用斜体。代码片段为 Python 风格伪代码，C++ 实现与之逻辑等价。

---

## 1. 问题定义

给定一条由离散路径点组成的 2D 路径（可能来自 A* 等全局规划器），以及一张代价地图（costmap），输出一条满足以下条件的平滑路径：

1. **运动学一致性**：相邻状态之间的位置、朝向、曲率满足离散运动学模型。
2. **曲率约束**：路径上任意点的曲率不超过 `max_curvature`（即转弯半径不小于 `1/max_curvature`）。
3. **障碍物避让**：路径与障碍物的距离不低于安全阈值。
4. **端点锚定**：起点位置/朝向严格固定，终点位置在可配置的容差框内。
5. **路径平滑**：曲率变化率小，路径自然流畅。

---

## 2. 状态表示

每个离散状态是一个 5 维向量：

$$
\mathbf{s}_i = (x_i,\ y_i,\ \theta_i,\ \kappa_i,\ ds_i)
$$

| 分量 | 含义 | 单位 |
|------|------|------|
| $x_i$ | 世界坐标系 x 位置 | m |
| $y_i$ | 世界坐标系 y 位置 | m |
| $\theta_i$ | 航向角（yaw） | rad |
| $\kappa_i$ | 曲率（$= 1/R$，$R$ 为转弯半径） | $1/m$ |
| $ds_i$ | 从状态 $i$ 到状态 $i+1$ 的弧长步长 | m |

所有状态展平为一维数组供优化器使用：

$$
\mathbf{X} = [x_0, y_0, \theta_0, \kappa_0, ds_0,\ x_1, y_1, \theta_1, \kappa_1, ds_1,\ \ldots]
$$

数组长度 $= 5N$，其中 $N$ 为状态总数。

---

## 3. 输入输出约定

### 3.1 输入

- **路径** `path`: $[(x_0, y_0, d_0),\ (x_1, y_1, d_1),\ \ldots]$，其中第三分量 $d_i$ 是方向符号：$+1$ 表示前进，$-1$ 表示倒车。
- **起点方向** `start_dir`: $(dx, dy)$ 向量，表示起点切向。
- **终点方向** `end_dir`: $(dx, dy)$ 向量，表示终点切向。
- **代价地图** `costmap`: 可选，栅格地图，包含原点、分辨率、障碍物占据栅格。
- **预计算 ESDF** `precomputed_esdf`: 可选，与 costmap 同尺寸的有符号距离场扁平数组。

### 3.2 输出

- **平滑路径** `smoothed_path`: $[(x_0, y_0, \text{yaw}_0),\ \ldots]$，第三分量是航向角（弧度）。
- **候选路径** `candidate_path`: 后验校验失败时仍可返回，用于诊断。
- **诊断元数据**: `optimized_knot_count`, `target_spacing`, `success`, `report`.

---

## 4. 总体流程（9 步）

```
输入路径 ──→ ① 下采样 ──→ ② 展开状态链 ──→ ③ 初始化变量
                                                 │
                                                 ▼
⑨ 上采样输出 ←── ⑧ 后验校验 ←── ⑦ Ceres 求解 ←── ⑥ 施加变量边界 ←── ⑤ 拼接残差
                                                   ↑
                                              ④ 准备 ESDF
```

每一步的详细说明见后续章节。

---

## 5. 第 ① 步：输入路径下采样

**目的**：减少参与优化的状态数，同时保留关键几何特征（特别是换向点）。

**算法**：

```
function downsample_input_path(path, downsampling_factor):
    if downsampling_factor <= 1 or len(path) <= 2:
        return path

    result = [path[0]]
    last_kept = 0

    for i in 1 .. len(path)-2:
        prev_sign  = sign(path[i-1].z)
        curr_sign  = sign(path[i].z)
        next_sign  = sign(path[i+1].z)
        around_cusp = (curr_sign != prev_sign) or (curr_sign != next_sign)

        if around_cusp or (i - last_kept) >= downsampling_factor:
            result.append(path[i])
            last_kept = i

    if result[-1] != path[-1]:
        result.append(path[-1])

    return result
```

**关键细节**：
- 换向点（cusp）附近的点永远不被丢弃。
- `downsampling_factor` 默认为 1（不下采样）。

---

## 6. 第 ② 步：展开状态链（buildProcessedPath）

**目的**：把原始路径转换为带元数据的状态链。

### 6.1 算法

```
function build_processed_path(path, start_dir, end_dir, params, costmap):
    processed = new KinematicProcessedPath()
    processed.start_theta = atan2(start_dir.y, start_dir.x)
    processed.end_theta   = atan2(end_dir.y, end_dir.x)

    sampled = downsample_input_path(path, params.path_downsampling_factor)

    # Step A: 计算每段的 gear 方向
    gear_directions = []
    for i in 0 .. len(sampled)-2:
        if params.reversing_enabled:
            gear_directions.append(-1.0 if sampled[i].z < 0 else +1.0)
        else:
            gear_directions.append(+1.0)

    # Step B: 展开状态，插入 cusp
    processed.reference_points.append((sampled[0].x, sampled[0].y))

    for i in 0 .. len(sampled)-2:
        current_gear = gear_directions[i]
        next_gear = gear_directions[i+1] if i+1 < len(gear_directions) else current_gear

        processed.gears.append(current_gear)
        processed.is_cusp_segment.append(False)
        processed.reference_points.append((sampled[i+1].x, sampled[i+1].y))

        # 如果 gear 变化，在换向点插入 cusp 状态
        if i+2 < len(sampled) and current_gear != next_gear:
            processed.gears.append(0.0)              # cusp: gear = 0
            processed.is_cusp_segment.append(True)
            processed.reference_points.append((sampled[i+1].x, sampled[i+1].y))

    processed.state_count = len(processed.reference_points)

    # Step C: 初始化 theta, kappa, ds
    theta = [0.0] * N
    kappa = [0.0] * N    # 初始曲率全为零
    ds    = [0.0] * N

    spacing_sum = 0
    spacing_count = 0

    for i in 0 .. N-2:
        dx = ref[i+1].x - ref[i].x
        dy = ref[i+1].y - ref[i].y
        seg_len = hypot(dx, dy)

        if is_cusp_segment[i]:
            theta[i] = theta[i-1] if i > 0 else start_theta
            ds[i] = 0.0
            continue

        if seg_len > 1e-6:
            heading = atan2(dy, dx)
            if gears[i] < 0:  heading += π   # 倒车时朝向反转
            theta[i] = normalize_angle(heading)
            ds[i] = seg_len
            spacing_sum += seg_len
            spacing_count += 1
        else:
            theta[i] = theta[i-1] if i > 0 else start_theta

    # 最后一个点的 theta
    theta[N-1] = theta[N-2] if N > 1 else start_theta

    if params.keep_start_orientation: theta[0] = start_theta
    if params.keep_goal_orientation:  theta[N-1] = end_theta

    # 计算目标间距
    processed.target_spacing = (spacing_sum / spacing_count) if spacing_count > 0
                               else (costmap.resolution if costmap else 0.2)

    # Step D: 展平为初始变量数组
    processed.initial_variables = []
    for i in 0 .. N-1:
        processed.initial_variables.extend([ref[i].x, ref[i].y, theta[i], kappa[i], ds[i]])

    return processed
```

### 6.2 Cusp 插入规则

假设原始路径有 gear 变化：`... [+1, +1, -1, -1] ...`

在 gear 变化的换向点处，会插入一个额外的 cusp 状态：

```
状态索引:    ...  i   i+1(C)  i+2  ...
gear:        ... +1    0      -1   ...
is_cusp:     ... false true   false ...
reference:   ...  P    P'     P'   ...
```

- Cusp 状态的位置与前一个路径点相同。
- `gear = 0` 表示静止过渡。
- `ds = 0` 表示零步长（不移动）。
- cusp 段的含义是：车辆在该点停下，切换档位，再继续。

---

## 7. 第 ③ 步：ESDF 准备

**目的**：为障碍物残差和后验校验提供距离场。

**ESDF（Euclidean Signed Distance Field）**：对 costmap 中每个栅格 $(i,j)$，计算到最近障碍物表面的有符号欧几里得距离。正表示在自由空间，负表示在障碍物内部。

```
function initialize_esdf_values(costmap, params, precomputed_esdf):
    if not params.obstacle_terms_enabled():
        return empty array

    if precomputed_esdf is not None:
        assert len(precomputed_esdf) == costmap.size_x * costmap.size_y
        esdf_values = precomputed_esdf
    else:
        esdf_values = compute_esdf(costmap, obstacle_threshold=LETHAL_OBSTACLE)

    # 构建 Ceres Grid2D + BiCubicInterpolator 供残差使用
    esdf_grid = Grid2D(esdf_values, rows=costmap.size_y, cols=costmap.size_x)
    esdf_interpolator = BiCubicInterpolator(esdf_grid)
```

**注意**：ESDF 的坐标约定是 `(row, col)`，即 `(y, x)`。插值时需要偏移 0.5 格以对齐栅格中心。

---

## 8. 第 ⑤ 步：残差拼接（buildProblem）

这是整个优化的核心。所有残差由 Ceres AutoDiff 驱动（或 Python 版用 `scipy.optimize.least_squares`）。

### 8.1 残差总览

| 残差类型 | 每组残差数 | 作用于 | 说明 |
|----------|-----------|--------|------|
| **Transition** | 7 | $(s_i, s_{i+1})$ | 运动学一致性 |
| **Boundary (start)** | 4 | $s_0$ | 锚定起点 |
| **Boundary (goal)** | 4 | $s_{N-1}$ | 锚定终点（含容差） |
| **Reference** | 2 | $s_i$ | 靠近参考路径 |
| **Obstacle** | 动态 (1 或 $M$) | $s_i$ | 障碍物避让 |

### 8.2 运动学过渡残差（TransitionCostFunctor）— 7 个残差

这是核心残差，约束相邻状态满足离散运动学模型。

**输入**：
- 当前状态 $\mathbf{s}_i = (x, y, \theta, \kappa, ds)$
- 下一状态 $\mathbf{s}_{i+1} = (x', y', \theta', \kappa', ds')$
- gear $g$（$+1$ 前进，$-1$ 倒车，$0$ cusp）
- `is_cusp_segment` 布尔值

**运动学模型**：

方向因子：
$$
\text{dir} = \begin{cases} +1 & \text{if } g \geq 0 \\ -1 & \text{if } g < 0 \end{cases}
$$

用**梯形曲率积分**预测下一朝向：
$$
\theta_{\text{pred}} = \theta + \text{dir} \cdot ds \cdot \frac{\kappa + \kappa'}{2}
$$

用**Euler midpoint**预测下一位置：
$$
\theta_{\text{mid}} = \frac{\theta + \theta_{\text{pred}}}{2}
$$
$$
x_{\text{pred}} = x + \text{dir} \cdot ds \cdot \cos(\theta_{\text{mid}})
$$
$$
y_{\text{pred}} = y + \text{dir} \cdot ds \cdot \sin(\theta_{\text{mid}})
$$

**正常段残差**（`is_cusp_segment = false`）：

| 索引 | 公式 | 含义 | 权重 |
|------|------|------|------|
| [0] | $w_m \cdot (x' - x_{\text{pred}})$ | 位置 x 误差 | `model_weight` |
| [1] | $w_m \cdot (y' - y_{\text{pred}})$ | 位置 y 误差 | `model_weight` |
| [2] | $w_m \cdot \text{angle\_diff}(\theta', \theta_{\text{pred}})$ | 朝向误差 | `model_weight` |
| [3] | $w_c \cdot \frac{\kappa + \kappa'}{2}$ | 曲率大小惩罚 | `curvature_weight` |
| [4] | $w_{cr} \cdot \frac{\kappa' - \kappa}{\sqrt{ds}}$ | 曲率变化率 | `curvature_rate_weight` |
| [5] | $w_s \cdot \frac{ds - ds_{\text{target}}}{ds_{\text{target}}}$ | 步长误差（归一化） | `spacing_weight` |
| [6] | $w_l \cdot ds$ | 长度惩罚 | `length_weight` |

其中：
- $\text{angle\_diff}(a, b) = \text{normalize\_angle}(a - b)$，归一化到 $(-\pi, \pi]$
- $\text{normalize\_angle}(\alpha) = \text{atan2}(\sin(\alpha), \cos(\alpha))$
- 残差 [4] 的分母在 $ds < 10^{-3}$ 时用 $0.03$ 代替，避免除零
- Ceres 的目标函数是 $\frac{1}{2}\sum r_i^2$，所以权重 $w$ 实际效果是 $\frac{1}{2} w^2 \cdot (\text{物理量})^2$

**Cusp 段残差**（`is_cusp_segment = true`）：

| 索引 | 公式 | 含义 |
|------|------|------|
| [0] | $w_f \cdot (x' - x)$ | 强制 x 不变 |
| [1] | $w_f \cdot (y' - y)$ | 强制 y 不变 |
| [2] | $w_f \cdot \text{angle\_diff}(\theta', \theta)$ | 强制朝向不变 |
| [3] | 0 | — |
| [4] | 0 | — |
| [5] | $w_s \cdot 10 \cdot ds$ | 强惩罚非零步长 |
| [6] | $w_l \cdot ds$ | 长度惩罚 |

`fix_weight` $w_f$ 是直接权重（不取平方根），默认值 100。

### 8.3 边界残差（BoundaryCostFunctor）— 4 个残差

**作用**：锚定起点和终点。

**输入**：
- 状态 $\mathbf{s} = (x, y, \theta, \kappa, ds)$
- 参考点 $(x_{\text{ref}}, y_{\text{ref}})$
- 目标朝向 $\theta_{\text{ref}}$
- 纵向容差 $t_{\text{lon}}$、横向容差 $t_{\text{lat}}$、朝向容差 $t_{\theta}$
- `keep_orientation` 布尔值

**坐标变换**：将位置误差投影到目标坐标系：

$$
\begin{aligned}
\Delta x &= x - x_{\text{ref}} \\
\Delta y &= y - y_{\text{ref}} \\
e_{\text{lon}} &= \cos(\theta_{\text{ref}}) \cdot \Delta x + \sin(\theta_{\text{ref}}) \cdot \Delta y \\
e_{\text{lat}} &= -\sin(\theta_{\text{ref}}) \cdot \Delta x + \cos(\theta_{\text{ref}}) \cdot \Delta y
\end{aligned}
$$

**残差**：

| 索引 | 公式 | 条件 |
|------|------|------|
| [0] | $w_f \cdot \max(0,\ |e_{\text{lon}}| - t_{\text{lon}})$ | 始终 |
| [1] | $w_f \cdot \max(0,\ |e_{\text{lat}}| - t_{\text{lat}})$ | 始终 |
| [2] | $w_f \cdot \max(0,\ |\text{angle\_diff}(\theta, \theta_{\text{ref}})| - t_{\theta})$ | 仅当 `keep_orientation=true` |

**起点 vs 终点的参数差异**：

| 参数 | 起点 | 终点 |
|------|------|------|
| `t_lon` | 0 | `goal_longitudinal_tolerance` |
| `t_lat` | 0 | `goal_lateral_tolerance` |
| `t_theta` | 0 | `goal_orientation_tolerance` |
| `keep_orientation` | `keep_start_orientation` | `keep_goal_orientation` |

容差为 0 时退化为绝对硬锚定。

**终点参考朝向**：当 `keep_goal_orientation=false` 时，终点的容差框朝向由末段几何朝向决定：

```
function goal_position_frame_heading(reference_points, end_theta, keep_goal_orientation):
    if keep_goal_orientation or len(reference_points) < 2:
        return end_theta
    delta = reference_points[-1] - reference_points[-2]
    if delta.norm() < EPSILON:
        return end_theta
    return atan2(delta.y, delta.x)
```

### 8.4 参考路径残差（ReferenceCostFunctor）— 2 个残差

**作用**：把优化后的路径点拉回原始参考位置，防止漂移。

| 索引 | 公式 |
|------|------|
| [0] | $w_r \cdot (x - x_{\text{ref}})$ |
| [1] | $w_r \cdot (y - y_{\text{ref}})$ |

权重 $w_r$ = `reference_path_weight_sqrt`。当 $w_r < 10^{-9}$ 时不添加此残差。

### 8.5 障碍物残差（ObstacleCostFunctor）— 动态数量

**作用**：基于 ESDF 的距离惩罚，将路径推离障碍物。

**单点模式**（`cost_check_points` 为空）：每个状态 1 个残差。

**多点模式**（`cost_check_points` 非空）：每个检测点 1 个残差。`cost_check_points` 是 $[x_{\text{local}}, y_{\text{local}}, \text{weight}, \ldots]$ 的列表，每 3 个数为一组。

**惩罚模型**：

```
function obstacle_penalty(world_x, world_y):
    # 世界坐标 → 栅格坐标
    grid_x = (world_x - origin_x) / resolution
    grid_y = (world_y - origin_y) / resolution

    # 越界检查
    if grid_x < 1.5 or grid_y < 1.5 or grid_x >= size_x - 1.5 or grid_y >= size_y - 1.5:
        return 1.0   # 常数边界残差

    # ESDF 双三次插值（注意 row/col 顺序和 0.5 偏移）
    distance = esdf_interpolator.Evaluate(grid_y - 0.5, grid_x - 0.5)

    # 减去机器人半径
    surface_distance = distance - cost_check_radius

    # 安全距离检查
    if surface_distance >= obstacle_safe_distance:
        return 0.0

    # hinge residual（Ceres 会平方形成二次代价）
    return (obstacle_safe_distance - surface_distance) / obstacle_safe_distance
```

**多点检测的坐标变换**：

```
for each (local_x, local_y, point_weight) in cost_check_points:
    world_x = x + cos(θ) * local_x - sin(θ) * local_y
    world_y = y + sin(θ) * local_x + cos(θ) * local_y
    residual = pose_weight * point_weight * obstacle_penalty(world_x, world_y)
```

**障碍权重**：所有状态使用统一的 `costmap_weight_sqrt`，不为 cusp 单独配置障碍权重。

---

## 9. 第 ⑥ 步：施加变量边界（applyBounds）

除软残差外，部分约束通过显式参数边界实现：

| 变量 | 下界 | 上界 | 条件 |
|------|------|------|------|
| $x_i$ | $x_{\text{ref},i} - d_{\max}$ | $x_{\text{ref},i} + d_{\max}$ | `reference_point_max_deviation_m > 0` |
| $y_i$ | $y_{\text{ref},i} - d_{\max}$ | $y_{\text{ref},i} + d_{\max}$ | 同上 |
| $\kappa_i$ | $-\kappa_{\max}$ | $\kappa_{\max}$ | 始终，$\kappa_{\max} = \max(\text{max\_curvature}, 10^{-6})$ |
| $ds_i$ | $10^{-6}$（非 cusp 且非末点） | `max_spacing`（若 > 0） | 始终 |
| $ds_i$ | $0$ | `max_spacing`（若 > 0） | cusp 段或末点 |

**关键设计**：曲率和步长用硬边界而非软残差，避免求解器在明显不合理的区域浪费迭代。

---

## 10. 第 ⑦ 步：Ceres 求解

### 10.1 求解器配置

```cpp
Solver::Options options;
options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;  // 默认
options.max_num_iterations = 50;                       // OptimizerParams.max_iterations
options.function_tolerance = 1e-6;
options.gradient_tolerance = 1e-10;
options.parameter_tolerance = 1e-8;
options.max_solver_time_in_seconds = params.max_time;  // 默认 10s
options.num_threads = 1;                               // 小规模问题线程开销大于加速
```

### 10.2 C++ vs Python 实现差异

| 方面 | C++ | Python |
|------|-----|--------|
| 自动微分 | Ceres AutoDiff (`Jet<double,N>`) | `scipy.optimize.least_squares` (数值差分) |
| ESDF 插值 | `ceres::BiCubicInterpolator<Grid2D<double>>` | 手写双三次插值 |
| 求解器 | Ceres Solver | scipy L-BFGS-B / Trust Region |
| 障碍物残差 | `DynamicAutoDiffCostFunction` (动态残差数) | NumPy 向量化计算 |

---

## 11. 第 ⑧ 步：后验校验（SmootherValidator）

**目的**：把"数值上收敛"和"工程上可交付"明确分开。即便 Ceres 返回 `SUCCESS`，也不代表结果可用。

### 11.1 校验流程（按顺序执行，任一失败即拒绝）

#### 11.1.1 状态向量形状与有限值

```
for i in 0 .. N-1:
    assert variables[i*5 .. i*5+5] 全部 isfinite
```

#### 11.1.2 边界约束

**起点**：
- 位置偏移 $= \sqrt{(x_0 - x_{\text{ref},0})^2 + (y_0 - y_{\text{ref},0})^2}$
- 容差 $= \max(\text{resolution} \times 0.5, 10^{-3})$
- 若 `keep_start_orientation`，朝向偏移 $= |\text{angle\_diff}(\theta_0, \theta_{\text{start}})|$，容差 $= 0.1$ rad

**终点**：
- 位置误差投影到目标坐标系（与边界残差相同的 lon/lat 分解）
- 纵向/横向容差取 `max(用户设定值, 位置容差)`
- 额外增加 `convergence_epsilon = 5e-4` 以容忍数值微小偏差
- 若 `keep_goal_orientation`，朝向误差 $> \max(\text{goal\_orientation\_tolerance}, 0.1)$ 则拒绝

#### 11.1.3 段一致性

对每一对相邻状态 $(i, i+1)$：

**Cusp 段**：
- 位移 $= \|(x', y') - (x, y)\| < \text{位置容差}$
- 朝向差 $= |\text{angle\_diff}(\theta', \theta)| < 0.1$ rad

**非 Cusp 段**：
- 位移 $> \text{displacement\_tol}$（$\max(\text{resolution} \times 0.25, 10^{-4})$），否则拒绝（段坍缩）
- 位移在朝向方向上的投影 $= \Delta x \cos\theta + \Delta y \sin\theta$
  - 前进段 ($g \geq 0$)：投影必须 $> 0$
  - 倒车段 ($g < 0$)：投影必须 $< 0$

#### 11.1.4 曲率约束

**双重检查**：
1. 显式状态曲率 $|\kappa_i| > \kappa_{\max} + 10^{-4}$ 则拒绝
2. 几何曲率 $= \frac{|\text{angle\_diff}(\theta', \theta)|}{\text{displacement}} > \kappa_{\max} + 10^{-4}$ 则拒绝

几何曲率检查覆盖了"$\kappa$ 合法但输出轨迹几何超限"的边界情况。

#### 11.1.5 障碍物净空

对每个状态的足迹采样点：
- 世界坐标查询 ESDF 值
- 若 ESDF 值 $< \text{cost\_check\_radius}$，拒绝（碰撞）
- 若 ESDF 值无限（越界），拒绝

---

## 12. 第 ⑨ 步：路径上采样（upsamplePathKinematic）

**目的**：在优化后的离散状态之间插入中间点，使输出路径更密。

**算法**：对每一对相邻状态 $(i, i+1)$，在段内均匀插入 `upsample_factor - 1` 个中间点。

```
for each segment (i, i+1):
    if cusp or gear == 0 or ds <= 1e-6:
        直接复制 next_pose
        continue

    direction = +1 if gear >= 0 else -1
    step = ds / upsample_factor

    # 段内运动学插值
    for j in 1 .. upsample_factor-1:
        t0 = (j-1) / upsample_factor
        t1 = j / upsample_factor
        κ₀ = κ + (κ' - κ) * t0
        κ₁ = κ + (κ' - κ) * t1

        θ_mid = θ_interp + direction * step * 0.5 * κ₀
        x_interp += direction * step * cos(θ_mid)
        y_interp += direction * step * sin(θ_mid)
        θ_interp = normalize(θ_interp + direction * step * 0.5 * (κ₀ + κ₁))

    # 闭合误差校正：将预测端点与实际 next_pose 的偏差均匀摊开
    closure = next_pose - predicted_end
    for j in 1 .. upsample_factor-1:
        t = j / upsample_factor
        output[j] = sample[j-1] + t * closure
```

**注意**：闭合误差校正是因为优化后的状态只在有限权重下逼近运动学一致性，直接用运动学模型前推会产生端点漂移。

---

## 13. 参数参考（SmootherParams）

### 13.1 运动学与参考路径权重

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_weight_sqrt` | double | 0.0 | 运动学一致性残差权重（已取平方根） |
| `kinematic_curvature_weight_sqrt` | double | 0.0 | 曲率大小惩罚权重 |
| `kinematic_curvature_rate_weight_sqrt` | double | 0.0 | 曲率变化率惩罚权重 |
| `kinematic_spacing_weight_sqrt` | double | 1.0 | 步长接近目标间距的正则权重 |
| `path_length_weight_sqrt` | double | 0.0 | 总长度惩罚权重 |
| `reference_path_weight_sqrt` | double | 0.0 | 参考路径吸附权重 |
| `reference_point_max_deviation_m` | double | 0.0 | 每个优化点相对参考点的最大偏移（≤0 关闭） |
| `fix_weight` | double | 100.0 | 起终点边界和 cusp 约束的直接权重（不取平方根） |
| `max_curvature` | double | 0.0 | 最大曲率，单位 $1/m$ |
| `max_time` | double | 10.0 | Ceres 求解最大墙钟时间（秒） |

### 13.2 障碍物与足迹

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `costmap_weight_sqrt` | double | 0.0 | 统一障碍物权重 |
| `obstacle_safe_distance` | double | 0.5 | 最小安全净空（m） |
| `cost_check_radius` | double | 0.0 | 机器人圆形足迹半径（m） |
| `cost_check_points` | vector\<double\> | [] | 局部检测点 $[x, y, w, \ldots]$ |
| `use_exact_esdf` | bool | true | 使用精确 ESDF 后端 |

### 13.3 路径重采样与方向

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path_downsampling_factor` | int | 1 | 输入路径下采样步长 |
| `path_upsampling_factor` | int | 1 | 输出路径上采样倍数 |
| `reversing_enabled` | bool | true | 是否解析路径第三分量的倒车语义 |

### 13.4 起终点约束

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `keep_start_orientation` | bool | true | 固定起点朝向 |
| `keep_goal_orientation` | bool | true | 固定终点朝向 |
| `goal_longitudinal_tolerance` | double | 0.0 | 终点纵向容差（m） |
| `goal_lateral_tolerance` | double | 0.0 | 终点横向容差（m） |
| `goal_orientation_tolerance` | double | 0.0 | 终点朝向容差（rad） |

---

## 14. 求解器配置（OptimizerParams）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug` | bool | false | 是否输出逐迭代日志 |
| `linear_solver` | enum | `SparseNormalCholesky` | Ceres 线性求解器 |
| `max_iterations` | int | 50 | 最大非线性迭代次数 |
| `parameter_tolerance` | double | 1e-8 | 参数步长收敛阈值 |
| `function_tolerance` | double | 1e-6 | 目标函数收敛阈值 |
| `gradient_tolerance` | double | 1e-10 | 梯度收敛阈值 |

---

## 15. 错误处理

### 15.1 错误码（ErrorCode）

| 错误码 | 枚举值 | 字符串 | 含义 |
|--------|--------|--------|------|
| 1001 | `InvalidPath` | `CS_INVALID_PATH` | 输入路径太短 |
| 2001 | `FailedToSmoothPath` | `CS_SMOOTHING_FAILED` | 求解或后验校验失败 |
| 3001 | `InvalidCostmap` | `CS_INVALID_COSTMAP` | 启用障碍项但 costmap 为空 |
| 3002 | `PrecomputedEsdfSizeMismatch` | `CS_PRECOMPUTED_ESDF_SIZE_MISMATCH` | 预计算 ESDF 尺寸不匹配 |

### 15.2 平滑失败原因（SmoothingFailureReason）

| 枚举值 | 字符串 | 触发条件 |
|--------|--------|----------|
| `SolverRejectedSolution` | `solver_rejected_solution` | Ceres 返回不可用解 |
| `NoCostImprovement` | `no_cost_improvement` | 目标函数无改善 |
| `InvalidStateVector` | `invalid_state_vector` | 状态向量维度错误 |
| `NonFiniteState` | `nonfinite_state` | 状态含 NaN/Inf |
| `StartPositionConstraint` | `start_position_constraint` | 起点位置偏移超限 |
| `StartOrientationConstraint` | `start_orientation_constraint` | 起点朝向偏移超限 |
| `GoalPositionConstraint` | `goal_position_constraint` | 终点位置超出容差框 |
| `GoalOrientationConstraint` | `goal_orientation_constraint` | 终点朝向超出容差 |
| `CuspHoldConstraint` | `cusp_hold_constraint` | Cusp 段位置/朝向变化 |
| `CollapsedSegment` | `collapsed_segment` | 非 Cusp 段坍缩为零长度 |
| `MotionDirectionConstraint` | `motion_direction_constraint` | 位移方向与 gear 不一致 |
| `PathOutOfBounds` | `path_out_of_bounds` | 路径点超出地图范围 |
| `FootprintCollision` | `footprint_collision` | 足迹采样点与障碍物碰撞 |
| `CurvatureConstraint` | `curvature_constraint` | 曲率超出最大限制 |

### 15.3 失败传播路径

```
smooth() 调用
  ├─ request.failure == nullptr → 抛异常 (FailedToSmoothPath / InvalidPath / InvalidCostmap)
  └─ request.failure != nullptr → 写入 SmoothingFailureInfo，返回 result.success = false
       └─ result.candidate_path 仍保留诊断候选
```

`SmoothingFailureInfo` 结构：

```cpp
struct SmoothingFailureInfo {
    SmoothingFailureReason reason;
    std::string message;
    int failed_index;                    // 失败状态索引
    double actual_curvature;             // 曲率校验失败时的实测值
    double max_curvature;                // 曲率上限
    double turning_radius;               // 实测转弯半径
    double goal_longitudinal_error;      // 终点纵向误差
    double goal_lateral_error;           // 终点横向误差
    double goal_longitudinal_tolerance;  // 纵向容差
    double goal_lateral_tolerance;       // 横向容差
};
```

---

## 16. 文件结构与分层

### 16.1 C++ 头文件

```
include/constrained_smoother/
├── kinematic_smoother.hpp              # 顶层编排：initialize() + smooth()
├── kinematic_smoother_problem_builder.hpp  # 状态展开 + 问题拼接 + 变量边界 + 解包
├── kinematic_smoother_costs.hpp        # 4 个 cost functor 定义
├── smoother_validator.hpp              # 后验硬校验
├── smoother_request.hpp                # SmootherRequest + SmootherResult
├── options.hpp                         # SmootherParams + OptimizerParams
├── exceptions.hpp                      # 错误码 + 异常类 + SmoothingFailureInfo
├── solver_utils.hpp                    # solveProblemOrReportFailure()
├── utils.hpp                           # normalizeAngle, goalPositionFrameHeading 等
├── costmap2d.hpp                       # 薄 shim → esdf_core::Costmap2D
└── esdf.hpp                            # 薄 shim → esdf_core::ESDF
```

**header-only 设计**：`constrained_smoother` 库本身是 header-only（`INTERFACE` CMake target）。`costmap2d.hpp` 和 `esdf.hpp` 是转发层，实际实现位于独立的 `esdf_core` 库。

### 16.2 Python 包

```
constrained_smoother/
├── __init__.py
├── smoother.py           # KinematicSmoother（镜像 C++ 顶层）
├── problem_builder.py    # 状态展开 + 残差构建 + 变量边界
├── costs.py              # transition_residuals / boundary_residuals / ...
├── validator.py          # 后验校验
├── costmap2d.py          # Costmap2D 类
├── esdf.py               # ESDF 计算
├── options.py            # SmootherParams + OptimizerParams
├── smoother_request.py   # SmootherRequest + SmootherResult
├── exceptions.py         # 异常类 + SmoothingFailureInfo
├── solver_utils.py       # scipy 求解器封装
├── utils.py              # normalize_angle, angle_diff, ...
├── astar_esdf.py         # A* + ESDF 联合规划器
└── tests/
    └── test_core.py      # 单元测试
```

### 16.3 Pybind11 绑定

`pybind/py_constrained_smoother.cpp` 暴露：

- `KinematicSmoother` 类：`initialize()` / `smooth()` / `try_smooth()`
- `SmootherParams` / `OptimizerParams` 结构
- `Costmap2D` 类
- `ErrorCode` / `SmoothingFailureReason` 枚举
- 统一的结构化返回字典格式：

```python
{
    "ok": bool,
    "path": list | None,
    "smoothed_path": list | None,
    "candidate_path": list | None,
    "optimized_knot_count": int,
    "target_spacing_m": float,
    "error_code": str | None,
    "error_message": str | None,
    "error_reason": str | None,
    "error_details": {"failed_index": int, ...} | None,
}
```

---

## 17. 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Eigen3 | ≥ 3.4 | 矩阵/向量运算 |
| Ceres Solver | ≥ 2.0 (含 SuiteSparse) | 非线性最小二乘求解 |
| GTest | — | C++ 单元测试 |
| pybind11 | — | Python 绑定（可选） |
| scipy | — | Python 版求解器 |
| numpy | — | Python 数值运算 |

**构建**：

```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON -DBUILD_TESTS=ON
make -j$(nproc)
```

---

## 18. 关键设计决策与常见误读

### 18.1 设计决策

1. **`*_sqrt` 权重约定**：大多数权重以平方根形式传入，因为残差 $r$ 被 Ceres 平方后 $\frac{1}{2} w^2 r^2$，开方使参数空间更线性，调参更直觉。

2. **`fix_weight` 不取平方根**：边界约束需要强锚定，直接用大数更直观。

3. **双三次插值用于 ESDF**：Ceres AutoDiff 需要可微的 ESDF 查询。双三次插值提供 $C^1$ 连续性和解析梯度。

4. **Header-only 库**：所有模板代码在头文件中，避免链接问题。代价是编译时间较长。

5. **后验校验独立于求解器**：优化器只关心最小化目标函数，硬约束由独立的 validator 执行。这种分离使得校验逻辑可以独立演进。

### 18.2 常见误读

| 误读 | 正确理解 |
|------|----------|
| 只优化 $(x, y)$ | $\theta$, $\kappa$, $ds$ 都是显式优化变量 |
| Cusp 是普通点 | Cusp 是显式插入的停驻过渡状态（`gear=0, ds=0`） |
| `max_curvature` 约束半径 | 约束曲率（$1/m$），不是半径 |
| 成功返回即可用 | 还需通过后验硬校验 |
| `kinematic_max_spacing` 是软约束 | 是 `ds` 的显式硬上界 |
| `path_length_weight_sqrt` 约束总长度 | 直接压缩每段 `ds` |
| 参考路径残差约束朝向 | 只约束 $(x, y)$ 位置 |

---

## 19. 最小可复现伪代码

以下是整个 `smooth()` 的最小完整伪代码：

```python
def smooth(path, start_dir, end_dir, costmap, params, optimizer_params, precomputed_esdf=None):
    # ① 输入校验
    assert len(path) >= 2
    if params.obstacle_terms_enabled():
        assert costmap is not None

    # ② ESDF 准备
    esdf_values = initialize_esdf(costmap, params, precomputed_esdf)

    # ③ 展开状态链
    processed = build_processed_path(path, start_dir, end_dir, params, costmap)
    N = processed.state_count

    # ④ 初始变量
    variables = processed.initial_variables  # length = 5*N

    # ⑤ 构建残差函数
    def residual_fn(X):
        residuals = []
        for i in range(N - 1):
            residuals += transition_residuals(X[i*5:(i+1)*5], X[(i+1)*5:(i+2)*5], ...)
        residuals += boundary_residuals(X[0:5], start_ref, ...)
        residuals += boundary_residuals(X[(N-1)*5:N*5], goal_ref, ...)
        if params.reference_weight > 0:
            for i in range(N):
                residuals += reference_residuals(X[i*5:(i+1)*5], ref_points[i], ...)
        if params.obstacle_terms_enabled():
            for i in range(N):
                residuals += obstacle_residuals(X[i*5:(i+1)*5], esdf_values, costmap, ...)
        return np.array(residuals)

    # ⑥ 变量边界
    lower, upper = compute_bounds(processed, params)

    # ⑦ 求解
    result = least_squares(residual_fn, variables, bounds=(lower, upper), ...)
    variables = result.x

    # ⑧ 后验校验
    candidate = unpack_path(variables, N)
    if not validate_kinematic_solution(variables, processed, costmap, params, esdf_values):
        return SmootherResult(success=False, candidate_path=candidate)

    # ⑨ 上采样
    smoothed = upsample_path_kinematic(variables, processed, params)
    return SmootherResult(success=True, smoothed_path=smoothed, candidate_path=candidate)
```

---

## 20. 附录：角度归一化

所有角度运算使用 `atan2(sin, cos)` 归一化到 $(-\pi, \pi]$：

```python
def normalize_angle(a):
    return atan2(sin(a), cos(a))

def angle_diff(a, b):
    return normalize_angle(a - b)
```

这是 AutoDiff 安全的（处处可微），避免了 `fmod` 的不连续性。
