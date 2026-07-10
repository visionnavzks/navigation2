# 运动学平滑器设计说明

本文档说明 `my/kinematic_smoother` 中 `KinematicSmoother` 的实现思路，覆盖当前保留的 C++ 版本 `include/kinematic_smoother/kinematic_smoother.hpp`。

当前 C++ 版本已经拆成"顶层编排 + 共享运行骨架 + 运动学问题构建器 + cost functor"几层，不再把状态展开、问题拼接和执行主线都塞在一个头文件里。

文中流程顺序与代码里新增的"第 1 步 / 第 2 步"注释保持一致，适合一边看文档一边顺着实现读。

## 这版 smoother 在做什么

`KinematicSmoother` 把每个离散状态显式写成：

- `x`：位置横坐标
- `y`：位置纵坐标
- `theta`：航向角
- `kappa`：曲率
- `ds`：到下一个状态的弧长步长

因此它不只是"把点拉平"，而是在相邻状态之间施加离散运动学一致性约束。

## 核心输入输出约定

对外接口仍然使用路径形式：

- 输入路径的第三个分量仍沿用方向语义。
- C++ 版本从 `path[i].z()` 推断每一段是前进还是倒车。

但在内部，优化器实际处理的是一条展平后的状态链：

- 每个状态是 `(x, y, theta, kappa, ds)`。
- 所有状态按固定顺序展开成一维变量数组，交给 Ceres 最小二乘求解器。

求解完成后：

- 对外只恢复 `(x, y, yaw)`。
- 内部的 `kappa` 和 `ds` 不直接作为公共输出返回。

## 总体流程

`KinematicSmoother::smooth(...)` 当前遵循同一条主线：

1. 校验输入合法性。
    - 至少要有起点和终点。
2. 准备 ESDF 或障碍物上下文。
    - C++ 版会构建或接收预计算 ESDF，供障碍物残差和后验校验共用。
3. 把原始路径展开成运动学状态链。
    - 遇到换向时插入 cusp 停驻状态。
4. 用参考几何初始化 `(x, y, theta, kappa, ds)`。
    - 这是整个非线性求解的初值来源。
5. 构建残差问题。
    - 包括过渡残差、边界残差、参考路径残差，以及障碍物残差。
6. 施加显式变量边界。
    - 主要是曲率上下界、非负弧长约束和最大步长上界。
7. 交给求解器优化。
    - 当前版本使用 Ceres。
8. 执行后验校验。
    - C++ 版会检查有限值、边界约束、换向一致性、cusp 停驻行为、曲率约束和障碍物净空。
9. 组装 `SmootherResult`。
    - `candidate_path` 保存解包后的候选路径。
    - `smoothed_path` 保存通过后验校验后的最终 `(x, y, yaw)` 输出。

## 当前 C++ 分层

当前运动学版实现建议按下面的对象边界来理解：

1. 顶层对象：`include/kinematic_smoother/kinematic_smoother.hpp`
    - 对外暴露 `initialize()` 和 `smooth()`。
    - 持有长期状态，比如 ESDF 缓存、validator 和求解器配置基线。
2. 单次执行对象：`KinematicSmoother::Run`
    - 表示一次 `smooth()` 调用的生命周期。
    - 负责驱动"准备 -> 求解 -> 校验 -> 回写输出"。
3. 问题构建器：`include/kinematic_smoother/kinematic_smoother_problem_builder.hpp`
    - 负责 ESDF 准备、状态展开、变量初值生成、残差拼接、显式边界约束和输出解包。
4. 残差定义：`include/kinematic_smoother/kinematic_smoother_costs.hpp`
    - 定义过渡（7 残差）、边界（4 残差）、参考路径（2 残差）和障碍物（动态残差）各类 cost functor。

共享层还包括：

- `include/kinematic_smoother/smoother_request.hpp`
    - 统一单次调用请求结构。
- `include/kinematic_smoother/options.hpp`
    - 求解器参数和运行时配置。
- `include/kinematic_smoother/exceptions.hpp`
    - 稳定错误码、失败原因枚举和结构化失败信息。

## Web 层如何驱动它

当前 `web/app.py` 不再把 `/api/plan` 的输入拆成大量独立局部变量，而是先收束成 `PlanRequestConfig`：

- `PlanRequestConfig.from_payload()` 负责一次性解析和归一化前端参数。
- `build_footprint_model()` 负责生成 planner / smoother 共用的检查点与半径模型。
- `build_smoother_params()` 负责把 Web 层权重转换成 pybind `SmootherParams`。
- `build_optimizer_params()` 负责把 Web 层求解器参数转换成 pybind `OptimizerParams`。

这样 `/api/plan` 主流程就只保留四件事：

1. 构造请求配置。
2. 运行 planner stage。
3. 运行 `KinematicSmoother`。
4. 做矩形足迹后验校验并组装响应。

这层设计的目标不是引入额外抽象，而是把"请求解析"和"算法执行"分开，让接口层修改不会污染求解主线。

## 第一个关键点：为什么要插入 cusp 状态

换向点不能简单地当成一个普通路径点，因为它在运动学上意味着"前一个段"和"后一个段"的档位方向不同。

实现上会在换向处额外插入一个 cusp 停驻状态：

- 位置与换向点相同。
- 对应的 gear 记为 `0.0`。
- 该段会被标记为 `is_cusp_segment = true`。

这样做的目的有两个：

1. 显式表达"这里需要停一下再换向"。
2. 避免把前进段和倒车段直接连成一个连续运动学过渡，从而产生不合理的预测状态。

## 第二个关键点：状态是怎么初始化的

在状态展开后，求解器并不会凭空从零开始猜，而是先利用参考几何构造一个可行初值：

1. `x` 和 `y` 直接来自展开后的参考点。
2. `theta` 来自相邻参考点的几何朝向。
    - 如果该段是倒车，则会额外加上 `pi`，使朝向与运动方向一致。
3. `kappa` 初始为零。
4. `ds` 初始为相邻参考点的欧式距离。
    - cusp 段的 `ds` 会直接置零。

这一步对求解稳定性很重要，因为运动学残差是强非线性的。

## 第三个关键点：残差是怎么拼起来的

### 1. 过渡残差（TransitionCostFunctor，7 个残差）

这是核心残差，约束相邻两个状态满足离散运动学模型。

对于普通段，运动学模型采用梯形曲率积分预测朝向，再用 Euler midpoint 近似预测位置：

- `[0]` `x` 位置误差（预测与实际偏差，乘 `model_weight`）
- `[1]` `y` 位置误差（预测与实际偏差，乘 `model_weight`）
- `[2]` `theta` 朝向误差（预测与实际偏差，乘 `model_weight`）
- `[3]` 平均曲率惩罚（鼓励路径趋向直行，乘 `curvature_weight`）
- `[4]` 曲率变化率惩罚（以弧长平方根归一化，乘 `curvature_rate_weight`）
- `[5]` 相邻有效步长差分（约束 `ds_i` 接近 `ds_{i+1}`，归一化后无量纲，乘 `spacing_weight`）
- `[6]` 密度归一化长度惩罚（残差为 `ds / sqrt(ds_ref)`，使代价不随结点密度变化，乘 `length_weight`）

对于 cusp 段：

- `[0]` `[1]` `[2]` 强约束位置和朝向不变（乘 `fix_weight`）。
- `[5]` 强惩罚非零步长（乘 `fix_weight`）。cusp 及其相邻段不参与均匀间距差分。
- `[6]` 按 `ds_ref` 归一化后压缩长度（乘 `length_weight`）。

涉及的权重参数：

| 残差 | 对应 `SmootherParams` 字段 |
| --- | --- |
| 模型约束 [0-2] | `model_weight` |
| 曲率 [3] | `kinematic_curvature_weight` |
| 曲率变化率 [4] | `kinematic_curvature_rate_weight` |
| 步长 [5] | `kinematic_spacing_weight` |
| 长度 [6] | `path_length_weight` |

### 2. 边界残差（BoundaryCostFunctor，3 个残差）

边界残差负责锚定起点和终点，输出 3 个残差：

- `[0]` 目标坐标系 lon 方向位置误差（超出 `goal_longitudinal_tolerance` 才惩罚）
- `[1]` 目标坐标系 lat 方向位置误差（超出 `goal_lateral_tolerance` 才惩罚）
- `[2]` 朝向误差（仅在 `keep_start_orientation` / `keep_goal_orientation` 为 true 且超出 `goal_orientation_tolerance` 时惩罚）

起点和终点都使用同一个 functor 实例化，参数不同。

容差参数为 0 时退化为绝对硬锚定；大于 0 时表达"范围停"语义。

### 3. 参考路径残差（ReferenceCostFunctor，2 个残差）

如果启用参考路径权重，就会额外把解轻柔地拉回原始几何：

- `[0]` `x` 方向偏差
- `[1]` `y` 方向偏差

这个残差只约束平面位置，不直接约束 `theta`、`kappa` 或 `ds`。

权重由 `reference_path_weight` 控制。此外，`reference_point_max_deviation_m` 可以为每个优化点设置最大偏移半径（通过显式参数边界实现），超过该半径的参考点吸引力不再有效——这是一种"硬空间约束"。

### 4. 障碍物残差（ObstacleCostFunctor，动态残差）

C++ 版本会为每个状态连接一个障碍物净空残差：

- 如果没有提供足迹采样点，就以状态中心为检查点，输出 1 个残差。
- 如果提供了 `cost_check_points`，就把局部采样点旋转到世界坐标后逐点检查，每个三元组 `(x_local, y_local, weight)` 输出 1 个残差。
- 所有状态使用统一的 `obstacle_weight` 障碍物权重。

惩罚模型为二次惩罚：当到障碍物表面的距离低于 `obstacle_safe_distance` 时才生效，越近惩罚越大。

## 第四个关键点：变量边界为什么单独施加

曲率和步长的约束不是通过软残差实现的，而是显式参数边界：

- `kappa` 被限制在 `[-max_curvature, max_curvature]`。
- `ds` 被限制为非负。
- 当 `kinematic_max_spacing > 0` 时，`ds` 还会被限制不超过该上界。

这样做的好处是：

- 避免求解器在明显不合理的区域里浪费迭代。
- 把"绝对不能越界"的约束交给优化器底层边界机制处理。

## 第五个关键点：后验校验为什么还要保留

即便求解器返回成功，也不代表结果一定可用。

当前 C++ 版会在优化完成后继续做硬性校验，按顺序检查：

1. **状态向量形状和有限值**：维度是否正确，每个 `(x, y, theta, kappa, ds)` 是否有限。
2. **边界约束**：起点位置/朝向是否保持固定；终点位置是否在 lon/lat 容差框内，朝向是否满足容差。
3. **段一致性**：
    - cusp 段是否真的保持停驻（位置和朝向不变）。
    - 非 cusp 段是否没有坍缩到零长度。
    - 各段位移方向是否与 gear 一致。
4. **曲率约束**：
    - 显式 `kappa` 是否超出 `max_curvature`。
    - 相邻姿态形成的几何曲率是否超出 `max_curvature`（覆盖 kappa 合法但输出轨迹几何超限的情形）。
5. **障碍物净空**：足迹采样点是否与障碍物冲突或越界。

曲率校验失败时会额外回传 `actual_curvature`、`max_curvature` 和 `turning_radius`；终点位置校验失败时会额外回传 `goal_longitudinal_error`、`goal_lateral_error` 和对应容差。

这一步是为了把"数值上能收敛"和"工程上能使用"明确区分开。

## 按代码阅读的顺序

如果你准备直接看 C++ 实现，推荐按下面顺序读：

1. `smooth(...)` in `kinematic_smoother.hpp`
    - 先建立总流程视角。
2. `KinematicSmoother::Run`
    - 理解单次调用如何持有请求、驱动准备、求解和结果回写。
3. `KinematicSmootherProblemBuilder::buildProcessedPath(...)`
    - 理解状态链是怎么从原始路径展开出来的。
4. `KinematicSmootherProblemBuilder::buildProblem(...)`
    - 理解各类残差是怎么进入求解问题的。
5. `KinematicSmootherProblemBuilder::applyBounds(...)`
    - 理解哪些约束是边界，而不是软残差。
6. `KinematicSmootherProblemBuilder::unpackPath(...)`
    - 理解内部状态是如何恢复成对外输出路径的。

如果你准备连同 Web 接口一起看，建议在这之后再读：

7. `PlanRequestConfig` in `web/app.py`
8. `/api/plan` in `web/app.py`

## 最短阅读路径

如果你只想先把 C++ 版的整体骨架看明白，建议按下面顺序读：

1. `include/kinematic_smoother/kinematic_smoother.hpp`
    - 先看类注释、`smooth(...)` 入口和内部 `Run` 的三阶段生命周期。
2. `include/kinematic_smoother/smoother_request.hpp`
    - 搞清楚单次调用上下文里哪些字段是输入、哪些会被原地修改。
3. `include/kinematic_smoother/options.hpp`
    - 理解所有权重、约束和求解器配置的含义。
4. `include/kinematic_smoother/kinematic_smoother_problem_builder.hpp`
    - 理解 ESDF 准备、状态展开、问题拼接、边界约束和输出解包。
5. `include/kinematic_smoother/kinematic_smoother_costs.hpp`
    - 再回头看过渡、边界、参考路径和障碍物残差各自编码了什么。
6. `include/kinematic_smoother/smoother_validator.hpp`
    - 最后确认哪些运动学约束只是优化偏好，哪些会在求解后被硬性拒绝。

如果你接下来要改失败处理或对外错误语义，再补读 [错误码参考](error-codes.md) 和 [包使用指南](package-guide.md) 里的"失败传播路径"小节。

## 常见误读

最容易误读的地方通常有这些：

- 这版 smoother 不是只优化 `(x, y)`。
- `theta`、`kappa`、`ds` 都是显式优化变量。
- cusp 不是普通点，而是一个显式插入的停驻过渡状态。
- `max_curvature` 约束的是曲率（`1/m`），不是半径。
- C++ 版成功返回前还有一步独立的后验校验。
- `kinematic_max_spacing` 是 `ds` 的硬上界，不是软约束。
- `path_length_weight` 按 `ds_ref` 归一化后压缩每段 `ds`；在间距近似均匀时，其平方和等价于总弧长代价。
- `reference_point_max_deviation_m` 是显式参数边界，不是参考路径残差的一部分。

## 常见误改点

如果你准备改实现，最容易把行为改坏的地方通常有这些：

- 改 `buildProcessedPath(...)` 时只顾状态维度，不同步 `state_count`、`gears`、`is_cusp_segment` 的索引契约。
    - 这三个量是一整套展开语义，错一处通常会连锁破坏求解和后验校验。
- 在 `buildProblem(...)` 里改残差顺序或边界逻辑时，忘记它和 `applyBounds(...)`、validator 是配套设计。
    - 软残差、显式边界和后验拒绝各自承担不同职责，不建议混写。
- 把 cusp 段当成普通运动段处理。
    - cusp 段的 `gear == 0`，语义是停驻过渡，不是普通短段。
- 修改 pybind 返回格式时，不同步 [错误码参考](error-codes.md) 的结构化错误约定。
    - 这类漂移最容易让 web 和 notebook 调用层出现"能跑但语义对不上"的问题。
- 添加新的 `SmootherParams` 字段时，不同步 `build_smoother_params()` 和 `PlanRequestConfig`。
    - Web 层参数映射和 C++ 参数是一一对应的，缺一就会出现"前端配了但没传进去"。

## 建议结合阅读的文件

如果你要修改这部分实现，建议一起阅读：

- `include/kinematic_smoother/kinematic_smoother.hpp`
- `include/kinematic_smoother/kinematic_smoother_problem_builder.hpp`
- `include/kinematic_smoother/kinematic_smoother_costs.hpp`
- `include/kinematic_smoother/smoother_validator.hpp`
- `include/kinematic_smoother/options.hpp`
- `include/kinematic_smoother/exceptions.hpp`

这样可以同时看到：

- 后验校验如何把求解结果转化为可交付行为。
- 结构化失败信息如何从校验器传播到调用层。
