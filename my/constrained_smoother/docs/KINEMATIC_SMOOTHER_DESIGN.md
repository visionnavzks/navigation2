# 运动学平滑器设计说明

本文档说明 `my/constrained_smoother` 中 `KinematicSmoother` 的实现思路，覆盖 C++ 版本 `include/constrained_smoother/kinematic_smoother.hpp` 和 Python 版本 `include/constrained_smoother/kinematic_smoother.py`。

当前 C++ 版本已经拆成“顶层编排 + 共享运行骨架 + 运动学问题构建器 + cost functor”几层，不再把状态展开、问题拼接和执行主线都塞在一个头文件里。

文中流程顺序与代码里新增的“第 1 步 / 第 2 步”注释保持一致，适合一边看文档一边顺着实现读。

## 这版 smoother 在做什么

几何版 `Smoother` 直接优化路径点位置，再从局部切向恢复 yaw。

`KinematicSmoother` 更进一步，把每个离散状态显式写成：

- `x`：位置横坐标
- `y`：位置纵坐标
- `theta`：航向角
- `kappa`：曲率
- `ds`：到下一个状态的弧长步长

因此它不只是“把点拉平”，而是在相邻状态之间施加离散运动学一致性约束。

## 核心输入输出约定

对外接口仍然使用路径形式：

- 输入路径的第三个分量仍沿用方向语义。
- C++ 版本从 `path[i].z()` 推断每一段是前进还是倒车。
- Python 版本允许显式传入 `gear_directions`。

但在内部，优化器实际处理的是一条展平后的状态链：

- 每个状态是 `(x, y, theta, kappa, ds)`。
- 所有状态按固定顺序展开成一维变量数组，交给 Ceres 或 SciPy 最小二乘求解器。

求解完成后：

- 对外只恢复 `(x, y, yaw)`。
- 内部的 `kappa` 和 `ds` 不直接作为公共输出返回。

## 总体流程

`KinematicSmoother::smooth(...)` 与 Python 版 `_optimize_impl(...)` 都遵循同一条主线：

1. 校验输入合法性。
    - 至少要有起点和终点。
    - Python 版还会校验 `raw_path` 形状和 `gear_directions` 长度。
2. 准备 ESDF 或障碍物上下文。
    - C++ 版会构建或接收预计算 ESDF，供障碍物残差和后验校验共用。
    - Python 版当前不带 ESDF 障碍物项，主要聚焦运动学残差本身。
3. 把原始路径展开成运动学状态链。
    - 遇到换向时插入 cusp 停驻状态。
4. 用参考几何初始化 `(x, y, theta, kappa, ds)`。
    - 这是整个非线性求解的初值来源。
5. 构建残差问题。
    - 包括过渡残差、边界残差、参考路径残差，以及 C++ 版本中的障碍物残差。
6. 施加显式变量边界。
    - 主要是曲率上下界和非负弧长约束。
7. 交给求解器优化。
    - C++ 版使用 Ceres。
    - Python 版使用 `scipy.optimize.least_squares`。
8. 执行后验校验。
    - C++ 版会检查有限值、边界约束、换向一致性、cusp 停驻行为和障碍物净空。
9. 回写公共输出路径。
    - 只保留 `(x, y, yaw)`。

## 当前 C++ 分层

当前运动学版实现建议按下面的对象边界来理解：

1. 顶层对象：`include/constrained_smoother/kinematic_smoother.hpp`
    - 对外暴露 `initialize()`、`smooth()` 和 `getLastOptimizedKnotCount()`。
    - 持有长期状态，比如 ESDF 缓存、validator 和最近一次优化状态数。
2. 单次执行对象：`KinematicSmoother::Run`
    - 表示一次 `smooth()` 调用的生命周期。
    - 负责驱动“准备 -> 求解 -> 校验 -> 回写输出”。
3. 问题构建器：`include/constrained_smoother/kinematic_smoother_problem_builder.hpp`
    - 负责 ESDF 准备、状态展开、变量初值生成、残差拼接、显式边界约束和输出解包。
4. 残差定义：`include/constrained_smoother/kinematic_smoother_costs.hpp`
    - 定义过渡、边界、参考路径和障碍物各类 cost functor。

共享层还包括：

- `include/constrained_smoother/smoother_base.hpp`
    - 统一 solver 配置、调试状态和公共输入校验。
- `include/constrained_smoother/smoother_request.hpp`
    - 统一单次调用请求结构。
- `include/constrained_smoother/smoother_run_base.hpp`
    - 统一 `prepare -> solve -> finalize` 的执行骨架。

## 第一个关键点：为什么要插入 cusp 状态

换向点不能简单地当成一个普通路径点，因为它在运动学上意味着“前一个段”和“后一个段”的档位方向不同。

实现上会在换向处额外插入一个 cusp 停驻状态：

- 位置与换向点相同。
- 对应的 gear 记为 `0.0`。
- 该段会被标记为 `is_cusp_segment = true`。

这样做的目的有两个：

1. 显式表达“这里需要停一下再换向”。
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

### 1. 过渡残差

这是核心残差，约束相邻两个状态满足离散运动学模型。

对于普通段：

- 用当前状态的 `theta`、`kappa`、`ds` 预测下一个状态。
- 惩罚预测状态与真实优化变量之间的偏差。
- 同时惩罚曲率变化和步长偏离目标值。

对于 cusp 段：

- 不再使用常规运动学推进。
- 改为强约束“位置不变、角度不突变、步长接近 0”。

### 2. 边界残差

边界残差负责：

- 固定起点和终点位置。
- 按配置固定起点和终点朝向。
- 在终点压制最后一个虚拟步长。

### 3. 参考路径残差

如果启用参考路径权重，就会额外把解轻柔地拉回原始几何。

这个残差只约束平面位置，不直接约束 `theta`、`kappa` 或 `ds`。

### 4. 障碍物残差（C++ 版）

C++ 版本会为每个状态连接一个障碍物净空残差：

- 如果没有提供足迹采样点，就以状态中心为检查点。
- 如果提供了 `cost_check_points`，就把局部采样点旋转到世界坐标后逐点检查。
- cusp 前后会自动使用更高的障碍物权重。

## 第四个关键点：变量边界为什么单独施加

曲率和步长的约束不是通过软残差实现的，而是显式参数边界：

- `kappa` 被限制在 `[-max_curvature, max_curvature]`。
- `ds` 被限制为非负。

这样做的好处是：

- 避免求解器在明显不合理的区域里浪费迭代。
- 把“绝对不能越界”的约束交给优化器底层边界机制处理。

## 第五个关键点：后验校验为什么还要保留

即便求解器返回成功，也不代表结果一定可用。

当前 C++ 版会在优化完成后继续做硬性校验，检查：

- 状态值是否有限。
- 起终点位置和朝向是否保持固定。
- 各段 gear 与位姿变化是否一致。
- cusp 段是否真的保持停驻行为。
- 足迹采样点是否与障碍物冲突或越界。

这一步是为了把“数值上能收敛”和“工程上能使用”明确区分开。

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

如果你准备先看 Python 实现，推荐顺序是：

1. `_optimize_impl(...)`
2. `_residuals(...)`
3. `_kinematic_residuals(...)`
4. `_smoothness_residuals(...)`
5. `_spacing_residuals(...)`
6. `_boundary_residuals(...)`

Python 版更短，更适合先建立直觉；C++ 版则包含更完整的工程化约束和后验校验。

## 最短阅读路径

如果你只想先把 C++ 版的整体骨架看明白，建议按下面顺序读：

1. `include/constrained_smoother/kinematic_smoother.hpp`
    - 先看类注释、`smooth(...)` 入口和内部 `Run` 的三阶段生命周期。
2. `include/constrained_smoother/smoother_request.hpp`
    - 搞清楚单次调用上下文里哪些字段是输入、哪些会被原地修改。
3. `include/constrained_smoother/kinematic_smoother_problem_builder.hpp`
    - 理解 ESDF 准备、状态展开、问题拼接、边界约束和输出解包。
4. `include/constrained_smoother/kinematic_smoother_costs.hpp`
    - 再回头看过渡、边界、参考路径和障碍物残差各自编码了什么。
5. `include/constrained_smoother/smoother_validator.hpp`
    - 最后确认哪些运动学约束只是优化偏好，哪些会在求解后被硬性拒绝。

如果你接下来要改失败处理或对外错误语义，再补读 [Error Codes](error-codes.md) 和 [Package Guide](package-guide.md) 里的“失败传播路径”小节。

## 常见误读

最容易误读的地方通常有这些：

- 这版 smoother 不是只优化 `(x, y)`。
- `theta`、`kappa`、`ds` 都是显式优化变量。
- cusp 不是普通点，而是一个显式插入的停驻过渡状态。
- `max_curvature` 约束的是曲率，不是半径。
- C++ 版成功返回前还有一步独立的后验校验。

## 常见误改点

如果你准备改实现，最容易把行为改坏的地方通常有这些：

- 改 `buildProcessedPath(...)` 时只顾状态维度，不同步 `state_count`、`gears`、`is_cusp_segment` 的索引契约。
    - 这三个量是一整套展开语义，错一处通常会连锁破坏求解和后验校验。
- 在 `buildProblem(...)` 里改残差顺序或边界逻辑时，忘记它和 `applyBounds(...)`、validator 是配套设计。
    - 软残差、显式边界和后验拒绝各自承担不同职责，不建议混写。
- 把 cusp 段当成普通运动段处理。
    - cusp 段的 `gear == 0`，语义是停驻过渡，不是普通短段。
- 修改 Python 原型或 pybind 返回格式时，不同步 README / [Error Codes](error-codes.md) 的结构化错误约定。
    - 这类漂移最容易让 web 和 notebook 调用层出现“能跑但语义对不上”的问题。

## 建议结合阅读的文件

如果你要修改这部分实现，建议一起阅读：

- `include/constrained_smoother/kinematic_smoother.hpp`
- `include/constrained_smoother/kinematic_smoother_problem_builder.hpp`
- `include/constrained_smoother/kinematic_smoother_costs.hpp`
- `include/constrained_smoother/kinematic_smoother.py`
- `include/constrained_smoother/smoother_validator.hpp`
- `include/constrained_smoother/options.hpp`
- `docs/SMOOTHER_DESIGN.md`

这样可以同时看到：

- 运动学版和几何版的建模差异。
- Python 原型和 C++ 工程实现的对应关系。
- 后验校验如何把求解结果转化为可交付行为。