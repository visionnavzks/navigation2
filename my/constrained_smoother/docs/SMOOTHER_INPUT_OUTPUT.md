# 几何平滑器输入输出说明

本文档专门描述 `my/constrained_smoother` 中几何版 `Smoother` 的系统边界、输入输出语义以及失败返回约定。
如果你要理解内部残差构造、cusp 处理或求解器分层，请回到 `SMOOTHER_DESIGN.md`；如果你主要关心如何正确调用 `Smoother::smooth(...)`，优先阅读本文。

## 核心约定

这个平滑器在不同阶段故意让路径第三个分量承担两种不同语义。

优化之前：

- 每个姿态表示为 `(x, y, direction_sign)`。
- `direction_sign > 0` 表示该局部路径段为前进。
- `direction_sign < 0` 表示该局部路径段为倒车。

优化和重建之后：

- 每个姿态返回为 `(x, y, yaw)`。
- `yaw` 在二维位置优化完成后，根据局部切向方向重建得到。

这是整个模块里最重要的约定之一。很多集成错误都来自把输入路径误认为已经存储了 yaw。

## 系统输入与输出

这一节只描述几何版 `Smoother` 这一个系统边界，不展开 Web 界面或独立 A* 规划器的 HTTP / UI 交互。

### 对外输入

当前主入口是 `Smoother::smooth(...)`，调用层需要提供下面这些输入：

1. `path: std::vector<Eigen::Vector3d>`
    - 原地传入、也原地返回。
    - 输入时每个点必须是 `(x, y, direction_sign)`。
    - 至少要有两个点，否则会被立即拒绝。
2. `start_dir: Eigen::Vector2d`
    - 起点切向方向向量，不是 yaw 角。
    - 当 `keep_start_orientation=true` 时，用它来锚定第二个点的位置。
3. `end_dir: Eigen::Vector2d`
    - 终点切向方向向量，不是 yaw 角。
    - 当 `keep_goal_orientation=true` 时，用它来锚定倒数第二个点的位置。
4. `costmap: const Costmap2D *`
    - 当障碍物相关项启用时，它是障碍物距离场的来源。
    - 当 `costmap_weight_sqrt=0` 且 `cusp_costmap_weight_sqrt=0` 时，可以为 `nullptr`。
    - 生命周期必须覆盖整个 `smooth()` 调用。
5. `params: const SmootherParams &`
    - 提供残差权重、曲率阈值、下采样倍率、上采样倍率、是否保持端点朝向，以及终点 lon/lat 容差等运行参数。
6. `precomputed_esdf: const std::vector<double> *`（可选）
    - 若提供，则直接复用这份扁平化 ESDF。
    - 若为空，则在启用障碍物项时内部根据 `costmap` 现场生成。
    - 若障碍物项关闭，这份输入会被忽略。
    - 若尺寸与 `costmap` 不匹配，会抛出 `PrecomputedEsdfSizeMismatch`。
7. `failure: SmoothingFailureInfo *`（可选）
    - 仅用于“失败走普通返回值而不是异常”的调用模式。
    - 为空时，求解失败和后验校验失败通常通过异常传播。

从实现角度看，上述输入在入口处会被折叠为一个 `SmootherRequest`，供单次执行对象 `Smoother::Run` 在整个生命周期内共享。

### 输入语义和所有权约束

- `path` 是唯一会被原地改写的输入缓冲区。
- `start_dir` 和 `end_dir` 始终按几何切向量解释，不应传 yaw 标量的 `cos/sin` 之外的任何编码。
- `costmap` 和 `precomputed_esdf` 都是借用视图，`Smoother` 不拥有它们的生命周期。
- 若障碍物项关闭，则允许 `costmap == nullptr`，同时 `precomputed_esdf` 也不会被消费。
- `params` 中多数权重使用平方根形式，因为残差会先缩放、再由 Ceres 在目标函数中平方。
- 即使 `reversing_enabled=false` 是保留字段，几何版这条代码路径仍然按输入路径现有的方向符号工作。

### 内部中间产物

虽然对外接口只暴露一次 `smooth()` 调用，但系统内部会在三个阶段之间传递几类关键中间量：

1. 参考路径快照 `reference_path_`
    - 保存原始输入几何，供求解后校验边界约束时对照使用。
2. 工作路径 `path_optim_`
    - 由输入路径复制而来。
    - 如果启用端点朝向约束，第二个点和倒数第二个点会先被重定位。
3. 激活掩码 `optimized_`
    - 标记哪些点在下采样、cusp 保留和冻结规则之后仍参与求解。
4. `ceres::Problem`
    - 承载三点平滑残差、可选四点曲率变化率残差、障碍物净空残差、可选的终点位置带宽残差和边界冻结约束。
5. 扁平化 ESDF 缓存 `esdf_values_`
    - 当障碍物项启用时，同时服务于优化阶段的障碍物残差与求解后的净空校验。

这些中间量解释了为什么系统的“输入路径”和“最终输出路径”之间并不是一一对应的原样求解，而是会经过“锚定、下采样求解、上采样重建”三段变换。

### 成功输出

当 `smooth()` 返回 `true` 时，对外可观察到的输出有四类：

1. `path` 被原地改写为输出路径。
    - 每个点现在表示 `(x, y, yaw)`。
    - `yaw` 是重建值，不是优化阶段直接求解的状态。
2. 路径点数量可能变化。
    - 当 `path_upsampling_factor > 1` 时，系统会在关键控制点之间补插中间点。
3. 起点位置保持固定，终点位置则分两种情况。
    - 若 `goal_longitudinal_tolerance=0` 且 `goal_lateral_tolerance=0`，终点位置仍然固定。
    - 若配置了终点 lon/lat 容差，终点位置只需落在允许带宽内。
    - 若启用了端点朝向保持，对应切向约束也应在输出上继续成立。
4. `getLastOptimizedKnotCount()` 可返回最近一次真正参与求解的控制点数量。
    - 这个值反映的是下采样后的优化控制点数，不是最终输出路径长度。

有一个容易忽略的边界情况：如果输入路径虽然合法，但所有内部点都被跳过，或者构建后没有可优化自由度，系统可能不会真正调用 Ceres，但仍会返回 `true`，并基于锚定后的几何关系重建输出路径。

### 失败输出

失败分成两类，输出面不同：

1. 输入前置条件失败。
    - 典型情况包括路径过短、在启用障碍物项时 `costmap` 无效、预计算 ESDF 尺寸不匹配。
    - 这类错误直接抛结构化异常，如 `InvalidPath`、`InvalidCostmap`、`PrecomputedEsdfSizeMismatch`。
2. 求解失败或后验校验失败。
    - 如果 `failure == nullptr`，会抛 `FailedToSmoothPath`。
    - 如果传入非空 `failure`，则返回 `false`，同时写入 `SmoothingFailureInfo`。

`SmoothingFailureInfo` 当前会回传这些稳定字段：

- `reason`：枚举化失败原因，例如 `SolverRejectedSolution`、`NoCostImprovement`、`FootprintCollision`、`CurvatureConstraint`。
- `message`：给日志和调试界面看的说明文本。
- `failed_index`：如果失败可定位到路径中的某个点，则记录该索引。
- `actual_curvature`、`max_curvature`、`turning_radius`：供曲率相关失败诊断使用。

因此，从系统接口角度可以把输出面概括成下面这个判定表：

| 情况 | 返回值 | `path` | `failure` | 异常 |
| --- | --- | --- | --- | --- |
| 输入非法 | 无 | 不保证可用 | 不写入 | 抛输入异常 |
| 求解/校验失败，且未传 `failure` | 无 | 不保证可用 | 无 | 抛 `FailedToSmoothPath` |
| 求解/校验失败，且传入 `failure` | `false` | 不保证可用 | 写入失败详情 | 不抛 |
| 成功 | `true` | 改写为 `(x, y, yaw)` | 保持调用方自有策略 | 不抛 |

### 最小调用心智模型

如果只保留最关键的输入输出关系，可以把几何版 smoother 理解成：

```text
输入:
  path(x, y, direction_sign)
    + start_dir / end_dir
    + optional costmap
    + params
    + optional precomputed_esdf

处理:
  锚定边界
  -> 下采样并保留 cusp
  -> 构建残差问题
  -> 求解二维位置
  -> 重建 yaw 与插值点
  -> 后验校验

输出:
  success => bool true + path(x, y, yaw)
  failure => exception 或 bool false + SmoothingFailureInfo
```