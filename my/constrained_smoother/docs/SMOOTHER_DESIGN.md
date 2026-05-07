# 几何平滑器设计说明

本文档说明 `my/constrained_smoother` 中几何版约束平滑器的内部架构，面向需要修改实现而不仅仅是调用 API 的读者。

## 范围

独立版包中包含三个相关子系统：

- 基于 Ceres 的 C++ 几何约束平滑器。
- 由规划器与平滑器共享的 ESDF 生成工具。
- 用于可视化路径结果的小型规划器和 Web 演示界面。

本文聚焦于几何版 smoother 当前的 C++ 分层实现。
现在顶层编排位于 `include/constrained_smoother/smoother.hpp`，而路径准备与重建、问题构建、共享执行骨架分别被拆到独立头文件中。
文中流程顺序与代码里新增的“第 1 步 / 第 2 步”注释保持一致，方便你在文档和实现之间来回对照。

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

## 高层流程

`Smoother::smooth(...)` 会依次执行以下阶段：

1. 校验输入形状。
   - 少于两个点的路径会立即被拒绝。
2. 保存参考路径快照。
   - 原始路径会被保留，供后续后验校验固定边界位置时使用。
3. 构建 ESDF 插值器。
   - 要么复用调用方传入的扁平化 ESDF，要么从 costmap 现算。
4. 准备优化路径。
   - 复制输入路径。
   - 根据需要重新定位第二个点和倒数第二个点，以施加端点朝向锚定。
5. 单次遍历路径并添加残差块。
   - 对内部点做下采样。
   - 即使正常下采样会跳过，也必须保留 cusp。
   - 添加主三点平滑残差。
   - 当局部运动方向一致时，添加可选四点曲率变化率残差。
6. 冻结端点锚定。
   - 起点和终点位置始终固定。
   - 如果启用了切向约束，对应锚点在所有残差连接完成后也会被冻结。
7. 使用 Ceres 求解。
   - 如果求解结果不可用，或目标函数没有改进，则直接判为失败。
8. 重建对外输出路径。
   - 对被跳过的路径段使用三次 Bezier 插值补点。
   - 根据局部切向方向重新计算 yaw。
9. 执行后验校验。
   - 检查输出是否有限。
   - 检查边界约束。
   - 检查曲率限制。
   - 检查是否满足和优化阶段一致的 ESDF 净空要求。

## 当前代码分层

当前几何版实现已经不再是“一个头文件包办全部细节”，而是分成四层：

1. 顶层对象：`include/constrained_smoother/smoother.hpp`
   - 对外暴露 `initialize()`、`smooth()` 和 `getLastOptimizedKnotCount()`。
   - 持有长期状态，比如 ESDF 缓存、validator 和最近一次优化点数。
2. 单次执行对象：`Smoother::Run`
   - 表示一次 `smooth()` 调用的生命周期。
   - 持有本次请求、参考路径快照、工作路径、副本问题对象和优化标记。
3. 路径侧 helper：`include/constrained_smoother/smoother_path_ops.hpp`
   - 负责端点朝向锚定、上采样和 yaw 重建。
4. 问题构建 helper：`include/constrained_smoother/smoother_problem_builder.hpp`
   - 负责 ESDF 准备、主残差连接、cusp 邻域重赋权和边界冻结。

共享层还包括：

- `include/constrained_smoother/smoother_base.hpp`
  - 统一 solver 配置、调试状态和公共输入校验。
- `include/constrained_smoother/smoother_request.hpp`
  - 统一单次调用请求结构。
- `include/constrained_smoother/smoother_run_base.hpp`
  - 统一 `prepare -> solve -> finalize` 执行骨架。

## 按代码阅读的主线

如果你是从 `smoother.hpp` 直接往下读，建议按下面顺序理解：

1. `smooth(...)` in `smoother.hpp`
   - 这是总入口，负责把请求对象交给内部 `Run`。
2. `Smoother::Run`
   - 这是单次执行对象，负责把“路径准备、问题构建、求解、路径重建、后验校验”串成一条主线。
3. `SmootherPathOps`
   - 这里处理参考路径副本、端点朝向锚定和输出重建。
4. `SmootherProblemBuilder`
   - 这里处理 ESDF 初始化、残差连接、cusp 邻域回溯重赋权和边界冻结。
5. `SmootherValidator`
   - 这里统一做求解后的硬性校验。

如果你先建立这条主线，再回头看具体 cost functor，会更容易理解每一段代码为什么出现在那个位置。

## 最短阅读路径

如果你只想用最少跳转快速建立实现心智模型，建议按下面顺序读：

1. `include/constrained_smoother/smoother.hpp`
   - 先看类注释、`smooth(...)` 入口和内部 `Run` 的三阶段生命周期。
2. `include/constrained_smoother/smoother_request.hpp`
   - 搞清楚单次调用上下文里哪些字段是输入、哪些会被原地修改。
3. `include/constrained_smoother/smoother_path_ops.hpp`
   - 理解端点锚定、关键点链和输出 yaw 是怎么重建出来的。
4. `include/constrained_smoother/smoother_problem_builder.hpp`
   - 理解 ESDF、下采样、cusp 重赋权和残差连接是怎么进入求解问题的。
5. `include/constrained_smoother/smoother_validator.hpp`
   - 最后确认哪些条件只是优化目标，哪些条件会在求解后被硬性拒绝。

如果你接下来要改失败处理或对外错误语义，再补读 [ERROR_CODES.md](../ERROR_CODES.md) 和 [README.md](../README.md) 里的“失败传播路径”小节。

## 为什么求解器只优化二维位置

独立版平滑器是一个几何优化器，而不是完整的运动学状态优化器。

- Ceres 中的变量虽然是路径里的三维向量，但优化阶段第三个分量仍表示 direction sign，而不是转向状态。
- yaw 是在求解结束后根据邻域几何关系重建出来的。
- 因此曲率和曲率变化率惩罚，本质上都是由路径几何形状推导出的代理量。

这种设计让实现保持轻量，并且与原始 Nav2 constrained smoother 保持一致，但它也意味着求解器无法表达真正的原地转向动作。

## 残差模型

主残差来自 `SmootherCostFunction`，由 `SmootherProblemBuilder` 负责连接。

主要项包括：

- 平滑项。
  - 鼓励相邻路径点形成平滑局部曲线。
- 参考路径距离项。
  - 防止优化结果偏离原始路径过远。
- 障碍物净空项。
  - 使用 costmap 的 ESDF 采样结果惩罚净空不足。
- 曲率限制项。
  - 对超过 `max_curvature` 的局部曲率施加惩罚。
- 曲率变化率项。
  - 可选四点残差，用于抑制曲率突变。

`SmootherParams` 中的权重都采用平方根形式，因为每个残差会先被缩放，再由 Ceres 在目标函数里做平方。

从代码实现角度看，残差是按下面顺序接入问题的：

1. 遍历路径，确定哪些点会真正保留下来参与优化。
2. 对每一个可连接的三点窗口添加主残差。
3. 如果当前四点窗口跨越的所有段方向一致，再补一个曲率变化率残差。
4. 如果当前点位于 cusp 邻域，则对障碍物残差使用更高权重。

## 下采样策略

平滑器默认不会优化输入路径中的每一个原始点。

- `path_downsampling_factor` 允许在连接残差时跳过一部分内部点。
- 起点和终点始终保留。
- cusp 始终保留，即使它打破了常规下采样节奏。

因此，实际优化是在一个缩减后的控制点集合上进行的，而 `getLastOptimizedKnotCount()` 会报告最终保留下来的点数。

这也是为什么一条路径即使输入合法，最终构建出来的优化问题仍然可能非常简单。如果所有内部点都被跳过或冻结，构建器会返回 `false`，不会启动 Ceres，而最终路径会直接从施加锚定后的参考几何重建出来。

## Cusp 处理

当相邻两个被接受的路径点发生 direction sign 翻转时，会检测到一个 cusp。

Cusp 之所以重要，有两个原因：

- 它保留了运动方向确实发生变化这一关键信息。
- 它通常是整条路径里最容易碰撞的位置，尤其是在使用扩展足迹时。

实现上会对 cusp 做两类特殊处理：

1. cusp 点不会被常规下采样跳过。
2. 在 cusp 周围的可配置弧长邻域内，会提高障碍物权重。

重赋权逻辑会维护一个最近障碍物代价函数的双端队列。一旦检测到 cusp，就会回看这些邻近残差，并在 `cusp_zone_length` 范围内把它们的障碍物权重向 `cusp_costmap_weight_sqrt` 方向插值提升。

## 端点朝向锚定

平滑器始终固定起点和终点的位置，并可选地施加朝向约束。

- `keep_start_orientation` 会把第二个点移动到 `start_dir` 定义的射线上。
- `keep_goal_orientation` 会把倒数第二个点移动到 `end_dir` 定义的射线上。

这些锚定在构建残差之前施加，随后在 Ceres 问题中被冻结。

如果路径只有三个点且两个选项都开启，那么唯一的中间点会在起点和终点两侧的锚定建议之间做折中。

## 路径重建

优化结束后，求解器并不会直接返回最终对外路径，`SmootherPathOps::populateOutput(...)` 还会执行第二阶段重建。

- 只遍历保留下来的优化控制点。
- 当 `path_upsampling_factor > 1` 时，使用三次 Bezier 曲线为被跳过的区间补点。
- 为每个返回姿态根据局部切向方向赋值 yaw。

因此，输出 yaw 是一个重建值，而不是直接被优化的状态变量。

这部分在代码里的顺序同样很重要：

1. 先识别哪些优化控制点构成新的关键帧链。
2. 再为关键点恢复切向方向和 yaw。
3. 如果关键点之间原本跨过了被下采样的区段，则用 Bezier 插值补点。
4. 最后为补出来的中间点逐个补齐 yaw。

## 后验校验

独立版实现不会因为数值上收敛就默认结果安全，而是会在优化之后显式执行后验校验。

`SmootherValidator` 目前会检查：

- 坐标和 yaw 是否都是有限值。
- 起点和终点位置是否保持固定。
- 起点和终点朝向约束是否满足。
- 曲率限制是否满足。
- 是否仍满足与优化阶段一致的 ESDF 障碍物净空要求。

如果任何一项校验失败，`smooth(...)` 会根据调用方式选择抛异常或填充 `SmoothingFailureInfo`。

## ESDF 所有权

扁平化 ESDF 向量存放在 `Smoother` 实例内部，这份存储会被两个消费者复用：

- 优化阶段障碍物残差使用的双三次插值器。
- 后验校验阶段按网格采样有符号距离的检查逻辑。

正因为这份数据被多个阶段共享，调用方传入的 `precomputed_esdf` 才必须与 costmap 尺寸完全一致。

## 建议结合阅读的文件

如果你要改行为，这几个文件构成了最小且足够有用的阅读集合：

- `include/constrained_smoother/smoother.hpp`
- `include/constrained_smoother/smoother_path_ops.hpp`
- `include/constrained_smoother/smoother_problem_builder.hpp`
- `include/constrained_smoother/smoother_cost_function.hpp`
- `include/constrained_smoother/smoother_validator.hpp`
- `include/constrained_smoother/options.hpp`
- `docs/KINEMATIC_SMOOTHER_DESIGN.md`

把这些文件一起看，基本就能串起从参数定义、残差构建到安全校验的完整链路；如果你还想比较运动学版和平面几何版的差异，再接着看 `KINEMATIC_SMOOTHER_DESIGN.md`。

## 常见误读

新读者最容易误解的点主要有这些：

- 输入第三个分量不是 yaw。
- `max_curvature` 表示曲率，不是最小转弯半径。
- `reversing_enabled=false` 时，运动学版会忽略输入中的倒车标记并按全前进路径处理；几何版仍按现有路径方向语义工作。
- 即使 Ceres 返回成功，结果仍然要通过显式后验校验。
- cusp 表示方向符号翻转，而不是原地打方向盘的动作。

## 常见误改点

如果你准备改实现，最容易把行为改坏的地方通常有这些：

- 误把输入路径第三个分量当成 yaw。
   - 对几何版来说，优化前它仍然是 `direction_sign`，过早按 yaw 解释会破坏 cusp 检测和重建逻辑。
- 在 `SmootherPathOps` 里修改残差或 ESDF 逻辑。
   - 这个 helper 只负责路径副本和输出重建；残差连接应留在 `SmootherProblemBuilder`。
- 在 `SmootherProblemBuilder` 里直接写后验拒绝策略。
   - 求解后是否接受结果应继续收口在 `SmootherValidator`，不要把“优化目标”和“交付门槛”混在一起。
- 新增 failure reason 时只改抛错文本，不改稳定 reason / code 文档。
   - 相关改动至少要同步 `exceptions.hpp`、`ERROR_CODES.md`，必要时同步 README 的失败传播说明。

## 建议扩展点

如果后续要演进实现，影响最小的切入点主要有这些：

- 在保持现有路径遍历结构不变的前提下，在 `addPathResidualBlocks(...)` 中添加新的残差项。
- 在对下游开放新的求解模式之前，先扩展 `SmootherValidator`。
- 通过 `SmoothingFailureInfo` 增加更丰富的诊断信息，而不是继续堆叠自由文本错误消息。

保持这些边界稳定，有助于维持“优化、重建、校验”三阶段的清晰分层。