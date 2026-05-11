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
   - 障碍物距离场的来源。
   - 生命周期必须覆盖整个 `smooth()` 调用。
5. `params: const SmootherParams &`
   - 提供残差权重、曲率阈值、下采样倍率、上采样倍率、是否保持端点朝向等运行参数。
6. `precomputed_esdf: const std::vector<double> *`（可选）
   - 若提供，则直接复用这份扁平化 ESDF。
   - 若为空，则内部根据 `costmap` 现场生成。
   - 若尺寸与 `costmap` 不匹配，会抛出 `PrecomputedEsdfSizeMismatch`。
7. `failure: SmoothingFailureInfo *`（可选）
   - 仅用于“失败走普通返回值而不是异常”的调用模式。
   - 为空时，求解失败和后验校验失败通常通过异常传播。

从实现角度看，上述输入在入口处会被折叠为一个 `SmootherRequest`，供单次执行对象 `Smoother::Run` 在整个生命周期内共享。

### 输入语义和所有权约束

- `path` 是唯一会被原地改写的输入缓冲区。
- `start_dir` 和 `end_dir` 始终按几何切向量解释，不应传 yaw 标量的 `cos/sin` 之外的任何编码。
- `costmap` 和 `precomputed_esdf` 都是借用视图，`Smoother` 不拥有它们的生命周期。
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
   - 承载三点平滑残差、可选四点曲率变化率残差、障碍物净空残差和边界冻结约束。
5. 扁平化 ESDF 缓存 `esdf_values_`
   - 同时服务于优化阶段的障碍物残差与求解后的净空校验。

这些中间量解释了为什么系统的“输入路径”和“最终输出路径”之间并不是一一对应的原样求解，而是会经过“锚定、下采样求解、上采样重建”三段变换。

### 成功输出

当 `smooth()` 返回 `true` 时，对外可观察到的输出有四类：

1. `path` 被原地改写为输出路径。
   - 每个点现在表示 `(x, y, yaw)`。
   - `yaw` 是重建值，不是优化阶段直接求解的状态。
2. 路径点数量可能变化。
   - 当 `path_upsampling_factor > 1` 时，系统会在关键控制点之间补插中间点。
3. 起点和终点位置保持固定。
   - 若启用了端点朝向保持，对应切向约束也应在输出上继续成立。
4. `getLastOptimizedKnotCount()` 可返回最近一次真正参与求解的控制点数量。
   - 这个值反映的是下采样后的优化控制点数，不是最终输出路径长度。

有一个容易忽略的边界情况：如果输入路径虽然合法，但所有内部点都被跳过，或者构建后没有可优化自由度，系统可能不会真正调用 Ceres，但仍会返回 `true`，并基于锚定后的几何关系重建输出路径。

### 失败输出

失败分成两类，输出面不同：

1. 输入前置条件失败。
   - 典型情况包括路径过短、`costmap` 无效、预计算 ESDF 尺寸不匹配。
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
  + costmap
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

如果把参与优化的二维控制点记为

$$
p_i = \begin{bmatrix} x_i \\ y_i \end{bmatrix},
\qquad
\bar p_i = \text{参考路径上对应的原始位置},
$$

那么几何版实际最小化的是一组残差平方和：

$$
J = \sum_i \left\|r_i^{\text{main}}\right\|_2^2 + \sum_i \left\|r_i^{\text{rate}}\right\|_2^2,
$$

其中主残差块 `r_i^{main}` 来自 `SmootherCostFunction`，维度是 6；
可选的曲率变化率残差块 `r_i^{rate}` 来自 `CurvatureRateCostFunction`，维度是 2。

### 记号约定

对某个三点窗口 $(p_{i-1}, p_i, p_{i+1})$，定义

$$
\Delta_i^- = p_i - p_{i-1},
\qquad
\Delta_i^+ = p_{i+1} - p_i,
$$

$$
\rho_i = \frac{\lVert \Delta_i^- \rVert}{\lVert \Delta_i^+ \rVert}.
$$

如果当前中心点 $p_i$ 是 cusp，代码会把这个比值带符号地传进残差：

$$
\rho_i^{\pm} = s_i \rho_i,
\qquad
s_i =
\begin{cases}
-1, & p_i \text{ 是 cusp} \\
1, & \text{否则}
\end{cases}
$$

这就是实现里的 `last_to_current_length_ratio_`。

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

### 主残差块的实际公式

对每个被接受的三点窗口，`SmootherCostFunction` 会生成

$$
r_i^{\text{main}} =
\begin{bmatrix}
r_{s,x} \\
r_{s,y} \\
r_{\kappa} \\
r_{d,x} \\
r_{d,y} \\
r_{obs}
\end{bmatrix}.
$$

#### 1. 平滑项

实现里的平滑项是一个二维向量残差：

$$
r_i^{\text{smooth}}
=
\sqrt{w_s}\left(\rho_i^{\pm} \Delta_i^+ - \Delta_i^-\right).
$$

展开成两个标量分量，就是：

$$
r_{s,x} = \sqrt{w_s}\left(\rho_i^{\pm} \Delta_{i,x}^+ - \Delta_{i,x}^-\right),
\qquad
r_{s,y} = \sqrt{w_s}\left(\rho_i^{\pm} \Delta_{i,y}^+ - \Delta_{i,y}^-\right).
$$

当没有 cusp 时，这一项鼓励前后段方向和长度比例保持平滑；当中心点是 cusp 时，$\rho_i^{\pm}$ 会带负号，相当于把后继段按“翻转方向”参与比较。

#### 2. 曲率限制项

实现先用三点几何恢复局部圆心 $c_i$。若当前窗口不是直线或退化情形，则局部转弯半径为

$$
R_i = \lVert p_i - c_i \rVert,
\qquad
\kappa_i = \frac{1}{R_i}.
$$

对应残差是一个单边惩罚：

$$
r_{\kappa} = \sqrt{w_{\kappa}}\; \max\left(0, \kappa_i - \kappa_{\max}\right).
$$

如果三点近似共线，或曲率没有超过 `max_curvature`，这一项就是 0。

#### 3. 参考路径距离项

这部分是最直接的二范数吸引项：

$$
r_i^{\text{dist}} = \sqrt{w_d}(p_i - \bar p_i).
$$

展开后：

$$
r_{d,x} = \sqrt{w_d}(x_i - \bar x_i),
\qquad
r_{d,y} = \sqrt{w_d}(y_i - \bar y_i).
$$

它的作用是阻止优化结果在没有必要时偏离参考路径太远。

#### 4. 障碍物净空项

令 ESDF 在世界坐标点 $q$ 处的有符号距离为 $d_{esdf}(q)$，安全净空阈值为

$$
d_{safe} = \max(\texttt{obstacle\_safe\_distance}, 10^{-6}),
$$

若使用圆形简化足迹，则实现先扣除足迹半径

$$
d_{surf}(q) = d_{esdf}(q) - \max(\texttt{cost\_check\_radius}, 0).
$$

然后定义内部惩罚函数

$$
\phi(q) =
\begin{cases}
0, & d_{surf}(q) \ge d_{safe} \\
\left(\dfrac{d_{safe} - d_{surf}(q)}{d_{safe}}\right)^2, & d_{surf}(q) < d_{safe}
\end{cases}
$$

如果没有配置 `cost_check_points`，障碍物残差就是

$$
r_{obs} = \sqrt{w_o}\; \phi(p_i).
$$

如果配置了扩展足迹采样点 $(u_j, v_j, \beta_j)$，则先根据当前点的局部切向方向构造位姿变换 $T_i$，把局部足迹点变到世界坐标：

$$
q_{i,j} = T_i
\begin{bmatrix}
u_j \\
v_j \\
1
\end{bmatrix}.
$$

随后实现中的单个标量障碍物残差是

$$
r_{obs} = \sqrt{w_o} \sum_j \beta_j \phi(q_{i,j}).
$$

这里有两个实现细节值得单独记住：

1. `\phi(\cdot)` 内部已经先做了一次平方。
   - 因为 Ceres 还会再对整个残差平方，所以真正进入目标函数的是四次型的净空缺口惩罚。
2. 多个扩展足迹采样点不是“各自独立成残差后再求平方和”。
   - 当前实现是先把 $\sum_j \beta_j \phi(q_{i,j})$ 聚合成一个标量残差，再由 Ceres 对这个总和平方。

### 曲率变化率残差的实际公式

当四点窗口 $(p_{i-1}, p_i, p_{i+1}, p_{i+2})$ 跨越的所有局部段方向一致时，系统才会额外添加 `CurvatureRateCostFunction`：

$$
r_i^{\text{rate}} = \sqrt{w_{\dot \kappa}}
\left(p_{i+2} - 3p_{i+1} + 3p_i - p_{i-1}\right).
$$

它是一个二维向量残差，本质上是对控制点链三阶有限差分的惩罚，用来抑制局部曲率代理量的剧烈变化。

也正因为它要求四段局部方向一致，只要窗口穿过 cusp，这一项就不会被接入优化问题。

### 目标函数和参数的对应关系

如果忽略边界冻结和是否接入的条件判断，可以把实现近似理解成：

$$
J
=
\sum_i
\Big(
\lVert r_i^{\text{smooth}} \rVert_2^2
+ r_{\kappa,i}^2
+ \lVert r_i^{\text{dist}} \rVert_2^2
+ r_{obs,i}^2
\Big)
+ \sum_i \lVert r_i^{\text{rate}} \rVert_2^2.
$$

其中各权重和 `SmootherParams` 字段的对应关系是：

- $w_s$ 对应 `smooth_weight_sqrt^2`
- $w_{\kappa}$ 对应 `curvature_weight_sqrt^2`
- $w_d$ 对应 `distance_weight_sqrt^2`
- $w_o$ 对应 `costmap_weight_sqrt^2`，或者 cusp 邻域里的增强权重
- $w_{\dot \kappa}$ 对应 `curvature_rate_weight_sqrt^2`

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

几何版里的 cusp 不是“原地打方向盘”的动作，而是输入路径中局部运动方向从前进切到倒车，或从倒车切到前进的那个几何拐点。

### 识别时机

cusp 的识别发生在 `SmootherProblemBuilder::addPathResidualBlocks(...)` 的单次遍历里，而且发生在常规下采样判定之前。

- 判断依据不是 yaw，而是优化前路径第三个分量 `direction_sign`。
- 对于内部点 `i`，如果 `path_optim[i][2] * last_direction < 0`，当前点就被视为 cusp。
- `last_direction` 会随着遍历每一个原始点而更新，所以 cusp 判定是基于相邻原始路径点的方向符号翻转，而不是基于“上一个被保留的优化点”。
- 最后一个点不会被当作 cusp，因为 cusp 语义上需要同时连接前后两个局部路径段。

Cusp 之所以重要，有两个原因：

- 它保留了运动方向确实发生变化这一关键信息。
- 它通常是整条路径里最容易碰撞的位置，尤其是在使用扩展足迹时。

### 第 1 类处理：强制保留，不参与常规下采样跳过

一旦某个点被识别为 cusp，代码不会再走普通的 `path_downsampling_factor` 跳过逻辑。

- 这保证了方向切换点一定会出现在优化控制点链里。
- 如果没有这条规则，下采样可能把 cusp 整个跳掉，求解器就会把前进段和倒车段错误地当成同一条连续曲线来平滑。
- 因此，`getLastOptimizedKnotCount()` 统计出来的控制点数，通常会比“纯按固定步长下采样”略多，因为 cusp 会额外插入必须保留的关键点。

### 第 2 类处理：改变残差几何语义，而不只是保留一个点

cusp 不只是“多保留一个控制点”这么简单，它还会改变三点残差对局部几何的理解方式。

当构建器为某个三点窗口创建 `SmootherCostFunction` 时，会传入一个
`last_to_current_length_ratio`。正常情况下这个比值是正的，表示前后两个局部段按同向曲线处理；但如果“上一个被接受的控制点本身是 cusp”，这个比值会被乘以 `-1`。

这个负号会带来两个直接后果：

1. 平滑项不再把 cusp 两侧视为同向连续曲线。
   - 在残差内部，前后段差分会按“翻转后的后继段”来比较，避免把 gear 切换位置错误地拉直。
2. 曲率和切向方向的几何辅助函数会进入 cusp 模式。
   - `arcCenter(...)` 和 `tangentDir(...)` 在 `is_cusp=true` 时会把后继段方向取反，再计算圆心和切线。
   - 这样得到的是“跨过方向切换之后仍保持几何连续”的代理曲率，而不是把 cusp 当成普通单向转弯。

此外，四点曲率变化率残差还有一个额外限制：只有四个相邻控制点跨越的所有局部段方向符号都一致时，才会添加这一项。只要窗口内穿过了 cusp，曲率变化率残差就不会接入问题。

### 第 3 类处理：在 cusp 前后半区提高障碍物权重

几何版对 cusp 的第二个重点处理是障碍物代价重赋权。

- `cusp_zone_length` 表示总的 cusp 敏感区长度。
- 实现里实际使用的是 `cusp_half_length = cusp_zone_length / 2`。
- 也就是说，权重增强区会以 cusp 为中心，向前覆盖半段、向后再覆盖半段，而不是只对 cusp 之后的一段路径生效。

权重插值公式是线性的。若某个残差距离 cusp 的弧长距离为 `d`，则其障碍物权重会在下面两个端点之间过渡：

- `d = 0` 时，使用 `cusp_costmap_weight_sqrt`
- `d = cusp_half_length` 时，退回普通的 `costmap_weight_sqrt`

从实现机制看，这个过程分成“回看 cusp 之前”和“处理 cusp 之后”两半：

1. 遍历到 cusp 之前。
   - 构建器会把最近创建的障碍物残差函数和对应段长放进一个双端队列 `potential_cusp_funcs`。
   - 队列只保留累计弧长不超过 `cusp_half_length` 的那部分最近残差。
2. 一旦检测到 cusp。
   - 构建器会从离 cusp 最近的残差开始反向遍历这个队列。
   - 按距离把这些“已经创建完”的残差障碍物权重向 `cusp_costmap_weight_sqrt` 提升。
   - 这一步是回溯修改，所以 cusp 前半区不需要二次建图或重新建问题。
3. cusp 之后继续向前遍历。
   - 构建器把 `len_since_cusp` 置零。
   - 在后续残差创建时，只要离 cusp 的累计弧长仍在 `cusp_half_length` 之内，就直接按同一条线性规则给更高的障碍物权重。

这里有一个容易忽略的边界：被重赋权的只有障碍物残差项。

- 平滑项权重不会因为 cusp 提高。
- 参考路径距离项权重不会因为 cusp 提高。
- 曲率项和曲率变化率项也不会因为 cusp 改权。

换句话说，cusp 邻域的设计目标不是“整体更强地锁死路径”，而是“在方向切换附近更谨慎地满足净空要求”。

### 它和运动学版 cusp 的区别

这份文档讲的是几何版 smoother。这里的 cusp 处理和运动学版有一个重要区别：

- 几何版不会为 cusp 额外引入显式保持段，也没有单独的 `CuspHoldConstraint` 后验校验。
- 几何版的 cusp 影响主要体现在残差构建和障碍物重赋权上。
- 求解结束后，`SmootherValidator` 仍然只检查有限值、边界、曲率和净空，并不会单独再验证“cusp 保持段是否成立”。

因此，几何版里的 cusp 更准确地说是“优化问题构建时的重要局部语义”，而不是像运动学版那样进入显式状态约束系统的单独段类型。

### 如果你要改 cusp 行为，先看这几处

- `include/constrained_smoother/smoother_problem_builder.hpp`
  - 看 cusp 检测、控制点保留、双端队列回溯重赋权和负段长比值是怎么传进 cost functor 的。
- `include/constrained_smoother/smoother_cost_function.hpp`
  - 看负的 `last_to_current_length_ratio` 如何改变平滑项、曲率项和切向方向的几何解释。
- `include/constrained_smoother/utils.hpp`
  - 看 `arcCenter(...)` 和 `tangentDir(...)` 在 `is_cusp=true` 时怎样把后继段翻转后再计算。

如果你误把输入第三个分量当成 yaw，而不是 `direction_sign`，上面这整套 cusp 检测、重赋权和几何解释都会一起失效。

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