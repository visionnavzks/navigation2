# 包使用指南

`my/constrained_smoother` 是一个只保留运动学后端的独立实验包：

- C++ `KinematicSmoother`
- C++ A* + ESDF 工具
- Flask Web Lab

## 核心 API 约定

1. 输入路径使用 `(x, y, direction_sign)`，第三个分量表示前进/倒车方向。
2. 输出路径通过显式结果对象返回，结果中的第三个分量是弧度制 `yaw`。
3. `cost_check_points` 直接按 `(x_local, y_local, weight)` 三元组使用。
4. `reversing_enabled=false` 会把整条路径按前进段处理。
5. `max_curvature` 的单位是 `1/m`（曲率，不是半径）。

## 使用入口

- 纯 C++ 调用：`KinematicSmoother::smooth(const SmootherRequest&)`
- Python / pybind 安全调用：`KinematicSmoother.try_smooth(...)` 与 `KinematicSmoother.try_smooth_with_planner_esdf(...)`
- Python / pybind 异常调用：`KinematicSmoother.smooth(...)` 与 `KinematicSmoother.smooth_with_planner_esdf(...)`
- Web API 调用：`POST /api/plan`

## 调用链

```text
smooth()
  -> Run::prepare()
    -> KinematicSmootherProblemBuilder::initializeEsdfValues()
    -> KinematicSmootherProblemBuilder::buildProcessedPath()
    -> KinematicSmootherProblemBuilder::buildProblem()
    -> KinematicSmootherProblemBuilder::applyBounds()
  -> Run::solve()
  -> Run::finalize()
    -> KinematicSmootherProblemBuilder::unpackPath()
    -> SmootherValidator::validateKinematicSolution()
```

## 请求结构

当前统一使用 `SmootherRequest` 结构体，不再有多参数重载：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `path` | `const vector<Vector3d>&` | 只读输入路径；第三分量表示方向，不会在 smooth() 中被原地改写 |
| `start_dir` | `Vector2d` | 起点切向方向（向量语义，不是 `yaw` 标量） |
| `end_dir` | `Vector2d` | 终点切向方向（向量语义，不是 `yaw` 标量） |
| `costmap` | `Costmap2D*` | 优化使用的代价地图，上层需保证生命周期覆盖整个调用 |
| `params` | `SmootherParams` | 残差权重、边界约束和运行参数 |
| `precomputed_esdf` | `vector<double>*` | 可选的预计算 ESDF；为空则由构建器根据 costmap 现场生成 |
| `failure` | `SmoothingFailureInfo*` | 可选失败回传槽；为空时失败通过异常传播 |

`KinematicSmoother::smooth(...)` 现在返回 `SmootherResult`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `candidate_path` | `vector<Vector3d>` | 求解后直接解包得到的候选路径；若后验校验失败，仍可用于诊断 |
| `smoothed_path` | `vector<Vector3d>` | 通过后验校验并按运动学模型上采样后的最终路径 |
| `optimized_knot_count` | `size_t` | 本次参与优化的状态点数量 |
| `target_spacing` | `double` | 本次优化使用的目标 knot 间距（米） |
| `success` | `bool` | 是否得到可交付的最终平滑路径 |

## 参数说明

### SmootherParams

参数按四类组织：

#### 运动学和参考路径权重

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_weight` | `double` | `0.0` | 运动学状态转移一致性残差的权重（传入平方后的值，内部自动开方） |
| `reference_path_weight` | `double` | `0.0` | 控制点贴近参考路径的权重（传入平方后的值，内部自动开方）；设为 0 时路径只受运动学与障碍物项驱动 |
| `reference_point_max_deviation_m` | `double` | `0.0` | 每个优化点相对对应参考点的最大 x/y 偏移半径（米）；<= 0 表示关闭 |
| `kinematic_curvature_weight` | `double` | `0.0` | 显式曲率状态 `kappa` 的正则权重（传入平方后的值，内部自动开方） |
| `kinematic_curvature_rate_weight` | `double` | `0.0` | 曲率变化率项的权重（传入平方后的值，内部自动开方） |
| `kinematic_spacing_weight` | `double` | `1.0` | 弧长步长 `ds` 贴近目标间距的正则权重（传入平方后的值，内部自动开方）；避免步长变量在无约束时完全漂移 |
| `kinematic_max_spacing` | `double` | `0.0` | 弧长步长 `ds` 的上界（米）；<= 0 表示不启用上界 |
| `path_length_weight` | `double` | `0.0` | 总路径长度惩罚的权重（传入平方后的值，内部自动开方）；值越大越倾向于压缩总弧长 |
| `fix_weight` | `double` | `100.0` | cusp 保持段与起终点边界残差共用的直接约束权重；不会做 sqrt 变换 |
| `max_curvature` | `double` | `0.0` | 最大曲率约束（`1/m`） |
| `max_time` | `double` | `10.0` | 传给 Ceres 的最大墙钟时间（秒） |

> 大多数权重由调用方传入平方后的值（即实际权重），代码内部自动开方后再乘到残差上，随后由 Ceres 在目标函数中完成平方。

#### 障碍物与足迹检查

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `obstacle_weight` | `double` | `0.0` | 障碍物净空残差的统一权重（传入平方后的值，内部自动开方） |
| `use_exact_esdf` | `bool` | `true` | 为 `true` 时使用精确有符号距离场后端 |
| `obstacle_safe_distance` | `double` | `0.5` | 对障碍物距离场期望满足的最小有符号净空（米） |
| `cost_check_radius` | `double` | `0.0` | 当 `cost_check_points` 为空时使用的圆形足迹采样半径（米） |
| `cost_check_points` | `vector<double>` | `{}` | 障碍物足迹检查的局部坐标三元组 `(x, y, weight)`；为空时退回单圆模型 |

辅助方法 `obstacleTermsEnabled()` 返回当前是否真的启用了任何障碍物残差（`obstacle_weight` 大于阈值）。

#### 路径重采样与方向语义

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `path_downsampling_factor` | `int` | `1` | 连接残差块之前的路径下采样步长；值越大参与求解的状态数越少 |
| `path_upsampling_factor` | `int` | `1` | 重建最终路径时的插值倍数；值越大输出路径越密 |
| `reversing_enabled` | `bool` | `true` | 为 `false` 时忽略方向分量，整条路径按前进段处理 |

#### 起终点约束

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `goal_longitudinal_tolerance` | `double` | `0.0` | 终点在目标坐标系前向轴上的允许位置容差（米）；0 表示严格固定 |
| `goal_lateral_tolerance` | `double` | `0.0` | 终点在目标坐标系横向轴上的允许位置容差（米）；0 表示严格固定 |
| `goal_orientation_tolerance` | `double` | `0.0` | 终点朝向允许容差（弧度）；仅在 `keep_goal_orientation=true` 时生效 |
| `keep_goal_orientation` | `bool` | `true` | 通过锚定终点前一个点来固定终点切向方向 |
| `keep_start_orientation` | `bool` | `true` | 通过锚定第二个点来固定起点切向方向 |

### OptimizerParams

传递给 Ceres 的求解器级配置：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `debug` | `bool` | `false` | 开启逐迭代详细日志和最终摘要输出 |
| `linear_solver` | `LinearSolver` | `SparseNormalCholesky` | 线性求解器选择：`SparseNormalCholesky`（默认，适合当前稀疏结构）或 `DenseQr`（适合小型稠密问题） |
| `max_iterations` | `int` | `50` | 最大非线性迭代次数 |
| `parameter_tolerance` | `double` | `1e-8` | 参数步长收敛阈值 |
| `function_tolerance` | `double` | `1e-6` | 目标函数值收敛阈值 |
| `gradient_tolerance` | `double` | `1e-10` | 梯度收敛阈值 |

## 失败传播路径

- **输入前置条件失败**：直接抛结构化异常（`InvalidPath`、`InvalidCostmap`、`PrecomputedEsdfSizeMismatch`）。
- **求解或后验校验失败**：通过 `throwOrStoreSmoothingFailure(...)` 返回或抛出。
  - `failure != nullptr` 时写入 `SmoothingFailureInfo` 并返回 `false`。
  - `failure == nullptr` 时抛 `FailedToSmoothPath`。

Web 层会再做一层统一包装：

- 请求级错误走 `error`。
- 平滑失败走 `smooth_error`。
- 矩形足迹后验校验走 `candidate_rectangle_validation` / `final_rectangle_validation`。

## Web `/api/plan` 结构

当前 `web/app.py` 的规划主线按下面的阶段展开：

1. `PlanRequestConfig.from_payload()`
   - 把前端请求参数标准化为一个不可变配置对象。
2. planner stage
   - 手工参考路径模式走 `_run_manual_reference_stage(...)`。
   - 自动规划模式走 `_run_astar_stage(...)`。
3. smoother stage
   - 调用 `KinematicSmoother.try_smooth(...)` 或 `try_smooth_with_planner_esdf(...)`。
4. rectangle validation stage
   - 对候选路径做最终矩形足迹验证。
5. response assembly
   - 返回 pipeline、路径几何、优化器配置和诊断字段。

## 构建

### C++ 库

```bash
cmake -S . -B build
cmake --build build --parallel
```

### Python 绑定

```bash
cmake -S . -B build-py \
  -DBUILD_TESTS=OFF \
  -DBUILD_PYTHON=ON \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build-py --target py_constrained_smoother --parallel
```

### Web Lab

```bash
./run_web_app.sh
```

Web Lab 固定使用运动学后端，不再支持几何 smoother 或后端切换。

## 相关文档

- [运动学平滑器设计](KINEMATIC_SMOOTHER_DESIGN.md)
- [错误码参考](error-codes.md)
