# 错误码参考

本页是独立运动学平滑器包的稳定错误码和失败原因的完整参考。

关于 C++ 层失败传播的高层流程，也请参阅 [包使用指南](package-guide.md) 的"失败传播路径"小节。

- 包使用指南解释了什么时候失败会立即抛出、什么时候会写入 `SmoothingFailureInfo`。
- 本页解释稳定 `code` 和 `reason` 值到达调用层后的含义。

## 设计原则

- 错误码是稳定标识符；调用方应按 `code` 分支处理，不要依赖自由格式的 `message` 文本。
- `message` 可能随版本调整以提升可读性。
- 高层 API 应优先使用结构化返回值，而不是让原生异常穿透。

## 核心平滑器错误码

| 错误码 | 层级 | 含义 | 典型触发场景 | 建议处理方式 |
| --- | --- | --- | --- | --- |
| `CS_INVALID_PATH` | C++，pybind `try_*` | 输入路径过短或格式异常。 | 传给平滑器的路径少于 2 个节点。 | 平滑前校验路径长度；保留参考路径。 |
| `CS_SMOOTHING_FAILED` | C++，pybind `try_*`，Web `smooth_error` | 优化器运行但未产出可用解。 | 解不可用、目标函数未改善、后验校验发现边界 / 运动方向 / 足迹冲突。 | 回退到参考路径，检查 `error_reason` 或 `smooth_error.details.failure_reason` 确认具体失败原因。 |
| `CS_INVALID_COSTMAP` | C++，pybind `try_*` | 启用了障碍物或 planner 地图项但未提供有效 costmap。 | 在障碍物残差、障碍物校验或 ESDF 复用需要地图时传入了 null costmap。 | 提供有效 costmap，或在重试前禁用依赖障碍物的残差项。 |
| `CS_PRECOMPUTED_ESDF_SIZE_MISMATCH` | C++，pybind `try_*` | 预计算 ESDF 尺寸与 costmap 不匹配。 | 复用 planner ESDF 时地图尺寸不一致。 | 丢弃缓存的 ESDF，从当前地图重新计算。 |

## 错误表面映射

下表展示同一个后端失败在不同公共层的呈现方式。

| 场景 | C++ 层 | pybind `try_*` 层 | Web 层 |
| --- | --- | --- | --- |
| 无效路径 / costmap / ESDF 尺寸 | 立即抛 `InvalidPath`、`InvalidCostmap` 或 `PrecomputedEsdfSizeMismatch` | 返回 `ok=false` 和对应的稳定 `error_code` | 作为请求或 smoother 初始化失败上报 |
| 求解器解不可用 / 目标函数无改善 | `throwOrStoreSmoothingFailure(...)` → `FailedToSmoothPath` 或 `failure` 载荷 | 返回 `ok=false`、`error_code=CS_SMOOTHING_FAILED`，附带 `error_reason` / `error_details` | 通常出现在 `smooth_error` 中，包含 reason / details |
| 后验校验边界 / 碰撞 / 曲率拒绝 | 同上，走 `throwOrStoreSmoothingFailure(...)` 路径 | 同上，`CS_SMOOTHING_FAILED`，通过 `error_reason` 区分 | `smooth_error` 和矩形校验字段可能同时暴露拒绝原因 |
| Web 专属请求校验失败 | 不适用 | 不适用 | 端点返回 `CS_INVALID_REQUEST` 或其他 Web 专属错误码 |

使用 [包使用指南](package-guide.md) 的失败传播流程了解什么时候错误会被抛出或存储，用下表解释到达调用层的稳定 `code` / `reason` 值。

## Web API 错误码

| 错误码 | 端点 | 含义 | 典型触发场景 | 建议处理方式 |
| --- | --- | --- | --- | --- |
| `CS_INVALID_REQUEST` | `/api/obstacles`，`/api/plan` | 请求载荷校验失败。 | 缺少障碍物列表、数值格式错误、无效形状。 | 修复请求载荷后重试。 |
| `CS_ASTAR_NO_PATH` | `/api/plan` | A* 未找到可行路径。 | 起/终点在地图外、起/终点足迹被阻挡、或通道完全堵塞。 | 检查 `error.details.reason`，调整端点、足迹或障碍物布局。 |
| `CS_FINAL_PATH_NONFINITE` | `/api/plan` `smooth_error` | 最终后验校验发现非有限姿态值。 | 平滑候选路径的 `x`、`y` 或 `yaw` 包含 `NaN` 或 `Inf`。 | 拒绝候选路径，检查优化器输出。 |
| `CS_FINAL_PATH_OUT_OF_BOUNDS` | `/api/plan` `smooth_error` | 最终后验校验发现机器人足迹超出地图。 | 平滑候选路径的矩形足迹超出了 costmap 范围。 | 拒绝候选路径；减小变形量或调整约束。 |
| `CS_FINAL_PATH_COLLISION` | `/api/plan` `smooth_error` | 最终后验校验发现足迹碰撞。 | 矩形足迹校验后平滑候选路径与致命栅格重叠。 | 拒绝候选路径，回退到参考路径。 |
| `CS_INTERNAL_ERROR` | 任意 Web 端点 | 意外的服务器端错误。 | 未处理的 Python 异常或运行时故障。 | 检查日志和服务器状态后重试。 |

## `CS_SMOOTHING_FAILED` 失败原因

错误码固定为 `CS_SMOOTHING_FAILED`，后端会同时报告一个稳定的 reason 字符串。

| Reason | 含义 | 典型触发场景 |
| --- | --- | --- |
| `solver_rejected_solution` | Ceres 求解后 `IsSolutionUsable()` 为 false | 求解器未收敛或数值不稳定 |
| `no_cost_improvement` | 最终目标函数值未低于初始值 | 初值已经接近局部最优但不可用 |
| `invalid_state_vector` | 打包后的状态向量维度不正确 | 内部状态展开与求解器维度不一致 |
| `nonfinite_state` | 返回的状态包含 `NaN` 或 `Inf` | 数值溢出或除零 |
| `start_position_constraint` | 返回路径偏离了固定的起点位置 | 优化器移动了起点 |
| `start_orientation_constraint` | 返回路径违反了固定的起点朝向 | 优化器旋转了起点 |
| `goal_position_constraint` | 返回路径偏离了终点位置（超出 lon/lat 容差框） | 优化器移动了终点或容差过紧 |
| `goal_orientation_constraint` | 返回路径违反了终点朝向约束（超出朝向容差） | 优化器旋转了终点 |
| `cusp_hold_constraint` | cusp 重复状态本应保持静止，但发生了移动或旋转 | 换向过渡点位置/朝向漂移 |
| `collapsed_segment` | 非 cusp 段坍缩到接近零长度 | 相邻点合并 |
| `motion_direction_constraint` | 段位移方向与输入 gear 或端点朝向约束矛盾 | 前进段实际倒车或反之 |
| `path_out_of_bounds` | 足迹检查点在后验校验中离开地图范围 | 路径超出 costmap 边界 |
| `footprint_collision` | 足迹检查点的净空小于配置的碰撞半径 | 路径过于接近障碍物 |
| `curvature_constraint` | 返回路径在后验校验中超出了最大曲率限制 | 显式 `kappa` 或几何曲率超限 |

当可用时，`error_details.failed_index` 标识失败的状态或段索引。

## `SmoothingFailureInfo` 详细字段

后验校验失败时，`SmoothingFailureInfo` 会根据失败类型填充额外的诊断字段：

| 字段 | 类型 | 填充时机 | 含义 |
| --- | --- | --- | --- |
| `reason` | `SmoothingFailureReason` | 始终 | 失败原因枚举 |
| `message` | `string` | 始终 | 人类可读的错误描述 |
| `failed_index` | `int` | 始终 | 失败的状态或段索引（-1 表示不适用） |
| `actual_curvature` | `double` | 曲率约束失败 | 实际曲率值（`1/m`） |
| `max_curvature` | `double` | 曲率约束失败 | 配置的曲率上限（`1/m`） |
| `turning_radius` | `double` | 曲率约束失败 | 对应的转弯半径（`m`） |
| `goal_longitudinal_error` | `double` | 终点位置约束失败 | 终点在目标坐标系前向轴上的偏差（`m`） |
| `goal_lateral_error` | `double` | 终点位置约束失败 | 终点在目标坐标系横向轴上的偏差（`m`） |
| `goal_longitudinal_tolerance` | `double` | 终点位置约束失败 | 前向轴容差（`m`） |
| `goal_lateral_tolerance` | `double` | 终点位置约束失败 | 横向轴容差（`m`） |

## Python 安全 API 返回格式

### pybind 平滑器包装器

- `KinematicSmoother.try_smooth(...)`
- `KinematicSmoother.try_smooth_with_planner_esdf(...)`

返回格式：

```python
{
    "ok": bool,
    "path": list | None,
    "error_code": str | None,       # CS_* 错误码
    "error_message": str | None,
    "error_reason": str | None,     # SmoothingFailureReason 字符串
    "error_details": dict | None,   # 可包含 failed_index、曲率信息等
}
```

### `/api/plan` 平滑校验字段

当平滑运行时，Web API 可能返回以下额外字段：

```python
{
    "smooth_success": bool,
    "smooth_error": {
        "code": str,
        "message": str,
        "source": "smoother" | "post_validation",
        "details": {
            "failure_reason": str,
            "failed_index": int,
        } | dict | None,
    } | None,
    "candidate_rectangle_validation": {
        "valid": bool,
        "error_code": str | None,
        "message": str,
        "first_failure": dict | None,
        "validated_path": "smoothed_candidate",
    } | None,
    "final_rectangle_validation": {
        "valid": bool,
        "error_code": str | None,
        "message": str,
        "validated_path": "smoothed_path" | "reference_fallback",
    },
}
```

如果 `candidate_rectangle_validation.valid` 为 `false`，说明平滑候选路径在优化后被矩形足迹校验拒绝，`smooth_success` 保持 `false`，Web 响应仍然返回该候选路径供可视化。
