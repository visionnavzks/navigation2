# Package Guide

`my/constrained_smoother` 现在是一个只保留运动学后端的独立实验包：

- C++ `KinematicSmoother`
- C++ A* + ESDF 工具
- Flask Web Lab

## Key API Conventions

1. 输入路径使用 `(x, y, direction_sign)`。
2. 输出路径把第三个分量改写成弧度制 `yaw`。
3. `cost_check_points` 直接按 `(x_local, y_local, weight)` 使用。
4. `reversing_enabled=false` 会把整条路径按前进段处理。
5. `max_curvature` 的单位是 `1 / m`。

## 主要文档

- [Kinematic Smoother Design](KINEMATIC_SMOOTHER_DESIGN.md)
- [Error Codes](error-codes.md)

## 使用入口

- 纯 C++ 调用：`KinematicSmoother::smooth(...)`
- Python / pybind 调用：`KinematicSmoother.try_smooth(...)` 与 `KinematicSmoother.try_smooth_with_planner_esdf(...)`
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

## 失败传播路径

- 输入前置条件失败：直接抛结构化异常。
- 求解或后验校验失败：通过 `throwOrStoreSmoothingFailure(...)` 返回或抛出。

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

## Build

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

Web Lab 现在固定使用运动学后端，不再支持几何 smoother 或后端切换。