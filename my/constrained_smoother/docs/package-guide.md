# Package Guide

`my/constrained_smoother` 是一个独立于 ROS 的约束平滑器实验包，当前把三类东西放在一起：

- 基于 Ceres 和 Eigen 的 C++ constrained smoother。
- 轻量级 C++ A* 规划器和 ESDF 工具。
- 基于 Flask 的 Web Lab，用于查看 costmap、规划结果和 smoother 行为。

## Key API Conventions

1. 输入路径使用 `(x, y, direction_sign)`，不是 `(x, y, yaw)`。
2. 输出路径会把第三个分量改写成弧度制 `yaw`。
3. `SmootherParams` 中大部分权重使用平方根形式，例如 `smooth_weight_sqrt = sqrt(weight)`。
4. `cost_check_points` 直接按 `(x_local, y_local, weight)` 使用，不会被额外预处理。
5. `reversing_enabled=false` 会让运动学 smoother 把所有段都当成前进段处理。
6. `max_curvature` 表示曲率 `1 / m`，不是最小转弯半径。

## 文档入口

- [Geometric Smoother I/O](SMOOTHER_INPUT_OUTPUT.md)
- [Geometric Smoother Design](SMOOTHER_DESIGN.md)
- [Kinematic Smoother Design](KINEMATIC_SMOOTHER_DESIGN.md)
- [MPC Cost Notes](mpc_cost.md)
- [Error Codes](error-codes.md)

## 调用链速览

几何版：

```text
smooth()
  -> Run::prepare()
    -> SmootherPathOps::initializeOptimizationPath()
    -> SmootherProblemBuilder::buildProblem()
  -> Run::solve()
  -> Run::finalize()
    -> SmootherPathOps::populateOutput()
    -> SmootherValidator::validateSmoothedPath()
```

运动学版：

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

当前 C++ 层的失败传播分成两类：

- 输入前置条件失败：直接抛结构化异常。
- 求解或后验校验失败：走 `throwOrStoreSmoothingFailure(...)`。

如果 `smooth(..., failure)` 传入了非空 `failure`，错误会写入 `SmoothingFailureInfo` 并返回 `false`；如果 `failure == nullptr`，则会抛 `FailedToSmoothPath`。

稳定错误码和原因字符串的完整目录见 [Error Codes](error-codes.md)。

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --parallel
```

### Build with Python bindings

```bash
mkdir -p build
cd build
cmake .. -DBUILD_PYTHON=ON -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build . --parallel
```

### Build and run the Web Lab

```bash
./run_web_app.sh
```

### Serve these docs locally

```bash
uvx --with mkdocs-material mkdocs serve -f mkdocs.yml
```

### Build the static docs site

```bash
uvx --with mkdocs-material mkdocs build -f mkdocs.yml
```