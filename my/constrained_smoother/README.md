# Constrained Smoother (Standalone)

A standalone, ROS-independent extraction of `nav2_constrained_smoother` with three pieces bundled together:

- A C++ constrained path smoother built on Ceres and Eigen.
- A lightweight C++ A* planner plus ESDF utilities.
- A Flask-based Web Lab for inspecting costmaps, planner output, and smoother behavior.

The smoother solves a nonlinear least-squares problem over 2D path geometry with smoothness, curvature, distance-to-reference, and obstacle-clearance terms.

## Key API Conventions

These are the most important behavior contracts in the current standalone implementation.

1. Input paths use `(x, y, direction_sign)`, not `(x, y, yaw)`.
	 - `direction_sign` should typically be `+1` for forward and `-1` for reverse.
2. Output paths overwrite the third component with `yaw` in radians.
	 - After `smooth()` returns, `path[i][2]` is no longer a direction sign.
3. `SmootherParams` expects square-root weights.
	 - Set `smooth_weight_sqrt = sqrt(weight)` for the geometric smoother, `model_weight_sqrt = sqrt(weight)` for the kinematic transition term, `costmap_weight_sqrt = sqrt(weight)`, and so on.
4. `cost_check_points` is used as-is.
	 - The standalone build does not preprocess footprint sample weights the way the ROS plugin layer does.
	 - Pass triples of `(x_local, y_local, weight)` in the robot local frame.
5. `reversing_enabled=false` forces the kinematic smoother to treat every segment as forward motion.
6. `max_curvature` is curvature in `1 / m`, not minimum turning radius.
7. Steering is not an explicit optimization state.
	 - The smoother optimizes 2D path geometry and reconstructs `yaw` afterward from local tangents.
	 - This means the standalone build can penalize curvature, but it cannot represent stop-and-steer maneuvers where the robot stays in place and only the steering angle changes.
	 - As a result, cusp-like features in this demo should be read as geometric direction-switch transitions, not as true in-place steering actions.

## 实现导读

如果你需要按“实际执行步骤”来理解内部数据流、残差连接方式、cusp 处理和后验校验流程，可以从下面两份文档开始：

- [docs/SMOOTHER_DESIGN.md](docs/SMOOTHER_DESIGN.md)
	- 对应几何版 `Smoother`，说明当前“顶层编排 + path ops + problem builder”的分层，以及 ESDF 准备、cusp 重赋权、路径重建与后验校验。
- [docs/KINEMATIC_SMOOTHER_DESIGN.md](docs/KINEMATIC_SMOOTHER_DESIGN.md)
	- 对应 `KinematicSmoother` 的 C++ 与 Python 两个实现，说明当前“顶层编排 + problem builder”的分层，以及状态展开、cusp 插入、残差拼接、变量边界和求解后校验。

### 当前 C++ 分层

如果你准备修改实现，先建立下面这张分层图会更省时间：

- `include/constrained_smoother/smoother_base.hpp`
	- 两种 smoother 共享的 solver 配置、输入前置校验和调试开关。
- `include/constrained_smoother/smoother_request.hpp`
	- 两种 smoother 共享的单次调用请求对象。
- `include/constrained_smoother/smoother_run_base.hpp`
	- 两种 smoother 共享的单次执行骨架，负责统一 `prepare -> solve -> finalize` 主线。
- `include/constrained_smoother/smoother.hpp`
	- 几何版 smoother 的顶层编排类。
- `include/constrained_smoother/smoother_path_ops.hpp`
	- 几何版路径锚定、上采样和 yaw 重建。
- `include/constrained_smoother/smoother_problem_builder.hpp`
	- 几何版 ESDF 准备、残差连接和问题冻结。
- `include/constrained_smoother/kinematic_smoother.hpp`
	- 运动学版 smoother 的顶层编排类。
- `include/constrained_smoother/kinematic_smoother_problem_builder.hpp`
	- 运动学版 ESDF 准备、状态展开、问题拼接、边界约束和结果解包。
- `include/constrained_smoother/kinematic_smoother_costs.hpp`
	- 运动学版各类 cost functor。

### 调用链速览

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

### 失败传播路径

当前 C++ 层的失败传播分成两类：

- 输入前置条件失败：直接抛结构化异常。
	- 例如 `InvalidPath`、`InvalidCostmap`、`PrecomputedEsdfSizeMismatch`。
- 求解或后验校验失败：走 `throwOrStoreSmoothingFailure(...)`。
	- 如果 `smooth(..., failure)` 传入了非空 `failure`，错误会写入 `SmoothingFailureInfo` 并返回 `false`。
	- 如果 `failure == nullptr`，则会抛 `FailedToSmoothPath`。

可以把这条主线理解成：

```text
bad input
  -> structured exception

solver / post-validation failure
  -> throwOrStoreSmoothingFailure(...)
    -> store into failure and return false
    -> or throw FailedToSmoothPath
```

稳定错误码和原因字符串的完整目录见 [docs/error-codes.md](docs/error-codes.md)。
如果你在排查失败语义，建议把本节和 [docs/error-codes.md](docs/error-codes.md) 一起看：这里讲传播路径，错误码文档讲稳定 code / reason 的含义。

### 最常见改动入口

如果你已经知道自己要改哪一类行为，可以直接从这里开始：

- 想改几何版残差或 cusp 邻域逻辑：看 `include/constrained_smoother/smoother_problem_builder.hpp`。
- 想改几何版端点锚定、上采样或 yaw 重建：看 `include/constrained_smoother/smoother_path_ops.hpp`。
- 想改运动学状态展开、边界约束或问题拼接：看 `include/constrained_smoother/kinematic_smoother_problem_builder.hpp`。
- 想改运动学 cost functor：看 `include/constrained_smoother/kinematic_smoother_costs.hpp`。
- 想改统一失败传播、错误码或异常语义：看 `include/constrained_smoother/exceptions.hpp`、[docs/error-codes.md](docs/error-codes.md) 和本 README 的“失败传播路径”小节。
- 想改后验校验策略：看 `include/constrained_smoother/smoother_validator.hpp`。
- 想改顶层调用编排或单次执行生命周期：看 `include/constrained_smoother/smoother.hpp`、`include/constrained_smoother/kinematic_smoother.hpp` 和 `include/constrained_smoother/smoother_run_base.hpp`。

### 接口选择指南

如果你在接入上层系统时不确定该走哪一类接口，可以按下面的原则选：

- 想在纯 C++ 场景里快速失败，并让调用栈自然中断：用 `smooth(...)` 的异常路径。
	- 适合测试、命令行工具和“失败就是异常事件”的上层逻辑。
- 想在纯 C++ 场景里把失败当成普通控制流处理：给 `smooth(..., failure)` 传非空 `failure`。
	- 适合需要记录失败原因、保留参考路径并继续执行的上层逻辑。
- 想给 Python / Web / UI 层提供稳定的结构化返回：优先走 pybind `try_*` 接口。
	- 这层会把 native 异常和 `failure` 统一折叠成稳定的 `error_code` / `error_reason` / `error_details`。

如果你在这几类接口之间切换，建议同时看本 README 的“失败传播路径”和 [docs/error-codes.md](docs/error-codes.md)。

对 Python / Web 接入再细分一点：

- Python 脚本或 notebook 想快速试算法，且失败时愿意直接抛异常中断：可以直接用 pybind 暴露的 `smooth(...)`。
- Python 服务、前端桥接层或批处理脚本想稳定收集失败信息：优先用 `try_smooth(...)` / `try_smooth_with_planner_esdf(...)`。
- 纯 Python SciPy 原型只关心运动学状态优化，不依赖 C++ Ceres / ESDF 集成：用 `include/constrained_smoother/kinematic_smoother.py` 里的 `try_optimize(...)`。
	- 它和 C++ 一样把输入第三列解释为 `direction_sign`，不会把它当成 yaw。
- Flask Web API 场景通常不应直接把 native 异常冒到前端，而应继续沿用结构化 `error` / `smooth_error` 返回。

### 后端切换最小改动指南

如果你只是想在几何版 `Smoother` 和运动学版 `KinematicSmoother` 之间切换，调用层通常不需要整体重写：

- 保持不变的部分：输入路径仍然是 `(x, y, direction_sign)`，输出路径仍然把第三个分量改写成 yaw。
- 保持不变的部分：C++ 层都支持异常式 `smooth(...)` 和带 `failure` 的结构化控制流。
- 保持不变的部分：pybind 层都提供 `smooth(...)`、`try_smooth(...)`、`smooth_with_planner_esdf(...)`、`try_smooth_with_planner_esdf(...)`，而且 `try_*` 返回面保持同构。
- 更值得重调的部分：`smooth_weight_sqrt`、`distance_weight_sqrt`、`costmap_weight_sqrt` 往往需要重新平衡，因为两个后端的状态空间和残差模型不同。
- 更值得重调的部分：运动学版现在使用独立的 `kinematic_curvature_weight_sqrt`、`kinematic_curvature_rate_weight_sqrt`，不要再直接复用几何版 `curvature_weight_sqrt`、`curvature_rate_weight_sqrt`。
- 更值得重调的部分：`keep_start_orientation` / `keep_goal_orientation` 在运动学版里会和 gear 一致性约束一起起作用，若目标朝向与段方向冲突，更容易触发后验拒绝。

一个实用经验是：

- 从几何版切到运动学版时，先保留调用面和错误处理面，只重新调残差权重与边界相关参数。
- 从运动学版切回几何版时，先确认你是否真的还需要显式状态变量 `theta / kappa / ds` 带来的约束表达能力。

## Dependencies

- Ceres Solver
- Eigen3
- Google Test (optional, for tests)
- pybind11 (optional, for Python bindings)
- Flask and NumPy (optional, for the Web Lab)

## ESDF Backends

The standalone project now keeps two ESDF implementations:

- `Exact` uses the vendored `distance_transform` implementation of the Felzenszwalb/Huttenlocher algorithm.
- `Approximate` keeps the older 8-neighbor propagation implementation as a simpler fallback.

Public `compute_esdf()` now means signed ESDF. Planner and smoother both use the same signed distance semantics, while the simpler backend remains available through the `use_exact` switch for comparison and debugging.

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

This produces `py_constrained_smoother.*.so` in `build/`.

### Build and run the Web Lab

```bash
./run_web_app.sh
```

The script creates a local uv-managed virtual environment, installs the Python dependencies needed by the Web Lab, rebuilds and installs the pybind11 module, and then starts the Flask app on port 5002.
It disables Flask's code reloader by default so the freshly rebuilt Python extension is not hot-reloaded while the shared object is still being replaced.

## Docs

The design notes under `docs/` can now be served as a local Material for MkDocs site from this package directory.

### Preview locally

```bash
./run_docs.sh
```

The script launches the local MkDocs Material preview on `127.0.0.1:8000` by default.
If that port is already busy, it automatically picks the next free port.
Override `CS_DOCS_HOST` or `CS_DOCS_PORT` if you want a different bind address.

### Build static docs

```bash
uvx --with mkdocs-material mkdocs build -f mkdocs.yml
```

The generated site is written to `site/`.

## Run Tests

```bash
cd build
ctest --output-on-failure
```

### 测试入口索引

如果你准备改某一层实现，通常可以先从这些测试入口开始：

- 几何残差和几何工具函数：`CostFunctionTest.*`、`UtilsTest.*`
- 几何路径重建 helper：`SmootherPathOpsTest.*`
- 运动学问题构建器：`KinematicSmootherProblemBuilderTest.*`
- 几何版顶层行为与失败传播：`SmootherTest.*`、`ErrorTest.*`
- 运动学版顶层行为与失败传播：`KinematicSmootherTest.*`
- 基础 costmap 行为：`CostmapTest.*`

这些测试当前都集中在 `test/test_smoother.cpp`，可以先用测试名定位，再回到对应实现文件。

### 改动到验证

如果你改完代码后不想重新判断该跑什么，可以直接按下面这张表执行：

- 改几何版顶层编排、path ops、problem builder、validator：在 `build/` 里重编 `test_smoother` 并跑 `ctest -R test_smoother`。
- 改运动学版顶层编排、problem builder、validator：同样先跑 `build/` 里的 `test_smoother`。
- 改 pybind 暴露层：先在 `build-py313/` 里重编 `py_constrained_smoother`，再回到 `build/` 跑 `test_smoother`。
- 改 README / 设计文档 / 错误码文档：至少做一次轻量一致性检查；如果同时碰了头文件或绑定层，还是按上面对应目标重编。
- 改 Web Lab 或 `/api/plan` 行为：除了上面的编译验证外，再实际启动 `./run_web_app.sh` 或手动运行 `web/app.py` 做接口冒烟。

## C++ Usage

```cpp
#include <cmath>
#include <vector>

#include "Eigen/Core"
#include "constrained_smoother/costmap2d.hpp"
#include "constrained_smoother/smoother.hpp"

constrained_smoother::Costmap2D costmap(100, 100, 0.05, 0.0, 0.0);
// Fill costmap with obstacle data here.

constrained_smoother::SmootherParams params;
params.smooth_weight_sqrt = std::sqrt(50.0);
params.costmap_weight_sqrt = std::sqrt(0.015);
params.distance_weight_sqrt = std::sqrt(0.0);
params.curvature_weight_sqrt = std::sqrt(30.0);
params.curvature_rate_weight_sqrt = std::sqrt(5.0);  // optional D3 curvature-rate proxy
params.max_curvature = 1.0 / 0.4;  // minimum turning radius = 0.4 m
params.keep_start_orientation = true;
params.keep_goal_orientation = true;

constrained_smoother::OptimizerParams opt_params;
opt_params.max_iterations = 50;

constrained_smoother::Smoother smoother;
smoother.initialize(opt_params);

// On input, z is direction_sign (+1 / -1).
std::vector<Eigen::Vector3d> path = {
	{0.5, 0.5, 1.0},
	{1.0, 0.7, 1.0},
	{1.6, 1.0, 1.0},
	{2.2, 1.4, 1.0},
};

Eigen::Vector2d start_dir(1.0, 0.0);
Eigen::Vector2d end_dir(1.0, 0.0);

smoother.smooth(path, start_dir, end_dir, &costmap, params);

// After smoothing, z has been rewritten to yaw in radians.
for (const auto & pose : path) {
	const double yaw = pose.z();
	(void)yaw;
}
```

## Python Usage

```python
import math
import numpy as np
import py_constrained_smoother as pcs

costmap = pcs.Costmap2D(100, 100, 0.05, 0.0, 0.0)

params = pcs.SmootherParams()
params.smooth_weight_sqrt = math.sqrt(50.0)
params.costmap_weight_sqrt = math.sqrt(0.015)
params.distance_weight_sqrt = math.sqrt(0.0)
params.curvature_weight_sqrt = math.sqrt(30.0)
params.curvature_rate_weight_sqrt = math.sqrt(5.0)  # optional D3 curvature-rate proxy
params.max_curvature = 2.5
params.keep_start_orientation = True
params.keep_goal_orientation = True

opt_params = pcs.OptimizerParams()
opt_params.max_iterations = 50

smoother = pcs.Smoother()
smoother.initialize(opt_params)

# On input, the third component is direction_sign, not yaw.
path = [
		np.array([0.5, 0.5, 1.0]),
		np.array([1.0, 0.7, 1.0]),
		np.array([1.6, 1.0, 1.0]),
		np.array([2.2, 1.4, 1.0]),
]
start_dir = np.array([1.0, 0.0])
end_dir = np.array([1.0, 0.0])

smoothed = smoother.smooth(path, start_dir, end_dir, costmap, params)
optimized_knot_count = smoother.get_last_optimized_knot_count()

# On output, smoothed[i][2] is yaw in radians.
print(optimized_knot_count, float(smoothed[0][2]))

safe_result = smoother.try_smooth(path, start_dir, end_dir, costmap, params)
if not safe_result["ok"]:
		print(
			safe_result["error_code"],
			safe_result["error_reason"],
			safe_result["error_message"],
			safe_result["error_details"],
		)
```

异常式和结构化接口的选择可以简单理解成：

```python
# 失败时直接抛异常，适合快速试验或测试。
smoothed = smoother.smooth(path, start_dir, end_dir, costmap, params)

# 失败时不抛异常，适合脚本、服务和 UI 层。
result = smoother.try_smooth(path, start_dir, end_dir, costmap, params)
if result["ok"]:
    smoothed = result["path"]
else:
    print(result["error_code"], result["error_reason"])
```

## Error Codes

The standalone project now uses stable error codes instead of relying on free-form messages alone.

- C++ exceptions expose `code()` and `codeString()`.
- Python bindings expose `ErrorCode`, `error_code_to_string(...)`, and `ERROR_*` constants.
- For Python callers that want structured failures without exception handling, prefer `try_smooth(...)` and `try_smooth_with_planner_esdf(...)`.
	- These methods return a dict with `ok`, `path`, `error_code`, `error_reason`, `error_message`, and `error_details`.
- The pure Python SciPy helper in `include/constrained_smoother/kinematic_smoother.py` now also exposes `try_optimize(...)`.
	- It returns `ok`, `states`, `optimizer_result`, `error_code`, and `error_message`.
- The Flask web API returns an `error` object on failures:
- `POST /api/plan` also performs one final rectangle-footprint validation on the smoothed candidate before accepting it.
	- If that post-validation fails, the API sets `smooth_success=false`, fills `smooth_error`, includes `candidate_rectangle_validation`, and still returns the smoothed candidate for visualization.
	- `final_rectangle_validation` always describes the path that is actually returned to the frontend.

See [docs/error-codes.md](docs/error-codes.md) for the full catalog and handling guidance.

```json
{
	"success": false,
	"message": "A* could not find a path because the goal pose lies inside a lethal obstacle cell.",
	"error": {
		"code": "CS_ASTAR_NO_PATH",
		"message": "A* could not find a path because the goal pose lies inside a lethal obstacle cell.",
		"source": "planner",
		"details": {
			"reason": "goal_in_lethal_obstacle",
			"goal": {
				"endpoint": "goal",
				"world_x": 18.0,
				"world_y": 18.0,
				"mx": 180,
				"my": 180,
				"cell_cost": 254
			}
		}
	}
}
```

## Web Lab

The Web Lab is an interactive scene editor and visualizer around the C++ A* planner and smoother.

Current behavior:

- The map is a synthetic 20 m x 20 m costmap with draggable rectangular lethal obstacles plus inflated safety cells.
- Start and goal markers are draggable.
- Obstacle rectangles are draggable.
- Left-drag on empty space pans the camera.
- Double-click or use Reset View to restore full-map framing.
- Slider and toggle changes trigger automatic replanning.
- The toolbar can switch between costmap and ESDF views.
- The sidebar exposes heading constraints, planner penalty settings, footprint mode, solver controls, and live metrics.
- The frontend shows a cursor inspector, optimized-point inspector, and a curvature chart for the current smoothed path.

### Run the Web Lab

```bash
# 1. Build the Python bindings first.
# 2. Activate your environment if needed.
cd my/constrained_smoother
python3 web/app.py
```

Open `http://localhost:5002` in your browser.

When launched from `my/constrained_smoother`, `web/app.py` adds both the package root and `build/` directory to `sys.path`, so an extra `PYTHONPATH` export is usually not required once the pybind module has been built.

### Web API Summary

- `GET /api/costmap`
	- Returns the current costmap grid, optional ESDF grid, and map metadata.
- `POST /api/obstacles`
	- Accepts `obstacle_rects_cells` and rebuilds the scene costmap.
	- Validation failures return `success=false` plus a structured `error.code`.
- `POST /api/plan`
	- Runs A* and then the constrained smoother.
	- Accepts start and goal positions, start and goal yaw constraints, footprint mode, planner penalty settings, and solver parameters.
	- Returns raw A* points, downsampled reference points, smoothed points, `opt_theta`, timing, lengths, and optimized knot counts.
	- Hard failures return `success=false` plus a structured `error` object.
	- Smoother-only failures keep `success=true` so the UI can fall back to the reference path, and additionally populate `smooth_error`.
	- If the optimizer returns a path but final rectangle-footprint validation rejects it, the response also includes `candidate_rectangle_validation` with the exact rejection reason.

The smoother route currently derives its planner safe distance from the shared hinge-loss threshold, and in point-robot mode it adds the point-robot radius on top of that shared threshold.
The standalone A* now also performs hard footprint feasibility checks: point-robot mode rejects cells whose ESDF clearance is smaller than the configured radius, and rectangle mode rejects any axis-aligned pose whose box footprint overlaps lethal cells. Rectangle A* checking intentionally ignores yaw.

## Original Source

Extracted from [navigation2/nav2_constrained_smoother](https://github.com/ros-navigation/navigation2/tree/main/nav2_constrained_smoother).

## License

Apache License 2.0
