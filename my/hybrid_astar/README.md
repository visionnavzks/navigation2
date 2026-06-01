# Hybrid A* Planner 技术文档

## 1. 概述

`hybrid_astar` 是一个独立的 Hybrid A* 路径规划器，支持 Ackermann 车辆运动学约束（Dubin / Reeds-Shepp 模型）。命名空间为 `hybrid_astar`，不依赖 Nav2 或其他 ROS 包，可独立编译和集成。

### 运动模型

| 模型 | 说明 |
|---|---|
| `DUBIN` | 仅前进，最小转弯半径约束 |
| `REEDS_SHEPP` | 前进 + 倒车，最小转弯半径约束 |

### 与原 smac_planner 的区别

- 命名空间：`smac_planner` → `hybrid_astar`
- 移除 `Node2D`、`NodeLattice` 及所有相关模板实例化
- 移除 `MotionModel::TWOD`、`STATE_LATTICE`、`OMNI`
- 移除 `LatticeMetadata`、`MotionPrimitive` 等 Lattice 专用类型
- 单库编译，无共享模板依赖

---

## 2. 目录结构

```
hybrid_astar/
├── CMakeLists.txt
├── build.sh
├── include/my/hybrid_astar/
│   ├── a_star.hpp                 # A* 核心算法（模板类）
│   ├── analytic_expansion.hpp     # 解析扩展（Dubin/Reeds-Shepp 直连）
│   ├── collision_checker.hpp      # 碰撞检测（半径/轮廓模式）
│   ├── constants.hpp              # 运动模型枚举、代价常量
│   ├── costmap_2d.hpp             # 独立 2D 代价地图
│   ├── costmap_downsampler.hpp    # 代价地图降采样
│   ├── distance_heuristic.hpp     # 运动学距离启发（OMPL 查找表）
│   ├── goal_manager.hpp           # 目标状态管理（多目标/全方向）
│   ├── node_basic.hpp             # 优先队列轻量包装
│   ├── node_hybrid.hpp            # Hybrid A* 节点 + 运动表
│   ├── obstacle_heuristic.hpp     # 障碍物启发（D* Lite 风格）
│   ├── smac_planner_hybrid.hpp    # 顶层规划器接口
│   ├── smoother.hpp               # 路径平滑器
│   ├── types.hpp                  # 数据结构定义
│   └── utils.hpp                  # 工具函数
├── src/                           # 对应实现文件
├── test/                          # GTest 单元测试
└── thirdparty/
    └── robin_hood.h               # 快速哈希表
```

---

## 3. 架构总览

```
SmacPlannerHybrid                    顶层规划器
├── AStarAlgorithm<NodeHybrid>       A* 搜索核心
│   ├── NodeHybrid                   SE2 搜索节点
│   │   ├── HybridMotionTable        运动原语预计算
│   │   ├── ObstacleHeuristic        障碍物距离场
│   │   └── DistanceHeuristic        运动学距离查找表
│   ├── NodeBasic<NodeHybrid>        优先队列元素包装
│   ├── AnalyticExpansion<NodeHybrid> 解析路径扩展
│   └── GoalManager<NodeHybrid>      目标状态管理
├── GridCollisionChecker             碰撞检测
├── CostmapDownsampler               代价地图降采样
└── Smoother                         路径平滑
```

---

## 4. 核心数据结构

### 4.1 搜索状态空间

搜索空间为三维 SE2：`(x, y, θ)`，其中：
- `(x, y)`：栅格坐标（连续浮点）
- `θ`：角度 bin 索引（离散整数，0 ~ `angle_quantization-1`）

线性索引映射：

```
index = angle + x × angle_quantization + y × width × angle_quantization
```

### 4.2 NodeHybrid — 搜索节点

```cpp
class NodeHybrid {
    NodeHybrid * parent;                    // 父节点（回溯路径）
    Coordinates pose;                       // {x, y, theta} 连续坐标

    float _cell_cost;                       // 代价地图值（碰撞检测缓存）
    float _accumulated_cost;                // g(n)：从起点到此节点的累计代价
    uint64_t _index;                        // 线性索引
    bool _was_visited;                      // 是否已从队列中取出扩展
    unsigned int _motion_primitive_index;   // 到达此节点的运动原语索引
    TurnDirection _turn_dir;                // 转向方向
    bool _is_node_valid;                    // 碰撞检测结果缓存
    NodeContext * _ctx;                     // 共享上下文指针
};
```

`NodeContext` 绑定共享状态：

```cpp
struct NodeContext {
    HybridMotionTable motion_table;                    // 运动原语表
    std::unique_ptr<ObstacleHeuristic> obstacle_heuristic;  // 障碍物启发
    std::unique_ptr<DistanceHeuristic<NodeHybrid>> distance_heuristic;  // 距离启发
};
```

### 4.3 HybridMotionTable — 运动原语预计算

初始化时根据运动模型（Dubin/Reeds-Shepp）和最小转弯半径，预计算：

| 字段 | 维度 | 说明 |
|---|---|---|
| `delta_xs[prim][angle]` | `[N_prim × N_angle]` | 每个运动原语在每个角度下的 X 偏移 |
| `delta_ys[prim][angle]` | `[N_prim × N_angle]` | 每个运动原语在每个角度下的 Y 偏移 |
| `trig_values[angle]` | `[N_angle]` | 每个角度的 `{cos, sin}` 缓存 |
| `travel_costs[prim]` | `[N_prim]` | 每个运动原语的基础行进代价 |

运动原语数量：
- Dubin：3 个（直行、左转、右转），可插值扩展到 `3 + 2×(N-1)` 个
- Reeds-Shepp：6 个（直行/倒车 + 4 个转弯），可插值扩展到 `6 + 4×(N-1)` 个

### 4.4 Graph — 节点存储

```cpp
typedef robin_hood::unordered_node_map<uint64_t, NodeHybrid> Graph;
```

- **哈希表**，key 为线性索引，value 为 `NodeHybrid` 对象
- 按需插入（`addToGraph` 只在访问时创建节点）
- 初始 `reserve(100000)`，按需增长
- 搜索结束后 `clearGraph()` 通过 `std::swap` 释放

---

## 5. 搜索算法

### 5.1 主循环 (`AStarAlgorithm::createPath`)

```
1. 初始化 open list（优先队列），将起点加入
2. while (iterations < max_iterations && queue 非空):
   a. 取 f(n) 最小的节点 current_node
   b. 检查是否已访问 → 跳过重复节点
   c. 标记为已访问
   d. 尝试解析扩展（Analytic Expansion）
   e. 检查是否到达目标 → 回溯路径
   f. 扩展邻居节点:
      - 对每个运动原语计算投影坐标
      - 碰撞检测
      - 计算 g_cost = current.g + traversal_cost
      - 如果 g_cost < neighbor.g → 更新代价和父节点
      - 加入 open list: f = g + h
3. 返回最短容差路径或超时
```

### 5.2 代价计算

#### 5.2.1 单元格代价 `_cell_cost`

从 `Costmap2D` 读取（0~255），碰撞检测时缓存到节点中，避免重复查询。

#### 5.2.2 行进代价 `getTraversalCost`

```
travel_cost_raw = travel_costs[motion_index]
                  × (travel_distance_reward + cost_penalty × normalized_cost)

其中:
  normalized_cost = child_cell_cost / 252.0
  travel_distance_reward = 1.0 - retrospective_penalty
```

根据转向方向加罚：

| 条件 | 代价倍数 |
|---|---|
| 直行 / 倒车 | `× 1.0` |
| 继续同方向转弯 | `× non_straight_penalty`（默认 1.05） |
| 换转弯方向 | `× (non_straight_penalty + change_penalty)` |
| 倒车方向 | 再 `× reverse_penalty`（默认 2.0） |

#### 5.2.3 启发值 `h(n)`

```
h(n) = max(obstacle_heuristic, distance_heuristic)
```

两者取 max 保证既考虑障碍物绕行（admissible），又考虑运动学约束（consistent）。

### 5.3 邻居扩展

`NodeHybrid::getNeighbors` 对每个运动原语：

1. 计算投影坐标 `(Δx + node.x, Δy + node.y, new_heading)`
2. 线性索引 → 从 graph 中获取/创建邻居节点
3. 如果未访问且碰撞检测通过 → 加入邻居列表

---

## 6. 启发函数

### 6.1 障碍物启发 (`ObstacleHeuristic`)

**本质：从 goal 出发的 Dijkstra 搜索**，在 2x 降采样的代价地图上计算。

#### 数据结构

```
obstacle_heuristic_lookup_table_[index]   // 降采样地图上的距离值
  - 负值 = open set（未关闭）
  - 正值 = closed set（已确定最短距离）
  - 0.0  = 未访问
```

#### 单步代价

```
线性模式:  travel_cost = 邻接距离 × (1.0 + cost_penalty × cost / 252.0)
二次模式:  travel_cost = 邻接距离 × (1.0 + cost_penalty × cost² / 252²)
```

- 邻接距离：上下左右 = 1.0，对角线 = √2
- `cost`：代价值地图原始值（0~252）
- `cost_penalty`：调参系数（`search_info.cost_penalty`，默认 2.0）

#### 关键特性

| 特性 | 实现 |
|---|---|
| 降采样 | 默认 2x，2×2 窗口取最小 cost（更乐观） |
| 障碍物阻挡 | `cost >= INSCRIBED_COST`（253）直接跳过 |
| 边界保护 | 距地图边缘 ≤3 格的节点跳过 |
| 动态扩展 | 按需扩展到当前查询节点，缓存复用 |
| 启发加速 | 用 2D 欧几里得距离做 A* 启发，加速 Dijkstra |

#### 查询流程

```
getObstacleHeuristic(node_coords):
  1. 查表 → 如果 > 0，直接返回（已缓存）
  2. 如果 < 0，在 open set 中，用 2D 距离重排优先级
  3. 从 queue 中扩展，直到到达 start_index 或 queue 为空
  4. 返回缓存值
```

### 6.2 距离启发 (`DistanceHeuristic`)

**本质：Dubin/Reeds-Shepp 距离查找表**，预计算 OMPL 状态空间距离。

#### 查找表

```
dist_heuristic_lookup_table_[x][y][heading]
维度: size_lookup × ceil(size_lookup/2) × num_angle_quantization
```

利用象限对称性，只存 2/4 象限（Y 取绝对值镜像）。

#### 查询流程

```
getDistanceHeuristic(node_coords, goal_coords):
  1. 旋转平移：将 node 坐标变换到以 goal 为原点的相对坐标系
  2. 如果在查找窗口内 → 查表
  3. 如果超出窗口但 obstacle_heuristic ≤ 0 → 实时 OMPL 计算
  4. 否则返回 0
```

---

## 7. 解析扩展 (`AnalyticExpansion`)

在 A* 搜索过程中，定期尝试用 Dubin/Reeds-Shepp 曲线直接连接当前节点到目标：

```
tryAnalyticExpansion(current_node):
  1. 计算当前节点到目标的距离
  2. 按 d / analytic_expansion_ratio 频率尝试
  3. 对每个目标节点:
     a. 用 OMPL 状态空间插值路径
     b. 沿路径每 √2 间隔采样，碰撞检测
     c. 计算代价评分
  4. 尝试增大转弯半径（0.5 步进，最多 4x）优化路径
  5. 选择评分最低的解析路径
```

#### 路径评分

```
score = Σ (distance × (1.0 + cost_penalty × node_cost / 252.0))
```

优先选择低代价路径。

#### 方向变化计数（Reeds-Shepp）

```cpp
countDirectionChanges(path):  // 统计路径中前进/后退切换次数
  越少越好 → 操作更简单
```

---

## 8. 碰撞检测 (`GridCollisionChecker`)

### 8.1 两种模式

| 模式 | `use_radius` | 检测方式 |
|---|---|---|
| 半径模式 | `true` | 只检查中心点，`cost >= 253` 即碰撞 |
| 轮廓模式 | `false` | 旋转后的多边形顶点逐一检测 |

### 8.2 轮廓模式流程

```
setFootprint(footprint, angle_quantization):
  对每个角度 bin (0 ~ 71):
    oriented_footprints_[bin] = rotate(footprint, bin × 2π/72)

inCollision(x, y, angle_bin):
  1. 边界检查 → 超出地图 = 碰撞
  2. 中心点代价检查:
     - < possible_collision_cost 且 > 0 → 快速通过
     - == UNKNOWN_COST 且不穿越未知 → 碰撞
     - == INSCRIBED_COST 或 OCCUPIED_COST → 碰撞
  3. 取该角度的旋转轮廓:
     oriented_footprint = oriented_footprints_[angle_bin]
  4. 遍历每个轮廓顶点:
     - mapToWorld() 转世界坐标
     - worldToMap() 转栅格坐标
     - 超出地图 → 碰撞
     - cost >= OCCUPIED_COST (254) → 碰撞
     - cost == UNKNOWN_COST 且不穿越未知 → 碰撞
  5. 所有点通过 → 无碰撞
```

### 8.3 `circumscribed_cost` 优化

如果中心点代价 < `circumscribed_cost`，说明机器人完全在自由空间内，跳过轮廓检测。适用于 `use_radius = false` 但希望快速排除的情况。

---

## 9. 代价地图

### 9.1 Costmap2D

独立的 2D 代价地图，不依赖 Nav2 的 `Costmap2D`：

```cpp
class Costmap2D {
    unsigned int size_x_, size_y_;     // 栅格尺寸
    double resolution_;                // 分辨率 (m/cell)
    double origin_x_, origin_y_;       // 原点世界坐标
    std::vector<unsigned char> cost_map_;  // 代价数据
};
```

### 9.2 CostmapDownsampler

降采样代价地图，用于加速障碍物启发计算：

```
downsample(factor):
  对每个新 cell (new_mx, new_my):
    取原始地图 [x_offset, x_offset+factor) × [y_offset, y_offset+factor) 区域:
      - use_min_cost_neighbor=true → 取最小值（更乐观）
      - use_min_cost_neighbor=false → 取最大值（更保守）
```

### 9.3 代价值含义

| 值 | 常量 | 含义 |
|---|---|---|
| 0 | `FREE_COST` | 自由空间 |
| 1~252 | `MAX_NON_OBSTACLE_COST` | 代价值（越大约靠近障碍物） |
| 253 | `INSCRIBED_COST` | 内切圆区域（机器人完全覆盖） |
| 254 | `OCCUPIED_COST` | 占据（障碍物） |
| 255 | `UNKNOWN_COST` | 未知区域 |

---

## 10. 路径平滑 (`Smoother`)

### 10.1 算法

基于加权平均的迭代平滑：

```
new_x = x + w_data × (original_x - x) + w_smooth × (x_prev + x_next - 2×x)
new_y = y + w_data × (original_y - y) + w_smooth × (y_prev + y_next - 2×y)
```

- `w_data`：数据保真权重（保持接近原始路径）
- `w_smooth`：平滑权重（减少曲率）
- 迭代直到变化 < tolerance 或达到最大次数

### 10.2 约束

- 碰撞检测：平滑后如果新位置代价 > 252 且 != 255 → 回退到上次平滑结果
- 方向分段：检测方向突变（> π/2），分段独立平滑
- 边界条件：用 Dubin 曲线强制起始和终止方向

### 10.3 参数 (`SmootherParams`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `tolerance_` | 1e-3 | 收敛阈值 |
| `max_its_` | 200 | 最大迭代次数 |
| `w_data_` | 0.3 | 数据保真权重 |
| `w_smooth_` | 0.3 | 平滑权重 |
| `do_refinement_` | false | 是否迭代精炼 |
| `refinement_num_` | 3 | 精炼次数 |

---

## 11. 目标管理 (`GoalManager`)

支持多种目标朝向模式：

| 模式 | 说明 |
|---|---|
| `DEFAULT` | 单个目标，指定朝向 |
| `BIDIRECTIONAL` | 两个目标：原始朝向 + 180° 反向 |
| `ALL_DIRECTION` | 所有角度 bin 都作为目标 |

#### 流程

```
1. setGoal() → 根据模式创建目标节点，加入 _goals_state
2. removeInvalidGoals() → 碰撞检测，移除无效目标
3. prepareGoalsForAnalyticExpansion() → 分为 coarse/fine 两组
4. isGoal(node) → 判断是否到达任一目标
```

---

## 12. 规划器接口

### 12.1 配置 (`SmacPlannerHybridConfig`)

```cpp
struct SmacPlannerHybridConfig {
    // 搜索参数
    bool downsample_costmap{false};        // 是否降采样代价地图
    int downsampling_factor{1};            // 降采样倍数
    unsigned int angle_quantization_bins{72};  // 角度量化数（5° 一个 bin）
    float tolerance{0.25f};                // 容差 (m)
    bool allow_unknown{true};              // 是否穿越未知区域
    int max_iterations{1000000};           // 最大迭代次数
    int max_on_approach_iterations{1000};  // 接近目标时的最大迭代
    int terminal_checking_interval{5000};  // 超时检查间隔
    double max_planning_time{5.0};         // 最大规划时间 (s)
    double lookup_table_size{20.0};        // 距离启发表大小 (m)
    bool debug_visualizations{false};      // 是否输出展开日志
    std::string motion_model_for_search{"DUBIN"};  // 运动模型
    std::string goal_heading_mode{"DEFAULT"};      // 目标朝向模式
    int coarse_search_resolution{1};       // ALL_DIRECTION 粗搜索分辨率

    // 搜索调参
    SearchInfo search_info;                 // 见 5.2 节

    // 平滑参数
    SmootherParams smoother_params;         // 见 10.3 节

    // 机器人参数
    Footprint robot_footprint;              // 轮廓顶点列表
    bool use_radius{false};                 // 是否用半径碰撞检测
    double circumscribed_cost{-1.0};       // 外接圆代价（-1 自动计算）
    double inflation_radius{0.5};           // 膨胀半径 (m)
    double circumscribed_radius{0.5};       // 外接圆半径 (m)
};
```

### 12.2 使用示例

```cpp
#include "my/hybrid_astar/smac_planner_hybrid.hpp"

using namespace hybrid_astar;

// 1. 创建代价地图
Costmap2D costmap(size_x, size_y, resolution, origin_x, origin_y);
// ... 填充代价数据 ...

// 2. 配置规划器
SmacPlannerHybridConfig config;
config.motion_model_for_search = "REEDS_SHEPP";
config.search_info.minimum_turning_radius = 1.0;
config.angle_quantization_bins = 72;
config.tolerance = 0.2;

// 3. 设置机器人轮廓
Footprint footprint = {{-0.2, -0.3}, {0.2, -0.3}, {0.2, 0.3}, {-0.2, 0.3}};
config.robot_footprint = footprint;
config.use_radius = false;

// 4. 初始化
SmacPlannerHybrid planner;
planner.configure(&costmap, config);

// 5. 规划
Pose start = {0.0, 0.0, 0.0};
Pose goal = {5.0, 3.0, 1.57};
Path path = planner.createPlan(start, goal);
```

---

## 13. 预计算表总结

| 表 | 所属类 | 维度 | 内容 |
|---|---|---|---|
| `delta_xs` | HybridMotionTable | `[N_prim × N_angle]` | 运动原语 X 偏移 |
| `delta_ys` | HybridMotionTable | `[N_prim × N_angle]` | 运动原语 Y 偏移 |
| `trig_values` | HybridMotionTable | `[N_angle]` | 角度 cos/sin 缓存 |
| `travel_costs` | HybridMotionTable | `[N_prim]` | 运动原语基础代价 |
| `obstacle_heuristic_lookup_table_` | ObstacleHeuristic | `[ds_x × ds_y]` | Dijkstra 距离场 |
| `dist_heuristic_lookup_table_` | DistanceHeuristic | `[x × y/2 × θ]` | Dubin/RS 距离 |
| `oriented_footprints_` | GridCollisionChecker | `[N_angle × N_pts]` | 旋转后轮廓 |
| `_graph` | AStarAlgorithm | 动态哈希表 | 搜索节点 |
| `_queue` | AStarAlgorithm | 优先队列 | Open list |

---

## 14. 编译

```bash
cd my/hybrid_astar
chmod +x build.sh
./build.sh
```

依赖：`steering_functions_lite`（`../cpp-dubins-rs` 子目录）、`GTest`（仅测试）

---

## 15. 测试

| 测试 | 验证内容 |
|---|---|
| `test_collision_checker` | 碰撞检测、代价查询、未知区域穿越 |
| `test_a_star` | Hybrid A* 基本规划 |
| `test_nodehybrid` | NodeHybrid 基本功能 |
| `test_smoother` | 路径平滑（直线、空路径、短路径） |
| `test_utils` | 坐标转换、外接圆代价计算 |
| `test_goal_manager` | 目标添加/移除/清理/解析扩展分组 |
