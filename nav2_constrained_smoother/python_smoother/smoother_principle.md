# Constrained Smoother 原理文档

`nav2_constrained_smoother` 使用数值优化方法对路径进行平滑处理。它通过构建一个非线性最小二乘问题，在保持路径尽可能接近原始路径的同时，优化路径的平滑度、曲率连续性，并避开障碍物。

## 1. 核心算法

该平滑器使用 **Levenberg-Marquardt (LM)** 算法求解非线性最小二乘问题。
目标是寻找一组优化的路径点 $P = \{p_0, p_1, ..., p_N\}$，使得总代价函数 $J(P)$ 最小。

$$
\min_{P} J(P) = \min_{P} \sum_{i} r_i(P)^2
$$

其中 $r_i(P)$ 是各项残差（Residuals）。

Python 实现中使用了 `scipy.optimize.least_squares`。

## 2. 优化变量

优化变量为路径中间点的坐标：
$$ P_{opt} = \{p_1, p_2, ..., p_{N-1}\} $$
其中 $p_i = (x_i, y_i)$。

起点 $p_0$ 和终点 $p_N$ 通常由输入路径固定，以保证连接性。如果启用了 `keep_start_orientation` 或 `keep_goal_orientation`，则可能会固定 $p_1$ 或 $p_{N-1}$ 以保持端点切向方向。

## 3. 代价函数 (Cost Function)

总代价由以下几部分组成：

### 3.1 平滑代价 (Smoothness Cost)

平滑代价旨在最小化路径的离散加加速度（Jerky motion），通常通过最小化二阶差分来实现。这使得路径点分布更加均匀且直线更直。

$$
r_{smooth, i} = w_{smooth} \cdot \| (p_{i+1} - p_i) - (p_i - p_{i-1}) \|
$$

展开后即为：
$$
r_{smooth, i} = w_{smooth} \cdot \| p_{i+1} - 2p_i + p_{i-1} \|
$$

### 3.2 原始路径偏离代价 (Original Path Adherence)

为了防止优化后的路径偏离全局规划器生成的原始路径太远，引入偏离代价。

$$
r_{dist, i} = w_{dist} \cdot \| p_i - p_{original, i} \|
$$

### 3.3 曲率代价 (Curvature Constraint)

为了满足机器人的最小转弯半径 $R_{min}$，需要限制路径的最大曲率 $\kappa_{max} = \frac{1}{R_{min}}$。
曲率 $\kappa_i$ 是通过三个连续点 $p_{i-1}, p_i, p_{i+1}$ 计算的（Menger Curvature）。

计算公式为：
$$
\kappa_i = \frac{4 \cdot \text{Area}(p_{i-1}, p_i, p_{i+1})}{|p_{i-1}-p_i| \cdot |p_i-p_{i+1}| \cdot |p_{i-1}-p_{i+1}|}
$$

残差项为软约束（Soft Constraint）：
$$
r_{curve, i} = \begin{cases} 
\sqrt{w_{curve}} \cdot (\kappa_i - \kappa_{max}) & \text{if } \kappa_i > \kappa_{max} \\
0 & \text{otherwise}
\end{cases}
$$

这表示只有当曲率超过最大允许值时才产生代价。

### 3.4 障碍物/代价地图代价 (Costmap Cost)

为了确保路径不发生碰撞并尽可能远离高代价区域，引入代价地图惩罚。

对于路径上的每个点 $p_i$，其代价为：
$$
r_{cost, i} = \sqrt{w_{cost}} \cdot \text{Costmap}(p_i)
$$

**Footprint 扩展：**
如果定义了机器人的 `cost_check_points`（例如矩形边角），则会对每个 footprint 点进行检查。
对于路径点 $p_i$ 及其切向方向 $\theta_i$，将 footprint 点 $f_j$ 变换到世界坐标系：
$$
p_{world, j} = R(\theta_i) \cdot f_j + p_i
$$
此时总代价包含所有 footprint 点的代价：
$$
r_{cost, i, j} = \sqrt{w_{cost} \cdot w_{point, j}} \cdot \text{Costmap}(p_{world, j})
$$

## 4. 总结

最终的优化问题是寻找 $P$，使得所有残差平方和最小：

$$
J(P) = \sum \|r_{smooth}\|^2 + \sum \|r_{dist}\|^2 + \sum \|r_{curve}\|^2 + \sum \|r_{cost}\|^2
$$

通过调整权重参数 ($w_{smooth}, w_{dist}, w_{curve}, w_{cost}$)，可以平衡路径的平滑度、跟随精度和安全性。
