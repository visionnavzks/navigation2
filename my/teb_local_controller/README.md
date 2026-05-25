# TEB Local Controller Notes

这个目录实现了一个基于 CasADi `Opti` 的简化 TEB/MPC 轨迹跟踪器，用于验证以下几个核心想法：

- 参考轨迹由直线段和圆弧段组成。
- 轨迹优化的时间步长 `dt` 可以伸缩，而不是固定常数。
- 控制量选择为 `dt / jerk / dkappa`。
- 优化目标既包含几何跟踪，也包含速度、加速度、曲率和控制平滑性。
- demo 中使用的参考轨迹会先从初始状态在参考路径上的投影点开始对齐，再送入优化器。

## 文件说明

- `teb_mpc.py`
  核心状态、参考轨迹构造器和 `TEBMPCController` 求解器实现。
- `demo_support.py`
  demo 专用参考轨迹、随机初始状态、投影对齐和参数摘要。
- `app.py`
  Flask Web demo。

## 状态、控制量与参考量

优化器中的状态定义为：

$$
X_i = [x_i, y_i, \theta_i, v_i, a_i, \kappa_i]
$$

其中：

- $x, y$：平面位置
- $\theta$：航向角
- $v$：纵向速度
- $a$：纵向加速度
- $\kappa$：路径曲率

控制量定义为：

$$
U_i = [dt_i, jerk_i, d\kappa_i]
$$

其中：

- $dt_i$：第 $i$ 段时间步长，可优化、可伸缩
- $jerk_i$：加加速度
- $d\kappa_i$：曲率导数

参考轨迹 `ReferenceTrajectory` 包含：

$$
\{x_{ref}, y_{ref}, \theta_{ref}, v_{ref}, a_{ref}, \kappa_{ref}, s_{ref}, dt_{ref}\}
$$

其中 `s_ref` 是累计弧长，`dt_ref` 是参考时间步长；若构造参考轨迹时未显式传入，则按 `ds / cruise_speed` 推导。当前实现中它主要用于给 `dt` 提供初始尺度，并用于构造停车参考。

## 轨迹生成

参考轨迹由 `build_reference_trajectory()` 构造。

### 直线段

对长度为 `length` 的直线段，按照弧长采样间隔 `ds` 离散：

- 航向角保持不变
- 曲率为零
- 每个采样点的弧长累计增加

### 圆弧段

对半径为 `radius`、转角为 `angle` 的圆弧段：

- 曲率为 $\kappa = \pm 1 / |radius|$
- 按弧长均匀离散
- 利用圆弧几何关系更新位置和航向

这样可以得到一条几何上连续、曲率已知的参考路径。

## demo 参考对齐

demo 中不会直接使用整条原始参考路径，而是先做一次“初始状态到参考路径的投影对齐”。

步骤如下：

1. 在参考折线上找到初始状态 `(x, y)` 的最近投影点；如果初始状态落在参考路径首尾之外，则允许在首尾切向上做线性延长后再投影。
2. 计算该投影点对应的参考弧长 $s_0$。
3. 之后不再直接裁剪离散点，而是通过连续的弧长采样器生成新的参考轨迹：
  - 当 $s < 0$ 时，位置沿起点切向线性延长，航向固定为起点航向。
  - 当 $0 \le s \le s_{end}$ 时，按原始参考轨迹插值采样。
  - 当 $s > s_{end}$ 且起点仍在原始终点之前时，重采样在原始终点处截断，因此参考点数量可能减少。
  - 当 $s > s_{end}$ 且起点本身已经在终点外时，不再继续延长原路径，而是从当前状态平滑收敛到终点切向的延长线，并沿这条延长线减速停车；这段停车参考的航向和曲率由几何轨迹反推，并额外按 `max_lat_accel`、`max_kappa`、`max_dkappa` 做整形。
4. 这条对齐后的参考轨迹作为优化器输入。

这样做的原因是：

- 第一个优化节点不应该总是被拉回原始参考路径的第一个离散点。
- 对于随机初始状态或偏离参考的情况，从投影点开始更合理。
- 这也是后续做基于弧长 $s$ 的参考跟踪的自然过渡版本。

## 动力学离散模型

优化器采用离散时间单轨近似模型，状态转移在 `TEBMPCController.solve()` 中定义。

### 加速度和曲率更新

$$
a_{i+1} = a_i + dt_i \cdot jerk_i
$$

$$
\kappa_{i+1} = \kappa_i + dt_i \cdot d\kappa_i
$$

### 速度更新

$$
v_{i+1} = v_i + dt_i \cdot a_i + \frac{1}{2} dt_i^2 \cdot jerk_i
$$

### 中点近似

为了减少直接欧拉离散误差，位置和航向更新使用中点量：

$$
v_{mid} = v_i + \frac{1}{2} dt_i a_i + \frac{1}{8} dt_i^2 jerk_i
$$

$$
\kappa_{mid} = \kappa_i + \frac{1}{2} dt_i d\kappa_i
$$

### 航向更新

$$
\theta_{i+1} = \theta_i + dt_i \cdot v_{mid} \cdot \kappa_{mid}
$$

### 位置更新

当中点曲率接近零时，使用直线推进：

$$
x_{i+1} = x_i + dt_i \cdot v_{mid} \cdot \cos(\theta_i)
$$

$$
y_{i+1} = y_i + dt_i \cdot v_{mid} \cdot \sin(\theta_i)
$$

否则使用常曲率圆弧的解析推进：

$$
x_{i+1} = x_i + \frac{\sin(\theta_{i+1}) - \sin(\theta_i)}{\kappa_{mid}}
$$

$$
y_{i+1} = y_i + \frac{\cos(\theta_i) - \cos(\theta_{i+1})}{\kappa_{mid}}
$$

这个模型的特点是：

- 比最简单的前向欧拉更平滑
- 能把 `dt` 的伸缩直接作用到几何推进距离上
- 在较大 `dt` 下比简单中点直线推进更贴近单轨几何运动
- 能把 `jerk` 和 `d\kappa` 作为自然控制量放进优化器

## 代价函数

总代价由三部分组成：

$$
J = J_{track} + J_{control} + J_{terminal}
$$

### 1. 跟踪代价

对每个节点 $i$：

$$
J_{track} = \sum_i \Big[
w_{pos}((x_i-x_{ref,i})^2 + (y_i-y_{ref,i})^2)
+ w_{speed}(v_i-v_{ref,i})^2
\Big]
$$

解释：

- 位置误差使用二次项
- 中间节点只跟踪位置和速度
- 航向误差不在中间阶段跟踪，而是只放到终端代价中

### 2. 控制代价

对每段控制量 $i$：

$$
J_{control} = \sum_i \Big[
w_{time}dt_i
+ w_{jerk} jerk_i^2
+ w_{d\kappa} d\kappa_i^2
\Big]
 + \sum_{i>0} w_{dt}(dt_i-dt_{i-1})^2
$$

解释：

- `w_time` 直接推动总时间变短
- `w_dt` 控制相邻时间步长的平滑程度
- `w_jerk` 抑制过大的加加速度
- `w_dkappa` 抑制曲率变化过快

### 3. 终端代价

末端节点额外加重：

$$
J_{terminal} = w_{terminal} \cdot \Big[
(x_N-x_{ref,N})^2 + (y_N-y_{ref,N})^2
+ (v_N-v_{ref,N})^2
+ (1-\cos(\theta_N-\theta_{ref,N}))
\Big]
$$

这样可以防止优化只顾中间过程而末端对不齐。

## 约束条件

### 初始状态约束

优化器强制第一个状态节点等于当前初始状态：

- $x_0, y_0, \theta_0$
- $v_0, a_0, \kappa_0$

### 动态约束

所有相邻状态必须满足上面的离散动力学方程。

### 盒约束

代码中当前使用的约束包括：

- $dt_{min} \le dt_i \le dt_{max}$
- $0 \le v_i \le max\_speed$
- $|a_i| \le max\_accel$
- $|v_i^2 \kappa_i| \le max\_lat\_accel$
- $|jerk_i| \le max\_jerk$
- $|\kappa_i| \le max\_kappa$
- $|d\kappa_i| \le max\_dkappa$

这些约束在 `opti.bounded(...)` 中统一设置。

## 参数说明

控制器默认参数包括：

### 时间相关

- `dt_ref = 0.1`（控制器参数默认值；若 `build_reference_trajectory()` 未显式传入 `dt_ref`，则参考轨迹会按 `ds / cruise_speed` 自动推导）
- `selection_length = 0.0`（从当前投影点开始选取多少米参考路径用于优化；`0` 表示一直取到当前路径终点）
- `extra_points = 0`（对齐后的参考轨迹点数调整量；正值增加点，负值减少点，最终至少保留 2 个点）
- `dt_min = 0.03`
- `dt_max = 0.35`

### 状态/控制边界

- `max_speed = 2.5`
- `max_accel = 1.0`
- `max_lat_accel = 1.5`
- `max_jerk = 3.0`
- `max_kappa = 2.0`
- `max_dkappa = 1.5`

### 权重

- `w_pos = 0.0`
- `w_pos_terminal = 30.0`（过程终点）
- `w_theta = 15.0`（过程终点）
- `w_speed = 0.0`
- `w_time = 1.0`
- `w_speed_terminal = 0.0`（过程终点）
- `w_pos_terminal_real = 60.0`（真实路径终点）
- `w_theta_terminal_real = 15.0`（真实路径终点）
- `w_speed_terminal_real = 60.0`（真实路径终点）
- `w_dt = 10.0`
- `w_jerk = 0.5`
- `w_dkappa = 0.5`

说明：`w_*_terminal` 与 `w_*_terminal_real` 是两组互斥参数，不叠加。若当前优化目标就是原始路径真正终点，则只使用 `w_*_terminal_real`；若当前优化终点不是原始路径真正终点，则只使用 `w_*_terminal`。

### IPOPT

- `ipopt_max_iter = 500`
- `ipopt_tol = 1e-6`
- `ipopt_print_level = 0`

## 求解流程

一次典型求解流程如下：

1. 构造参考轨迹 `ReferenceTrajectory`
2. 给定当前初始状态 `VehicleState`
3. 将初始状态投影到参考轨迹上
4. 从投影点开始构建对齐后的参考轨迹
5. 使用 `TEBMPCController.solve(...)` 建立 CasADi `Opti` 问题
6. 调用 IPOPT 求解
7. 返回优化状态、控制量和各项代价值

返回结果中主要包括：

- `x, y, theta`
- `v, a, kappa`
- `dt, jerk, dkappa`
- `time`：由 `dt` 累加得到的时间轴
- `costs`：`track / control / terminal / total`

## demo 说明

### Web demo

运行：

```bash
source .venv/bin/activate
python my/teb_local_controller/app.py
```

访问：

```text
http://127.0.0.1:5002
```

## 当前局限

当前实现是一个面向验证的简化版本，不是完整生产控制器。主要局限包括：

- 参考仍然是离散轨迹，不是连续弧长参数化的 `ref(s)` 求值器
- 代价函数仍按离散参考数组索引跟踪，而不是显式优化进度变量 $s_i$
- 没有障碍物、碰撞、安全距离或拓扑切换约束
- 没有多候选带优化，只做单条轨迹的时间弹性优化
- 没有真正 receding-horizon 的滚动窗接口封装

## 下一步建议

如果要进一步把这个实现推向更合理的 MPC/TEB 形式，优先级建议是：

1. 把参考从离散数组升级成基于弧长 $s$ 的连续插值器。
2. 在优化器里显式引入参考进度变量 $s_i$，而不是固定索引对应。
3. 增加 tracking error 曲线与时域联动高亮。
4. 若要接近真实 TEB，再增加障碍物和软约束项。