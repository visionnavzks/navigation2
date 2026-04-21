import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import least_squares

# ==========================================
# 1. 基础类：机器人形状分解 & ESDF 模拟
# ==========================================

class RobotFootprint:
    """
    负责将矩形机器人分解为多个覆盖圆 (Multi-Circle Decomposition)
    采用"宁长勿宽"的策略，适合通过窄缝。
    """
    def __init__(self, length, width, num_circles=3):
        self.L = length
        self.W = width
        # 半径略微大于宽的一半，增加一点安全余量
        self.radius = (width / 2.0) + 0.05 
        
        # 计算圆心分布
        # 策略：首尾圆心向外推，尽量覆盖角点
        # 胶囊体逻辑：limit_x 是圆心距离中心的极限距离
        limit_x = max(0, (length / 2.0) - self.radius * 0.6)
        
        if num_circles <= 1:
            self.offsets = np.array([0.0])
        else:
            self.offsets = np.linspace(-limit_x, limit_x, num_circles)
            
    def get_circle_centers(self, x, y, theta):
        """
        根据机器人的位姿 (x, y, theta) 计算所有覆盖圆的世界坐标
        Returns: Shape (K, 2)
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        centers_x = x + self.offsets * cos_t
        centers_y = y + self.offsets * sin_t
        
        return np.stack([centers_x, centers_y], axis=1)

class MockESDFMap:
    """
    模拟 ESDF 地图。
    实际使用时，请替换为从 C++ 绑定的 costmap/esdf 查询接口。
    """
    def __init__(self):
        # 定义两个圆形障碍物: (x, y, radius)
        self.obstacles = [
            (3.0, 2.0, 0.8),  # 障碍物 1
            (6.0, 4.0, 1.0)   # 障碍物 2
        ]

    def get_distance_and_gradient(self, points):
        """
        计算点集到最近障碍物的符号距离和梯度。
        points: (N, 2) array
        Return: 
           dist: (N,) 距离 (负数表示在障碍物内)
           grad: (N, 2) 梯度向量 (指向远离障碍物的方向)
        """
        N = points.shape[0]
        min_dists = np.full(N, 100.0) # 初始极大值
        gradients = np.zeros((N, 2))

        # 简单的解析几何计算距离
        for ox, oy, r in self.obstacles:
            dx = points[:, 0] - ox
            dy = points[:, 1] - oy
            dist_center = np.hypot(dx, dy)
            
            # Signed Distance: 表面为0，内部为负
            sdf = dist_center - r
            
            # 更新最近距离
            mask = sdf < min_dists
            min_dists[mask] = sdf[mask]
            
            # 计算梯度 (归一化向量)
            # 防止除以0
            valid_grad = dist_center > 1e-6
            
            # 只有当这是目前最近的障碍物时才更新梯度
            update_mask = mask & valid_grad
            if np.any(update_mask):
                inv_dist = 1.0 / dist_center[update_mask]
                gradients[update_mask, 0] = dx[update_mask] * inv_dist
                gradients[update_mask, 1] = dy[update_mask] * inv_dist
                
        return min_dists, gradients

# ==========================================
# 2. 核心算法：残差函数
# ==========================================

def get_residuals(vars, N, raw_path, gears, robot, esdf, settings):
    """
    计算所有优化目标的残差向量。
    vars: 扁平化的状态向量 [x0, y0, th0, k0, s0, x1, ...]
    """
    # 1. 解包变量 (N x 5)
    # columns: 0:x, 1:y, 2:theta, 3:kappa, 4:delta_s
    states = vars.reshape((N, 5))
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    kappa = states[:, 3]
    ds = states[:, 4]

    residuals = []
    
    # 权重配置
    w_model = settings['w_model']   # 运动学约束权重 (很高)
    w_smooth = settings['w_smooth'] # 平滑权重
    w_ref = settings['w_ref']       # 参考路径跟随权重
    w_obs = settings['w_obs']       # 障碍物斥力权重
    w_s = settings['w_s']           # 间距均匀化权重
    w_fix = settings['w_fix']       # 起终点固定权重

    # --- A. 运动学模型一致性 (Kinematic Constraints) ---
    # 预测下一个点：x_{i+1} = f(x_i, u_i)
    # 使用中点积分法提高精度
    for i in range(N - 1):
        direction = gears[i] # +1 or -1
        d_s = ds[i]
        k = kappa[i]
        
        # 中点航向角
        mid_theta = theta[i] + direction * d_s * k * 0.5
        
        # 预测值
        pred_x = x[i] + direction * d_s * np.cos(mid_theta)
        pred_y = y[i] + direction * d_s * np.sin(mid_theta)
        pred_theta = theta[i] + direction * d_s * k
        
        # 残差：预测值 - 实际变量值
        residuals.append(w_model * (x[i+1] - pred_x))
        residuals.append(w_model * (y[i+1] - pred_y))
        
        # 角度差归一化到 [-pi, pi]
        ang_diff = theta[i+1] - pred_theta
        ang_diff = (ang_diff + np.pi) % (2 * np.pi) - np.pi
        residuals.append(w_model * ang_diff)

    # --- B. 平滑性 (Smoothness) ---
    # 惩罚曲率变化率 (Jerk)
    # 注意：在 Cusp (换挡) 点，不计算平滑性，允许曲率突变
    for i in range(N - 1):
        if gears[i] == gears[i+1] if i+1 < len(gears) else True:
            # 只有当这一段和下一段方向相同时，才要求曲率连续
            d_k = kappa[i+1] - kappa[i]
            # 为了数值稳定，除以 ds 可能导致极大值，这里简化为直接惩罚 delta_kappa
            residuals.append(w_smooth * d_k) 

    # --- C. 障碍物避障 (Obstacle Avoidance - Hinge Loss) ---
    # 对每个轨迹点，计算多个圆心的位置
    # 展平所有圆心以进行批量查询
    # total_checks = N * num_circles
    all_centers = []
    for i in range(N):
        centers = robot.get_circle_centers(x[i], y[i], theta[i]) # (K, 2)
        all_centers.append(centers)
    
    all_centers_np = np.vstack(all_centers) # (N*K, 2)
    
    # 查询 ESDF
    dists, _ = esdf.get_distance_and_gradient(all_centers_np)
    
    # 安全阈值
    safe_threshold = robot.radius + settings['safety_margin']
    
    # Hinge Loss: 只有距离小于阈值时才产生残差
    # r = max(0, safe - dist)
    obs_res = np.zeros_like(dists)
    mask = dists < safe_threshold
    obs_res[mask] = safe_threshold - dists[mask]
    
    residuals.extend(w_obs * obs_res)

    # --- D. 参考路径跟随 (Reference Path) ---
    # 只约束位置
    ref_x = raw_path[:, 0]
    ref_y = raw_path[:, 1]
    dist_sq = (x - ref_x)**2 + (y - ref_y)**2
    # 使用 sqrt 使得残差为距离，least_squares 会对其平方
    residuals.extend(w_ref * np.sqrt(dist_sq + 1e-6))

    # --- E. 间距正则化 (Regularization) ---
    # 希望 delta_s 接近 target_spacing
    residuals.extend(w_s * (ds[:-1] - settings['target_spacing']))

    # --- F. 边界强约束 (Boundary Conditions) ---
    # 固定起点和终点的位置与朝向
    # 起点
    residuals.append(w_fix * (x[0] - raw_path[0, 0]))
    residuals.append(w_fix * (y[0] - raw_path[0, 1]))
    # residuals.append(w_fix * (theta[0] - raw_path_theta[0])) # 需外部传入起点角度

    # 终点
    residuals.append(w_fix * (x[-1] - raw_path[-1, 0]))
    residuals.append(w_fix * (y[-1] - raw_path[-1, 1]))

    return np.array(residuals)


# ==========================================
# 3. 主优化器类
# ==========================================

class KinematicSmoother:
    def __init__(self, robot_L=1.0, robot_W=0.6):
        self.robot = RobotFootprint(robot_L, robot_W, num_circles=3)
        self.esdf = MockESDFMap()
        
        # 优化参数配置
        self.settings = {
            'w_model': 10.0,    # 物理约束最重要
            'w_obs': 20.0,      # 避障权重高 (安全第一)
            'w_smooth': 5.0,    # 平滑权重
            'w_ref': 0.5,       # 允许稍微偏离原路径以避障
            'w_s': 0.1,         # 间距约束较弱
            'w_fix': 100.0,     # 强行固定起终点
            'safety_margin': 0.10, # 额外保留 10cm 安全距离
            'target_spacing': 0.15 # 期望点间距 0.15m
        }

    def solve(self, raw_path, gears, start_yaw):
        """
        执行优化
        raw_path: (N, 2) array
        gears: (N-1,) array, +1 or -1
        start_yaw: float, initial heading
        """
        N = len(raw_path)
        
        # 1. 构建初始猜测 (Initial Guess)
        # 简单的差分计算朝向，如果是倒车则反转
        x_init = raw_path[:, 0]
        y_init = raw_path[:, 1]
        
        # 初始化 Theta
        theta_init = np.zeros(N)
        theta_init[0] = start_yaw
        for i in range(1, N):
            dx = x_init[i] - x_init[i-1]
            dy = y_init[i] - y_init[i-1]
            yaw = np.arctan2(dy, dx)
            
            # 如果是倒车，速度向量反向，但车头朝向通常定义为车体前方
            # Nav2 中倒车时：yaw 指向车头，运动方向相反
            # 所以如果 gear=-1, yaw = velocity_angle + pi
            prev_gear = gears[i-1] if i-1 < len(gears) else gears[-1]
            if prev_gear < 0:
                yaw += np.pi
                
            # 归一化
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            theta_init[i] = yaw

        # 初始化 Kappa (设为0)
        kappa_init = np.zeros(N)
        
        # 初始化 delta_s (欧氏距离)
        ds_init = np.zeros(N)
        for i in range(N-1):
            ds_init[i] = np.hypot(x_init[i+1]-x_init[i], y_init[i+1]-y_init[i])
        
        # 拼装初始向量
        initial_guess = np.column_stack([x_init, y_init, theta_init, kappa_init, ds_init]).flatten()
        
        # 2. 调用优化器
        print(f"Start Optimization with {N} points...")
        result = least_squares(
            fun=get_residuals,
            x0=initial_guess,
            args=(N, raw_path, gears, self.robot, self.esdf, self.settings),
            method='lm', # Levenberg-Marquardt
            max_nfev=100,
            verbose=1
        )
        
        # 3. 解析结果
        opt_states = result.x.reshape((N, 5))
        return opt_states

# ==========================================
# 4. 运行演示
# ==========================================

def main():
    # 1. 生成一条带倒车的粗糙路径 (人字形)
    # A -> B (Forward), B -> C (Reverse)
    # A(0,0), B(4,0), C(4, 4)
    
    # 段1：直走，穿过障碍物1附近
    path1_x = np.linspace(0, 4.5, 20)
    path1_y = np.linspace(0, 0.5, 20) # 稍微有点歪
    path1 = np.column_stack([path1_x, path1_y])
    gears1 = np.ones(19) # Forward
    
    # 段2：倒车，避开障碍物2
    path2_x = np.linspace(4.5, 4.0, 15)
    path2_y = np.linspace(0.5, 5.0, 15)
    path2 = np.column_stack([path2_x, path2_y])
    gears2 = np.full(14, -1) # Reverse
    
    # 合并 (注意连接点)
    raw_path = np.vstack([path1, path2[1:]])
    gears = np.concatenate([gears1, np.array([0]), gears2]) # 中间补个0占位或处理逻辑需一致
    # 修正 gears 长度应为 N-1
    gears = np.concatenate([gears1, np.array([-1]), gears2]) # 简单起见，这里假设尖点归属下一段
    
    # 2. 运行平滑器
    smoother = KinematicSmoother(robot_L=1.0, robot_W=0.6)
    opt_states = smoother.solve(raw_path, gears, start_yaw=0.0)
    
    # 3. 可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 画障碍物
    for ox, oy, r in smoother.esdf.obstacles:
        circle = plt.Circle((ox, oy), r, color='black', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)
        # 画安全边界
        safe_r = r + smoother.robot.radius + smoother.settings['safety_margin']
        ax.add_patch(plt.Circle((ox, oy), safe_r, color='red', fill=False, linestyle='--', alpha=0.3))

    # 画原始路径
    ax.plot(raw_path[:,0], raw_path[:,1], 'g.--', label='Raw Path', alpha=0.5)
    
    # 画优化后的路径
    opt_x = opt_states[:, 0]
    opt_y = opt_states[:, 1]
    opt_theta = opt_states[:, 2]
    ax.plot(opt_x, opt_y, 'b.-', linewidth=2, label='Smoothed Path')
    
    # 画机器人 Footprint (每隔几个点画一次)
    for i in range(0, len(opt_states), 4):
        x, y, theta = opt_x[i], opt_y[i], opt_theta[i]
        
        # 画矩形轮廓
        # 计算矩形四个角点
        L, W = smoother.robot.L, smoother.robot.W
        corners = np.array([
            [L/2, W/2], [L/2, -W/2], [-L/2, -W/2], [-L/2, W/2]
        ])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rot_corners = corners @ R.T + np.array([x, y])
        poly = patches.Polygon(rot_corners, closed=True, edgecolor='blue', facecolor='none', alpha=0.5)
        ax.add_patch(poly)
        
        # 画分解圆 (Debug)
        centers = smoother.robot.get_circle_centers(x, y, theta)
        for cx, cy in centers:
            c = plt.Circle((cx, cy), smoother.robot.radius, color='cyan', alpha=0.2, lw=0)
            ax.add_patch(c)
            
        # 标箭头表示朝向
        arrow_len = 0.3
        ax.arrow(x, y, arrow_len*np.cos(theta), arrow_len*np.sin(theta), head_width=0.1, color='blue')

    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    ax.set_title("Kinematic Smoother with Rectangular Footprint & ESDF")
    
    plt.show()

if __name__ == "__main__":
    main()