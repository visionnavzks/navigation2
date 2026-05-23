import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
import matplotlib.pyplot as plt

class ESDF2D:
    def __init__(self, occupancy_grid, resolution, origin_x=0.0, origin_y=0.0, use_bicubic=True):
        """
        初始化 ESDF 地图
        :param occupancy_grid: 2D numpy array, 0=free, 1=obstacle (occupied)
        :param resolution: grid resolution (meters/pixel)
        :param origin_x: world x coordinate of grid[0,0]
        :param origin_y: world y coordinate of grid[0,0]
        :param use_bicubic: True for Bicubic interpolation (smoother gradients), False for Bilinear
        """
        self.grid = np.array(occupancy_grid, dtype=bool)
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.interp_order = 3 if use_bicubic else 1
        
        # 1. 计算 ESDF 场
        self.esdf_field = self._compute_esdf(self.grid)
        
        # 2. 预计算梯度场 (Gradient Field)
        # np.gradient 返回的是 (d_row, d_col) -> (dy, dx)
        # 注意: numpy 坐标系是 (row, col)，对应 (y, x)
        grad_y, grad_x = np.gradient(self.esdf_field, edge_order=2)
        
        # 归一化梯度吗？
        # 在优化中，我们通常需要的是真实的距离变化率。
        # 由于 grid 间距是 1 (index space)，而实际物理距离变化需要除以 resolution?
        # 不，ESDF 值本身已经是物理距离了。
        # 梯度 = d(Distance)/d(Index)。为了转成 d(Distance)/d(Meter)，需要除以 resolution。
        self.grad_field_x = grad_x / self.resolution
        self.grad_field_y = grad_y / self.resolution

    def _compute_esdf(self, grid):
        """
        计算有符号距离场:
        Positive (+) -> Outside obstacle (Free space)
        Negative (-) -> Inside obstacle
        """
        # 1. 计算 Free Space 到最近 Obstacle 的距离 (EDT)
        # invert grid: 1 for free, 0 for obstacle
        dist_outside = distance_transform_edt(~grid)
        
        # 2. 计算 Obstacle 内部到最近 Free Space 的距离
        # grid: 1 for obstacle, 0 for free. 
        # distance_transform_edt 计算到最近的 0 的距离。
        # 对于全 0 区域（free），dist_inside 应为 0。
        # 如果只有 free，dist_inside 会很大，这里需要 trick。
        # 我们只关心障碍物内部的距离。
        dist_inside = distance_transform_edt(grid)
        
        # 3. 合并 (Signed Distance)
        # 外面是正的，里面是负的
        # distance_transform_edt 返回的是像素距离，需要乘分辨率
        # 边界处理：由于 EDT 在边界处 dist_outside 和 dist_inside 都会接近 0，
        # 直接相减通常没问题。
        # 注意：dist_inside 在障碍物中心是最大的。我们希望内部是负数。
        
        # 为了精确处理边界（避免 0.5 像素偏差），通常 EDT 计算的是中心到中心的距离。
        # 这里简化处理：
        esdf = (dist_outside - dist_inside) * self.resolution
        
        return esdf

    def get_distance(self, x, y):
        """
        查询世界坐标 (x, y) 处的距离
        支持标量或 numpy array 输入
        """
        gx, gy = self._world_to_grid(x, y)
        
        # map_coordinates expect input shape (2, N) for [rows, cols]
        # 注意：rows 对应 y, cols 对应 x
        coords = np.array([gy, gx])
        
        # mode='nearest' 保证出界后取最近边界的值（保持梯度方向大致正确）
        dists = map_coordinates(self.esdf_field, coords, order=self.interp_order, mode='nearest')
        
        return dists

    def get_gradient(self, x, y):
        """
        查询世界坐标 (x, y) 处的梯度 (dx, dy)
        返回: (grad_x, grad_y) 指向距离增加的方向（远离障碍物）
        """
        gx, gy = self._world_to_grid(x, y)
        coords = np.array([gy, gx])
        
        gx_val = map_coordinates(self.grad_field_x, coords, order=self.interp_order, mode='nearest')
        gy_val = map_coordinates(self.grad_field_y, coords, order=self.interp_order, mode='nearest')
        
        return gx_val, gy_val

    def get_cost_and_gradient(self, x, y, safe_dist):
        """
        专门为优化器设计的辅助函数。
        计算 Cost = (safe_dist - dist)^2  if dist < safe_dist else 0
        以及对应的关于 x, y 的梯度。
        """
        dist = self.get_distance(x, y)
        
        # 初始化
        cost = np.zeros_like(dist)
        grad_x_cost = np.zeros_like(dist)
        grad_y_cost = np.zeros_like(dist)
        
        # 找出小于安全距离的点 (需要受罚)
        mask = dist < safe_dist
        
        if np.any(mask):
            # Cost = (d_safe - d)^2
            diff = safe_dist - dist[mask]
            cost[mask] = diff ** 2
            
            # Gradient of Cost w.r.t x:
            # d(Cost)/dx = 2 * (d_safe - d) * (-1) * d(d)/dx
            #            = -2 * (d_safe - d) * dist_grad_x
            
            # 获取距离场的梯度 (指向远离障碍物方向)
            if np.isscalar(x):
                gx, gy = self.get_gradient(x, y) # Scalar
                gx = gx if mask else 0
                gy = gy if mask else 0
                grad_x_cost = -2 * diff * gx
                grad_y_cost = -2 * diff * gy
            else:
                gx, gy = self.get_gradient(x[mask], y[mask])
                grad_x_cost[mask] = -2 * diff * gx
                grad_y_cost[mask] = -2 * diff * gy
            
        return cost, grad_x_cost, grad_y_cost

    def _world_to_grid(self, x, y):
        gx = (x - self.origin_x) / self.resolution
        gy = (y - self.origin_y) / self.resolution
        return gx, gy

# --- 测试代码 ---
if __name__ == "__main__":
    # 1. 创建一个模拟地图 (100x100)
    # 0 = Free, 1 = Obstacle
    grid = np.zeros((100, 100))
    
    # 添加围墙
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    
    # 添加一个障碍物块
    grid[30:50, 30:50] = 1
    
    # **制造窄通道** (两墙之间留 4 个像素宽)
    grid[30:70, 60:65] = 1 # 左墙
    grid[30:70, 69:74] = 1 # 右墙 (通道在 65-69 之间)
    
    # 2. 初始化 ESDF
    resolution = 0.1 # 10cm per pixel
    esdf = ESDF2D(grid, resolution, use_bicubic=True)
    
    # 3. 可视化 ESDF 场
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("ESDF Field (Positive=Safe)")
    im = plt.imshow(esdf.esdf_field, origin='lower', cmap='RdBu')
    plt.colorbar(im, label='Distance (m)')
    
    # 4. 测试梯度流 (Quiver Plot)
    # 在窄通道附近采样
    x = np.linspace(5.5, 8.0, 20) # 对应 index 55-80
    y = np.linspace(4.0, 6.0, 20) # 对应 index 40-60
    X, Y = np.meshgrid(x, y)
    
    # 获取梯度
    GX, GY = esdf.get_gradient(X.flatten(), Y.flatten())
    GX = GX.reshape(X.shape)
    GY = GY.reshape(Y.shape)
    
    plt.subplot(1, 2, 2)
    plt.title("Gradient Field in Narrow Passage")
    plt.imshow(esdf.grid, origin='lower', cmap='gray_r', extent=[0, 10, 0, 10], alpha=0.3)
    
    # 画出梯度的反方向（因为优化器是下降，cost梯度是指向障碍物的，但ESDF梯度是指向远离障碍物的）
    # 如果我们要最大化距离，就顺着箭头走。
    # 如果我们要最小化 Cost (d_safe - d)^2，力是推着我们顺着箭头走的。
    plt.quiver(X, Y, GX, GY, color='red', scale=20, width=0.003)
    plt.xlim(5.5, 8.0)
    plt.ylim(4.0, 6.0)
    
    plt.tight_layout()
    plt.show()
    
    # 5. 测试数值查询
    test_x, test_y = 6.7, 5.0 # 窄通道中心 (index ~67, 50)
    d = esdf.get_distance(test_x, test_y)
    print(f"Distance at ({test_x}, {test_y}): {d:.4f} m")
    
    cost, gxc, gyc = esdf.get_cost_and_gradient(test_x, test_y, safe_dist=0.5)
    print(f"Optimization Cost info at center: Cost={cost}, Grad=({gxc:.4f}, {gyc:.4f})")
    
    # 测试靠近墙的地方
    wall_x, wall_y = 6.55, 5.0 # 非常靠近左墙
    d_wall = esdf.get_distance(wall_x, wall_y)
    cost_wall, gxc_wall, gyc_wall = esdf.get_cost_and_gradient(wall_x, wall_y, safe_dist=0.5)
    print(f"Distance near wall ({wall_x}, {wall_y}): {d_wall:.4f} m")
    print(f"Gradient push near wall: ({gxc_wall:.4f}, {gyc_wall:.4f})")
    print("Notice the strong gradient pushing positive X (to the right, away from left wall)")