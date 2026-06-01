
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

class ESDFMap:
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
        self.height, self.width = self.esdf_field.shape
        
        # 2. 预计算梯度场 (Gradient Field)
        # np.gradient 返回的是 (d_row, d_col) -> (dy, dx)
        # 注意: numpy 坐标系是 (row, col)，对应 (y, x)
        grad_y, grad_x = np.gradient(self.esdf_field, edge_order=2)
        
        # 归一化梯度吗？
        # W e want physical gradient: d(Distance)/d(Meter)
        # d(D)/d(x_m) = d(D)/d(x_px) * d(x_px)/d(x_m) = d(D)/d(x_px) * (1/res)
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
        # grid is 1 for obstacle. ~grid is 1 for free.
        # dt(~grid) gives dist to nearest 0 (obstacle)
        dist_outside = distance_transform_edt(~grid)
        
        # 2. 计算 Obstacle 内部到最近 Free Space 的距离
        # grid: 1 for obstacle, 0 for free. 
        # dt(grid) gives dist to nearest 0 (free)
        dist_inside = distance_transform_edt(grid)
        
        # 3. 合并 (Signed Distance)
        # Dist = Outside - Inside
        # 边界处通常 Outside=0.5, Inside=0.5 (approx), diff=0.
        esdf = (dist_outside - dist_inside) * self.resolution
        
        return esdf

    def get_distance(self, x, y):
        """
        查询世界坐标 (x, y) 处的距离
        支持标量或 numpy array 输入
        """
        gx, gy = self._world_to_grid(x, y)
        
        input_is_scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
        
        if input_is_scalar:
            # (ndim, 1)
            coords = np.array([[gy], [gx]])
        else:
            # (ndim, N)
            coords = np.array([gy, gx])
            
        # mode='nearest' 保证出界后取最近边界的值（保持梯度方向大致正确）
        dists = map_coordinates(self.esdf_field, coords, order=self.interp_order, mode='nearest')
        
        if input_is_scalar:
            return dists[0]
        return dists

    def get_gradient(self, x, y):
        """
        查询世界坐标 (x, y) 处的梯度 (dx, dy)
        返回: (grad_x, grad_y) 指向距离增加的方向（远离障碍物）
        """
        gx, gy = self._world_to_grid(x, y)
        
        input_is_scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
        
        if input_is_scalar:
            coords = np.array([[gy], [gx]])
        else:
            coords = np.array([gy, gx])
        
        gx_val = map_coordinates(self.grad_field_x, coords, order=self.interp_order, mode='nearest')
        gy_val = map_coordinates(self.grad_field_y, coords, order=self.interp_order, mode='nearest')
        
        if input_is_scalar:
            return gx_val[0], gy_val[0]
        return gx_val, gy_val

    def get_distance_and_gradient(self, x, y):
        """
        Compatibility method for KinematicSmoother
        Returns: dist, np.array([grad_x, grad_y])
        """
        dist = self.get_distance(x, y)
        gx, gy = self.get_gradient(x, y)
        return dist, np.array([gx, gy])

    def _world_to_grid(self, x, y):
        gx = (x - self.origin_x) / self.resolution
        gy = (y - self.origin_y) / self.resolution
        return gx, gy
