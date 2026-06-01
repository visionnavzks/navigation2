import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImprovedRobotDecomposer:
    def __init__(self, length, width, mode='smart_capsule', tolerance=0.05):
        """
        :param mode: 
            'tight': R=W/2, 侧边切齐, 忽略角点
            'conservative': R=W/2 * sqrt(2), 覆盖所有, 但侧边很胖
            'smart_capsule': R=W/2, 保持宽度, 拉长首尾圆心以覆盖角点 (推荐)
        :param tolerance: 在 'smart_capsule' 模式下，允许两个圆之间凹陷的最大深度 (米)
                          数值越小，圆越多，拟合越平滑。
        """
        self.L = length
        self.W = width
        self.offsets = []
        self.radius = 0.0
        self.mode = mode
        self.tolerance = tolerance
        
        self._compute_geometry()

    def _compute_geometry(self):
        # 基础半径设为宽度的一半
        base_radius = self.W / 2.0
        
        if self.mode == 'tight':
            self.radius = base_radius
            # 圆心不能超出车身，贴着前后边缘
            limit_x = max(0, self.L / 2.0 - self.radius)
            self._fill_circles(limit_x)
            
        elif self.mode == 'conservative':
            # 传统全覆盖，副作用是变胖
            self.radius = np.hypot(self.L/2, self.W/2) if self.L < self.W else np.hypot(self.W/2, self.W/2)
            # 因为半径很大，可能只需要一个圆，或者圆心很靠近中心
            limit_x = max(0, self.L / 2.0 - self.radius)
            self._fill_circles(limit_x)

        elif self.mode == 'smart_capsule':
            # 策略：宽度绝对优先。保持 R = W/2 (加一点点epsilon防止数值误差)
            self.radius = base_radius + 0.01 
            
            # 计算要覆盖角点 (L/2, W/2)，圆心 X 需要在哪里？
            # dist^2 = (L/2 - cx)^2 + (W/2 - 0)^2 <= R^2
            # (L/2 - cx)^2 <= R^2 - (W/2)^2
            # 注意：如果 R ~= W/2，这里 R^2 - (W/2)^2 接近 0
            # 这意味着圆心必须非常接近 X = L/2
            
            # 实际上，对于完美的矩形角点，用圆覆盖必然导致圆头突出
            # 我们计算刚好覆盖角点时的圆心位置：
            # (L/2 - cx)^2 = R^2 - (W/2)^2 
            # 如果 R 略大于 W/2，这一项 > 0。
            # 但通常为了覆盖角点，我们需要把圆心推到接近 L/2 的位置
            
            # 简化逻辑：让圆心位于 x_offset，使得圆的边缘刚好包住角点
            # 这种情况下，其实就是让圆心向外推，形成一个胶囊形状
            # 我们直接让圆心推到 L/2 的位置是不行的（那样圆的一半都在车外）
            # 我们允许车头突出，以换取车宽不增加。
            
            # 计算覆盖角点所需的圆心位置 cx
            # cx = L/2 - sqrt(R^2 - (W/2)^2) 
            # 如果 R=W/2，则 cx = L/2。也就是说圆心要压在车头线上！
            
            term = self.radius**2 - (self.W/2.0)**2
            if term < 0: term = 0
            limit_x = self.L / 2.0 - np.sqrt(term)
            
            # 此时 limit_x 可能会很大（接近 L/2），导致圆突出车身
            # 这是为了不增加宽度而必须付出的代价（长度增加）
            self._fill_circles(limit_x)

    def _fill_circles(self, limit_x):
        """根据允许的 limit_x 和 tolerance 自动填充中间的圆"""
        if limit_x <= 1e-3:
            self.offsets = np.array([0.0])
            return

        # 计算中间需要多少个圆来保证凹陷不超过 tolerance
        # 两个圆心距离为 d，半径 R。
        # 交叉点的纵向距离 h = sqrt(R^2 - (d/2)^2)
        # 凹陷深度 gap = R - h
        # 我们要求 gap <= tolerance  =>  h >= R - tolerance
        # R^2 - (d/2)^2 >= (R - tol)^2
        # (d/2)^2 <= R^2 - (R - tol)^2
        # d_max = 2 * sqrt( R^2 - (R-tol)^2 )
        
        min_val = self.radius**2 - (self.radius - self.tolerance)**2
        if min_val < 0: min_val = 0
        d_max = 2 * np.sqrt(min_val)
        
        total_len = 2 * limit_x
        num_intervals = int(np.ceil(total_len / d_max))
        num_circles = num_intervals + 1
        
        # 至少2个圆（头尾）
        num_circles = max(2, num_circles)
        
        self.offsets = np.linspace(-limit_x, limit_x, num_circles)

    def plot(self, ax, title_suffix=""):
        # 绘制真实车体轮廓
        rect = patches.Rectangle((-self.L/2, -self.W/2), self.L, self.W, 
                              fill=False, edgecolor='blue', lw=2, linestyle='--', label='Real Footprint')
        ax.add_patch(rect)
        
        # 绘制分解圆
        # 计算所有圆的联合边界（仅仅为了绘图）
        for i, ox in enumerate(self.offsets):
            circ = patches.Circle((ox, 0), self.radius, color='red', alpha=0.3, lw=0)
            ax.add_patch(circ)
            ax.plot(ox, 0, 'r.', markersize=3)
            
            # 绘制圆的轮廓
            circ_outline = patches.Circle((ox, 0), self.radius, fill=False, edgecolor='red', lw=1, alpha=0.6)
            ax.add_patch(circ_outline)

        # 设置绘图属性
        ax.set_aspect('equal')
        max_limit = max(self.L, self.W) / 2 + self.radius + 0.2
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
        ax.grid(True, alpha=0.2)
        
        # 统计数据
        bloat_w = (self.radius * 2 - self.W) * 100
        extension_l = ((self.offsets[-1] + self.radius) * 2 - self.L) * 100
        title = f"{self.mode}\nN={len(self.offsets)}, R={self.radius:.2f}m"
        title += f"\nWidth Bloat: +{bloat_w:.1f}cm"
        title += f"\nLen Extend: +{extension_l:.1f}cm"
        ax.set_title(title, fontsize=9)

if __name__ == "__main__":
    # 对比三种模式
    L, W = 4.5, 2.0
    modes = ['tight', 'conservative', 'smart_capsule']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, mode in zip(axes, modes):
        robot = ImprovedRobotDecomposer(L, W, mode=mode, tolerance=0.02) # tolerance 2cm
        robot.plot(ax)
        
    plt.tight_layout()
    plt.show()