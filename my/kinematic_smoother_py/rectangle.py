import numpy as np
import matplotlib.pyplot as plt

class RobotDecomposer:
    def __init__(self, length, width, mode='balanced'):
        """
        :param mode: 
            'tight': 半径 = W/2 (侧边切齐，但角有缝隙)
            'balanced': 半径 = W/2 * 1.1 (安全裕度)
            'full_cover': 确保覆盖长方形所有顶点
        """
        self.L = length
        self.W = width
        self.offsets = []
        self.radius = 0.0
        
        self._compute_geometry(mode)

    def _compute_geometry(self, mode):
        # 1. 确定半径
        if mode == 'tight':
            self.radius = self.W / 2.0
        elif mode == 'full_cover':
            # 这是一个简单的数学推导：要盖住角，半径至少要能触达局部长方形的顶点
            # 假设两个圆心间距为 d，则 R^2 = (W/2)^2 + (d/2)^2
            # 这里取一个经验值：W/2 的根号2倍
            self.radius = (self.W / 2.0) * np.sqrt(2)
        else: # balanced
            self.radius = (self.W / 2.0) * 1.1

        # 2. 确定圆心摆放的物理极限 (保证圆不超出长方形的 L 边界)
        # 圆心最远只能距离边缘 self.W/2，否则侧边会露出来或圆头突出去太多
        limit_x = max(0, (self.L / 2.0) - (self.W / 2.0))
        
        if limit_x == 0:
            self.offsets = np.array([0.0])
        else:
            # 3. 计算圆的数量
            # 间距建议：圆心距 d <= 1.5 * radius 保证重叠度
            d_desired = self.radius
            num_circles = int(np.ceil((2 * limit_x) / d_desired)) + 1
            num_circles = max(2, num_circles)
            
            self.offsets = np.linspace(-limit_x, limit_x, num_circles)

    def get_world_circles(self, x, y, yaw):
        """将分解圆投影到世界坐标系"""
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        return [(x + ox * cos_y, y + ox * sin_y, self.radius) for ox in self.offsets]

    def plot(self, ax=None, title_suffix=""):
        """可视化当前的分解效果"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            show_plot = True
        else:
            show_plot = False
            
        # 绘制车体
        rect = plt.Rectangle((-self.L/2, -self.W/2), self.L, self.W, 
                              fill=False, color='blue', lw=2, label='Robot' if (ax.get_legend() is None) else "")
        ax.add_patch(rect)
        # 绘制圆
        for i, ox in enumerate(self.offsets):
            circ = plt.Circle((ox, 0), self.radius, color='red', alpha=0.2, 
                              label='Decomposition' if (i == 0 and ax.get_legend() is None) else "")
            ax.add_patch(circ)
            ax.plot(ox, 0, 'r.', markersize=4)
            
        ax.set_aspect('equal')
        ax.set_title(f"Mode: {self.radius:.2f}R | {len(self.offsets)} Circles" + title_suffix)
        ax.grid(True, alpha=0.3)
        
        # Set limits with some padding
        pad = self.radius + 0.5
        ax.set_xlim(-self.L/2 - pad, self.L/2 + pad)
        ax.set_ylim(-self.W/2 - pad, self.W/2 + pad)
        
        # Avoid duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        if show_plot:
            plt.show()

if __name__ == "__main__":
    # Draw all situations: Different sizes and modes
    modes = ['tight', 'balanced', 'full_cover']
    sizes = [
        (4.5, 2.0, "Truck"), 
        (0.6, 0.5, "Small Bot")
    ]
    
    fig, axes = plt.subplots(len(sizes), len(modes), figsize=(15, 8))
    
    # Ensure axes is 2D array even if 1 row
    if len(sizes) == 1:
        axes = np.array([axes])
    if len(modes) == 1 and len(sizes) > 1:
        axes = axes.reshape(-1, 1)
        
    for i, (l, w, name) in enumerate(sizes):
        for j, mode in enumerate(modes):
            rd = RobotDecomposer(length=l, width=w, mode=mode)
            ax = axes[i, j]
            rd.plot(ax=ax, title_suffix=f"\n{name}: {l}x{w} ({mode})")
            
    plt.tight_layout()
    plt.savefig('rectangle_decomposition.png')
    print("Saved visualization to rectangle_decomposition.png")
    plt.show()