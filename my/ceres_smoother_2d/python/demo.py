"""
Ceres 二维 ESDF 路径平滑器的 Python demo。

用法：
    python demo.py [path_to_occupancy_map.png]

输出：
    - smooth_result.png：包含 ESDF 热力图和路径的 matplotlib 可视化图
    - smooth_result_interactive.png：交互式 matplotlib 图（若有显示环境）
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

# 将 build 目录加入路径，以加载 nanobind 模块。
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
sys.path.insert(0, build_dir)

import ceres_smoother_2d as cs2d


def generate_test_path(esdf_map, n_points=50):
    """生成一条避开障碍的测试参考路径。"""
    wx = esdf_map.world_width
    wy = esdf_map.world_height

    # 在自由空间中寻找合适的 Y 层。
    best_y = wy * 0.5
    best_score = -1.0
    for row in range(0, esdf_map.height, 5):
        y = esdf_map.origin_y + (row + 0.5) * esdf_map.resolution
        score = 0.0
        count = 0
        for col in range(0, esdf_map.width, 10):
            x = esdf_map.origin_x + (col + 0.5) * esdf_map.resolution
            d = esdf_map.get_distance(x, y)
            if d > 0:
                score += d
                count += 1
        if count > 0:
            avg = score / count
            if avg > best_score:
                best_score = avg
                best_y = y

    # 在该 Y 层生成正弦路径。
    xs, ys = [], []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = 0.1 * wx + t * 0.8 * wx
        y = best_y + 2.0 * np.sin(2 * np.pi * t)
        d = esdf_map.get_distance(x, y)
        if d > 0.05:
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        xs = [0.1 * wx + t * 0.8 * wx for t in np.linspace(0, 1, 30)]
        ys = [best_y] * 30

    return np.array(xs), np.array(ys)


def visualize(esdf_map, ref_x, ref_y, res_x, res_y, output_path):
    """创建信息较完整的 matplotlib 可视化。"""
    # 提取 ESDF 栅格用于可视化。
    esdf_arr = np.array(esdf_map.get_esdf_array()).reshape(esdf_map.height, esdf_map.width)
    occ_arr = np.array(esdf_map.get_occupancy_array()).reshape(esdf_map.height, esdf_map.width)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- 面板 1：占据图 + 路径 ---
    ax = axes[0]
    ax.imshow(occ_arr, cmap='gray_r', origin='lower',
              extent=[esdf_map.origin_x, esdf_map.origin_x + esdf_map.world_width,
                      esdf_map.origin_y, esdf_map.origin_y + esdf_map.world_height],
              alpha=0.8)
    ax.plot(ref_x, ref_y, 'b-o', markersize=3, linewidth=1.5, label='Reference', alpha=0.7)
    ax.plot(res_x, res_y, 'g-', linewidth=2.5, label='Smoothed')
    ax.plot(res_x[0], res_y[0], 'go', markersize=8, zorder=5)
    ax.plot(res_x[-1], res_y[-1], 'rs', markersize=8, zorder=5)
    ax.set_title('Occupancy Map + Paths')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # --- 面板 2：ESDF 热力图 + 平滑路径 ---
    ax = axes[1]
    # 裁剪 ESDF 范围，改善可视化效果。
    esdf_vis = np.clip(esdf_arr, -2, 5)
    im = ax.imshow(esdf_vis, cmap='RdBu_r', origin='lower',
                   extent=[esdf_map.origin_x, esdf_map.origin_x + esdf_map.world_width,
                           esdf_map.origin_y, esdf_map.origin_y + esdf_map.world_height])
    plt.colorbar(im, ax=ax, label='Distance (m)', shrink=0.8)
    ax.plot(ref_x, ref_y, 'b--', linewidth=1, alpha=0.5, label='Reference')
    ax.plot(res_x, res_y, 'k-', linewidth=2.5, label='Smoothed')
    ax.plot(res_x[0], res_y[0], 'go', markersize=8, zorder=5)
    ax.plot(res_x[-1], res_y[-1], 'rs', markersize=8, zorder=5)
    ax.set_title('ESDF + Smoothed Path')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # --- 面板 3：间隙曲线 ---
    ax = axes[2]
    clearances = [esdf_map.get_distance(x, y) for x, y in zip(res_x, res_y)]
    ds = np.zeros(len(res_x))
    for i in range(1, len(res_x)):
        ds[i] = ds[i-1] + np.sqrt((res_x[i]-res_x[i-1])**2 + (res_y[i]-res_y[i-1])**2)

    ax.fill_between(ds, clearances, alpha=0.3, color='green')
    ax.plot(ds, clearances, 'g-', linewidth=2, label='Clearance')
    ax.axhline(y=0.3, color='r', linestyle='--', linewidth=1, label='Safe distance (0.3m)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_title('Obstacle Clearance Profile')
    ax.set_xlabel('Path distance (m)')
    ax.set_ylabel('Clearance (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    map_path = "/home/zks/ws/gits/navigation2/my/maps/occupancy_map.png"
    if len(sys.argv) > 1:
        map_path = sys.argv[1]

    print("=== Ceres 2D Path Smoother — Python Demo ===")
    print(f"Map: {map_path}")

    # 加载地图。
    resolution = 0.05
    esdf_map = cs2d.ESDFMap(map_path, resolution, 0.0, 0.0, 127)
    print(f"Map: {esdf_map.width}x{esdf_map.height} "
          f"({esdf_map.world_width:.1f}x{esdf_map.world_height:.1f} m)")

    # 生成路径。
    ref_x, ref_y = generate_test_path(esdf_map)
    print(f"Reference path: {len(ref_x)} points")

    # 配置平滑器。
    params = cs2d.SmootherParams()
    params.max_iterations = 200
    params.w_smooth = 100.0
    params.w_max_curvature = 50.0
    params.min_turning_radius = 0.5
    params.w_reference = 10.0
    params.w_obstacle = 200.0
    params.safety_margin = 0.3
    params.verbose = False

    # 执行平滑。
    smoother = cs2d.PathSmoother2D(params)
    result = smoother.smooth(ref_x.tolist(), ref_y.tolist(), esdf_map)

    print(f"\nOptimization: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Time: {result.solve_time_ms:.1f} ms")
    print(f"  Final cost: {result.final_cost:.2f}")

    res_x = np.array(result.x)
    res_y = np.array(result.y)

    # 验证间隙。
    min_clearance = min(esdf_map.get_distance(x, y) for x, y in zip(res_x, res_y))
    max_deviation = max(np.sqrt((rx-nx)**2 + (ry-ny)**2)
                        for rx, ry, nx, ny in zip(res_x, res_y, ref_x, ref_y))
    print(f"  Min clearance: {min_clearance:.3f} m (safety_margin={params.safety_margin} m)")
    print(f"  Max deviation: {max_deviation:.3f} m")

    # 可视化。
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "smooth_result.png")
    visualize(esdf_map, ref_x, ref_y, res_x, res_y, output_path)


if __name__ == "__main__":
    main()
