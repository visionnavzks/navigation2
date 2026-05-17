# DS Local Controller Notes

这个目录是 `my/teb_local_controller` 的一份独立 ds 版本，不再用时间步长 `dt` 做参数化，而是直接按弧长 `s` 推进。

## 核心变化

- 状态改成 `[x, y, theta, kappa]`
- 控制量改成 `[ds, dkappa]`
- 前端图表统一按累计弧长 `s` 展示
- 随机初始状态不再包含 `v / a`
- Web demo 默认端口改为 `5003`，避免和原版冲突

## 模型

离散空间模型使用中点近似：

$$
\kappa_{i+1} = \kappa_i + ds_i \cdot d\kappa_i
$$

$$
\kappa_{mid} = \kappa_i + \frac{1}{2} ds_i \cdot d\kappa_i
$$

$$
\theta_{i+1} = \theta_i + ds_i \cdot \kappa_{mid}
$$

$$
\theta_{mid} = \theta_i + \frac{1}{2} ds_i \cdot \kappa_{mid}
$$

$$
x_{i+1} = x_i + ds_i \cdot \cos(\theta_{mid})
$$

$$
y_{i+1} = y_i + ds_i \cdot \sin(\theta_{mid})
$$

代价函数由三部分组成：

- 几何跟踪：位置、航向、曲率贴近参考
- 控制正则：`ds` 贴近参考离散间距，`dkappa` 保持平滑
- 终端加强：末端位置、航向、曲率对齐

## 主要文件

- `teb_mpc.py`
  ds 版本的参考轨迹构造和 `DSMPCController` 求解器。
- `demo_support.py`
  参考轨迹、随机初始状态、投影对齐和配置摘要。
- `app.py`
  Flask Web demo，默认监听 `http://127.0.0.1:5003/`。
- `matplotlib_demo.py`
  一个简化的 Matplotlib 可视化脚本，绘制路径、航向、曲率和控制量曲线。
- `templates/index.html`
  页面结构。
- `static/js/app.js`
  交互、绘图和统计展示。

## 运行 Web Demo

在仓库根目录激活虚拟环境后执行：

```bash
python my/teb_local_controller_ds/app.py
```

然后访问 `http://127.0.0.1:5003/`。

## 运行 Matplotlib Demo

```bash
python my/teb_local_controller_ds/matplotlib_demo.py --random
```

可选参数：

- `--seed 123` 固定随机种子
- `--save out.png` 直接保存图片而不弹窗