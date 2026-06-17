"""Flask + Plotly Web demo for Ceres 2D Smoother with A* planning.

Run:
    ./run_web.sh
Then open http://127.0.0.1:5000/
"""

import base64
import io
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import ceres_smoother_2d as cs2d

app = Flask(__name__)

MAP_PATH = os.environ.get(
    "MAP_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "maps", "occupancy_map.png"),
)
RESOLUTION = float(os.environ.get("RESOLUTION", "0.05"))
esdf_map = None
occ_grid = None
COSTMAP_B64 = None
ESDF_B64 = None
ESDF_SCHEMES = None
OCC_STATUS_GRID = None

FREE_RGB = np.array([255, 255, 255], dtype=np.uint8)
OBSTACLE_RGB = np.array([0, 0, 0], dtype=np.uint8)
UNKNOWN_RGB = np.array([128, 128, 128], dtype=np.uint8)

ESDF_SCHEME_DEFS = [
    {
        "key": "clearance",
        "label": "Clearance: collision / margin / free",
        "vmin": -1.0,
        "vmax": 3.0,
        "stops": [
            (0.00, "#4c0519"),
            (0.18, "#b91c1c"),
            (0.25, "#f97316"),
            (0.38, "#fde68a"),
            (0.58, "#67e8f9"),
            (1.00, "#1d4ed8"),
        ],
        "swatch": "linear-gradient(to right,#4c0519,#b91c1c,#f97316,#fde68a,#67e8f9,#1d4ed8)",
    },
    {
        "key": "signed",
        "label": "Signed: obstacle red / free blue",
        "vmin": -2.0,
        "vmax": 2.0,
        "stops": [
            (0.00, "#7f1d1d"),
            (0.30, "#ef4444"),
            (0.50, "#f8fafc"),
            (0.68, "#38bdf8"),
            (1.00, "#0f766e"),
        ],
        "swatch": "linear-gradient(to right,#7f1d1d,#ef4444,#f8fafc,#38bdf8,#0f766e)",
    },
    {
        "key": "safety",
        "label": "Safety bands: wall / low / clear",
        "vmin": -0.5,
        "vmax": 2.5,
        "stops": [
            (0.00, "#991b1b"),
            (0.16, "#ef4444"),
            (0.26, "#f59e0b"),
            (0.42, "#fef08a"),
            (0.64, "#86efac"),
            (1.00, "#0284c7"),
        ],
        "swatch": "linear-gradient(to right,#991b1b,#ef4444,#f59e0b,#fef08a,#86efac,#0284c7)",
    },
    {
        "key": "relief",
        "label": "Relief: dark wall / bright clearance",
        "vmin": -0.8,
        "vmax": 2.0,
        "stops": [
            (0.00, "#020617"),
            (0.24, "#7f1d1d"),
            (0.29, "#f97316"),
            (0.48, "#cbd5e1"),
            (1.00, "#ffffff"),
        ],
        "swatch": "linear-gradient(to right,#020617,#7f1d1d,#f97316,#cbd5e1,#fff)",
    },
]


def _encode_png(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _palette(values, stops):
    values = np.clip(values, 0.0, 1.0)
    out = np.zeros(values.shape + (3,), dtype=np.float64)
    stops = [(pos, _rgb(color)) for pos, color in stops]
    for i, (left_pos, left_rgb) in enumerate(stops[:-1]):
        right_pos, right_rgb = stops[i + 1]
        mask = (values >= left_pos) & (values <= right_pos)
        denom = max(right_pos - left_pos, 1e-9)
        t = ((values[mask] - left_pos) / denom)[:, None]
        out[mask] = left_rgb * (1.0 - t) + right_rgb * t
    out[values <= stops[0][0]] = stops[0][1]
    out[values >= stops[-1][0]] = stops[-1][1]
    return out.astype(np.uint8)


def _source_map_masks(path):
    from PIL import Image
    src_png = np.array(Image.open(path).convert("L"))
    unknown_mask = (src_png > 1) & (src_png <= 127)
    obstacle_mask = src_png <= 1
    free_mask = src_png > 127
    status = np.zeros_like(src_png, dtype=np.uint8)
    status[free_mask] = 0          # 空白/Free
    status[obstacle_mask] = 1       # 占用/Obstacle
    status[unknown_mask] = 2        # 未知/Unknown
    # ESDF uses row-flipped indexing so convert to same row orientation as world grid.
    return free_mask, obstacle_mask, unknown_mask, status[::-1, :]


def _build_costmap_png(free_mask, obstacle_mask, unknown_mask):
    occ_rgb = np.zeros(free_mask.shape + (3,), dtype=np.uint8)
    occ_rgb[free_mask] = FREE_RGB
    occ_rgb[obstacle_mask] = OBSTACLE_RGB
    occ_rgb[unknown_mask] = UNKNOWN_RGB
    return _encode_png(occ_rgb)


def _build_esdf_schemes(esdf_png_rows, unknown_mask):
    schemes = []
    for scheme in ESDF_SCHEME_DEFS:
        t = (np.clip(esdf_png_rows, scheme["vmin"], scheme["vmax"]) - scheme["vmin"]) / (
            scheme["vmax"] - scheme["vmin"])
        rgb = _palette(t, scheme["stops"])
        rgb[unknown_mask] = UNKNOWN_RGB
        schemes.append({
            "key": scheme["key"],
            "label": scheme["label"],
            "swatch": scheme["swatch"],
            "png": _encode_png(rgb),
        })
    return schemes


def init_map():
    global esdf_map, occ_grid, COSTMAP_B64, ESDF_B64, ESDF_SCHEMES, OCC_STATUS_GRID

    esdf_map = cs2d.ESDFMap(MAP_PATH, RESOLUTION, 0.0, 0.0, 127)
    occ_grid = np.array(esdf_map.get_occupancy_array()).reshape(
        esdf_map.height, esdf_map.width)
    free_mask, obstacle_mask, unknown_mask, status_grid = _source_map_masks(MAP_PATH)
    OCC_STATUS_GRID = status_grid

    COSTMAP_B64 = _build_costmap_png(free_mask, obstacle_mask, unknown_mask)

    # ESDFMap stores rows flipped (row r = PNG row H-1-r). Flip back before
    # creating display images so PNG row 0 corresponds to world y_max.
    ed = np.array(esdf_map.get_esdf_array()).reshape(esdf_map.height, esdf_map.width)
    ESDF_SCHEMES = _build_esdf_schemes(ed[::-1, :], unknown_mask)
    ESDF_B64 = ESDF_SCHEMES[0]["png"]
    print(f"[init] {esdf_map.width}x{esdf_map.height} ({esdf_map.world_width:.1f}x{esdf_map.world_height:.1f}m)")


def compute_path_cost_breakdown(xs, ys, pm, map_obj):
    """Recompute readable path costs on the returned path.

    Ceres reports 0.5 * sum(residual^2). These terms use the same convention.
    Reference cost is omitted because the optimizer's internal reference path
    can be resampled to a different point count than the final returned path.
    """
    n = len(xs)
    terms = dict(length=0.0, smooth=0.0, curvature=0.0, obstacle=0.0, penetration=0.0)
    obstacle_active = 0
    penetration_active = 0
    obstacle_threshold = _obstacle_cost_distance(pm)
    min_obstacle_distance = float("inf")
    min_obstacle_margin = float("inf")
    if n < 2:
        terms["total"] = 0.0
        terms["obstacle_active_points"] = 0
        terms["obstacle_threshold"] = round(obstacle_threshold, 6)
        terms["min_obstacle_distance"] = None
        terms["min_obstacle_margin"] = None
        return terms

    for i in range(n - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        terms["length"] += 0.5 * pm.w_length * (dx * dx + dy * dy)

    max_kappa = 1.0 / pm.min_turning_radius if pm.min_turning_radius > 0 else float("inf")
    for i in range(1, n - 1):
        sx = xs[i + 1] - 2.0 * xs[i] + xs[i - 1]
        sy = ys[i + 1] - 2.0 * ys[i] + ys[i - 1]
        terms["smooth"] += 0.5 * pm.w_smooth * (sx * sx + sy * sy)

        d = map_obj.get_distance(xs[i], ys[i])
        margin = d - obstacle_threshold
        min_obstacle_distance = min(min_obstacle_distance, d)
        min_obstacle_margin = min(min_obstacle_margin, margin)
        violation = obstacle_threshold - d
        if violation > 0.0:
            obstacle_active += 1
            terms["obstacle"] += 0.5 * pm.w_obstacle * violation * violation
        penetration = -d
        if penetration > 0.0:
            penetration_active += 1
            terms["penetration"] += 0.5 * pm.w_penetration * penetration * penetration

        v1x = xs[i] - xs[i - 1]
        v1y = ys[i] - ys[i - 1]
        v2x = xs[i + 1] - xs[i]
        v2y = ys[i + 1] - ys[i]
        dot = v1x * v2x + v1y * v2y
        norm_v1 = math.sqrt(v1x * v1x + v1y * v1y + 1e-12)
        norm_v2 = math.sqrt(v2x * v2x + v2y * v2y + 1e-12)
        current_ds = 0.5 * (norm_v1 + norm_v2)
        theta = math.atan2(abs(v1x * v2y - v1y * v2x), dot)
        theta_limit = max_kappa * current_ds
        violation = theta - theta_limit
        if violation > 0.0:
            terms["curvature"] += 0.5 * pm.w_max_curvature * violation * violation

    terms["total"] = (
        terms["length"] + terms["smooth"] + terms["curvature"] +
        terms["obstacle"] + terms["penetration"]
    )
    rounded = {k: round(v, 6) for k, v in terms.items()}
    rounded["obstacle_active_points"] = obstacle_active
    rounded["penetration_active_points"] = penetration_active
    rounded["obstacle_threshold"] = round(obstacle_threshold, 6)
    rounded["min_obstacle_distance"] = (
        round(min_obstacle_distance, 6) if min_obstacle_distance < float("inf") else None
    )
    rounded["min_obstacle_margin"] = (
        round(min_obstacle_margin, 6) if min_obstacle_margin < float("inf") else None
    )
    return rounded


def _grid_index(wx, wy):
    row = max(0, min(esdf_map.height - 1, int(wy / esdf_map.resolution)))
    col = max(0, min(esdf_map.width - 1, int(wx / esdf_map.resolution)))
    return row, col


def _is_free(wx, wy):
    row, col = _grid_index(wx, wy)
    return bool(occ_grid[row, col] == 0)


def _make_smoother_params(body):
    pm = cs2d.SmootherParams()
    pm.max_iterations = int(body.get("max_iterations", 200))
    pm.w_smooth = float(body.get("w_smooth", 100))
    pm.w_max_curvature = float(body.get("w_max_curvature", 1000))
    pm.min_turning_radius = float(body.get("min_turning_radius", 0.5))
    pm.w_reference = float(body.get("w_reference", 0))
    pm.w_length = float(body.get("w_length", 0.5))
    pm.w_obstacle = float(body.get("w_obstacle", 1))
    pm.w_penetration = float(body.get("w_penetration", body.get("w_penetrate", 0)))
    pm.safety_margin = float(body.get("safety_margin", 1.0))
    if "robot_radius" in body and body.get("robot_radius") not in (None, ""):
        pm.robot_radius = float(body["robot_radius"])
    else:
        pm.robot_radius = 0.5
    pm.max_time_seconds = float(body.get("max_time_seconds", 2.0))
    pm.target_spacing = float(body.get("target_spacing", 0.3))
    pm.resample_after_smooth = bool(body.get("resample_after_smooth", False))
    pm.resample_before_smooth = bool(body.get("resample_before_smooth", True))
    return pm


def _obstacle_cost_distance(pm):
    """Effective obstacle cost distance = safety_margin + robot_radius."""
    return pm.safety_margin + pm.robot_radius


def _downsample_path(points, step):
    step = max(1, int(step))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _path_length(points):
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points)))


def _compute_curvature_profile(xs, ys):
    """Compute discrete curvature at each interior point.

    Uses the Menger curvature formula: kappa = 2*sin(theta) / ds,
    where theta is the turning angle between consecutive segments
    and ds is the average step size.
    Returns (max_kappa, curvatures_list).
    """
    n = len(xs)
    if n < 3:
        return 0.0, [0.0] * n
    curvatures = [0.0]
    max_k = 0.0
    for i in range(1, n - 1):
        v1x = xs[i] - xs[i - 1]
        v1y = ys[i] - ys[i - 1]
        v2x = xs[i + 1] - xs[i]
        v2y = ys[i + 1] - ys[i]
        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y
        theta = abs(math.atan2(cross, dot))
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        ds = 0.5 * (n1 + n2)
        kappa = 2.0 * math.sin(theta) / ds if ds > 1e-12 else 0.0
        curvatures.append(kappa)
        if kappa > max_k:
            max_k = kappa
    curvatures.append(0.0)
    return max_k, curvatures


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/costmap")
def api_costmap():
    return jsonify(
        full_width=esdf_map.width, full_height=esdf_map.height,
        resolution=esdf_map.resolution,
        extent_x=esdf_map.world_width, extent_y=esdf_map.world_height,
        png=COSTMAP_B64, esdf_png=ESDF_B64, esdf_schemes=ESDF_SCHEMES,
    )


@app.route("/api/query")
def api_query():
    """Query map data at a world coordinate (x, y)."""
    try:
        x = float(request.args.get("x", 0))
        y = float(request.args.get("y", 0))
    except (ValueError, TypeError):
        return jsonify(error="invalid coordinates"), 400

    esdf_val = esdf_map.get_distance(x, y)
    res = esdf_map.resolution
    px = int(x / res) if res > 0 else 0
    py = int(y / res) if res > 0 else 0
    in_bounds = (0 <= px < esdf_map.width and 0 <= py < esdf_map.height)
    occ = int(OCC_STATUS_GRID[py, px]) if in_bounds else None
    return jsonify(
        x=round(x, 3), y=round(y, 3),
        esdf=round(esdf_val, 4) if in_bounds else None,
        occ=occ,
        pixel_x=px, pixel_y=py,
        in_bounds=in_bounds,
    )


@app.route("/api/plan", methods=["POST"])
def api_plan():
    body = request.get_json(force=True, silent=True) or {}
    try:
        sx, sy = float(body["start"][0]), float(body["start"][1])
        gx, gy = float(body["goal"][0]), float(body["goal"][1])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"无效的起点/终点: {e}"), 400

    start_ok = _is_free(sx, sy)
    goal_ok = _is_free(gx, gy)

    if not start_ok or not goal_ok:
        r = ""
        if not start_ok: r = f"起点({sx:.1f},{sy:.1f})在障碍物中"
        if not goal_ok: r += ("，" if r else "") + f"终点({gx:.1f},{gy:.1f})在障碍物中"
        return jsonify(found=False, reason=r, plan_ms=0, start_ok=start_ok, goal_ok=goal_ok)

    pm = _make_smoother_params(body)
    robot_radius = pm.robot_radius

    t0 = time.perf_counter()
    astar_res = cs2d.astar_solve(esdf_map, sx, sy, gx, gy, robot_radius)
    plan_ms = (time.perf_counter() - t0) * 1000

    if not astar_res.success:
        return jsonify(found=False, reason=f"无路径 (起点/终点不连通, A* 半径 {robot_radius:.2f}m)",
                       plan_ms=round(plan_ms,1), start_ok=True, goal_ok=True)

    raw = list(zip(astar_res.x, astar_res.y))

    dp = _downsample_path(raw, body.get("downsample", 3))
    xs = [p[0] for p in dp]; ys = [p[1] for p in dp]

    sm = cs2d.PathSmoother2D(pm)
    t1 = time.perf_counter()
    res = sm.smooth(xs, ys, esdf_map)
    smooth_ms = (time.perf_counter() - t1) * 1000
    cls = [esdf_map.get_distance(x, y) for x, y in zip(res.x, res.y)]
    cost_breakdown = compute_path_cost_breakdown(res.x, res.y, pm, esdf_map)

    smoothed_points = list(zip(res.x, res.y))
    max_kappa, kappa_profile = _compute_curvature_profile(res.x, res.y)

    return jsonify(
        found=True, start_ok=True, goal_ok=True,
        plan_ms=round(plan_ms,1), smooth_ms=round(smooth_ms,1),
        robot_radius=robot_radius,
        robot_radius_fallback=False,
        raw_x=[p[0] for p in raw], raw_y=[p[1] for p in raw],
        raw_points=len(raw), raw_length=round(_path_length(raw),2),
        ds_x=xs, ds_y=ys, ds_points=len(xs),
        smooth_x=res.x, smooth_y=res.y, smooth_points=len(res.x),
        smooth_length=round(_path_length(smoothed_points),2),
        success=res.success, final_cost=res.final_cost,
        solver_cost=res.final_cost,
        solver_report=res.report,
        path_cost=cost_breakdown["total"],
        cost_breakdown=cost_breakdown,
        iterations=res.iterations, solve_time_ms=res.solve_time_ms,
        min_clearance=round(min(cls),4) if cls else 0,
        clearances=cls,
        max_curvature=round(max_kappa,6),
        curvature_profile=[round(k,6) for k in kappa_profile],
    )


if __name__ == "__main__":
    init_map()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    debug = os.environ.get("CERES_WEB_DEBUG", "0").lower() not in ("0", "false", "no")
    use_reloader = os.environ.get("CERES_WEB_RELOAD", "0").lower() not in ("0", "false", "no")
    print(f"\n  → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=use_reloader)
