/*  Clothoid Smoother Web Demo – Canvas Frontend  */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];

  /* ── i18n ──────────────────────────────────────────────────── */
  const I18N = {
    en: {
      hero_eyebrow: "Clothoid-based Path Smoother",
      hero_title: "A* + Clothoid Smoother Lab",
      hero_subtitle: "Visualize clothoid-based path smoothing on an occupancy map. Drag start/goal, tune parameters, and compare raw vs optimized paths.",
      hero_status: "Drag the Start and Goal markers on the map, then click Run Planning.",
      tab_session: "Session", tab_weights: "Weights", tab_solver: "Solver", tab_layers: "Layers",
      session_title: "Session", drag_scene: "Drag scene",
      start: "Start", goal: "Goal", cursor: "Cursor", zoom: "Zoom",
      start_heading: "Start Heading", goal_heading: "Goal Heading",
      start_constraint: "Start Constraint", goal_constraint: "Goal Constraint",
      keep_start: "Keep start orientation", keep_goal: "Keep goal orientation",
      start_heading_label: "Start Heading: {v} deg", goal_heading_label: "Goal Heading: {v} deg",
      weights_title: "Smoother Weights", clothoid: "Clothoid",
      model_weight: "Model Weight: {v}", fix_weight: "Fix Weight: {v}",
      obstacle_weight: "Obstacle Weight: {v}", ref_path_weight: "Reference Path Weight: {v}",
      curvature_weight: "Curvature Weight: {v}", curvature_rate_weight: "Curvature-Rate Weight: {v}",
      spacing_weight: "Spacing Weight: {v}", path_length_weight: "Path Length Weight: {v}",
      max_curvature: "Max Curvature (1/m): {v}", min_turn_radius: "Min turning radius: {v} m",
      solver_title: "Solver", ceres: "Ceres",
      ref_spacing: "Reference Spacing (m): {v}", max_iterations: "Max Iterations: {v}",
      max_time: "Max Solver Time (s): {v}", safe_distance: "Obstacle Safe Distance (m): {v}",
      debug: "Debug logging",
      layers_title: "Layers", toggle: "toggle",
      costmap: "Costmap", costmap_hint: "obstacle field",
      esdf: "ESDF", esdf_hint: "distance field",
      start_goal: "Start / Goal", start_goal_hint: "endpoints",
      astar_raw: "A* Raw", astar_hint: "dense grid path",
      reference: "Reference", reference_hint: "downsampled input",
      smoothed: "Smoothed", smoothed_hint: "clothoid output",
      run_planning: "Run Planning", reset_view: "Reset View",
      world: "World", cells: "Cells",
      map_overview: "Map", map_title: "Map Overview", occupancy: "occupancy",
      grid: "Grid", resolution: "Resolution", origin: "Origin",
      stats: "Stats", run_statistics: "Run Statistics",
      astar_time: "A* Time", smooth_time: "Smooth Time",
      astar_pts: "A* Points", ref_pts: "Ref Points", opt_knots: "Opt Knots",
      opt_pts: "Opt Points", raw_length: "Raw Length", ref_length: "Ref Length", opt_length: "Opt Length",
      run_note: "Run planning to see statistics.",
      status_running: "Running planning...",
      status_ok: "Smoothed in {t}ms ({k} knots, {n} pts)",
      status_fallback: "Smoother fallback: {msg}",
      status_error: "Network error: {msg}",
      profile_title: "Path Profiles", idle: "idle", ok: "ok",
      curvature_title: "Curvature k(s)", peak: "Peak |kappa|", mean: "Mean |kappa|",
      no_data: "no data",
    },
    zh: {
      hero_eyebrow: "基于Clothoid的路径平滑器",
      hero_title: "A* + Clothoid 平滑器实验室",
      hero_subtitle: "在占用地图上可视化clothoid路径平滑。拖动起点/终点，调节参数，对比原始与优化路径。",
      hero_status: "在地图上拖动起点和终点标记，然后点击「运行规划」。",
      tab_session: "会话", tab_weights: "权重", tab_solver: "求解器", tab_layers: "图层",
      session_title: "会话", drag_scene: "拖动场景",
      start: "起点", goal: "终点", cursor: "光标", zoom: "缩放",
      start_heading: "起点朝向", goal_heading: "终点朝向",
      start_constraint: "起点约束", goal_constraint: "终点约束",
      keep_start: "保持起点朝向", keep_goal: "保持终点朝向",
      start_heading_label: "起点朝向: {v} 度", goal_heading_label: "终点朝向: {v} 度",
      weights_title: "平滑器权重", clothoid: "Clothoid",
      model_weight: "模型权重: {v}", fix_weight: "固定权重: {v}",
      obstacle_weight: "障碍物权重: {v}", ref_path_weight: "参考路径权重: {v}",
      curvature_weight: "曲率权重: {v}", curvature_rate_weight: "曲率变化率权重: {v}",
      spacing_weight: "间距权重: {v}", path_length_weight: "路径长度权重: {v}",
      max_curvature: "最大曲率 (1/m): {v}", min_turn_radius: "最小转弯半径: {v} m",
      solver_title: "求解器", ceres: "Ceres",
      ref_spacing: "参考间距 (m): {v}", max_iterations: "最大迭代次数: {v}",
      max_time: "最大求解时间 (s): {v}", safe_distance: "障碍物安全距离 (m): {v}",
      debug: "调试日志",
      layers_title: "图层", toggle: "切换",
      costmap: "代价地图", costmap_hint: "障碍物场",
      esdf: "ESDF", esdf_hint: "距离场",
      start_goal: "起点 / 终点", start_goal_hint: "端点",
      astar_raw: "A* 原始", astar_hint: "密集网格路径",
      reference: "参考路径", reference_hint: "降采样输入",
      smoothed: "平滑路径", smoothed_hint: "clothoid输出",
      run_planning: "运行规划", reset_view: "重置视图",
      world: "世界", cells: "栅格",
      map_overview: "地图", map_title: "地图概览", occupancy: "占用地图",
      grid: "栅格", resolution: "分辨率", origin: "原点",
      stats: "统计", run_statistics: "运行统计",
      astar_time: "A* 耗时", smooth_time: "平滑耗时",
      astar_pts: "A* 路点", ref_pts: "参考路点", opt_knots: "优化节点",
      opt_pts: "优化路点", raw_length: "原始长度", ref_length: "参考长度", opt_length: "优化长度",
      run_note: "运行规划以查看统计信息。",
      status_running: "正在运行规划...",
      status_ok: "平滑完成 {t}ms ({k} 节点, {n} 路点)",
      status_fallback: "平滑器回退: {msg}",
      status_error: "网络错误: {msg}",
      profile_title: "路径曲线图", idle: "空闲", ok: "完成",
      curvature_title: "曲率 k(s)", peak: "峰值 |kappa|", mean: "均值 |kappa|",
      no_data: "暂无数据",
    },
  };

  let lang = "zh";
  function t(key, vars) {
    let s = (I18N[lang] || I18N.en)[key] || (I18N.en)[key] || key;
    if (vars) Object.entries(vars).forEach(([k, v]) => { s = s.replace(`{${k}}`, v); });
    return s;
  }

  /* ── state ─────────────────────────────────────────────────── */
  let costmapData = null;
  let esdfData = null;
  let mapMeta = null;
  let astarResult = null;

  let startWorld = { x: 10.0, y: 10.0 };
  let goalWorld = { x: 50.0, y: 30.0 };
  let startDragging = false;
  let goalDragging = false;
  let panDragging = false;
  let lastMouseWorld = null;

  let viewOffsetX = 0;
  let viewOffsetY = 0;
  let viewScale = 1.0;

  /* ── canvas setup ─────────────────────────────────────────── */
  const canvas = $("#map-canvas");
  const ctx = canvas.getContext("2d");

  /* ── ESDF colormap (viridis-like) ─────────────────────────── */
  function esdfColor(val, maxVal) {
    const t = Math.min(1.0, val / maxVal);
    // Viridis-inspired: dark purple -> blue -> teal -> green -> yellow
    const r = Math.floor(68 + t * (253 - 68));
    const g = Math.floor(1 + t * (231 - 1));
    const b = Math.floor(84 + t * (37 - 84));
    return [r, g, b];
  }

  /* ── API ───────────────────────────────────────────────────── */
  async function fetchCostmap() {
    const r = await fetch("/api/costmap");
    const d = await r.json();
    costmapData = d.data;
    esdfData = d.esdf;
    mapMeta = {
      size_x: d.size_x, size_y: d.size_y,
      resolution: d.resolution,
      origin_x: d.origin_x, origin_y: d.origin_y,
    };
    updateMapInfo();
    draw();
  }

  async function runPlanning() {
    const btn = $("#run-btn");
    btn.disabled = true;
    setStatus(t("status_running"), "");
    try {
      const body = {
        start_x: startWorld.x, start_y: startWorld.y,
        goal_x: goalWorld.x, goal_y: goalWorld.y,
        start_yaw_deg: parseFloat($("#start_yaw_deg").value),
        goal_yaw_deg: parseFloat($("#goal_yaw_deg").value),
        keep_start_orientation: $("#keep_start_orientation").checked,
        keep_goal_orientation: $("#keep_goal_orientation").checked,
        reference_spacing_target_m: parseFloat($("#reference_spacing_target_m").value),
        max_curvature: parseFloat($("#max_curvature").value),
        max_time: parseFloat($("#max_time").value),
        max_iterations: parseInt($("#max_iterations").value),
        costmap_weight: parseFloat($("#costmap_weight").value),
        model_weight: parseFloat($("#model_weight").value),
        fix_weight: parseFloat($("#fix_weight").value),
        kinematic_curvature_weight: parseFloat($("#kinematic_curvature_weight").value),
        kinematic_curvature_rate_weight: parseFloat($("#kinematic_curvature_rate_weight").value),
        kinematic_spacing_weight: parseFloat($("#kinematic_spacing_weight").value),
        path_length_weight: parseFloat($("#path_length_weight").value),
        reference_path_weight: parseFloat($("#reference_path_weight").value),
        obstacle_safe_distance: parseFloat($("#obstacle_safe_distance").value),
        debug: $("#debug").checked,
      };
      const r = await fetch("/api/smooth", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) {
        setStatus(d.message || "Request failed", "error");
        return;
      }
      astarResult = d;
      updateStats(d);
      if (d.smooth_success) {
        setStatus(t("status_ok", { t: d.smooth_time_ms, k: d.num_opt_knots, n: d.num_opt_pts }), "ok");
      } else {
        const msg = d.smooth_message || (d.smooth_error && d.smooth_error.code) || "smoother failed";
        setStatus(t("status_fallback", { msg }), "error");
      }
      draw();
    } catch (e) {
      setStatus(t("status_error", { msg: e.message }), "error");
    } finally {
      btn.disabled = false;
    }
  }

  /* ── helpers ───────────────────────────────────────────────── */
  function setStatus(msg, cls) {
    const el = $("#status-msg");
    el.textContent = msg;
    el.className = "status-msg" + (cls ? " " + cls : "");
    const hero = $("#hero-status");
    if (hero) hero.textContent = msg;
  }

  function updateMapInfo() {
    if (!mapMeta) return;
    const ww = (mapMeta.size_x * mapMeta.resolution).toFixed(1);
    const wh = (mapMeta.size_y * mapMeta.resolution).toFixed(1);
    const txt = `${ww} x ${wh} m`;
    $("#map-world-size-toolbar").textContent = txt;
    $("#map-cells-toolbar").textContent = `${mapMeta.size_x} x ${mapMeta.size_y}`;
    $("#map-grid").textContent = `${mapMeta.size_x} x ${mapMeta.size_y}`;
    $("#map-world-size").textContent = txt;
    $("#map-resolution").textContent = mapMeta.resolution + " m";
    $("#map-origin").textContent = `(${mapMeta.origin_x}, ${mapMeta.origin_y})`;
  }

  function updateStats(d) {
    const ss = $("#smooth-state");
    if (ss) ss.textContent = d.smooth_success ? t("ok") : "fallback";
    $("#info-astar-time").textContent = d.astar_time_ms + " ms";
    $("#info-smooth-time").textContent = d.smooth_time_ms + " ms";
    $("#info-astar-pts").textContent = d.num_astar_pts;
    $("#info-ref-pts").textContent = d.num_ref_pts;
    $("#info-opt-knots").textContent = d.num_opt_knots;
    $("#info-opt-pts").textContent = d.num_opt_pts;
    $("#info-raw-length").textContent = d.raw_path_length_m + " m";
    $("#info-ref-length").textContent = d.ref_path_length_m + " m";
    $("#info-opt-length").textContent = d.opt_path_length_m + " m";
    const ps = $("#pipeline-summary");
    if (ps) ps.textContent = d.smooth_success
      ? `A* ${d.astar_time_ms}ms -> Smooth ${d.smooth_time_ms}ms`
      : `A* ${d.astar_time_ms}ms -> FAILED`;
    updateCurvatureProfile(d);
  }

  /* ── curvature chart ───────────────────────────────────────── */
  function updateCurvatureProfile(d) {
    const state = $("#curvature-state");
    if (!d || !d.opt_x || d.opt_x.length < 3) {
      if (state) state.textContent = t("no_data");
      return;
    }
    const xs = d.opt_x, ys = d.opt_y;
    const n = xs.length;
    const sArr = [0];
    for (let i = 1; i < n; i++) {
      sArr.push(sArr[i-1] + Math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1]));
    }
    const kappa = [];
    for (let i = 1; i < n - 1; i++) {
      const dx1 = xs[i] - xs[i-1], dy1 = ys[i] - ys[i-1];
      const dx2 = xs[i+1] - xs[i], dy2 = ys[i+1] - ys[i];
      const d1 = Math.hypot(dx1, dy1), d2 = Math.hypot(dx2, dy2);
      if (d1 < 1e-9 || d2 < 1e-9) { kappa.push(0); continue; }
      const cross = (dx1/d1)*(dy2/d2) - (dy1/d1)*(dx2/d2);
      const dot = (dx1/d1)*(dx2/d2) + (dy1/d1)*(dy2/d2);
      kappa.push(2 * Math.atan2(cross, 1 + dot) / ((d1 + d2) / 2));
    }
    kappa.push(kappa.length > 0 ? kappa[kappa.length-1] : 0);
    const sK = sArr.slice(1, -1);

    if (window.Plotly) {
      Plotly.newPlot("curvature-chart", [{
        x: sK, y: kappa.slice(1, -1), type: "scatter", mode: "lines",
        line: { color: "#bf3657", width: 2 },
        name: "kappa",
      }], {
        margin: { l: 48, r: 16, t: 8, b: 36 },
        xaxis: { title: "s (m)", gridcolor: "rgba(104,86,58,0.12)" },
        yaxis: { title: "kappa (1/m)", gridcolor: "rgba(104,86,58,0.12)" },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(255,250,240,0.94)",
        font: { family: "Avenir Next, sans-serif", color: "#223127", size: 11 },
      }, { responsive: true, displayModeBar: false });
    }
    const absK = kappa.map(Math.abs);
    const peak = Math.max(...absK);
    const mean = absK.reduce((a,b)=>a+b, 0) / absK.length;
    $("#curvature-peak").textContent = peak.toFixed(4) + " /m";
    $("#curvature-mean").textContent = mean.toFixed(4) + " /m";
    if (state) state.textContent = t("ok");
  }

  /* ── coordinate transforms ─────────────────────────────────── */
  function worldToCanvas(wx, wy) {
    if (!mapMeta) return [0, 0];
    const cx = (wx - mapMeta.origin_x) / mapMeta.resolution;
    const cy_img = (wy - mapMeta.origin_y) / mapMeta.resolution;
    const cy = mapMeta.size_y - 1 - cy_img;
    const cw = mapMeta.size_x * viewScale;
    const ch = mapMeta.size_y * viewScale;
    const px = (canvas.width - cw) / 2 + cx * viewScale + viewOffsetX;
    const py = (canvas.height - ch) / 2 + cy * viewScale + viewOffsetY;
    return [px, py];
  }

  function canvasToWorld(px, py) {
    if (!mapMeta) return [0, 0];
    const cw = mapMeta.size_x * viewScale;
    const ch = mapMeta.size_y * viewScale;
    const cx = px - (canvas.width - cw) / 2 - viewOffsetX;
    const cy_canvas = py - (canvas.height - ch) / 2 - viewOffsetY;
    const cx_cell = cx / viewScale;
    const cy_img = mapMeta.size_y - 1 - cy_canvas / viewScale;
    const wx = mapMeta.origin_x + (cx_cell + 0.5) * mapMeta.resolution;
    const wy = mapMeta.origin_y + (cy_img + 0.5) * mapMeta.resolution;
    return [wx, wy];
  }

  /* ── draw ──────────────────────────────────────────────────── */
  function draw() {
    if (!costmapData || !mapMeta) return;
    const W = mapMeta.size_x, H = mapMeta.size_y;
    const imgData = ctx.createImageData(W, H);
    const showEsdf = $("#layer_esdf") && $("#layer_esdf").checked;

    let esdfMax = 1.0;
    if (showEsdf && esdfData) {
      for (let i = 0; i < esdfData.length; i++) {
        if (esdfData[i] > esdfMax) esdfMax = esdfData[i];
      }
    }

    for (let i = 0; i < W * H; i++) {
      let r, g, b;
      if (showEsdf && esdfData) {
        const [cr, cg, cb] = esdfColor(esdfData[i], esdfMax);
        r = cr; g = cg; b = cb;
      } else {
        const val = costmapData[i];
        if (val === 255) { r = 176; g = 184; b = 191; }       // unknown: gray
        else if (val >= 254) { r = 47; g = 52; b = 64; }       // lethal: dark
        else if (val > 0) {
          const t = val / 253;
          r = Math.floor(239 + t * (217 - 239));
          g = Math.floor(226 + t * (122 - 226));
          b = Math.floor(194 + t * (43 - 194));
        } else { r = 255; g = 252; b = 246; }                   // free: warm white
      }
      imgData.data[i*4] = r; imgData.data[i*4+1] = g; imgData.data[i*4+2] = b; imgData.data[i*4+3] = 255;
    }

    const offCanvas = document.createElement("canvas");
    offCanvas.width = W; offCanvas.height = H;
    offCanvas.getContext("2d").putImageData(imgData, 0, 0);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    const cw = W * viewScale, ch = H * viewScale;
    const dx = (canvas.width - cw) / 2 + viewOffsetX;
    const dy = (canvas.height - ch) / 2 + viewOffsetY;
    ctx.drawImage(offCanvas, dx, dy, cw, ch);

    // Draw grid axes
    drawAxes(dx, dy, cw, ch);

    if (astarResult) {
      if ($("#layer_astar") && $("#layer_astar").checked && astarResult.astar_x) {
        drawPath(astarResult.astar_x, astarResult.astar_y, "rgba(43,113,186,0.7)", 2);
      }
      if ($("#layer_reference") && $("#layer_reference").checked && astarResult.ref_x) {
        drawPath(astarResult.ref_x, astarResult.ref_y, "#d97a2b", 2.5);
      }
      if ($("#layer_smoothed") && $("#layer_smoothed").checked && astarResult.opt_x) {
        drawPath(astarResult.opt_x, astarResult.opt_y, "#bf3657", 3);
      }
    }

    if ($("#layer_markers") && $("#layer_markers").checked) {
      drawMarker(startWorld.x, startWorld.y, "#208d76", "S");
      drawMarker(goalWorld.x, goalWorld.y, "#d94f34", "G");
    }

    updateReadouts();
  }

  function drawAxes(dx, dy, cw, ch) {
    if (!mapMeta) return;
    ctx.save();
    ctx.strokeStyle = "rgba(15,92,80,0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "rgba(15,92,80,0.6)";
    ctx.textAlign = "center";
    const step_m = 5;
    const step_cells = step_m / mapMeta.resolution;
    for (let cx = 0; cx <= mapMeta.size_x; cx += step_cells) {
      const px = dx + cx * viewScale;
      if (px < dx || px > dx + cw) continue;
      ctx.beginPath(); ctx.moveTo(px, dy); ctx.lineTo(px, dy + ch); ctx.stroke();
      const wx = (cx * mapMeta.resolution).toFixed(0);
      ctx.fillText(wx + "m", px, dy + ch + 12);
    }
    ctx.textAlign = "right";
    for (let cy = 0; cy <= mapMeta.size_y; cy += step_cells) {
      const py = dy + cy * viewScale;
      if (py < dy || py > dy + ch) continue;
      ctx.beginPath(); ctx.moveTo(dx, py); ctx.lineTo(dx + cw, py); ctx.stroke();
      const wy = ((mapMeta.size_y - cy) * mapMeta.resolution).toFixed(0);
      ctx.fillText(wy + "m", dx - 4, py + 3);
    }
    ctx.restore();
  }

  function drawPath(xs, ys, color, width) {
    if (!xs || xs.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.beginPath();
    const [sx, sy] = worldToCanvas(xs[0], ys[0]);
    ctx.moveTo(sx, sy);
    for (let i = 1; i < xs.length; i++) {
      const [px, py] = worldToCanvas(xs[i], ys[i]);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  function drawMarker(wx, wy, color, label) {
    const [px, py] = worldToCanvas(wx, wy);
    ctx.beginPath();
    ctx.arc(px, py, 12, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, px, py);
  }

  function updateReadouts() {
    $("#start-coord").textContent = `(${startWorld.x.toFixed(1)}, ${startWorld.y.toFixed(1)})`;
    $("#goal-coord").textContent = `(${goalWorld.x.toFixed(1)}, ${goalWorld.y.toFixed(1)})`;
    $("#zoom-level").textContent = viewScale.toFixed(2) + "x";
    const sdeg = $("#start_yaw_deg").value;
    const gdeg = $("#goal_yaw_deg").value;
    $("#val_start_yaw_deg").textContent = sdeg + (lang === "zh" ? " 度" : " deg");
    $("#val_goal_yaw_deg").textContent = gdeg + (lang === "zh" ? " 度" : " deg");
    $("#start-heading-readout").textContent = sdeg + (lang === "zh" ? " 度" : " deg");
    $("#goal-heading-readout").textContent = gdeg + (lang === "zh" ? " 度" : " deg");
    $("#start-constraint-readout").textContent = $("#keep_start_orientation").checked ? (lang === "zh" ? "启用" : "Enabled") : (lang === "zh" ? "禁用" : "Disabled");
    $("#goal-constraint-readout").textContent = $("#keep_goal_orientation").checked ? (lang === "zh" ? "启用" : "Enabled") : (lang === "zh" ? "禁用" : "Disabled");
  }

  /* ── mouse interaction ─────────────────────────────────────── */
  function getMouseWorld(e) {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    return canvasToWorld(px, py);
  }

  function hitTest(wx, wy, target, radius) {
    return Math.hypot(wx - target.x, wy - target.y) < radius;
  }

  canvas.addEventListener("mousedown", (e) => {
    const [wx, wy] = getMouseWorld(e);
    if (hitTest(wx, wy, startWorld, 1.5)) { startDragging = true; return; }
    if (hitTest(wx, wy, goalWorld, 1.5)) { goalDragging = true; return; }
    panDragging = true;
    lastMouseWorld = { x: e.clientX, y: e.clientY };
    canvas.classList.add("is-dragging");
  });

  canvas.addEventListener("mousemove", (e) => {
    const [wx, wy] = getMouseWorld(e);
    if (mapMeta) {
      $("#cursor-coord").textContent = `(${wx.toFixed(2)}, ${wy.toFixed(2)})`;
    }
    if (startDragging) { startWorld.x = Math.max(0, Math.min(mapMeta.size_x * mapMeta.resolution, wx)); startWorld.y = Math.max(0, Math.min(mapMeta.size_y * mapMeta.resolution, wy)); draw(); return; }
    if (goalDragging) { goalWorld.x = Math.max(0, Math.min(mapMeta.size_x * mapMeta.resolution, wx)); goalWorld.y = Math.max(0, Math.min(mapMeta.size_y * mapMeta.resolution, wy)); draw(); return; }
    if (panDragging && lastMouseWorld) {
      viewOffsetX += e.clientX - lastMouseWorld.x;
      viewOffsetY += e.clientY - lastMouseWorld.y;
      lastMouseWorld = { x: e.clientX, y: e.clientY };
      draw();
    }
  });

  canvas.addEventListener("mouseup", () => { startDragging = false; goalDragging = false; panDragging = false; lastMouseWorld = null; canvas.classList.remove("is-dragging"); });
  canvas.addEventListener("mouseleave", () => { startDragging = false; goalDragging = false; panDragging = false; lastMouseWorld = null; canvas.classList.remove("is-dragging"); });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    viewScale = Math.max(0.2, Math.min(5.0, viewScale * factor));
    draw();
  }, { passive: false });

  canvas.addEventListener("dblclick", () => { viewOffsetX = 0; viewOffsetY = 0; viewScale = 1.0; draw(); });

  /* ── panel switcher ────────────────────────────────────────── */
  $$(".rail-switcher__nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-panel-target");
      const nav = btn.closest(".rail-switcher__nav");
      nav.querySelectorAll(".rail-switcher__nav-btn").forEach((b) => { b.classList.remove("is-active"); b.setAttribute("aria-selected", "false"); });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");
      const container = nav.closest(".rail-switcher").querySelector(".rail-switcher__panel-column") || nav.closest(".rail-switcher");
      container.querySelectorAll(".rail-switcher__panel").forEach((p) => { p.hidden = true; p.classList.remove("is-active"); });
      const panel = container.querySelector(`[data-panel-id="${target}"]`);
      if (panel) { panel.hidden = false; panel.classList.add("is-active"); }
    });
  });

  /* ── slider readouts ───────────────────────────────────────── */
  function updateSliderLabels() {
    $$("input[type='range']").forEach((sl) => {
      const key = "val_" + sl.id;
      const valEl = $("#" + key);
      if (valEl) {
        const v = sl.value;
        if (sl.id === "max_curvature") {
          const r = parseFloat(v);
          valEl.textContent = v;
          const tr = $("#val_min_turn_radius");
          if (tr) tr.textContent = t("min_turn_radius", { v: r > 0 ? (1/r).toFixed(2) : "--" });
        } else {
          valEl.textContent = v;
        }
      }
    });
  }

  $$("input[type='range']").forEach((sl) => {
    sl.addEventListener("input", () => { updateSliderLabels(); });
  });

  /* ── language switch ───────────────────────────────────────── */
  function applyLanguage() {
    // Update text content for all i18n elements
    $$("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      const text = t(key);
      if (text !== key) el.textContent = text;
    });
    // Update data-i18n-html elements
    $$("[data-i18n-html]").forEach((el) => {
      const key = el.getAttribute("data-i18n-html");
      const text = t(key);
      if (text !== key) el.innerHTML = text;
    });
    // Update button/label text directly
    const runBtn = $("#run-btn");
    if (runBtn) runBtn.textContent = t("run_planning");
    const clearBtn = $("#clear-btn");
    if (clearBtn) clearBtn.textContent = t("reset_view");

    // Tab labels
    $$(".rail-switcher__nav-btn").forEach((btn) => {
      const target = btn.getAttribute("data-panel-target");
      const keyMap = {
        "panel-session": "tab_session", "panel-weights": "tab_weights",
        "panel-solver": "tab_solver", "panel-layers": "tab_layers",
        "panel-map-overview": "tab_session", "panel-run-statistics": "tab_stats",
      };
      const k = keyMap[target];
      if (k) { const s = btn.querySelector("span"); if (s) s.textContent = t(k); else btn.textContent = t(k); }
    });

    // Section headings
    $$(".section-heading h2").forEach((h2) => {
      const section = h2.closest("[data-panel-id]");
      if (!section) return;
      const id = section.getAttribute("data-panel-id");
      const map = {
        "panel-session": "session_title", "panel-weights": "weights_title",
        "panel-solver": "solver_title", "panel-layers": "layers_title",
        "panel-map-overview": "map_title", "panel-run-statistics": "run_statistics",
      };
      if (map[id]) h2.textContent = t(map[id]);
    });

    updateSliderLabels();
    updateReadouts();
  }

  const langBtn = $("#language-switch");
  if (langBtn) {
    langBtn.addEventListener("change", () => { lang = langBtn.value; applyLanguage(); });
  }

  /* ── buttons ───────────────────────────────────────────────── */
  $("#run-btn").addEventListener("click", runPlanning);
  $("#clear-btn").addEventListener("click", () => { viewOffsetX = 0; viewOffsetY = 0; viewScale = 1.0; draw(); });

  /* ── init ──────────────────────────────────────────────────── */
  fetchCostmap();
})();
