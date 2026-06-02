// Hybrid A* Debugger — interactive map frontend
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('map-canvas');
  const ctx = canvas.getContext('2d');
  const loupeCanvas = document.getElementById('loupe-canvas');
  const loupeCtx = loupeCanvas.getContext('2d');

  // ---- i18n ----
  const LANGUAGE_STORAGE_KEY = 'hybrid-astar-ui-language';
  const messages = {
    en: {
      'hero.title': 'Hybrid A* Planner Debugger',
      'hero.subtitle': 'Interactive costmap editor with Hybrid A* path planning visualization. Drag start/goal markers and obstacle blocks to explore planner behavior.',
      'hero.status': 'Left-drag endpoints or obstacle blocks to edit the scene. Scroll to zoom, double-click to reset view.',
      'language.label': 'Language',
      'session.title': 'Session',
      'session.start': 'Start',
      'session.goal': 'Goal',
      'session.cursor': 'Cursor',
      'session.zoom': 'Zoom',
      'session.startHeading': 'Start Heading',
      'session.goalHeading': 'Goal Heading',
      'session.startHeadingLabel': 'Start Heading: <span id="val_start_theta_deg">0</span> deg',
      'session.goalHeadingLabel': 'Goal Heading: <span id="val_goal_theta_deg">0</span> deg',
      'session.planBtn': '▶ Plan',
      'session.clearBtn': 'Clear Path',
      'session.resetBtn': 'Reset View',
      'motion.title': 'Motion Model',
      'motion.modelLabel': 'Model',
      'motion.dubin': 'Dubin (forward only)',
      'motion.reedsShepp': 'Reeds-Shepp (forward + reverse)',
      'motion.hint': 'Dubin uses 3 primitives (straight, left, right). Reeds-Shepp adds reverse and can produce cusps.',
      'planner.title': 'Planner Parameters',
      'planner.minTurnRadius': 'Min Turning Radius (m): <span id="val_minimum_turning_radius">1.50</span>',
      'planner.tolerance': 'Tolerance (m): <span id="val_tolerance">0.25</span>',
      'planner.angleBins': 'Angle Bins: <span id="val_angle_bins">72</span>',
      'planner.maxTime': 'Max Planning Time (s): <span id="val_max_planning_time">5.0</span>',
      'planner.maxIter': 'Max Iterations: <span id="val_max_iterations">1000000</span>',
      'planner.reversePenalty': 'Reverse Penalty: <span id="val_reverse_penalty">2.00</span>',
      'planner.costPenalty': 'Cost Penalty: <span id="val_cost_penalty">2.00</span>',
      'planner.smoothPath': 'Smooth Path',
      'planner.allowUnknown': 'Allow Unknown',
      'layers.title': 'Layers',
      'layers.costmap': 'Costmap',
      'layers.path': 'Planned Path',
      'layers.axes': 'Map Axes',
      'layers.grid': 'Grid Lines',
      'map.title': 'Map Overview',
      'map.grid': 'Grid',
      'map.resolution': 'Resolution',
      'map.origin': 'Origin',
      'stats.title': 'Run Statistics',
      'stats.status': 'Status',
      'stats.time': 'Time',
      'stats.poses': 'Path Poses',
      'stats.length': 'Path Length',
      'loupe.title': 'Cursor Inspector',
      'loupe.cost': 'Cell Cost',
      'loupe.world': 'World XY',
      'status.planning': 'Planning...',
      'status.pathFound': 'Path found: {poses} poses in {time} ms',
      'status.noPath': 'No path: {message}',
      'status.loadError': 'Failed to load costmap: {message}',
      'status.planError': 'Planning failed: {message}',
      'status.obstacleError': 'Failed to update obstacles: {message}',
    },
    zh: {
      'hero.title': 'Hybrid A* 规划器调试器',
      'hero.subtitle': '交互式代价地图编辑器，支持 Hybrid A* 路径规划可视化。拖拽起终点标记和障碍物方块来探索规划器行为。',
      'hero.status': '左键拖拽端点或障碍物方块来编辑场景，滚轮缩放，双击恢复全景。',
      'language.label': '语言',
      'session.title': '会话',
      'session.start': '起点',
      'session.goal': '终点',
      'session.cursor': '光标',
      'session.zoom': '缩放',
      'session.startHeading': '起点朝向',
      'session.goalHeading': '终点朝向',
      'session.startHeadingLabel': '起点朝向: <span id="val_start_theta_deg">0</span> deg',
      'session.goalHeadingLabel': '终点朝向: <span id="val_goal_theta_deg">0</span> deg',
      'session.planBtn': '▶ 规划',
      'session.clearBtn': '清除路径',
      'session.resetBtn': '重置视图',
      'motion.title': '运动模型',
      'motion.modelLabel': '模型',
      'motion.dubin': 'Dubin（仅前进）',
      'motion.reedsShepp': 'Reeds-Shepp（前进+倒车）',
      'motion.hint': 'Dubin 使用 3 种运动原语（直行、左转、右转）。Reeds-Shepp 增加倒车，可产生尖点。',
      'planner.title': '规划器参数',
      'planner.minTurnRadius': '最小转弯半径 (m): <span id="val_minimum_turning_radius">1.50</span>',
      'planner.tolerance': '终点容差 (m): <span id="val_tolerance">0.25</span>',
      'planner.angleBins': '角度量化桶数: <span id="val_angle_bins">72</span>',
      'planner.maxTime': '最大规划时间 (s): <span id="val_max_planning_time">5.0</span>',
      'planner.maxIter': '最大迭代次数: <span id="val_max_iterations">1000000</span>',
      'planner.reversePenalty': '倒车惩罚: <span id="val_reverse_penalty">2.00</span>',
      'planner.costPenalty': '代价惩罚: <span id="val_cost_penalty">2.00</span>',
      'planner.smoothPath': '平滑路径',
      'planner.allowUnknown': '允许未知区域',
      'layers.title': '图层',
      'layers.costmap': '代价地图',
      'layers.path': '规划路径',
      'layers.axes': '地图坐标轴',
      'layers.grid': '网格线',
      'map.title': '地图概览',
      'map.grid': '栅格',
      'map.resolution': '分辨率',
      'map.origin': '原点',
      'stats.title': '运行统计',
      'stats.status': '状态',
      'stats.time': '时间',
      'stats.poses': '路径位姿数',
      'stats.length': '路径长度',
      'loupe.title': '光标检查器',
      'loupe.cost': '栅格代价值',
      'loupe.world': '世界坐标',
      'status.planning': '规划中...',
      'status.pathFound': '找到路径: {poses} 个位姿, 耗时 {time} ms',
      'status.noPath': '未找到路径: {message}',
      'status.loadError': '加载代价地图失败: {message}',
      'status.planError': '规划失败: {message}',
      'status.obstacleError': '更新障碍物失败: {message}',
    },
  };

  let currentLang = localStorage.getItem(LANGUAGE_STORAGE_KEY) || 'en';
  if (!messages[currentLang]) currentLang = 'en';

  function t(key, replacements) {
    let text = (messages[currentLang] && messages[currentLang][key]) || messages.en[key] || key;
    if (replacements) {
      for (const [k, v] of Object.entries(replacements)) {
        text = text.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
      }
    }
    return text;
  }

  function applyLanguage() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
      el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      el.innerHTML = t(el.dataset.i18nHtml);
    });
    // Update select option text for motion model
    const dm = document.getElementById('motion_model');
    if (dm) {
      dm.options[0].text = t('motion.dubin');
      dm.options[1].text = t('motion.reedsShepp');
    }
    // Update language selector
    const langSwitch = document.getElementById('language-switch');
    if (langSwitch) langSwitch.value = currentLang;
  }

  // Set up language switcher
  const langSwitch = document.getElementById('language-switch');
  if (langSwitch) {
    langSwitch.value = currentLang;
    langSwitch.addEventListener('change', () => {
      currentLang = langSwitch.value;
      localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLang);
      applyLanguage();
    });
  }
  applyLanguage();

  // ---- State ----
  const state = {
    costmap: null,
    sizeX: 200,
    sizeY: 200,
    resolution: 0.1,
    originX: 0,
    originY: 0,
    obstacleRects: [],
    start: { x: 1.0, y: 1.0, theta: 0 },
    goal: { x: 19.0, y: 19.0, theta: 0 },
    path: [],
    // View transform
    viewScale: 1,
    viewOffsetX: 0,
    viewOffsetY: 0,
    // Interaction
    dragging: null,  // 'start' | 'goal' | 'obstacle' | 'pan'
    dragObstacleIdx: -1,
    dragStartCanvas: null,
    dragStartWorld: null,
    lastMouseCanvas: { x: 0, y: 0 },
    lastMouseWorld: { x: 0, y: 0 },
    hoveredObstacle: -1,
  };

  // ---- Coordinate transforms ----
  function worldToCanvas(wx, wy) {
    const cx = (wx - state.originX) / state.resolution * state.viewScale + state.viewOffsetX;
    const cy = (state.sizeY - (wy - state.originY) / state.resolution) * state.viewScale + state.viewOffsetY;
    return { x: cx, y: cy };
  }

  function canvasToWorld(cx, cy) {
    const gridX = (cx - state.viewOffsetX) / state.viewScale;
    const gridY = (cy - state.viewOffsetY) / state.viewScale;
    const wx = gridX * state.resolution + state.originX;
    const wy = (state.sizeY - gridY) * state.resolution + state.originY;
    return { x: wx, y: wy };
  }

  function canvasToGrid(cx, cy) {
    const gridX = (cx - state.viewOffsetX) / state.viewScale;
    const gridY = (cy - state.viewOffsetY) / state.viewScale;
    return { x: Math.floor(gridX), y: Math.floor(gridY) };
  }

  function resetView() {
    const padX = 20, padY = 20;
    state.viewScale = Math.min(
      (canvas.width - 2 * padX) / state.sizeX,
      (canvas.height - 2 * padY) / state.sizeY
    );
    state.viewOffsetX = (canvas.width - state.sizeX * state.viewScale) / 2;
    state.viewOffsetY = (canvas.height - state.sizeY * state.viewScale) / 2;
    document.getElementById('zoom-level').textContent = (state.viewScale / (canvas.width / state.sizeX)).toFixed(2) + 'x';
  }

  // ---- Drawing ----
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#e8e0d4';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (document.getElementById('show-costmap').checked && state.costmap) {
      drawCostmap();
    }
    if (document.getElementById('show-grid').checked) {
      drawGrid();
    }
    if (document.getElementById('show-axes').checked) {
      drawAxes();
    }
    drawObstacles();
    if (document.getElementById('show-path').checked && state.path.length > 0) {
      drawPath();
    }
    drawMarker(state.start, '#147a6a', 'S');
    drawMarker(state.goal, '#c44e36', 'G');
  }

  function drawCostmap() {
    const imgData = ctx.createImageData(state.sizeX, state.sizeY);
    for (let gy = 0; gy < state.sizeY; gy++) {
      for (let gx = 0; gx < state.sizeX; gx++) {
        const cost = state.costmap[gy * state.sizeX + gx];
        // Flip Y: canvas row 0 = world top (highest gy), so map gy → imgY
        const imgY = (state.sizeY - 1 - gy);
        const idx = (imgY * state.sizeX + gx) * 4;
        if (cost === 0) {
          imgData.data[idx] = 240; imgData.data[idx+1] = 236; imgData.data[idx+2] = 228; imgData.data[idx+3] = 255;
        } else if (cost < 128) {
          const t = cost / 128;
          imgData.data[idx] = Math.round(240 - 100 * t);
          imgData.data[idx+1] = Math.round(236 - 80 * t);
          imgData.data[idx+2] = Math.round(228 - 40 * t);
          imgData.data[idx+3] = 255;
        } else if (cost < 254) {
          const t = (cost - 128) / 126;
          imgData.data[idx] = Math.round(140 + 80 * t);
          imgData.data[idx+1] = Math.round(156 - 100 * t);
          imgData.data[idx+2] = Math.round(188 - 140 * t);
          imgData.data[idx+3] = 255;
        } else {
          imgData.data[idx] = 40; imgData.data[idx+1] = 40; imgData.data[idx+2] = 40; imgData.data[idx+3] = 255;
        }
      }
    }

    // Draw to offscreen canvas, then scale up
    const off = document.createElement('canvas');
    off.width = state.sizeX; off.height = state.sizeY;
    off.getContext('2d').putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    const tl = worldToCanvas(state.originX, state.originY + state.sizeY * state.resolution);
    const br = worldToCanvas(state.originX + state.sizeX * state.resolution, state.originY);
    ctx.drawImage(off, tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    ctx.imageSmoothingEnabled = true;
  }

  function drawGrid() {
    ctx.strokeStyle = 'rgba(104,86,58,0.08)';
    ctx.lineWidth = 0.5;
    const step = Math.max(1, Math.round(5 / state.viewScale));
    for (let gx = 0; gx <= state.sizeX; gx += step) {
      const p = worldToCanvas(gx * state.resolution, 0);
      const p2 = worldToCanvas(gx * state.resolution, state.sizeY * state.resolution);
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
    for (let gy = 0; gy <= state.sizeY; gy += step) {
      const p = worldToCanvas(0, gy * state.resolution);
      const p2 = worldToCanvas(state.sizeX * state.resolution, gy * state.resolution);
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
  }

  function drawAxes() {
    ctx.save();
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(104,86,58,0.5)';
    ctx.strokeStyle = 'rgba(104,86,58,0.3)';
    ctx.lineWidth = 1;

    const step = Math.max(1, Math.round(2 / state.viewScale));
    for (let gx = 0; gx <= state.sizeX; gx += step) {
      const wx = gx * state.resolution;
      const p = worldToCanvas(wx, 0);
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x, p.y + 5); ctx.stroke();
      if (gx % (step * 2) === 0) ctx.fillText(wx.toFixed(1), p.x - 8, p.y + 14);
    }
    for (let gy = 0; gy <= state.sizeY; gy += step) {
      const wy = gy * state.resolution;
      const p = worldToCanvas(0, wy);
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x - 5, p.y); ctx.stroke();
      if (gy % (step * 2) === 0) ctx.fillText(wy.toFixed(1), p.x - 30, p.y + 4);
    }
    ctx.restore();
  }

  function drawObstacles() {
    state.obstacleRects.forEach((r, i) => {
      const tl = worldToCanvas(r[0] * state.resolution, r[3] * state.resolution);
      const br = worldToCanvas(r[2] * state.resolution, r[1] * state.resolution);
      ctx.strokeStyle = i === state.hoveredObstacle ? '#c44e36' : 'rgba(104,86,58,0.5)';
      ctx.lineWidth = i === state.hoveredObstacle ? 2 : 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      ctx.setLineDash([]);
    });
  }

  function drawPath() {
    const path = state.path;
    if (path.length < 2) return;

    // Draw path line with gradient
    ctx.strokeStyle = '#2b71ba';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    const p0 = worldToCanvas(path[0][0], path[0][1]);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < path.length; i++) {
      const p = worldToCanvas(path[i][0], path[i][1]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();

    // Draw heading arrows at regular intervals
    const arrowInterval = Math.max(1, Math.floor(path.length / 20));
    const arrowLen = 12;
    for (let i = 0; i < path.length; i += arrowInterval) {
      const px = path[i][0], py = path[i][1], theta = path[i][2];
      const p = worldToCanvas(px, py);
      const ex = p.x + arrowLen * Math.cos(theta);
      const ey = p.y - arrowLen * Math.sin(theta);

      // Arrow shaft
      ctx.strokeStyle = '#147a6a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      // Arrowhead
      const headLen = 5;
      const headAngle = 0.5;
      ctx.fillStyle = '#147a6a';
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(
        ex - headLen * Math.cos(theta - headAngle),
        ey + headLen * Math.sin(theta - headAngle)
      );
      ctx.lineTo(
        ex - headLen * Math.cos(theta + headAngle),
        ey + headLen * Math.sin(theta + headAngle)
      );
      ctx.closePath();
      ctx.fill();
    }

    // Draw dots at each pose (smaller, more transparent)
    ctx.fillStyle = 'rgba(43, 113, 186, 0.3)';
    for (let i = 0; i < path.length; i++) {
      const p = worldToCanvas(path[i][0], path[i][1]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawMarker(pose, color, label) {
    const p = worldToCanvas(pose.x, pose.y);
    const r = 8;

    // Circle
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, p.x, p.y);

    // Heading arrow
    const arrowLen = 18;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(
      p.x + arrowLen * Math.cos(pose.theta),
      p.y - arrowLen * Math.sin(pose.theta)
    );
    ctx.stroke();
  }

  // ---- Loupe ----
  function drawLoupe(gridX, gridY) {
    const loupeSize = 9;
    const pixSize = Math.floor(loupeCanvas.width / loupeSize);
    loupeCtx.clearRect(0, 0, loupeCanvas.width, loupeCanvas.height);

    if (!state.costmap) return;

    for (let dy = 0; dy < loupeSize; dy++) {
      for (let dx = 0; dx < loupeSize; dx++) {
        const cx = gridX + dx - Math.floor(loupeSize / 2);
        const cy = gridY + dy - Math.floor(loupeSize / 2);
        let color = '#ccc';
        if (cx >= 0 && cx < state.sizeX && cy >= 0 && cy < state.sizeY) {
          const cost = state.costmap[cy * state.sizeX + cx];
          if (cost === 0) color = '#f0e4e0';
          else if (cost < 128) color = `rgb(${240 - Math.round(100*cost/128)}, ${236 - Math.round(80*cost/128)}, ${228 - Math.round(40*cost/128)})`;
          else if (cost < 254) color = `rgb(${140 + Math.round(80*(cost-128)/126)}, ${156 - Math.round(100*(cost-128)/126)}, ${188 - Math.round(140*(cost-128)/126)})`;
          else color = '#282828';
        }
        loupeCtx.fillStyle = color;
        loupeCtx.fillRect(dx * pixSize, dy * pixSize, pixSize - 1, pixSize - 1);
      }
    }

    // Highlight center cell
    const center = Math.floor(loupeSize / 2) * pixSize;
    loupeCtx.strokeStyle = '#c44e36';
    loupeCtx.lineWidth = 2;
    loupeCtx.strokeRect(center, center, pixSize - 1, pixSize - 1);
  }

  // ---- Mouse interaction ----
  function clientToCanvasPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function hitTestObstacle(cx, cy) {
    const w = canvasToWorld(cx, cy);
    for (let i = 0; i < state.obstacleRects.length; i++) {
      const r = state.obstacleRects[i];
      const wx0 = r[0] * state.resolution;
      const wy0 = r[1] * state.resolution;
      const wx1 = r[2] * state.resolution;
      const wy1 = r[3] * state.resolution;
      if (w.x >= wx0 && w.x <= wx1 && w.y >= wy0 && w.y <= wy1) return i;
    }
    return -1;
  }

  function hitTestMarker(cx, cy, pose) {
    const p = worldToCanvas(pose.x, pose.y);
    return Math.hypot(cx - p.x, cy - p.y) < 14;
  }

  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    const cp = clientToCanvasPoint(e);

    if (hitTestMarker(cp.x, cp.y, state.start)) {
      state.dragging = 'start';
    } else if (hitTestMarker(cp.x, cp.y, state.goal)) {
      state.dragging = 'goal';
    } else {
      const obsIdx = hitTestObstacle(cp.x, cp.y);
      if (obsIdx >= 0) {
        state.dragging = 'obstacle';
        state.dragObstacleIdx = obsIdx;
        state.dragStartCanvas = cp;
        state.dragStartWorld = canvasToWorld(cp.x, cp.y);
      } else {
        state.dragging = 'pan';
        state.dragStartCanvas = cp;
        state.dragStartWorld = { x: state.viewOffsetX, y: state.viewOffsetY };
      }
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const cp = clientToCanvasPoint(e);
    const w = canvasToWorld(cp.x, cp.y);
    state.lastMouseCanvas = cp;
    state.lastMouseWorld = w;

    // Update cursor readouts
    const grid = canvasToGrid(cp.x, cp.y);
    const inBounds = grid.x >= 0 && grid.x < state.sizeX && grid.y >= 0 && grid.y < state.sizeY;
    if (inBounds) {
      document.getElementById('cursor-coord').textContent = `(${w.x.toFixed(2)}, ${w.y.toFixed(2)})`;
      document.getElementById('cursor-world').textContent = `W: (${w.x.toFixed(2)}, ${w.y.toFixed(2)})`;
      document.getElementById('cursor-cell').textContent = `C: (${grid.x}, ${grid.y})`;
    } else {
      document.getElementById('cursor-coord').textContent = '--';
      document.getElementById('cursor-world').textContent = '--';
      document.getElementById('cursor-cell').textContent = '--';
    }

    // Loupe
    if (inBounds) {
      const cost = state.costmap ? state.costmap[grid.y * state.sizeX + grid.x] : 0;
      document.getElementById('loupe-cost').textContent = cost;
      document.getElementById('loupe-world').textContent = `(${w.x.toFixed(2)}, ${w.y.toFixed(2)})`;
      drawLoupe(grid.x, grid.y);
    }

    // Hover detection
    state.hoveredObstacle = inBounds ? hitTestObstacle(cp.x, cp.y) : -1;
    canvas.style.cursor = 'crosshair';
    if (inBounds && (hitTestMarker(cp.x, cp.y, state.start) || hitTestMarker(cp.x, cp.y, state.goal))) {
      canvas.style.cursor = 'grab';
    } else if (state.hoveredObstacle >= 0) {
      canvas.style.cursor = 'move';
    }

    // Dragging
    if (!state.dragging) return;
    canvas.style.cursor = 'grabbing';

    if (state.dragging === 'start') {
      state.start.x = Math.max(0, Math.min(w.x, state.sizeX * state.resolution));
      state.start.y = Math.max(0, Math.min(w.y, state.sizeY * state.resolution));
      updateReadouts();
      runPlanning();
    } else if (state.dragging === 'goal') {
      state.goal.x = Math.max(0, Math.min(w.x, state.sizeX * state.resolution));
      state.goal.y = Math.max(0, Math.min(w.y, state.sizeY * state.resolution));
      updateReadouts();
      runPlanning();
    } else if (state.dragging === 'obstacle') {
      const idx = state.dragObstacleIdx;
      const dx = w.x - state.dragStartWorld.x;
      const dy = w.y - state.dragStartWorld.y;
      const r = state.obstacleRects[idx];
      const w0 = r[0] * state.resolution, h0 = r[1] * state.resolution;
      const w1 = r[2] * state.resolution, h1 = r[3] * state.resolution;
      const nw0 = w0 + dx, nh0 = h0 + dy;
      const nw1 = w1 + dx, nh1 = h1 + dy;
      if (nw0 >= 0 && nw1 <= state.sizeX * state.resolution &&
          nh0 >= 0 && nh1 <= state.sizeY * state.resolution) {
        state.obstacleRects[idx] = [
          Math.round(nw0 / state.resolution),
          Math.round(nh0 / state.resolution),
          Math.round(nw1 / state.resolution),
          Math.round(nh1 / state.resolution),
        ];
        state.dragStartWorld = w;
        // Rebuild costmap server-side for accurate visual feedback
        rebuildCostmapAsync();
        draw();
      }
    } else if (state.dragging === 'pan') {
      state.viewOffsetX = state.dragStartWorld.x + (cp.x - state.dragStartCanvas.x);
      state.viewOffsetY = state.dragStartWorld.y + (cp.y - state.dragStartCanvas.y);
      draw();
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    if (state.dragging === 'obstacle') {
      updateObstacles();
    } else if (state.dragging === 'start' || state.dragging === 'goal') {
      runPlanning();
    }
    state.dragging = null;
  });

  canvas.addEventListener('mouseleave', () => {
    if (state.dragging === 'obstacle') {
      updateObstacles();
    }
    state.dragging = null;
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const cp = clientToCanvasPoint(e);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.max(0.5, Math.min(state.viewScale * factor, 20));

    // Zoom centered on cursor
    state.viewOffsetX = cp.x - (cp.x - state.viewOffsetX) * (newScale / state.viewScale);
    state.viewOffsetY = cp.y - (cp.y - state.viewOffsetY) * (newScale / state.viewScale);
    state.viewScale = newScale;

    document.getElementById('zoom-level').textContent =
      (state.viewScale / (canvas.width / state.sizeX)).toFixed(2) + 'x';
    draw();
  });

  canvas.addEventListener('dblclick', () => {
    resetView();
    draw();
  });

  // ---- Slider / select bindings ----
  const sliderConfig = [
    { id: 'start_theta_deg', display: 'val_start_theta_deg', stateKey: null },
    { id: 'goal_theta_deg', display: 'val_goal_theta_deg', stateKey: null },
    { id: 'minimum_turning_radius', display: 'val_minimum_turning_radius', stateKey: null },
    { id: 'tolerance', display: 'val_tolerance', stateKey: null },
    { id: 'angle_bins', display: 'val_angle_bins', stateKey: null },
    { id: 'max_planning_time', display: 'val_max_planning_time', stateKey: null },
    { id: 'max_iterations', display: 'val_max_iterations', stateKey: null },
    { id: 'reverse_penalty', display: 'val_reverse_penalty', stateKey: null },
    { id: 'cost_penalty', display: 'val_cost_penalty', stateKey: null },
  ];

  let planDebounce = null;
  sliderConfig.forEach(cfg => {
    const el = document.getElementById(cfg.id);
    if (!el) return;
    el.addEventListener('input', () => {
      document.getElementById(cfg.display).textContent = el.value;
      if (cfg.id === 'start_theta_deg' || cfg.id === 'goal_theta_deg') {
        state.start.theta = parseFloat(document.getElementById('start_theta_deg').value) * Math.PI / 180;
        state.goal.theta = parseFloat(document.getElementById('goal_theta_deg').value) * Math.PI / 180;
        document.getElementById('start-heading-readout').textContent = document.getElementById('start_theta_deg').value + ' deg';
        document.getElementById('goal-heading-readout').textContent = document.getElementById('goal_theta_deg').value + ' deg';
        draw();
      }
      // Debounce replanning for parameter changes
      if (planDebounce) clearTimeout(planDebounce);
      planDebounce = setTimeout(() => runPlanning(), 300);
    });
  });

  document.getElementById('motion_model').addEventListener('change', () => runPlanning());
  document.getElementById('smooth_path').addEventListener('change', () => runPlanning());
  document.getElementById('allow_unknown').addEventListener('change', () => runPlanning());

  // Layer toggles
  ['show-costmap', 'show-path', 'show-axes', 'show-grid'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => draw());
  });

  // ---- Readouts ----
  function updateReadouts() {
    document.getElementById('start-coord').textContent =
      `(${state.start.x.toFixed(2)}, ${state.start.y.toFixed(2)})`;
    document.getElementById('goal-coord').textContent =
      `(${state.goal.x.toFixed(2)}, ${state.goal.y.toFixed(2)})`;
    document.getElementById('start-heading-readout').textContent =
      Math.round(state.start.theta * 180 / Math.PI) + ' deg';
    document.getElementById('goal-heading-readout').textContent =
      Math.round(state.goal.theta * 180 / Math.PI) + ' deg';
  }

  // ---- API calls ----
  let _costmapRebuildTimer = null;
  function rebuildCostmapAsync() {
    if (_costmapRebuildTimer) clearTimeout(_costmapRebuildTimer);
    _costmapRebuildTimer = setTimeout(async () => {
      try {
        const resp = await fetch('/api/obstacles', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ obstacle_rects: state.obstacleRects }),
        });
        const data = await resp.json();
        if (data.ok) {
          const cmResp = await fetch('/api/costmap');
          const cmData = await cmResp.json();
          state.costmap = cmData.data;
          draw();
        }
      } catch (err) { /* ignore during drag */ }
    }, 80);
  }

  async function loadCostmap() {
    try {
      const resp = await fetch('/api/costmap');
      const data = await resp.json();
      state.costmap = data.data;
      state.sizeX = data.size_x;
      state.sizeY = data.size_y;
      state.resolution = data.resolution;
      state.originX = data.origin_x;
      state.originY = data.origin_y;
      state.obstacleRects = data.obstacle_rects;
      if (data.start) {
        state.start.x = data.start[0] * state.resolution;
        state.start.y = data.start[1] * state.resolution;
        state.start.theta = data.start[2];
        document.getElementById('start_theta_deg').value = Math.round(state.start.theta * 180 / Math.PI);
        document.getElementById('val_start_theta_deg').textContent = Math.round(state.start.theta * 180 / Math.PI);
      }
      if (data.goal) {
        state.goal.x = data.goal[0] * state.resolution;
        state.goal.y = data.goal[1] * state.resolution;
        state.goal.theta = data.goal[2];
        document.getElementById('goal_theta_deg').value = Math.round(state.goal.theta * 180 / Math.PI);
        document.getElementById('val_goal_theta_deg').textContent = Math.round(state.goal.theta * 180 / Math.PI);
      }
      document.getElementById('map-grid-size').textContent = `${data.size_x} × ${data.size_y}`;
      document.getElementById('map-resolution').textContent = `${data.resolution.toFixed(2)} m`;
      document.getElementById('map-origin').textContent = `(${data.origin_x}, ${data.origin_y})`;

      resetView();
      updateReadouts();
      draw();
      runPlanning();
    } catch (err) {
      showStatus(t('status.loadError', { message: err.message }), 'error');
    }
  }

  let planningInFlight = false;
  async function runPlanning() {
    if (planningInFlight) return;
    planningInFlight = true;

    const statusEl = document.getElementById('status-msg');
    statusEl.className = 'status-msg visible';
    statusEl.textContent = t('status.planning');
    statusEl.style.color = 'var(--muted)';

    try {
      const resp = await fetch('/api/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_x: state.start.x,
          start_y: state.start.y,
          start_theta: state.start.theta,
          goal_x: state.goal.x,
          goal_y: state.goal.y,
          goal_theta: state.goal.theta,
          motion_model: document.getElementById('motion_model').value,
          tolerance: parseFloat(document.getElementById('tolerance').value),
          angle_bins: parseInt(document.getElementById('angle_bins').value),
          max_planning_time: parseFloat(document.getElementById('max_planning_time').value),
          allow_unknown: document.getElementById('allow_unknown').checked,
          smooth_path: document.getElementById('smooth_path').checked,
          minimum_turning_radius: parseFloat(document.getElementById('minimum_turning_radius').value),
          reverse_penalty: parseFloat(document.getElementById('reverse_penalty').value),
          cost_penalty: parseFloat(document.getElementById('cost_penalty').value),
          max_iterations: parseInt(document.getElementById('max_iterations').value),
        }),
      });
      const data = await resp.json();

      state.path = data.path || [];

      document.getElementById('run-status').textContent = data.ok ? '✓ Found' : '✗ Failed';
      document.getElementById('run-status').style.color = data.ok ? 'var(--accent)' : 'var(--danger)';
      document.getElementById('run-time').textContent = data.elapsed_ms ? data.elapsed_ms.toFixed(1) + ' ms' : '--';
      document.getElementById('run-path-length').textContent = data.path_length || '0';

      // Compute path arc length
      let arcLen = 0;
      for (let i = 1; i < state.path.length; i++) {
        arcLen += Math.hypot(
          state.path[i][0] - state.path[i-1][0],
          state.path[i][1] - state.path[i-1][1]
        );
      }
      document.getElementById('run-path-distance').textContent = arcLen.toFixed(2) + ' m';

      document.getElementById('run-message').textContent = data.message || '';

      if (data.ok) {
        showStatus(t('status.pathFound', { poses: data.path_length, time: data.elapsed_ms.toFixed(1) }), 'ok');
      } else {
        showStatus(t('status.noPath', { message: data.message }), 'error');
      }

      draw();
    } catch (err) {
      showStatus(t('status.planError', { message: err.message }), 'error');
      state.path = [];
      draw();
    } finally {
      planningInFlight = false;
    }
  }

  async function updateObstacles() {
    try {
      const resp = await fetch('/api/obstacles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ obstacle_rects: state.obstacleRects }),
      });
      const data = await resp.json();
      if (data.ok) {
        const cmResp = await fetch('/api/costmap');
        const cmData = await cmResp.json();
        state.costmap = cmData.data;
        draw();
        runPlanning();
      }
    } catch (err) {
      showStatus(t('status.obstacleError', { message: err.message }), 'error');
    }
  }

  function showStatus(msg, type) {
    const el = document.getElementById('status-msg');
    el.className = `status-msg visible ${type}`;
    el.textContent = msg;
  }

  // ---- Button handlers ----
  document.getElementById('run-btn').addEventListener('click', () => runPlanning());
  document.getElementById('clear-path-btn').addEventListener('click', () => {
    state.path = [];
    document.getElementById('run-status').textContent = '--';
    document.getElementById('run-time').textContent = '--';
    document.getElementById('run-path-length').textContent = '--';
    document.getElementById('run-path-distance').textContent = '--';
    document.getElementById('run-message').textContent = '';
    document.getElementById('status-msg').className = 'status-msg';
    draw();
  });
  document.getElementById('reset-view-btn').addEventListener('click', () => {
    resetView();
    draw();
  });

  // ---- Init ----
  loadCostmap();
});
