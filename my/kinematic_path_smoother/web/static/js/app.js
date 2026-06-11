const scene = structuredClone(window.DEFAULT_SCENE);
let solution = null;
let drag = null;
let solveTimer = null;
let activePointerId = null;

const $ = id => document.getElementById(id);
const mapCanvas = $('map');
const mapCtx = mapCanvas.getContext('2d');
const curvatureCanvas = $('curvature-chart');
const spacingCanvas = $('spacing-chart');
const curvatureCtx = curvatureCanvas.getContext('2d');
const spacingCtx = spacingCanvas.getContext('2d');

const view = {
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  initialized: false,
};

const controls = {
  startX: $('start-x'),
  startY: $('start-y'),
  goalX: $('goal-x'),
  goalY: $('goal-y'),
  startYaw: $('start-yaw'),
  goalYaw: $('goal-yaw'),
  referenceStride: $('reference-stride'),
  modelWeight: $('model-weight'),
  referenceWeight: $('reference-weight'),
  curvatureWeight: $('curvature-weight'),
  curvatureRateWeight: $('curvature-rate-weight'),
  lengthWeight: $('length-weight'),
  maxCurvature: $('max-curvature'),
  keepStart: $('keep-start'),
  keepGoal: $('keep-goal'),
  goalLonTol: $('goal-lon-tol'),
  goalLatTol: $('goal-lat-tol'),
  footprintRadius: $('footprint-radius'),
  upsample: $('upsample'),
  maxIterations: $('max-iterations'),
};

const layers = {
  costmap: $('layer-costmap'),
  grid: $('layer-grid'),
  raw: $('layer-raw'),
  reference: $('layer-reference'),
  smooth: $('layer-smooth'),
  knots: $('layer-knots'),
  footprint: $('layer-footprint'),
};

const valueLabels = [
  ['start-yaw', 'start-yaw-value', ' deg', 0],
  ['goal-yaw', 'goal-yaw-value', ' deg', 0],
  ['reference-stride', 'reference-stride-value', '', 0],
  ['model-weight', 'model-weight-value', '', 1],
  ['reference-weight', 'reference-weight-value', '', 1],
  ['curvature-weight', 'curvature-weight-value', '', 1],
  ['curvature-rate-weight', 'curvature-rate-weight-value', '', 1],
  ['length-weight', 'length-weight-value', '', 2],
  ['max-curvature', 'max-curvature-value', ' 1/m', 1],
  ['goal-lon-tol', 'goal-lon-tol-value', ' m', 2],
  ['goal-lat-tol', 'goal-lat-tol-value', ' m', 2],
  ['footprint-radius', 'footprint-radius-value', ' m', 2],
  ['upsample', 'upsample-value', 'x', 0],
  ['max-iterations', 'max-iterations-value', '', 0],
];

function meters(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} m` : '--';
}

function signed(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--';
}

function pathLength(path) {
  if (!path || path.length < 2) return 0;
  let sum = 0;
  for (let i = 1; i < path.length; i++) {
    sum += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
  }
  return sum;
}

function profile(path) {
  const curvature = [];
  const spacing = [];
  let arc = 0;
  if (!path || path.length < 2) return { curvature, spacing };
  for (let i = 1; i < path.length; i++) {
    const ds = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    spacing.push([arc, ds]);
    arc += ds;
  }
  for (let i = 1; i < path.length - 1; i++) {
    const a = path[i - 1];
    const b = path[i];
    const c = path[i + 1];
    const h0 = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const h1 = Math.atan2(c[1] - b[1], c[0] - b[0]);
    const ds = Math.max(1e-6, Math.hypot(c[0] - a[0], c[1] - a[1]));
    const dtheta = Math.atan2(Math.sin(h1 - h0), Math.cos(h1 - h0));
    curvature.push([i, dtheta / ds]);
  }
  return { curvature, spacing };
}

function updateLabels() {
  for (const [inputId, labelId, suffix, digits] of valueLabels) {
    const value = Number($(inputId).value);
    $(labelId).textContent = `${value.toFixed(digits)}${suffix}`;
  }
}

function syncInputsFromScene() {
  controls.startX.value = scene.start.x.toFixed(2);
  controls.startY.value = scene.start.y.toFixed(2);
  controls.goalX.value = scene.goal.x.toFixed(2);
  controls.goalY.value = scene.goal.y.toFixed(2);
  controls.startYaw.value = scene.start.yaw_deg;
  controls.goalYaw.value = scene.goal.yaw_deg;
  updateLabels();
}

function syncInputsFromEndpoints() {
  controls.startX.value = scene.start.x.toFixed(2);
  controls.startY.value = scene.start.y.toFixed(2);
  controls.goalX.value = scene.goal.x.toFixed(2);
  controls.goalY.value = scene.goal.y.toFixed(2);
  updateLabels();
}

function updateSceneFromInputs() {
  const bounds = worldBounds();
  scene.start.x = clamp(Number(controls.startX.value), 0.05, bounds.w - 0.05);
  scene.start.y = clamp(Number(controls.startY.value), 0.05, bounds.h - 0.05);
  scene.goal.x = clamp(Number(controls.goalX.value), 0.05, bounds.w - 0.05);
  scene.goal.y = clamp(Number(controls.goalY.value), 0.05, bounds.h - 0.05);
  scene.start.yaw_deg = Number(controls.startYaw.value);
  scene.goal.yaw_deg = Number(controls.goalYaw.value);
  updateLabels();
}

function params() {
  return {
    model_weight: Number(controls.modelWeight.value),
    reference_weight: Number(controls.referenceWeight.value),
    curvature_weight: Number(controls.curvatureWeight.value),
    curvature_rate_weight: Number(controls.curvatureRateWeight.value),
    length_weight: Number(controls.lengthWeight.value),
    max_curvature: Number(controls.maxCurvature.value),
    keep_start_orientation: controls.keepStart.checked,
    keep_goal_orientation: controls.keepGoal.checked,
    goal_longitudinal_tolerance: Number(controls.goalLonTol.value),
    goal_lateral_tolerance: Number(controls.goalLatTol.value),
    footprint_radius: Number(controls.footprintRadius.value),
    path_upsampling_factor: Number(controls.upsample.value),
    reference_stride: Number(controls.referenceStride.value),
    max_iterations: Number(controls.maxIterations.value),
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function worldBounds() {
  return {
    w: scene.width * scene.resolution,
    h: scene.height * scene.resolution,
  };
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function resetView() {
  resizeCanvas(mapCanvas);
  const bounds = worldBounds();
  const margin = 54 * (window.devicePixelRatio || 1);
  view.scale = Math.min((mapCanvas.width - 2 * margin) / bounds.w, (mapCanvas.height - 2 * margin) / bounds.h);
  view.offsetX = (mapCanvas.width - bounds.w * view.scale) * 0.5;
  view.offsetY = (mapCanvas.height + bounds.h * view.scale) * 0.5;
  view.initialized = true;
}

function worldToCanvas(x, y) {
  if (!view.initialized) resetView();
  return { x: view.offsetX + x * view.scale, y: view.offsetY - y * view.scale };
}

function canvasToWorld(x, y) {
  if (!view.initialized) resetView();
  return { x: (x - view.offsetX) / view.scale, y: (view.offsetY - y) / view.scale };
}

function pointerPosition(event) {
  const rect = mapCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * mapCanvas.width / rect.width,
    y: (event.clientY - rect.top) * mapCanvas.height / rect.height,
  };
}

function canvasPointForPose(pose) {
  return worldToCanvas(pose.x, pose.y);
}

function drawCostmap() {
  if (!solution?.costmap || !layers.costmap.checked) return;
  const cell = scene.resolution * view.scale;
  mapCtx.save();
  for (let y = 0; y < scene.height; y++) {
    for (let x = 0; x < scene.width; x++) {
      const cost = solution.costmap[y * scene.width + x];
      if (cost <= 0) continue;
      const p = worldToCanvas(x * scene.resolution, (y + 1) * scene.resolution);
      mapCtx.fillStyle = cost >= 254 ? '#111827' : '#94a3b8';
      mapCtx.fillRect(p.x, p.y, Math.ceil(cell), Math.ceil(cell));
    }
  }
  mapCtx.restore();
}

function drawGrid() {
  if (!layers.grid.checked) return;
  const bounds = worldBounds();
  mapCtx.save();
  mapCtx.strokeStyle = '#d8e0ea';
  mapCtx.lineWidth = 1;
  const step = view.scale * scene.resolution < 7 ? 5 : 1;
  for (let x = 0; x <= scene.width; x += step) {
    const p0 = worldToCanvas(x * scene.resolution, 0);
    const p1 = worldToCanvas(x * scene.resolution, bounds.h);
    mapCtx.beginPath();
    mapCtx.moveTo(p0.x, p0.y);
    mapCtx.lineTo(p1.x, p1.y);
    mapCtx.stroke();
  }
  for (let y = 0; y <= scene.height; y += step) {
    const p0 = worldToCanvas(0, y * scene.resolution);
    const p1 = worldToCanvas(bounds.w, y * scene.resolution);
    mapCtx.beginPath();
    mapCtx.moveTo(p0.x, p0.y);
    mapCtx.lineTo(p1.x, p1.y);
    mapCtx.stroke();
  }
  mapCtx.restore();
}

function drawObstacles() {
  mapCtx.save();
  mapCtx.strokeStyle = '#f97316';
  mapCtx.lineWidth = 2;
  for (const obstacle of scene.obstacles) {
    const p = worldToCanvas(obstacle.x * scene.resolution, (obstacle.y + obstacle.h) * scene.resolution);
    const w = obstacle.w * scene.resolution * view.scale;
    const h = obstacle.h * scene.resolution * view.scale;
    if (!solution?.costmap || !layers.costmap.checked) {
      mapCtx.fillStyle = '#111827';
      mapCtx.fillRect(p.x, p.y, w, h);
    }
    mapCtx.strokeRect(p.x, p.y, w, h);
  }
  mapCtx.restore();
}

function drawPath(path, color, width, dash = []) {
  if (!path || path.length < 2) return;
  mapCtx.save();
  mapCtx.strokeStyle = color;
  mapCtx.lineWidth = width;
  mapCtx.lineJoin = 'round';
  mapCtx.lineCap = 'round';
  mapCtx.setLineDash(dash);
  mapCtx.beginPath();
  path.forEach((point, index) => {
    const p = worldToCanvas(point[0], point[1]);
    if (index === 0) mapCtx.moveTo(p.x, p.y);
    else mapCtx.lineTo(p.x, p.y);
  });
  mapCtx.stroke();
  mapCtx.restore();
}

function drawKnots(path) {
  if (!path || !layers.knots.checked) return;
  mapCtx.save();
  mapCtx.fillStyle = '#6d5bd0';
  for (const point of path) {
    const p = worldToCanvas(point[0], point[1]);
    mapCtx.beginPath();
    mapCtx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
    mapCtx.fill();
  }
  mapCtx.restore();
}

function drawFootprints(path) {
  if (!path || !layers.footprint.checked) return;
  const radius = Number(controls.footprintRadius.value) * view.scale;
  if (radius <= 0) return;
  mapCtx.save();
  mapCtx.strokeStyle = 'rgba(37, 99, 235, 0.28)';
  mapCtx.lineWidth = 1;
  const stride = Math.max(1, Math.floor(path.length / 80));
  for (let i = 0; i < path.length; i += stride) {
    const p = worldToCanvas(path[i][0], path[i][1]);
    mapCtx.beginPath();
    mapCtx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    mapCtx.stroke();
  }
  mapCtx.restore();
}

function drawMarker(pose, color, label) {
  const p = worldToCanvas(pose.x, pose.y);
  const yaw = pose.yaw_deg * Math.PI / 180;
  mapCtx.save();
  mapCtx.translate(p.x, p.y);
  mapCtx.fillStyle = color;
  mapCtx.strokeStyle = '#ffffff';
  mapCtx.lineWidth = 3;
  mapCtx.beginPath();
  mapCtx.arc(0, 0, 9, 0, Math.PI * 2);
  mapCtx.fill();
  mapCtx.stroke();
  mapCtx.rotate(-yaw);
  mapCtx.strokeStyle = color;
  mapCtx.lineWidth = 4;
  mapCtx.beginPath();
  mapCtx.moveTo(0, 0);
  mapCtx.lineTo(24, 0);
  mapCtx.stroke();
  mapCtx.restore();
  mapCtx.fillStyle = '#162033';
  mapCtx.font = '800 12px system-ui';
  mapCtx.fillText(label, p.x + 12, p.y - 12);
}

function drawMap() {
  resizeCanvas(mapCanvas);
  if (!view.initialized) resetView();
  mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
  const bounds = worldBounds();
  const a = worldToCanvas(0, 0);
  const b = worldToCanvas(bounds.w, bounds.h);
  mapCtx.fillStyle = '#f8fafc';
  mapCtx.fillRect(a.x, b.y, b.x - a.x, a.y - b.y);
  drawGrid();
  drawCostmap();
  drawObstacles();
  if (solution) {
    if (layers.raw.checked) drawPath(solution.raw_path, '#64748b', 2, [5, 6]);
    if (layers.reference.checked) drawPath(solution.reference_path, '#d97706', 3, [10, 6]);
    if (layers.smooth.checked) drawPath(solution.smoothed_path, solution.success ? '#08766f' : '#e11d48', 4);
    drawFootprints(solution.smoothed_path);
    drawKnots(solution.optimized_path);
  }
  drawMarker(scene.start, '#2563eb', 'S');
  drawMarker(scene.goal, '#e11d48', 'G');
}

function drawChart(canvas, ctx, points, color, zeroLine = true) {
  resizeCanvas(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#d7dee8';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (canvas.height * i) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  if (!points || points.length < 2) return;
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  if (zeroLine) {
    minY = Math.min(minY, 0);
    maxY = Math.max(maxY, 0);
  }
  if (Math.abs(maxY - minY) < 1e-9) {
    minY -= 1;
    maxY += 1;
  }
  const pad = 18 * (window.devicePixelRatio || 1);
  const sx = x => pad + ((x - minX) / Math.max(1e-9, maxX - minX)) * (canvas.width - 2 * pad);
  const sy = y => canvas.height - pad - ((y - minY) / (maxY - minY)) * (canvas.height - 2 * pad);
  if (zeroLine && minY <= 0 && maxY >= 0) {
    ctx.strokeStyle = '#b7c2d0';
    ctx.beginPath();
    ctx.moveTo(pad, sy(0));
    ctx.lineTo(canvas.width - pad, sy(0));
    ctx.stroke();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = sx(point[0]);
    const y = sy(point[1]);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function updateAnalysis() {
  const smooth = solution?.smoothed_path || [];
  const rawLength = pathLength(solution?.raw_path);
  const refLength = pathLength(solution?.reference_path);
  const smoothLength = pathLength(smooth);
  const prof = profile(smooth);
  const peak = prof.curvature.reduce((max, item) => Math.max(max, Math.abs(item[1])), 0);
  const meanSpacing = prof.spacing.length
    ? prof.spacing.reduce((sum, item) => sum + item[1], 0) / prof.spacing.length
    : 0;

  $('metric-raw-length').textContent = meters(rawLength);
  $('metric-ref-length').textContent = meters(refLength);
  $('metric-smooth-length').textContent = meters(smoothLength);
  $('metric-peak-curvature').textContent = peak ? `${peak.toFixed(3)} 1/m` : '--';
  $('curvature-summary').textContent = peak ? `peak ${peak.toFixed(3)} 1/m` : '--';
  $('spacing-summary').textContent = meanSpacing ? `mean ${meanSpacing.toFixed(3)} m` : '--';
  $('diag-backend').textContent = solution?.backend || '--';
  $('diag-time').textContent = solution ? `${solution.elapsed_ms} ms` : '--';
  $('diag-knots').textContent = solution?.stats ? String(solution.stats.optimized_knot_count) : '--';
  $('diag-failure').textContent = solution?.failure ? `${solution.failure.reason}: ${solution.failure.message}` : '--';
  $('diagnostic-status').textContent = solution ? (solution.success ? 'accepted' : 'rejected') : '--';

  drawChart(curvatureCanvas, curvatureCtx, prof.curvature, '#08766f', true);
  drawChart(spacingCanvas, spacingCtx, prof.spacing, '#6d5bd0', false);
}

function setStatus(text, detail = '') {
  $('status-line').textContent = text;
  $('stats-line').textContent = detail;
}

async function solve() {
  updateSceneFromInputs();
  setStatus('Running', '');
  drawMap();
  const payload = {
    width: scene.width,
    height: scene.height,
    resolution: scene.resolution,
    origin_x: scene.origin_x,
    origin_y: scene.origin_y,
    start: scene.start,
    goal: scene.goal,
    obstacles: scene.obstacles,
    params: params(),
  };
  try {
    const response = await fetch('/api/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || data.error || 'request failed');
    solution = data;
    $('backend-badge').textContent = data.backend === 'cpp' ? 'C++' : 'Python';
    setStatus(
      data.success ? 'Accepted' : 'Rejected',
      `${data.elapsed_ms} ms / A* ${data.stats.raw_points} / Ref ${data.stats.reference_points} / Out ${data.smoothed_path.length}`,
    );
  } catch (error) {
    solution = null;
    setStatus('Failed', error.message);
  }
  updateAnalysis();
  drawMap();
}

function scheduleSolve() {
  clearTimeout(solveTimer);
  solveTimer = setTimeout(solve, 180);
}

function obstacleCanvasRect(obstacle) {
  const topLeft = worldToCanvas(
    obstacle.x * scene.resolution,
    (obstacle.y + obstacle.h) * scene.resolution,
  );
  return {
    x: topLeft.x,
    y: topLeft.y,
    w: obstacle.w * scene.resolution * view.scale,
    h: obstacle.h * scene.resolution * view.scale,
  };
}

function hitTest(pos) {
  const markerRadius = 22 * (window.devicePixelRatio || 1);
  const start = canvasPointForPose(scene.start);
  const goal = canvasPointForPose(scene.goal);
  if (Math.hypot(pos.x - start.x, pos.y - start.y) <= markerRadius) {
    return { type: 'start', grabbed: { x: scene.start.x, y: scene.start.y } };
  }
  if (Math.hypot(pos.x - goal.x, pos.y - goal.y) <= markerRadius) {
    return { type: 'goal', grabbed: { x: scene.goal.x, y: scene.goal.y } };
  }

  for (let i = 0; i < scene.obstacles.length; i++) {
    const obstacle = scene.obstacles[i];
    const rect = obstacleCanvasRect(obstacle);
    if (pos.x >= rect.x && pos.x <= rect.x + rect.w && pos.y >= rect.y && pos.y <= rect.y + rect.h) {
      return {
        type: 'obstacle',
        index: i,
        grabbed: { x: obstacle.x, y: obstacle.y },
      };
    }
  }
  return { type: 'pan' };
}

mapCanvas.addEventListener('pointerdown', event => {
  if (event.button !== undefined && event.button !== 0) return;
  event.preventDefault();
  const pos = pointerPosition(event);
  const world = canvasToWorld(pos.x, pos.y);
  const hit = hitTest(pos);
  drag = {
    ...hit,
    pointerStart: pos,
    worldStart: world,
    viewStart: { x: view.offsetX, y: view.offsetY },
    moved: false,
  };
  activePointerId = event.pointerId;
  clearTimeout(solveTimer);
  mapCanvas.classList.add('is-dragging');
  if (mapCanvas.setPointerCapture) {
    mapCanvas.setPointerCapture(event.pointerId);
  }
});

mapCanvas.addEventListener('pointermove', event => {
  const pos = pointerPosition(event);
  const world = canvasToWorld(pos.x, pos.y);
  $('cursor-readout').textContent = `${world.x.toFixed(2)}, ${world.y.toFixed(2)}`;
  if (!drag) {
    mapCanvas.classList.toggle('is-hover-handle', hitTest(pos).type !== 'pan');
    return;
  }
  if (activePointerId !== null && event.pointerId !== activePointerId) return;
  event.preventDefault();

  const dx = pos.x - drag.pointerStart.x;
  const dy = pos.y - drag.pointerStart.y;
  drag.moved = drag.moved || Math.hypot(dx, dy) > 2 * (window.devicePixelRatio || 1);
  const bounds = worldBounds();
  if (drag.type === 'start' || drag.type === 'goal') {
    const pose = drag.type === 'start' ? scene.start : scene.goal;
    pose.x = clamp(drag.grabbed.x + dx / view.scale, 0.05, bounds.w - 0.05);
    pose.y = clamp(drag.grabbed.y - dy / view.scale, 0.05, bounds.h - 0.05);
    syncInputsFromEndpoints();
  } else if (drag.type === 'obstacle') {
    const obstacle = scene.obstacles[drag.index];
    obstacle.x = clamp(
      Math.round(drag.grabbed.x + dx / (view.scale * scene.resolution)),
      0,
      scene.width - obstacle.w,
    );
    obstacle.y = clamp(
      Math.round(drag.grabbed.y - dy / (view.scale * scene.resolution)),
      0,
      scene.height - obstacle.h,
    );
  } else {
    view.offsetX = drag.viewStart.x + dx;
    view.offsetY = drag.viewStart.y + dy;
  }
  drawMap();
});

function finishPointerDrag(event) {
  if (!drag) return;
  if (event && activePointerId !== null && event.pointerId !== activePointerId) return;
  event?.preventDefault();
  const shouldSolve = drag.moved && drag.type !== 'pan';
  drag = null;
  activePointerId = null;
  mapCanvas.classList.remove('is-dragging');
  if (event && mapCanvas.hasPointerCapture?.(event.pointerId)) {
    mapCanvas.releasePointerCapture(event.pointerId);
  }
  if (shouldSolve) solve();
}

mapCanvas.addEventListener('pointerup', finishPointerDrag);
mapCanvas.addEventListener('pointercancel', finishPointerDrag);
mapCanvas.addEventListener('lostpointercapture', () => {
  if (!drag) return;
  const shouldSolve = drag.moved && drag.type !== 'pan';
  drag = null;
  activePointerId = null;
  mapCanvas.classList.remove('is-dragging');
  if (shouldSolve) solve();
});

mapCanvas.addEventListener('wheel', event => {
  event.preventDefault();
  const pos = pointerPosition(event);
  const before = canvasToWorld(pos.x, pos.y);
  const factor = event.deltaY < 0 ? 1.12 : 0.89;
  view.scale = clamp(view.scale * factor, 24, 260);
  view.offsetX = pos.x - before.x * view.scale;
  view.offsetY = pos.y + before.y * view.scale;
  drawMap();
}, { passive: false });

mapCanvas.addEventListener('dblclick', () => {
  resetView();
  drawMap();
});

for (const tab of document.querySelectorAll('.tab')) {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(item => item.classList.toggle('is-active', item === tab));
    document.querySelectorAll('.panel').forEach(panel => panel.classList.toggle('is-active', panel.id === tab.dataset.panel));
  });
}

for (const control of Object.values(controls)) {
  control.addEventListener('input', () => {
    updateSceneFromInputs();
    updateLabels();
    drawMap();
    scheduleSolve();
  });
}

for (const layer of Object.values(layers)) {
  layer.addEventListener('change', () => {
    drawMap();
    updateAnalysis();
  });
}

$('solve-btn').addEventListener('click', solve);
$('reset-btn').addEventListener('click', () => {
  Object.assign(scene, structuredClone(window.DEFAULT_SCENE));
  solution = null;
  view.initialized = false;
  syncInputsFromScene();
  solve();
});

window.addEventListener('resize', () => {
  view.initialized = false;
  drawMap();
  updateAnalysis();
});

syncInputsFromScene();
resetView();
drawMap();
solve();
