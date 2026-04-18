// A* + Constrained Smoother — interactive map frontend
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('map-canvas');
  const ctx = canvas.getContext('2d');
  const runBtn = document.getElementById('run-btn');
  const clearBtn = document.getElementById('clear-btn');
  const statusMsg = document.getElementById('status-msg');

  // ---- State -----------------------------------------------------------------
  const state = {
    costmap: null,      // {size_x, size_y, resolution, origin_x, origin_y, data}
    start: null,        // {x, y}  world coords
    goal: null,         // {x, y}  world coords
    clickPhase: 'start', // 'start' | 'goal'
    paths: null,        // result from /api/plan
    // view transform (pan + zoom)
    viewScale: 1,
    viewOffsetX: 0,
    viewOffsetY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragOffsetX: 0,
    dragOffsetY: 0,
  };

  // ---- Slider binding -------------------------------------------------------
  const sliders = [
    'smooth_weight', 'costmap_weight', 'distance_weight',
    'curvature_weight', 'max_curvature',
    'max_iterations', 'path_downsampling_factor', 'path_upsampling_factor',
  ];
  sliders.forEach(id => {
    const el = document.getElementById(id);
    const lbl = document.getElementById('val_' + id);
    if (el && lbl) {
      el.addEventListener('input', () => { lbl.textContent = el.value; });
    }
  });

  // ---- Helpers ---------------------------------------------------------------
  function setStatus(msg, cls) {
    statusMsg.textContent = msg;
    statusMsg.className = 'status-msg ' + (cls || '');
  }

  function worldToCanvas(wx, wy) {
    if (!state.costmap) return {x: 0, y: 0};
    const cm = state.costmap;
    const cellX = (wx - cm.origin_x) / cm.resolution;
    const cellY = (wy - cm.origin_y) / cm.resolution;
    // Map cell to pixel (y-axis flipped: world-up → canvas-down)
    const px = cellX * (canvas.width / cm.size_x);
    const py = (cm.size_y - cellY) * (canvas.height / cm.size_y);
    // Apply view transform
    return {
      x: px * state.viewScale + state.viewOffsetX,
      y: py * state.viewScale + state.viewOffsetY,
    };
  }

  function canvasToWorld(cx, cy) {
    if (!state.costmap) return {x: 0, y: 0};
    const cm = state.costmap;
    // Invert view transform
    const px = (cx - state.viewOffsetX) / state.viewScale;
    const py = (cy - state.viewOffsetY) / state.viewScale;
    const cellX = px / (canvas.width / cm.size_x);
    const cellY = cm.size_y - py / (canvas.height / cm.size_y);
    return {
      x: cm.origin_x + cellX * cm.resolution,
      y: cm.origin_y + cellY * cm.resolution,
    };
  }

  // ---- Drawing ---------------------------------------------------------------
  function draw() {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.restore();

    if (!state.costmap) return;

    // Draw costmap
    drawCostmap();

    // Draw paths
    if (state.paths) {
      drawPath(state.paths.astar_x, state.paths.astar_y, 'rgba(33,150,243,0.35)', 1.5);
      drawPath(state.paths.ref_x, state.paths.ref_y, '#ff9800', 2, true);
      drawPath(state.paths.opt_x, state.paths.opt_y, '#e91e63', 2.5, true);
    }

    // Draw start / goal markers
    if (state.start) drawMarker(state.start.x, state.start.y, '#4caf50', 'S');
    if (state.goal) drawMarker(state.goal.x, state.goal.y, '#f44336', 'G');
  }

  let costmapImageData = null;

  function buildCostmapImage() {
    if (!state.costmap) return;
    const cm = state.costmap;
    const imgData = ctx.createImageData(cm.size_x, cm.size_y);
    for (let my = 0; my < cm.size_y; my++) {
      for (let mx = 0; mx < cm.size_x; mx++) {
        const cost = cm.data[my * cm.size_x + mx];
        // Flip y for canvas
        const canvasRow = cm.size_y - 1 - my;
        const idx = (canvasRow * cm.size_x + mx) * 4;

        if (cost >= 254) {
          // Lethal
          imgData.data[idx] = 40;
          imgData.data[idx + 1] = 40;
          imgData.data[idx + 2] = 50;
          imgData.data[idx + 3] = 255;
        } else if (cost > 0) {
          // Inflated
          const t = cost / 253;
          imgData.data[idx] = Math.floor(30 + 60 * t);
          imgData.data[idx + 1] = Math.floor(30 + 20 * t);
          imgData.data[idx + 2] = Math.floor(50 + 40 * t);
          imgData.data[idx + 3] = 255;
        } else {
          // Free
          imgData.data[idx] = 18;
          imgData.data[idx + 1] = 22;
          imgData.data[idx + 2] = 30;
          imgData.data[idx + 3] = 255;
        }
      }
    }
    costmapImageData = imgData;
  }

  function drawCostmap() {
    if (!costmapImageData) return;
    const cm = state.costmap;
    // Create temporary canvas for costmap image
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = cm.size_x;
    tmpCanvas.height = cm.size_y;
    tmpCanvas.getContext('2d').putImageData(costmapImageData, 0, 0);
    // Draw scaled
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(
      state.viewScale * (canvas.width / cm.size_x), 0,
      0, state.viewScale * (canvas.height / cm.size_y),
      state.viewOffsetX, state.viewOffsetY
    );
    ctx.drawImage(tmpCanvas, 0, 0);
    ctx.restore();
  }

  function drawPath(xs, ys, color, width, dots) {
    if (!xs || xs.length < 2) return;
    ctx.beginPath();
    const p0 = worldToCanvas(xs[0], ys[0]);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < xs.length; i++) {
      const p = worldToCanvas(xs[i], ys[i]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.stroke();

    if (dots) {
      ctx.fillStyle = color;
      for (let i = 0; i < xs.length; i++) {
        const p = worldToCanvas(xs[i], ys[i]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, width, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawMarker(wx, wy, color, label) {
    const p = worldToCanvas(wx, wy);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, p.x, p.y);
  }

  // ---- Interaction -----------------------------------------------------------
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const world = canvasToWorld(cx, cy);

    if (state.clickPhase === 'start') {
      state.start = world;
      state.goal = null;
      state.paths = null;
      state.clickPhase = 'goal';
      setStatus('Start set. Click to set Goal.', '');
      runBtn.disabled = true;
    } else {
      state.goal = world;
      state.clickPhase = 'start';
      setStatus('Goal set. Click "Run Planning" or click map again.', '');
      runBtn.disabled = false;
      // Auto-run planning
      runPlanning();
    }
    draw();
  });

  // Pan
  canvas.addEventListener('mousedown', (e) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      state.dragging = true;
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.dragOffsetX = state.viewOffsetX;
      state.dragOffsetY = state.viewOffsetY;
      e.preventDefault();
    }
  });
  canvas.addEventListener('mousemove', (e) => {
    if (state.dragging) {
      state.viewOffsetX = state.dragOffsetX + (e.clientX - state.dragStartX);
      state.viewOffsetY = state.dragOffsetY + (e.clientY - state.dragStartY);
      draw();
    }
  });
  window.addEventListener('mouseup', () => { state.dragging = false; });

  // Zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const newScale = state.viewScale * factor;
    // Zoom toward mouse position
    state.viewOffsetX = mx - (mx - state.viewOffsetX) * (newScale / state.viewScale);
    state.viewOffsetY = my - (my - state.viewOffsetY) * (newScale / state.viewScale);
    state.viewScale = newScale;
    draw();
  });

  // ---- Run Planning ----------------------------------------------------------
  runBtn.addEventListener('click', () => runPlanning());
  clearBtn.addEventListener('click', () => {
    state.paths = null;
    state.start = null;
    state.goal = null;
    state.clickPhase = 'start';
    runBtn.disabled = true;
    setStatus('Cleared. Click map to set Start.', '');
    clearInfo();
    draw();
  });

  function getParams() {
    const vals = {};
    sliders.forEach(id => {
      const el = document.getElementById(id);
      if (el) vals[id] = parseFloat(el.value);
    });
    return vals;
  }

  async function runPlanning() {
    if (!state.start || !state.goal) return;
    setStatus('Planning…', '');
    runBtn.disabled = true;

    const body = {
      start_x: state.start.x,
      start_y: state.start.y,
      goal_x: state.goal.x,
      goal_y: state.goal.y,
      ...getParams(),
    };

    try {
      const res = await fetch('/api/plan', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.success) {
        state.paths = data;
        setStatus(
          `Done — A*: ${data.astar_time_ms}ms, Smooth: ${data.smooth_time_ms}ms` +
          (data.smooth_success ? '' : ' (smooth failed, showing ref)'),
          data.smooth_success ? 'ok' : 'error'
        );
        updateInfo(data);
      } else {
        setStatus(data.message || 'Planning failed.', 'error');
      }
    } catch (err) {
      setStatus('Network error: ' + err.message, 'error');
    }
    runBtn.disabled = false;
    draw();
  }

  function updateInfo(d) {
    document.getElementById('info-astar-time').textContent = d.astar_time_ms;
    document.getElementById('info-smooth-time').textContent = d.smooth_time_ms;
    document.getElementById('info-astar-pts').textContent = d.num_astar_pts;
    document.getElementById('info-ref-pts').textContent = d.num_ref_pts;
    document.getElementById('info-opt-pts').textContent = d.num_opt_pts;
  }
  function clearInfo() {
    ['info-astar-time','info-smooth-time','info-astar-pts','info-ref-pts','info-opt-pts']
      .forEach(id => { document.getElementById(id).textContent = '--'; });
  }

  // ---- Bootstrap -------------------------------------------------------------
  async function loadCostmap() {
    setStatus('Loading costmap…', '');
    try {
      const res = await fetch('/api/costmap');
      state.costmap = await res.json();
      buildCostmapImage();
      setStatus('Click on the map to set a Start point.', '');
      draw();
    } catch (err) {
      setStatus('Failed to load costmap: ' + err.message, 'error');
    }
  }

  loadCostmap();
});
