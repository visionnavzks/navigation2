from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from zhengli.kinematic_smoother.kinematic_smoother import KinematicSmoother


DEFAULTS = {
    "forward_length": 10.0,
    "reverse_end_x": 0.0,
    "reverse_end_y": 5.0,
    "forward_points": 20,
    "reverse_points": 20,
    "w_model": 50.0,
    "ref_weight": 0.5,
    "w_smooth": 20.0,
    "w_s": 1.0,
    "w_fix": 100.0,
    "target_spacing": 0.5,
    "max_iter": 100,
}

UNSAFE_BROWSER_PORTS = {
    1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53,
    69, 77, 79, 87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117,
    119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513, 514,
    515, 526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989,
    990, 993, 995, 1719, 1720, 1723, 2049, 3659, 4045, 5060, 5061, 6000,
    6566, 6665, 6666, 6667, 6668, 6669, 6697, 10080,
}

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Kinematic Smoother Web Demo</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{
  --bg:#f1ece2;
  --bg-deep:#e6dcc9;
  --panel:#fbf8f1;
  --panel-strong:#fffdf8;
  --ink:#1f2833;
  --muted:#5d6a74;
  --line:rgba(31,40,51,0.12);
  --accent:#126e82;
  --accent-soft:rgba(18,110,130,0.12);
  --accent-warm:#d97706;
  --accent-warm-soft:rgba(217,119,6,0.14);
  --danger:#b42318;
  --shadow:0 18px 48px rgba(45,37,28,0.12);
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:radial-gradient(circle at top left, #f8f3ea 0%, var(--bg) 45%, var(--bg-deep) 100%);color:var(--ink)}
body{font-family:"Avenir Next","Segoe UI",sans-serif}
h1,h2,h3{font-family:"Iowan Old Style","Palatino Linotype",serif;margin:0}
button,input,select{font:inherit}
.app{
  min-height:100vh;
  display:grid;
  grid-template-columns:360px minmax(0,1fr);
}
.sidebar{
  padding:24px 20px;
  border-right:1px solid var(--line);
  background:linear-gradient(180deg, rgba(255,253,248,0.96), rgba(248,242,230,0.92));
  box-shadow:inset -1px 0 0 rgba(255,255,255,0.45);
  overflow:auto;
}
.sidebar-header{
  margin-bottom:20px;
}
.eyebrow{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  background:var(--accent-soft);
  color:var(--accent);
  font-size:12px;
  font-weight:700;
  letter-spacing:0.04em;
  text-transform:uppercase;
}
.sidebar-header h1{
  font-size:30px;
  line-height:1.02;
  margin-top:12px;
}
.sidebar-header p{
  margin:10px 0 0;
  color:var(--muted);
  line-height:1.55;
  font-size:14px;
}
.control-section{
  background:rgba(255,255,255,0.58);
  border:1px solid rgba(31,40,51,0.08);
  border-radius:22px;
  padding:16px 14px;
  box-shadow:0 8px 24px rgba(62,48,33,0.05);
}
.control-section + .control-section{
  margin-top:14px;
}
.control-section h2{
  font-size:18px;
  margin-bottom:6px;
}
.control-section p{
  margin:0 0 12px;
  color:var(--muted);
  font-size:13px;
  line-height:1.45;
}
.control{
  margin-bottom:12px;
}
.control:last-child{margin-bottom:0}
.control-label{
  display:flex;
  justify-content:space-between;
  gap:12px;
  font-size:12px;
  font-weight:700;
  color:var(--ink);
  margin-bottom:6px;
}
.control-value{
  color:var(--accent);
  font-variant-numeric:tabular-nums;
}
input[type=range]{
  width:100%;
  accent-color:var(--accent);
}
.toolbar{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top:16px;
}
.btn{
  border:none;
  border-radius:999px;
  padding:11px 16px;
  cursor:pointer;
  transition:transform .16s ease, box-shadow .16s ease, opacity .16s ease;
}
.btn:disabled{cursor:wait;opacity:.65;transform:none}
.btn-primary{
  background:linear-gradient(135deg, #165d6f, var(--accent));
  color:#fff;
  box-shadow:0 10px 24px rgba(18,110,130,0.28);
}
.btn-secondary{
  background:rgba(255,255,255,0.72);
  color:var(--ink);
  border:1px solid rgba(31,40,51,0.08);
}
.btn:hover:not(:disabled){transform:translateY(-1px)}
.toggle{
  display:flex;
  align-items:center;
  gap:8px;
  margin-top:12px;
  font-size:13px;
  color:var(--muted);
}
.workspace{
  padding:24px;
  overflow:auto;
}
.hero{
  display:flex;
  justify-content:space-between;
  gap:18px;
  align-items:flex-start;
  margin-bottom:16px;
}
.hero-copy h2{
  font-size:33px;
  line-height:1.05;
}
.hero-copy p{
  max-width:760px;
  margin:10px 0 0;
  color:var(--muted);
  line-height:1.55;
}
.status-card{
  min-width:240px;
  background:var(--panel-strong);
  border:1px solid rgba(31,40,51,0.08);
  border-radius:24px;
  padding:16px 18px;
  box-shadow:var(--shadow);
}
.status-row{
  display:flex;
  justify-content:space-between;
  gap:12px;
  align-items:center;
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border-radius:999px;
  padding:6px 10px;
  background:var(--accent-soft);
  color:var(--accent);
  font-size:12px;
  font-weight:700;
}
.badge.error{
  background:rgba(180,35,24,0.12);
  color:var(--danger);
}
.status-message{
  margin-top:10px;
  color:var(--muted);
  font-size:13px;
  line-height:1.5;
}
.metrics{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:14px;
  margin-bottom:14px;
}
.metric{
  background:var(--panel);
  border:1px solid rgba(31,40,51,0.08);
  border-radius:22px;
  padding:14px 16px;
  box-shadow:var(--shadow);
}
.metric-title{
  font-size:12px;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:0.05em;
}
.metric-value{
  margin-top:8px;
  font-size:26px;
  font-weight:700;
  line-height:1.1;
}
.metric-sub{
  margin-top:6px;
  font-size:13px;
  color:var(--muted);
}
.results-grid{
  display:grid;
  grid-template-columns:minmax(0,1.65fr) minmax(320px,0.75fr);
  gap:14px;
  align-items:start;
}
.plot-card,
.info-card{
  background:var(--panel-strong);
  border:1px solid rgba(31,40,51,0.08);
  border-radius:26px;
  box-shadow:var(--shadow);
  overflow:hidden;
}
.plot-header,
.info-header{
  padding:16px 18px 0;
}
.plot-header h3,
.info-header h3{
  font-size:20px;
}
.plot-header p,
.info-header p{
  margin:6px 0 0;
  color:var(--muted);
  font-size:13px;
  line-height:1.45;
}
.plot-area{
  height:470px;
  padding:4px 10px 12px;
}
.info-body{
  padding:14px 18px 18px;
}
.kv-list{
  display:grid;
  gap:10px;
}
.kv-row{
  display:flex;
  justify-content:space-between;
  gap:12px;
  border-bottom:1px dashed rgba(31,40,51,0.12);
  padding-bottom:8px;
  font-size:14px;
}
.kv-row:last-child{border-bottom:none;padding-bottom:0}
.kv-row span:first-child{color:var(--muted)}
.kv-row span:last-child{font-variant-numeric:tabular-nums}
.mini-grid{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:14px;
  margin-top:14px;
}
.mini-plot{
  height:280px;
  padding:4px 8px 12px;
}
@media (max-width: 1320px){
  .metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
  .results-grid{grid-template-columns:1fr}
  .mini-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width: 980px){
  .app{grid-template-columns:1fr}
  .sidebar{border-right:none;border-bottom:1px solid var(--line)}
  .workspace{padding:18px}
  .hero{flex-direction:column}
  .mini-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sidebar-header">
      <span class="eyebrow">Kinematic Smoother</span>
      <h1>左侧改参数，右侧看轨迹。</h1>
      <p>这个页面直接调用当前仓库里的平滑器。左侧调原始 cusp 形状和优化权重，右侧会刷新路径、航向、曲率和采样间距。</p>
    </div>

    <section class="control-section">
      <h2>Path Shape</h2>
      <p>控制原始前进段长度和倒车终点，快速观察 cusp 附近的几何变化。</p>

      <div class="control">
        <div class="control-label"><span>Forward length</span><span class="control-value" id="value-forward_length"></span></div>
        <input id="forward_length" type="range" min="4" max="20" step="0.5" />
      </div>
      <div class="control">
        <div class="control-label"><span>Reverse end x</span><span class="control-value" id="value-reverse_end_x"></span></div>
        <input id="reverse_end_x" type="range" min="-6" max="12" step="0.25" />
      </div>
      <div class="control">
        <div class="control-label"><span>Reverse end y</span><span class="control-value" id="value-reverse_end_y"></span></div>
        <input id="reverse_end_y" type="range" min="-6" max="10" step="0.25" />
      </div>
      <div class="control">
        <div class="control-label"><span>Forward points</span><span class="control-value" id="value-forward_points"></span></div>
        <input id="forward_points" type="range" min="8" max="60" step="1" />
      </div>
      <div class="control">
        <div class="control-label"><span>Reverse points</span><span class="control-value" id="value-reverse_points"></span></div>
        <input id="reverse_points" type="range" min="8" max="60" step="1" />
      </div>
    </section>

    <section class="control-section">
      <h2>Solver Weights</h2>
      <p>这些参数直接透传给 `KinematicSmoother`，用于看模型一致性、平滑性和参考贴合程度的权衡。</p>

      <div class="control">
        <div class="control-label"><span>w_model</span><span class="control-value" id="value-w_model"></span></div>
        <input id="w_model" type="range" min="0" max="120" step="1" />
      </div>
      <div class="control">
        <div class="control-label"><span>ref_weight</span><span class="control-value" id="value-ref_weight"></span></div>
        <input id="ref_weight" type="range" min="0" max="3" step="0.05" />
      </div>
      <div class="control">
        <div class="control-label"><span>w_smooth</span><span class="control-value" id="value-w_smooth"></span></div>
        <input id="w_smooth" type="range" min="0" max="60" step="1" />
      </div>
      <div class="control">
        <div class="control-label"><span>w_s</span><span class="control-value" id="value-w_s"></span></div>
        <input id="w_s" type="range" min="0" max="10" step="0.1" />
      </div>
      <div class="control">
        <div class="control-label"><span>w_fix</span><span class="control-value" id="value-w_fix"></span></div>
        <input id="w_fix" type="range" min="10" max="250" step="5" />
      </div>
      <div class="control">
        <div class="control-label"><span>target_spacing</span><span class="control-value" id="value-target_spacing"></span></div>
        <input id="target_spacing" type="range" min="0.1" max="1.5" step="0.05" />
      </div>
      <div class="control">
        <div class="control-label"><span>max_iter</span><span class="control-value" id="value-max_iter"></span></div>
        <input id="max_iter" type="range" min="20" max="200" step="5" />
      </div>

      <div class="toolbar">
        <button id="solve-btn" class="btn btn-primary">Solve</button>
        <button id="reset-btn" class="btn btn-secondary">Reset</button>
      </div>
      <label class="toggle"><input id="auto-solve" type="checkbox" checked /> Auto solve</label>
    </section>
  </aside>

  <main class="workspace">
    <section class="hero">
      <div class="hero-copy">
        <span class="eyebrow">Result View</span>
        <h2>实时看 cusp 附近怎么被拉顺。</h2>
        <p>黑色虚线是原始路径，蓝绿色是优化结果，橙色点标记 `ds` 接近 0 的 cusp 插入点。状态卡会显示优化是否收敛、迭代次数和残差规模。</p>
      </div>
      <div class="status-card">
        <div class="status-row">
          <strong>Solver status</strong>
          <span id="status-badge" class="badge">Idle</span>
        </div>
        <div id="status-message" class="status-message">等待初始求解。</div>
      </div>
    </section>

    <section class="metrics">
      <article class="metric">
        <div class="metric-title">Processed Points</div>
        <div class="metric-value" id="metric-points">-</div>
        <div class="metric-sub" id="metric-points-sub">raw → optimized</div>
      </article>
      <article class="metric">
        <div class="metric-title">Path Length</div>
        <div class="metric-value" id="metric-length">-</div>
        <div class="metric-sub">optimized integrated ds</div>
      </article>
      <article class="metric">
        <div class="metric-title">Max |kappa|</div>
        <div class="metric-value" id="metric-kappa">-</div>
        <div class="metric-sub">largest absolute curvature</div>
      </article>
      <article class="metric">
        <div class="metric-title">Residual RMS</div>
        <div class="metric-value" id="metric-rms">-</div>
        <div class="metric-sub">mean residual magnitude</div>
      </article>
    </section>

    <section class="results-grid">
      <article class="plot-card">
        <div class="plot-header">
          <h3>Path and Orientation</h3>
          <p>路径图里同时画原始折线、平滑结果和稀疏朝向短线，便于看 cusp 前后的姿态过渡。</p>
        </div>
        <div id="path-plot" class="plot-area"></div>
      </article>

      <article class="info-card">
        <div class="info-header">
          <h3>Run Summary</h3>
          <p>这里展示本次求解的关键统计，便于快速判断参数变化是否真的改善了结果。</p>
        </div>
        <div class="info-body">
          <div id="summary-list" class="kv-list"></div>
        </div>
      </article>
    </section>

    <section class="mini-grid">
      <article class="plot-card">
        <div class="plot-header">
          <h3>Theta</h3>
          <p>使用解包后的角度曲线，更容易看出 cusp 附近的姿态衔接趋势。</p>
        </div>
        <div id="theta-plot" class="mini-plot"></div>
      </article>
      <article class="plot-card">
        <div class="plot-header">
          <h3>Curvature</h3>
          <p>曲率分布可以直观看出平滑程度和局部过激弯折。</p>
        </div>
        <div id="kappa-plot" class="mini-plot"></div>
      </article>
      <article class="plot-card">
        <div class="plot-header">
          <h3>Delta S</h3>
          <p>红色虚线是目标间距，橙色点帮助识别 cusp 对应的近零间距位置。</p>
        </div>
        <div id="ds-plot" class="mini-plot"></div>
      </article>
      <article class="plot-card">
        <div class="plot-header">
          <h3>dcurv</h3>
          <p>曲率变化率越平，说明轨迹在弯折处越连续。</p>
        </div>
        <div id="dcurv-plot" class="mini-plot"></div>
      </article>
    </section>
  </main>
</div>

<script>
const DEFAULTS = __DEFAULTS__;
const CONTROLS = [
  {id:'forward_length', digits:2},
  {id:'reverse_end_x', digits:2},
  {id:'reverse_end_y', digits:2},
  {id:'forward_points', digits:0},
  {id:'reverse_points', digits:0},
  {id:'w_model', digits:1},
  {id:'ref_weight', digits:2},
  {id:'w_smooth', digits:1},
  {id:'w_s', digits:2},
  {id:'w_fix', digits:0},
  {id:'target_spacing', digits:2},
  {id:'max_iter', digits:0},
];

let debounceTimer = null;
let currentRequest = 0;

function setStatus(kind, label, message) {
  const badge = document.getElementById('status-badge');
  const msg = document.getElementById('status-message');
  badge.textContent = label;
  badge.className = kind === 'error' ? 'badge error' : 'badge';
  msg.textContent = message;
}

function setBusy(isBusy) {
  document.getElementById('solve-btn').disabled = isBusy;
}

function formatFixed(value, digits) {
  return Number(value).toFixed(digits);
}

function updateControlLabel(id) {
  const meta = CONTROLS.find(item => item.id === id);
  const el = document.getElementById(id);
  const valueEl = document.getElementById(`value-${id}`);
  valueEl.textContent = formatFixed(el.value, meta.digits);
}

function applyDefaults() {
  CONTROLS.forEach(({id}) => {
    const el = document.getElementById(id);
    el.value = DEFAULTS[id];
    updateControlLabel(id);
  });
}

function collectPayload() {
  const payload = {};
  CONTROLS.forEach(({id}) => {
    payload[id] = Number(document.getElementById(id).value);
  });
  return payload;
}

function debounceSolve() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    solve();
  }, 250);
}

function buildHeadingTrace(solution, spacing) {
  const xs = [];
  const ys = [];
  const n = solution.x.length;
  const stride = Math.max(1, Math.floor(n / 18));
  const segLen = Math.max(0.28, spacing * 0.8);
  for (let i = 0; i < n; i += stride) {
    xs.push(solution.x[i], solution.x[i] + segLen * Math.cos(solution.theta_rad[i]), null);
    ys.push(solution.y[i], solution.y[i] + segLen * Math.sin(solution.theta_rad[i]), null);
  }
  return {
    x: xs,
    y: ys,
    type: 'scatter',
    mode: 'lines',
    name: 'Heading',
    line: {color: 'rgba(217,119,6,0.72)', width: 1.5},
    opacity: 0.55,
    hoverinfo: 'skip',
  };
}

function buildPoseArrowTrace(pose, label, color, length, dashed=false, opacity=0.8) {
  return {
    x: [pose.x, pose.x + length * Math.cos(pose.theta_rad)],
    y: [pose.y, pose.y + length * Math.sin(pose.theta_rad)],
    type: 'scatter',
    mode: 'lines+markers',
    name: label,
    line: {color, width: 3, dash: dashed ? 'dash' : 'solid'},
    marker: {size: [10, 9], color, symbol: ['circle', 'arrow-bar-up']},
    opacity,
    hovertemplate: `${label}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>`,
  };
}

function buildDirectionalPathTraces(x, y, segmentGears, labelPrefix, style) {
  const traces = [];
  let current = null;
  let forwardShown = false;
  let reverseShown = false;

  function flushCurrent() {
    if (!current || current.x.length < 2) {
      current = null;
      return;
    }
    const isForward = current.direction === 'forward';
    const label = `${labelPrefix} ${isForward ? 'forward' : 'reverse'}`;
    traces.push({
      x: current.x,
      y: current.y,
      type: 'scatter',
      mode: 'lines+markers',
      name: label,
      legendgroup: label,
      showlegend: isForward ? !forwardShown : !reverseShown,
      line: {
        color: isForward ? style.forwardColor : style.reverseColor,
        width: style.lineWidth,
        dash: style.dash,
      },
      marker: {
        size: style.markerSize,
        symbol: style.markerSymbol,
        color: isForward ? style.forwardColor : style.reverseColor,
      },
      opacity: style.opacity,
      hovertemplate: `${label}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>`,
    });
    if (isForward) {
      forwardShown = true;
    } else {
      reverseShown = true;
    }
    current = null;
  }

  for (let i = 0; i < segmentGears.length; i += 1) {
    const gear = segmentGears[i];
    const direction = gear > 0 ? 'forward' : gear < 0 ? 'reverse' : 'cusp';

    if (direction === 'cusp') {
      flushCurrent();
      continue;
    }

    if (!current || current.direction !== direction) {
      flushCurrent();
      current = {
        direction,
        x: [x[i], x[i + 1]],
        y: [y[i], y[i + 1]],
      };
      continue;
    }

    current.x.push(x[i + 1]);
    current.y.push(y[i + 1]);
  }

  flushCurrent();
  return traces;
}

function renderMetrics(summary) {
  document.getElementById('metric-points').textContent = `${summary.raw_points} → ${summary.optimized_points}`;
  document.getElementById('metric-points-sub').textContent = `${summary.cusp_points} cusp point(s)`;
  document.getElementById('metric-length').textContent = `${summary.path_length_optimized.toFixed(2)} m`;
  document.getElementById('metric-kappa').textContent = summary.max_abs_kappa.toFixed(3);
  document.getElementById('metric-rms').textContent = summary.residual_rms.toFixed(4);
}

function renderSummary(summary, params) {
  const items = [
    ['Solve time', `${summary.solve_time_ms.toFixed(1)} ms`],
    ['Iterations', `${summary.nfev}`],
    ['Cost', summary.cost.toFixed(6)],
    ['Optimality', summary.optimality.toExponential(2)],
    ['Mean ds', `${summary.mean_ds.toFixed(3)} m`],
    ['Min / max ds', `${summary.min_ds.toFixed(3)} / ${summary.max_ds.toFixed(3)}`],
    ['Raw length', `${summary.path_length_raw.toFixed(3)} m`],
    ['Target spacing', `${params.target_spacing.toFixed(3)} m`],
    ['Status code', `${summary.status}`],
  ];
  document.getElementById('summary-list').innerHTML = items.map(([k, v]) => (
    `<div class="kv-row"><span>${k}</span><span>${v}</span></div>`
  )).join('');
}

function getDcurvSeries(solution) {
  return {
    indices: solution.index,
    values: solution.dcurv,
  };
}

function buildCuspMarkerTrace(x, y, label='Cusp') {
  return {
    x,
    y,
    type: 'scatter',
    mode: 'markers',
    name: label,
    marker: {size: 9, color: '#d97706', line: {width: 1, color: '#fff7ed'}},
    opacity: 0.95,
  };
}

function renderPlots(payload) {
  const raw = payload.raw;
  const solution = payload.solution;
  const summary = payload.summary;
  const targetSpacing = payload.params.target_spacing;
  const rawPose = payload.raw_pose;
  const optimizedPose = payload.optimized_pose;
  const cuspIndices = solution.cusp_indices;
  const cuspX = solution.cusp_indices.map(i => solution.x[i]);
  const cuspY = solution.cusp_indices.map(i => solution.y[i]);
  const cuspDs = solution.cusp_indices.map(i => solution.ds[i]);
  const cuspTheta = solution.cusp_indices.map(i => solution.theta_deg[i]);
  const cuspKappa = solution.cusp_indices.map(i => solution.kappa[i]);
  const dcurv = getDcurvSeries(solution);
  const cuspDcurv = solution.cusp_indices.map(() => 0);
  const poseArrowLength = Math.max(0.45, targetSpacing * 1.4);
  const pathTraces = [
    ...buildDirectionalPathTraces(raw.x, raw.y, raw.segment_gears, 'Raw', {
      forwardColor: '#2f855a',
      reverseColor: '#c2410c',
      lineWidth: 2,
      dash: 'dash',
      markerSize: 6,
      markerSymbol: 'x',
      opacity: 0.42,
    }),
    ...buildDirectionalPathTraces(solution.x, solution.y, solution.segment_gears, 'Optimized', {
      forwardColor: '#16a34a',
      reverseColor: '#dc2626',
      lineWidth: 3,
      dash: 'solid',
      markerSize: 5,
      markerSymbol: 'circle',
      opacity: 0.68,
    }),
    buildHeadingTrace(solution, targetSpacing),
    buildPoseArrowTrace(rawPose.start, 'Raw start heading', '#2f855a', poseArrowLength, true, 0.5),
    buildPoseArrowTrace(rawPose.end, 'Raw end heading', '#b45309', poseArrowLength, true, 0.5),
    buildPoseArrowTrace(optimizedPose.start, 'Optimized start heading', '#16a34a', poseArrowLength, false, 0.78),
    buildPoseArrowTrace(optimizedPose.end, 'Optimized end heading', '#dc2626', poseArrowLength, false, 0.78),
    {
      x: cuspX,
      y: cuspY,
      type: 'scatter',
      mode: 'markers',
      name: 'Cusp',
      marker: {size: 10, color: '#d97706', line: {width: 1, color: '#fff7ed'}},
      opacity: 0.9,
    },
  ];

  Plotly.react('path-plot', pathTraces, {
    margin: {l: 58, r: 26, t: 16, b: 56},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      title: 'X [m]',
      gridcolor: 'rgba(31,40,51,0.08)',
      zerolinecolor: 'rgba(31,40,51,0.12)',
      scaleanchor: 'y',
    },
    yaxis: {
      title: 'Y [m]',
      gridcolor: 'rgba(31,40,51,0.08)',
      zerolinecolor: 'rgba(31,40,51,0.12)',
    },
    legend: {orientation: 'h', y: 1.1, x: 0},
  }, {responsive: true, scrollZoom: true});

  Plotly.react('theta-plot', [
    {
      x: solution.index,
      y: solution.theta_deg,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'theta',
      line: {color: '#126e82', width: 2.4},
      marker: {size: 5, color: '#126e82'},
    },
    buildCuspMarkerTrace(cuspIndices, cuspTheta),
  ], {
    margin: {l: 58, r: 18, t: 12, b: 48},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {title: 'Point index', gridcolor: 'rgba(31,40,51,0.08)'},
    yaxis: {title: 'Theta [deg]', gridcolor: 'rgba(31,40,51,0.08)'},
    showlegend: false,
  }, {responsive: true, scrollZoom: true});

  Plotly.react('kappa-plot', [
    {
      x: solution.index,
      y: solution.kappa,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'kappa',
      line: {color: '#7c5c2b', width: 2.4},
      marker: {size: 5, color: '#7c5c2b'},
    },
    buildCuspMarkerTrace(cuspIndices, cuspKappa),
    {
      x: [solution.index[0], solution.index[solution.index.length - 1]],
      y: [0, 0],
      type: 'scatter',
      mode: 'lines',
      name: 'zero',
      line: {color: 'rgba(31,40,51,0.35)', width: 1, dash: 'dot'},
      hoverinfo: 'skip',
    },
  ], {
    margin: {l: 58, r: 18, t: 12, b: 48},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {title: 'Point index', gridcolor: 'rgba(31,40,51,0.08)'},
    yaxis: {title: 'Kappa [1/m]', gridcolor: 'rgba(31,40,51,0.08)'},
    showlegend: false,
  }, {responsive: true, scrollZoom: true});

  Plotly.react('ds-plot', [
    {
      x: solution.index,
      y: solution.ds,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'ds',
      line: {color: '#126e82', width: 2.4},
      marker: {size: 5, color: '#126e82'},
    },
    {
      x: [solution.index[0], solution.index[solution.index.length - 1]],
      y: [targetSpacing, targetSpacing],
      type: 'scatter',
      mode: 'lines',
      name: 'target',
      line: {color: '#b42318', width: 1.5, dash: 'dash'},
      hoverinfo: 'skip',
    },
    {
      x: solution.cusp_indices,
      y: cuspDs,
      type: 'scatter',
      mode: 'markers',
      name: 'cusp',
      marker: {size: 10, color: '#d97706'},
    },
  ], {
    margin: {l: 58, r: 18, t: 12, b: 48},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {title: 'Point index', gridcolor: 'rgba(31,40,51,0.08)'},
    yaxis: {title: 'Delta s [m]', gridcolor: 'rgba(31,40,51,0.08)'},
    showlegend: false,
  }, {responsive: true, scrollZoom: true});

  Plotly.react('dcurv-plot', [
    {
      x: dcurv.indices,
      y: dcurv.values,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'dcurv',
      line: {color: '#7b2cbf', width: 2.4},
      marker: {size: 5, color: '#7b2cbf'},
    },
    buildCuspMarkerTrace(cuspIndices, cuspDcurv, 'Cusp (undefined dcurv)'),
    {
      x: [dcurv.indices[0], dcurv.indices[dcurv.indices.length - 1]],
      y: [0, 0],
      type: 'scatter',
      mode: 'lines',
      name: 'zero',
      line: {color: 'rgba(31,40,51,0.35)', width: 1, dash: 'dot'},
      hoverinfo: 'skip',
    },
  ], {
    margin: {l: 58, r: 18, t: 12, b: 48},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {title: 'Point index', gridcolor: 'rgba(31,40,51,0.08)'},
    yaxis: {title: 'dk/ds [1/m^2]', gridcolor: 'rgba(31,40,51,0.08)'},
    showlegend: false,
  }, {responsive: true, scrollZoom: true});

  renderMetrics(summary);
  renderSummary(summary, payload.params);
}

async function solve() {
  const requestId = ++currentRequest;
  setBusy(true);
  setStatus('info', 'Solving', '正在调用后端优化器，请稍等。');
  try {
    const response = await fetch('/api/solve', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(collectPayload()),
    });
    const payload = await response.json();
    if (requestId !== currentRequest) {
      return;
    }
    if (!response.ok) {
      throw new Error(payload.error || 'Solve failed');
    }
    renderPlots(payload);
    const summary = payload.summary;
    const label = summary.success ? 'Converged' : 'Warning';
    const message = `${summary.message} | nfev=${summary.nfev}, cost=${summary.cost.toFixed(6)}, time=${summary.solve_time_ms.toFixed(1)} ms`;
    setStatus(summary.success ? 'info' : 'error', label, message);
  } catch (error) {
    if (requestId !== currentRequest) {
      return;
    }
    setStatus('error', 'Error', error.message);
  } finally {
    if (requestId === currentRequest) {
      setBusy(false);
    }
  }
}

function init() {
  applyDefaults();
  CONTROLS.forEach(({id}) => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      updateControlLabel(id);
      if (document.getElementById('auto-solve').checked) {
        debounceSolve();
      }
    });
  });
  document.getElementById('solve-btn').addEventListener('click', solve);
  document.getElementById('reset-btn').addEventListener('click', () => {
    applyDefaults();
    solve();
  });
  solve();
}

window.addEventListener('load', init);
</script>
</body>
</html>
"""


def create_cusp_path(
    forward_length: float,
    reverse_end_x: float,
    reverse_end_y: float,
    forward_points: int,
    reverse_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    forward_points = max(2, int(forward_points))
    reverse_points = max(2, int(reverse_points))

    x1 = np.linspace(0.0, forward_length, forward_points)
    y1 = np.zeros_like(x1)

    x2 = np.linspace(forward_length, reverse_end_x, reverse_points)
    y2 = np.linspace(0.0, reverse_end_y, reverse_points)

    path = np.column_stack((np.concatenate((x1, x2[1:])), np.concatenate((y1, y2[1:]))))
    gears = np.ones(len(path) - 1, dtype=float)
    gears[forward_points - 1 :] = -1.0
    return path, gears


def _safe_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _safe_int(value: Any, *, name: str) -> int:
    return int(round(_safe_float(value, name=name)))


def _sanitize_params(req: dict[str, Any]) -> dict[str, Any]:
    params = dict(DEFAULTS)
    params["forward_length"] = float(np.clip(_safe_float(req.get("forward_length", params["forward_length"]), name="forward_length"), 4.0, 20.0))
    params["reverse_end_x"] = float(np.clip(_safe_float(req.get("reverse_end_x", params["reverse_end_x"]), name="reverse_end_x"), -6.0, 12.0))
    params["reverse_end_y"] = float(np.clip(_safe_float(req.get("reverse_end_y", params["reverse_end_y"]), name="reverse_end_y"), -6.0, 10.0))
    params["forward_points"] = int(np.clip(_safe_int(req.get("forward_points", params["forward_points"]), name="forward_points"), 8, 60))
    params["reverse_points"] = int(np.clip(_safe_int(req.get("reverse_points", params["reverse_points"]), name="reverse_points"), 8, 60))
    params["w_model"] = float(np.clip(_safe_float(req.get("w_model", params["w_model"]), name="w_model"), 0.0, 120.0))
    params["ref_weight"] = float(np.clip(_safe_float(req.get("ref_weight", params["ref_weight"]), name="ref_weight"), 0.0, 3.0))
    params["w_smooth"] = float(np.clip(_safe_float(req.get("w_smooth", params["w_smooth"]), name="w_smooth"), 0.0, 60.0))
    params["w_s"] = float(np.clip(_safe_float(req.get("w_s", params["w_s"]), name="w_s"), 0.0, 10.0))
    params["w_fix"] = float(np.clip(_safe_float(req.get("w_fix", params["w_fix"]), name="w_fix"), 10.0, 250.0))
    params["target_spacing"] = float(np.clip(_safe_float(req.get("target_spacing", params["target_spacing"]), name="target_spacing"), 0.1, 1.5))
    params["max_iter"] = int(np.clip(_safe_int(req.get("max_iter", params["max_iter"]), name="max_iter"), 20, 200))

    reverse_leg = math.hypot(params["forward_length"] - params["reverse_end_x"], params["reverse_end_y"])
    if reverse_leg < 0.2:
        raise ValueError("reverse leg is too short; move the reverse endpoint farther from the cusp")

    return params


def _build_payload(params: dict[str, Any]) -> dict[str, Any]:
    path, gears = create_cusp_path(
        forward_length=params["forward_length"],
        reverse_end_x=params["reverse_end_x"],
        reverse_end_y=params["reverse_end_y"],
        forward_points=params["forward_points"],
        reverse_points=params["reverse_points"],
    )

    smoother = KinematicSmoother(
        w_model=params["w_model"],
        ref_weight=params["ref_weight"],
        w_smooth=params["w_smooth"],
        w_s=params["w_s"],
        w_fix=params["w_fix"],
        target_spacing=params["target_spacing"],
        max_iter=params["max_iter"],
    )
    solve_start = perf_counter()
    result = smoother.optimize(path, gears, return_result=True, verbose=0)
    solve_time_ms = (perf_counter() - solve_start) * 1000.0
    solution = result.x.reshape((-1, 5))
    processed_gears = []
    for i, curr_gear in enumerate(gears):
      next_gear = gears[i + 1] if i + 1 < len(gears) else curr_gear
      processed_gears.append(float(curr_gear))
      if i < len(gears) - 1 and curr_gear != next_gear:
        processed_gears.append(0.0)
    processed_gears = np.array(processed_gears, dtype=float)

    x = solution[:, 0]
    y = solution[:, 1]
    theta = solution[:, 2]
    kappa = solution[:, 3]
    ds = solution[:, 4]
    theta_deg = np.rad2deg(np.unwrap(theta))
    cusp_threshold = max(1e-3, params["target_spacing"] * 0.05)
    cusp_indices = np.flatnonzero(ds[:-1] < cusp_threshold) if len(ds) > 1 else np.array([], dtype=int)
    raw_length = float(np.sum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))
    optimized_ds = ds[:-1] if len(ds) > 1 else np.array([], dtype=float)
    optimized_length = float(np.sum(optimized_ds)) if optimized_ds.size else 0.0
    residual_rms = float(np.sqrt(np.mean(np.square(result.fun)))) if result.fun.size else 0.0
    dcurv = [None] * len(kappa)
    for i in range(len(kappa) - 1):
      if processed_gears[i] == 0.0:
        continue
      step = max(float(ds[i]), 0.03)
      dcurv[i] = float((kappa[i + 1] - kappa[i]) / step)

    raw_start_theta = math.atan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])
    raw_end_theta = math.atan2(path[-1, 1] - path[-2, 1], path[-1, 0] - path[-2, 0])
    if gears[-1] < 0:
      raw_end_theta += math.pi
    raw_start_theta = float((raw_start_theta + math.pi) % (2.0 * math.pi) - math.pi)
    raw_end_theta = float((raw_end_theta + math.pi) % (2.0 * math.pi) - math.pi)

    summary = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "solve_time_ms": float(solve_time_ms),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "residual_rms": residual_rms,
        "raw_points": int(len(path)),
        "optimized_points": int(len(solution)),
        "cusp_points": int(len(cusp_indices)),
        "path_length_raw": raw_length,
        "path_length_optimized": optimized_length,
        "max_abs_kappa": float(np.max(np.abs(kappa))) if len(kappa) else 0.0,
        "max_abs_dcurv": float(max((abs(value) for value in dcurv if value is not None), default=0.0)),
        "mean_ds": float(np.mean(optimized_ds)) if optimized_ds.size else 0.0,
        "min_ds": float(np.min(optimized_ds)) if optimized_ds.size else 0.0,
        "max_ds": float(np.max(optimized_ds)) if optimized_ds.size else 0.0,
    }

    return {
        "params": params,
        "raw": {
            "x": path[:, 0].tolist(),
            "y": path[:, 1].tolist(),
            "gears": gears.tolist(),
          "segment_gears": gears.tolist(),
        },
        "raw_pose": {
          "start": {
            "x": float(path[0, 0]),
            "y": float(path[0, 1]),
            "theta_rad": raw_start_theta,
            "theta_deg": float(np.rad2deg(raw_start_theta)),
          },
          "end": {
            "x": float(path[-1, 0]),
            "y": float(path[-1, 1]),
            "theta_rad": raw_end_theta,
            "theta_deg": float(np.rad2deg(raw_end_theta)),
          },
        },
        "optimized_pose": {
          "start": {
            "x": float(x[0]),
            "y": float(y[0]),
            "theta_rad": float(theta[0]),
            "theta_deg": float(np.rad2deg(theta[0])),
          },
          "end": {
            "x": float(x[-1]),
            "y": float(y[-1]),
            "theta_rad": float(theta[-1]),
            "theta_deg": float(np.rad2deg(theta[-1])),
          },
        },
        "solution": {
            "index": np.arange(len(solution)).tolist(),
            "x": x.tolist(),
            "y": y.tolist(),
            "theta_rad": theta.tolist(),
            "theta_deg": theta_deg.tolist(),
            "kappa": kappa.tolist(),
            "dcurv": dcurv,
            "ds": ds.tolist(),
          "segment_gears": processed_gears.tolist(),
            "cusp_indices": cusp_indices.astype(int).tolist(),
        },
        "summary": summary,
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        html = HTML_PAGE.replace("__DEFAULTS__", json.dumps(DEFAULTS, sort_keys=True))
        return Response(html, mimetype="text/html")

    @app.get("/api/init")
    def api_init() -> Any:
        return jsonify(_build_payload(dict(DEFAULTS)))

    @app.post("/api/solve")
    def api_solve() -> Any:
        req = request.get_json(silent=True) or {}
        try:
            params = _sanitize_params(req)
            return jsonify(_build_payload(params))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kinematic smoother web demo")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=5080, help="Bind port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.port in UNSAFE_BROWSER_PORTS:
        raise SystemExit(
            f"Port {args.port} is blocked by modern browsers as an unsafe port. "
            "Use a different port such as 5080 or 5000."
        )
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
