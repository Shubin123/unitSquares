import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { construct, finalize, validate, CSV_HEADER, csvRows } from './geometry.js';
import { createGPUBackend } from './gpu-solver.js';
import { createCPUBackend } from './cpu-solver.js';
import { Optimizer } from './optimizer.js';

const N_MAX = 1000;
const LS_KEY = 'squares-in-squares:improved:v1';
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 6) => (Number.isFinite(v) ? v.toFixed(d) : '—');

// ---------------------------------------------------------------- solutions

const records = await fetch('data/records.json').then((r) => r.json()).then((j) => j.s).catch(() => ({}));
const recordKeys = Object.keys(records).map(Number).sort((a, b) => a - b);
const recordMax = 323; // dense coverage of the table ends here
function recordFor(n) {
  if (n > recordMax) return null;
  for (const k of recordKeys) if (k >= n) return records[k];
  return null;
}

const bundled = await fetch('data/improved.json').then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
let local = {};
try { local = JSON.parse(localStorage.getItem(LS_KEY) || '{}'); } catch { local = {}; }

const unpack = (e, label) => ({ s: e.s, squares: e.q.map(([x, y, t]) => ({ x, y, t })), kind: 'optimized', label });
const cache = new Map();
function solution(n) {
  if (cache.has(n)) return cache.get(n);
  const c = construct(n);
  let best = { ...c, squares: finalize(c.squares) ?? c.squares };
  best.s = validate(best.squares).s ?? c.s;
  for (const [src, label] of [[bundled, 'GPU search (bundled)'], [local, 'GPU search (this browser)']]) {
    const e = src[n];
    if (e && e.s < best.s - 1e-9) {
      const cand = unpack(e, label);
      const v = validate(cand.squares);
      if (v.ok && v.s < best.s - 1e-9) best = { ...cand, s: v.s };
    }
  }
  cache.set(n, best);
  return best;
}
function quickS(n) {
  const e = Math.min(bundled[n]?.s ?? Infinity, local[n]?.s ?? Infinity);
  return Math.min(construct(n).s, e);
}
function saveLocal(n, sol) {
  local[n] = { s: sol.s, q: sol.squares.map((q) => [q.x, q.y, q.t]) };
  try { localStorage.setItem(LS_KEY, JSON.stringify(local)); } catch { /* storage unavailable */ }
  cache.set(n, sol);
}

// ---------------------------------------------------------------- three.js scene

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const host = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f9c, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
scene.add(sun, sun.target);

const world = new THREE.Group();
scene.add(world);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshStandardMaterial({ roughness: 0.95 }));
floor.receiveShadow = true;
world.add(floor);
const frame = new THREE.Group();
world.add(frame);
const frameMat = new THREE.MeshStandardMaterial({ color: 0x30343c, roughness: 0.6 });

const H = 0.22;
const bodyGeo = new THREE.BoxGeometry(1, 1, H);
const capGeo = new THREE.PlaneGeometry(0.9, 0.9);
let bodies = null, caps = null, capacity = 0;
function ensureCapacity(n) {
  if (n <= capacity) return;
  capacity = Math.max(n, capacity * 2, 64);
  for (const m of [bodies, caps]) if (m) { world.remove(m); m.dispose(); }
  bodies = new THREE.InstancedMesh(bodyGeo, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 }), capacity);
  caps = new THREE.InstancedMesh(capGeo, new THREE.MeshStandardMaterial({ roughness: 0.4 }), capacity);
  bodies.castShadow = true; bodies.receiveShadow = true;
  world.add(bodies, caps);
}

const tmp = new THREE.Object3D(), col = new THREE.Color(), col2 = new THREE.Color();
const C_AXIS = new THREE.Color('#3d6fd6'), C_TILT = new THREE.Color('#e0832f'), C_HOT = new THREE.Color('#e5383b');

// Draw squares {x, y, t} (or a Float32Array of x,y,t,_) with optional per-square penalty.
function drawSquares(list, n, energy) {
  ensureCapacity(n);
  for (let i = 0; i < n; i++) {
    const packed = ArrayBuffer.isView(list);
    const x = packed ? list[4 * i] : list[i].x, y = packed ? list[4 * i + 1] : list[i].y, t = packed ? list[4 * i + 2] : list[i].t;
    tmp.position.set(x, y, H / 2); tmp.rotation.set(0, 0, t); tmp.updateMatrix();
    bodies.setMatrixAt(i, tmp.matrix);
    tmp.position.z = H + 0.002; tmp.updateMatrix();
    caps.setMatrixAt(i, tmp.matrix);
    const tilt = Math.abs(Math.sin(2 * t)) > 1e-3;
    col.copy(tilt ? C_TILT : C_AXIS);
    const e = energy ? energy[4 * i + 3] : 0;
    if (e > 1e-7) col.lerp(C_HOT, Math.min(1, Math.sqrt(e) * 20));
    bodies.setColorAt(i, col.clone().multiplyScalar(0.62));
    caps.setColorAt(i, col2.copy(col).lerp(new THREE.Color(1, 1, 1), 0.18));
  }
  bodies.count = caps.count = n;
  for (const m of [bodies, caps]) { m.instanceMatrix.needsUpdate = true; m.instanceColor.needsUpdate = true; }
}

let viewL = 0;
function setContainer(L) {
  floor.scale.set(L, L, 1); floor.position.set(L / 2, L / 2, 0);
  frame.clear();
  const w = Math.max(0.04, L * 0.008), h = 0.12;
  for (const [x, y, sx, sy] of [[L / 2, -w / 2, L + 2 * w, w], [L / 2, L + w / 2, L + 2 * w, w], [-w / 2, L / 2, w, L], [L + w / 2, L / 2, w, L]]) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, h), frameMat);
    b.position.set(x, y, h / 2); b.castShadow = true;
    frame.add(b);
  }
  if (Math.abs(L - viewL) / Math.max(L, 1) > 0.15) fitCamera(L, tiltMode);
  viewL = L;
  sun.position.set(L * 0.2, -L * 0.4, L * 1.4); sun.target.position.set(L / 2, L / 2, 0);
  const sc = sun.shadow.camera; sc.left = sc.bottom = -L; sc.right = sc.top = L; sc.far = L * 4; sc.updateProjectionMatrix();
}

let tiltMode = true;
function fitCamera(L, tilt) {
  tiltMode = tilt;
  const c = new THREE.Vector3(L / 2, L / 2, 0);
  const dist = L / (2 * Math.tan((camera.fov * Math.PI) / 360)) * 1.25;
  camera.up.set(0, tilt ? 0 : 1, tilt ? 1 : 0);
  if (tilt) camera.position.set(L / 2, L / 2 - dist * 0.62, dist * 0.82);
  else camera.position.set(L / 2, L / 2, dist);
  controls.target.copy(c);
  camera.updateProjectionMatrix();
  controls.update();
}

function applyTheme() {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  scene.background = new THREE.Color(css('--surface'));
  floor.material.color.set(dark ? '#262a31' : '#efede6');
  frameMat.color.set(dark ? '#8a8f9c' : '#30343c');
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { applyTheme(); drawChart(); });
applyTheme();

new ResizeObserver(() => {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}).observe(host);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

$('btn-top').onclick = () => fitCamera(viewL, false);
$('btn-tilt').onclick = () => fitCamera(viewL, true);

// ---------------------------------------------------------------- UI state

let n = Math.min(N_MAX, Math.max(1, +(new URLSearchParams(location.search).get('n')) || 11));
let opt = null, running = false;

function showStats(sol) {
  const rec = recordFor(n), grid = Math.ceil(Math.sqrt(n));
  $('hud-n').textContent = n;
  $('hud-s').textContent = fmt(sol.s, 9);
  $('hud-kind').textContent = sol.label;
  $('st-s').textContent = fmt(sol.s, 9);
  $('st-score').textContent = fmt(sol.s / Math.sqrt(n) - 1, 6);
  $('st-grid').textContent = grid;
  $('st-rec').textContent = rec ? fmt(rec, 9) : 'n/a';
  const gap = rec ? sol.s - rec : NaN;
  const gEl = $('st-gap');
  gEl.textContent = rec ? (gap < 1e-7 ? (gap < -1e-7 ? `${fmt(gap, 6)} (new!)` : 'matched') : `+${fmt(gap, 6)}`) : 'n/a';
  gEl.className = rec ? (gap < 1e-7 ? 'good' : 'warn') : '';
  $('st-kind').textContent = sol.label;
}

function selectN(v) {
  stopOpt();
  n = Math.min(N_MAX, Math.max(1, Math.round(v) || 1));
  $('n-input').value = n; $('n-range').value = n;
  history.replaceState(null, '', `?n=${n}`);
  const sol = solution(n);
  setContainer(sol.s);
  drawSquares(sol.squares, n);
  showStats(sol);
  drawChart();
}

$('n-input').onchange = (e) => selectN(+e.target.value);
$('n-range').oninput = (e) => selectN(+e.target.value);
$('n-prev').onclick = () => selectN(n - 1);
$('n-next').onclick = () => selectN(n + 1);
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') selectN(n - 1);
  if (e.key === 'ArrowRight') selectN(n + 1);
});

// ---------------------------------------------------------------- optimizer

let backendPromise = createGPUBackend().catch((err) => {
  console.warn('WebGPU unavailable, using CPU', err);
  return createCPUBackend();
});
backendPromise.then((b) => {
  $('backend').textContent = b.kind === 'webgpu' ? `WebGPU · ${b.name}` : 'CPU fallback';
  $('backend').title = b.name;
});

function stopOpt() {
  running = false;
  $('btn-opt').textContent = 'Optimize';
  $('btn-opt').classList.add('primary');
}

async function startOpt() {
  if (n < 2) return;
  const backend = await backendPromise;
  const sel = $('sel-reps').value;
  opt = new Optimizer(backend, n, solution(n), { replicas: sel === 'auto' ? undefined : +sel });
  running = true;
  $('btn-opt').textContent = 'Stop';
  $('btn-opt').classList.remove('primary');
  $('op-reps').textContent = opt.R;
  const myN = n, me = opt;
  let lastDraw = 0;
  while (running && opt === me && n === myN) {
    const improved = await me.step();
    if (!running || n !== myN) break;
    if (improved) {
      saveLocal(myN, improved);
      showStats(improved);
      drawChart();
    }
    const tp = me.throughput();
    $('op-rate').textContent = `${(tp.pairsPerSec / 1e9).toFixed(2)} G pair/s`;
    $('op-batches').textContent = tp.batches;
    $('op-imp').textContent = me.stats.improvements;
    const now = performance.now();
    if (now - lastDraw > 60) {
      lastDraw = now;
      if ($('chk-live').checked && me.lastView) {
        setContainer(me.lastView.L);
        drawSquares(me.lastView.pos, myN, me.lastView.grad);
        $('hud-s').textContent = `${fmt(me.lastView.L, 6)} (live)`;
      } else {
        const sol = solution(myN);
        setContainer(sol.s); drawSquares(sol.squares, myN); $('hud-s').textContent = fmt(sol.s, 9);
      }
    }
    if (backend.kind === 'cpu') await new Promise((r) => setTimeout(r, 0));
  }
  const sol = solution(myN);
  if (n === myN) { setContainer(sol.s); drawSquares(sol.squares, myN); showStats(sol); }
}

$('btn-opt').onclick = () => (running ? stopOpt() : startOpt());
$('chk-live').onchange = () => { if (!running) selectN(n); };

// ---------------------------------------------------------------- export

function download(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$('btn-csv-n').onclick = () => download(`squares_n${n}.csv`, [CSV_HEADER, ...csvRows(n, solution(n).squares)].join('\n') + '\n');
$('btn-csv-all').onclick = async () => {
  const btn = $('btn-csv-all'); btn.disabled = true;
  const lines = [CSV_HEADER];
  let id = 0;
  for (let k = 1; k <= N_MAX; k++) {
    const rows = csvRows(k, solution(k).squares, id);
    id += rows.length; lines.push(...rows);
    if (k % 50 === 0) { btn.textContent = `Building… ${k}/${N_MAX}`; await new Promise((r) => setTimeout(r, 0)); }
  }
  download('submission.csv', lines.join('\n') + '\n');
  btn.textContent = 'Full submission'; btn.disabled = false;
};
$('btn-reset').onclick = () => {
  local = {};
  try { localStorage.removeItem(LS_KEY); } catch { /* ignore */ }
  cache.clear(); quick = null; selectN(n);
};

// ---------------------------------------------------------------- chart

let quick = null;
const canvas = $('chart'), ctx = canvas.getContext('2d');
const YMAX = 0.45;
function drawChart() {
  quick ??= Array.from({ length: N_MAX + 1 }, (_, k) => (k ? quickS(k) : 0));
  if (cache.has(n)) quick[n] = cache.get(n).s;
  const dpr = devicePixelRatio || 1, W = canvas.clientWidth, Hh = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = Hh * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, Hh);
  const pad = { l: 36, r: 8, t: 8, b: 18 };
  const X = (k) => pad.l + ((k - 1) / (N_MAX - 1)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - Math.min(v, YMAX) / YMAX) * (Hh - pad.t - pad.b);
  ctx.font = '10px JetBrains Mono, monospace'; ctx.fillStyle = css('--muted'); ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
  for (const v of [0, 0.1, 0.2, 0.3, 0.4]) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(W - pad.r, Y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(1), 4, Y(v) + 3);
  }
  for (const k of [1, 200, 400, 600, 800, 1000]) ctx.fillText(k, Math.min(X(k) - 6, W - pad.r - ctx.measureText(k).width), Hh - 4);
  const series = (fn, color, w) => {
    ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath();
    for (let k = 1; k <= N_MAX; k++) { const v = fn(k); if (v === null) continue; k === 1 ? ctx.moveTo(X(k), Y(v)) : ctx.lineTo(X(k), Y(v)); }
    ctx.stroke();
  };
  series((k) => Math.ceil(Math.sqrt(k)) / Math.sqrt(k) - 1, css('--grid'), 1);
  series((k) => quick[k] / Math.sqrt(k) - 1, css('--cur'), 1.5);
  ctx.fillStyle = css('--rec');
  for (let k = 1; k <= recordMax; k++) { const r = recordFor(k); if (r) ctx.fillRect(X(k) - 1, Y(r / Math.sqrt(k) - 1) - 1, 2, 2); }
  ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(X(n), pad.t); ctx.lineTo(X(n), Hh - pad.b); ctx.stroke(); ctx.setLineDash([]);
  let total = 0;
  for (let k = 1; k <= N_MAX; k++) total += quick[k] / Math.sqrt(k) - 1;
  $('total').textContent = `Σ (s/√n − 1) over n = 1…1000: ${total.toFixed(4)} (trivial grid: 28.9377)`;
  canvas._map = (px) => Math.round(1 + ((px - pad.l) / (W - pad.l - pad.r)) * (N_MAX - 1));
}
canvas.addEventListener('click', (e) => selectN(canvas._map(e.offsetX)));
canvas.addEventListener('mousemove', (e) => {
  const k = Math.min(N_MAX, Math.max(1, canvas._map(e.offsetX))), tip = $('tip'), r = recordFor(k);
  tip.hidden = false;
  tip.textContent = `n=${k}  s=${quick[k].toFixed(5)}${r ? `  best=${r.toFixed(5)}` : ''}`;
  tip.style.left = `${Math.min(e.offsetX + 24, canvas.clientWidth - 220)}px`; tip.style.top = `${e.offsetY - 10}px`;
});
canvas.addEventListener('mouseleave', () => { $('tip').hidden = true; });
addEventListener('resize', drawChart);

selectN(n);
