// Exact geometry for packing unit squares in a square.
// A square is {x, y, t}: centre and rotation in radians. All math here is f64.

export const SQRT2 = Math.SQRT2;
const HALF_DIAG2 = 2; // (sum of circumradii)^2 = (2 * sqrt(2)/2)^2

// Half-extent of a rotated unit square along either axis: 0.5(|cos t| + |sin t|).
export const halfExtent = (t) => 0.5 * (Math.abs(Math.cos(t)) + Math.abs(Math.sin(t)));

// Separating-axis penetration depth of two unit squares (> 0 means overlap).
// For unit squares the projected radii on all four candidate axes collapse to
// 0.5 + k(dt), so the depth is 0.5 + k - max |axis . d|.
export function penetration(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  if (dx * dx + dy * dy >= HALF_DIAG2) return -1;
  const ca = Math.cos(a.t), sa = Math.sin(a.t), cb = Math.cos(b.t), sb = Math.sin(b.t);
  const dt = b.t - a.t;
  const k = 0.5 * (Math.abs(Math.cos(dt)) + Math.abs(Math.sin(dt)));
  const m = Math.max(
    Math.abs(ca * dx + sa * dy), Math.abs(-sa * dx + ca * dy),
    Math.abs(cb * dx + sb * dy), Math.abs(-sb * dx + cb * dy));
  return 0.5 + k - m;
}

export function bbox(squares) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const q of squares) {
    const k = halfExtent(q.t);
    x0 = Math.min(x0, q.x - k); y0 = Math.min(y0, q.y - k);
    x1 = Math.max(x1, q.x + k); y1 = Math.max(y1, q.y + k);
  }
  return { x0, y0, x1, y1, s: Math.max(x1 - x0, y1 - y0) };
}

// Candidate pairs whose circumcircles may touch, via a uniform grid hash.
export function nearPairs(squares, reach = Math.sqrt(HALF_DIAG2) + 0.05) {
  const cell = reach, grid = new Map(), pairs = [];
  const key = (i, j) => i * 73856093 ^ j * 19349663;
  squares.forEach((q, idx) => {
    const ci = Math.floor(q.x / cell), cj = Math.floor(q.y / cell);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      const bucket = grid.get(key(ci + di, cj + dj));
      if (!bucket) continue;
      for (const o of bucket) {
        const p = squares[o], dx = p.x - q.x, dy = p.y - q.y;
        if (dx * dx + dy * dy < reach * reach) pairs.push(o, idx);
      }
    }
    const k0 = key(ci, cj);
    if (!grid.has(k0)) grid.set(k0, []);
    grid.get(k0).push(idx);
  });
  return pairs;
}

export function maxPenetration(squares, pairs = nearPairs(squares)) {
  let worst = -Infinity, at = null;
  for (let p = 0; p < pairs.length; p += 2) {
    const d = penetration(squares[pairs[p]], squares[pairs[p + 1]]);
    if (d > worst) { worst = d; at = [pairs[p], pairs[p + 1]]; }
  }
  return { worst, at };
}

// Strict validity: every pair separated by at least `margin`.
export function validate(squares, margin = 1e-10) {
  for (const q of squares) if (![q.x, q.y, q.t].every(Number.isFinite)) return { ok: false, reason: 'non-finite' };
  const { worst, at } = maxPenetration(squares);
  if (worst > -margin && at) return { ok: false, reason: `squares ${at[0]} and ${at[1]} overlap by ${worst.toExponential(2)}` };
  return { ok: true, s: bbox(squares).s };
}

// ---------------------------------------------------------------- constructions

export function gridPacking(n) {
  const m = Math.ceil(Math.sqrt(n));
  const squares = [];
  for (let k = 0; k < n; k++) squares.push({ x: (k % m) + 0.5, y: Math.floor(k / m) + 0.5, t: 0 });
  return { s: m, squares, kind: 'grid', label: `${m}×${m} grid` };
}

// Two unrotated "staircases" with `a` steps in opposite corners.
function staircase(a, s, corners) {
  const out = [];
  for (let i = 0; i < a; i++) for (let j = 0; i + j <= a - 1; j++) for (const [fx, fy] of corners)
    out.push({ x: fx ? s - i - 0.5 : i + 0.5, y: fy ? s - j - 0.5 : j + 0.5, t: 0 });
  return out;
}

// Göbel strip: n = a(a+1) + 2 + b, s = a + 1 + √2/2, b = 1 + ⌊(a-1)√2⌋.
// A 45° strip of b squares runs along the anti-diagonal between the corner
// squares at (0, s) and (s, 0), sandwiched by two a-step staircases.
export function gobelStrip(a) {
  const b = 1 + Math.floor((a - 1) * SQRT2);
  const s = a + 1 + SQRT2 / 2;
  const squares = staircase(a, s, [[0, 0], [1, 1]]);
  squares.push({ x: 0.5, y: s - 0.5, t: 0 }, { x: s - 0.5, y: 0.5, t: 0 });
  const len = (s - 2) * SQRT2, slack = (len - b) / 2, d = 1 / SQRT2;
  for (let k = 0; k < b; k++) {
    const u = slack + k + 0.5;
    squares.push({ x: 1 + u * d, y: s - 1 - u * d, t: Math.PI / 4 });
  }
  return { s, squares, kind: 'gobel-strip', label: `Göbel strip a=${a}, b=${b}` };
}

// Göbel square: n = 2a(a+1) + b², s = a + 1 + b√2/2, needs a-1 < b√2/2 < a+1.
// A b×b block tilted 45° in the centre, four a-step staircases in the corners.
export function gobelSquare(a, b) {
  const h = b * SQRT2 / 2;
  if (!(a - 1 < h && h < a + 1)) return null;
  const s = a + 1 + h, c = s / 2;
  const squares = staircase(a, s, [[0, 0], [1, 0], [0, 1], [1, 1]]);
  for (let p = 0; p < b; p++) for (let q = 0; q < b; q++) {
    const lx = p - (b - 1) / 2, ly = q - (b - 1) / 2;
    squares.push({ x: c + (lx - ly) / SQRT2, y: c + (lx + ly) / SQRT2, t: Math.PI / 4 });
  }
  return { s, squares, kind: 'gobel-square', label: `Göbel square a=${a}, b=${b}` };
}

// Wrap a packing of side s in k L-shaped borders of unrotated squares:
// each border adds ⌊s⌋ + ⌊s+1⌋ squares and grows the side by exactly 1.
export function withBorders(base, k) {
  if (k === 0) return base;
  let { s } = base;
  const squares = base.squares.map((q) => ({ x: q.x + k, y: q.y + k, t: q.t }));
  for (let r = k - 1; r >= 0; r--) {
    s += 1;
    for (let i = 0; i < Math.floor(s + 1e-9); i++) squares.push({ x: r + i + 0.5, y: r + 0.5, t: 0 });
    for (let j = 1; j < Math.floor(s + 1e-9); j++) squares.push({ x: r + 0.5, y: r + j + 0.5, t: 0 });
  }
  return { s, squares, kind: base.kind, label: `${base.label} + ${k} border${k > 1 ? 's' : ''}` };
}

// Every construction family indexed by capacity; s(n) is monotone so a packing
// for n' ≥ n gives one for n by deleting squares.
let catalog = null;
function buildCatalog(maxN) {
  const bases = [];
  for (let a = 1; a * (a + 1) < maxN; a++) {
    const b = 1 + Math.floor((a - 1) * SQRT2);
    bases.push({ count: a * (a + 1) + 2 + b, s: a + 1 + SQRT2 / 2, make: () => gobelStrip(a) });
  }
  for (let a = 0; 2 * a * (a + 1) < maxN; a++) for (let b = 1; b < 2 * a + 4; b++) {
    const h = b * SQRT2 / 2;
    if (a - 1 < h && h < a + 1) bases.push({ count: 2 * a * (a + 1) + b * b, s: a + 1 + h, make: () => gobelSquare(a, b) });
  }
  const list = [];
  for (const base of bases) {
    let { count, s } = base;
    for (let k = 0; count < maxN * 1.1; k++) {
      list.push({ count, s, make: () => withBorders(base.make(), k) });
      count += Math.floor(s + 1 + 1e-9) + Math.floor(s + 1e-9); s += 1;
    }
  }
  return list.sort((p, q) => p.s - q.s);
}

export function construct(n) {
  catalog ??= buildCatalog(1100);
  let best = gridPacking(n);
  for (const c of catalog) {
    if (c.s >= best.s - 1e-12) break;
    if (c.count >= n) { best = c.make(); break; }
  }
  return { ...best, squares: best.squares.slice(0, n) };
}

// ---------------------------------------------------------------- finishing

// f64 penalty descent that removes residual overlaps left by the f32 GPU search.
export function polish(squares, L, iters = 400) {
  const q = squares.map((p) => ({ ...p }));
  const pairs = nearPairs(q, 1.6);
  const g = new Float64Array(q.length * 3);
  for (let it = 0; it < iters; it++) {
    g.fill(0);
    let worst = 0;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p], j = pairs[p + 1], a = q[i], b = q[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const ca = Math.cos(a.t), sa = Math.sin(a.t), cb = Math.cos(b.t), sb = Math.sin(b.t);
      const dt = b.t - a.t, cd = Math.cos(dt), sd = Math.sin(dt);
      const k = 0.5 * (Math.abs(cd) + Math.abs(sd));
      const dk = 0.5 * (Math.sign(cd) * -sd + Math.sign(sd) * cd); // dk/d(dt)
      const pr = [ca * dx + sa * dy, -sa * dx + ca * dy, cb * dx + sb * dy, -sb * dx + cb * dy];
      let w = 0;
      for (let z = 1; z < 4; z++) if (Math.abs(pr[z]) > Math.abs(pr[w])) w = z;
      const ov = 0.5 + k - Math.abs(pr[w]) + 1e-9; // aim for a hair of clearance
      if (ov <= 0) continue;
      worst = Math.max(worst, ov);
      const sg = Math.sign(pr[w]);
      const ax = [[ca, sa], [-sa, ca], [cb, sb], [-sb, cb]][w];
      // d(ov)/d(a.t) and d(ov)/d(b.t)
      const dpa = w === 0 ? pr[1] : w === 1 ? -pr[0] : 0;
      const dpb = w === 2 ? (-sb * dx + cb * dy) : w === 3 ? -(cb * dx + sb * dy) : 0;
      g[3 * i] += 2 * ov * sg * ax[0]; g[3 * i + 1] += 2 * ov * sg * ax[1];
      g[3 * j] -= 2 * ov * sg * ax[0]; g[3 * j + 1] -= 2 * ov * sg * ax[1];
      g[3 * i + 2] += 2 * ov * (-dk - sg * dpa);
      g[3 * j + 2] += 2 * ov * (dk - sg * dpb);
    }
    for (let i = 0; i < q.length; i++) {
      const p = q[i], k = halfExtent(p.t);
      const dk = 0.5 * (Math.sign(Math.cos(p.t)) * -Math.sin(p.t) + Math.sign(Math.sin(p.t)) * Math.cos(p.t));
      for (const [c, gi] of [['x', 0], ['y', 1]]) {
        const lo = Math.max(0, k - p[c]), hi = Math.max(0, p[c] + k - L);
        g[3 * i + gi] += 2 * (hi - lo);
        g[3 * i + 2] += 2 * (lo + hi) * dk;
      }
    }
    if (worst < 1e-12) break;
    for (let i = 0; i < q.length; i++) {
      q[i].x -= 0.1 * g[3 * i]; q[i].y -= 0.1 * g[3 * i + 1]; q[i].t -= 0.1 * g[3 * i + 2];
    }
  }
  return q;
}

// Scale centres about the centroid by the smallest factor that leaves every pair
// strictly separated, then translate the bounding box to the origin.
export function finalize(squares, margin = 1e-10) {
  const n = squares.length;
  if (n === 0) return [];
  const cx = squares.reduce((s, q) => s + q.x, 0) / n, cy = squares.reduce((s, q) => s + q.y, 0) / n;
  const pairs = nearPairs(squares, 1.6);
  const scaled = (e) => squares.map((q) => ({ x: cx + (q.x - cx) * (1 + e), y: cy + (q.y - cy) * (1 + e), t: q.t }));
  const okAt = (e) => maxPenetration(scaled(e), pairs).worst <= -margin;
  let hi = 1e-9;
  while (!okAt(hi)) { hi *= 4; if (hi > 0.05) return null; }
  let lo = hi > 1e-9 ? hi / 4 : 0;
  if (hi > 1e-9) for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; okAt(mid) ? (hi = mid) : (lo = mid); }
  const out = scaled(hi);
  const bb = bbox(out);
  for (const q of out) { q.x -= bb.x0; q.y -= bb.y0; }
  return validate(out, margin * 0.5).ok ? out : null;
}

// f64 squeeze: shrink the container around a valid packing, re-polish, and keep
// whatever still finalizes, doubling or halving the step like a line search.
export function tighten(sol, { rounds = 24, iters = 1500 } = {}) {
  let best = sol, d = 1e-4 * sol.s;
  for (let r = 0; r < rounds && d > 1e-10; r++) {
    const L = best.s - d, sc = L / best.s;
    const out = finalize(polish(best.squares.map((q) => ({ x: q.x * sc, y: q.y * sc, t: q.t })), L, iters));
    const s = out && bbox(out).s;
    if (out && s < best.s - 1e-12) { best = { ...best, s, squares: out }; d *= 2; } else d /= 4;
  }
  return best;
}

// ---------------------------------------------------------------- submission

export const CSV_HEADER = 'id,n,square_id,x,y,angle';
export function csvRows(n, squares, startId = 0) {
  return squares.map((q, i) => `${startId + i},${n},${i},${q.x.toPrecision(17)},${q.y.toPrecision(17)},${q.t.toPrecision(17)}`);
}
