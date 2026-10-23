// Host-side controller: adaptive container shrinking over a population of
// replicas (parallel tempering + restarts from the incumbent), with every
// improvement re-verified in f64 before it is accepted.

import { polish, finalize, tighten } from './geometry.js';

const TOL_E = 4e-8;   // per-square penalty below which an f32 state is a candidate
const PATIENCE = 6;   // batches without feasibility before backing off
const MIN_STEP = 2e-5;

export function defaultReplicas(backend, n) {
  if (backend.kind === 'cpu') return Math.max(1, Math.min(8, Math.round(2e3 / (n * n))));
  return Math.max(4, Math.min(2048, Math.round(6e6 / (n * n))));
}

export class Optimizer {
  constructor(backend, n, best, { replicas, hot = 48, cold = 48 } = {}) {
    this.b = backend; this.n = n; this.R = replicas ?? defaultReplicas(backend, n);
    this.best = best; this.hot = hot; this.cold = cold;
    this.params = { lrPos: 0.04, lrAng: 0.04, mom: 0.9, wall: 4, maxStep: 0.05 };
    this.b.init(n, this.R);
    this.b.setParams(this.params);
    this.reps = [];
    this.host = new Float32Array(n * this.R * 4);
    for (let r = 0; r < this.R; r++) {
      this.reps.push({ T0: 0.006 * 0.5 ** (r % 7), L: best.s, hi: null, step: 0.01, fails: 0 });
      this.seed(r, r % 4 === 3 ? 'random' : 'best');
    }
    this.stats = { batches: 0, pairEvals: 0, t0: performance.now(), improvements: 0, feasible: 0 };
    this.lastView = null;
  }

  seed(r, mode) {
    const { n, best } = this, rep = this.reps[r];
    rep.step = 0.002 + 0.02 * Math.random();
    rep.hi = null; rep.fails = 0;
    rep.L = best.s - rep.step;
    const p = new Float32Array(n * 4);
    if (mode === 'random') {
      for (let i = 0; i < n; i++) {
        p[4 * i] = 0.5 + Math.random() * (rep.L - 1); p[4 * i + 1] = 0.5 + Math.random() * (rep.L - 1);
        p[4 * i + 2] = Math.random() * Math.PI / 2;
      }
    } else {
      const sc = rep.L / best.s, sig = 0.03 * Math.random();
      best.squares.forEach((q, i) => {
        p[4 * i] = q.x * sc + sig * gauss(); p[4 * i + 1] = q.y * sc + sig * gauss(); p[4 * i + 2] = q.t + sig * gauss();
      });
      // Kick: relocate a few squares to random spots so the search can change topology.
      const kicks = Math.floor(Math.random() * Math.min(4, n));
      for (let k = 0; k < kicks; k++) {
        const i = Math.floor(Math.random() * n);
        p[4 * i] = 0.5 + Math.random() * (rep.L - 1); p[4 * i + 1] = 0.5 + Math.random() * (rep.L - 1);
        p[4 * i + 2] = Math.random() * Math.PI / 2;
      }
    }
    this.host.set(p, r * n * 4);
    this.b.writeReplica(r, p);
  }

  rescale(r, newL) {
    const { n } = this, rep = this.reps[r], sc = newL / rep.L;
    const p = this.host.slice(r * n * 4, (r + 1) * n * 4);
    for (let i = 0; i < n; i++) { p[4 * i] *= sc; p[4 * i + 1] *= sc; }
    rep.L = newL;
    this.host.set(p, r * n * 4);
    this.b.writeReplica(r, p);
  }

  repArray(hot) {
    const a = new Float32Array(this.R * 2);
    this.reps.forEach((rep, r) => { a[2 * r] = rep.L; a[2 * r + 1] = hot ? rep.T0 : 0; });
    return a;
  }

  // One batch: hot Langevin phase, cold quench, read back, adapt every replica.
  async step() {
    const { n, R } = this;
    this.b.setReps(this.repArray(true)); this.b.run(this.hot);
    this.b.setReps(this.repArray(false)); this.b.run(this.cold);
    const { pos, grad } = await this.b.read();
    this.host = pos;
    this.stats.batches++;
    this.stats.pairEvals += R * n * n * (this.hot + this.cold + 1);

    let improved = null, viewR = 0, viewScore = Infinity;
    for (let r = 0; r < R; r++) {
      const rep = this.reps[r];
      let maxE = 0, sumE = 0;
      for (let i = 0; i < n; i++) { const e = grad[4 * (r * n + i) + 3]; sumE += e; if (e > maxE) maxE = e; }
      const score = rep.L + 10 * sumE;
      if (score < viewScore) { viewScore = score; viewR = r; }
      if (maxE < TOL_E) {
        this.stats.feasible++;
        if (rep.L < this.best.s - 1e-9) {
          const cand = this.verify(r);
          if (cand && cand.s < this.best.s - 1e-9) { this.best = cand; improved = cand; this.stats.improvements++; }
        }
        rep.hi = rep.L; rep.fails = 0;
        rep.step = Math.min(rep.step * 1.3, 0.05);
        this.rescale(r, Math.min(rep.L, this.best.s) - rep.step);
      } else if (++rep.fails >= PATIENCE) {
        rep.fails = 0;
        rep.step *= 0.5;
        if (rep.hi === null || rep.step < MIN_STEP || rep.hi > this.best.s + 0.05) this.seed(r, Math.random() < 0.75 ? 'best' : 'random');
        else this.rescale(r, Math.min(rep.hi, this.best.s) - rep.step);
      }
    }
    this.lastView = { r: viewR, L: this.reps[viewR].L, pos: pos.subarray(viewR * n * 4, (viewR + 1) * n * 4), grad: grad.subarray(viewR * n * 4, (viewR + 1) * n * 4) };
    return improved;
  }

  verify(r) {
    const { n } = this, p = this.host, L = this.reps[r].L;
    const sq = [];
    for (let i = 0; i < n; i++) { const I = 4 * (r * n + i); sq.push({ x: p[I], y: p[I + 1], t: p[I + 2] }); }
    const out = finalize(polish(sq, L));
    if (!out) return null;
    let s = 0;
    for (const q of out) { const k = 0.5 * (Math.abs(Math.cos(q.t)) + Math.abs(Math.sin(q.t))); s = Math.max(s, q.x + k, q.y + k); }
    const sol = { s, squares: out, kind: 'optimized', label: `GPU search (${this.b.kind})` };
    return n <= 400 ? tighten(sol) : sol;
  }

  throughput() {
    const dt = (performance.now() - this.stats.t0) / 1000;
    return { pairsPerSec: this.stats.pairEvals / dt, batches: this.stats.batches, elapsed: dt };
  }

  destroy() { this.b.destroy(); }
}

function gauss() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()); }
