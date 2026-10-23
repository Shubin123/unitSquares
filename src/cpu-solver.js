// CPU fallback with the same interface and math as the WebGPU backend.

export function createCPUBackend() {
  let n = 0, R = 0, pos, grad, vel, reps, P = {};
  const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

  function computeGrad() {
    for (let r = 0; r < R; r++) {
      const base = r * n, L = reps[2 * r];
      for (let i = 0; i < n; i++) {
        const I = 4 * (base + i), x = pos[I], y = pos[I + 1], t = pos[I + 2];
        const ca = Math.cos(t), sa = Math.sin(t);
        let gx = 0, gy = 0, gt = 0, e = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const J = 4 * (base + j), dx = pos[J] - x, dy = pos[J + 1] - y;
          if (dx * dx + dy * dy >= 2) continue;
          const cb = Math.cos(pos[J + 2]), sb = Math.sin(pos[J + 2]);
          const dl = pos[J + 2] - t, cd = Math.cos(dl), sd = Math.sin(dl);
          const k = 0.5 * (Math.abs(cd) + Math.abs(sd));
          const dk = -0.5 * (Math.sign(cd) * -sd + Math.sign(sd) * cd);
          const p0 = ca * dx + sa * dy, p1 = -sa * dx + ca * dy, p2 = cb * dx + sb * dy, p3 = -sb * dx + cb * dy;
          let m = Math.abs(p0), ax = ca, ay = sa, s = Math.sign(p0), dp = p1;
          if (Math.abs(p1) > m) { m = Math.abs(p1); ax = -sa; ay = ca; s = Math.sign(p1); dp = -p0; }
          if (Math.abs(p2) > m) { m = Math.abs(p2); ax = cb; ay = sb; s = Math.sign(p2); dp = 0; }
          if (Math.abs(p3) > m) { m = Math.abs(p3); ax = -sb; ay = cb; s = Math.sign(p3); dp = 0; }
          const ov = 0.5 + k - m;
          if (ov <= 0) continue;
          e += ov * ov;
          gx += 2 * ov * s * ax; gy += 2 * ov * s * ay; gt += 2 * ov * (dk - s * dp);
        }
        const kk = 0.5 * (Math.abs(ca) + Math.abs(sa));
        const dkk = 0.5 * (Math.sign(ca) * -sa + Math.sign(sa) * ca);
        const lx = Math.max(0, kk - x), ly = Math.max(0, kk - y), hx = Math.max(0, x + kk - L), hy = Math.max(0, y + kk - L);
        e += P.wall * (lx * lx + ly * ly + hx * hx + hy * hy);
        gx += P.wall * 2 * (hx - lx); gy += P.wall * 2 * (hy - ly); gt += P.wall * 2 * (lx + ly + hx + hy) * dkk;
        grad[I] = gx; grad[I + 1] = gy; grad[I + 2] = gt; grad[I + 3] = e;
      }
    }
  }

  return {
    kind: 'cpu', name: 'CPU (JavaScript, f64)',
    init(n_, R_) {
      n = n_; R = R_;
      pos = new Float64Array(n * R * 4); grad = new Float64Array(n * R * 4); vel = new Float64Array(n * R * 4);
      reps = new Float64Array(R * 2);
    },
    setParams(p) { P = p; },
    writeReplica(r, pos4) { pos.set(pos4, r * n * 4); vel.fill(0, r * n * 4, (r + 1) * n * 4); },
    setReps(r2) { reps.set(r2); },
    run(iters) {
      for (let k = 0; k < iters; k++) {
        computeGrad();
        for (let idx = 0; idx < n * R; idx++) {
          const T = reps[2 * Math.floor(idx / n) + 1], I = 4 * idx;
          for (let c = 0; c < 3; c++) {
            const lr = c < 2 ? P.lrPos : P.lrAng;
            vel[I + c] = Math.min(P.maxStep, Math.max(-P.maxStep, P.mom * vel[I + c] - lr * grad[I + c]));
            pos[I + c] += vel[I + c] + (T > 0 ? T * gauss() : 0);
          }
        }
      }
    },
    async read() { computeGrad(); return { pos: Float32Array.from(pos), grad: Float32Array.from(grad) }; },
    destroy() {},
  };
}
