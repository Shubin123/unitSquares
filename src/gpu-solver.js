// WebGPU backend: R independent replicas of an n-square packing relaxed in
// parallel. One invocation per square, workgroup-shared tiles of 64 neighbours,
// analytic separating-axis gradients (positions and angles), heavy-ball
// momentum, and per-replica Langevin noise for parallel tempering.

const WG = 64;

const COMMON = /* wgsl */`
struct Params { n: u32, R: u32, lrPos: f32, lrAng: f32, mom: f32, wall: f32, maxStep: f32, pad: f32 };
@group(0) @binding(0) var<uniform> P: Params;
`;

const GRAD_WGSL = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> grad: array<vec4f>;
@group(0) @binding(3) var<storage, read> reps: array<vec2f>;
var<workgroup> tile: array<vec4f, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let n = P.n;
  let base = wid.y * n;
  let i = wid.x * ${WG}u + lid.x;
  let valid = i < n;
  var me = vec4f(0.0);
  if (valid) { me = pos[base + i]; }
  let ca = cos(me.z); let sa = sin(me.z);
  var g = vec3f(0.0);
  var e = 0.0;
  let ntiles = (n + ${WG - 1}u) / ${WG}u;
  for (var t = 0u; t < ntiles; t++) {
    let j0 = t * ${WG}u;
    if (j0 + lid.x < n) { tile[lid.x] = pos[base + j0 + lid.x]; }
    workgroupBarrier();
    let cnt = min(${WG}u, n - j0);
    for (var q = 0u; q < cnt; q++) {
      let o = tile[q];
      let d = o.xy - me.xy;
      if (j0 + q == i || dot(d, d) >= 2.0) { continue; }
      let cb = cos(o.z); let sb = sin(o.z);
      let dl = o.z - me.z;
      let cd = cos(dl); let sd = sin(dl);
      let k = 0.5 * (abs(cd) + abs(sd));
      let dk = -0.5 * (sign(cd) * -sd + sign(sd) * cd);     // dk/d(theta_i)
      let p0 = ca * d.x + sa * d.y;
      let p1 = -sa * d.x + ca * d.y;
      let p2 = cb * d.x + sb * d.y;
      let p3 = -sb * d.x + cb * d.y;
      var m = abs(p0); var ax = vec2f(ca, sa); var s = sign(p0); var dp = p1;
      if (abs(p1) > m) { m = abs(p1); ax = vec2f(-sa, ca); s = sign(p1); dp = -p0; }
      if (abs(p2) > m) { m = abs(p2); ax = vec2f(cb, sb); s = sign(p2); dp = 0.0; }
      if (abs(p3) > m) { m = abs(p3); ax = vec2f(-sb, cb); s = sign(p3); dp = 0.0; }
      let ov = 0.5 + k - m;
      if (ov <= 0.0) { continue; }
      e += ov * ov;
      g += 2.0 * ov * vec3f(s * ax, dk - s * dp);
    }
    workgroupBarrier();
  }
  if (!valid) { return; }
  let L = reps[wid.y].x;
  let c = cos(me.z); let sn = sin(me.z);
  let kk = 0.5 * (abs(c) + abs(sn));
  let dkk = 0.5 * (sign(c) * -sn + sign(sn) * c);
  let lo = max(vec2f(0.0), vec2f(kk) - me.xy);
  let hi = max(vec2f(0.0), me.xy + vec2f(kk - L));
  e += P.wall * (dot(lo, lo) + dot(hi, hi));
  g += P.wall * 2.0 * vec3f(hi - lo, (lo.x + lo.y + hi.x + hi.y) * dkk);
  grad[base + i] = vec4f(g, e);
}`;

const STEP_WGSL = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> grad: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> rng: array<u32>;
@group(0) @binding(5) var<storage, read> reps: array<vec2f>;

fn pcg(s: ptr<function, u32>) -> f32 {
  let st = *s * 747796405u + 2891336453u;
  *s = st;
  var w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
  w = (w >> 22u) ^ w;
  return (f32(w >> 8u) + 0.5) / 16777216.0;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= P.n * P.R) { return; }
  let T = reps[idx / P.n].y;
  var p = pos[idx];
  let g = grad[idx];
  var v = vel[idx];
  v = vec4f(clamp(P.mom * v.xyz - vec3f(P.lrPos * g.xy, P.lrAng * g.z), vec3f(-P.maxStep), vec3f(P.maxStep)), 0.0);
  p = vec4f(p.xyz + v.xyz, p.w);
  if (T > 0.0) {
    var s = rng[idx];
    let r1 = sqrt(-2.0 * log(pcg(&s))); let a1 = 6.2831853 * pcg(&s);
    let r2 = sqrt(-2.0 * log(pcg(&s))); let a2 = 6.2831853 * pcg(&s);
    p = vec4f(p.xyz + T * vec3f(r1 * cos(a1), r1 * sin(a1), r2 * cos(a2)), p.w);
    rng[idx] = s;
  }
  pos[idx] = p;
  vel[idx] = v;
}`;

export async function createGPUBackend() {
  if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();
  const info = adapter.info || {};
  const name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' ') || 'WebGPU device';
  const gradPipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: GRAD_WGSL }), entryPoint: 'main' } });
  const stepPipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: STEP_WGSL }), entryPoint: 'main' } });

  let n = 0, R = 0, buf = null, gradBG, stepBG;
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

  return {
    kind: 'webgpu', name,
    init(n_, R_) {
      if (buf) Object.values(buf).forEach((b) => b.destroy());
      n = n_; R = R_;
      const N = n * R;
      buf = {
        params: device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | CD }),
        pos: device.createBuffer({ size: N * 16, usage: S | CD | CS }),
        grad: device.createBuffer({ size: N * 16, usage: S | CS }),
        vel: device.createBuffer({ size: N * 16, usage: S | CD }),
        rng: device.createBuffer({ size: N * 4, usage: S | CD }),
        reps: device.createBuffer({ size: Math.max(16, R * 8), usage: S | CD }),
        readPos: device.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | CD }),
        readGrad: device.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | CD }),
      };
      const seeds = new Uint32Array(N);
      for (let o = 0; o < N; o += 16384) crypto.getRandomValues(seeds.subarray(o, o + 16384));
      device.queue.writeBuffer(buf.rng, 0, seeds);
      const bind = (pipe, list) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: list.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
      gradBG = bind(gradPipe, [buf.params, buf.pos, buf.grad, buf.reps]);
      stepBG = bind(stepPipe, [buf.params, buf.pos, buf.grad, buf.vel, buf.rng, buf.reps]);
    },
    setParams({ lrPos, lrAng, mom, wall, maxStep }) {
      const ab = new ArrayBuffer(32), u = new Uint32Array(ab), f = new Float32Array(ab);
      u[0] = n; u[1] = R; f[2] = lrPos; f[3] = lrAng; f[4] = mom; f[5] = wall; f[6] = maxStep;
      device.queue.writeBuffer(buf.params, 0, ab);
    },
    writeReplica(r, pos4) {
      device.queue.writeBuffer(buf.pos, r * n * 16, pos4);
      device.queue.writeBuffer(buf.vel, r * n * 16, new Float32Array(n * 4));
    },
    setReps(reps2) { device.queue.writeBuffer(buf.reps, 0, reps2); },
    run(iters) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      const gx = Math.ceil(n / WG), sx = Math.ceil((n * R) / WG);
      for (let k = 0; k < iters; k++) {
        pass.setPipeline(gradPipe); pass.setBindGroup(0, gradBG); pass.dispatchWorkgroups(gx, R);
        pass.setPipeline(stepPipe); pass.setBindGroup(0, stepBG); pass.dispatchWorkgroups(sx);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
    },
    async read() {
      const enc = device.createCommandEncoder();
      // A final gradient pass so energies match the positions we read back.
      const pass = enc.beginComputePass();
      pass.setPipeline(gradPipe); pass.setBindGroup(0, gradBG); pass.dispatchWorkgroups(Math.ceil(n / WG), R);
      pass.end();
      enc.copyBufferToBuffer(buf.pos, 0, buf.readPos, 0, n * R * 16);
      enc.copyBufferToBuffer(buf.grad, 0, buf.readGrad, 0, n * R * 16);
      device.queue.submit([enc.finish()]);
      await Promise.all([buf.readPos.mapAsync(GPUMapMode.READ), buf.readGrad.mapAsync(GPUMapMode.READ)]);
      const pos = new Float32Array(buf.readPos.getMappedRange().slice(0));
      const grad = new Float32Array(buf.readGrad.getMappedRange().slice(0));
      buf.readPos.unmap(); buf.readGrad.unmap();
      return { pos, grad };
    },
    destroy() { if (buf) Object.values(buf).forEach((b) => b.destroy()); buf = null; },
  };
}
