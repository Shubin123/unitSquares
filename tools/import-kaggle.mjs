// Merge public Kaggle notebook outputs into per-puzzle best packings.
// Every packing is re-verified with the strict f64 SAT check (and scaled out by
// the minimum needed for a 1e-10 clearance) before it can win a puzzle.
//
//   curl -sL -o out/<user>_<slug>.csv \
//     https://www.kaggle.com/api/v1/kernels/output/download/<user>/<slug>/submission.csv
//   node tools/import-kaggle.mjs out/*.csv
//
// Writes data/kaggle/index.json ({n: {s, src}}) and data/kaggle/<n>.bin
// (Float64 x, y, angle per square) for every n that beats the constructions.

import fs from 'fs';
import path from 'path';
import { construct, finalize, validate } from '../src/geometry.js';

const OUT = new URL('../data/kaggle/', import.meta.url).pathname;
const files = process.argv.slice(2);
if (!files.length) { console.error('usage: node tools/import-kaggle.mjs <submission.csv>...'); process.exit(1); }

function parse(file) {
  const byN = new Map();
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  const head = lines[0].trim().split(',');
  const col = Object.fromEntries(head.map((h, i) => [h, i]));
  for (let k = 1; k < lines.length; k++) {
    if (!lines[k]) continue;
    const f = lines[k].split(',');
    const n = +f[col.n];
    if (!byN.has(n)) byN.set(n, []);
    byN.get(n)[+f[col.square_id]] = { x: +f[col.x], y: +f[col.y], t: +f[col.angle] };
  }
  return byN;
}

const best = new Map();
for (let n = 1; n <= 1000; n++) best.set(n, { s: construct(n).s, src: null, squares: null });

for (const file of files) {
  const src = path.basename(file, '.csv').replace('_', '/');
  let wins = 0, rejected = 0;
  for (const [n, sq] of parse(file)) {
    if (n < 1 || n > 1000 || sq.length !== n || sq.some((q) => !q)) { rejected++; continue; }
    const fin = finalize(sq);
    if (!fin) { rejected++; continue; }
    const v = validate(fin);
    if (!v.ok) { rejected++; continue; }
    if (v.s < best.get(n).s - 1e-9) { best.set(n, { s: v.s, src, squares: fin }); wins++; }
  }
  console.log(`${src}: ${wins} puzzles improved, ${rejected} rejected`);
}

fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });
const index = {};
let total = 0;
for (const [n, b] of best) {
  total += b.s / Math.sqrt(n) - 1;
  if (!b.squares) continue;
  index[n] = { s: b.s, src: b.src };
  const a = new Float64Array(n * 3);
  b.squares.forEach((q, i) => { a[3 * i] = q.x; a[3 * i + 1] = q.y; a[3 * i + 2] = q.t; });
  fs.writeFileSync(path.join(OUT, `${n}.bin`), Buffer.from(a.buffer));
}
fs.writeFileSync(path.join(OUT, 'index.json'), JSON.stringify(index));
console.log(`${Object.keys(index).length} puzzles from Kaggle notebooks; merged Σ(s/√n − 1) = ${total.toFixed(9)}`);
