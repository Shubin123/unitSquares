// Build submission.csv for n = 1…1000 from the constructions plus the bundled
// GPU-search improvements, validating every puzzle in f64.
//
//   node tools/make-submission.mjs [out=submission.csv]

import fs from 'fs';
import { construct, finalize, validate, CSV_HEADER, csvRows } from '../src/geometry.js';

const out = process.argv[2] || 'submission.csv';
const improved = JSON.parse(fs.readFileSync(new URL('../data/improved.json', import.meta.url), 'utf8'));
const lines = [CSV_HEADER];
let id = 0, total = 0, fromSearch = 0;

for (let n = 1; n <= 1000; n++) {
  const c = construct(n);
  let squares = finalize(c.squares), s = validate(squares).s;
  const e = improved[n];
  if (e) {
    const cand = e.q.map(([x, y, t]) => ({ x, y, t }));
    const v = validate(cand);
    if (v.ok && cand.length === n && v.s < s) { squares = cand; s = v.s; fromSearch++; }
  }
  const check = validate(squares);
  if (!check.ok || squares.length !== n) throw new Error(`n=${n}: ${check.reason}`);
  total += s / Math.sqrt(n) - 1;
  const rows = csvRows(n, squares, id);
  id += rows.length;
  lines.push(...rows);
}

fs.writeFileSync(out, lines.join('\n') + '\n');
console.log(`wrote ${out}: ${id} rows, ${fromSearch} puzzles from GPU search, Σ(s/√n − 1) = ${total.toFixed(6)}`);
