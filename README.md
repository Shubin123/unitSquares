# Squares in Squares

**Live demo:** https://shubin123.github.io/unitSquares/

An interactive three.js viewer and GPU solver for the Kaggle
[Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares) problem:
pack *n* non-overlapping unit squares (any rotation) into the smallest square, for
*n* = 1…1000.

The site renders the current best packing for every *n*. It can also run a
WebGPU search in your browser that tries to beat it. Every packing is
re-verified in double precision before it is shown or exported. You can download
a single puzzle or a full `id,n,square_id,x,y,angle` submission CSV.

## How the solutions are built

### 1. Exact constructions (all n, no search)

`src/geometry.js` builds each packing from closed-form families and keeps the
smallest side for each *n*. Since s(n) is monotone, a packing for n′ ≥ n gives one
for *n* once the extra squares are deleted.

| family | squares | side |
|---|---|---|
| trivial grid | ⌈√n⌉² | ⌈√n⌉ |
| Göbel strip (a ≥ 1) | a(a+1) + 2 + b, with b = 1 + ⌊(a−1)√2⌋ | a + 1 + √2/2 |
| Göbel square (a−1 < b√2/2 < a+1) | 2a(a+1) + b² | a + 1 + b√2/2 |
| + k L-shaped borders | + Σ (⌊s⌋ + ⌊s+1⌋) | s + k |

The Göbel families tilt a strip or a b×b block of squares by 45°, with unrotated
"staircases" in the corners. The border step explains records such as
s(65) → s(82) → s(101), each one unit larger. Together these cut the summed waste
Σ(s/√n − 1) over n = 1…1000 from **28.94** (grids) to **24.21**.

### 2. GPU search (WebGPU compute shaders)

`src/gpu-solver.js` and `src/optimizer.js` run a population of R replicas, up to
2048, all on the GPU at once:

* **Penalty energy.** The energy is the sum of squared overlap depths between
  squares, plus a wall term. For two unit squares the separating-axis radii
  collapse to `0.5 + ½(|cos Δθ| + |sin Δθ|)`, so the overlap depth and its exact
  gradient with respect to x, y, *and θ* are cheap closed forms. The search needs
  no finite differences or polygon clipping.
* **Kernel layout.** One GPU thread handles one square. Neighbours are streamed
  through 64-wide shared-memory tiles, and a circumcircle test rejects distant
  pairs early. Each batch encodes about 100 gradient and update dispatches into a
  single command buffer, so the browser makes one round trip per batch.
* **Parallel tempering.** Each replica has its own temperature for Langevin
  noise. Every batch runs a hot phase and then a cold quench before readback.
* **Adaptive container shrinking.** A replica that becomes feasible shrinks its
  container, with a step that grows after each success. A replica that stays
  jammed backs off and halves its step. If it collapses, it restarts from the
  incumbent with a small perturbation and a few relocated squares, or from
  random positions.
* **f64 finishing.** The GPU works in f32. A candidate is re-polished on the
  CPU in f64, then squeezed with a doubling/halving line search on the side.
  Finally its centres are scaled out just enough that every pair is strictly
  separated (margin 1e-10) under an exact SAT check.

On an Apple M-series GPU this runs at 2–25 G pair checks/s, depending on *n*.
Starting from a 4×4 grid, it usually finds s(11) = 3.877084 (the record, to
1e-8) within 10 s. The CPU fallback, which browsers without WebGPU use, finds
s(5) = 2 + √2/2 from a 3×3 grid.

`data/improved.json` holds GPU-found packings that beat the constructions. It was
produced offline by `tools/precompute.cjs`, which drives the site in headless
Chrome. With 15 s per *n* over n ≤ 200, it improved n = 11, 17, 18, 19, 26 and 37.
n = 11 and 26 now match the records to 1e-6; the others are within 0.003–0.016.
That brings the submission total to **24.064**. For larger *n*, the records
come from more elaborate structured families that this local search does not
find in seconds. Improvements you find in your own browser are kept in `localStorage` and
included in your exported CSV.

## Repo layout

```
index.html, style.css     page (no build step; three.js from jsDelivr via import map)
src/geometry.js           SAT overlap, constructions, f64 polish/finalize, CSV
src/gpu-solver.js         WGSL kernels + WebGPU backend
src/cpu-solver.js         same math on the CPU (fallback)
src/optimizer.js          replica controller (tempering, shrinking, restarts)
src/main.js               three.js scene, UI, chart, export
data/records.json         best-known s(n) from David Ellsworth's table (n ≤ 323)
data/improved.json        bundled GPU-search improvements
tools/make-submission.mjs build + validate submission.csv offline
tools/precompute.cjs      headless-Chrome WebGPU batch search
tools/parse_records.py    regenerate records.json
```

```sh
python3 -m http.server 8765        # then open http://localhost:8765/?n=11
node tools/make-submission.mjs     # writes submission.csv (500,500 rows)
```

## Notes and caveats

* I could not read the competition's Evaluation tab or its public notebooks,
  because Kaggle requires a login. The CSV layout (`id,n,square_id,x,y,angle`,
  with centres and angles in radians) and the per-puzzle quantity s/√n − 1 come
  from the public overview text, so check them against the official page before
  you submit. Packings are exported with a 1e-9 relative safety gap, so they are
  valid even under a strict no-touching rule.
* The best-known values come from
  [David Ellsworth's Squares in Squares page](https://kingbird.myphotos.cc/packing/squares_in_squares.html),
  which continues Erich Friedman's survey. Those values are used only for the
  "gap to record" comparison.

MIT licensed.
