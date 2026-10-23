// Drive the site in headless Chrome (WebGPU) to search each n for a few seconds
// and merge every verified improvement into data/improved.json.
//
//   npm i -D playwright && python3 -m http.server 8765 &
//   node tools/precompute.cjs [secondsPerN=20] [maxN=200] [port=8765]

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const SECS = +(process.argv[2] || 20), MAX_N = +(process.argv[3] || 200), PORT = +(process.argv[4] || 8765);
const OUT = path.join(__dirname, '..', 'data', 'improved.json');
const KEY = 'squares-in-squares:improved:v1';

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('pageerror', e.message));
  const base = JSON.parse(fs.readFileSync(OUT, 'utf8'));
  const save = (store) => {
    const merged = JSON.parse(fs.readFileSync(OUT, 'utf8'));
    for (const [n, e] of Object.entries(store)) if (!merged[n] || e.s < merged[n].s - 1e-12) merged[n] = e;
    const sorted = Object.fromEntries(Object.keys(merged).sort((a, b) => a - b).map((k) => [k, merged[k]]));
    fs.writeFileSync(OUT, JSON.stringify(sorted));
  };
  for (let n = 2; n <= MAX_N; n++) {
    await page.goto(`http://127.0.0.1:${PORT}/?n=${n}`);
    await page.waitForFunction(() => document.getElementById('st-gap').textContent !== '—');
    const gap = await page.textContent('#st-gap');
    if (!gap.startsWith('+')) continue; // already matches the record
    await page.click('#btn-opt');
    await page.waitForTimeout(SECS * 1000);
    await page.click('#btn-opt');
    await page.waitForTimeout(300);
    console.log(`n=${n}  before ${gap}  after ${await page.textContent('#st-gap')}  s=${await page.textContent('#st-s')}`);
    save(await page.evaluate((k) => JSON.parse(localStorage.getItem(k) || '{}'), KEY));
  }
  await browser.close();
  console.log('kept', Object.keys(JSON.parse(fs.readFileSync(OUT, 'utf8'))).length, 'improved packings (was', Object.keys(base).length + ')');
})();
