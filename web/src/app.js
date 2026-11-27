import { price, greeks, impliedVol } from './bsm.js';

// ---------------------------------------------------------------- state and inputs
const FIELDS = [
  { key: 'S', label: 'Spot S', min: 1, max: 300, step: 0.5, dec: 2, scale: 1, ok: (v) => v > 0 },
  { key: 'K', label: 'Strike K', min: 1, max: 300, step: 0.5, dec: 2, scale: 1, ok: (v) => v > 0 },
  { key: 'T', label: 'Expiry T (years)', min: 0.01, max: 5, step: 0.01, dec: 2, scale: 1, ok: (v) => v > 0 },
  { key: 'r', label: 'Rate r (%)', min: -2, max: 20, step: 0.1, dec: 2, scale: 0.01, ok: () => true },
  { key: 'sigma', label: 'Volatility σ (%)', min: 1, max: 200, step: 0.5, dec: 1, scale: 0.01, ok: (v) => v > 0 },
  { key: 'q', label: 'Dividend yield q (%)', min: 0, max: 15, step: 0.1, dec: 2, scale: 0.01, ok: () => true },
];
const shown = { S: 100, K: 100, T: 1, r: 5, sigma: 20, q: 0 }; // values as displayed (percent for r, sigma, q)
const bad = {};
const $ = (id) => document.getElementById(id);

function model() {
  const m = {};
  for (const f of FIELDS) m[f.key] = shown[f.key] * f.scale;
  return m;
}

function buildInputs() {
  const host = $('inputs');
  for (const f of FIELDS) {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.innerHTML = `<label for="n-${f.key}">${f.label}</label>
      <input type="range" id="r-${f.key}" min="${f.min}" max="${f.max}" step="${f.step}" value="${shown[f.key]}" aria-label="${f.label} slider">
      <input type="number" id="n-${f.key}" step="any" value="${shown[f.key]}" inputmode="decimal">`;
    host.appendChild(wrap);
    const range = wrap.querySelector('input[type=range]');
    const num = wrap.querySelector('input[type=number]');
    range.addEventListener('input', () => {
      num.value = range.value;
      accept(f, Number(range.value), wrap);
    });
    num.addEventListener('input', () => {
      const v = num.value.trim() === '' ? NaN : Number(num.value);
      if (Number.isFinite(v)) range.value = String(v);
      accept(f, v, wrap);
    });
  }
}

function accept(f, v, wrap) {
  const valid = Number.isFinite(v) && f.ok(v);
  wrap.classList.toggle('bad', !valid);
  if (valid) {
    shown[f.key] = v;
    delete bad[f.key];
  } else {
    bad[f.key] = f;
  }
  const msg = $('input-msg');
  const keys = Object.keys(bad);
  msg.textContent = keys.length
    ? `${bad[keys[0]].label} must be a number${['S', 'K', 'T', 'sigma'].includes(keys[0]) ? ' greater than zero' : ''}. Showing the last valid values.`
    : '';
  render();
}

// ---------------------------------------------------------------- formatting
function fmt(x, d = 4) {
  if (!Number.isFinite(x)) return '–';
  return x.toFixed(d);
}
function sig(x, n = 6) {
  if (!Number.isFinite(x)) return '–';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-4) return x.toExponential(n - 1);
  return Number(x.toPrecision(n)).toString();
}
function tickFmt(step) {
  const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return (x) => (Math.abs(x) < step * 1e-9 ? '0' : x.toFixed(d));
}
function niceTicks(lo, hi, n = 5) {
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9; t += step) ticks.push(t);
  return { ticks, step };
}

// ---------------------------------------------------------------- prices and Greeks table
const GREEKS = [
  { key: 'delta', name: 'Delta', raw: 'per 1.00 of spot', trader: 'per 1.00 of spot', div: 1 },
  { key: 'gamma', name: 'Gamma', raw: 'per 1.00 of spot²', trader: 'per 1.00 of spot²', div: 1 },
  { key: 'vega', name: 'Vega', raw: 'per 1.00 of vol (100 pts)', trader: 'per 1 vol point', div: 100 },
  { key: 'theta', name: 'Theta', raw: 'per year', trader: 'per calendar day', div: 365 },
  { key: 'rho', name: 'Rho', raw: 'per 1.00 of rate (100 pts)', trader: 'per 1 rate point', div: 100 },
  { key: 'vanna', name: 'Vanna', raw: 'dDelta per 1.00 of vol', trader: 'dDelta per 1 vol point', div: 100 },
  { key: 'volga', name: 'Volga', raw: 'dVega per 1.00 of vol', trader: 'dVega per 1 vol point', div: 10000 },
];
const SAME_FOR_CALL_AND_PUT = new Set(['gamma', 'vega', 'vanna', 'volga']);

function renderPrices(m) {
  const c = price('call', m.S, m.K, m.T, m.r, m.sigma, m.q);
  const p = price('put', m.S, m.K, m.T, m.r, m.sigma, m.q);
  $('call-price').textContent = fmt(c);
  $('put-price').textContent = fmt(p);
  const gap = c - p - (m.S * Math.exp(-m.q * m.T) - m.K * Math.exp(-m.r * m.T));
  $('parity').textContent = gap.toExponential(2);

  const trader = $('trader').checked;
  const gc = greeks('call', m.S, m.K, m.T, m.r, m.sigma, m.q);
  const gp = greeks('put', m.S, m.K, m.T, m.r, m.sigma, m.q);
  $('greeks-body').innerHTML = GREEKS.map((g) => {
    const d = trader ? g.div : 1;
    return `<tr><th scope="row">${g.name}<span class="unit">${trader ? g.trader : g.raw}</span></th><td class="num">${sig(gc[g.key] / d)}</td><td class="num">${sig(gp[g.key] / d)}</td></tr>`;
  }).join('');
}

// ---------------------------------------------------------------- charts
const tip = $('tip');

function legendHtml(series) {
  return `<div class="legend">${series
    .map((s) => `<span><svg viewBox="0 0 28 10" aria-hidden="true"><line x1="0" y1="5" x2="28" y2="5" style="stroke:${s.color};stroke-width:2;${s.dash ? 'stroke-dasharray:6 4' : ''}"/></svg>${s.name}</span>`)
    .join('')}</div>`;
}

function lineChart(host, { xs, series, xTitle, yTitle, marker, yFmt }) {
  const W = Math.max(320, host.clientWidth || 600);
  const H = W < 520 ? 280 : 340;
  const M = { l: 62, r: 18, t: 12, b: 44 };
  const pw = W - M.l - M.r;
  const ph = H - M.t - M.b;
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) for (const y of s.ys) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.06;
  const { ticks: yt, step: ys } = niceTicks(lo - padY, hi + padY, 5);
  const yMin = Math.min(lo - padY, yt[0]);
  const yMax = Math.max(hi + padY, yt[yt.length - 1]);
  const { ticks: xt, step: xsStep } = niceTicks(x0, x1, W < 520 ? 4 : 7);
  const sx = (x) => M.l + ((x - x0) / (x1 - x0)) * pw;
  const sy = (y) => M.t + (1 - (y - yMin) / (yMax - yMin)) * ph;
  const fy = yFmt || tickFmt(ys);
  const fx = tickFmt(xsStep);

  let svg = `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${yTitle} against ${xTitle}">`;
  for (const t of yt) svg += `<line class="grid-l" x1="${M.l}" x2="${W - M.r}" y1="${sy(t)}" y2="${sy(t)}"/><text class="axis-t" x="${M.l - 8}" y="${sy(t) + 4}" text-anchor="end">${fy(t)}</text>`;
  for (const t of xt) svg += `<text class="axis-t" x="${sx(t)}" y="${H - M.b + 18}" text-anchor="middle">${fx(t)}</text>`;
  svg += `<line class="axis-l" x1="${M.l}" x2="${W - M.r}" y1="${M.t + ph}" y2="${M.t + ph}"/>`;
  svg += `<text class="axis-t" x="${M.l + pw / 2}" y="${H - 6}" text-anchor="middle">${xTitle}</text>`;
  svg += `<text class="axis-t" transform="translate(14 ${M.t + ph / 2}) rotate(-90)" text-anchor="middle">${yTitle}</text>`;
  if (marker !== undefined && marker >= x0 && marker <= x1) {
    svg += `<line x1="${sx(marker)}" x2="${sx(marker)}" y1="${M.t}" y2="${M.t + ph}" style="stroke:var(--axis);stroke-width:1;stroke-dasharray:3 4"/>`;
  }
  for (const s of series) {
    const d = s.ys.map((y, i) => `${i ? 'L' : 'M'}${sx(xs[i]).toFixed(2)} ${sy(y).toFixed(2)}`).join('');
    svg += `<path d="${d}" fill="none" style="stroke:${s.color};stroke-width:2;stroke-linejoin:round;${s.dash ? 'stroke-dasharray:6 4' : ''}"/>`;
  }
  // Direct labels at the right end, the higher end value above its line and the lower one below.
  const order = [...series].sort((a, b) => b.ys[b.ys.length - 1] - a.ys[a.ys.length - 1]);
  order.forEach((s, i) => {
    const y = sy(s.ys[s.ys.length - 1]);
    const dy = series.length === 1 ? -8 : i === 0 ? -8 : 16;
    svg += `<text x="${W - M.r - 4}" y="${y + dy}" text-anchor="end" style="font-size:12px;fill:var(--text)">${s.name}</text>`;
  });
  // Marker dots at the current spot, with a surface ring.
  if (marker !== undefined && marker >= x0 && marker <= x1) {
    const mi = interp(xs, marker);
    for (const s of series) {
      const y = s.ys[mi.i] * (1 - mi.f) + s.ys[Math.min(mi.i + 1, xs.length - 1)] * mi.f;
      svg += `<circle cx="${sx(marker)}" cy="${sy(y)}" r="4.5" style="fill:${s.color};stroke:var(--surface);stroke-width:2"/>`;
    }
  }
  svg += `<g id="hov" style="display:none"><line id="hov-l" y1="${M.t}" y2="${M.t + ph}" style="stroke:var(--text-2);stroke-width:1"/>${series
    .map((s, k) => `<circle id="hov-${k}" r="4.5" style="fill:${s.color};stroke:var(--surface);stroke-width:2"/>`)
    .join('')}</g>`;
  svg += `<rect id="hit" x="${M.l}" y="${M.t}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;

  host.innerHTML = legendHtml(series) + svg;
  const el = host.querySelector('svg.plot');
  const hov = el.querySelector('#hov');
  const hovL = el.querySelector('#hov-l');
  const hit = el.querySelector('#hit');
  const move = (ev) => {
    const box = el.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    const x = x0 + ((px - M.l) / pw) * (x1 - x0);
    const i = Math.max(0, Math.min(xs.length - 1, Math.round(((x - x0) / (x1 - x0)) * (xs.length - 1))));
    hov.style.display = '';
    hovL.setAttribute('x1', sx(xs[i]));
    hovL.setAttribute('x2', sx(xs[i]));
    series.forEach((s, k) => {
      const c = el.querySelector(`#hov-${k}`);
      c.setAttribute('cx', sx(xs[i]));
      c.setAttribute('cy', sy(s.ys[i]));
    });
    tip.hidden = false;
    tip.innerHTML = `<b>${xTitle} ${xs[i].toFixed(2)}</b>` + series.map((s) => `<div><i style="background:${s.color}"></i>${s.name}: ${sig(s.ys[i], 6)}</div>`).join('');
    const tw = tip.offsetWidth;
    tip.style.left = `${Math.min(window.innerWidth - tw - 8, ev.clientX + 14)}px`;
    tip.style.top = `${Math.max(8, ev.clientY - 20 - tip.offsetHeight)}px`;
  };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerdown', move);
  hit.addEventListener('pointerleave', () => { hov.style.display = 'none'; tip.hidden = true; });
}

function interp(xs, x) {
  const n = xs.length - 1;
  const t = ((x - xs[0]) / (xs[n] - xs[0])) * n;
  const i = Math.min(n - 1, Math.max(0, Math.floor(t)));
  return { i, f: t - i };
}

function tableHtml(xs, series, picks = 11) {
  const idx = Array.from({ length: picks }, (_, k) => Math.round((k * (xs.length - 1)) / (picks - 1)));
  return `<table class="tbl"><thead><tr><th scope="col">Spot</th>${series.map((s) => `<th scope="col" class="num">${s.name}</th>`).join('')}</tr></thead><tbody>${idx
    .map((i) => `<tr><td>${xs[i].toFixed(2)}</td>${series.map((s) => `<td class="num">${sig(s.ys[i], 6)}</td>`).join('')}</tr>`)
    .join('')}</tbody></table>`;
}

function spotGrid(m) {
  const lo = Math.min(0.5 * m.K, 0.9 * m.S);
  const hi = Math.max(1.5 * m.K, 1.1 * m.S);
  const n = 160;
  return Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
}

function renderCharts(m) {
  const xs = spotGrid(m);
  // Chart 1: value today against payoff at expiry
  const type = $('c1-type').value;
  const phi = type === 'call' ? 1 : -1;
  const s1 = [
    { name: 'Value today', color: 'var(--s1)', ys: xs.map((S) => price(type, S, m.K, m.T, m.r, m.sigma, m.q)) },
    { name: 'Payoff at expiry', color: 'var(--s2)', dash: true, ys: xs.map((S) => Math.max(phi * (S - m.K), 0)) },
  ];
  lineChart($('chart1'), { xs, series: s1, xTitle: 'Spot price', yTitle: `${type === 'call' ? 'Call' : 'Put'} value`, marker: m.S });
  $('table1').innerHTML = tableHtml(xs, s1);

  // Chart 2: one Greek against spot
  const gk = $('c2-greek').value;
  const same = SAME_FOR_CALL_AND_PUT.has(gk);
  const ys = (t) => xs.map((S) => greeks(t, S, m.K, m.T, m.r, m.sigma, m.q)[gk]);
  const s2 = same
    ? [{ name: 'Call and put', color: 'var(--s1)', ys: ys('call') }]
    : [{ name: 'Call', color: 'var(--s1)', ys: ys('call') }, { name: 'Put', color: 'var(--s2)', dash: true, ys: ys('put') }];
  const g = GREEKS.find((x) => x.key === gk);
  lineChart($('chart2'), { xs, series: s2, xTitle: 'Spot price', yTitle: `${g.name} (${g.raw})`, marker: m.S });
  $('c2-note').textContent = same
    ? `${g.name} is the same for a call and a put with the same strike and expiry (differentiate put-call parity), so one line is shown.`
    : '';
  $('table2').innerHTML = tableHtml(xs, s2);
}

// ---------------------------------------------------------------- implied volatility
function renderIv(m) {
  const type = $('iv-type').value;
  const raw = $('iv-price').value.trim();
  const out = $('iv-out');
  const quote = raw === '' ? NaN : Number(raw);
  const res = impliedVol(type, quote, m.S, m.K, m.T, m.r, m.q);
  if (res.converged) {
    const back = price(type, m.S, m.K, m.T, m.r, res.iv, m.q);
    out.innerHTML = `<div class="big">${(res.iv * 100).toFixed(4)}%</div>
      <div class="meta">${res.iterations} iterations · repricing error ${Math.abs(back - quote).toExponential(1)}</div>`;
  } else {
    out.innerHTML = `<div class="big bad">No implied volatility</div><div class="meta">${res.message}</div>`;
  }
}

// ---------------------------------------------------------------- wiring
function render() {
  const m = model();
  renderPrices(m);
  renderCharts(m);
  renderIv(m);
}

function init() {
  buildInputs();
  $('c2-greek').innerHTML = GREEKS.map((g) => `<option value="${g.key}">${g.name}</option>`).join('');
  $('trader').addEventListener('change', () => renderPrices(model()));
  $('c1-type').addEventListener('change', () => renderCharts(model()));
  $('c2-greek').addEventListener('change', () => renderCharts(model()));
  $('iv-type').addEventListener('change', () => renderIv(model()));
  $('iv-price').addEventListener('input', () => renderIv(model()));
  $('iv-fill').addEventListener('click', () => {
    const m = model();
    $('iv-price').value = price($('iv-type').value, m.S, m.K, m.T, m.r, m.sigma, m.q).toFixed(4);
    renderIv(m);
  });
  $('theme').addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
  });
  let w = 0;
  new ResizeObserver(() => {
    const nw = Math.round($('chart1').clientWidth);
    if (nw !== w) { w = nw; renderCharts(model()); }
  }).observe($('chart1'));
  render();
}

init();
