import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { normCdf, normPdf, price, greeks, impliedVol } from '../src/bsm.js';

const DATA = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../data');

function readCsv(name) {
  const [head, ...lines] = fs.readFileSync(path.join(DATA, name), 'utf8').trim().split(/\r?\n/);
  const cols = head.split(',');
  return lines.map((l) => Object.fromEntries(l.split(',').map((v, i) => [cols[i], v])));
}

// Small seeded PRNG so the random tests are reproducible.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const uniform = (r, a, b) => a + (b - a) * r();

function randomCases(seed, n) {
  const r = rng(seed);
  return Array.from({ length: n }, (_, i) => {
    const S = uniform(r, 20, 200);
    return {
      type: i % 2 ? 'put' : 'call',
      S,
      K: S * uniform(r, 0.6, 1.6),
      T: uniform(r, 0.05, 3),
      r: uniform(r, -0.01, 0.1),
      sigma: uniform(r, 0.05, 1.2),
      q: uniform(r, 0, 0.05),
    };
  });
}

const px = (c) => price(c.type, c.S, c.K, c.T, c.r, c.sigma, c.q);

test('normal CDF matches 50-digit values on [-12, 12] to ~1e-15 relative', () => {
  const rows = readCsv('normcdf_reference.csv');
  assert.ok(rows.length > 150);
  let worst = 0;
  for (const { z, cdf } of rows) {
    const ref = Number(cdf);
    worst = Math.max(worst, Math.abs(normCdf(Number(z)) - ref) / ref);
  }
  assert.ok(worst < 1e-15, `worst relative error ${worst}`);
});

test('normal CDF/PDF basic properties', () => {
  assert.equal(normCdf(0), 0.5);
  assert.ok(Math.abs(normPdf(0) - 0.3989422804014327) < 1e-16);
  assert.ok(Number.isNaN(normCdf(NaN)));
  assert.equal(normCdf(40), 1);
  assert.ok(normCdf(-40) < 1e-300 || normCdf(-40) === 0);
  for (let z = -6; z <= 6; z += 0.37) assert.ok(Math.abs(normCdf(z) + normCdf(-z) - 1) < 1e-15);
});

test('textbook values (Hull; ATM benchmark)', () => {
  assert.ok(Math.abs(price('call', 42, 40, 0.5, 0.1, 0.2) - 4.7594) < 5e-5);
  assert.ok(Math.abs(price('put', 42, 40, 0.5, 0.1, 0.2) - 0.8086) < 5e-5);
  assert.ok(Math.abs(price('call', 100, 100, 1, 0.05, 0.2) - 10.4506) < 5e-5);
  assert.ok(Math.abs(price('put', 100, 100, 1, 0.05, 0.2) - 5.5735) < 5e-5);
});

test('price, Greeks and implied vol agree with the 50-digit reference contracts', () => {
  const rows = readCsv('reference.csv');
  assert.ok(rows.length >= 500);
  const names = ['delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'volga'];
  const maxAbsErr = Object.fromEntries(names.map((n) => [n, 0]));
  const maxVal = Object.fromEntries(names.map((n) => [n, 0]));
  let worstPrice = 0;
  let worstIv = 0;
  for (const row of rows) {
    const [S, K, T, r, q, sigma] = ['S', 'K', 'T', 'r', 'q', 'sigma'].map((k) => Number(row[k]));
    const refPrice = Number(row.price);
    worstPrice = Math.max(worstPrice, Math.abs(price(row.kind, S, K, T, r, sigma, q) - refPrice) / refPrice);
    const g = greeks(row.kind, S, K, T, r, sigma, q);
    for (const n of names) {
      maxAbsErr[n] = Math.max(maxAbsErr[n], Math.abs(g[n] - Number(row[n])));
      maxVal[n] = Math.max(maxVal[n], Math.abs(Number(row[n])));
    }
    const iv = impliedVol(row.kind, refPrice, S, K, T, r, q);
    assert.ok(iv.converged, `no convergence for ${JSON.stringify(row)}`);
    worstIv = Math.max(worstIv, Math.abs(iv.iv - sigma));
  }
  assert.ok(worstPrice < 1e-13, `price relative error ${worstPrice}`);
  for (const n of names) assert.ok(maxAbsErr[n] / maxVal[n] < 1e-13, `${n}: ${maxAbsErr[n] / maxVal[n]}`);
  assert.ok(worstIv < 1e-9, `iv error ${worstIv}`);
});

test('put-call parity over random parameter sets', () => {
  let worst = 0;
  for (const c of randomCases(1, 5000)) {
    const call = price('call', c.S, c.K, c.T, c.r, c.sigma, c.q);
    const put = price('put', c.S, c.K, c.T, c.r, c.sigma, c.q);
    const rhs = c.S * Math.exp(-c.q * c.T) - c.K * Math.exp(-c.r * c.T);
    worst = Math.max(worst, Math.abs(call - put - rhs) / (c.S * Math.exp(-c.q * c.T)));
  }
  assert.ok(worst < 1e-13, `worst parity gap ${worst}`);
});

test('degenerate limits: T = 0 and sigma = 0', () => {
  assert.ok(Math.abs(price('call', 110, 100, 0, 0.05, 0.2) - 10) < 1e-14);
  assert.equal(price('call', 90, 100, 0, 0.05, 0.2), 0);
  assert.ok(Math.abs(price('put', 90, 100, 0, 0.05, 0.2) - 10) < 1e-14);
  const fwd = Math.max(100 * Math.exp(-0.02) - 95 * Math.exp(-0.05), 0);
  assert.ok(Math.abs(price('call', 100, 95, 1, 0.05, 0, 0.02) - fwd) < 1e-13);
});

test('Greeks agree with five-point finite differences of the price', () => {
  const fd = (f, x, h) => (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h);
  for (const c of randomCases(3, 150)) {
    const g = greeks(c.type, c.S, c.K, c.T, c.r, c.sigma, c.q);
    const P = (S, T, r, s) => price(c.type, S, c.K, T, r, s, c.q);
    const checks = {
      delta: fd((S) => P(S, c.T, c.r, c.sigma), c.S, 1e-3 * c.S),
      vega: fd((s) => P(c.S, c.T, c.r, s), c.sigma, 1e-4),
      theta: -fd((T) => P(c.S, T, c.r, c.sigma), c.T, 1e-3),
      rho: fd((r) => P(c.S, c.T, r, c.sigma), c.r, 1e-4),
      gamma: fd((S) => greeks(c.type, S, c.K, c.T, c.r, c.sigma, c.q).delta, c.S, 1e-3 * c.S),
      vanna: fd((s) => greeks(c.type, c.S, c.K, c.T, c.r, s, c.q).delta, c.sigma, 1e-4),
      volga: fd((s) => greeks(c.type, c.S, c.K, c.T, c.r, s, c.q).vega, c.sigma, 1e-4),
    };
    for (const [name, num] of Object.entries(checks)) {
      const err = Math.abs(g[name] - num) / (Math.abs(num) + 1e-3);
      assert.ok(err < 1e-5, `${name}: analytic ${g[name]} vs fd ${num}`);
    }
  }
});

test('call and put Greeks obey the relations from differentiating put-call parity', () => {
  for (const c of randomCases(4, 500)) {
    const a = greeks('call', c.S, c.K, c.T, c.r, c.sigma, c.q);
    const b = greeks('put', c.S, c.K, c.T, c.r, c.sigma, c.q);
    assert.ok(Math.abs(a.delta - b.delta - Math.exp(-c.q * c.T)) < 1e-13);
    assert.ok(Math.abs(a.gamma - b.gamma) < 1e-15);
    assert.ok(Math.abs(a.vega - b.vega) < 1e-12);
    assert.ok(Math.abs(a.theta - b.theta - (c.q * c.S * Math.exp(-c.q * c.T) - c.r * c.K * Math.exp(-c.r * c.T))) < 1e-10);
    assert.ok(Math.abs(a.rho - b.rho - c.K * c.T * Math.exp(-c.r * c.T)) < 1e-10);
  }
});

test('implied vol round trip; non-convergence only for quotes with no time value', () => {
  let converged = 0;
  for (const c of randomCases(5, 3000)) {
    const p = px(c);
    const res = impliedVol(c.type, p, c.S, c.K, c.T, c.r, c.q);
    if (res.converged) {
      converged += 1;
      assert.ok(Math.abs(price(c.type, c.S, c.K, c.T, c.r, res.iv, c.q) - p) <= 1e-9 * p);
      if (greeks(c.type, c.S, c.K, c.T, c.r, c.sigma, c.q).vega > 1e-3) assert.ok(Math.abs(res.iv - c.sigma) < 1e-7);
    } else {
      assert.ok(['at-or-below-intrinsic', 'time-value-below-rounding-noise'].includes(res.reason), res.reason);
    }
  }
  assert.ok(converged > 0.95 * 3000);
});

test('implied vol converges in the far out-of-the-money wings', () => {
  for (const K of [40, 60, 150, 250]) {
    const type = K > 100 ? 'call' : 'put';
    const p = price(type, 100, K, 0.1, 0.04, 0.12, 0);
    assert.ok(p > 0 && p < 1e-20);
    const res = impliedVol(type, p, 100, K, 0.1, 0.04, 0);
    assert.ok(res.converged && Math.abs(res.iv - 0.12) < 1e-6, `K=${K}: ${JSON.stringify(res)}`);
  }
});

test('quotes within rounding noise of intrinsic value are rejected, never guessed at', () => {
  for (const type of ['call', 'put']) {
    const K = type === 'call' ? 90 : 110;
    const intrinsic = price(type, 100, K, 0.5, 0.04, 0, 0.015);
    for (const ulps of [0, 1, 2, 4, 7]) {
      const res = impliedVol(type, intrinsic * (1 + ulps * Number.EPSILON), 100, K, 0.5, 0.04, 0.015);
      assert.ok(!res.converged && Number.isNaN(res.iv), `${type} ${ulps} ulps`);
    }
    const ok = impliedVol(type, price(type, 100, K, 0.5, 0.04, 0.25, 0.015), 100, K, 0.5, 0.04, 0.015);
    assert.ok(ok.converged && Math.abs(ok.iv - 0.25) < 1e-9);
  }
});

test('invalid and arbitrage-violating quotes give a reason, not a number', () => {
  const S = 100;
  const K = 100;
  const T = 1;
  const r = 0.04;
  const q = 0.015;
  const discS = S * Math.exp(-q * T);
  const discK = K * Math.exp(-r * T);
  for (const type of ['call', 'put']) {
    const upper = type === 'call' ? discS : discK;
    const cases = [
      [-1, 'at-or-below-intrinsic'],
      [0, 'at-or-below-intrinsic'],
      [upper, 'at-or-above-upper-bound'],
      [upper + 1, 'at-or-above-upper-bound'],
      [NaN, 'not-finite'],
      [Infinity, 'not-finite'],
    ];
    for (const [quote, reason] of cases) {
      const res = impliedVol(type, quote, S, K, T, r, q);
      assert.equal(res.converged, false);
      assert.ok(Number.isNaN(res.iv));
      assert.equal(res.reason, reason, `${type} ${quote}`);
      assert.ok(res.message.length > 10);
    }
  }
  const p = price('call', 100, 100, 1, 0.04, 3.0, 0.015); // 300% vol
  assert.equal(impliedVol('call', p, 100, 100, 1, 0.04, 0.015, { sigmaHi: 2 }).reason, 'vol-outside-search-bracket');
  assert.ok(Math.abs(impliedVol('call', p, 100, 100, 1, 0.04, 0.015).iv - 3) < 1e-8);
});

test('invalid inputs throw RangeError', () => {
  assert.throws(() => price('call', -1, 100, 1, 0.05, 0.2), RangeError);
  assert.throws(() => price('call', 100, 0, 1, 0.05, 0.2), RangeError);
  assert.throws(() => price('call', 100, 100, -1, 0.05, 0.2), RangeError);
  assert.throws(() => price('call', 100, 100, 1, 0.05, -0.2), RangeError);
  assert.throws(() => price('call', NaN, 100, 1, 0.05, 0.2), RangeError);
  assert.throws(() => price('straddle', 100, 100, 1, 0.05, 0.2), RangeError);
  assert.throws(() => greeks('call', 100, 100, 0, 0.05, 0.2), RangeError);
  assert.throws(() => greeks('call', 100, 100, 1, 0.05, 0), RangeError);
  assert.throws(() => impliedVol('call', 5, 100, 100, 0, 0.05), RangeError);
});
