// Black-Scholes-Merton pricing, analytic Greeks and an implied-volatility solver.
//
// A JavaScript port of the Python package (python/bsm) and the C++ library (cpp/): the same
// formulas, conventions and solver. It runs in the browser for the demo page and is tested
// against the same 50-digit reference values as the other two implementations.
//
// Conventions: raw derivatives. vega per 1.00 of volatility, rho per 1.00 of rate, theta = dV/dt
// per year (negative for long options). Rates, volatility and yield are decimals (0.05 = 5%).

const INV_SQRT_2PI = 0.3989422804014326779;
const EPS = Number.EPSILON;

// ---------------------------------------------------------------------------------------------
// Standard normal distribution, accurate to ~4e-16 relative error on [-12, 12] (checked against
// 50-digit values in tests). JavaScript has no erfc, so N(z) is built from two pieces:
//   * |z| < 1.2: the positive-term series  N(z) = 1/2 + pdf(z) * sum z^(2n+1) / (2n+1)!!
//   * otherwise: the Gaussian continued fraction  N(-z) = pdf(z) / (z + 1/(z + 2/(z + 3/(z + ...))))
// ---------------------------------------------------------------------------------------------

// exp(-z^2 / 2) with z^2 split exactly: z^2 = zh^2 + (z - zh)(z + zh) and zh has few mantissa bits,
// so zh^2 is exact. Avoids losing z^2 * 1e-16 of relative accuracy in the tails.
function expHalfNegSq(z) {
  if (Math.abs(z) > 40) return 0; // underflows to 0 (and avoids 0 * Infinity in the split below)
  const zh = Math.round(z * 16) / 16;
  return Math.exp(-0.5 * zh * zh) * Math.exp(-0.5 * (z - zh) * (z + zh));
}

export function normPdf(z) {
  return INV_SQRT_2PI * expHalfNegSq(z);
}

// N(-z) for z >= 0.
function upperTail(z) {
  if (z < 1.2) {
    let term = z;
    let sum = z;
    for (let n = 1; n < 400; n++) {
      term *= (z * z) / (2 * n + 1);
      sum += term;
      if (term < sum * 1e-17) break;
    }
    return 0.5 - normPdf(z) * sum;
  }
  const terms = z >= 5 ? 30 : z >= 3 ? 80 : 300;
  let t = z;
  for (let k = terms; k >= 1; k--) t = z + k / t;
  return normPdf(z) / t;
}

export function normCdf(z) {
  if (Number.isNaN(z)) return NaN;
  return z >= 0 ? 1 - upperTail(z) : upperTail(-z);
}

// ---------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------

function phiOf(type) {
  if (type === 'call' || type === 'c') return 1;
  if (type === 'put' || type === 'p') return -1;
  throw new RangeError("type must be 'call' or 'put'");
}

function requireFinite(name, x) {
  if (!Number.isFinite(x)) throw new RangeError(`${name} must be a finite number`);
}

function validate(S, K, T, r, sigma, q, { strict }) {
  [['S', S], ['K', K], ['T', T], ['r', r], ['sigma', sigma], ['q', q]].forEach(([n, x]) => requireFinite(n, x));
  if (S <= 0 || K <= 0) throw new RangeError('S and K must be strictly positive');
  if (strict ? T <= 0 || sigma <= 0 : T < 0 || sigma < 0) {
    throw new RangeError(strict ? 'T and sigma must be strictly positive' : 'T and sigma must be non-negative');
  }
}

// ---------------------------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------------------------

// Core formula on validated inputs. phi = +1 call, -1 put.
function priceCore(phi, S, K, T, r, sigma, q) {
  const discS = S * Math.exp(-q * T);
  const discK = K * Math.exp(-r * T);
  const volSqrtT = sigma * Math.sqrt(T);
  if (volSqrtT <= 0) return Math.max(phi * (discS - discK), 0); // forward intrinsic value
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / volSqrtT;
  const d2 = d1 - volSqrtT;
  return phi * (discS * normCdf(phi * d1) - discK * normCdf(phi * d2));
}

/** Price of a European option. T = 0 or sigma = 0 return the discounted forward intrinsic value. */
export function price(type, S, K, T, r, sigma, q = 0) {
  const phi = phiOf(type);
  validate(S, K, T, r, sigma, q, { strict: false });
  return priceCore(phi, S, K, T, r, sigma, q);
}

/** Analytic Greeks (requires T > 0 and sigma > 0). */
export function greeks(type, S, K, T, r, sigma, q = 0) {
  const phi = phiOf(type);
  validate(S, K, T, r, sigma, q, { strict: true });
  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
  const d2 = d1 - sigma * sqrtT;
  const es = Math.exp(-q * T);
  const ek = Math.exp(-r * T);
  const pdf1 = normPdf(d1);
  const vega = S * es * pdf1 * sqrtT;
  return {
    delta: phi * es * normCdf(phi * d1),
    gamma: (es * pdf1) / (S * sigma * sqrtT),
    vega,
    theta: (-S * es * pdf1 * sigma) / (2 * sqrtT) - phi * r * K * ek * normCdf(phi * d2) + phi * q * S * es * normCdf(phi * d1),
    rho: phi * K * T * ek * normCdf(phi * d2),
    vanna: (-es * pdf1 * d2) / sigma,
    volga: (vega * d1 * d2) / sigma,
  };
}

// ---------------------------------------------------------------------------------------------
// Implied volatility: safeguarded Newton on log-price, on the out-of-the-money side
// ---------------------------------------------------------------------------------------------

function initialGuess(p, S, K, T, r, q) {
  const discS = S * Math.exp(-q * T);
  const x = Math.abs(Math.log(S / K) + (r - q) * T);
  const mk = Math.sqrt((2 * x) / T); // Manaster-Koltveit
  const atm = Math.sqrt((2 * Math.PI) / T) * (p / discS); // Brenner-Subrahmanyam
  return Math.max(0.5 * mk, atm);
}

const REASONS = {
  'not-finite': 'The price is not a finite number.',
  'at-or-below-intrinsic': 'The price is at or below the option’s intrinsic value, so no volatility can produce it (arbitrage).',
  'at-or-above-upper-bound': 'The price is at or above the model’s upper bound (the discounted spot for a call, the discounted strike for a put).',
  'time-value-below-rounding-noise':
    'The price is intrinsic value to within rounding error, so it carries no volatility information.',
  'vol-outside-search-bracket': 'The implied volatility lies outside the search range.',
  'did-not-converge': 'The solver did not converge.',
};

/**
 * Black-Scholes implied volatility.
 * Returns { iv, iterations, converged, reason, message }; iv is NaN when there is no solution and
 * `reason` says why (see REASONS). Throws RangeError for invalid contract terms.
 */
export function impliedVol(type, marketPrice, S, K, T, r, q = 0, opts = {}) {
  const { tol = 1e-12, maxIter = 100, sigmaLo = 1e-8, sigmaHi = 10 } = opts;
  const phi = phiOf(type);
  validate(S, K, T, r, 1, q, { strict: true });
  const reject = (reason) => ({ iv: NaN, iterations: 0, converged: false, reason, message: REASONS[reason] });

  const discS = S * Math.exp(-q * T);
  const discK = K * Math.exp(-r * T);
  const lower = Math.max(phi * (discS - discK), 0);
  const upper = phi > 0 ? discS : discK;
  if (!Number.isFinite(marketPrice)) return reject('not-finite');
  if (!(marketPrice > lower)) return reject('at-or-below-intrinsic');
  if (!(marketPrice < upper)) return reject('at-or-above-upper-bound');

  // Convert an in-the-money quote to the equivalent out-of-the-money option by put-call parity:
  // the opposite option's price is the quote minus forward intrinsic value (the time value).
  const itm = lower > 0;
  const phiO = itm ? -phi : phi;
  const p = itm ? marketPrice - lower : marketPrice;
  // Time value within rounding noise of the intrinsic value carries no volatility information.
  const noise = itm ? 8 * EPS * (discS + discK) : 0;
  if (!(p > noise)) return reject('time-value-below-rounding-noise');
  if (!(p > priceCore(phiO, S, K, T, r, sigmaLo, q)) || !(p < priceCore(phiO, S, K, T, r, sigmaHi, q))) {
    return reject('vol-outside-search-bracket');
  }

  let lo = sigmaLo;
  let hi = sigmaHi;
  let sigma = Math.min(Math.max(initialGuess(p, S, K, T, r, q), 1e-3), 0.5 * sigmaHi);
  const sqrtT = Math.sqrt(T);
  const logP = Math.log(p);

  for (let it = 1; it <= maxIter; it++) {
    const model = priceCore(phiO, S, K, T, r, sigma, q);
    const g = Math.log(model) - logP; // -Infinity if the model price underflows to 0
    if (g > 0) hi = sigma;
    else lo = sigma;
    if (Math.abs(g) <= tol || hi - lo <= 1e-15 * Math.max(sigma, 1e-8)) {
      return { iv: sigma, iterations: it, converged: true, reason: null, message: '' };
    }
    const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const vega = S * Math.exp(-q * T) * normPdf(d1) * sqrtT;
    let next = sigma - (g * model) / vega;
    if (!Number.isFinite(next) || next <= lo || next >= hi) next = 0.5 * (lo + hi);
    sigma = next;
  }
  return reject('did-not-converge');
}
