"""Implied-volatility solver.

Given a market price, find sigma such that ``price(S, K, T, r, sigma) == market``.

Method: a *safeguarded Newton* iteration. Newton's method on the Black-Scholes
price converges quadratically near the root, but it can shoot out of range when
vega is tiny (deep in/out of the money, very short maturities). So each iterate
keeps a bracket ``[lo, hi]`` that is guaranteed to contain the root (price is
strictly increasing in sigma, so the sign of the residual tells us which side we
are on), and any Newton step that leaves the bracket is replaced by a bisection
step. The result converges for every arbitrage-free price and never diverges.

Two refinements make it robust in the wings:

* **Work on the out-of-the-money side.** An in-the-money quote is converted with
  put-call parity to the out-of-the-money option with the same strike and
  maturity: the two differ only by the known forward intrinsic value, so the
  *time value* ``price - intrinsic`` is the price of the OTM option. This avoids
  solving for sigma from a price dominated by intrinsic value.
* **Newton on log-price.** Out-of-the-money prices behave like ``exp(-c / sigma^2)``
  in the wings; ``ln(price)`` is far better conditioned than ``price`` and Newton
  converges in a handful of steps even for prices of 1e-60.

The solver is vectorised: it iterates on all still-active elements at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import norm_pdf, prepare, squeeze
from .pricing import price_arrays


# An in-the-money quote must carry time value above this many ulps of S e^{-qT} + K e^{-rT} to be usable.
TIME_VALUE_NOISE = 8.0 * np.finfo(float).eps


@dataclass
class IVResult:
    """Detailed solver output (arrays have the broadcast input shape)."""

    iv: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray


def _no_arbitrage_bounds(S, K, T, r, q, phi):
    disc_s = S * np.exp(-q * T)
    disc_k = K * np.exp(-r * T)
    lower = np.maximum(phi * (disc_s - disc_k), 0.0)
    upper = np.where(phi > 0, disc_s, disc_k)
    return lower, upper


def _initial_guess(p, S, K, T, r, q):
    """Starting volatility from moneyness and price.

    Manaster-Koltveit: sigma* = sqrt(2 |ln(F/K)| / T) is the vol at which vega is
    maximal for that moneyness; half of it is a good start for out-of-the-money
    options. At the money that is zero, so also consider the Brenner-Subrahmanyam
    at-the-money approximation sigma ~ sqrt(2 pi / T) * price / (S e^{-qT}) and take
    the larger of the two. (Compared against a constant start and the unscaled
    Manaster-Koltveit guess on a wide grid, this needed the fewest iterations.)
    """
    disc_s = S * np.exp(-q * T)
    x = np.abs(np.log(S / K) + (r - q) * T)
    mk = np.sqrt(2.0 * x / T)
    atm = np.sqrt(2.0 * np.pi / T) * p / disc_s
    return np.maximum(0.5 * mk, atm)


def implied_vol(
    price,
    S,
    K,
    T,
    r,
    kind="call",
    q=0.0,
    *,
    tol=1e-12,
    max_iter=100,
    sigma_lo=1e-8,
    sigma_hi=10.0,
    full_output=False,
):
    """Black-Scholes implied volatility.

    Parameters
    ----------
    price : market option price(s).
    tol : relative price tolerance, ``|ln(model / market)| <= tol``.
    max_iter : iteration cap per element.
    sigma_lo, sigma_hi : search bracket for sigma (default 1e-8 to 1000% vol).
    full_output : if True return an :class:`IVResult` with iteration counts and
        convergence flags instead of just the volatilities.

    Returns
    -------
    Implied volatility (float or array). Elements where no solution exists are
    ``nan``: prices outside the no-arbitrage bounds (at or below intrinsic value,
    or at or above the upper bound), non-finite prices, in-the-money quotes whose
    time value is within rounding noise (8 ulps of S e^{-qT} + K e^{-rT}), or an implied vol
    outside ``[sigma_lo, sigma_hi]``.
    """
    raw = (price, S, K, T, r, q, kind)
    p = np.asarray(price, dtype=float)
    S_, K_, T_, r_, _, q_, phi = prepare(S, K, T, r, 1.0, q, kind, allow_degenerate=False)
    p, S_, K_, T_, r_, q_, phi = np.broadcast_arrays(p, S_, K_, T_, r_, q_, phi)
    shape = p.shape

    # Work on flat copies so we can index the active subset cheaply.
    p, S_, K_, T_, r_, q_, phi = (np.array(a, dtype=float).ravel() for a in (p, S_, K_, T_, r_, q_, phi))
    n = p.size

    lower, upper = _no_arbitrage_bounds(S_, K_, T_, r_, q_, phi)
    valid = np.isfinite(p) & (p > lower) & (p < upper)

    # Convert in-the-money quotes to the equivalent out-of-the-money option by parity:
    # for the opposite option type, price = quote - forward intrinsic value.
    itm = lower > 0.0
    phi_o = np.where(itm, -phi, phi)
    p_o = np.where(itm, p - lower, p)
    # An ITM quote whose time value is within rounding noise of the forward intrinsic value
    # carries no volatility information: the subtraction S e^{-qT} - K e^{-rT} is only accurate to
    # a few ulps of its *operands*, and what a solver returns for such a quote depends on how the
    # compiler/CPU rounds it (e.g. fused multiply-add). Reject it instead of returning a number.
    noise = np.where(itm, TIME_VALUE_NOISE * (S_ * np.exp(-q_ * T_) + K_ * np.exp(-r_ * T_)), 0.0)
    valid &= p_o > noise

    # The root must lie inside [sigma_lo, sigma_hi]: check the bracket ends.
    with np.errstate(invalid="ignore"):
        p_at_lo = price_arrays(S_, K_, T_, r_, np.full(n, sigma_lo), q_, phi_o)
        p_at_hi = price_arrays(S_, K_, T_, r_, np.full(n, sigma_hi), q_, phi_o)
        valid &= (p_o > p_at_lo) & (p_o < p_at_hi)

    lo = np.full(n, sigma_lo)
    hi = np.full(n, sigma_hi)
    sigma = np.clip(_initial_guess(np.where(valid, p_o, 1.0), S_, K_, T_, r_, q_), 1e-3, 0.5 * sigma_hi)
    iterations = np.zeros(n, dtype=int)
    converged = np.zeros(n, dtype=bool)
    active = valid.copy()

    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        s = sigma[idx]
        Si, Ki, Ti, ri, qi, phii, pi = S_[idx], K_[idx], T_[idx], r_[idx], q_[idx], phi_o[idx], p_o[idx]

        model = price_arrays(Si, Ki, Ti, ri, s, qi, phii)
        iterations[idx] = it
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.log(model) - np.log(pi)  # log-price residual; -inf if model underflows to 0

        done = np.abs(g) <= tol
        # Tighten the bracket using the sign of the residual (price increases with sigma).
        too_high = g > 0
        hi[idx] = np.where(too_high, s, hi[idx])
        lo[idx] = np.where(too_high, lo[idx], s)
        # Bracket collapsed to floating-point resolution: accept.
        done |= (hi[idx] - lo[idx]) <= 1e-15 * np.maximum(s, 1e-8)

        # Newton step on g(sigma) = ln(model) - ln(p), with g' = vega / model; bisection fallback.
        d1 = (np.log(Si / Ki) + (ri - qi + 0.5 * s**2) * Ti) / (s * np.sqrt(Ti))
        vega = Si * np.exp(-qi * Ti) * norm_pdf(d1) * np.sqrt(Ti)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = s - g * model / vega
        mid = 0.5 * (lo[idx] + hi[idx])
        bad = ~np.isfinite(newton) | (newton <= lo[idx]) | (newton >= hi[idx])
        step = np.where(bad, mid, newton)

        sigma[idx] = np.where(done, s, step)
        converged[idx] = done
        active[idx] = ~done

    iv = np.where(converged, sigma, np.nan).reshape(shape)
    if full_output:
        return IVResult(iv=iv, iterations=iterations.reshape(shape), converged=converged.reshape(shape))
    return squeeze(iv, *raw)
