"""Black-Scholes-Merton prices for European options with a continuous dividend yield.

With spot S, strike K, maturity T (years), risk-free rate r, volatility sigma and
dividend yield q (all continuously compounded):

    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)

    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

All functions are vectorised: any argument may be a NumPy array, and arguments
broadcast against each other. ``kind`` may be a single string or an array of strings.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF

from ._common import prepare, squeeze


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes ``(d1, d2)``. Requires T > 0 and sigma > 0."""
    S, K, T, r, sigma, q, _ = prepare(S, K, T, r, sigma, q, "call", allow_degenerate=False)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return squeeze(d1, S, K, T, r, sigma, q), squeeze(d2, S, K, T, r, sigma, q)


def price_arrays(S, K, T, r, sigma, q, phi):
    """Core formula on already-validated, broadcast float arrays (no checks).

    ``phi`` is +1 for calls and -1 for puts. Used directly by the implied-vol
    solver so it can avoid re-validating inputs on every iteration.
    """
    disc_s = S * np.exp(-q * T)  # present value of the underlying
    disc_k = K * np.exp(-r * T)  # present value of the strike

    vol_sqrt_t = sigma * np.sqrt(T)
    degenerate = vol_sqrt_t <= 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    regular = phi * (disc_s * ndtr(phi * d1) - disc_k * ndtr(phi * d2))
    limit = np.maximum(phi * (disc_s - disc_k), 0.0)
    return np.where(degenerate, limit, regular)


def price(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes-Merton price of a European call or put.

    Degenerate cases (T = 0 or sigma = 0) return the discounted forward
    intrinsic value, which is the correct limit of the formula.
    """
    raw_inputs = (S, K, T, r, sigma, q)
    S, K, T, r, sigma, q, phi = prepare(S, K, T, r, sigma, q, kind, allow_degenerate=True)
    out = price_arrays(S, K, T, r, sigma, q, phi)
    return squeeze(out, *raw_inputs, kind)


def call_price(S, K, T, r, sigma, q=0.0):
    """Convenience wrapper: European call."""
    return price(S, K, T, r, sigma, "call", q)


def put_price(S, K, T, r, sigma, q=0.0):
    """Convenience wrapper: European put."""
    return price(S, K, T, r, sigma, "put", q)


def put_call_parity_gap(S, K, T, r, sigma, q=0.0):
    """Return ``C - P - (S e^{-qT} - K e^{-rT})``; zero up to rounding if parity holds."""
    c = price(S, K, T, r, sigma, "call", q)
    p = price(S, K, T, r, sigma, "put", q)
    S, K, T, r, _, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    return c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))
