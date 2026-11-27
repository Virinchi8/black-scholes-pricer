"""Analytic Black-Scholes-Merton Greeks.

Conventions (raw mathematical derivatives, *not* trader-scaled):

* ``delta``  dV/dS
* ``gamma``  d2V/dS2
* ``vega``   dV/dsigma, per 1.00 (100 vol points); divide by 100 for "per 1%"
* ``theta``  dV/dt where t is calendar time, per year; divide by 365 for "per day".
  Theta is the *time-decay* sign convention: it is normally negative for long options.
* ``rho``    dV/dr, per 1.00 (100 rate points); divide by 100 for "per 1%"
* ``vanna``  d2V/(dS dsigma)
* ``volga``  d2V/dsigma2  (a.k.a. vomma)

Every function requires T > 0 and sigma > 0 (the Greeks are not smooth at the
degenerate limit).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import ndtr

from ._common import norm_pdf, prepare, squeeze


def _core(S, K, T, r, sigma, q, kind):
    raw = (S, K, T, r, sigma, q, kind)
    S, K, T, r, sigma, q, phi = prepare(S, K, T, r, sigma, q, kind, allow_degenerate=False)
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return raw, (S, K, T, r, sigma, q, phi, sqrt_t, d1, d2)


def delta(S, K, T, r, sigma, kind="call", q=0.0):
    raw, (S, K, T, r, sigma, q, phi, _, d1, _) = _core(S, K, T, r, sigma, q, kind)
    return squeeze(phi * np.exp(-q * T) * ndtr(phi * d1), *raw)


def gamma(S, K, T, r, sigma, q=0.0):
    raw, (S, K, T, r, sigma, q, _, sqrt_t, d1, _) = _core(S, K, T, r, sigma, q, "call")
    return squeeze(np.exp(-q * T) * norm_pdf(d1) / (S * sigma * sqrt_t), *raw[:-1])


def vega(S, K, T, r, sigma, q=0.0):
    raw, (S, K, T, r, sigma, q, _, sqrt_t, d1, _) = _core(S, K, T, r, sigma, q, "call")
    return squeeze(S * np.exp(-q * T) * norm_pdf(d1) * sqrt_t, *raw[:-1])


def theta(S, K, T, r, sigma, kind="call", q=0.0):
    raw, (S, K, T, r, sigma, q, phi, sqrt_t, d1, d2) = _core(S, K, T, r, sigma, q, kind)
    decay = -S * np.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * sqrt_t)
    carry_r = -phi * r * K * np.exp(-r * T) * ndtr(phi * d2)
    carry_q = phi * q * S * np.exp(-q * T) * ndtr(phi * d1)
    return squeeze(decay + carry_r + carry_q, *raw)


def rho(S, K, T, r, sigma, kind="call", q=0.0):
    raw, (S, K, T, r, sigma, q, phi, _, _, d2) = _core(S, K, T, r, sigma, q, kind)
    return squeeze(phi * K * T * np.exp(-r * T) * ndtr(phi * d2), *raw)


def vanna(S, K, T, r, sigma, q=0.0):
    raw, (S, K, T, r, sigma, q, _, _, d1, d2) = _core(S, K, T, r, sigma, q, "call")
    return squeeze(-np.exp(-q * T) * norm_pdf(d1) * d2 / sigma, *raw[:-1])


def volga(S, K, T, r, sigma, q=0.0):
    raw, (S, K, T, r, sigma, q, _, sqrt_t, d1, d2) = _core(S, K, T, r, sigma, q, "call")
    vega_ = S * np.exp(-q * T) * norm_pdf(d1) * sqrt_t
    return squeeze(vega_ * d1 * d2 / sigma, *raw[:-1])


def greeks(S, K, T, r, sigma, kind="call", q=0.0) -> Dict[str, object]:
    """All Greeks in one dictionary (see module docstring for conventions)."""
    return {
        "delta": delta(S, K, T, r, sigma, kind, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, kind, q),
        "rho": rho(S, K, T, r, sigma, kind, q),
        "vanna": vanna(S, K, T, r, sigma, q),
        "volga": volga(S, K, T, r, sigma, q),
    }
