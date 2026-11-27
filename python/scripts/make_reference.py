"""Generate ``data/reference.csv``: ground-truth option values in 50-digit arithmetic.

Both the Python and the C++ implementation are tested against this file, so each
is checked against high-precision truth rather than only against the other.

    pip install mpmath
    python scripts/make_reference.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from mpmath import exp, log, mp, mpf, ncdf, npdf, sqrt

mp.dps = 50

OUT = Path(__file__).resolve().parents[2] / "data" / "reference.csv"
OUT_NORM = OUT.with_name("normcdf_reference.csv")
OUT.parent.mkdir(exist_ok=True)

COLUMNS = ["kind", "S", "K", "T", "r", "q", "sigma", "price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]


def reference_row(kind, S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = (mpf(repr(float(x))) for x in (S, K, T, r, q, sigma))
    phi = 1 if kind == "call" else -1
    srt = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + sigma**2 / 2) * T) / srt
    d2 = d1 - srt
    es, ek = exp(-q * T), exp(-r * T)
    price = phi * (S * es * ncdf(phi * d1) - K * ek * ncdf(phi * d2))
    delta = phi * es * ncdf(phi * d1)
    gamma = es * npdf(d1) / (S * srt)
    vega = S * es * npdf(d1) * sqrt(T)
    theta = -S * es * npdf(d1) * sigma / (2 * sqrt(T)) - phi * r * K * ek * ncdf(phi * d2) + phi * q * S * es * ncdf(phi * d1)
    rho = phi * K * T * ek * ncdf(phi * d2)
    vanna = -es * npdf(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return [price, delta, gamma, vega, theta, rho, vanna, volga]


def main(n=600, seed=20261127):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = "call" if i % 2 == 0 else "put"
        S = 100.0
        K = float(np.round(100.0 * np.exp(rng.uniform(-0.35, 0.35)), 4))
        T = float(np.round(rng.uniform(0.1, 3.0), 4))
        r = float(np.round(rng.uniform(0.0, 0.08), 4))
        q = float(np.round(rng.uniform(0.0, 0.04), 4))
        sigma = float(np.round(rng.uniform(0.10, 1.00), 4))
        vals = reference_row(kind, S, K, T, r, q, sigma)
        rows.append([kind, S, K, T, r, q, sigma] + [mp.nstr(v, 25) for v in vals])
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


def write_normcdf_reference():
    """N(z) in 50-digit arithmetic on a grid out to the far tails (used by the JavaScript tests)."""
    zs = [i / 8 for i in range(-96, 97)]  # -12 ... 12
    with OUT_NORM.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z", "cdf"])
        for z in zs:
            w.writerow([repr(float(z)), mp.nstr(ncdf(mpf(z)), 25)])
    print(f"wrote {len(zs)} rows to {OUT_NORM}")


if __name__ == "__main__":
    main()
    write_normcdf_reference()
