"""Reproduce every number and figure quoted in the README.

Run from the ``python/`` directory:

    pip install -e ".[dev]"
    python scripts/make_results.py

Writes PNGs and ``results.json`` to ``../docs``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bsm  # noqa: E402

DOCS = Path(__file__).resolve().parents[2] / "docs"
DOCS.mkdir(exist_ok=True)

# Chart palette (light mode, validated: categorical slots 1-3 pass all-pairs CVD checks).
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "font.size": 9,
        "lines.linewidth": 2.0,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)

R, Q, SIGMA, K0 = 0.05, 0.0, 0.20, 100.0
results: dict = {}


# ---------------------------------------------------------------- 1. reference values
results["hull_example"] = {
    "call": bsm.price(42, 40, 0.5, 0.10, 0.20, "call"),
    "put": bsm.price(42, 40, 0.5, 0.10, 0.20, "put"),
}
results["atm_benchmark"] = {
    "call": bsm.price(100, 100, 1, 0.05, 0.20, "call"),
    "put": bsm.price(100, 100, 1, 0.05, 0.20, "put"),
}

# ---------------------------------------------------------------- 2. put-call parity
rng = np.random.default_rng(2026)
n = 200_000
S = rng.uniform(20, 200, n)
K = S * rng.uniform(0.5, 1.8, n)
T = rng.uniform(0.01, 5.0, n)
r = rng.uniform(-0.01, 0.12, n)
sg = rng.uniform(0.03, 1.5, n)
q = rng.uniform(0.0, 0.06, n)
rel_gap = np.abs(bsm.put_call_parity_gap(S, K, T, r, sg, q)) / (S * np.exp(-q * T))
results["parity"] = {"n": n, "max_relative_gap": float(rel_gap.max()), "median_relative_gap": float(np.median(rel_gap))}

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.hist(np.log10(np.maximum(rel_gap, 1e-18)), bins=60, color=BLUE, edgecolor="white", linewidth=0.6)
ax.set_xlabel(r"$\log_{10}\,|C - P - (Se^{-qT} - Ke^{-rT})|\,/\,Se^{-qT}$   (exact zeros plotted at -18)")
ax.set_ylabel("count")
ax.set_title(f"Put-call parity residual over {n:,} random parameter sets")
fig.savefig(DOCS / "fig_parity.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. price and payoff
spot = np.linspace(60, 140, 400)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=False)
for ax, kind in zip(axes, ("call", "put")):
    payoff = np.maximum(spot - K0, 0) if kind == "call" else np.maximum(K0 - spot, 0)
    ax.plot(spot, payoff, color=INK2, linestyle=(0, (4, 3)), label="payoff at expiry")
    ax.plot(spot, bsm.price(spot, K0, 0.25, R, SIGMA, kind), color=BLUE, label="T = 0.25 y")
    ax.plot(spot, bsm.price(spot, K0, 1.0, R, SIGMA, kind), color=ORANGE, linestyle="-.", label="T = 1 y")
    ax.set_title(f"European {kind}  (K=100, r=5%, σ=20%)")
    ax.set_xlabel("spot S")
    ax.set_ylabel("option value")
axes[0].legend(loc="upper left")
fig.tight_layout()
fig.savefig(DOCS / "fig_price_payoff.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. Greeks vs spot
maturities = [(0.1, BLUE, "-"), (0.5, ORANGE, "-."), (1.0, AQUA, (0, (1, 1)))]
fns = [
    ("Delta", lambda s, t: bsm.delta(s, K0, t, R, SIGMA, "call", Q)),
    ("Gamma", lambda s, t: bsm.gamma(s, K0, t, R, SIGMA, Q)),
    ("Vega (per 1.00 vol)", lambda s, t: bsm.vega(s, K0, t, R, SIGMA, Q)),
    ("Theta (per year)", lambda s, t: bsm.theta(s, K0, t, R, SIGMA, "call", Q)),
    ("Rho (per 1.00 rate)", lambda s, t: bsm.rho(s, K0, t, R, SIGMA, "call", Q)),
    ("Vanna", lambda s, t: bsm.vanna(s, K0, t, R, SIGMA, Q)),
]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
spot = np.linspace(60, 140, 400)
for ax, (name, fn) in zip(axes.ravel(), fns):
    for T_, col, ls in maturities:
        ax.plot(spot, fn(spot, T_), color=col, linestyle=ls, label=f"T = {T_:g} y")
    ax.set_title(name)
    ax.set_xlabel("spot S")
axes[0, 0].legend(loc="center left")
fig.suptitle("Call Greeks vs spot  (K=100, r=5%, σ=20%, q=0)", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(DOCS / "fig_greeks.png")
plt.close(fig)

# ---------------------------------------------------------------- 5. Greeks vs finite differences
def fd5(f, x, h):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


m = 2000
Sg, Kg = S[:m], K[:m]
Tg, rg, sgg, qg = T[:m], r[:m], sg[:m], q[:m]
fd_checks = {
    "delta": (bsm.delta(Sg, Kg, Tg, rg, sgg, "call", qg), fd5(lambda s: bsm.price(s, Kg, Tg, rg, sgg, "call", qg), Sg, 1e-3 * Sg)),
    "gamma": (bsm.gamma(Sg, Kg, Tg, rg, sgg, qg), fd5(lambda s: bsm.delta(s, Kg, Tg, rg, sgg, "call", qg), Sg, 1e-3 * Sg)),
    "vega": (bsm.vega(Sg, Kg, Tg, rg, sgg, qg), fd5(lambda s: bsm.price(Sg, Kg, Tg, rg, s, "call", qg), sgg, 1e-3)),
    "theta": (bsm.theta(Sg, Kg, Tg, rg, sgg, "call", qg), -fd5(lambda t: bsm.price(Sg, Kg, t, rg, sgg, "call", qg), Tg, 1e-3)),
    "rho": (bsm.rho(Sg, Kg, Tg, rg, sgg, "call", qg), fd5(lambda x: bsm.price(Sg, Kg, Tg, x, sgg, "call", qg), rg, 1e-4)),
    "vanna": (bsm.vanna(Sg, Kg, Tg, rg, sgg, qg), fd5(lambda s: bsm.delta(Sg, Kg, Tg, rg, s, "call", qg), sgg, 1e-3)),
    "volga": (bsm.volga(Sg, Kg, Tg, rg, sgg, qg), fd5(lambda s: bsm.vega(Sg, Kg, Tg, rg, s, qg), sgg, 1e-3)),
}
results["greeks_vs_finite_difference"] = {}
for name, (analytic, numeric) in fd_checks.items():
    # error relative to the Greek's own scale over the sample (avoids dividing by ~0)
    scale = np.max(np.abs(analytic))
    results["greeks_vs_finite_difference"][name] = {
        "max_abs_error_over_max_abs_value": float(np.max(np.abs(analytic - numeric)) / scale)
    }

# ---------------------------------------------------------------- 5b. vs 50-digit reference
import csv  # noqa: E402

with (DOCS.parent / "data" / "reference.csv").open() as f:
    _rows = list(csv.DictReader(f))
_kind = np.array([r["kind"] for r in _rows])
_c = {k: np.array([float(r[k]) for r in _rows]) for k in _rows[0] if k != "kind"}
_args = (_c["S"], _c["K"], _c["T"], _c["r"], _c["sigma"])
ref_check = {"rows": len(_rows)}
ref_check["price_max_relative_error"] = float(
    np.max(np.abs(bsm.price(*_args, _kind, _c["q"]) - _c["price"]) / _c["price"])
)
for _name in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga"):
    _fn = getattr(bsm, _name)
    _out = _fn(*_args, _kind, _c["q"]) if _name in ("delta", "theta", "rho") else _fn(*_args, _c["q"])
    ref_check[f"{_name}_max_abs_error_over_max_abs_value"] = float(
        np.max(np.abs(_out - _c[_name])) / np.max(np.abs(_c[_name]))
    )
_iv = bsm.implied_vol(_c["price"], _c["S"], _c["K"], _c["T"], _c["r"], _kind, _c["q"])
ref_check["implied_vol_max_abs_error"] = float(np.max(np.abs(_iv - _c["sigma"])))
results["python_vs_50_digit_reference"] = ref_check

# ---------------------------------------------------------------- 6. implied-vol solver
rng = np.random.default_rng(99)
n_iv = 100_000
S_i = np.full(n_iv, 100.0)
K_i = 100.0 * np.exp(rng.uniform(-0.9, 0.9, n_iv))  # log-moneyness out to ±0.9 (K from ~40 to ~246)
T_i = rng.uniform(0.02, 3.0, n_iv)
sig_i = rng.uniform(0.05, 1.5, n_iv)
kind_i = np.where(rng.random(n_iv) < 0.5, "call", "put")
r_i, q_i = 0.04, 0.015
p_i = bsm.price(S_i, K_i, T_i, r_i, sig_i, kind_i, q_i)
res = bsm.implied_vol(p_i, S_i, K_i, T_i, r_i, kind_i, q_i, full_output=True)

phi_i = np.where(kind_i == "call", 1.0, -1.0)
intrinsic = np.maximum(phi_i * (S_i * np.exp(-q_i * T_i) - K_i * np.exp(-r_i * T_i)), 0.0)
has_tv = p_i > intrinsic
vega_i = bsm.vega(S_i, K_i, T_i, r_i, sig_i, q_i)
ident = has_tv & (vega_i > 1e-3)
err = np.abs(res.iv - sig_i)
rel_price_err = np.abs(bsm.price(S_i[res.converged], K_i[res.converged], T_i[res.converged], r_i, res.iv[res.converged], kind_i[res.converged], q_i) - p_i[res.converged]) / p_i[res.converged]
results["implied_vol"] = {
    "n": n_iv,
    "share_with_time_value_in_float64": float(has_tv.mean()),
    "converged_among_those": float(res.converged[has_tv].mean()),
    "unconverged_with_time_value": int((has_tv & ~res.converged).sum()),
    "max_sigma_error_identifiable": float(np.nanmax(err[ident])),
    "max_relative_price_error_all_converged": float(rel_price_err.max()),
    "share_price_error_below_1e-12": float((rel_price_err <= 1e-12).mean()),
    "iterations_mean": float(res.iterations[res.converged].mean()),
    "iterations_median": float(np.median(res.iterations[res.converged])),
    "iterations_p99": float(np.percentile(res.iterations[res.converged], 99)),
    "iterations_max": int(res.iterations[res.converged].max()),
}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
conv = res.converged
counts = np.bincount(res.iterations[conv])
its = np.arange(counts.size)
axes[0].bar(its[counts > 0], counts[counts > 0], color=BLUE, edgecolor="white", linewidth=0.6)
axes[0].set_xlabel("iterations to converge")
axes[0].set_ylabel("count")
axes[0].set_title("Solver iterations (strikes 40-246, T 0.02-3 y)")
e = err[ident]
axes[1].hist(np.log10(np.maximum(e, 1e-17)), bins=60, color=BLUE, edgecolor="white", linewidth=0.6)
axes[1].set_xlabel(r"$\log_{10}\,|\hat\sigma - \sigma|$   (exact matches plotted at -17)")
axes[1].set_ylabel("count")
axes[1].set_title("Recovered-vol error (vega > 1e-3)")
fig.tight_layout()
fig.savefig(DOCS / "fig_iv_solver.png")
plt.close(fig)

# ---------------------------------------------------------------- 7. throughput (indicative)
big = 1_000_000
Sb = np.full(big, 100.0)
Kb = 100.0 * np.exp(rng.uniform(-0.4, 0.4, big))
Tb = rng.uniform(0.05, 2.0, big)
sgb = rng.uniform(0.1, 0.8, big)
t0 = time.perf_counter()
pb = bsm.price(Sb, Kb, Tb, 0.04, sgb, "call", 0.0)
t1 = time.perf_counter()
bsm.implied_vol(pb, Sb, Kb, Tb, 0.04, "call", 0.0)
t2 = time.perf_counter()
results["python_throughput_1M_options"] = {"price_seconds": t1 - t0, "implied_vol_seconds": t2 - t1}

(DOCS / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
