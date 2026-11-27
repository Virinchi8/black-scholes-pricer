"""Check the Python implementation against 50-digit reference values (data/reference.csv)."""

import csv
from pathlib import Path

import numpy as np
import pytest

import bsm

REF = Path(__file__).resolve().parents[2] / "data" / "reference.csv"


@pytest.fixture(scope="module")
def ref():
    with REF.open() as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    kind = cols["kind"]
    num = {k: cols[k].astype(float) for k in cols if k != "kind"}
    return kind, num


def _rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def test_reference_file_is_present_and_sized(ref):
    kind, num = ref
    assert len(kind) >= 500


def test_price(ref):
    kind, c = ref
    p = bsm.price(c["S"], c["K"], c["T"], c["r"], c["sigma"], kind, c["q"])
    assert _rel(p, c["price"]).max() < 1e-13


@pytest.mark.parametrize("name", ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"])
def test_greeks(ref, name):
    kind, c = ref
    args = (c["S"], c["K"], c["T"], c["r"], c["sigma"])
    fn = getattr(bsm, name)
    if name in ("delta", "theta", "rho"):
        out = fn(*args, kind, c["q"])
    else:
        out = fn(*args, c["q"])
    scale = np.max(np.abs(c[name]))
    assert np.max(np.abs(out - c[name])) / scale < 1e-13


def test_implied_vol_recovers_true_sigma(ref):
    kind, c = ref
    iv = bsm.implied_vol(c["price"], c["S"], c["K"], c["T"], c["r"], kind, c["q"])
    assert np.nanmax(np.abs(iv - c["sigma"])) < 1e-9
    assert not np.isnan(iv).any()
