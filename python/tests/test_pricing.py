import numpy as np
import pytest

import bsm


class TestKnownValues:
    def test_hull_textbook_example(self):
        # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5
        assert bsm.price(42, 40, 0.5, 0.10, 0.20, "call") == pytest.approx(4.7594, abs=5e-5)
        assert bsm.price(42, 40, 0.5, 0.10, 0.20, "put") == pytest.approx(0.8086, abs=5e-5)

    def test_standard_atm_one_year(self):
        # S=K=100, r=5%, sigma=20%, T=1 is a widely quoted benchmark.
        assert bsm.call_price(100, 100, 1, 0.05, 0.20) == pytest.approx(10.4506, abs=5e-5)
        assert bsm.put_price(100, 100, 1, 0.05, 0.20) == pytest.approx(5.5735, abs=5e-5)

    def test_dividend_yield_lowers_call(self):
        no_div = bsm.call_price(100, 100, 1, 0.05, 0.2, q=0.0)
        div = bsm.call_price(100, 100, 1, 0.05, 0.2, q=0.03)
        assert div < no_div


class TestPutCallParity:
    """C - P = S e^{-qT} - K e^{-rT} must hold for every parameter set."""

    def test_random_cloud(self, random_params):
        S, K, T, r, sigma, q = random_params
        gap = bsm.put_call_parity_gap(S, K, T, r, sigma, q)
        scale = S * np.exp(-q * T)
        assert np.max(np.abs(gap) / scale) < 1e-13

    @pytest.mark.parametrize("q", [0.0, 0.02])
    def test_explicit_scalar(self, q):
        S, K, T, r, sigma = 100.0, 95.0, 0.75, 0.04, 0.3
        lhs = bsm.price(S, K, T, r, sigma, "call", q) - bsm.price(S, K, T, r, sigma, "put", q)
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=1e-12)

    def test_extreme_parameters(self):
        # Very low and very high vol, very short and long maturities.
        S = np.array([100.0, 100.0, 100.0, 100.0])
        K = np.array([100.0, 50.0, 200.0, 100.0])
        T = np.array([1e-4, 10.0, 0.01, 30.0])
        sigma = np.array([0.01, 2.0, 0.5, 0.05])
        gap = bsm.put_call_parity_gap(S, K, T, 0.03, sigma, 0.01)
        assert np.max(np.abs(gap)) < 1e-10


class TestNoArbitrageBounds:
    def test_call_and_put_within_bounds(self, random_params):
        S, K, T, r, sigma, q = random_params
        c = bsm.price(S, K, T, r, sigma, "call", q)
        p = bsm.price(S, K, T, r, sigma, "put", q)
        disc_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
        tol = 1e-12 * S
        assert np.all(c >= np.maximum(disc_s - disc_k, 0) - tol)
        assert np.all(c <= disc_s + tol)
        assert np.all(p >= np.maximum(disc_k - disc_s, 0) - tol)
        assert np.all(p <= disc_k + tol)

    def test_monotone_in_sigma_and_spot(self):
        sig = np.linspace(0.05, 1.5, 50)
        assert np.all(np.diff(bsm.price(100, 105, 1, 0.03, sig, "call")) > 0)
        assert np.all(np.diff(bsm.price(100, 105, 1, 0.03, sig, "put")) > 0)
        spot = np.linspace(50, 150, 50)
        assert np.all(np.diff(bsm.price(spot, 100, 1, 0.03, 0.3, "call")) > 0)
        assert np.all(np.diff(bsm.price(spot, 100, 1, 0.03, 0.3, "put")) < 0)

    def test_convex_in_strike(self):
        # Butterfly spreads cannot have negative value.
        K = np.linspace(60, 140, 81)
        c = bsm.price(100, K, 0.5, 0.03, 0.25, "call")
        assert np.all(c[:-2] - 2 * c[1:-1] + c[2:] >= -1e-12)


class TestDegenerateLimits:
    def test_zero_maturity_is_intrinsic(self):
        assert bsm.price(110, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(10.0)
        assert bsm.price(90, 100, 0.0, 0.05, 0.2, "call") == 0.0
        assert bsm.price(90, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(10.0)

    def test_zero_vol_is_discounted_forward_intrinsic(self):
        S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.02
        expected = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        assert bsm.price(S, K, T, r, 0.0, "call", q) == pytest.approx(expected)

    def test_small_vol_approaches_zero_vol_limit(self):
        limit = bsm.price(100, 95, 1, 0.05, 0.0, "call")
        assert bsm.price(100, 95, 1, 0.05, 1e-6, "call") == pytest.approx(limit, abs=1e-6)


class TestInterface:
    def test_scalar_in_scalar_out(self):
        assert isinstance(bsm.price(100, 100, 1, 0.05, 0.2, "call"), float)

    def test_array_in_array_out_and_broadcasting(self):
        K = np.array([90.0, 100.0, 110.0])
        out = bsm.price(100, K, 1, 0.05, 0.2, "call")
        assert out.shape == (3,)
        grid = bsm.price(100, K[:, None], np.array([0.5, 1.0])[None, :], 0.05, 0.2, "put")
        assert grid.shape == (3, 2)

    def test_mixed_kinds_in_one_call(self):
        kinds = np.array(["call", "put", "C", "P"])
        out = bsm.price(100, 100, 1, 0.05, 0.2, kinds)
        assert out[0] == pytest.approx(out[2]) and out[1] == pytest.approx(out[3])
        assert out[0] != pytest.approx(out[1])

    def test_kind_is_case_insensitive(self):
        assert bsm.price(100, 100, 1, 0.05, 0.2, "CALL") == bsm.price(100, 100, 1, 0.05, 0.2, "call")

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(S=-1, K=100, T=1, r=0.05, sigma=0.2),
            dict(S=100, K=0, T=1, r=0.05, sigma=0.2),
            dict(S=100, K=100, T=-0.1, r=0.05, sigma=0.2),
            dict(S=100, K=100, T=1, r=0.05, sigma=-0.2),
            dict(S=np.nan, K=100, T=1, r=0.05, sigma=0.2),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            bsm.price(**kwargs)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            bsm.price(100, 100, 1, 0.05, 0.2, "straddle")
        with pytest.raises(ValueError):
            bsm.price(100, 100, 1, 0.05, 0.2, 1)
