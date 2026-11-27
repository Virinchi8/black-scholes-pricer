"""Analytic Greeks are verified against central finite differences of the price."""

import numpy as np
import pytest

import bsm


def _fd(f, x, h):
    """Fourth-order (five-point) central difference of f at x with step h.

    A plain two-point stencil has O(h^2) truncation error, which is visible
    against very small Greeks (deep out-of-the-money deltas ~1e-4).
    """
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12.0 * h)


def _subset(random_params, n=300):
    S, K, T, r, sigma, q = (a[:n] for a in random_params)
    return S, K, T, r, sigma, q


@pytest.mark.parametrize("kind", ["call", "put"])
class TestFiniteDifferences:
    def test_delta(self, random_params, kind):
        S, K, T, r, sigma, q = _subset(random_params)
        h = 1e-4 * S
        fd = _fd(lambda s: bsm.price(s, K, T, r, sigma, kind, q), S, h)
        assert np.allclose(bsm.delta(S, K, T, r, sigma, kind, q), fd, rtol=1e-6, atol=1e-8)

    def test_gamma(self, random_params, kind):
        S, K, T, r, sigma, q = _subset(random_params)
        h = 1e-3 * S
        fd = _fd(lambda s: bsm.delta(s, K, T, r, sigma, kind, q), S, h)
        assert np.allclose(bsm.gamma(S, K, T, r, sigma, q), fd, rtol=1e-5, atol=1e-9)

    def test_vega(self, random_params, kind):
        S, K, T, r, sigma, q = _subset(random_params)
        h = 1e-5
        fd = _fd(lambda s: bsm.price(S, K, T, r, s, kind, q), sigma, h)
        assert np.allclose(bsm.vega(S, K, T, r, sigma, q), fd, rtol=1e-6, atol=1e-8)

    def test_theta_is_minus_dV_dT(self, random_params, kind):
        S, K, T, r, sigma, q = _subset(random_params)
        h = 1e-5
        fd = -_fd(lambda t: bsm.price(S, K, t, r, sigma, kind, q), T, h)
        assert np.allclose(bsm.theta(S, K, T, r, sigma, kind, q), fd, rtol=1e-5, atol=1e-7)

    def test_rho(self, random_params, kind):
        S, K, T, r, sigma, q = _subset(random_params)
        h = 1e-6
        fd = _fd(lambda rr: bsm.price(S, K, T, rr, sigma, kind, q), r, h)
        assert np.allclose(bsm.rho(S, K, T, r, sigma, kind, q), fd, rtol=1e-6, atol=1e-7)


class TestSecondOrder:
    def test_vanna(self, random_params):
        S, K, T, r, sigma, q = _subset(random_params)
        # vanna = d delta / d sigma
        fd = _fd(lambda s: bsm.delta(S, K, T, r, s, "call", q), sigma, 1e-5)
        assert np.allclose(bsm.vanna(S, K, T, r, sigma, q), fd, rtol=1e-5, atol=1e-8)

    def test_vanna_equals_dvega_dS(self, random_params):
        S, K, T, r, sigma, q = _subset(random_params)
        fd = _fd(lambda s: bsm.vega(s, K, T, r, sigma, q), S, 1e-3 * S)
        assert np.allclose(bsm.vanna(S, K, T, r, sigma, q), fd, rtol=1e-5, atol=1e-8)

    def test_volga(self, random_params):
        S, K, T, r, sigma, q = _subset(random_params)
        fd = _fd(lambda s: bsm.vega(S, K, T, r, s, q), sigma, 1e-5)
        assert np.allclose(bsm.volga(S, K, T, r, sigma, q), fd, rtol=1e-5, atol=1e-7)


class TestCallPutRelations:
    """Differentiating put-call parity gives exact relations between call and put Greeks."""

    def test_delta_difference(self, random_params):
        S, K, T, r, sigma, q = random_params
        d = bsm.delta(S, K, T, r, sigma, "call", q) - bsm.delta(S, K, T, r, sigma, "put", q)
        assert np.allclose(d, np.exp(-q * T), atol=1e-13)

    def test_theta_difference(self, random_params):
        S, K, T, r, sigma, q = random_params
        d = bsm.theta(S, K, T, r, sigma, "call", q) - bsm.theta(S, K, T, r, sigma, "put", q)
        expected = q * S * np.exp(-q * T) - r * K * np.exp(-r * T)
        assert np.allclose(d, expected, rtol=1e-10, atol=1e-10)

    def test_rho_difference(self, random_params):
        S, K, T, r, sigma, q = random_params
        d = bsm.rho(S, K, T, r, sigma, "call", q) - bsm.rho(S, K, T, r, sigma, "put", q)
        assert np.allclose(d, K * T * np.exp(-r * T), rtol=1e-10, atol=1e-10)


class TestSignsAndBounds:
    def test_signs(self, random_params):
        S, K, T, r, sigma, q = random_params
        assert np.all(bsm.gamma(S, K, T, r, sigma, q) > 0)
        assert np.all(bsm.vega(S, K, T, r, sigma, q) > 0)
        dc = bsm.delta(S, K, T, r, sigma, "call", q)
        dp = bsm.delta(S, K, T, r, sigma, "put", q)
        assert np.all((dc > 0) & (dc < 1)) and np.all((dp < 0) & (dp > -1))
        assert np.all(bsm.rho(S, K, T, r, sigma, "call", q) > 0)
        assert np.all(bsm.rho(S, K, T, r, sigma, "put", q) < 0)

    def test_forward_atm_call_delta(self):
        # At K = forward, d1 = sigma sqrt(T)/2, so delta = N(0.1) for sigma=0.2, T=1.
        d = bsm.delta(100, 100 * np.exp(0.05), 1.0, 0.05, 0.2, "call")
        assert d == pytest.approx(0.5398278372770290, abs=1e-12)

    def test_greeks_dict_has_all_keys(self):
        g = bsm.greeks(100, 100, 1, 0.05, 0.2, "put", 0.01)
        assert set(g) == {"delta", "gamma", "vega", "theta", "rho", "vanna", "volga"}

    def test_requires_positive_vol_and_maturity(self):
        with pytest.raises(ValueError):
            bsm.gamma(100, 100, 0.0, 0.05, 0.2)
        with pytest.raises(ValueError):
            bsm.vega(100, 100, 1.0, 0.05, 0.0)
