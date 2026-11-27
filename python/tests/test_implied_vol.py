import numpy as np
import pytest
from scipy.optimize import brentq

import bsm


def _grid():
    """Moneyness x maturity x vol x kind grid, flattened to 1-D arrays."""
    m = np.array([0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.4])
    T = np.array([0.02, 0.1, 0.5, 1.0, 3.0])
    sig = np.array([0.05, 0.15, 0.3, 0.6, 1.2, 2.5])
    kinds = np.array(["call", "put"])
    M, TT, SG, KD = np.meshgrid(m, T, sig, kinds, indexing="ij")
    return 100.0, 100.0 * M.ravel(), TT.ravel(), SG.ravel(), KD.ravel()


R, Q = 0.04, 0.015


class TestRoundTrip:
    def test_price_is_reproduced_on_grid(self):
        S, K, T, sig, kind = _grid()
        p = bsm.price(S, K, T, R, sig, kind, Q)
        res = bsm.implied_vol(p, S, K, T, R, kind, Q, full_output=True)
        phi = np.where(kind == "call", 1.0, -1.0)
        intrinsic = np.maximum(phi * (S * np.exp(-Q * T) - K * np.exp(-R * T)), 0.0)
        # A price equal to intrinsic value *in floating point* (time value below ~1e-15,
        # deep ITM and short-dated) carries no volatility information: nan is correct.
        has_time_value = p > intrinsic
        assert has_time_value.sum() > 380
        assert res.converged[has_time_value].all()
        assert not res.converged[~has_time_value].any()
        ok = res.converged
        p_back = bsm.price(S, K[ok], T[ok], R, res.iv[ok], kind[ok], Q)
        assert np.max(np.abs(p_back - p[ok]) / p[ok]) < 1e-9

    def test_vol_is_recovered_where_it_is_identifiable(self):
        S, K, T, sig, kind = _grid()
        p = bsm.price(S, K, T, R, sig, kind, Q)
        iv = bsm.implied_vol(p, S, K, T, R, kind, Q)
        vega = bsm.vega(S, K, T, R, sig, Q)
        identifiable = vega > 1e-3  # otherwise a 1e-16 price error moves sigma noticeably
        assert identifiable.sum() > 100
        assert np.nanmax(np.abs(iv[identifiable] - sig[identifiable])) < 1e-7

    def test_agrees_with_brent_reference(self):
        rng = np.random.default_rng(7)
        n = 200
        S = np.full(n, 100.0)
        K = rng.uniform(70, 140, n)
        T = rng.uniform(0.1, 2.0, n)
        sig = rng.uniform(0.1, 0.8, n)
        kind = np.where(rng.random(n) < 0.5, "call", "put")
        p = bsm.price(S, K, T, R, sig, kind, Q)
        iv = bsm.implied_vol(p, S, K, T, R, kind, Q)
        for i in range(n):
            ref = brentq(lambda s: bsm.price(S[i], K[i], T[i], R, s, kind[i], Q) - p[i], 1e-6, 5.0, xtol=1e-14)
            assert iv[i] == pytest.approx(ref, abs=1e-8)

    def test_call_and_put_give_same_vol_via_parity(self):
        S, K, T, sig = 100.0, np.linspace(80, 125, 19), 0.75, 0.33
        c = bsm.price(S, K, T, R, sig, "call", Q)
        p = bsm.price(S, K, T, R, sig, "put", Q)
        iv_c = bsm.implied_vol(c, S, K, T, R, "call", Q)
        iv_p = bsm.implied_vol(p, S, K, T, R, "put", Q)
        assert np.allclose(iv_c, iv_p, atol=1e-8)
        assert np.allclose(iv_c, sig, atol=1e-8)

    def test_mixed_kinds_in_one_call(self):
        kinds = np.array(["call", "put", "c", "p"])
        p = bsm.price(100, 105, 1.0, R, 0.25, kinds, Q)
        assert np.allclose(bsm.implied_vol(p, 100, 105, 1.0, R, kinds, Q), 0.25, atol=1e-9)


class TestConvergenceBehaviour:
    def test_near_atm_converges_in_few_iterations(self):
        K = np.linspace(95, 105, 21)
        p = bsm.price(100, K, 0.5, R, 0.2, "call", Q)
        res = bsm.implied_vol(p, 100, K, 0.5, R, "call", Q, full_output=True)
        assert res.converged.all()
        assert res.iterations.max() <= 15
        assert res.iterations.mean() <= 9

    def test_wings_still_converge_within_cap(self):
        # Deep OTM / short maturity: vega is tiny, plain Newton would diverge.
        K = np.array([60.0, 70.0, 150.0, 200.0])
        p = bsm.price(100, K, 0.05, R, 0.4, "call", Q)
        keep = p > 1e-300
        res = bsm.implied_vol(p[keep], 100, K[keep], 0.05, R, "call", Q, full_output=True)
        assert res.converged.all()
        assert res.iterations.max() < 100


class TestInvalidPrices:
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_arbitrage_violations_give_nan(self, kind):
        S, K, T = 100.0, 100.0, 1.0
        disc_s, disc_k = S * np.exp(-Q * T), K * np.exp(-R * T)
        lower = max((disc_s - disc_k) if kind == "call" else (disc_k - disc_s), 0.0)
        upper = disc_s if kind == "call" else disc_k
        for bad in (lower, lower - 0.5, upper, upper + 1.0, 0.0, -1.0, np.nan, np.inf):
            assert np.isnan(bsm.implied_vol(bad, S, K, T, R, kind, Q))

    def test_price_below_intrinsic_for_itm_call(self):
        # Deep ITM call worth less than S - K e^{-rT} is an arbitrage.
        assert np.isnan(bsm.implied_vol(5.0, 120, 100, 1.0, 0.05, "call"))

    def test_vol_outside_bracket_gives_nan(self):
        p = bsm.price(100, 100, 1.0, R, 3.0, "call", Q)  # 300% vol
        assert np.isnan(bsm.implied_vol(p, 100, 100, 1.0, R, "call", Q, sigma_hi=2.0))
        assert bsm.implied_vol(p, 100, 100, 1.0, R, "call", Q) == pytest.approx(3.0, abs=1e-8)

    def test_one_bad_element_does_not_poison_the_rest(self):
        good = bsm.price(100, 100, 1.0, R, 0.3, "call", Q)
        out = bsm.implied_vol(np.array([good, -1.0, good]), 100, 100, 1.0, R, "call", Q)
        assert np.isnan(out[1]) and out[0] == pytest.approx(0.3, abs=1e-9) and out[2] == out[0]

    def test_invalid_contract_terms_raise(self):
        with pytest.raises(ValueError):
            bsm.implied_vol(5.0, 100, 100, 0.0, 0.05, "call")
        with pytest.raises(ValueError):
            bsm.implied_vol(5.0, 100, -100, 1.0, 0.05, "call")


class TestInterface:
    def test_scalar_in_scalar_out(self):
        p = bsm.price(100, 100, 1.0, R, 0.2, "call", Q)
        assert isinstance(bsm.implied_vol(p, 100, 100, 1.0, R, "call", Q), float)

    def test_shape_is_preserved(self):
        K = np.linspace(90, 110, 6).reshape(2, 3)
        p = bsm.price(100, K, 1.0, R, 0.25, "put", Q)
        assert bsm.implied_vol(p, 100, K, 1.0, R, "put", Q).shape == (2, 3)
