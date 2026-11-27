import numpy as np
import pytest


@pytest.fixture(scope="module")
def random_params():
    """A reproducible cloud of reasonable option parameters (arrays of length 2000)."""
    rng = np.random.default_rng(12345)
    n = 2000
    S = rng.uniform(20, 200, n)
    K = S * rng.uniform(0.6, 1.6, n)
    T = rng.uniform(0.05, 3.0, n)
    r = rng.uniform(-0.01, 0.10, n)
    sigma = rng.uniform(0.05, 1.2, n)
    q = rng.uniform(0.0, 0.05, n)
    return S, K, T, r, sigma, q
