"""Shared helpers: input validation, broadcasting and option-type handling."""

from __future__ import annotations

import numpy as np

SQRT_2PI = np.sqrt(2.0 * np.pi)

_CALL_NAMES = {"c", "call"}
_PUT_NAMES = {"p", "put"}


def norm_pdf(x):
    """Standard normal probability density."""
    return np.exp(-0.5 * x * x) / SQRT_2PI


def option_sign(kind):
    """Map option type(s) to phi = +1 (call) / -1 (put).

    ``kind`` may be a single string ("call", "put", "c", "p", case-insensitive)
    or an array-like of such strings, so calls and puts can be mixed in one
    vectorised call.
    """
    arr = np.asarray(kind)
    if arr.dtype.kind not in ("U", "S", "O"):
        raise ValueError("kind must be 'call'/'put' (or 'c'/'p'), got non-string input")
    lowered = np.char.lower(arr.astype(str))
    is_call = np.isin(lowered, list(_CALL_NAMES))
    is_put = np.isin(lowered, list(_PUT_NAMES))
    if not np.all(is_call | is_put):
        raise ValueError("kind must be 'call'/'put' (or 'c'/'p')")
    return np.where(is_call, 1.0, -1.0)


def prepare(S, K, T, r, sigma, q, kind, *, allow_degenerate):
    """Validate and broadcast inputs to a common shape.

    Returns ``(S, K, T, r, sigma, q, phi)`` as float arrays. When
    ``allow_degenerate`` is False, ``T`` and ``sigma`` must be strictly positive
    (needed wherever we divide by ``sigma * sqrt(T)``).
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    phi = option_sign(kind)
    S, K, T, r, sigma, q, phi = np.broadcast_arrays(S, K, T, r, sigma, q, phi)

    for name, v in (("S", S), ("K", K), ("T", T), ("r", r), ("sigma", sigma), ("q", q)):
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be finite")
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive")
    if allow_degenerate:
        if np.any(T < 0) or np.any(sigma < 0):
            raise ValueError("T and sigma must be non-negative")
    else:
        if np.any(T <= 0) or np.any(sigma <= 0):
            raise ValueError("T and sigma must be strictly positive for this function")
    return S, K, T, r, sigma, q, phi


def squeeze(out, *inputs):
    """Return a Python float if every input was a scalar, otherwise the array."""
    if all(np.ndim(x) == 0 for x in inputs):
        return float(out)
    return out
