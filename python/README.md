# bsm (Python package)

Vectorised Black-Scholes-Merton pricing, analytic Greeks and an implied-volatility solver.

```python
import bsm

bsm.price(100, 105, 0.5, 0.04, 0.25, "call", q=0.01)
bsm.greeks(100, 105, 0.5, 0.04, 0.25, "put", q=0.01)
bsm.implied_vol(6.61, 100, 110, 0.5, 0.03, "call")
```

See the [repository README](../README.md) for the method, accuracy results, plots and the C++ port.
