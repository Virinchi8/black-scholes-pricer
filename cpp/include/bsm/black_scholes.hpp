// Black-Scholes-Merton pricing, analytic Greeks and an implied-volatility solver.
//
// European options on an asset with continuous dividend yield q:
//   d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
//   call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
//   put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
//
// Greek conventions match the Python package: raw derivatives, vega per 1.00 of
// volatility, rho per 1.00 of rate, theta = dV/dt per year (negative for long options).
#pragma once

#include <limits>

namespace bsm {

enum class OptionType { Call = 1, Put = -1 };

struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    double vanna;  // d2V / (dS dsigma)
    double volga;  // d2V / dsigma^2
};

struct IVResult {
    double iv;       // NaN when no solution exists
    int iterations;  // iterations used (0 if the quote was rejected up front)
    bool converged;
};

double norm_pdf(double x) noexcept;
double norm_cdf(double x) noexcept;

// Price of a European option. T == 0 or sigma == 0 return the discounted forward
// intrinsic value. Throws std::invalid_argument for S <= 0, K <= 0, T < 0, sigma < 0
// or non-finite inputs.
double price(OptionType type, double S, double K, double T, double r, double sigma, double q = 0.0);

// Analytic Greeks. Requires T > 0 and sigma > 0 (throws std::invalid_argument otherwise).
Greeks greeks(OptionType type, double S, double K, double T, double r, double sigma, double q = 0.0);

// Implied volatility via safeguarded Newton on log-price, working on the
// out-of-the-money side (in-the-money quotes are converted by put-call parity).
// Returns iv = NaN, converged = false for quotes with no solution: price at or below
// intrinsic value, at or above the upper no-arbitrage bound, non-finite, an
// in-the-money quote whose time value is within rounding noise (8 ulps of S e^{-qT} + K e^{-rT}), or
// an implied vol outside [sigma_lo, sigma_hi]. Throws std::invalid_argument for invalid
// contract terms (S <= 0, K <= 0, T <= 0, non-finite inputs).
IVResult implied_vol(OptionType type, double market_price, double S, double K, double T, double r,
                     double q = 0.0, double tol = 1e-12, int max_iter = 100, double sigma_lo = 1e-8,
                     double sigma_hi = 10.0);

}  // namespace bsm
