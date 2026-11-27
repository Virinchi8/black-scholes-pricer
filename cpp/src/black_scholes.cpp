#include "bsm/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bsm {

namespace {

constexpr double kInvSqrt2Pi = 0.39894228040143267794;  // 1 / sqrt(2 pi)
constexpr double kInvSqrt2 = 0.70710678118654752440;    // 1 / sqrt(2)
constexpr double kTwoPi = 6.28318530717958647692;

inline double sign(OptionType t) { return t == OptionType::Call ? 1.0 : -1.0; }

void require_finite_positive_SK(double S, double K) {
    if (!std::isfinite(S) || !std::isfinite(K) || S <= 0.0 || K <= 0.0)
        throw std::invalid_argument("S and K must be finite and strictly positive");
}

void require_finite(double T, double r, double sigma, double q) {
    if (!std::isfinite(T) || !std::isfinite(r) || !std::isfinite(sigma) || !std::isfinite(q))
        throw std::invalid_argument("T, r, sigma and q must be finite");
}

// Core formula on validated inputs. phi = +1 call, -1 put.
double price_core(double phi, double S, double K, double T, double r, double sigma, double q) {
    const double disc_s = S * std::exp(-q * T);
    const double disc_k = K * std::exp(-r * T);
    const double vol_sqrt_t = sigma * std::sqrt(T);
    if (vol_sqrt_t <= 0.0) return std::max(phi * (disc_s - disc_k), 0.0);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t;
    const double d2 = d1 - vol_sqrt_t;
    return phi * (disc_s * norm_cdf(phi * d1) - disc_k * norm_cdf(phi * d2));
}

// Starting volatility: max(0.5 * Manaster-Koltveit, Brenner-Subrahmanyam ATM approximation).
double initial_guess(double p, double S, double K, double T, double r, double q) {
    const double disc_s = S * std::exp(-q * T);
    const double x = std::abs(std::log(S / K) + (r - q) * T);
    const double mk = std::sqrt(2.0 * x / T);
    const double atm = std::sqrt(kTwoPi / T) * p / disc_s;
    return std::max(0.5 * mk, atm);
}

}  // namespace

double norm_pdf(double x) noexcept { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double norm_cdf(double x) noexcept { return 0.5 * std::erfc(-x * kInvSqrt2); }

double price(OptionType type, double S, double K, double T, double r, double sigma, double q) {
    require_finite_positive_SK(S, K);
    require_finite(T, r, sigma, q);
    if (T < 0.0 || sigma < 0.0) throw std::invalid_argument("T and sigma must be non-negative");
    return price_core(sign(type), S, K, T, r, sigma, q);
}

Greeks greeks(OptionType type, double S, double K, double T, double r, double sigma, double q) {
    require_finite_positive_SK(S, K);
    require_finite(T, r, sigma, q);
    if (T <= 0.0 || sigma <= 0.0) throw std::invalid_argument("T and sigma must be strictly positive for Greeks");

    const double phi = sign(type);
    const double sqrt_t = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t);
    const double d2 = d1 - sigma * sqrt_t;
    const double es = std::exp(-q * T);
    const double ek = std::exp(-r * T);
    const double pdf1 = norm_pdf(d1);

    Greeks g{};
    g.delta = phi * es * norm_cdf(phi * d1);
    g.gamma = es * pdf1 / (S * sigma * sqrt_t);
    g.vega = S * es * pdf1 * sqrt_t;
    g.theta = -S * es * pdf1 * sigma / (2.0 * sqrt_t) - phi * r * K * ek * norm_cdf(phi * d2) +
              phi * q * S * es * norm_cdf(phi * d1);
    g.rho = phi * K * T * ek * norm_cdf(phi * d2);
    g.vanna = -es * pdf1 * d2 / sigma;
    g.volga = g.vega * d1 * d2 / sigma;
    return g;
}

IVResult implied_vol(OptionType type, double market_price, double S, double K, double T, double r, double q,
                     double tol, int max_iter, double sigma_lo, double sigma_hi) {
    require_finite_positive_SK(S, K);
    require_finite(T, r, 0.0, q);
    if (T <= 0.0) throw std::invalid_argument("T must be strictly positive");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const IVResult rejected{nan, 0, false};
    const double phi = sign(type);

    // No-arbitrage bounds: forward intrinsic value <= price < discounted underlying (call) / strike (put).
    const double disc_s = S * std::exp(-q * T);
    const double disc_k = K * std::exp(-r * T);
    const double lower = std::max(phi * (disc_s - disc_k), 0.0);
    const double upper = phi > 0 ? disc_s : disc_k;
    if (!std::isfinite(market_price) || !(market_price > lower) || !(market_price < upper)) return rejected;

    // Convert an in-the-money quote to the out-of-the-money option via put-call parity:
    // the opposite option's price is the quote minus forward intrinsic value (the time value).
    const bool itm = lower > 0.0;
    const double phi_o = itm ? -phi : phi;
    const double p = itm ? market_price - lower : market_price;
    if (!(p > 0.0)) return rejected;

    // The root must lie inside [sigma_lo, sigma_hi].
    if (!(p > price_core(phi_o, S, K, T, r, sigma_lo, q)) || !(p < price_core(phi_o, S, K, T, r, sigma_hi, q)))
        return rejected;

    double lo = sigma_lo, hi = sigma_hi;
    double sigma = std::clamp(initial_guess(p, S, K, T, r, q), 1e-3, 0.5 * sigma_hi);
    const double sqrt_t = std::sqrt(T);
    const double log_p = std::log(p);

    for (int it = 1; it <= max_iter; ++it) {
        const double model = price_core(phi_o, S, K, T, r, sigma, q);
        const double g = std::log(model) - log_p;  // -inf if the model price underflows to 0

        bool done = std::abs(g) <= tol;
        if (g > 0.0) hi = sigma; else lo = sigma;
        done = done || (hi - lo) <= 1e-15 * std::max(sigma, 1e-8);
        if (done) return {sigma, it, true};

        const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t);
        const double vega = S * std::exp(-q * T) * norm_pdf(d1) * sqrt_t;
        double next = sigma - g * model / vega;
        if (!std::isfinite(next) || next <= lo || next >= hi) next = 0.5 * (lo + hi);
        sigma = next;
    }
    return {nan, max_iter, false};
}

}  // namespace bsm
