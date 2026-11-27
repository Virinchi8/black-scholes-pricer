// Throughput benchmark: prices and implied vols for 1,000,000 options.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "bsm/black_scholes.hpp"

int main() {
    const int n = 1'000'000;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ku(-0.4, 0.4), tu(0.05, 2.0), su(0.1, 0.8);
    std::vector<double> K(n), T(n), sig(n), p(n), iv(n);
    for (int i = 0; i < n; ++i) {
        K[i] = 100.0 * std::exp(ku(rng));
        T[i] = tu(rng);
        sig[i] = su(rng);
    }

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    for (int i = 0; i < n; ++i) p[i] = bsm::price(bsm::OptionType::Call, 100.0, K[i], T[i], 0.04, sig[i], 0.0);
    auto t1 = clock::now();
    long long iterations = 0, converged = 0;
    for (int i = 0; i < n; ++i) {
        const bsm::IVResult r = bsm::implied_vol(bsm::OptionType::Call, p[i], 100.0, K[i], T[i], 0.04, 0.0);
        iv[i] = r.iv;
        iterations += r.iterations;
        converged += r.converged;
    }
    auto t2 = clock::now();

    // Accuracy is only meaningful where the quote identifies the volatility (vega not ~0): a deep
    // in-the-money price that equals intrinsic value to within rounding says nothing about sigma.
    double max_err = 0;
    long long identifiable = 0;
    for (int i = 0; i < n; ++i) {
        const double vega = bsm::greeks(bsm::OptionType::Call, 100.0, K[i], T[i], 0.04, sig[i]).vega;
        if (vega > 1e-3 && !std::isnan(iv[i])) {
            ++identifiable;
            max_err = std::max(max_err, std::abs(iv[i] - sig[i]));
        }
    }

    auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    std::printf("price:       %.3f s for %d options\n", sec(t0, t1), n);
    std::printf("implied vol: %.3f s for %d options (mean %.2f iterations)\n", sec(t1, t2), n,
                static_cast<double>(iterations) / n);
    std::printf("  %lld converged; %lld unidentifiable quotes (time value within rounding noise of intrinsic)\n",
                converged, static_cast<long long>(n) - converged);
    std::printf("  max |iv - sigma| over %lld quotes with vega > 1e-3: %.2e\n", identifiable, max_err);
    return 0;
}
