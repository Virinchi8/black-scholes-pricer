// Self-contained test runner (no external dependencies, so it builds offline).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bsm/black_scholes.hpp"

using bsm::Greeks;
using bsm::OptionType;

namespace {

int g_checks = 0;
int g_failures = 0;
const char* g_current = "";

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        ++g_checks;                                                                      \
        if (!(cond)) {                                                                   \
            ++g_failures;                                                                \
            std::printf("  FAIL [%s] %s:%d  %s\n", g_current, __FILE__, __LINE__, #cond); \
        }                                                                                \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                          \
    do {                                                                                               \
        ++g_checks;                                                                                    \
        const double _a = (a), _b = (b), _t = (tol);                                                   \
        if (!(std::abs(_a - _b) <= _t)) {                                                              \
            ++g_failures;                                                                              \
            std::printf("  FAIL [%s] %s:%d  |%s - %s| = %.3e > %.3e\n", g_current, __FILE__, __LINE__, \
                        #a, #b, std::abs(_a - _b), _t);                                                \
        }                                                                                              \
    } while (0)

#define CHECK_THROWS(expr)                                                                \
    do {                                                                                  \
        ++g_checks;                                                                       \
        bool _threw = false;                                                              \
        try { (void)(expr); } catch (const std::invalid_argument&) { _threw = true; }     \
        if (!_threw) {                                                                    \
            ++g_failures;                                                                 \
            std::printf("  FAIL [%s] %s:%d  expected invalid_argument: %s\n", g_current, \
                        __FILE__, __LINE__, #expr);                                       \
        }                                                                                 \
    } while (0)

struct Case {
    OptionType type;
    double S, K, T, r, sigma, q;
};

std::vector<Case> random_cases(unsigned seed, int n) {
    std::mt19937_64 rng(seed);
    auto u = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
    std::vector<Case> v;
    for (int i = 0; i < n; ++i) {
        const double S = u(20, 200);
        v.push_back({i % 2 ? OptionType::Put : OptionType::Call, S, S * u(0.6, 1.6), u(0.05, 3.0),
                     u(-0.01, 0.10), u(0.05, 1.2), u(0.0, 0.05)});
    }
    return v;
}

double px(const Case& c) { return bsm::price(c.type, c.S, c.K, c.T, c.r, c.sigma, c.q); }

// Five-point central difference.
double fd5(const std::function<double(double)>& f, double x, double h) {
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12.0 * h);
}

void test_known_values() {
    g_current = "known values";
    CHECK_NEAR(bsm::price(OptionType::Call, 42, 40, 0.5, 0.10, 0.20), 4.7594, 5e-5);  // Hull
    CHECK_NEAR(bsm::price(OptionType::Put, 42, 40, 0.5, 0.10, 0.20), 0.8086, 5e-5);
    CHECK_NEAR(bsm::price(OptionType::Call, 100, 100, 1, 0.05, 0.20), 10.4506, 5e-5);
    CHECK_NEAR(bsm::price(OptionType::Put, 100, 100, 1, 0.05, 0.20), 5.5735, 5e-5);
    CHECK_NEAR(bsm::norm_cdf(0.0), 0.5, 1e-16);
    CHECK_NEAR(bsm::norm_cdf(1.96), 0.9750021048517795, 1e-15);
    CHECK_NEAR(bsm::norm_pdf(0.0), 0.3989422804014327, 1e-16);
}

void test_put_call_parity() {
    g_current = "put-call parity";
    double worst = 0;
    for (const auto& c : random_cases(1, 5000)) {
        const double call = bsm::price(OptionType::Call, c.S, c.K, c.T, c.r, c.sigma, c.q);
        const double put = bsm::price(OptionType::Put, c.S, c.K, c.T, c.r, c.sigma, c.q);
        const double rhs = c.S * std::exp(-c.q * c.T) - c.K * std::exp(-c.r * c.T);
        worst = std::max(worst, std::abs(call - put - rhs) / (c.S * std::exp(-c.q * c.T)));
    }
    CHECK(worst < 1e-13);
    std::printf("  max relative parity gap over 5000 cases: %.2e\n", worst);
}

void test_degenerate_and_bounds() {
    g_current = "degenerate limits and bounds";
    CHECK_NEAR(bsm::price(OptionType::Call, 110, 100, 0.0, 0.05, 0.2), 10.0, 1e-14);
    CHECK_NEAR(bsm::price(OptionType::Call, 90, 100, 0.0, 0.05, 0.2), 0.0, 1e-14);
    CHECK_NEAR(bsm::price(OptionType::Put, 90, 100, 0.0, 0.05, 0.2), 10.0, 1e-14);
    const double fwd = std::max(100 * std::exp(-0.02) - 95 * std::exp(-0.05), 0.0);
    CHECK_NEAR(bsm::price(OptionType::Call, 100, 95, 1.0, 0.05, 0.0, 0.02), fwd, 1e-13);
    for (const auto& c : random_cases(2, 2000)) {
        const double p = px(c);
        const double ds = c.S * std::exp(-c.q * c.T), dk = c.K * std::exp(-c.r * c.T);
        const double lo = std::max(c.type == OptionType::Call ? ds - dk : dk - ds, 0.0);
        const double hi = c.type == OptionType::Call ? ds : dk;
        CHECK(p >= lo - 1e-12 * c.S && p <= hi + 1e-12 * c.S);
    }
}

void test_greeks_finite_differences() {
    g_current = "greeks vs finite differences";
    double worst_rel[7] = {0};
    for (const auto& c : random_cases(3, 400)) {
        const Greeks g = bsm::greeks(c.type, c.S, c.K, c.T, c.r, c.sigma, c.q);
        auto P = [&](double S, double K, double T, double r, double s) {
            return bsm::price(c.type, S, K, T, r, s, c.q);
        };
        const double fd_delta = fd5([&](double s) { return P(s, c.K, c.T, c.r, c.sigma); }, c.S, 1e-3 * c.S);
        const double fd_gamma = fd5(
            [&](double s) { return bsm::greeks(c.type, s, c.K, c.T, c.r, c.sigma, c.q).delta; }, c.S, 1e-3 * c.S);
        const double fd_vega = fd5([&](double s) { return P(c.S, c.K, c.T, c.r, s); }, c.sigma, 1e-4);
        const double fd_theta = -fd5([&](double t) { return P(c.S, c.K, t, c.r, c.sigma); }, c.T, 1e-3);
        const double fd_rho = fd5([&](double r) { return P(c.S, c.K, c.T, r, c.sigma); }, c.r, 1e-4);
        const double fd_vanna = fd5(
            [&](double s) { return bsm::greeks(c.type, c.S, c.K, c.T, c.r, s, c.q).delta; }, c.sigma, 1e-4);
        const double fd_volga = fd5(
            [&](double s) { return bsm::greeks(c.type, c.S, c.K, c.T, c.r, s, c.q).vega; }, c.sigma, 1e-4);
        const double an[7] = {g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga};
        const double fd[7] = {fd_delta, fd_gamma, fd_vega, fd_theta, fd_rho, fd_vanna, fd_volga};
        for (int i = 0; i < 7; ++i)
            worst_rel[i] = std::max(worst_rel[i], std::abs(an[i] - fd[i]) / (std::abs(fd[i]) + 1e-3));
    }
    const char* names[7] = {"delta", "gamma", "vega", "theta", "rho", "vanna", "volga"};
    for (int i = 0; i < 7; ++i) {
        CHECK(worst_rel[i] < 1e-5);
        std::printf("  %-5s worst |analytic - fd| / (|fd| + 1e-3): %.2e\n", names[i], worst_rel[i]);
    }
}

void test_call_put_relations() {
    g_current = "call/put Greek relations";
    for (const auto& c : random_cases(4, 1000)) {
        const Greeks a = bsm::greeks(OptionType::Call, c.S, c.K, c.T, c.r, c.sigma, c.q);
        const Greeks b = bsm::greeks(OptionType::Put, c.S, c.K, c.T, c.r, c.sigma, c.q);
        CHECK_NEAR(a.delta - b.delta, std::exp(-c.q * c.T), 1e-13);
        CHECK_NEAR(a.gamma, b.gamma, 1e-15);
        CHECK_NEAR(a.vega, b.vega, 1e-12);
        CHECK_NEAR(a.theta - b.theta, c.q * c.S * std::exp(-c.q * c.T) - c.r * c.K * std::exp(-c.r * c.T), 1e-10);
        CHECK_NEAR(a.rho - b.rho, c.K * c.T * std::exp(-c.r * c.T), 1e-10);
        CHECK(a.gamma > 0 && a.vega > 0);
    }
}

void test_implied_vol_roundtrip() {
    g_current = "implied vol";
    int total = 0, converged = 0, max_it = 0;
    for (const auto& c : random_cases(5, 5000)) {
        const double p = px(c);
        const bsm::IVResult r = bsm::implied_vol(c.type, p, c.S, c.K, c.T, c.r, c.q);
        ++total;
        const double vega = bsm::greeks(c.type, c.S, c.K, c.T, c.r, c.sigma, c.q).vega;
        if (r.converged) {
            ++converged;
            max_it = std::max(max_it, r.iterations);
            // The recovered vol must reproduce the quote.
            CHECK_NEAR(px({c.type, c.S, c.K, c.T, c.r, r.iv, c.q}), p, 1e-9 * p);
            if (vega > 1e-3) CHECK_NEAR(r.iv, c.sigma, 1e-7);
        } else {
            // Only acceptable if the quote carries no volatility information: its time value is within
            // rounding noise (8 ulps of the operands) of the forward intrinsic value.
            const double ds = c.S * std::exp(-c.q * c.T), dk = c.K * std::exp(-c.r * c.T);
            const double lo = std::max(c.type == OptionType::Call ? ds - dk : dk - ds, 0.0);
            CHECK(p - lo <= 16.0 * std::numeric_limits<double>::epsilon() * (ds + dk));
        }
    }
    std::printf("  %d/%d converged, max iterations %d\n", converged, total, max_it);
    CHECK(converged > 0.95 * total);
}

void test_implied_vol_wings_and_invalid() {
    g_current = "implied vol wings and invalid quotes";
    // Far out of the money, tiny prices: plain Newton on price would fail here.
    // (Only the OTM option: for a deep-ITM quote the time value is below double resolution, so
    // there is no volatility information left and NaN is the correct answer.)
    for (double K : {40.0, 60.0, 150.0, 250.0}) {
        const OptionType type = K > 100.0 ? OptionType::Call : OptionType::Put;
        const double p = bsm::price(type, 100, K, 0.1, 0.04, 0.12, 0.0);
        CHECK(p > 0.0 && p < 1e-20);  // genuinely tiny quotes
        const bsm::IVResult r = bsm::implied_vol(type, p, 100, K, 0.1, 0.04);
        CHECK(r.converged);
        CHECK_NEAR(r.iv, 0.12, 1e-6);
    }
    const double ds = 100 * std::exp(-0.015), dk = 100 * std::exp(-0.04);
    for (auto type : {OptionType::Call, OptionType::Put}) {
        // The exact lower boundary (forward intrinsic value) must come from the library's own
        // arithmetic: recomputing exp() here can differ by one ulp (e.g. compile-time constant
        // folding vs the runtime libm), which would turn "at intrinsic" into "1 ulp above".
        const double lo = bsm::price(type, 100, 100, 1.0, 0.04, 0.0, 0.015);
        const double hi = type == OptionType::Call ? ds : dk;
        for (double bad : std::vector<double>{lo, lo - 0.5, hi, hi + 1.0, 0.0, -1.0, std::nan(""),
                                              std::numeric_limits<double>::infinity()}) {
            const bsm::IVResult r = bsm::implied_vol(type, bad, 100, 100, 1.0, 0.04, 0.015);
            CHECK(!r.converged && std::isnan(r.iv));
        }
    }
    // Vol outside the search bracket is rejected, inside it is found.
    const double p = bsm::price(OptionType::Call, 100, 100, 1.0, 0.04, 3.0, 0.015);
    CHECK(!bsm::implied_vol(OptionType::Call, p, 100, 100, 1.0, 0.04, 0.015, 1e-12, 100, 1e-8, 2.0).converged);
    CHECK_NEAR(bsm::implied_vol(OptionType::Call, p, 100, 100, 1.0, 0.04, 0.015).iv, 3.0, 1e-8);
}

void test_noise_level_quotes_are_rejected() {
    g_current = "noise-level quotes";
    // A quote within a few ulps of intrinsic value must be rejected on every compiler/CPU, never
    // "solved": the result for it would otherwise depend on fused multiply-add and rounding.
    const double eps = std::numeric_limits<double>::epsilon();
    for (auto type : {OptionType::Call, OptionType::Put}) {
        const double K = type == OptionType::Call ? 90.0 : 110.0;
        const double intrinsic = bsm::price(type, 100, K, 0.5, 0.04, 0.0, 0.015);
        for (int ulps : {0, 1, 2, 4, 7}) {
            const bsm::IVResult r =
                bsm::implied_vol(type, intrinsic * (1.0 + ulps * eps), 100, K, 0.5, 0.04, 0.015);
            CHECK(!r.converged && std::isnan(r.iv));
        }
        // Genuine time value is accepted.
        const double quote = bsm::price(type, 100, K, 0.5, 0.04, 0.25, 0.015);
        const bsm::IVResult ok = bsm::implied_vol(type, quote, 100, K, 0.5, 0.04, 0.015);
        CHECK(ok.converged);
        CHECK_NEAR(ok.iv, 0.25, 1e-9);
    }
}

void test_input_validation() {
    g_current = "input validation";
    CHECK_THROWS(bsm::price(OptionType::Call, -1, 100, 1, 0.05, 0.2));
    CHECK_THROWS(bsm::price(OptionType::Call, 100, 0, 1, 0.05, 0.2));
    CHECK_THROWS(bsm::price(OptionType::Call, 100, 100, -1, 0.05, 0.2));
    CHECK_THROWS(bsm::price(OptionType::Call, 100, 100, 1, 0.05, -0.2));
    CHECK_THROWS(bsm::price(OptionType::Call, std::nan(""), 100, 1, 0.05, 0.2));
    CHECK_THROWS(bsm::greeks(OptionType::Call, 100, 100, 0.0, 0.05, 0.2));
    CHECK_THROWS(bsm::greeks(OptionType::Call, 100, 100, 1.0, 0.05, 0.0));
    CHECK_THROWS(bsm::implied_vol(OptionType::Call, 5.0, 100, 100, 0.0, 0.05));
    CHECK_THROWS(bsm::implied_vol(OptionType::Call, 5.0, 100, -100, 1.0, 0.05));
}

// ---- 50-digit reference values shared with the Python tests -------------------------------------
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

void test_against_reference_csv() {
    g_current = "reference CSV";
#ifdef BSM_REFERENCE_CSV
    std::ifstream in(BSM_REFERENCE_CSV);
    CHECK(in.good());
    std::string line;
    std::getline(in, line);  // header
    int rows = 0;
    double worst_price = 0, worst_greek = 0, worst_iv = 0;
    double per_greek[7] = {0}, scale[7] = {0};  // max absolute error, max |reference value|
    while (std::getline(in, line)) {
        const auto f = split(line);
        if (f.size() < 15) continue;
        const OptionType type = f[0] == "call" ? OptionType::Call : OptionType::Put;
        double v[14];
        for (int i = 0; i < 14; ++i) v[i] = std::stod(f[i + 1]);
        const double S = v[0], K = v[1], T = v[2], r = v[3], q = v[4], sigma = v[5];
        const double ref_price = v[6];
        const double ref[7] = {v[7], v[8], v[9], v[10], v[11], v[12], v[13]};  // delta..volga

        worst_price = std::max(worst_price, std::abs(bsm::price(type, S, K, T, r, sigma, q) - ref_price) / ref_price);
        const Greeks g = bsm::greeks(type, S, K, T, r, sigma, q);
        const double got[7] = {g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga};
        for (int i = 0; i < 7; ++i) {
            per_greek[i] = std::max(per_greek[i], std::abs(got[i] - ref[i]));
            scale[i] = std::max(scale[i], std::abs(ref[i]));
        }
        const bsm::IVResult iv = bsm::implied_vol(type, ref_price, S, K, T, r, q);
        CHECK(iv.converged);
        worst_iv = std::max(worst_iv, std::abs(iv.iv - sigma));
        ++rows;
    }
    {
        const char* names[7] = {"delta", "gamma", "vega", "theta", "rho", "vanna", "volga"};
        for (int i = 0; i < 7; ++i) {
            const double rel = per_greek[i] / scale[i];  // absolute error relative to the column's largest value
            worst_greek = std::max(worst_greek, rel);
            std::printf("  %-5s max abs error / max |value| vs 50-digit reference: %.2e\n", names[i], rel);
        }
    }
    CHECK(rows >= 500);
    CHECK(worst_price < 1e-13);
    CHECK(worst_greek < 1e-12);
    CHECK(worst_iv < 1e-9);
    std::printf("  %d rows: worst price rel err %.2e, worst Greek err %.2e, worst |iv - sigma| %.2e\n", rows,
                worst_price, worst_greek, worst_iv);
#else
    std::printf("  (skipped: BSM_REFERENCE_CSV not defined)\n");
#endif
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests = {
        {"known values", test_known_values},
        {"put-call parity", test_put_call_parity},
        {"degenerate limits and bounds", test_degenerate_and_bounds},
        {"greeks vs finite differences", test_greeks_finite_differences},
        {"call/put Greek relations", test_call_put_relations},
        {"implied vol round trip", test_implied_vol_roundtrip},
        {"implied vol wings and invalid quotes", test_implied_vol_wings_and_invalid},
        {"noise-level quotes", test_noise_level_quotes_are_rejected},
        {"input validation", test_input_validation},
        {"reference CSV", test_against_reference_csv},
    };
    for (const auto& [name, fn] : tests) {
        std::printf("%s\n", name);
        fn();
    }
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
