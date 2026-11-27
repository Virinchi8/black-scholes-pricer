// Small command-line front end:
//   bsm_cli price  <call|put> S K T r sigma [q]
//   bsm_cli greeks <call|put> S K T r sigma [q]
//   bsm_cli iv     <call|put> market_price S K T r [q]
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "bsm/black_scholes.hpp"

namespace {

void usage() {
    std::fputs(
        "usage:\n"
        "  bsm_cli price  <call|put> S K T r sigma [q]\n"
        "  bsm_cli greeks <call|put> S K T r sigma [q]\n"
        "  bsm_cli iv     <call|put> market_price S K T r [q]\n",
        stderr);
}

bool parse_type(const std::string& s, bsm::OptionType& out) {
    if (s == "call" || s == "c") { out = bsm::OptionType::Call; return true; }
    if (s == "put" || s == "p") { out = bsm::OptionType::Put; return true; }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }
    const std::string cmd = argv[1];
    bsm::OptionType type;
    if (!parse_type(argv[2], type)) { usage(); return 2; }

    try {
        auto num = [&](int i, double fallback, bool required) {
            if (i < argc) return std::stod(argv[i]);
            if (required) throw std::invalid_argument("missing argument");
            return fallback;
        };
        if (cmd == "price" || cmd == "greeks") {
            const double S = num(3, 0, true), K = num(4, 0, true), T = num(5, 0, true);
            const double r = num(6, 0, true), sigma = num(7, 0, true), q = num(8, 0.0, false);
            if (cmd == "price") {
                std::printf("%.12f\n", bsm::price(type, S, K, T, r, sigma, q));
            } else {
                const bsm::Greeks g = bsm::greeks(type, S, K, T, r, sigma, q);
                std::printf("delta %.10f\ngamma %.10f\nvega  %.10f\ntheta %.10f\nrho   %.10f\nvanna %.10f\nvolga %.10f\n",
                            g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga);
            }
        } else if (cmd == "iv") {
            const double p = num(3, 0, true), S = num(4, 0, true), K = num(5, 0, true);
            const double T = num(6, 0, true), r = num(7, 0, true), q = num(8, 0.0, false);
            const bsm::IVResult res = bsm::implied_vol(type, p, S, K, T, r, q);
            if (!res.converged) {
                std::fputs("no implied volatility (price outside no-arbitrage bounds or solver failed)\n", stderr);
                return 1;
            }
            std::printf("%.12f  (%d iterations)\n", res.iv, res.iterations);
        } else {
            usage();
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return 0;
}
