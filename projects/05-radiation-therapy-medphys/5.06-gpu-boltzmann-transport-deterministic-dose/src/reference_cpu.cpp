// ===========================================================================
// src/reference_cpu.cpp  --  Loader, Gauss-Legendre quadrature, serial S_N solver
// ---------------------------------------------------------------------------
// Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- readable loops, no parallelism, no cleverness -- so that
//   when the GPU and CPU agree, we believe the GPU. The per-cell transport
//   physics is the shared boltzmann_sn.h; here we wrap it in source iteration.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, boltzmann_sn.h. Twin: kernels.cu (GPU).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill
#include <cmath>       // std::cos, std::fabs, M_PI fallback
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

#ifndef M_PI
#define M_PI 3.14159265358979323846   // some MSVC configs omit M_PI
#endif

// ---------------------------------------------------------------------------
// make_gauss_legendre: nodes/weights of the nord-point Gauss-Legendre rule on
//   [-1,1], the standard S_N angular quadrature. We find each root of the
//   Legendre polynomial P_n by Newton's method starting from the classic
//   cos(pi (k-0.25)/(n+0.5)) initial guess, then set the weight from the exact
//   formula w = 2 / ((1 - x^2) P_n'(x)^2). Computing it (instead of hardcoding a
//   table) keeps the code self-contained and works for any even nord.
//
//   Why this quadrature: it exactly integrates the scalar-flux angle integral
//   (2) for the angular polynomials that arise from isotropic scattering, and
//   sum(w) = 2 so a flat unit angular flux integrates to phi = 2 (= |[-1,1]|).
// ---------------------------------------------------------------------------
SnQuadrature make_gauss_legendre(int nord) {
    if (nord < 2 || (nord % 2) != 0)
        throw std::runtime_error("S_N order (nord) must be even and >= 2");

    SnQuadrature q;
    q.mu.assign(nord, 0.0);
    q.w.assign(nord, 0.0);

    const int    m = (nord + 1) / 2;     // roots are symmetric; compute half, mirror
    const double eps = 1e-15;            // Newton convergence: full double precision

    for (int k = 0; k < m; ++k) {
        // Initial guess for the k-th root (well-separated, converges in a few steps).
        double x = std::cos(M_PI * (k + 0.75) / (nord + 0.5));
        double dp = 0.0;                 // P_n'(x), filled by the recurrence below

        for (int it = 0; it < 100; ++it) {
            // Evaluate the Legendre polynomial P_n(x) and its derivative via the
            // stable three-term recurrence:
            //   (j+1) P_{j+1} = (2j+1) x P_j - j P_{j-1}.
            double p0 = 1.0;             // P_0(x)
            double p1 = x;               // P_1(x)
            for (int j = 2; j <= nord; ++j) {
                double p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j;
                p0 = p1; p1 = p2;        // shift window up by one degree
            }
            // Derivative from P_n and P_{n-1}: P_n'(x) = n (x P_n - P_{n-1})/(x^2-1).
            dp = nord * (x * p1 - p0) / (x * x - 1.0);
            double dx = -p1 / dp;        // Newton step: x <- x - P_n/P_n'
            x += dx;
            if (std::fabs(dx) < eps) break;
        }

        // Fill the +/- symmetric pair. Store negative node first so the array is
        // ascending in mu -- purely cosmetic, keeps output tidy.
        const double w = 2.0 / ((1.0 - x * x) * dp * dp);   // exact G-L weight
        q.mu[k]            = -x;   q.w[k]            = w;
        q.mu[nord - 1 - k] =  x;   q.w[nord - 1 - k] = w;
    }
    return q;
}

// ---------------------------------------------------------------------------
// load_slab: parse the SlabProblem text format (see data/README.md).
//   Fails LOUDLY (throws) on any malformed field so a demo never runs on junk.
// ---------------------------------------------------------------------------
SlabProblem load_slab(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open slab problem file: " + path);

    SlabProblem p;
    if (!(in >> p.ncell >> p.nord >> p.width >> p.max_iter >> p.tol
            >> p.psi_left_bc >> p.psi_right_bc))
        throw std::runtime_error(
            "bad header (expected 'ncell nord width max_iter tol psiL psiR') in " + path);

    if (p.ncell <= 0 || p.nord < 2 || (p.nord % 2) != 0 || p.width <= 0.0
        || p.max_iter <= 0 || p.tol <= 0.0)
        throw std::runtime_error("invalid slab header values in " + path);

    p.sigma_t.assign(p.ncell, 0.0);
    p.sigma_s.assign(p.ncell, 0.0);
    p.q.assign(p.ncell, 0.0);
    for (int i = 0; i < p.ncell; ++i) {
        if (!(in >> p.sigma_t[i] >> p.sigma_s[i] >> p.q[i]))
            throw std::runtime_error("not enough cell rows (need ncell) in " + path);
        // Physical sanity: scattering cannot exceed total (that would mean
        // negative absorption). Catching this here beats debugging NaNs later.
        if (p.sigma_t[i] < 0.0 || p.sigma_s[i] < 0.0 || p.sigma_s[i] > p.sigma_t[i])
            throw std::runtime_error("cell violates 0 <= sigma_s <= sigma_t in " + path);
    }
    return p;
}

// ---------------------------------------------------------------------------
// solve_sn_cpu: SOURCE ITERATION around the transport sweep.
//   Algorithm (mirrors kernels.cu exactly, so results match):
//     1. phi <- 0 (initial guess for the scalar flux).
//     2. Repeat until converged or max_iter:
//          a. build the cell source Q from the CURRENT phi (lagged),
//          b. SWEEP every ordinate across the slab, accumulating
//             phi_new[i] = sum_n w_n * psi_avg_{n,i}  (equation 3),
//          c. measure the change ||phi_new - phi||_inf / ||phi_new||_inf,
//          d. phi <- phi_new.
//   Complexity: O(iters * nord * ncell). Source iteration converges
//   geometrically with rate ~ c = max(Sigma_s/Sigma_t) (the scattering ratio):
//   more scattering -> slower convergence (that is what DSA accelerates; see
//   THEORY §real-world).
// ---------------------------------------------------------------------------
void solve_sn_cpu(const SlabProblem& p, const SnQuadrature& quad,
                  std::vector<double>& phi, int& iters) {
    const int n = p.ncell;
    const double h = p.h();

    phi.assign(n, 0.0);                 // (1) start from a zero-flux guess
    std::vector<double> phi_new(n, 0.0);

    iters = 0;
    for (int it = 0; it < p.max_iter; ++it) {
        // (2b) fresh accumulator; each ordinate ADDS its weighted share into it,
        // in a FIXED ordinate order -> deterministic, matches the GPU reduction.
        std::fill(phi_new.begin(), phi_new.end(), 0.0);
        for (int nq = 0; nq < p.nord; ++nq) {
            sn_sweep_one_ordinate(quad.mu[nq], quad.w[nq], n, h,
                                  p.sigma_t.data(), p.sigma_s.data(), p.q.data(),
                                  phi.data(),                 // lagged scalar flux
                                  p.psi_left_bc, p.psi_right_bc,
                                  phi_new.data());
        }

        // (2c) relative L-infinity change; the convergence test that stops SI.
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = std::fabs(phi_new[i] - phi[i]);
            if (d > num) num = d;
            const double a = std::fabs(phi_new[i]);
            if (a > den) den = a;
        }
        phi.swap(phi_new);              // (2d) accept the new iterate (O(1) swap)
        ++iters;

        const double rel = (den > 0.0) ? (num / den) : num;
        if (rel <= p.tol) break;        // converged: further sweeps would not move phi
    }
}

// ---------------------------------------------------------------------------
// deposition_field: dep[i] = Sigma_a[i] * phi[i], Sigma_a = Sigma_t - Sigma_s.
//   The rate at which particles are absorbed per unit volume -- our dose proxy.
//   Shared by CPU and GPU reporting so the "dose" columns agree exactly.
// ---------------------------------------------------------------------------
void deposition_field(const SlabProblem& p, const std::vector<double>& phi,
                      std::vector<double>& dep) {
    dep.assign(p.ncell, 0.0);
    for (int i = 0; i < p.ncell; ++i) {
        const double sigma_a = p.sigma_t[i] - p.sigma_s[i];   // absorption x-section
        dep[i] = sigma_a * phi[i];
    }
}
