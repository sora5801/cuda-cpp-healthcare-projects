// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Jacobi SCF dipole solver (truth)
// ---------------------------------------------------------------------------
// Project 2.27 : Polarizable Water Model GPU Dynamics
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- plain serial loops, no parallelism, no cleverness
//   -- so that when the GPU and CPU agree we believe the GPU. Every per-pair
//   arithmetic operation comes from the SHARED header polar.h, so the GPU kernel
//   (kernels.cu) runs the exact same math and the dipoles agree to round-off.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// THE ALGORITHM (Jacobi self-consistent field; THEORY.md §Algorithm)
//   1. E^perm_i  <- field at site i from all permanent charges + external field
//                   (computed ONCE; independent of the dipoles).
//   2. mu_i      <- alpha_i * E^perm_i                 (zeroth guess, "direct")
//   3. repeat:
//        E^dip_i <- sum_{j!=i} T_ij . mu_j             (field of the OTHER dipoles)
//        mu_i'   <- alpha_i * (E^perm_i + E^dip_i)     (re-induce every site)
//        dmu     <- max_i,component |mu_i' - mu_i|
//        mu      <- mu'                                 (Jacobi: update all at once)
//      until dmu <= tol or max_iters reached.
//   4. U_pol     <- -1/2 sum_i mu_i . E^perm_i          (induction energy)
//
//   "Jacobi" means every site is updated from the PREVIOUS sweep's dipoles (we
//   read mu, write mu_next, then swap). That is exactly the data pattern that
//   parallelizes: each site's update is independent within a sweep, so the GPU
//   kernel gives one site to one thread. (Gauss-Seidel would converge faster but
//   is inherently sequential -- a deliberate teaching trade-off, see THEORY.md.)
//
// READ THIS AFTER: reference_cpu.h, polar.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>      // std::fabs, std::fmax
#include <fstream>    // std::ifstream
#include <sstream>    // std::istringstream (token stream after stripping comments)
#include <stdexcept>  // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_system: parse the tiny whitespace text format (see data/README.md):
//
//   line 1 : N a_thole max_iters tol  Eext_x Eext_y Eext_z
//   next N : x y z q alpha            (one polarizable/charged site per line)
//
// Lines beginning with '#' are comments and are skipped. We read token-by-token
// after stripping comments so the sample file can be liberally annotated.
// ---------------------------------------------------------------------------
PolarSystem load_system(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open system file: " + path);

    // Strip '#' comments and blank lines into a single token stream. This keeps
    // the parser trivial (operator>>) while letting the data file teach.
    std::string cleaned;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        cleaned += line;
        cleaned += '\n';
    }
    std::istringstream ss(cleaned);

    PolarSystem sys;
    int N = 0;
    if (!(ss >> N >> sys.a_thole >> sys.max_iters >> sys.tol
             >> sys.Eext.x >> sys.Eext.y >> sys.Eext.z))
        throw std::runtime_error("bad header (expected 'N a_thole max_iters tol "
                                 "Eext_x Eext_y Eext_z') in " + path);
    if (N <= 0 || sys.max_iters <= 0 || sys.tol <= 0.0)
        throw std::runtime_error("invalid system parameters in " + path);

    sys.sites.resize(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        Site& s = sys.sites[static_cast<std::size_t>(i)];
        if (!(ss >> s.pos.x >> s.pos.y >> s.pos.z >> s.q >> s.alpha))
            throw std::runtime_error("bad site line " + std::to_string(i) + " in " + path);
        if (s.alpha < 0.0)
            throw std::runtime_error("negative polarizability at site " + std::to_string(i));
    }
    return sys;
}

// ---------------------------------------------------------------------------
// compute_permanent_field_cpu: E^perm_i = Eext + sum_{j!=i} q_j r_ij / r_ij^3.
//   O(N^2): every site sees every other site's permanent charge. Computed once
//   per configuration; the SCF loop reuses it. The GPU has a structurally
//   identical kernel (permanent_field_kernel) -- both call field_perm_pair().
// ---------------------------------------------------------------------------
void compute_permanent_field_cpu(const PolarSystem& sys, std::vector<Vec3>& Eperm) {
    const int N = num_sites(sys);
    Eperm.assign(static_cast<std::size_t>(N), Vec3{0.0, 0.0, 0.0});
    for (int i = 0; i < N; ++i) {
        Vec3 E = sys.Eext;                         // start from the uniform external field
        const Vec3 pi = sys.sites[static_cast<std::size_t>(i)].pos;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;                  // a charge exerts no field on itself
            const Site& sj = sys.sites[static_cast<std::size_t>(j)];
            if (sj.q == 0.0) continue;             // skip neutral sites (no permanent field)
            E = vadd(E, field_perm_pair(pi, sj.pos, sj.q));
        }
        Eperm[static_cast<std::size_t>(i)] = E;
    }
}

// ---------------------------------------------------------------------------
// solve_dipoles_cpu: the Jacobi SCF loop described in the file header.
// ---------------------------------------------------------------------------
SolveResult solve_dipoles_cpu(const PolarSystem& sys) {
    const int N = num_sites(sys);

    // Step 1: permanent field (fixed across the SCF loop).
    std::vector<Vec3> Eperm;
    compute_permanent_field_cpu(sys, Eperm);

    // Step 2: direct (zeroth) guess mu_i = alpha_i * E^perm_i. mu_next is the
    // Jacobi scratch buffer we write while reading mu (ping-pong via swap).
    std::vector<Vec3> mu(static_cast<std::size_t>(N));
    std::vector<Vec3> mu_next(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        const double a = sys.sites[static_cast<std::size_t>(i)].alpha;
        mu[static_cast<std::size_t>(i)] = vscale(Eperm[static_cast<std::size_t>(i)], a);
    }

    // Step 3: iterate to self-consistency.
    int iter = 0;
    double dmu_max = 0.0;
    for (iter = 1; iter <= sys.max_iters; ++iter) {
        dmu_max = 0.0;
        for (int i = 0; i < N; ++i) {
            const Site& si = sys.sites[static_cast<std::size_t>(i)];
            // Field at i from every OTHER induced dipole (this sweep's mu).
            Vec3 Edip{0.0, 0.0, 0.0};
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                const Site& sj = sys.sites[static_cast<std::size_t>(j)];
                if (sj.alpha == 0.0) continue;     // pure fixed charges carry no dipole
                Edip = vadd(Edip, field_dip_pair(si.pos, sj.pos,
                                                 mu[static_cast<std::size_t>(j)],
                                                 si.alpha, sj.alpha, sys.a_thole));
            }
            // Re-induce: mu_i = alpha_i * (E^perm_i + E^dip_i).
            const Vec3 Etot = vadd(Eperm[static_cast<std::size_t>(i)], Edip);
            const Vec3 mnew = vscale(Etot, si.alpha);
            mu_next[static_cast<std::size_t>(i)] = mnew;

            // Track the largest per-component change for the convergence test.
            const Vec3 d = vsub(mnew, mu[static_cast<std::size_t>(i)]);
            dmu_max = std::fmax(dmu_max, std::fabs(d.x));
            dmu_max = std::fmax(dmu_max, std::fabs(d.y));
            dmu_max = std::fmax(dmu_max, std::fabs(d.z));
        }
        mu.swap(mu_next);                          // commit this sweep's dipoles
        if (dmu_max <= sys.tol) break;             // converged
    }
    if (iter > sys.max_iters) iter = sys.max_iters; // loop ran to the cap

    // Step 4: polarization energy U_pol = -1/2 sum_i mu_i . E^perm_i.
    double U = 0.0;
    for (int i = 0; i < N; ++i)
        U += polarization_energy_site(mu[static_cast<std::size_t>(i)],
                                      Eperm[static_cast<std::size_t>(i)]);

    SolveResult r;
    r.mu = std::move(mu);
    r.iters = iter;
    r.final_dmu = dmu_max;
    r.U_pol = U;
    r.U_pol_kcal = energy_to_kcal_per_mol(U);
    return r;
}
