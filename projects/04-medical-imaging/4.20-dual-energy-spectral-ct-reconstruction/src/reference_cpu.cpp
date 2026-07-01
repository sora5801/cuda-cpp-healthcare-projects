// ===========================================================================
// src/reference_cpu.cpp  --  Scanner physics, data loader, and the trusted
//                            serial DECT decomposition baseline
// ---------------------------------------------------------------------------
// Project 4.20 : Dual-Energy / Spectral CT Reconstruction
//
// ROLE
//   (1) build_spectral_model(): the fixed scanner physics -- two X-ray spectra
//       (~80 kVp low, ~140 kVp high) and two basis-material attenuation curves
//       (soft-tissue/"water" and iodine/"bone"). Deterministic -> reproducible.
//   (2) load_sinogram(): parse the tiny text dataset (data/README.md format).
//   (3) linear_init(): a cheap linearised starting guess for Newton.
//   (4) decompose_cpu(): loop over bins, seed + Newton-solve via the shared
//       __host__ __device__ core decompose_bin() (dect.h). No cleverness on
//       purpose -- if CPU and GPU agree, we trust the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, dect.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::exp, std::pow, std::sqrt
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// build_spectral_model
//
//   We need, at each sampled energy E:
//     * a low- and a high-kVp SPECTRUM weight (how many photons at E), and
//     * the attenuation of the two basis materials at E.
//
//   Rather than ship real tabulated NIST/tube data (large, license-encumbered),
//   we build SMOOTH ANALYTIC approximations that are physically faithful in the
//   ways that matter for the demo:
//     - both spectra span ~30-140 keV;
//     - the 80 kVp spectrum peaks LOW (~50 keV), the 140 kVp peaks HIGH (~75
//       keV) -> the two views weight the energy axis differently, which is the
//       ENTIRE reason dual-energy can separate two materials;
//     - each basis material's mu(E) FALLS with energy (roughly ~E^-3 for the
//       photoelectric part plus a flat Compton floor), and iodine falls much
//       faster than water (its high atomic number Z), giving the two curves
//       DISTINCT energy dependence.
//   These are TEACHING approximations, clearly labelled synthetic (data/README).
//   THEORY "Where this sits in the real world" explains how production tools use
//   measured spectra + NIST XCOM cross-sections instead.
// ---------------------------------------------------------------------------
SpectralModel build_spectral_model() {
    SpectralModel sm;

    // Sample energies from E_MIN..E_MAX keV at NUM_ENERGIES equal points.
    const double E_MIN = 30.0;   // keV (below this, few photons survive filtration)
    const double E_MAX = 140.0;  // keV (tube potential upper bound)
    const double dE = (E_MAX - E_MIN) / (NUM_ENERGIES - 1);

    // A tube spectrum is well modelled by a skewed bump: rising from a low-energy
    // cutoff, peaking near ~1/3-1/2 of the kVp, then falling to the endpoint.
    // We use a simple Gaussian-in-energy bump centred at peak_keV; it is smooth,
    // strictly positive, and easy to normalise. (Not a Kramers spectrum -- a
    // deliberately simple stand-in; see THEORY.)
    auto bump = [](double E, double peak, double width) {
        const double z = (E - peak) / width;
        return std::exp(-0.5 * z * z);
    };

    double sum_lo = 0.0, sum_hi = 0.0;   // for normalising each spectrum to sum 1
    for (int k = 0; k < NUM_ENERGIES; ++k) {
        const double E = E_MIN + dE * k;
        sm.energy_keV[k] = E;

        // Low-kVp (80): peak ~50 keV, moderately narrow (fewer high-E photons).
        sm.w_lo[k] = bump(E, 50.0, 14.0);
        // High-kVp (140): peak ~75 keV, broader (reaches to 140 keV).
        sm.w_hi[k] = bump(E, 78.0, 24.0);
        sum_lo += sm.w_lo[k];
        sum_hi += sm.w_hi[k];

        // Basis-material attenuation curves mu(E) [1/cm]. Photoelectric part
        // ~ Z^~4 / E^3 dominates at low E; Compton part is nearly flat. Constants
        // are chosen so magnitudes and the water/iodine contrast RATIO are
        // realistic for a teaching demo (iodine ~10x water at 40 keV, converging
        // toward water at 140 keV -- exactly the DECT separation lever).
        const double E3 = (E / 60.0) * (E / 60.0) * (E / 60.0);  // (E/60keV)^3
        // Material 1 = soft tissue / water-like: low Z, gentle energy slope.
        sm.mu1[k] = 0.020 / E3 + 0.18;
        // Material 2 = iodine / bone-like: high Z, steep energy slope (big
        // photoelectric term) -> strong low-energy contrast that fades at high E.
        sm.mu2[k] = 1.900 / E3 + 0.32;
    }

    // Normalise each spectrum so its weights sum to 1. Then the "integral S_e dE"
    // denominator in the forward model (dect.h eq. 3) is exactly 1 and drops out,
    // and both spectra are on equal footing.
    for (int k = 0; k < NUM_ENERGIES; ++k) {
        sm.w_lo[k] /= sum_lo;
        sm.w_hi[k] /= sum_hi;
    }
    return sm;
}

// ---------------------------------------------------------------------------
// load_sinogram: read the tiny committed dataset. Format (data/README.md):
//   line 1 : "<n> <has_truth>"
//   next n : "<m_lo> <m_hi>"              (has_truth == 0)  OR
//            "<m_lo> <m_hi> <t1> <t2>"    (has_truth == 1)
//   Blank lines and lines beginning with '#' are ignored (so the sample can be
//   self-documenting). Throws on any malformed content.
// ---------------------------------------------------------------------------
DectSinogram load_sinogram(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sinogram file: " + path);

    // Small helper: fetch the next non-blank, non-comment line.
    auto next_line = [&](std::string& line) -> bool {
        while (std::getline(in, line)) {
            // Trim a trailing '\r' so Windows-CRLF files parse on any platform.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string t = line;
            std::size_t p = t.find_first_not_of(" \t");
            if (p == std::string::npos) continue;          // blank
            if (t[p] == '#') continue;                     // comment
            return true;
        }
        return false;
    };

    std::string line;
    if (!next_line(line)) throw std::runtime_error("empty sinogram file: " + path);
    int n = 0, has_truth = 0;
    {
        std::istringstream hs(line);
        if (!(hs >> n >> has_truth))
            throw std::runtime_error("bad header (expected '<n> <has_truth>') in " + path);
    }
    if (n <= 0) throw std::runtime_error("non-positive bin count in " + path);

    DectSinogram s;
    s.n = n;
    s.m_lo.resize(n);
    s.m_hi.resize(n);
    if (has_truth) { s.true_t1.resize(n); s.true_t2.resize(n); }

    for (int i = 0; i < n; ++i) {
        if (!next_line(line))
            throw std::runtime_error("unexpected end of data (needed " +
                                     std::to_string(n) + " bins) in " + path);
        std::istringstream ls(line);
        if (!(ls >> s.m_lo[i] >> s.m_hi[i]))
            throw std::runtime_error("bad measurement line " + std::to_string(i) + " in " + path);
        if (has_truth) {
            if (!(ls >> s.true_t1[i] >> s.true_t2[i]))
                throw std::runtime_error("bad truth columns on line " +
                                         std::to_string(i) + " in " + path);
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// linear_init: solve the LINEARISED forward model for a starting guess.
//   Define the spectrum-averaged ("effective") attenuation of each material for
//   each spectrum:
//       mubar1_lo = sum_k w_lo[k] * mu1[k]      (etc.)
//   Then for small path lengths f_e ~ mubar1_e*t1 + mubar2_e*t2, a 2x2 LINEAR
//   system in (t1,t2). Solve it in closed form. This lands close to the true
//   root so Newton needs only a handful of iterations. The same seed is used by
//   the GPU (kernels.cu) so both converge identically.
// ---------------------------------------------------------------------------
void linear_init(const SpectralModel& sm, double m_lo, double m_hi,
                 double& t1_init, double& t2_init) {
    // Spectrum-averaged attenuation coefficients (the linear-model matrix A).
    double a_lo_1 = 0.0, a_lo_2 = 0.0, a_hi_1 = 0.0, a_hi_2 = 0.0;
    for (int k = 0; k < NUM_ENERGIES; ++k) {
        a_lo_1 += sm.w_lo[k] * sm.mu1[k];
        a_lo_2 += sm.w_lo[k] * sm.mu2[k];
        a_hi_1 += sm.w_hi[k] * sm.mu1[k];
        a_hi_2 += sm.w_hi[k] * sm.mu2[k];
    }
    // Solve A * (t1,t2)^T = (m_lo, m_hi)^T by Cramer's rule (2x2).
    double det = a_lo_1 * a_hi_2 - a_lo_2 * a_hi_1;
    if (std::fabs(det) < 1e-12) det = (det < 0.0 ? -1e-12 : 1e-12);  // guard
    t1_init = ( a_hi_2 * m_lo - a_lo_2 * m_hi) / det;
    t2_init = (-a_hi_1 * m_lo + a_lo_1 * m_hi) / det;
    // Keep the seed on the physical (non-negative) branch.
    if (t1_init < 0.0) t1_init = 0.0;
    if (t2_init < 0.0) t2_init = 0.0;
}

// ---------------------------------------------------------------------------
// decompose_cpu: the serial reference. For every bin: build the linear seed,
// then run the shared Newton core decompose_bin() from dect.h. Because that core
// is the SAME function the GPU calls, this loop and the kernel produce
// bit-identical (t1,t2) -- the basis of our exact verification.
// ---------------------------------------------------------------------------
void decompose_cpu(const DectSinogram& sino, const SpectralModel& sm,
                   std::vector<double>& t1, std::vector<double>& t2,
                   std::vector<int>& iters) {
    t1.assign(static_cast<std::size_t>(sino.n), 0.0);
    t2.assign(static_cast<std::size_t>(sino.n), 0.0);
    iters.assign(static_cast<std::size_t>(sino.n), 0);
    for (int i = 0; i < sino.n; ++i) {
        double s1, s2;
        linear_init(sm, sino.m_lo[i], sino.m_hi[i], s1, s2);
        const DecompResult r = decompose_bin(sino.m_lo[i], sino.m_hi[i], sm,
                                              s1, s2, MAX_NEWTON_ITER, NEWTON_TOL);
        t1[static_cast<std::size_t>(i)] = r.t1;
        t2[static_cast<std::size_t>(i)] = r.t2;
        iters[static_cast<std::size_t>(i)] = r.iters;
    }
}
