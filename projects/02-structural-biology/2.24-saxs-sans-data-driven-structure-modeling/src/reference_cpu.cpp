// ===========================================================================
// src/reference_cpu.cpp  --  Plain-C++ baseline: Debye profile + SAXS analysis
// ---------------------------------------------------------------------------
// Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU is checked against. debye_profile_cpu() is a
//   single readable serial loop over q that calls the SAME per-q physics
//   (saxs_core.h) the GPU kernel uses, so when the two agree we trust the GPU.
//   The file also holds the host-only analysis (least-squares scale, reduced
//   chi-square, Guinier Rg) and the sample loader -- none of which need a GPU.
//
//   Compiled by the host C++ compiler only (no CUDA syntax here).
//
// READ THIS AFTER: reference_cpu.h and saxs_core.h. Compare debye_profile_cpu()
//   with debye_kernel in kernels.cu (its GPU twin -- same loop, one thread per q).
// ===========================================================================
#include "reference_cpu.h"
#include "saxs_core.h"        // debye_intensity_at_q (shared CPU/GPU physics)

#include <cmath>              // std::log, std::sqrt
#include <fstream>            // std::ifstream
#include <sstream>            // tokenising the header line
#include <stdexcept>          // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// debye_profile_cpu: forward-model I(q) at every q, serially.
//   Complexity O(n_q * n_atoms^2). Each q is computed by debye_intensity_at_q()
//   from saxs_core.h -- the identical function the GPU thread calls -- so the two
//   results differ only by floating-point summation order (tiny; see TOLERANCE
//   in main.cu and PATTERNS.md §4).
// ---------------------------------------------------------------------------
void debye_profile_cpu(const SaxsModel& m, std::vector<double>& I_model) {
    I_model.assign(static_cast<std::size_t>(m.n_q), 0.0);
    for (int k = 0; k < m.n_q; ++k) {
        // One independent intensity per q value: this is exactly the work item
        // the GPU parallelizes (one thread per k). On the CPU we just loop.
        I_model[static_cast<std::size_t>(k)] =
            debye_intensity_at_q(m.q[static_cast<std::size_t>(k)],
                                 m.x.data(), m.y.data(), m.z.data(),
                                 m.f.data(), m.n_atoms);
    }
}

// ---------------------------------------------------------------------------
// best_scale: weighted-least-squares scale c minimizing chi^2(c*model, exp).
//   d/dc of sum((c*I - E)/s)^2 = 0  ->  c = sum(I*E/s^2) / sum(I^2/s^2).
//   This puts the arbitrary-units model onto the experimental scale with ONE
//   degree of freedom (real SAXS fitting also floats a flat background; we keep
//   just the scale for clarity -- see THEORY §"real world").
// ---------------------------------------------------------------------------
double best_scale(const std::vector<double>& I_model,
                  const std::vector<double>& I_exp,
                  const std::vector<double>& sigma) {
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < I_model.size(); ++k) {
        const double w = 1.0 / (sigma[k] * sigma[k]);   // inverse-variance weight
        num += I_model[k] * I_exp[k] * w;
        den += I_model[k] * I_model[k] * w;
    }
    return (den > 0.0) ? num / den : 0.0;
}

// ---------------------------------------------------------------------------
// reduced_chi_square: chi^2 / n_q for the scaled model. ~1 => fits within noise.
// ---------------------------------------------------------------------------
double reduced_chi_square(const std::vector<double>& I_model, double c,
                          const std::vector<double>& I_exp,
                          const std::vector<double>& sigma) {
    double chi2 = 0.0;
    for (std::size_t k = 0; k < I_model.size(); ++k) {
        const double resid = (c * I_model[k] - I_exp[k]) / sigma[k];  // standardized residual
        chi2 += resid * resid;
    }
    return I_model.empty() ? 0.0 : chi2 / static_cast<double>(I_model.size());
}

// ---------------------------------------------------------------------------
// guinier_rg: Rg from a linear fit of ln I vs q^2 over the first n_fit points.
//   Guinier law: ln I(q) ≈ ln I(0) - (Rg^2/3) q^2. The slope of ln I against q^2
//   is -Rg^2/3, so Rg = sqrt(-3 * slope). We fit by ordinary least squares.
//   This recovers a real structural number (the molecule's size) from the curve.
// ---------------------------------------------------------------------------
double guinier_rg(const std::vector<double>& q, const std::vector<double>& I,
                  int n_fit) {
    // Clamp the fit window to the available points and require at least 2.
    if (n_fit > static_cast<int>(q.size())) n_fit = static_cast<int>(q.size());
    if (n_fit < 2) return -1.0;

    // Accumulate the sums needed for a least-squares line  Y = a + b*X,
    // with X = q^2 and Y = ln I. The slope b carries Rg.
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    int used = 0;
    for (int k = 0; k < n_fit; ++k) {
        if (I[static_cast<std::size_t>(k)] <= 0.0) continue;   // ln undefined for <=0
        const double X = q[static_cast<std::size_t>(k)] * q[static_cast<std::size_t>(k)];
        const double Y = std::log(I[static_cast<std::size_t>(k)]);
        sx += X; sy += Y; sxx += X * X; sxy += X * Y; ++used;
    }
    if (used < 2) return -1.0;
    const double denom = used * sxx - sx * sx;
    if (denom == 0.0) return -1.0;
    const double slope = (used * sxy - sx * sy) / denom;   // = -Rg^2/3
    if (slope >= 0.0) return -1.0;                         // unphysical (noisy fit)
    return std::sqrt(-3.0 * slope);
}

// ---------------------------------------------------------------------------
// load_model: parse the text sample format (see data/README.md). Layout:
//   line 1 : "n_atoms  n_q  true_rg"
//   next n_atoms lines : "x y z f"          (Å, Å, Å, electron count)
//   next n_q   lines  : "q  I_exp  sigma"   (1/Å, intensity, error bar)
//   '#' comment lines and blank lines are skipped. We read the whole file token
//   by token after stripping comments, which tolerates arbitrary whitespace.
// ---------------------------------------------------------------------------
SaxsModel load_model(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open SAXS model file: " + path);

    // Concatenate all non-comment content into one token stream.
    std::ostringstream cleaned;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t hash = line.find('#');         // strip trailing comments
        if (hash != std::string::npos) line.erase(hash);
        cleaned << line << ' ';
    }
    std::istringstream ts(cleaned.str());

    SaxsModel m;
    if (!(ts >> m.n_atoms >> m.n_q >> m.true_rg))
        throw std::runtime_error("malformed header (need: n_atoms n_q true_rg)");
    if (m.n_atoms <= 0 || m.n_q <= 0)
        throw std::runtime_error("n_atoms and n_q must be positive");

    // Atoms.
    m.x.resize(m.n_atoms); m.y.resize(m.n_atoms);
    m.z.resize(m.n_atoms); m.f.resize(m.n_atoms);
    for (int i = 0; i < m.n_atoms; ++i) {
        if (!(ts >> m.x[i] >> m.y[i] >> m.z[i] >> m.f[i]))
            throw std::runtime_error("truncated atom record");
    }

    // q grid + experimental curve.
    m.q.resize(m.n_q); m.I_exp.resize(m.n_q); m.sigma.resize(m.n_q);
    for (int k = 0; k < m.n_q; ++k) {
        if (!(ts >> m.q[k] >> m.I_exp[k] >> m.sigma[k]))
            throw std::runtime_error("truncated q/intensity record");
        if (m.sigma[k] <= 0.0)
            throw std::runtime_error("sigma must be > 0 (it is a divisor)");
    }
    return m;
}
