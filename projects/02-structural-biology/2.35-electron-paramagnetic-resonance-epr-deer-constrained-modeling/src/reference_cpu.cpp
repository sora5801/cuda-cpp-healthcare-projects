// ===========================================================================
// src/reference_cpu.cpp  --  CPU reference: loader, shared reweighting helpers,
//                            per-frame DEER back-calculation, max-entropy fit.
// ---------------------------------------------------------------------------
// Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
//
// This is the TRUSTED baseline against which the GPU result is verified. It is
// plain C++ (compiled by cl.exe / g++), and it shares ALL per-element math with
// the GPU via deer.h, so the two paths agree to ~1e-13. The reweighting itself
// (a tiny gradient descent over the M-vector of weights) is intentionally host
// code reused by both paths -- only the heavy per-frame histogram back-calc is
// offloaded to the GPU. See ../THEORY.md "GPU mapping".
//
// READ THIS AFTER: deer.h, reference_cpu.h.  READ BEFORE: main.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sqrt, std::exp, std::log
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_ensemble  --  parse the committed text sample (format in data/README.md)
// ---------------------------------------------------------------------------
// File layout (whitespace-separated, '#' lines ignored):
//   line 1 : M ROTAMERS NBINS         (header; ROTAMERS/NBINS must match build)
//   then for each frame m = 0..M-1:
//       a line:  truth_flag           (1 = synthetic "true" match, else 0)
//       ROTAMERS lines: x y z         (site-A rotamer endpoints, nm)
//       ROTAMERS lines: x y z         (site-B rotamer endpoints, nm)
//   then NBINS lines: target P_exp value per bin (will be re-normalized).
//
// We validate the header against the compiled constants so a mismatched sample
// fails loudly rather than silently producing garbage.
// ---------------------------------------------------------------------------
namespace {

// Read the next non-blank, non-comment token-line into an istringstream.
// Returns false at end-of-file. Centralizes the "skip # comments" rule.
bool next_line(std::ifstream& in, std::istringstream& iss) {
    std::string line;
    while (std::getline(in, line)) {
        // Trim a trailing '\r' so Windows-CRLF files parse on any platform.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;            // blank line
        if (line[first] == '#') continue;                    // comment line
        iss.clear();
        iss.str(line);
        return true;
    }
    return false;
}

Spin3 read_point(std::ifstream& in) {
    std::istringstream iss;
    if (!next_line(in, iss)) throw std::runtime_error("EPR sample: unexpected EOF reading a point");
    Spin3 p{};
    if (!(iss >> p.x >> p.y >> p.z))
        throw std::runtime_error("EPR sample: malformed 'x y z' point line");
    return p;
}

}  // namespace

Ensemble load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open EPR sample: " + path);

    std::istringstream iss;
    if (!next_line(in, iss)) throw std::runtime_error("EPR sample: missing header line");
    int M = 0, rot = 0, nb = 0;
    if (!(iss >> M >> rot >> nb))
        throw std::runtime_error("EPR sample: header must be 'M ROTAMERS NBINS'");
    if (rot != ROTAMERS_PER_SITE)
        throw std::runtime_error("EPR sample: ROTAMERS in file != compiled ROTAMERS_PER_SITE");
    if (nb != NBINS)
        throw std::runtime_error("EPR sample: NBINS in file != compiled NBINS");
    if (M <= 0) throw std::runtime_error("EPR sample: M must be positive");

    Ensemble e;
    e.M = M;
    e.siteA.resize(static_cast<std::size_t>(M) * ROTAMERS_PER_SITE);
    e.siteB.resize(static_cast<std::size_t>(M) * ROTAMERS_PER_SITE);
    e.truth.resize(M, 0);

    for (int m = 0; m < M; ++m) {
        if (!next_line(in, iss)) throw std::runtime_error("EPR sample: missing truth flag for a frame");
        int flag = 0;
        iss >> flag;                              // tolerate missing -> 0
        e.truth[m] = (flag != 0) ? 1 : 0;
        for (int i = 0; i < ROTAMERS_PER_SITE; ++i)
            e.siteA[static_cast<std::size_t>(m) * ROTAMERS_PER_SITE + i] = read_point(in);
        for (int i = 0; i < ROTAMERS_PER_SITE; ++i)
            e.siteB[static_cast<std::size_t>(m) * ROTAMERS_PER_SITE + i] = read_point(in);
    }

    // The experimental target distribution, one value per bin. We re-normalize to
    // sum 1 so the file can store unnormalized counts and we still compare two
    // proper probability distributions.
    e.target.resize(NBINS, 0.0);
    double tsum = 0.0;
    for (int b = 0; b < NBINS; ++b) {
        if (!next_line(in, iss)) throw std::runtime_error("EPR sample: missing target P(r) bin");
        double v = 0.0;
        iss >> v;
        if (v < 0.0) v = 0.0;                     // a probability cannot be negative
        e.target[b] = v;
        tsum += v;
    }
    if (tsum <= 0.0) throw std::runtime_error("EPR sample: target distribution sums to zero");
    for (int b = 0; b < NBINS; ++b) e.target[b] /= tsum;   // exact normalization

    return e;
}

// ---------------------------------------------------------------------------
// softmax_weights  --  log-weights g[] -> normalized positive weights w[].
//   w[m] = exp(g[m] - gmax) / sum_k exp(g[k] - gmax). Subtracting the max is the
//   numerically stable "log-sum-exp" trick (prevents exp() overflow); it does
//   not change the result because the constant cancels in the ratio.
// ---------------------------------------------------------------------------
void softmax_weights(const std::vector<double>& g, std::vector<double>& w) {
    const std::size_t M = g.size();
    w.resize(M);
    double gmax = g[0];
    for (std::size_t m = 1; m < M; ++m) if (g[m] > gmax) gmax = g[m];
    double Z = 0.0;
    for (std::size_t m = 0; m < M; ++m) { w[m] = std::exp(g[m] - gmax); Z += w[m]; }
    const double invZ = 1.0 / Z;
    for (std::size_t m = 0; m < M; ++m) w[m] *= invZ;       // now sum_m w[m] == 1
}

// ---------------------------------------------------------------------------
// mixed_distribution  --  P(r) = sum_m w[m] * P_m(r).  A dense matrix-vector
//   product: hist is [M x NBINS] row-major, w is length M, mixed is length NBINS.
//   This is the model distribution we compare against the experimental target.
// ---------------------------------------------------------------------------
void mixed_distribution(const std::vector<double>& hist, int M,
                        const std::vector<double>& w, std::vector<double>& mixed) {
    mixed.assign(NBINS, 0.0);
    for (int m = 0; m < M; ++m) {
        const double wm = w[m];
        const double* h = hist.data() + static_cast<std::size_t>(m) * NBINS;
        for (int b = 0; b < NBINS; ++b) mixed[b] += wm * h[b];
    }
}

// ---------------------------------------------------------------------------
// objective  --  L(g) = chi2( mix(softmax(g)), target ) + THETA * KL( w || 1/M )
//   Computes the two scalar pieces and returns their sum. The reweighting drives
//   this down. Splitting it out lets main report fit vs. regularizer separately.
// ---------------------------------------------------------------------------
double objective(const std::vector<double>& hist, int M,
                 const std::vector<double>& target, const std::vector<double>& g,
                 double* out_chi2, double* out_kl) {
    std::vector<double> w, mixed;
    softmax_weights(g, w);
    mixed_distribution(hist, M, w, mixed);
    const double chi2 = chi2_to_target(mixed.data(), target.data());   // shared (deer.h)
    const double kl   = kl_to_prior(w.data(), M);                      // shared (deer.h)
    if (out_chi2) *out_chi2 = chi2;
    if (out_kl)   *out_kl   = kl;
    return chi2 + THETA * kl;
}

// ---------------------------------------------------------------------------
// deer_backcalc_cpu  --  per-frame histograms, the GPU kernel's CPU twin.
//   Loops over the M frames and calls the SHARED deer_member_histogram() for
//   each. O(M * R^2). hist is filled [M x NBINS] row-major.
// ---------------------------------------------------------------------------
void deer_backcalc_cpu(const Ensemble& e, std::vector<double>& hist) {
    hist.assign(static_cast<std::size_t>(e.M) * NBINS, 0.0);
    for (int m = 0; m < e.M; ++m) {
        const Spin3* A = e.siteA.data() + static_cast<std::size_t>(m) * ROTAMERS_PER_SITE;
        const Spin3* B = e.siteB.data() + static_cast<std::size_t>(m) * ROTAMERS_PER_SITE;
        double* h = hist.data() + static_cast<std::size_t>(m) * NBINS;
        deer_member_histogram(A, B, h);          // identical to the device call
    }
}

// ---------------------------------------------------------------------------
// reweight_cpu  --  max-entropy (BioEn/EROS) reweighting by gradient descent.
// ---------------------------------------------------------------------------
// We minimize L(g) = chi2(mix(softmax(g)), target) + THETA*KL(softmax(g)||1/M)
// over the unconstrained log-weights g (start: g = 0 -> uniform weights). The
// gradient has a clean closed form because softmax + sum-of-squares + KL are all
// differentiable. With  w = softmax(g),  m_b = sum_k w_k P_k(b),  and the chi^2
// residual r_b = 2 (m_b - target_b):
//
//   d chi2 / d w_k = sum_b r_b P_k(b)            == "per-frame fit gradient"  fk
//   d KL   / d w_k = ln(w_k / w0) + 1            == "per-frame entropy grad"  ek
// and the softmax Jacobian turns a weight-space gradient G_k into a log-space
// gradient:  dL/dg_j = w_j ( G_j - sum_k w_k G_k )   (mean-subtracted, scaled).
//
// This analytic gradient is exact and cheap (O(M*NBINS) per step), so descent is
// deterministic and converges smoothly. We deliberately hand-roll it (rather than
// autodiff) because seeing the gradient is the whole teaching point. The GPU path
// reuses this SAME function -- only `hist` is produced differently -- guaranteeing
// the reweighting trajectories match exactly.
// ---------------------------------------------------------------------------
void reweight_cpu(const std::vector<double>& hist, int M,
                  const std::vector<double>& target, std::vector<double>& w,
                  double* out_chi2_before, double* out_chi2_after) {
    std::vector<double> g(M, 0.0);          // log-weights; g = 0 => uniform weights
    std::vector<double> mixed(NBINS, 0.0);  // current model P(r)
    std::vector<double> G(M, 0.0);          // weight-space gradient per frame
    std::vector<double> grad(M, 0.0);       // log-space gradient (what we step on)

    const double w0 = 1.0 / static_cast<double>(M);   // uniform prior population

    // chi^2 at the uniform start (before any reweighting) -- reported as "before".
    softmax_weights(g, w);
    mixed_distribution(hist, M, w, mixed);
    if (out_chi2_before) *out_chi2_before = chi2_to_target(mixed.data(), target.data());

    for (int it = 0; it < REWEIGHT_ITERS; ++it) {
        // 1. Current weights and model distribution.
        softmax_weights(g, w);
        mixed_distribution(hist, M, w, mixed);

        // 2. Weight-space gradient G_k = dL/dw_k = (chi2 part) + THETA*(KL part).
        for (int k = 0; k < M; ++k) {
            const double* hk = hist.data() + static_cast<std::size_t>(k) * NBINS;
            double fit = 0.0;
            for (int b = 0; b < NBINS; ++b) {
                const double resid = 2.0 * (mixed[b] - target[b]);   // d chi2 / d m_b
                fit += resid * hk[b];                                // chain through m_b
            }
            // KL gradient: d/dw_k [ sum w ln(w/w0) ] = ln(w_k/w0) + 1.
            const double ent = std::log(w[k] / w0) + 1.0;
            G[k] = fit + THETA * ent;
        }

        // 3. Map to log-space via the softmax Jacobian:
        //    dL/dg_j = w_j * ( G_j - <G>_w ),  where <G>_w = sum_k w_k G_k.
        double Gbar = 0.0;
        for (int k = 0; k < M; ++k) Gbar += w[k] * G[k];
        for (int j = 0; j < M; ++j) grad[j] = w[j] * (G[j] - Gbar);

        // 4. Gradient-descent step on the log-weights (fixed LR -> deterministic).
        for (int j = 0; j < M; ++j) g[j] -= REWEIGHT_LR * grad[j];
    }

    // Final weights + chi^2 after reweighting.
    softmax_weights(g, w);
    mixed_distribution(hist, M, w, mixed);
    if (out_chi2_after) *out_chi2_after = chi2_to_target(mixed.data(), target.data());
}
