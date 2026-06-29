// ===========================================================================
// src/reference_cpu.cpp  --  Loader, bath builder, serial walker driver, TI/BAR
// ---------------------------------------------------------------------------
// Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
//
// Compiled by the host C++ compiler ONLY. The per-walker Monte Carlo physics is
// in alchemy.h (shared with the GPU); here we (a) read the config, (b) build the
// synthetic solvent bath, (c) drive every walker serially as the trusted
// baseline, (d) reduce walkers -> per-window stats, and (e) turn those stats into
// a free energy by Thermodynamic Integration and by BAR.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // exp, log, fabs, sqrt
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_config: parse the single-line sample format. We validate aggressively so
// a malformed file fails LOUDLY (a silent default would mislead the learner).
// ---------------------------------------------------------------------------
AlchConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);
    AlchConfig c;
    int n_solvent = 0;
    if (!(in >> n_solvent >> c.sys.box >> c.sys.temperature >> c.sys.epsilon
             >> c.sys.sigma >> c.sys.q_solute >> c.sys.alpha_sc >> c.sys.max_step
             >> c.n_windows >> c.n_walkers >> c.n_equil >> c.n_prod
             >> c.seed >> c.bath_seed))
        throw std::runtime_error(
            "bad config (expected 'n_solvent box T epsilon sigma q_solute "
            "alpha_sc max_step n_windows n_walkers n_equil n_prod seed bath_seed') in " + path);
    c.sys.n_solvent = n_solvent;
    if (n_solvent <= 0 || c.n_windows < 2 || c.n_walkers <= 0 ||
        c.n_prod <= 0 || c.sys.temperature <= 0.0 || c.sys.box <= 0.0)
        throw std::runtime_error("invalid config values in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// build_bath: place n_solvent sites on a jittered spherical shell of radius
// ~0.6*box around the origin. We use the SAME counter-based RNG as the walkers
// (alchemy.h) seeded from `seed`, so the geometry is reproducible and identical
// to whatever scripts/make_synthetic.py documents. The shell keeps solvent away
// from r=0 (so plain LJ would already be finite) yet close enough that the
// solute feels it -- a clean, interpretable model bath.
// ---------------------------------------------------------------------------
BathStorage build_bath(const SystemParams& sys, int n_solvent, unsigned seed) {
    BathStorage b;
    b.x.resize(n_solvent);
    b.y.resize(n_solvent);
    b.z.resize(n_solvent);
    Rng rng = rng_init(0xB47Au + seed, 12345u);     // a fixed, documented stream
    const double shell = 0.6 * sys.box;             // nominal shell radius
    for (int i = 0; i < n_solvent; ++i) {
        // Draw a random direction by sampling a point in the cube and normalizing,
        // then jitter the radius by +/-15%. A tiny epsilon guards the (vanishingly
        // unlikely) exact-zero direction vector.
        double ux = 2.0 * rng_uniform(rng) - 1.0;
        double uy = 2.0 * rng_uniform(rng) - 1.0;
        double uz = 2.0 * rng_uniform(rng) - 1.0;
        const double norm = std::sqrt(ux * ux + uy * uy + uz * uz + 1e-12);
        const double r = shell * (0.85 + 0.30 * rng_uniform(rng));   // radius in [0.85,1.15]*shell
        b.x[i] = r * ux / norm;
        b.y[i] = r * uy / norm;
        b.z[i] = r * uz / norm;
    }
    return b;
}

// ---------------------------------------------------------------------------
// run_cpu: the serial reference. One independent Metropolis chain per
// (window, walker), each with a globally-unique id so its RNG stream is its own.
// The walker id MUST be computed identically here and in the kernel (window *
// n_walkers + walker) or the two would sample different chains and never agree.
// ---------------------------------------------------------------------------
void run_cpu(const AlchConfig& c, const BathStorage& bath,
             std::vector<WalkerResult>& walkers) {
    const SolventBath view = bath.view();
    const int W = total_walkers(c);
    walkers.assign(W, WalkerResult{});
    for (int w = 0; w < c.n_windows; ++w) {
        const double lam      = window_lambda(c, w);
        // Neighbour couplings for the BAR energy differences; clamp at the ends
        // so the end windows compare against themselves (delta = 0 there).
        const double lam_prev = window_lambda(c, (w > 0)               ? w - 1 : w);
        const double lam_next = window_lambda(c, (w < c.n_windows - 1) ? w + 1 : w);
        for (int k = 0; k < c.n_walkers; ++k) {
            const int gid = w * c.n_walkers + k;             // global walker id
            walkers[gid] = run_walker(view, c.sys, lam, lam_prev, lam_next,
                                      c.seed, uint64_t(gid),
                                      c.n_equil, c.n_prod);
        }
    }
}

// ---------------------------------------------------------------------------
// reduce_windows: average each window's walkers into one WindowStats. A plain
// ordered loop (no atomics) => deterministic and identical on any input. We sum
// the per-walker SUMS and sample COUNTS, then divide once -- numerically better
// than averaging per-walker means with unequal counts.
// ---------------------------------------------------------------------------
std::vector<WindowStats> reduce_windows(const AlchConfig& c,
                                        const std::vector<WalkerResult>& walkers) {
    std::vector<WindowStats> out(c.n_windows);
    for (int w = 0; w < c.n_windows; ++w) {
        double s_dudl = 0.0, s_fwd = 0.0, s_bwd = 0.0;
        long   n = 0, acc = 0;
        const long moves = long(c.n_walkers) * long(c.n_prod);   // total production moves
        for (int k = 0; k < c.n_walkers; ++k) {
            const WalkerResult& r = walkers[w * c.n_walkers + k];
            s_dudl += r.sum_dudl;
            s_fwd  += r.sum_du_fwd;
            s_bwd  += r.sum_du_bwd;
            n      += r.n_samples;
            acc    += r.n_accept;
        }
        WindowStats st;
        st.lambda      = window_lambda(c, w);
        st.mean_dudl   = (n > 0) ? s_dudl / double(n) : 0.0;
        st.mean_du_fwd = (n > 0) ? s_fwd  / double(n) : 0.0;
        st.mean_du_bwd = (n > 0) ? s_bwd  / double(n) : 0.0;
        st.accept_frac = (moves > 0) ? double(acc) / double(moves) : 0.0;
        st.n_samples   = n;
        out[w] = st;
    }
    return out;
}

// ---------------------------------------------------------------------------
// estimate_ti: Thermodynamic Integration by the trapezoidal rule.
//   delta-G(switch on) = integral_0^1 <dU/dlambda> d-lambda
//                      ~ sum over adjacent windows of (lam_{i+1}-lam_i)/2 *
//                        (<dU/dl>_i + <dU/dl>_{i+1}).
//   delta-G_solv (transferring the solute INTO solvent) is the NEGATIVE of the
//   switch-ON work in our convention (lambda=1 is fully solvated): adding the
//   solute-solvent coupling lowers the system free energy by |delta-G|, so the
//   solvation free energy is -integral. See THEORY section 2 for the sign.
// ---------------------------------------------------------------------------
double estimate_ti(const AlchConfig&, const std::vector<WindowStats>& stats) {
    double integral = 0.0;
    for (std::size_t i = 0; i + 1 < stats.size(); ++i) {
        const double dl = stats[i + 1].lambda - stats[i].lambda;
        integral += 0.5 * dl * (stats[i].mean_dudl + stats[i + 1].mean_dudl);
    }
    return -integral;   // delta-G_solv
}

// ---------------------------------------------------------------------------
// bar_pair: BAR free-energy difference beta*delta-f between two adjacent windows
// i and i+1, from their mean forward/backward energy differences.
//
//   BAR (Bennett 1976) is the minimum-variance free-energy estimator: it uses
//   energy-difference samples from BOTH the forward (i sampled, evaluated at i+1)
//   and backward (i+1 sampled, evaluated at i) directions. The full estimator
//   solves an implicit equation over the sample arrays. In the narrow,
//   closely-spaced-window regime where BAR is actually used, the two energy-
//   difference distributions are near-mirror Gaussians and the estimate reduces
//   to the deterministic closed form below -- which we use for the teaching
//   build (no iteration, so it is reproducible and CPU/GPU-agnostic). THEORY
//   section 6 gives the full self-consistent form as Exercise 4.
// ---------------------------------------------------------------------------
static double bar_pair(double mean_dU_fwd, double mean_dU_bwd, double beta) {
    const double a = beta * mean_dU_fwd;     // <beta*(U_{i+1}-U_i)> sampled in i
    const double b = beta * mean_dU_bwd;     // <beta*(U_i-U_{i+1})> sampled in i+1
    return 0.5 * (a - b);                    // beta * delta-f for the i->i+1 step
}

// estimate_bar: sum the per-adjacent-pair BAR free-energy increments and negate
// for the solvation convention (as in TI). Returns delta-G_solv in energy units.
double estimate_bar(const AlchConfig& c, const std::vector<WindowStats>& stats) {
    const double beta = 1.0 / c.sys.temperature;
    double df_total = 0.0;   // sum of beta*delta-f over windows (switch-on)
    for (std::size_t i = 0; i + 1 < stats.size(); ++i) {
        // window i's forward delta is to window i+1; window i+1's backward delta
        // is to window i -- exactly the pair BAR needs.
        df_total += bar_pair(stats[i].mean_du_fwd, stats[i + 1].mean_du_bwd, beta);
    }
    return -df_total / beta;   // back to energy units, solvation sign
}
