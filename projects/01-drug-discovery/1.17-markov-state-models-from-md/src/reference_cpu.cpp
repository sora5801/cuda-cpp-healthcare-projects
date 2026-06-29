// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared helpers, serial MSM reference
// ---------------------------------------------------------------------------
// Project 1.17 : Markov State Models from MD
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. Written to be
//   OBVIOUSLY correct -- plain serial loops, no parallelism -- so that when the
//   GPU agrees, we believe the GPU. Compiled by the host C++ compiler only (no
//   CUDA here); the per-frame math is shared with the kernels via msm.h.
//
//   The MSM pipeline lives here as a sequence of small, readable functions:
//     load_dataset -> init_centroids -> (assign/update)*iters
//        -> count_transitions -> build_transition_matrix
//        -> stationary_distribution -> slowest_timescale.
//   kernels.cu replaces ONLY the two hot loops (assign, count) with GPU twins
//   and reuses every other function below, which is why CPU == GPU exactly.
//
// READ THIS AFTER: msm.h and reference_cpu.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill, std::max
#include <cmath>       // std::log, std::sqrt, std::fabs
#include <fstream>     // std::ifstream
#include <limits>      // infinity
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_dataset: parse "N D K lag" then N*D floats. We validate aggressively so a
//   truncated or mis-shaped file produces a clear error rather than reading past
//   the data and silently clustering garbage.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);
    Dataset d;
    if (!(in >> d.N >> d.D >> d.K >> d.lag)
        || d.N <= 0 || d.D <= 0 || d.K <= 0 || d.K > d.N || d.lag <= 0 || d.lag >= d.N)
        throw std::runtime_error("bad header (expected 'N D K lag') in " + path);
    d.x.resize(static_cast<std::size_t>(d.N) * d.D);
    for (std::size_t i = 0; i < d.x.size(); ++i)
        if (!(in >> d.x[i])) throw std::runtime_error("dataset truncated in " + path);
    return d;
}

// ---------------------------------------------------------------------------
// init_centroids: deterministic FARTHEST-FIRST seeding (the greedy heart of
//   k-means++). Center 0 = frame 0; then repeatedly choose the frame farthest
//   from every center chosen so far. For an MSM this tends to place one initial
//   centroid in each well-separated conformational basin, so Lloyd's algorithm
//   converges to the metastable states instead of a poor local minimum. It is
//   fully deterministic (ties -> lowest index), so CPU and GPU start identically.
// ---------------------------------------------------------------------------
void init_centroids(const Dataset& d, std::vector<float>& centroids) {
    centroids.resize(static_cast<std::size_t>(d.K) * d.D);
    auto copy_center = [&](int k, int idx) {
        for (int j = 0; j < d.D; ++j)
            centroids[static_cast<std::size_t>(k) * d.D + j] =
                d.x[static_cast<std::size_t>(idx) * d.D + j];
    };
    copy_center(0, 0);                                   // first center = frame 0

    // min_d[i] = squared distance from frame i to its nearest chosen center.
    std::vector<double> min_d(d.N, std::numeric_limits<double>::infinity());
    for (int k = 1; k < d.K; ++k) {
        const float* last = &centroids[static_cast<std::size_t>(k - 1) * d.D];
        int best = 0; double best_d = -1.0;
        for (int i = 0; i < d.N; ++i) {
            // Fold in the distance to the center we JUST added, keeping a running
            // nearest-center distance, then track the global farthest frame.
            const double dd = km_sqdist(&d.x[static_cast<std::size_t>(i) * d.D], last, d.D);
            if (dd < min_d[i]) min_d[i] = dd;
            if (min_d[i] > best_d) { best_d = min_d[i]; best = i; }
        }
        copy_center(k, best);
    }
}

// ---------------------------------------------------------------------------
// update_centroids: turn fixed-point coordinate sums + integer counts into the
//   new centroids. Dividing the integer sums (built by atomicAdd on the GPU /
//   plain += on the CPU) by MSM_SCALE recovers the floating mean. Doing the
//   divide in ONE place, reused by both paths, guarantees identical centroids.
// ---------------------------------------------------------------------------
void update_centroids(const Dataset& d, const std::vector<unsigned long long>& sum,
                      const std::vector<unsigned int>& count, std::vector<float>& centroids) {
    for (int k = 0; k < d.K; ++k) {
        if (count[k] == 0) continue;                     // empty microstate: keep old center
        for (int j = 0; j < d.D; ++j) {
            const double mean = (static_cast<double>(sum[static_cast<std::size_t>(k) * d.D + j])
                                 / MSM_SCALE) / count[k];
            centroids[static_cast<std::size_t>(k) * d.D + j] = static_cast<float>(mean);
        }
    }
}

// ---------------------------------------------------------------------------
// compute_inertia: the k-means objective (sum of squared assignment distances).
//   A single scalar that summarizes how tight the microstate clustering is;
//   identical formula on both paths.
// ---------------------------------------------------------------------------
double compute_inertia(const Dataset& d, const std::vector<float>& centroids,
                       const std::vector<int>& labels) {
    double inertia = 0.0;
    for (int i = 0; i < d.N; ++i)
        inertia += km_sqdist(&d.x[static_cast<std::size_t>(i) * d.D],
                             &centroids[static_cast<std::size_t>(labels[i]) * d.D], d.D);
    return inertia;
}

// ---------------------------------------------------------------------------
// count_transitions_cpu: the heart of the MSM. Slide a window of width `lag`
//   over the time-ordered label sequence and tally each microstate->microstate
//   hop into the K x K count matrix C (row = "from", col = "to"). This is the
//   maximum-likelihood sufficient statistic for a Markov chain. INTEGER adds
//   -> the GPU twin (one thread per t) gives a bit-identical matrix.
// ---------------------------------------------------------------------------
void count_transitions_cpu(const Dataset& d, const std::vector<int>& labels,
                           std::vector<unsigned int>& counts) {
    counts.assign(static_cast<std::size_t>(d.K) * d.K, 0u);
    for (int t = 0; t + d.lag < d.N; ++t) {
        const int from = labels[t];          // microstate at time t
        const int to   = labels[t + d.lag];  // microstate at time t+tau
        counts[static_cast<std::size_t>(from) * d.K + to] += 1u;
    }
}

// ---------------------------------------------------------------------------
// build_transition_matrix: row-normalize the count matrix C into the transition
//   PROBABILITY matrix T. T[i][j] = C[i][j] / (sum over j of C[i][j]) is the
//   maximum-likelihood estimate of P(state j at t+tau | state i at t). A row
//   with no outgoing transitions becomes a self-absorbing state (T[i][i]=1) so
//   T remains a valid (row-stochastic) matrix. Identical on CPU/GPU because both
//   feed it the SAME integer count matrix.
// ---------------------------------------------------------------------------
void build_transition_matrix(int K, const std::vector<unsigned int>& counts,
                             std::vector<double>& T) {
    T.assign(static_cast<std::size_t>(K) * K, 0.0);
    for (int i = 0; i < K; ++i) {
        double row = 0.0;
        for (int j = 0; j < K; ++j) row += counts[static_cast<std::size_t>(i) * K + j];
        if (row == 0.0) {                    // never-left state -> self loop
            T[static_cast<std::size_t>(i) * K + i] = 1.0;
            continue;
        }
        for (int j = 0; j < K; ++j)
            T[static_cast<std::size_t>(i) * K + j] =
                counts[static_cast<std::size_t>(i) * K + j] / row;
    }
}

// ---------------------------------------------------------------------------
// stationary_distribution: pi such that pi T = pi (the equilibrium populations).
//   pi is the LEFT eigenvector of T for eigenvalue 1, i.e. the right eigenvector
//   of T^T. We find it by POWER ITERATION on T^T: repeatedly v <- T^T v and
//   renormalize; v converges to the dominant eigenvector (eigenvalue 1 for a
//   stochastic matrix). K is tiny (a handful of microstates), so a few hundred
//   iterations are instant and far simpler than calling a full eigensolver.
// ---------------------------------------------------------------------------
void stationary_distribution(int K, const std::vector<double>& T, std::vector<double>& pi) {
    pi.assign(K, 1.0 / K);                    // start from the uniform distribution
    std::vector<double> next(K, 0.0);
    for (int it = 0; it < 2000; ++it) {
        std::fill(next.begin(), next.end(), 0.0);
        // next[j] = sum_i pi[i] * T[i][j]   (this is pi advanced one step: pi T)
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                next[j] += pi[static_cast<std::size_t>(i)] * T[static_cast<std::size_t>(i) * K + j];
        double s = 0.0;
        for (int j = 0; j < K; ++j) s += next[j];
        if (s > 0.0) for (int j = 0; j < K; ++j) next[j] /= s;   // renormalize to a distribution
        pi.swap(next);
    }
}

// ---------------------------------------------------------------------------
// slowest_timescale: the second eigenvalue of T sets the slowest relaxation.
//   T's largest eigenvalue is 1 (stationary). The SECOND-largest magnitude,
//   lambda_2, controls how fast the slowest mode decays; the corresponding
//   IMPLIED TIMESCALE is t_2 = -tau / ln(lambda_2) (in frames). Physically this
//   is the molecule's slowest process (folding, a domain motion, a binding/
//   unbinding event).
//
//   We get lambda_2 by DEFLATED power iteration: power-iterate to the dominant
//   eigenpair (lambda_1=1 with eigenvector the all-ones direction for a
//   stochastic matrix's right eigenvector), then project that component out of
//   the iterate each step so the iteration converges to the SECOND eigenvector.
//   This is a teaching-simple substitute for a general eigensolver and is exact
//   enough for the small, well-conditioned T we build here.
// ---------------------------------------------------------------------------
double slowest_timescale(int K, int lag, const std::vector<double>& T, double* lambda2_out) {
    if (K < 2) { if (lambda2_out) *lambda2_out = 0.0; return 0.0; }

    // The right eigenvector of a row-stochastic matrix for eigenvalue 1 is the
    // constant vector (T * ones = ones). Deflating against this constant
    // direction leaves the subdominant modes.
    const double inv_sqrt_K = 1.0 / std::sqrt(static_cast<double>(K));
    std::vector<double> u1(K, inv_sqrt_K);   // unit-norm dominant right eigenvector

    // Start from a vector orthogonal-ish to u1; deflation each step enforces it.
    std::vector<double> v(K, 0.0);
    v[0] = 1.0; if (K > 1) v[1] = -1.0;       // a simple non-constant seed

    auto deflate = [&](std::vector<double>& w) {
        double dot = 0.0;                     // remove the u1 component: w <- w - (w.u1) u1
        for (int i = 0; i < K; ++i) dot += w[i] * u1[i];
        for (int i = 0; i < K; ++i) w[i] -= dot * u1[i];
    };
    auto normalize = [&](std::vector<double>& w) -> double {
        double n = 0.0; for (double x : w) n += x * x; n = std::sqrt(n);
        if (n > 0.0) for (double& x : w) x /= n;
        return n;
    };

    deflate(v); normalize(v);
    std::vector<double> w(K, 0.0);
    double lambda2 = 0.0;
    for (int it = 0; it < 3000; ++it) {
        // w = T v   (right-multiply: w[i] = sum_j T[i][j] v[j])
        for (int i = 0; i < K; ++i) {
            double acc = 0.0;
            for (int j = 0; j < K; ++j) acc += T[static_cast<std::size_t>(i) * K + j] * v[j];
            w[i] = acc;
        }
        deflate(w);                            // keep iterate free of the lambda_1 mode
        const double n = normalize(w);         // Rayleigh-ish: the growth factor ~ |lambda_2|
        lambda2 = n;
        v.swap(w);
    }
    if (lambda2_out) *lambda2_out = lambda2;
    // t_2 = -tau / ln(lambda_2); guard the degenerate lambda_2 -> {0,>=1} cases.
    if (lambda2 <= 0.0 || lambda2 >= 1.0) return 0.0;
    return -static_cast<double>(lag) / std::log(lambda2);
}

// ---------------------------------------------------------------------------
// msm_cpu: the full reference pipeline. Lloyd's k-means for a FIXED number of
//   iterations (deterministic; no convergence test that could differ from the
//   GPU), then count transitions, build T, and extract pi + the slowest
//   timescale. This is the trusted baseline main.cu verifies the GPU against.
// ---------------------------------------------------------------------------
MsmResult msm_cpu(const Dataset& d, int iters) {
    MsmResult r;
    init_centroids(d, r.centroids);
    r.labels.assign(d.N, 0);
    std::vector<unsigned long long> sum(static_cast<std::size_t>(d.K) * d.D);
    r.sizes.assign(d.K, 0);

    // --- k-means (ASSIGN + UPDATE), the part the GPU parallelizes -----------
    for (int it = 0; it < iters; ++it) {
        // ASSIGN: nearest centroid (microstate) for every frame.
        for (int i = 0; i < d.N; ++i)
            r.labels[i] = km_nearest(&d.x[static_cast<std::size_t>(i) * d.D],
                                     r.centroids.data(), d.K, d.D);
        // UPDATE: fixed-point accumulate, then divide (mirrors the GPU exactly).
        std::fill(sum.begin(), sum.end(), 0ull);
        std::fill(r.sizes.begin(), r.sizes.end(), 0u);
        for (int i = 0; i < d.N; ++i) {
            const int k = r.labels[i];
            for (int j = 0; j < d.D; ++j)
                sum[static_cast<std::size_t>(k) * d.D + j] +=
                    km_to_fixed(d.x[static_cast<std::size_t>(i) * d.D + j]);
            r.sizes[k] += 1u;
        }
        update_centroids(d, sum, r.sizes, r.centroids);
    }

    // --- transition matrix + spectral analysis (host-only, reused by GPU) ---
    count_transitions_cpu(d, r.labels, r.counts);
    build_transition_matrix(d.K, r.counts, r.T);
    stationary_distribution(d.K, r.T, r.pi);
    r.timescale = slowest_timescale(d.K, d.lag, r.T, &r.lambda2);
    return r;
}
