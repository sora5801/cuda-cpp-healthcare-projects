// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial ADMET baseline + data loader
// ---------------------------------------------------------------------------
// Project 1.16 : ADMET / Toxicity Prediction  (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   (1) load_admet()        : parse the tiny text dataset (data/README.md format).
//   (2) admet_predict_cpu() : the obviously-correct serial NxM prediction the GPU
//                             kernel is verified against -- no parallelism, no
//                             cleverness, just nested loops calling the SHARED
//                             admet_predict() from admet_core.h.
//   (3) admet_reduce()      : the deterministic integer reduction (per-endpoint
//                             flag counts, per-molecule totals, worst molecule).
//
//   Compiled by the host C++ compiler only (no CUDA). Because the per-element
//   math lives in admet_core.h behind the HD macro, THIS file and kernels.cu run
//   byte-for-byte identical arithmetic -> exact verification (PATTERNS.md sec.2).
//
// READ THIS AFTER: reference_cpu.h, admet_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream (header-line tokenizing)
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_admet: read the screening problem from text.
//
//   FILE FORMAT (documented fully in data/README.md):
//     line 1            : "<n> <D> <M>"  -- molecule count, descriptor len, #endpoints
//     next M lines      : "<endpoint_name> <b_t> <w_{t,0}> ... <w_{t,D-1}>"
//     next n lines      : "<mol_name> <x_{i,0}> ... <x_{i,D-1}>"
//   We validate D and M against the COMPILED ADMET_D / ADMET_M so a file built
//   for a different descriptor size fails loudly instead of reading garbage.
//
//   Why text (not a binary blob)? The sample is tiny and a learner can open it
//   and see exactly what feeds the kernel -- transparency over speed, per the
//   repo's didactic mission.
// ---------------------------------------------------------------------------
AdmetData load_admet(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ADMET data file: " + path);

    int n = 0, d = 0, m = 0;
    if (!(in >> n >> d >> m))
        throw std::runtime_error("bad header (expected '<n> <D> <M>') in " + path);
    if (d != ADMET_D)
        throw std::runtime_error("descriptor length mismatch: file D=" + std::to_string(d) +
                                 " but this build expects ADMET_D=" + std::to_string(ADMET_D));
    if (m != ADMET_M)
        throw std::runtime_error("endpoint count mismatch: file M=" + std::to_string(m) +
                                 " but this build expects ADMET_M=" + std::to_string(ADMET_M));
    if (n <= 0)
        throw std::runtime_error("non-positive molecule count in " + path);

    AdmetData data;
    data.n = n;
    data.desc.resize(static_cast<std::size_t>(n) * ADMET_D);
    data.weights.resize(static_cast<std::size_t>(ADMET_M) * ADMET_D);
    data.bias.resize(ADMET_M);
    data.mol_names.resize(n);
    data.endpoint_names.resize(ADMET_M);

    // --- M endpoint model rows: "<name> <bias> <w_0 ... w_{D-1}>" ----------
    for (int t = 0; t < ADMET_M; ++t) {
        if (!(in >> data.endpoint_names[static_cast<std::size_t>(t)]))
            throw std::runtime_error("unexpected EOF reading endpoint name " + std::to_string(t));
        if (!(in >> data.bias[static_cast<std::size_t>(t)]))
            throw std::runtime_error("unexpected EOF reading bias for endpoint " + std::to_string(t));
        for (int k = 0; k < ADMET_D; ++k)
            if (!(in >> data.weights[static_cast<std::size_t>(t) * ADMET_D + k]))
                throw std::runtime_error("unexpected EOF reading weight for endpoint " + std::to_string(t));
    }

    // --- n molecule rows: "<name> <x_0 ... x_{D-1}>" ----------------------
    for (int i = 0; i < n; ++i) {
        if (!(in >> data.mol_names[static_cast<std::size_t>(i)]))
            throw std::runtime_error("unexpected EOF reading molecule name " + std::to_string(i));
        for (int k = 0; k < ADMET_D; ++k)
            if (!(in >> data.desc[static_cast<std::size_t>(i) * ADMET_D + k]))
                throw std::runtime_error("unexpected EOF reading descriptor for molecule " + std::to_string(i));
    }
    return data;
}

// ---------------------------------------------------------------------------
// admet_predict_cpu: fill probs[i*M + t] = p_{i,t} for every (molecule, endpoint).
//
//   Two nested loops -> O(n * M * D) multiply-adds. Each cell is INDEPENDENT
//   (it reads only molecule i's descriptor and endpoint t's model), which is
//   precisely why the GPU gives each cell its own thread in kernels.cu. We call
//   the SHARED admet_predict() so the arithmetic matches the GPU exactly.
// ---------------------------------------------------------------------------
void admet_predict_cpu(const AdmetData& data, std::vector<double>& probs) {
    probs.assign(static_cast<std::size_t>(data.n) * ADMET_M, 0.0);
    for (int i = 0; i < data.n; ++i) {
        const double* x = &data.desc[static_cast<std::size_t>(i) * ADMET_D];  // molecule i
        for (int t = 0; t < ADMET_M; ++t) {
            const double* w = &data.weights[static_cast<std::size_t>(t) * ADMET_D]; // endpoint t
            const double  b = data.bias[static_cast<std::size_t>(t)];
            probs[static_cast<std::size_t>(i) * ADMET_M + t] = admet_predict(w, x, b, ADMET_D);
        }
    }
}

// ---------------------------------------------------------------------------
// admet_reduce: collapse the [n*M] probability matrix into the deterministic
// triage result. ALL accumulation is INTEGER (flag counts) so it is exactly
// reproducible and order-independent -- the same property that lets the GPU use
// atomicAdd on integers and still match this bit-for-bit (PATTERNS.md sec.3).
//
//   flagged_per_endpoint[t] : how many molecules have p_{i,t} >= threshold
//   total_flags[i]          : how many endpoints molecule i trips
//   worst_mol               : argmax over molecules, ranked by (total_flags,
//                             then summed probability, then -index) so ties are
//                             broken deterministically toward the lower index.
// ---------------------------------------------------------------------------
AdmetResult admet_reduce(const AdmetData& data, const std::vector<double>& probs) {
    AdmetResult r;
    r.flagged_per_endpoint.assign(ADMET_M, 0);
    r.total_flags.assign(static_cast<std::size_t>(data.n), 0);

    int    best_i      = -1;     // index of the worst (most toxic) molecule so far
    int    best_flags  = -1;     // its total flag count
    double best_score  = -1.0;   // its summed probability (tie-breaker)

    for (int i = 0; i < data.n; ++i) {
        int    flags = 0;        // endpoints molecule i trips
        double sump  = 0.0;      // sum of p_{i,t} over endpoints (tie-break only)
        for (int t = 0; t < ADMET_M; ++t) {
            const double p = probs[static_cast<std::size_t>(i) * ADMET_M + t];
            const int    f = admet_flagged(p, ADMET_THRESHOLD);   // 0 or 1
            r.flagged_per_endpoint[static_cast<std::size_t>(t)] += f;  // integer add
            flags += f;
            sump  += p;
        }
        r.total_flags[static_cast<std::size_t>(i)] = flags;

        // Deterministic argmax: more flags wins; equal flags -> larger summed
        // probability wins; a perfect tie keeps the LOWER index (we only replace
        // on a strict improvement, and we scan i ascending).
        if (flags > best_flags ||
            (flags == best_flags && sump > best_score)) {
            best_flags = flags;
            best_score = sump;
            best_i     = i;
        }
    }
    r.worst_mol       = best_i;
    r.worst_mol_score = best_score;
    return r;
}
