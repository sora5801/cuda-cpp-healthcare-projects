// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared metric, serial ICP reference
// ---------------------------------------------------------------------------
// Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
//
// Compiled by the host C++ compiler only (no CUDA constructs here). The
// per-point math -- nearest neighbour, the fixed-point covariance reduction,
// the 3x3 SVD, and solve_rigid -- all live in icp.h and are shared verbatim
// with the GPU kernels, so the serial ICP below and the GPU ICP produce the
// SAME transform each iteration. This file is the readable, obviously-correct
// baseline the GPU is checked against.
//
// READ THIS AFTER: reference_cpu.h, icp.h.  Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sqrt
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_clouds: parse the tiny committed sample. Format (see data/README.md):
//   np nq
//   [optional] a line whose first token is "GT", then 12 numbers: R(9) t(3)
//   np rows "x y z"   -> moving cloud P
//   nq rows "x y z"   -> fixed  cloud Q
// Throws std::runtime_error on any structural problem so the demo fails loudly.
// ---------------------------------------------------------------------------
Clouds load_clouds(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open cloud file: " + path);

    Clouds c;
    int np = 0, nq = 0;
    if (!(in >> np >> nq) || np <= 0 || nq <= 0)
        throw std::runtime_error("bad header (expected 'np nq') in " + path);

    c.gt = rigid_identity();

    // Peek the next token: if it is "GT" we read the ground-truth block, else we
    // put it back (it is the first coordinate of P) by remembering its value.
    std::string tok;
    if (!(in >> tok)) throw std::runtime_error("file truncated after header: " + path);

    // Helper to read a single float from a token we already hold, or from stream.
    auto read_first_float = [&](const std::string& first, float& dst) {
        dst = std::stof(first);
    };

    std::size_t read_start_index = 0;   // how many P-floats we have already read
    float pending = 0.0f;               // the first P x-coordinate, if buffered
    bool  have_pending = false;

    if (tok == "GT" || tok == "gt") {
        c.has_gt = true;
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 3; ++col)
                if (!(in >> c.gt.R[r][col])) throw std::runtime_error("bad GT rotation in " + path);
        for (int r = 0; r < 3; ++r)
            if (!(in >> c.gt.t[r])) throw std::runtime_error("bad GT translation in " + path);
    } else {
        // The token we grabbed is actually P[0].x -- buffer it for the loop.
        read_first_float(tok, pending);
        have_pending = true;
    }

    // Read the moving cloud P (np points, 3 floats each).
    c.P.resize(static_cast<std::size_t>(np));
    for (int i = 0; i < np; ++i) {
        float x, y, z;
        if (have_pending && i == 0 && read_start_index == 0) {
            x = pending;                       // reuse the buffered first x
            have_pending = false;
            if (!(in >> y >> z)) throw std::runtime_error("P truncated in " + path);
        } else {
            if (!(in >> x >> y >> z)) throw std::runtime_error("P truncated in " + path);
        }
        c.P[static_cast<std::size_t>(i)] = Vec3{ x, y, z };
    }

    // Read the fixed cloud Q (nq points).
    c.Q.resize(static_cast<std::size_t>(nq));
    for (int i = 0; i < nq; ++i) {
        float x, y, z;
        if (!(in >> x >> y >> z)) throw std::runtime_error("Q truncated in " + path);
        c.Q[static_cast<std::size_t>(i)] = Vec3{ x, y, z };
    }
    return c;
}

// ---------------------------------------------------------------------------
// rms_error: the alignment-quality metric (mm). For each moving point, apply
// the transform g, find its nearest fixed point, and accumulate the squared
// distance; return sqrt(mean). This is exactly the quantity ICP minimizes, so
// it must fall monotonically across iterations (a great thing to watch).
// ---------------------------------------------------------------------------
double rms_error(const std::vector<Vec3>& P, const std::vector<Vec3>& Q, const Rigid& g) {
    if (P.empty()) return 0.0;
    const int nq = static_cast<int>(Q.size());
    double acc = 0.0;
    for (const Vec3& p : P) {
        const Vec3 tp = rigid_apply(g, p);               // move p by current guess
        const int j = nearest_index(tp, Q.data(), nq);   // its closest fixed point
        acc += sqdist(tp, Q[static_cast<std::size_t>(j)]);
    }
    return std::sqrt(acc / static_cast<double>(P.size()));
}

// ---------------------------------------------------------------------------
// icp_cpu: the serial ICP reference. `iters` FIXED iterations (no early-out) so
// the result is perfectly deterministic and comparable to the GPU. Each pass:
//   1. Move every P point by the current running transform g.
//   2. CORRESPOND: nearest fixed point for each moved point.
//   3. Accumulate the fixed-point covariance/centroid sums over all pairs.
//   4. ALIGN: solve the incremental rigid transform, compose it onto g.
//   5. Record the RMS error for the convergence curve.
// Returns the final g (maps original P onto Q).
// ---------------------------------------------------------------------------
Rigid icp_cpu(const Clouds& c, int iters, std::vector<double>& history) {
    const int np = static_cast<int>(c.P.size());
    const int nq = static_cast<int>(c.Q.size());
    // Start from the CENTROID PRE-ALIGNMENT, not raw identity: this coarse guess
    // keeps ICP inside its convergence basin (see icp.h centroid_prealign). The
    // GPU path uses the SAME starting transform, so both converge identically.
    Rigid g = centroid_prealign(c.P.data(), np, c.Q.data(), nq);
    history.clear();
    history.reserve(static_cast<std::size_t>(iters));

    for (int it = 0; it < iters; ++it) {
        AccumFixed acc;
        accum_zero(acc);

        // Steps 1-3: for each moving point, transform, find its match, and add
        // the pair into the fixed-point accumulators (this is the reduction the
        // GPU parallelises with atomics -- but here it is a plain serial loop).
        for (int i = 0; i < np; ++i) {
            const Vec3 tp = rigid_apply(g, c.P[static_cast<std::size_t>(i)]);
            const int j = nearest_index(tp, c.Q.data(), nq);
            accum_pair(acc, tp, c.Q[static_cast<std::size_t>(j)]);
        }

        // Step 4: solve the incremental transform from the accumulators and
        // compose it onto the running estimate. Because solve_rigid works on the
        // ALREADY-transformed points tp, the solved transform is the increment
        // to apply AFTER g, i.e. g_new = increment ∘ g.
        const Rigid inc = solve_rigid(acc);
        g = rigid_compose(inc, g);

        // Step 5: record the alignment quality after this iteration.
        history.push_back(rms_error(c.P, c.Q, g));
    }
    return g;
}
