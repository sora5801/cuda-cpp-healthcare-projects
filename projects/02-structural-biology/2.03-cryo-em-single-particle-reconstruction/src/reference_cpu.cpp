// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust + data loader
// ---------------------------------------------------------------------------
// Project 2.3 : Cryo-EM Single-Particle Reconstruction  (reduced-scope, 2D)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- single readable loops, no parallelism, no
//   cleverness -- so that when the GPU and CPU agree, we believe the GPU. The
//   actual per-element physics (projection, correlation, back-projection) lives
//   in reference_cpu.h as `__host__ __device__` functions shared with the GPU,
//   so the two paths run BYTE-IDENTICAL arithmetic (docs/PATTERNS.md §2).
//
//   Two computations mirror the two GPU kernels:
//     match_cpu       -> the E-step: assign each particle its best ref angle.
//     reconstruct_cpu -> the M-step: back-project the assigned profiles to 2D.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_dataset: parse the tiny text dataset (format documented in
//   data/README.md). Layout (all whitespace-separated):
//     line: "n_particles  IMG_SIZE  N_ANGLES  PROJ_LEN"   (a header + self-check)
//     then IMG_SIZE*IMG_SIZE floats : the ground-truth density (row-major).
//     then N_ANGLES*PROJ_LEN  floats: the reference projection bank.
//     then, per particle: 1 int (true angle index) + PROJ_LEN floats (profile).
//   We re-derive the reference bank from the density at load time would also be
//   possible, but reading it lets make_synthetic.py own the geometry and keeps
//   the loader dumb. Throws std::runtime_error on any mismatch so a malformed
//   file fails loudly rather than silently producing garbage.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    int n = 0, img = 0, ang = 0, plen = 0;
    if (!(in >> n >> img >> ang >> plen))
        throw std::runtime_error("bad header (expected 'n IMG_SIZE N_ANGLES PROJ_LEN') in " + path);
    // The geometry is baked into this build at compile time, so a file made for a
    // different size cannot be matched correctly -> reject it with a clear message.
    if (img != IMG_SIZE || ang != N_ANGLES || plen != PROJ_LEN)
        throw std::runtime_error("geometry mismatch: file has (" +
            std::to_string(img) + "," + std::to_string(ang) + "," + std::to_string(plen) +
            ") but this build expects (" + std::to_string(IMG_SIZE) + "," +
            std::to_string(N_ANGLES) + "," + std::to_string(PROJ_LEN) +
            ") -- regenerate the sample or rebuild");
    if (n <= 0) throw std::runtime_error("non-positive particle count in " + path);

    Dataset ds;
    ds.n_particles = n;
    ds.true_img.resize(static_cast<std::size_t>(IMG_SIZE) * IMG_SIZE);
    ds.refs.resize(static_cast<std::size_t>(N_ANGLES) * PROJ_LEN);
    ds.particles.resize(static_cast<std::size_t>(n) * PROJ_LEN);
    ds.true_angle.resize(static_cast<std::size_t>(n));

    // Helper: read one float or throw on early EOF.
    auto rf = [&](float& v) {
        if (!(in >> v)) throw std::runtime_error("unexpected end of data in " + path);
    };
    for (auto& v : ds.true_img) rf(v);
    for (auto& v : ds.refs)     rf(v);
    for (int i = 0; i < n; ++i) {
        if (!(in >> ds.true_angle[i]))
            throw std::runtime_error("unexpected end of data (true_angle) in " + path);
        for (int s = 0; s < PROJ_LEN; ++s)
            rf(ds.particles[static_cast<std::size_t>(i) * PROJ_LEN + s]);
    }
    return ds;
}

// ---------------------------------------------------------------------------
// match_cpu (THE E-STEP): for each particle, score it against every reference
//   projection and pick the best -- this is the O(N*M) projection-matching sweep
//   that dominates real cryo-EM walltime (catalog deep-dive) and that the GPU
//   parallelizes one-thread-per-particle in kernels.cu.
//
//   Tie-break: on an exact score tie we keep the LOWER angle index (the `>`
//   comparison only replaces on a strictly greater score), so the assignment is
//   deterministic and matches the GPU's identical rule. Because ncc_score runs
//   the same float arithmetic here and on the device, the argmax is bit-exact.
//
//   assign[i]     : chosen reference-angle index for particle i (output).
//   best_score[i] : the winning NCC score (output; reported, not verified-exact).
//   Complexity: O(N * M * PROJ_LEN). For the sample that is tiny; at cryo-EM
//   scale (millions of particles, thousands of refs) it is the whole ball game.
// ---------------------------------------------------------------------------
void match_cpu(const Dataset& ds, std::vector<int>& assign, std::vector<float>& best_score) {
    assign.assign(static_cast<std::size_t>(ds.n_particles), 0);
    best_score.assign(static_cast<std::size_t>(ds.n_particles), -2.0f);  // below NCC's -1 floor
    for (int i = 0; i < ds.n_particles; ++i) {
        const float* p = &ds.particles[static_cast<std::size_t>(i) * PROJ_LEN];
        int   best_a = 0;
        float best_s = -2.0f;
        for (int a = 0; a < N_ANGLES; ++a) {
            const float* r = &ds.refs[static_cast<std::size_t>(a) * PROJ_LEN];
            const float s = ncc_score(p, r, PROJ_LEN);
            if (s > best_s) { best_s = s; best_a = a; }   // strict > -> lowest-index tie-break
        }
        assign[static_cast<std::size_t>(i)]     = best_a;
        best_score[static_cast<std::size_t>(i)] = best_s;
    }
}

// ---------------------------------------------------------------------------
// reconstruct_cpu (THE M-STEP): back-project every particle's profile, along its
//   ASSIGNED angle, into the 2D density. One output pixel at a time (a gather),
//   so there are no atomics and the accumulation order is fixed -> deterministic
//   and identical to the GPU. See backproject_pixel() in reference_cpu.h.
//
//   recon : [IMG_SIZE*IMG_SIZE] reconstructed density (output, row-major).
//   Complexity: O(IMG_SIZE^2 * N). The result approximates the true density up
//   to the unfiltered-back-projection blur (THEORY §"real world").
// ---------------------------------------------------------------------------
void reconstruct_cpu(const Dataset& ds, const std::vector<int>& assign,
                     std::vector<float>& recon) {
    recon.assign(static_cast<std::size_t>(IMG_SIZE) * IMG_SIZE, 0.0f);

    // Precompute the per-angle view direction once (shared with the GPU, which
    // uploads the identical array) so we never recompute trig per pixel.
    std::vector<double> ref_thetas(N_ANGLES);
    for (int a = 0; a < N_ANGLES; ++a) ref_thetas[a] = ref_angle(a);

    for (int py = 0; py < IMG_SIZE; ++py) {
        for (int px = 0; px < IMG_SIZE; ++px) {
            recon[static_cast<std::size_t>(py) * IMG_SIZE + px] =
                backproject_pixel(ds.particles.data(), assign.data(),
                                  ref_thetas.data(), ds.n_particles, px, py);
        }
    }
}
