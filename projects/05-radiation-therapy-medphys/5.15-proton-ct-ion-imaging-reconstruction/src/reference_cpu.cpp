// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial SART: the baseline we trust
// ---------------------------------------------------------------------------
// Project 5.15 : Proton CT & Ion Imaging Reconstruction
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- readable loops, no parallelism, no cleverness --
//   so that when the GPU and CPU agree we believe the GPU. It computes the SAME
//   SART reconstruction the kernels do, calling the SAME shared physics
//   (pct_physics.h) and using the SAME fixed-point accumulation, so the two
//   results are bit-identical (docs/PATTERNS.md sections 2 and 3).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//   Compare each function against its twin in kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_pct : parse the text list-mode format (see data/README.md).
//   Layout:
//     n half iters relax path_samples n_protons          <- header line
//     <n*n ground-truth RSP floats, row-major>            <- for reporting only
//     x0 y0 x1 y1 entry_angle exit_angle wepl             <- one row per proton
//   The ground-truth block lets the demo report reconstruction error against a
//   KNOWN answer (docs/PATTERNS.md section 6); it is NOT used by the solver.
// ---------------------------------------------------------------------------
PctProblem load_pct(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open pCT list-mode file: " + path);

    PctProblem prob;
    int n_protons = 0;
    if (!(in >> prob.geom.n >> prob.geom.half >> prob.iters >> prob.relax
             >> prob.path_samples >> n_protons))
        throw std::runtime_error("bad header (expected n half iters relax path_samples n_protons) in " + path);
    if (prob.geom.n <= 0 || prob.iters <= 0 || prob.path_samples <= 0 || n_protons <= 0)
        throw std::runtime_error("non-positive geometry/counts in " + path);

    // Ground-truth RSP image (row-major n*n).
    const std::size_t cells = static_cast<std::size_t>(prob.geom.n) * prob.geom.n;
    prob.truth.resize(cells);
    for (std::size_t i = 0; i < cells; ++i)
        if (!(in >> prob.truth[i]))
            throw std::runtime_error("ground-truth RSP block truncated in " + path);

    // Proton histories.
    prob.protons.resize(static_cast<std::size_t>(n_protons));
    for (int i = 0; i < n_protons; ++i) {
        Proton& p = prob.protons[static_cast<std::size_t>(i)];
        if (!(in >> p.x0 >> p.y0 >> p.x1 >> p.y1
                 >> p.entry_angle >> p.exit_angle >> p.wepl))
            throw std::runtime_error("proton list truncated in " + path);
    }
    return prob;
}

// ---------------------------------------------------------------------------
// world_to_voxel : nearest-voxel binning of a world point.
//   The image covers [-half, half]^2 with n voxels per side. Voxel (ix,iy) is
//   centred at world (-half + ix*vs, -half + iy*vs), vs = voxel size. We round
//   to the nearest voxel centre (a "nearest-neighbour" projector -- the simplest
//   that teaches the idea; a production pCT uses bilinear/Siddon weighting, an
//   exercise in README). Returns the row-major index iy*n+ix, or -1 if outside.
//   IDENTICAL logic runs on the device (kernels.cu) so binning matches exactly.
// ---------------------------------------------------------------------------
int world_to_voxel(const PctGeom& geom, float x, float y) {
    const float vs = geom.voxel_size();
    if (vs <= 0.0f) return -1;
    // (world + half)/vs -> fractional voxel coordinate; +0.5 then floor = round.
    const int ix = static_cast<int>(std::floor((x + geom.half) / vs + 0.5f));
    const int iy = static_cast<int>(std::floor((y + geom.half) / vs + 0.5f));
    if (ix < 0 || ix >= geom.n || iy < 0 || iy >= geom.n) return -1;
    return iy * geom.n + ix;
}

// ---------------------------------------------------------------------------
// forward_project_cpu : estimate a proton's WEPL through the CURRENT RSP image.
//   We sample the MLP at `path_samples` equally spaced chord fractions, look up
//   the RSP of the voxel each sample lands in, and multiply by the per-sample
//   segment length seg_len = chord_length / path_samples. Summing gives the
//   line integral of RSP along the curved path == the estimated WEPL.
//   (This is a midpoint-style quadrature of integral RSP ds; more samples ->
//   closer to the true integral. path_samples is a documented knob.)
// ---------------------------------------------------------------------------
float forward_project_cpu(const Proton& p, const PctGeom& geom,
                          const std::vector<float>& rsp, int path_samples) {
    // Chord length sets the physical measure ds each sample represents.
    const float dx = p.x1 - p.x0, dy = p.y1 - p.y0;
    const float chord_len = std::sqrt(dx * dx + dy * dy);
    const float seg_len = chord_len / path_samples;      // cm per sample

    float wepl_est = 0.0f;
    for (int s = 0; s < path_samples; ++s) {
        // Sample at the MIDPOINT of segment s: t = (s + 0.5)/path_samples.
        const float t = (s + 0.5f) / path_samples;
        float px, py;
        mlp_point(p, t, &px, &py);                        // curved MLP position
        const int v = world_to_voxel(geom, px, py);
        if (v >= 0)
            wepl_est += rsp[static_cast<std::size_t>(v)] * seg_len;
    }
    return wepl_est;
}

// ---------------------------------------------------------------------------
// reconstruct_cpu : serial SART. This is the reference the GPU must reproduce.
//
//   SART update (one sweep = one iteration over ALL protons):
//     For each proton p:
//        est_p  = forward_project(p)                 (current WEPL estimate)
//        r_p    = wepl_p - est_p                      (residual, cm)
//        Lp     = N_hit_p * seg_len                   (in-grid path length ||a_p||_1)
//        corr_p = r_p / Lp                            (RSP correction per cm)
//        for each in-grid sample voxel v of p:
//            num[v] += corr_p * seg_len   (a_pv-weighted correction, a_pv=seg_len)
//            den[v] += seg_len            (total length seen by voxel v)
//     Then, once all protons are tallied:
//        rsp[v] += relax * num[v] / den[v]     (den[v] > 0)
//   This is the textbook SART step sum_p a_pv (b_p - <a_p,x>)/||a_p||_1 over
//   sum_p a_pv, with a_pv = seg_len for each MLP sample landing in voxel v.
//
//   Accumulating num/den as FIXED-POINT int64 (value*PCT_FIXED_SCALE) makes the
//   sum ORDER-INDEPENDENT, so the parallel GPU tally (atomicAdd on int64) yields
//   the identical bits (docs/PATTERNS.md section 3). We convert back to float
//   before the division/update.
//
//   Complexity: O(iters * n_protons * path_samples). Serial here; the GPU does
//   the inner proton loop in parallel per sweep.
// ---------------------------------------------------------------------------
void reconstruct_cpu(const PctProblem& prob, std::vector<float>& result) {
    const PctGeom& geom = prob.geom;
    const std::size_t cells = static_cast<std::size_t>(geom.n) * geom.n;

    // RSP image, initialised to vacuum (0). SART fills it in.
    result.assign(cells, 0.0f);

    // Fixed-point accumulators (reused each sweep). int64 so many adds cannot
    // overflow for teaching-scale inputs.
    std::vector<std::int64_t> num_fx(cells), den_fx(cells);

    for (int it = 0; it < prob.iters; ++it) {
        // Clear the per-sweep tallies.
        std::fill(num_fx.begin(), num_fx.end(), std::int64_t(0));
        std::fill(den_fx.begin(), den_fx.end(), std::int64_t(0));

        // --- Tally every proton into the shared accumulators --------------
        for (const Proton& p : prob.protons) {
            // Segment length this proton assigns to each sample it hits.
            const float dx = p.x1 - p.x0, dy = p.y1 - p.y0;
            const float chord_len = std::sqrt(dx * dx + dy * dy);
            const float seg_len = chord_len / prob.path_samples;

            // Forward project + count in-grid hits in ONE pass so the residual
            // and the scatter use exactly the same samples.
            float est = 0.0f;
            int   n_hit = 0;
            for (int s = 0; s < prob.path_samples; ++s) {
                const float t = (s + 0.5f) / prob.path_samples;
                float px, py; mlp_point(p, t, &px, &py);
                const int v = world_to_voxel(geom, px, py);
                if (v >= 0) { est += result[static_cast<std::size_t>(v)] * seg_len; ++n_hit; }
            }
            if (n_hit == 0) continue;                       // proton missed the grid

            // SART correction: divide the WEPL residual by the proton's TOTAL
            // in-grid path length (its L1 row norm, ||a_p||_1 = n_hit*seg_len),
            // giving a per-unit-length RSP correction. Weighting by seg_len when
            // we scatter (below) then reconstructs the standard SART update.
            const float resid = p.wepl - est;               // WEPL residual (cm)
            const float corr  = resid / (n_hit * seg_len);  // RSP correction / cm

            // Second pass: scatter length-weighted correction into fixed point.
            for (int s = 0; s < prob.path_samples; ++s) {
                const float t = (s + 0.5f) / prob.path_samples;
                float px, py; mlp_point(p, t, &px, &py);
                const int v = world_to_voxel(geom, px, py);
                if (v < 0) continue;
                // Round-to-nearest into int64 fixed point (matches device rint).
                const std::int64_t num_add =
                    static_cast<std::int64_t>(std::llround(
                        static_cast<double>(corr) * static_cast<double>(seg_len) * PCT_FIXED_SCALE));
                const std::int64_t den_add =
                    static_cast<std::int64_t>(std::llround(
                        static_cast<double>(seg_len) * PCT_FIXED_SCALE));
                num_fx[static_cast<std::size_t>(v)] += num_add;
                den_fx[static_cast<std::size_t>(v)] += den_add;
            }
        }

        // --- Apply the SART update: rsp[v] += relax * num/den -------------
        for (std::size_t v = 0; v < cells; ++v) {
            if (den_fx[v] == 0) continue;                   // untouched voxel
            const double num = static_cast<double>(num_fx[v]) / PCT_FIXED_SCALE;
            const double den = static_cast<double>(den_fx[v]) / PCT_FIXED_SCALE;
            result[v] += static_cast<float>(prob.relax * (num / den));
        }
    }
}
