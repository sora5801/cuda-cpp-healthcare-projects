// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ gamma-index baseline we trust
// ---------------------------------------------------------------------------
// Project 5.9 -- Gamma-Index Dose Comparison
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- plain nested loops, no parallelism, no cleverness
//   -- so that when the GPU and CPU agree, we believe the GPU. The per-pair math
//   comes from gamma_core.h, the SAME header the GPU kernel includes, so the two
//   implementations are guaranteed to compute bit-identical gamma values
//   (PATTERNS.md §2). Verification then becomes an EXACT comparison, not a fuzzy
//   tolerance (see THEORY §6).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: gamma_core.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"
#include "gamma_core.h"     // gamma_sq_term(), GammaParams -- the shared physics

#include <algorithm>        // std::max
#include <cmath>            // std::sqrt, std::ceil

// ---------------------------------------------------------------------------
// map_max -- the largest dose in a map. Used to turn the "% dose difference"
//   criterion into an absolute dose (global gamma normalization) and to set the
//   low-dose analysis threshold. O(N), trivially correct.
// ---------------------------------------------------------------------------
static double map_max(const std::vector<float>& m) {
    double mx = 0.0;
    for (float v : m) mx = std::max(mx, static_cast<double>(v));
    return mx;
}

// ---------------------------------------------------------------------------
// gamma_map_cpu -- exhaustive, distance-limited gamma at every reference voxel.
//
//   The nested-loop structure (THEORY §3):
//     for each reference voxel (rx, ry):
//        best = +infinity
//        for each evaluated voxel (ex, ey) within the search window:
//           term = gamma_sq_term(dose_eval, dose_ref, dist^2, params)
//           best = min(best, term)
//        gamma[i] = sqrt(best)
//
//   WHY a search WINDOW and not the whole map: a candidate further than
//   gamma=1's worth of pure distance can never beat a nearby exact-dose match,
//   so we cap the search at `search_mm` (a few DTA criteria wide). This turns an
//   O(N^2) all-pairs scan into O(N*K) with K = window area -- the exact
//   reduction the catalog calls "distance-limited search". We use the SAME
//   window on the GPU so both sides scan an identical candidate set (essential
//   for the exact CPU==GPU check).
// ---------------------------------------------------------------------------
void gamma_map_cpu(const DoseProblem& prob, std::vector<float>& gamma_out) {
    const int W = prob.width;
    const int H = prob.height;
    gamma_out.assign(static_cast<std::size_t>(prob.size()), 0.0f);

    // --- Turn human-facing criteria into the precomputed inverse-squared
    //     normalizers the inner loop multiplies by (see gamma_core.h). The
    //     dose-difference criterion is GLOBAL: a percentage of the reference
    //     map's maximum dose (the most common clinical normalization).
    const double ref_max  = map_max(prob.ref);
    const double dd_crit  = prob.dd_percent * 0.01 * ref_max;   // [dose]
    const double dta_crit = prob.dta_mm;                        // [mm]
    GammaParams gp;
    gp.inv_dd_crit_sq  = 1.0 / (dd_crit  * dd_crit);
    gp.inv_dta_crit_sq = 1.0 / (dta_crit * dta_crit);

    // --- Search-window half-width in VOXELS. We search out to `search_mm`,
    //     chosen as a small multiple of the DTA criterion. Any evaluated point
    //     beyond this contributes a distance term > (search_mm/dta_crit)^2, so
    //     if that already exceeds the best gamma^2 found, it cannot win.
    const double search_mm  = 3.0 * dta_crit;                   // [mm]
    const int    radius_vox = static_cast<int>(
                                  std::ceil(search_mm / prob.spacing_mm));

    // Low-dose analysis threshold: only reference voxels above this absolute
    // dose are scored into the pass-rate (points below are set to gamma=0 and
    // excluded, mirroring clinical practice). We still WRITE a gamma value for
    // every voxel so the output map has full shape.
    const double dose_thresh = prob.dose_threshold_frac * ref_max;

    // --- The two outer loops: one iteration per reference voxel. This is the
    //     work the GPU assigns one-thread-per-iteration (kernels.cu).
    for (int ry = 0; ry < H; ++ry) {
        for (int rx = 0; rx < W; ++rx) {
            const int    ridx     = ry * W + rx;
            const double dose_ref = prob.ref[ridx];

            // Below-threshold background: not analyzed, gamma left at 0.
            if (dose_ref < dose_thresh) { gamma_out[ridx] = 0.0f; continue; }

            // Running minimum of gamma^2 over the candidate window.
            double best_sq = 1.0e30;   // "infinity" for our purposes

            // Clamp the window to the grid so we never index out of bounds.
            const int ex0 = std::max(0,     rx - radius_vox);
            const int ex1 = std::min(W - 1, rx + radius_vox);
            const int ey0 = std::max(0,     ry - radius_vox);
            const int ey1 = std::min(H - 1, ry + radius_vox);

            // --- Inner search: iterate evaluated voxels in a FIXED order
            //     (row-major over the window). Fixed order + the shared
            //     gamma_sq_term() is what makes the CPU and GPU minima identical.
            for (int ey = ey0; ey <= ey1; ++ey) {
                for (int ex = ex0; ex <= ex1; ++ex) {
                    // Physical squared distance between reference voxel (rx,ry)
                    // and evaluated voxel (ex,ey). dx,dy in voxels -> * spacing.
                    const double dx = (ex - rx) * prob.spacing_mm;   // [mm]
                    const double dy = (ey - ry) * prob.spacing_mm;   // [mm]
                    const double dist_sq = dx * dx + dy * dy;        // [mm^2]

                    const double dose_eval = prob.eval[ey * W + ex];
                    const double term = gamma_sq_term(dose_eval, dose_ref,
                                                      dist_sq, gp);
                    if (term < best_sq) best_sq = term;
                }
            }

            // One sqrt at the very end: gamma = min over the window of sqrt(.)
            gamma_out[ridx] = static_cast<float>(std::sqrt(best_sq));
        }
    }
}
