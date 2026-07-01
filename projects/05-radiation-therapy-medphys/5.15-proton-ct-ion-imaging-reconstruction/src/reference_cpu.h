// ===========================================================================
// src/reference_cpu.h  --  pCT problem, loader, and CPU SART reference
// ---------------------------------------------------------------------------
// Project 5.15 : Proton CT & Ion Imaging Reconstruction
//
// WHAT THIS PROJECT COMPUTES
//   Reconstruct a 2-D RELATIVE STOPPING POWER (RSP) map from list-mode proton
//   histories. Each proton contributes ONE equation:
//        integral over its MOST-LIKELY PATH of RSP(x,y)  =  measured WEPL
//   i.e. "the water-equivalent path length I measured equals the sum of the
//   stopping powers of the voxels I actually traversed." We solve the whole
//   over-determined system with SART (Simultaneous Algebraic Reconstruction
//   Technique) -- an iterative forward-project / residual / backproject loop.
//
// WHY A GPU
//   A clinical pCT scan is ~10^8 protons; each proton's MLP and its
//   forward/backprojection are INDEPENDENT of the others within one SART sweep,
//   so we give each proton its own GPU thread. This is the same "one thread per
//   history" massively-parallel pattern as Monte-Carlo dose (5.01), fused with
//   the per-path GATHER/scatter of CT backprojection (4.01). See docs/PATTERNS.md.
//
// DETERMINISM (docs/PATTERNS.md section 3)
//   Many protons cross the same voxel, so the backprojection is a many-writer
//   accumulation. Floating-point atomicAdd is NON-associative -> irreproducible.
//   We therefore accumulate the SART numerator/denominator in FIXED-POINT
//   integers (scaled to int64) which add commutatively; the GPU result is then
//   order-independent AND bit-identical to the CPU reference. reference_cpu.cpp
//   uses the SAME fixed-point accumulation so the two agree exactly.
//
//   Pure-C++ header (no CUDA). kernels.cu reuses PctProblem, PctGeom, Proton.
//   READ THIS AFTER: pct_physics.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pct_physics.h"   // PctGeom, Proton, mlp_point (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// PctProblem : one reconstruction problem -- the grid geometry, the list of
// measured protons, the SART controls, and (for teaching/verification) the
// GROUND-TRUTH RSP map the synthetic data was generated from.
// ---------------------------------------------------------------------------
struct PctProblem {
    PctGeom geom;                     // reconstruction grid (n, half)
    int     iters        = 0;        // SART iterations (full sweeps over all protons)
    float   relax        = 0.0f;     // SART relaxation factor lambda in (0,1]
    int     path_samples = 0;        // MLP quadrature samples per proton
    std::vector<Proton> protons;      // the measured list-mode histories
    std::vector<float>  truth;        // [n*n] ground-truth RSP (synthetic only; for reporting)
};

// ---------------------------------------------------------------------------
// FIXED-POINT SCALE for the deterministic accumulators.
//   We accumulate correction*weight and weight in int64 as value*FIXED_SCALE.
//   1e6 keeps ~6 decimal digits of the (order-1) RSP corrections -- plenty for
//   a teaching reconstruction, and integer adds are exact & order-independent.
//   Chosen so |sum| stays well inside int64 for ~10^4 protons.
// ---------------------------------------------------------------------------
static constexpr double PCT_FIXED_SCALE = 1.0e6;

// Load a PctProblem from the text format documented in data/README.md.
//   header: "<n> <half> <iters> <relax> <path_samples> <n_protons>"
//   then n*n ground-truth RSP floats (row-major),
//   then n_protons rows: "x0 y0 x1 y1 entry_angle exit_angle wepl".
PctProblem load_pct(const std::string& path);

// Map a world (x,y) in cm to the nearest voxel linear index, or -1 if outside
// the grid. Shared helper (mirrored on the device) so CPU/GPU bin identically.
int world_to_voxel(const PctGeom& geom, float x, float y);

// Forward-project ONE proton through the current RSP image: sample its MLP at
// `path_samples` points, accumulate RSP*seg_len. Returns the estimated WEPL.
// Shared by CPU and (mirrored equivalently) GPU. seg_len = chord_len/path_samples.
float forward_project_cpu(const Proton& p, const PctGeom& geom,
                          const std::vector<float>& rsp, int path_samples);

// CPU reference SART reconstruction (the trusted baseline).
//   * Starts from RSP = 0.
//   * Each iteration: for every proton, forward-project along its MLP, form the
//     residual (measured - estimated) WEPL, and scatter a length-weighted
//     correction into fixed-point numerator/denominator accumulators; then
//     update rsp[v] += relax * num[v]/den[v].
//   result is sized to n*n and filled with the reconstructed RSP.
void reconstruct_cpu(const PctProblem& prob, std::vector<float>& result);
