// ===========================================================================
// src/reference_cpu.h  --  Loader + serial DIR + serial dose warp/accumulate/DVH
// ---------------------------------------------------------------------------
// Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
//               (reduced-scope 2-D teaching version)
//
// Pure C++ (no CUDA). The per-voxel physics lives in demons.h (DIR) and dose.h
// (dose warp + DVH binning); kernels.cu reuses the SAME DM_HD/DS_HD functions, so
// the GPU results match this CPU reference within the tolerances documented in
// ../THEORY.md. This header is the trusted serial baseline the GPU is checked
// against, and whose wall time (measured in main.cu) makes the speed-up legible.
//
// WHAT'S DECLARED HERE
//   * ArtCase       -- one adaptive-radiotherapy problem instance (planning
//                      image, daily image, planning-frame dose, delivered dose).
//   * load_case     -- read the tiny synthetic sample (data/README.md format).
//   * register_cpu  -- the serial Thirion Demons solver (planning <- daily) -> DVF.
//   * warp_dose_cpu -- warp the delivered daily dose by the DVF into the planning
//                      frame (the "deformed dose" of one fraction).
//   * accumulate_cpu-- add a warped fraction dose into the running total.
//   * build_dvh_cpu -- bin a dose map into a DVH (counts per dose bin).
//   * dose_sum / dose_max -- deterministic scalar summaries reported to stdout.
//
// READ THIS AFTER: demons.h, dose.h. Then reference_cpu.cpp, then kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "demons.h"   // DemonsParams and the shared per-pixel DIR formulas
#include "dose.h"     // warp_dose_at, dvh_bin, DVH_BINS/DVH_MAX

// ---------------------------------------------------------------------------
// ArtCase -- one adaptive-radiotherapy problem instance.
//   All four grids are row-major [ny*nx] and share the same nx,ny (index=y*nx+x).
//     plan_img   : planning-day anatomy image (the FIXED image we register TO).
//     daily_img  : today's anatomy image (the MOVING image); differs from plan_img
//                  by a soft-tissue deformation (in the sample, a shifted blob).
//     plan_dose  : the dose the plan INTENDS to deliver, on the planning grid (Gy).
//     daily_dose : the dose actually delivered TODAY, laid down on today's grid
//                  (in the sample it equals plan_dose -- the linac fires the same
//                  fluence -- but the anatomy under it has moved, so where that
//                  dose lands in the body differs; the DVF corrects for that).
//   Intensities are in [0,1]; doses are in Gy (>= 0).
// ---------------------------------------------------------------------------
struct ArtCase {
    int nx = 0, ny = 0;                 // grid dimensions in voxels
    std::vector<double> plan_img;       // FIXED image  F                 [ny*nx]
    std::vector<double> daily_img;      // MOVING image M                 [ny*nx]
    std::vector<double> plan_dose;      // intended dose (planning frame)  [ny*nx], Gy
    std::vector<double> daily_dose;     // delivered dose (today's grid)   [ny*nx], Gy
};

// Load the sample text file (data/README.md documents the exact format):
//   line 1 : "nx ny"
//   then   : nx*ny plan_img, nx*ny daily_img, nx*ny plan_dose, nx*ny daily_dose
// Throws std::runtime_error on a missing/malformed file so demos fail loudly.
ArtCase load_case(const std::string& path);

// register_cpu: the SERIAL Demons reference. Registers daily_img (moving) onto
// plan_img (fixed) with P.iters iterations of (warp -> force -> add -> smooth),
// returning the displacement field ux,uy (each [ny*nx], zeroed inside). This is
// the DVF used to warp the dose. Complexity O(iters * nx * ny * radius).
void register_cpu(const ArtCase& c, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy);

// warp_dose_cpu: map the delivered daily dose into the planning frame via the
// DVF -- warped[i] = daily_dose(x+ux, y+uy) by bilinear gather (warp_dose_at).
// `warped` is resized to nx*ny. Shares the sampler with the GPU -> identical.
void warp_dose_cpu(const ArtCase& c,
                   const std::vector<double>& ux, const std::vector<double>& uy,
                   std::vector<double>& warped);

// accumulate_cpu: total[i] += add[i] for every voxel. This is the "summation of
// deformed doses": each fraction's warped dose is added into the running total in
// the common planning frame. total is resized/zeroed on the first call if empty.
void accumulate_cpu(std::vector<double>& total, const std::vector<double>& add);

// build_dvh_cpu: bin every voxel of `dose` into DVH_BINS counts (a differential
// dose-volume histogram: how many voxels fall in each dose interval). Returns a
// vector<unsigned> of length DVH_BINS. Integer counts -> exactly reproducible and
// directly comparable to the GPU's atomic-int histogram.
std::vector<unsigned> build_dvh_cpu(const std::vector<double>& dose);

// dose_sum / dose_max: deterministic scalar summaries of a dose map (total energy
// proxy and hot-spot). Reported to stdout at fixed precision; computed in double
// on the serial CPU field so they are bit-stable every run.
double dose_sum(const std::vector<double>& dose);
double dose_max(const std::vector<double>& dose);
