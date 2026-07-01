// ===========================================================================
// src/reference_cpu.h  --  CSR problem, DVH stats, and CPU reference optimizer
// ---------------------------------------------------------------------------
// Project 5.2 : Radiotherapy Treatment-Plan Optimization
//
// WHAT LIVES HERE
//   * Problem  -- the whole FMO problem: the sparse dose-influence matrix D in
//                 CSR format, the per-voxel objective specs, and optimizer knobs.
//   * PlanStats-- the deterministic plan-quality summary we print + verify
//                 (final objective, PTV coverage / homogeneity, OAR max dose).
//   * load_problem()  -- parse the tiny synthetic sample (see data/README.md).
//   * optimize_cpu()  -- the trusted serial reference optimizer (projected
//                 gradient descent). The GPU twin (kernels.cu) runs the SAME
//                 algorithm; main.cu compares their PlanStats + fluence.
//
//   This header is PURE C++ (no CUDA syntax): reference_cpu.cpp is built by the
//   host compiler, and kernels.cu (nvcc) also includes it to reuse Problem/
//   PlanStats. The per-voxel penalty/gradient math is in fmo.h (shared HD-core).
//
// READ THIS AFTER: fmo.h. Then reference_cpu.cpp (the CPU optimizer body),
//   then kernels.cuh / kernels.cu (the cuSPARSE GPU optimizer).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "fmo.h"   // VoxelSpec, StructKind, voxel_penalty/residual, project_nonneg

// ---------------------------------------------------------------------------
// Problem: everything needed to run FMO on one plan.
//
// The dose-influence matrix D is n_vox x n_beam, stored in COMPRESSED SPARSE
// ROW (CSR) format -- the layout cuSPARSE expects and the standard for dij
// matrices. CSR stores only the nonzeros of each row contiguously:
//   * row_ptr : length n_vox + 1. Row v's nonzeros are the half-open index
//               range [row_ptr[v], row_ptr[v+1]) into col_idx / values.
//               row_ptr[n_vox] == total number of nonzeros (nnz).
//   * col_idx : length nnz. col_idx[k] = the beamlet index j of nonzero k.
//   * values  : length nnz. values[k] = D[v, j] = dose per unit fluence (Gy).
// This stores nnz numbers instead of n_vox*n_beam, the whole reason a 10^6 x
// 10^4 matrix (10^10 entries) fits in GPU memory: only ~1% is nonzero.
// ---------------------------------------------------------------------------
struct Problem {
    int n_vox  = 0;                 // number of dose voxels   (rows of D)
    int n_beam = 0;                 // number of beamlets      (cols of D, = |x|)

    std::vector<int>   row_ptr;     // CSR row pointers, length n_vox + 1
    std::vector<int>   col_idx;     // CSR column indices, length nnz
    std::vector<float> values;      // CSR nonzero values D[v,j] (Gy/fluence), nnz

    std::vector<VoxelSpec> voxels;  // per-voxel objective spec, length n_vox

    // Optimizer hyper-parameters (read from the sample so it is self-describing).
    int   iters       = 0;          // number of projected-gradient iterations
    float step        = 0.0f;       // gradient step size eta (learning rate)
    float d_rx        = 0.0f;       // PTV prescription dose (Gy), for reporting

    int nnz() const { return row_ptr.empty() ? 0 : row_ptr.back(); }
};

// ---------------------------------------------------------------------------
// PlanStats: the deterministic plan-quality report (printed to stdout AND the
// object compared for CPU-vs-GPU agreement). Every field is a reproducible
// function of the final dose, so identical optimizers => identical stats.
//   objective     : final F(x), the minimized cost (lower = better plan).
//   ptv_mean      : mean PTV dose (Gy)   -- should sit near the prescription.
//   ptv_min       : min  PTV dose (Gy)   -- coverage: the cold spot.
//   ptv_max       : max  PTV dose (Gy)   -- hot spot inside the target.
//   oar_mean      : mean OAR dose (Gy)   -- sparing metric (lower = better).
//   oar_max       : max  OAR dose (Gy)   -- the peak organ dose vs tolerance.
//   homogeneity   : (ptv_max - ptv_min) / ptv_mean -- 0 = perfectly uniform PTV.
// These mirror clinical Dose-Volume-Histogram (DVH) point metrics.
// ---------------------------------------------------------------------------
struct PlanStats {
    double objective   = 0.0;
    double ptv_mean    = 0.0;
    double ptv_min     = 0.0;
    double ptv_max     = 0.0;
    double oar_mean    = 0.0;
    double oar_max     = 0.0;
    double homogeneity = 0.0;
};

// Parse the synthetic sample file into a Problem (format in data/README.md).
// Throws std::runtime_error on a missing/malformed file so demos fail loudly.
Problem load_problem(const std::string& path);

// Compute the dose d = D x (dense length n_vox) from a fluence x on the CPU by
// walking the CSR rows. Exposed so main.cu can turn the final fluence into a
// dose and derive the reported stats identically for both paths.
void csr_spmv_cpu(const Problem& p, const std::vector<float>& x,
                  std::vector<float>& dose);

// Turn a dose vector into the deterministic PlanStats summary (pure function of
// the dose + the voxel specs; used for both CPU and GPU final doses so the
// comparison is apples-to-apples).
PlanStats compute_stats(const Problem& p, const std::vector<float>& dose);

// The CPU reference optimizer: projected gradient descent on the fluence.
//   Inputs : the Problem (D in CSR + objective specs + hyper-params).
//   Output : x_out = the optimized fluence (length n_beam), from which main.cu
//            derives the final dose and stats. This is the trusted baseline the
//            GPU optimizer is checked against.
void optimize_cpu(const Problem& p, std::vector<float>& x_out);
