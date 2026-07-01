// ===========================================================================
// src/reference_cpu.h  --  Dataset model, design precompute, CPU GLM reference
// ---------------------------------------------------------------------------
// Project 4.16 : Functional MRI Analysis
//
// WHY A SEPARATE HEADER
//   reference_cpu.cpp is compiled by the PLAIN C++ compiler and must never see
//   any __global__/CUDA syntax, so its declarations cannot live in kernels.cuh.
//   Both main.cu and reference_cpu.cpp include THIS pure-C++ header (plus glm.h,
//   whose FMRI_HD functions are plain C++ when nvcc is absent) so they agree on
//   the data model and the reference-function signatures.
//
// WHAT LIVES HERE
//   * FmriDataset      -- the loaded problem (V voxels x T scans + design params)
//   * load_fmri()      -- parse the tiny committed sample file
//   * compute_XtX_inv()-- precompute the voxel-independent (X^T X)^-1 once
//   * glm_cpu()        -- the trusted per-voxel GLM baseline (loops fit_voxel())
//
//   The GPU kernel reuses ALL of the science in glm.h; only the outer "loop over
//   voxels" differs (a for-loop here, one-thread-per-voxel there).
//
// READ THIS AFTER: glm.h (the per-voxel math). Then reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "glm.h"   // GlmDesign, VoxelStat, fit_voxel, design_value (pure C++ here)

// ---------------------------------------------------------------------------
// FmriDataset: everything the analysis needs.
//   bold is stored VOXEL-MAJOR (row v = the T-length time-series of voxel v):
//   bold[v*T + t]. This layout means one voxel's whole series is contiguous, so
//   the CPU reads it with unit stride and each GPU thread streams its own row.
//   (An alternative time-major layout would coalesce better across threads at a
//   given t; we discuss that trade in THEORY §GPU-mapping and keep voxel-major
//   for a readable one-thread-one-row mapping.)
// ---------------------------------------------------------------------------
struct FmriDataset {
    int V = 0;                   // number of voxels
    GlmDesign design;            // T, TR_seconds, block_scans (defines X)
    std::vector<double> bold;    // [V * T] BOLD signal, voxel-major
    // Ground-truth bookkeeping from the synthetic generator (labels only; NOT
    // used by the fit). true_active[v] != 0 means voxel v had task signal added.
    std::vector<int> true_active;
};

// Parse the sample text file. Format (whitespace-separated, see data/README.md):
//   line/token stream:  V  T  TR_seconds  block_scans
//   then V rows, each:   active_flag  y_0 y_1 ... y_{T-1}
// Throws std::runtime_error if the file is missing or malformed.
FmriDataset load_fmri(const std::string& path);

// ---------------------------------------------------------------------------
// compute_XtX_inv: build the 3x3 normal-equations matrix X^T X from the design
//   params and invert it -- ONCE, because X (hence X^T X) is identical for every
//   voxel. Returns the determinant (0 => rank-deficient design, a bug in the
//   experiment setup) and writes the row-major inverse into out_inv[9]. Shared
//   by the CPU path and uploaded to the GPU (both use the same inverse).
// ---------------------------------------------------------------------------
double compute_XtX_inv(const GlmDesign& d, double out_inv[9]);

// ---------------------------------------------------------------------------
// glm_cpu: the CPU reference. For each voxel, call fit_voxel() (the shared HD
//   core) and store its t-statistic and task-beta. This is the trusted baseline
//   the GPU result is checked against -- same fit_voxel() => same numbers.
//   tstat and beta are resized to V.
// ---------------------------------------------------------------------------
void glm_cpu(const FmriDataset& ds, const double XtX_inv[9],
             std::vector<double>& tstat, std::vector<double>& beta);
