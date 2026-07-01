// ===========================================================================
// src/reference_cpu.cpp  --  CPU reference: loader, X^T X precompute, GLM loop
// ---------------------------------------------------------------------------
// Project 4.16 : Functional MRI Analysis
//
// This is the trusted, plain-C++ baseline. main.cu runs it AND the GPU kernel
// and asserts they agree (near-exactly, because both call the same fit_voxel()
// from glm.h). Nothing here uses CUDA -- it is compiled by cl.exe/g++.
//
// READ THIS AFTER: glm.h and reference_cpu.h. Then kernels.cu for the GPU twin.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_fmri: read the tiny committed sample. The format is deliberately plain
//   text (not NIfTI) so the demo has zero dependencies and the numbers are
//   human-inspectable. See data/README.md for the exact layout and provenance.
// ---------------------------------------------------------------------------
FmriDataset load_fmri(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open fMRI sample file: " + path);

    FmriDataset ds;
    // Header: V, T, TR_seconds, block_scans.
    if (!(in >> ds.V >> ds.design.T >> ds.design.TR_seconds >> ds.design.block_scans))
        throw std::runtime_error("malformed header (need: V T TR_seconds block_scans) in " + path);
    if (ds.V <= 0 || ds.design.T <= 0)
        throw std::runtime_error("V and T must be positive in " + path);
    if (ds.design.block_scans <= 0)
        throw std::runtime_error("block_scans must be positive in " + path);

    const int V = ds.V;
    const int T = ds.design.T;
    ds.bold.assign(static_cast<std::size_t>(V) * T, 0.0);
    ds.true_active.assign(V, 0);

    // Body: V rows, each "active_flag  y_0 ... y_{T-1}".
    for (int v = 0; v < V; ++v) {
        int flag = 0;
        if (!(in >> flag))
            throw std::runtime_error("unexpected EOF reading active_flag for voxel "
                                     + std::to_string(v) + " in " + path);
        ds.true_active[v] = flag;
        for (int t = 0; t < T; ++t) {
            double val = 0.0;
            if (!(in >> val))
                throw std::runtime_error("unexpected EOF reading BOLD sample (voxel "
                                         + std::to_string(v) + ", scan " + std::to_string(t)
                                         + ") in " + path);
            ds.bold[static_cast<std::size_t>(v) * T + t] = val;
        }
    }
    return ds;
}

// ---------------------------------------------------------------------------
// compute_XtX_inv: assemble X^T X (a 3x3 Gram matrix of the design columns) by
//   summing outer products over the T scans, then invert with invert_sym3().
//   X is voxel-independent, so this runs ONCE and the inverse is reused for all
//   voxels (CPU and GPU alike). Building X on the fly here mirrors exactly how
//   fit_voxel() builds it, so X^T X is consistent with the per-voxel fits.
// ---------------------------------------------------------------------------
double compute_XtX_inv(const GlmDesign& d, double out_inv[9]) {
    const int T = d.T;
    // Six unique entries of the symmetric 3x3 Gram matrix.
    double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
    for (int t = 0; t < T; ++t) {
        const double x0 = design_value(0, t, T, d.TR_seconds, d.block_scans);
        const double x1 = design_value(1, t, T, d.TR_seconds, d.block_scans);
        const double x2 = 1.0;    // intercept column
        a00 += x0 * x0; a01 += x0 * x1; a02 += x0 * x2;
        a11 += x1 * x1; a12 += x1 * x2;
        a22 += x2 * x2;
    }
    return invert_sym3(a00, a01, a02, a11, a12, a22, out_inv);
}

// ---------------------------------------------------------------------------
// glm_cpu: the reference activation map. One fit_voxel() call per voxel.
// ---------------------------------------------------------------------------
void glm_cpu(const FmriDataset& ds, const double XtX_inv[9],
             std::vector<double>& tstat, std::vector<double>& beta) {
    const int V = ds.V;
    const int T = ds.design.T;
    tstat.assign(V, 0.0);
    beta.assign(V, 0.0);
    for (int v = 0; v < V; ++v) {
        // Pointer to voxel v's contiguous time-series (voxel-major layout).
        const double* y = ds.bold.data() + static_cast<std::size_t>(v) * T;
        const VoxelStat s = fit_voxel(y, ds.design, XtX_inv);
        tstat[v] = s.tstat;
        beta[v]  = s.beta_task;
    }
}
