// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ FMO baseline we trust
// ---------------------------------------------------------------------------
// Project 5.2 : Radiotherapy Treatment-Plan Optimization
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU optimizer is checked against. Everything here is
//   written to be OBVIOUSLY correct -- plain serial loops, no parallelism, no
//   cleverness -- so that when the GPU and CPU agree we believe the GPU. It
//   implements: the sample loader, a CSR SpMV (d = D x), the DVH-style stats,
//   and the projected-gradient-descent optimizer. The per-voxel penalty and
//   gradient math come from fmo.h (shared with the GPU -> identical scalar math).
//
//   Compiled by the host C++ compiler only (NO CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: fmo.h, reference_cpu.h. Compare optimize_cpu() below against
//   optimize_gpu() in kernels.cu -- same algorithm, different engine.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sqrt, std::fabs
#include <fstream>     // std::ifstream
#include <limits>      // std::numeric_limits
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_problem: parse the tiny synthetic sample (layout documented verbatim in
//   data/README.md). The file is a self-describing FMO instance:
//
//     n_vox n_beam nnz iters step d_rx        (header line)
//     kind target weight                      (x n_vox : the VoxelSpec rows)
//     ...
//     row_ptr[0] row_ptr[1] ... row_ptr[n_vox]    (n_vox+1 ints)
//     col_idx[0] ... col_idx[nnz-1]               (nnz ints)
//     values[0]  ... values[nnz-1]                (nnz floats)
//
//   We read purely by whitespace-separated tokens (newlines are irrelevant to
//   operator>>), which keeps the parser short and robust. Every malformed or
//   short field throws so a broken sample fails loudly instead of silently
//   optimizing garbage.
// ---------------------------------------------------------------------------
Problem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open problem file: " + path);

    Problem p;
    int nnz = 0;
    // Header: sizes + optimizer hyper-parameters.
    if (!(in >> p.n_vox >> p.n_beam >> nnz >> p.iters >> p.step >> p.d_rx))
        throw std::runtime_error("bad header (expected 'n_vox n_beam nnz iters "
                                 "step d_rx') in " + path);
    if (p.n_vox <= 0 || p.n_beam <= 0 || nnz <= 0 || p.iters <= 0)
        throw std::runtime_error("invalid problem dimensions in " + path);

    // Per-voxel objective specs: kind, target dose (Gy), penalty weight.
    p.voxels.resize(p.n_vox);
    for (int v = 0; v < p.n_vox; ++v) {
        int kind; float target, weight;
        if (!(in >> kind >> target >> weight))
            throw std::runtime_error("truncated voxel spec list in " + path);
        p.voxels[v] = VoxelSpec{target, weight, kind};
    }

    // CSR arrays. row_ptr has n_vox+1 entries; col_idx/values have nnz entries.
    p.row_ptr.resize(p.n_vox + 1);
    for (int i = 0; i <= p.n_vox; ++i)
        if (!(in >> p.row_ptr[i])) throw std::runtime_error("truncated row_ptr in " + path);
    if (p.row_ptr.back() != nnz)
        throw std::runtime_error("row_ptr[n_vox] != nnz in " + path);

    p.col_idx.resize(nnz);
    for (int k = 0; k < nnz; ++k)
        if (!(in >> p.col_idx[k])) throw std::runtime_error("truncated col_idx in " + path);

    p.values.resize(nnz);
    for (int k = 0; k < nnz; ++k)
        if (!(in >> p.values[k])) throw std::runtime_error("truncated values in " + path);

    return p;
}

// ---------------------------------------------------------------------------
// csr_spmv_cpu: dose = D x, the forward dose map, by walking CSR rows.
//   For each voxel v, accumulate over its stored nonzeros:
//       dose[v] = sum_{k in [row_ptr[v], row_ptr[v+1])} values[k] * x[col_idx[k]]
//   This is the exact serial equivalent of the cuSPARSE SpMV the GPU uses, and
//   the operation main.cu reuses to convert a final fluence into a dose.
//   Complexity O(nnz). We accumulate in a double to keep the reference clean;
//   the GPU path accumulates in float (see THEORY.md section 5 on the tolerance).
// ---------------------------------------------------------------------------
void csr_spmv_cpu(const Problem& p, const std::vector<float>& x,
                  std::vector<float>& dose) {
    dose.assign(static_cast<std::size_t>(p.n_vox), 0.0f);
    for (int v = 0; v < p.n_vox; ++v) {
        float acc = 0.0f;                            // float to mirror the GPU SpMV
        const int beg = p.row_ptr[v], end = p.row_ptr[v + 1];
        for (int k = beg; k < end; ++k)
            acc += p.values[k] * x[p.col_idx[k]];    // D[v,j]*x[j]
        dose[v] = acc;
    }
}

// ---------------------------------------------------------------------------
// csr_spmvT_cpu: grad = D^T r, the transpose map that turns per-voxel residuals
//   into a per-BEAMLET gradient. D^T is (n_beam x n_vox); we never materialize
//   it -- we SCATTER each nonzero's contribution into the column it belongs to:
//       for each voxel v, for each nonzero k in row v:
//           grad[col_idx[k]] += values[k] * r[v]
//   On the CPU this scatter is a plain accumulate; on the GPU the same transpose
//   product is done by cuSPARSE with op = TRANSPOSE (no explicit atomics for us
//   to manage). Complexity O(nnz). Used inside optimize_cpu().
// ---------------------------------------------------------------------------
static void csr_spmvT_cpu(const Problem& p, const std::vector<float>& r,
                          std::vector<float>& grad) {
    grad.assign(static_cast<std::size_t>(p.n_beam), 0.0f);
    for (int v = 0; v < p.n_vox; ++v) {
        const float rv = r[v];
        const int beg = p.row_ptr[v], end = p.row_ptr[v + 1];
        for (int k = beg; k < end; ++k)
            grad[p.col_idx[k]] += p.values[k] * rv;  // scatter into beamlet column
    }
}

// ---------------------------------------------------------------------------
// compute_stats: dose vector -> deterministic PlanStats (DVH-style summary).
//   A pure function of the dose and the voxel specs, so feeding it the CPU dose
//   and the GPU dose and comparing the two is a fair, order-independent check.
//   Objective is summed in double for a stable headline number.
// ---------------------------------------------------------------------------
PlanStats compute_stats(const Problem& p, const std::vector<float>& dose) {
    PlanStats s;
    double ptv_sum = 0.0, oar_sum = 0.0;
    int    ptv_n = 0, oar_n = 0;
    double ptv_min =  std::numeric_limits<double>::infinity();
    double ptv_max = -std::numeric_limits<double>::infinity();
    double oar_max = -std::numeric_limits<double>::infinity();

    for (int v = 0; v < p.n_vox; ++v) {
        const double d = dose[v];
        s.objective += voxel_penalty(p.voxels[v], dose[v]);   // shared HD-core math
        if (p.voxels[v].kind == STRUCT_PTV) {
            ptv_sum += d; ++ptv_n;
            if (d < ptv_min) ptv_min = d;
            if (d > ptv_max) ptv_max = d;
        } else if (p.voxels[v].kind == STRUCT_OAR) {
            oar_sum += d; ++oar_n;
            if (d > oar_max) oar_max = d;
        }
    }
    s.ptv_mean = ptv_n ? ptv_sum / ptv_n : 0.0;
    s.ptv_min  = ptv_n ? ptv_min : 0.0;
    s.ptv_max  = ptv_n ? ptv_max : 0.0;
    s.oar_mean = oar_n ? oar_sum / oar_n : 0.0;
    s.oar_max  = oar_n ? oar_max : 0.0;
    // Homogeneity index: spread of PTV dose normalized by its mean (0 = uniform).
    s.homogeneity = (s.ptv_mean > 0.0) ? (s.ptv_max - s.ptv_min) / s.ptv_mean : 0.0;
    return s;
}

// ---------------------------------------------------------------------------
// optimize_cpu: projected gradient descent on the fluence -- the reference.
//
//   x <- 0                                (start with no beam -> zero dose)
//   repeat `iters` times:
//       d  = D x                          (forward SpMV: fluence -> dose)
//       r  = per-voxel residuals(d)       (shared fmo.h math)
//       g  = D^T r                        (transpose SpMV: dose grad -> beamlet grad)
//       x  = max(0, x - step * g)         (gradient step + non-negativity project)
//
//   Each iteration's cost is dominated by the two SpMVs (O(nnz)); the vector
//   math is O(n_vox)+O(n_beam). The GPU twin (optimize_gpu) runs this identical
//   loop with cuSPARSE doing the two SpMVs. We start x at zero so the trajectory
//   is fully determined by the data -> a reproducible result to verify against.
// ---------------------------------------------------------------------------
void optimize_cpu(const Problem& p, std::vector<float>& x_out) {
    std::vector<float> x(static_cast<std::size_t>(p.n_beam), 0.0f);   // fluence, x>=0
    std::vector<float> dose, resid, grad;

    for (int it = 0; it < p.iters; ++it) {
        // 1. Forward: current fluence -> dose in every voxel.
        csr_spmv_cpu(p, x, dose);

        // 2. Per-voxel residual r_v = dF/dd_v (drives the gradient). Uses the
        //    exact same voxel_residual() the GPU kernel calls.
        resid.assign(static_cast<std::size_t>(p.n_vox), 0.0f);
        for (int v = 0; v < p.n_vox; ++v)
            resid[v] = voxel_residual(p.voxels[v], dose[v]);

        // 3. Transpose: fold voxel residuals back onto beamlet gradients.
        csr_spmvT_cpu(p, resid, grad);

        // 4. Projected gradient step: descend, then clamp fluence to >= 0.
        for (int j = 0; j < p.n_beam; ++j)
            x[j] = project_nonneg(x[j] - p.step * grad[j]);
    }
    x_out = std::move(x);
}
