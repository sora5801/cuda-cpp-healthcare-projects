// ===========================================================================
// src/fmo.h  --  Shared (host + device) fluence-map-optimization core
// ---------------------------------------------------------------------------
// Project 5.2 : Radiotherapy Treatment-Plan Optimization
//
// WHAT THIS PROJECT COMPUTES
//   Fluence-Map Optimization (FMO), the numerical heart of IMRT/VMAT inverse
//   planning. A radiotherapy plan aims a set of "beamlets" (tiny pencil beams,
//   each with an adjustable intensity/fluence weight x_j >= 0) at a patient so
//   that the tumor (the PTV -- planning target volume) receives the prescribed
//   dose while healthy organs (OARs -- organs at risk) stay below tolerance.
//
//   The physics is LINEAR: the dose deposited in voxel v is the weighted sum of
//   the contributions of every beamlet that passes through it,
//
//        d_v = sum_j  D[v, j] * x_j          i.e.   d = D x
//
//   where D (the "dose-influence" / "dij" matrix) is precomputed by a dose
//   engine. D is huge (typically ~10^6 voxels x ~10^4 beamlets) but SPARSE
//   (a beamlet only irradiates a narrow corridor of voxels), so it is stored in
//   CSR format and the forward map d = D x is a sparse matrix-vector product
//   (SpMV) -- the single dominant cost of every optimizer iteration.
//
//   We minimize a weighted-quadratic objective over the fluence x >= 0:
//
//        F(x) = sum_v  w_v * pen_v(d_v)
//
//   with a per-structure one/two-sided quadratic penalty pen_v (below). Its
//   gradient w.r.t. the fluence is, by the chain rule,
//
//        dF/dx_j = sum_v  D[v, j] * (w_v * pen_v'(d_v))   i.e.  g = D^T r
//
//   another SpMV, this time with the TRANSPOSE of D. So each iteration is two
//   SpMVs (D x and D^T r) plus cheap per-voxel / per-beamlet vector math. We
//   minimize by PROJECTED gradient descent: step downhill, then clamp x to >= 0.
//
//   THE HD-CORE IDIOM (PATTERNS.md section 2): the per-voxel penalty value and
//   its derivative are the "one true formula" shared by the CPU reference and
//   the GPU kernels. Putting them here as `__host__ __device__` inline functions
//   guarantees both paths compute BYTE-IDENTICAL scalar math, so the only source
//   of CPU-vs-GPU difference is floating-point summation order in the SpMVs
//   (which we bound with a documented tolerance -- see THEORY.md section 5).
//   FMO_HD expands to __host__ __device__ under nvcc, to nothing under the host
//   compiler. Keep CUDA-only types (no __global__) out of this header so the
//   plain C++ compiler can include it too.
//
// READ THIS AFTER: nothing -- start here, then reference_cpu.h (the CSR problem
//   + the CPU optimizer), then kernels.cuh/.cu (the GPU optimizer via cuSPARSE).
// ===========================================================================
#pragma once

#ifdef __CUDACC__
#define FMO_HD __host__ __device__
#else
#define FMO_HD
#endif

// ---------------------------------------------------------------------------
// Structure classification of a voxel. FMO treats the three structure kinds
// with different one/two-sided quadratic penalties (this is the essence of an
// inverse-planning objective: it encodes the clinical trade-off numerically).
//   PTV  : the tumor. TWO-sided penalty -- punish both under- and over-dose,
//          driving d_v toward the prescription d_rx (coverage + homogeneity).
//   OAR  : an organ at risk. ONE-sided penalty -- punish only dose ABOVE the
//          tolerance d_max; dose below tolerance is free (we want it low but
//          any value under d_max is acceptable).
//   BODY : all other tissue. ONE-sided penalty above 0 with a small weight --
//          a gentle pressure to keep stray dose down and the problem bounded.
// ---------------------------------------------------------------------------
enum StructKind { STRUCT_PTV = 0, STRUCT_OAR = 1, STRUCT_BODY = 2 };

// Per-voxel objective parameters. One of these per voxel; small and trivially
// copyable so it lives happily in a device array. `target` is the reference
// dose (prescription for PTV, tolerance for OAR, 0 for BODY); `weight` is the
// clinical importance w_v; `kind` selects the penalty shape below.
struct VoxelSpec {
    float  target;   // reference dose in Gray (Gy): d_rx (PTV) or d_max (OAR/BODY)
    float  weight;   // penalty weight w_v (>= 0): larger => harder constraint
    int    kind;     // one of StructKind
};

// ---------------------------------------------------------------------------
// voxel_penalty: the per-voxel objective term  w_v * pen_v(d_v).
//   Returns the (non-negative) contribution of this voxel to F(x). Summed over
//   all voxels this IS the objective. The penalties are quadratic in the dose
//   DEVIATION so the objective is smooth and convex -- gradient descent finds
//   the global optimum for this (positive-semidefinite quadratic) problem.
//
//   PTV : (d - target)^2                        -- two-sided (any deviation hurts)
//   OAR : max(0, d - target)^2                  -- one-sided (only overdose hurts)
//   BODY: max(0, d - target)^2  (target = 0)    -- one-sided above zero
//
//   `d` is this voxel's current dose d_v (Gy). Complexity O(1); called once per
//   voxel per iteration by both the CPU loop and the GPU objective kernel.
// ---------------------------------------------------------------------------
FMO_HD inline float voxel_penalty(const VoxelSpec& s, float d) {
    const float dev = d - s.target;                 // signed dose deviation (Gy)
    if (s.kind == STRUCT_PTV) {
        return s.weight * dev * dev;                // two-sided quadratic
    }
    // OAR and BODY: only the positive part (overdose) is penalized.
    const float over = dev > 0.0f ? dev : 0.0f;     // max(0, d - target)
    return s.weight * over * over;
}

// ---------------------------------------------------------------------------
// voxel_residual: the per-voxel gradient factor  r_v = w_v * pen_v'(d_v).
//   This is dF/dd_v, i.e. how the objective changes as this voxel's dose
//   changes. The full fluence gradient is g = D^T r (chain rule): each beamlet's
//   gradient is the D-weighted sum of the residuals of the voxels it irradiates.
//
//   d/dd of (d - t)^2 is 2 (d - t); the one-sided penalties differentiate to
//   2 max(0, d - t). So:
//     PTV : 2 w (d - target)                      (can be negative -> push dose UP)
//     OAR : 2 w max(0, d - target)                (>= 0 -> only ever push dose DOWN)
//     BODY: 2 w max(0, d - target)  (target = 0)
//
//   Returned value has units Gy^-1 * (objective units); it feeds the transpose
//   SpMV. Complexity O(1) per voxel. Shared verbatim by CPU + GPU (the whole
//   point of this header) so gradients match to round-off.
// ---------------------------------------------------------------------------
FMO_HD inline float voxel_residual(const VoxelSpec& s, float d) {
    const float dev = d - s.target;
    if (s.kind == STRUCT_PTV) {
        return 2.0f * s.weight * dev;              // two-sided: sign carries direction
    }
    const float over = dev > 0.0f ? dev : 0.0f;
    return 2.0f * s.weight * over;                 // one-sided: >= 0
}

// ---------------------------------------------------------------------------
// project_nonneg: the PROJECTION step of projected gradient descent.
//   Fluence/intensity is a physical quantity that cannot be negative (a beamlet
//   cannot emit "anti-radiation"), so x_j is constrained to x_j >= 0. After each
//   gradient step we project back onto that feasible set, which for a simple box
//   constraint is just a clamp. This one-liner is shared so CPU and GPU clamp
//   identically. Returns max(0, xj).
// ---------------------------------------------------------------------------
FMO_HD inline float project_nonneg(float xj) {
    return xj > 0.0f ? xj : 0.0f;
}
