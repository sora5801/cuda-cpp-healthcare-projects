// ===========================================================================
// src/gamma_core.h  --  The ONE TRUE per-pair gamma math (CPU + GPU parity)
// ---------------------------------------------------------------------------
// Project 5.9 -- Gamma-Index Dose Comparison
//
// WHY THIS FILE EXISTS  (PATTERNS.md §2 -- the shared __host__ __device__ core)
//   The gamma index compares two dose distributions by, at every REFERENCE
//   point, searching nearby EVALUATED points for the smallest combined
//   dose-difference / distance disagreement. The CPU reference
//   (reference_cpu.cpp) and the GPU kernel (kernels.cu) must run *byte-for-byte
//   identical arithmetic* so that "GPU == CPU" is an EXACT check, not a fuzzy
//   one. The trick: put the per-pair physics in ONE header of
//   `__host__ __device__` inline functions, then have BOTH sides call it.
//
//   - reference_cpu.cpp is compiled by the plain host C++ compiler (cl.exe/g++).
//   - kernels.cu is compiled by nvcc.
//   Only nvcc defines the macro __CUDACC__, so we guard the CUDA decorators:
//   on the host compiler they simply vanish and this becomes ordinary C++.
//
//   HARD RULE for this header: NO CUDA-only types, NO __global__, NO <cuda*>
//   includes -- the host compiler must be able to include it verbatim.
//
// READ THIS BEFORE: reference_cpu.cpp and kernels.cu (both include this file).
// See ../THEORY.md §2 (the math) and §6 (why the shared core makes verification
// exact).
// ===========================================================================
#pragma once

// GAMMA_HD expands to `__host__ __device__` under nvcc (so the same inline
// function compiles for BOTH the CPU and every GPU thread), and to nothing for
// the host compiler (which has never heard of those keywords). This single
// macro is what guarantees CPU/GPU numerical parity -- see PATTERNS.md §2.
#ifdef __CUDACC__
#define GAMMA_HD __host__ __device__
#else
#define GAMMA_HD
#endif

// ---------------------------------------------------------------------------
// GammaParams -- the criteria that define "acceptable agreement".
//   Clinical convention is written as "P%/D mm" (e.g. 3%/3 mm): a point passes
//   if the evaluated dose is within P% of the dose difference criterion OR
//   there exists an evaluated point within D mm that has the matching dose (the
//   gamma index blends these two tolerances into one smooth number).
//
//   We precompute the INVERSE-SQUARED normalizers once on the host and pass
//   them in, so the hot inner loop multiplies instead of dividing (a division
//   is ~4-8x costlier than a multiply on the GPU, and this runs millions of
//   times). Storing them as `double` keeps the CPU and GPU reductions identical.
// ---------------------------------------------------------------------------
struct GammaParams {
    double inv_dd_crit_sq;   // 1 / (dose-difference criterion)^2   [1/dose^2]
    double inv_dta_crit_sq;  // 1 / (distance-to-agreement criterion)^2 [1/mm^2]
};

// ---------------------------------------------------------------------------
// gamma_sq_term -- the squared generalized distance in (dose, space) between a
//   single reference point and a single evaluated point.
//
//   Formal definition (THEORY §2):
//       Gamma^2(r_ref, r_eval) = (dose_eval - dose_ref)^2 / dd_crit^2
//                              + dist_mm^2               / dta_crit^2
//   and the gamma index at the reference point is
//       gamma(r_ref) = min over evaluated points of sqrt(Gamma^2).
//
//   We return the SQUARED value (no sqrt) because:
//     (a) sqrt is monotonic, so argmin of Gamma^2 == argmin of Gamma -- we can
//         do the whole min-search in squared space and sqrt only ONCE at the
//         end; and
//     (b) skipping millions of sqrts is a real speedup.
//
//   Parameters (units matter -- CLAUDE.md §6.1.3):
//     dose_eval : evaluated dose at the candidate point            [dose units]
//     dose_ref  : reference dose at the fixed reference point       [dose units]
//     dist_mm_sq: SQUARED physical distance between the two points  [mm^2]
//     p         : precomputed inverse-squared criteria (see above)
//
//   Returns: the dimensionless squared gamma term for this ONE pair.
//   Complexity: O(1) -- a handful of FLOPs; called once per (ref, eval) pair.
// ---------------------------------------------------------------------------
GAMMA_HD inline double gamma_sq_term(double dose_eval, double dose_ref,
                                     double dist_mm_sq, const GammaParams& p) {
    const double dose_diff = dose_eval - dose_ref;           // signed [dose]
    const double dose_term = dose_diff * dose_diff * p.inv_dd_crit_sq; // (dd/dd_crit)^2
    const double dist_term = dist_mm_sq * p.inv_dta_crit_sq;           // (r/dta_crit)^2
    return dose_term + dist_term;   // the two tolerances added in quadrature
}
