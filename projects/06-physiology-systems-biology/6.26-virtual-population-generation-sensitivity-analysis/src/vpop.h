// ===========================================================================
// src/vpop.h  --  Shared (host + device) virtual-population model + Saltelli/Sobol
// ---------------------------------------------------------------------------
// Project 6.26 : Virtual Population Generation & Sensitivity Analysis
//
// WHAT THIS PROJECT COMPUTES
//   Two linked ideas from quantitative systems pharmacology (QSP):
//
//   (1) VIRTUAL POPULATION GENERATION. A "virtual patient" is a draw of the
//       physiological parameters that drive a pharmacokinetic (PK) model:
//       absorption rate ka, clearance CL, distribution volume V, and oral
//       bioavailability F. We sample each parameter from a plausible range,
//       run a PK model for every patient, and summarize exposure -- here the
//       area-under-the-curve AUC (total drug exposure, mg*h/L).
//
//   (2) GLOBAL SENSITIVITY ANALYSIS (Sobol variance decomposition). "Which
//       parameter, if we could pin it down, would shrink the spread of AUC the
//       most?" Sobol answers this by decomposing Var(AUC) into contributions
//       from each parameter and their interactions. We estimate the FIRST-ORDER
//       index S_j (fraction of variance from parameter j alone) and the
//       TOTAL-ORDER index ST_j (j alone + all its interactions) with the
//       Saltelli sampling estimator. That needs N*(k+2) independent model
//       evaluations -- which is exactly the embarrassingly-parallel workload a
//       GPU eats for breakfast (one thread per model evaluation).
//
// THE MODEL (a deliberately simple, analytically-checkable PK model)
//   One-compartment model with first-order oral absorption. The plasma amount
//   solves  A(t) = (F*Dose*ka)/(ka - kel) * (e^{-kel t} - e^{-ka t}),  with the
//   elimination rate kel = CL/V and concentration C(t) = A(t)/V. The total
//   exposure has a CLOSED FORM:
//
//       AUC = integral_0^inf C(t) dt = F * Dose / CL.
//
//   That closed form is a GIFT for teaching: AUC depends ONLY on F and CL, and
//   NOT on ka or V. So a correct Sobol analysis must attribute ~all of the
//   variance to F and CL and ~zero to ka and V. The demo checks exactly this,
//   turning an abstract algorithm into a result you can sanity-check by hand.
//   (We still evaluate AUC by NUMERICALLY integrating C(t) with the trapezoid
//    rule, not the closed form -- so the code exercises a real forward model and
//    the closed form stays an INDEPENDENT check, per PATTERNS.md section 4.)
//
// WHY ONE SHARED HEADER (the __host__ __device__ idiom, PATTERNS.md section 2)
//   The per-sample math -- quasi-random sampling, parameter scaling, the PK
//   forward solve -- lives here as `VPOP_HD inline` functions. reference_cpu.cpp
//   (host compiler) and kernels.cu (nvcc) both include this header, so the CPU
//   reference and the GPU kernel run BYTE-IDENTICAL arithmetic. Verification is
//   then exact-to-round-off rather than approximate. Keep CUDA-only constructs
//   (no __global__, no <cuda_runtime.h>) OUT of this header so the host compiler
//   can include it cleanly.
//
// READ THIS AFTER: nothing (start here), then kernels.cuh, reference_cpu.h, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

// The __host__ __device__ decorators only exist under nvcc (__CUDACC__). When
// the plain host compiler builds reference_cpu.cpp, VPOP_HD expands to nothing,
// so the exact same source compiles for the CPU. This is the whole trick that
// gives us bit-for-bit CPU/GPU parity.
#ifdef __CUDACC__
#define VPOP_HD __host__ __device__
#else
#define VPOP_HD
#endif

// The number of uncertain parameters k in the sensitivity study. Fixed at 4 so
// we can keep the per-sample state in registers (a tiny fixed-size array) and so
// the Saltelli matrices have a known column count. Order below is FIXED and is
// the order the Sobol indices are reported in.
//   0: ka  (first-order absorption rate, 1/h)
//   1: CL  (clearance, L/h)
//   2: V   (central distribution volume, L)
//   3: F   (oral bioavailability, unitless fraction in (0,1])
#define VPOP_K 4

// -----------------------------------------------------------------------------
// Population configuration read from the data file. `lo[j]`/`hi[j]` are the
// lower/upper bounds of parameter j's uniform prior (the plausible physiological
// range for the virtual population). N is the Saltelli BASE sample size; the
// total number of model evaluations is N*(k+2) (see saltelli layout below).
// -----------------------------------------------------------------------------
struct VpopParams {
    double dose;             // administered oral dose (mg), a fixed constant
    double lo[VPOP_K];       // lower bound of each parameter's uniform range
    double hi[VPOP_K];       // upper bound of each parameter's uniform range
    double t_end;            // integration horizon (h); must be >> 1/kel so AUC converges
    int    steps;            // trapezoid steps over [0, t_end]  (grid = steps+1 points)
    int    N;                // Saltelli base sample size (rows of matrices A and B)
    uint64_t seed;           // (reserved) base seed; the Halton sequence is deterministic
};

// -----------------------------------------------------------------------------
// Saltelli evaluation layout. To estimate Sobol indices we need, per base row i:
//   f(A_i)                         -- model on sample matrix A
//   f(B_i)                         -- model on independent sample matrix B
//   f(AB_i^{(j)})  for j=0..k-1    -- A with ONLY column j swapped in from B
// So there are (k+2) "matrices" and we lay them out as a flat block index:
//   block 0        = A
//   block 1        = B
//   block 2+j      = AB^{(j)}   (A with column j taken from B)
// Global evaluation index g in [0, N*(k+2)) maps to (block, row) by:
//   block = g / N,   row = g % N.
// One GPU thread owns one g. This helper returns the number of blocks.
// -----------------------------------------------------------------------------
VPOP_HD inline int vpop_num_blocks() { return VPOP_K + 2; }
VPOP_HD inline long vpop_num_evals(int N) { return (long)N * (VPOP_K + 2); }

// -----------------------------------------------------------------------------
// van der Corport radical inverse in a given prime base -- the building block of
// the Halton low-discrepancy (quasi-random) sequence. It mirrors the integer
// `idx` around the "decimal" point in `base`: idx = d0 + d1*base + d2*base^2...
// maps to 0.d0 d1 d2... in base `base`. Successive indices spread points far
// more EVENLY than pseudo-random draws (low discrepancy), which makes the Sobol
// variance estimator converge much faster for the same N. cuRAND offers Sobol/
// Halton generators (see catalog); we hand-roll Halton here so the CPU and GPU
// draw byte-identical points and the header stays CUDA-free.
// -----------------------------------------------------------------------------
VPOP_HD inline double vpop_radical_inverse(unsigned int idx, unsigned int base) {
    double inv_base = 1.0 / (double)base;   // running 1/base, 1/base^2, ...
    double inv_bn   = inv_base;
    double result   = 0.0;
    while (idx > 0) {
        unsigned int digit = idx % base;    // next base-`base` digit of idx
        result += (double)digit * inv_bn;   // place it after the point
        idx /= base;                        // shift right one digit
        inv_bn *= inv_base;                 // move to the next place value
    }
    return result;                          // in [0, 1)
}

// The first 8 primes give us up to 8 independent, well-spread Halton dimensions.
// The Saltelli scheme uses 2*k = 8 dimensions total (k for matrix A, k for B),
// so 8 primes is exactly enough for k=4. Prime bases keep the dimensions
// mutually low-correlated (Halton's classic requirement).
VPOP_HD inline unsigned int vpop_prime(int d) {
    const unsigned int primes[8] = {2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u};
    return primes[d & 7];   // d in [0,8); mask guards accidental overrun
}

// -----------------------------------------------------------------------------
// Draw a single quasi-random coordinate in [0,1) for base-sample row `i`,
// dimension `dim`. We offset the Halton index by +1 so we never use index 0
// (whose radical inverse is exactly 0 in every base, which would pin every
// dimension to its lower bound for the first patient -- a classic Halton
// gotcha). `dim` in [0, 2k) selects the prime base / dimension.
// -----------------------------------------------------------------------------
VPOP_HD inline double vpop_unit(int i, int dim) {
    return vpop_radical_inverse((unsigned int)(i + 1), vpop_prime(dim));
}

// -----------------------------------------------------------------------------
// Build the k-dimensional parameter vector for global evaluation (block, row).
//   * The A-sample uses Halton dimensions 0..k-1.
//   * The B-sample uses Halton dimensions k..2k-1  (independent stream).
//   * Block b encodes the Saltelli rule:
//       b == 0            -> pure A row
//       b == 1            -> pure B row
//       b in [2, 2+k)     -> A row, but parameter (b-2) taken from B
//   Each unit coordinate u in [0,1) is mapped affinely into the physiological
//   range [lo_j, hi_j]. `out[j]` receives parameter j in physical units.
// -----------------------------------------------------------------------------
VPOP_HD inline void vpop_build_params(const VpopParams& P, int block, int row,
                                      double out[VPOP_K]) {
    for (int j = 0; j < VPOP_K; ++j) {
        // Decide, per parameter j, whether this block reads column j from A or B.
        bool take_from_B;
        if (block == 0)      take_from_B = false;              // matrix A: all from A
        else if (block == 1) take_from_B = true;               // matrix B: all from B
        else                 take_from_B = (j == block - 2);   // AB^{(j)}: only col j from B

        // A's dimensions are 0..k-1; B's are the next k dimensions.
        const int dim = take_from_B ? (VPOP_K + j) : j;
        const double u = vpop_unit(row, dim);                  // quasi-random in [0,1)
        out[j] = P.lo[j] + u * (P.hi[j] - P.lo[j]);            // scale into [lo,hi]
    }
}

// -----------------------------------------------------------------------------
// THE FORWARD MODEL: one-compartment oral PK, numerically integrated exposure.
// Given a parameter vector p = {ka, CL, V, F}, compute AUC = integral C(t) dt on
// [0, t_end] with the composite trapezoid rule over `steps` intervals.
//
//   kel = CL / V                       (first-order elimination rate, 1/h)
//   C(t) = (F*Dose*ka)/(V*(ka-kel)) * (e^{-kel t} - e^{-ka t})
//
// The (ka - kel) denominator vanishes when absorption and elimination rates
// coincide (the "flip-flop" degenerate case); we guard it with the analytic
// limit C(t) = (F*Dose*ka/V) * t * e^{-ka t}. Because ka and V do NOT appear in
// the closed-form AUC = F*Dose/CL, a correct Sobol analysis attributes ~0
// variance to them -- the built-in teaching check (see main.cu / THEORY).
// -----------------------------------------------------------------------------
VPOP_HD inline double vpop_auc(const VpopParams& P, const double p[VPOP_K]) {
    const double ka  = p[0];
    const double CL  = p[1];
    const double V   = p[2];
    const double F   = p[3];
    const double kel = CL / V;                       // elimination rate constant

    const double dt = P.t_end / (double)P.steps;     // trapezoid step (h)
    const double amp = F * P.dose * ka / V;          // concentration prefactor
    const double gap = ka - kel;                     // absorption/elimination gap

    double auc = 0.0;
    double c_prev = 0.0;                             // C(0) = 0 (nothing absorbed yet)
    for (int s = 1; s <= P.steps; ++s) {
        const double t = s * dt;
        double c;
        if (fabs(gap) > 1e-9) {
            c = (amp / gap) * (exp(-kel * t) - exp(-ka * t));   // standard Bateman form
        } else {
            c = amp * t * exp(-ka * t);                          // flip-flop limit ka==kel
        }
        auc += 0.5 * (c_prev + c) * dt;              // trapezoid area of this slice
        c_prev = c;
    }
    return auc;                                      // mg*h/L, approaches F*Dose/CL
}

// -----------------------------------------------------------------------------
// Evaluate the model for a single GLOBAL Saltelli index g in [0, N*(k+2)).
// Decodes g -> (block, row), builds the parameter vector, returns AUC. This is
// the ONE function the CPU loop and the GPU kernel both call, guaranteeing
// identical results per evaluation. `g` is `long` because N*(k+2) can exceed
// 2^31 for large studies.
// -----------------------------------------------------------------------------
VPOP_HD inline double vpop_eval(const VpopParams& P, long g) {
    const int block = (int)(g / P.N);
    const int row   = (int)(g % P.N);
    double p[VPOP_K];
    vpop_build_params(P, block, row, p);
    return vpop_auc(P, p);
}
