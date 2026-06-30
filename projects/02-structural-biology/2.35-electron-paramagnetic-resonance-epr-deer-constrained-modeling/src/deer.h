// ===========================================================================
// src/deer.h  --  Shared (host + device) DEER physics: spin-label rotamer
//                 convolution -> per-member P(r) histogram, and the chi^2 +
//                 relative-entropy objective used by ensemble reweighting.
// ---------------------------------------------------------------------------
// Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
//
// WHY THIS HEADER EXISTS (the most important idiom in the repo, PATTERNS.md §2)
//   The per-element physics lives here ONCE, as `__host__ __device__` inline
//   functions, so the CPU reference (reference_cpu.cpp, compiled by cl.exe) and
//   the GPU kernel (kernels.cu, compiled by nvcc) run BYTE-FOR-BYTE identical
//   math. That makes verification exact instead of approximate. DEER_HD expands
//   to `__host__ __device__` under nvcc and to nothing under the host compiler.
//   Keep CUDA-only constructs (no __global__, no <<<>>>) OUT of this header so
//   the plain C++ compiler can include it.
//
// THE SCIENCE IN ONE PARAGRAPH (full story in ../THEORY.md)
//   DEER (Double Electron-Electron Resonance, a.k.a. PELDOR) is a pulsed-EPR
//   experiment that measures the DISTRIBUTION of distances between two unpaired
//   electron spins. You attach a nitroxide spin label (commonly MTSSL) to two
//   engineered cysteines on a protein; the dipolar coupling between the two
//   electrons depends on 1/r^3, and inverting the time-domain signal yields a
//   probability distribution P(r) over the spin-spin distance r (typically
//   1.5-8 nm). Because a flexible protein samples MANY conformations, and each
//   spin label itself is a floppy tether sampling MANY ROTAMERS, the measured
//   P(r) is broad. To compare a structural model (an MD ensemble of M frames)
//   against the experiment, you BACK-CALCULATE P(r) from the model: for each
//   frame, place a cloud of label rotamers on each of the two sites, take every
//   pairwise spin-spin distance, and histogram it. The model's P(r) is then a
//   population-weighted sum over the M frames. Reweighting (BioEn/EROS) adjusts
//   those populations to best match the experimental P(r) without overfitting.
//
// WHAT THIS FILE PROVIDES
//   * Grid / units: the r-axis is a fixed histogram of NBINS bins (deer.cuh /
//     reference share the same constants, defined in deer_params.h).
//   * deer_member_histogram(): rotamer convolution for ONE frame -> its P(r).
//     This is the heavy, embarrassingly-parallel kernel body (one frame per
//     thread). O(R^2) spin-pairs per frame, where R = rotamers per site.
//   * chi2_to_target(): goodness-of-fit between a mixed P(r) and the target.
//   * kl_to_prior(): relative entropy (Kullback-Leibler) of weights vs. the
//     uniform prior -- the regularizer that keeps reweighting honest.
//
// READ THIS AFTER: deer_params.h (the constants).  READ BEFORE: kernels.cu,
// reference_cpu.cpp (both call these functions).
// ===========================================================================
#pragma once

#include "deer_params.h"   // NBINS, R_MIN_NM, R_BIN_NM, ROTAMERS_PER_SITE, ...

#include <cmath>           // std::sqrt, std::log, std::exp (host); device intrinsics map automatically

// DEER_HD: the decorator that makes a function callable from BOTH host and GPU.
//   Under nvcc (__CUDACC__ defined) it becomes `__host__ __device__`; under the
//   plain host compiler it vanishes, so reference_cpu.cpp still compiles.
#ifdef __CUDACC__
#define DEER_HD __host__ __device__
#else
#define DEER_HD
#endif

// ---------------------------------------------------------------------------
// Spin3: a minimal 3-D point in nanometres. We use double everywhere in the
// physics so the CPU and GPU agree to ~1e-13 (DEER distances are a few nm; the
// extra precision is essentially free at this problem size and buys exactness).
// ---------------------------------------------------------------------------
struct Spin3 {
    double x, y, z;   // nanometres
};

// ---------------------------------------------------------------------------
// r_bin_center(b): the distance (nm) at the centre of histogram bin b.
//   The r-axis is uniform: bin b covers [R_MIN_NM + b*R_BIN_NM, ...+R_BIN_NM),
//   so its centre is R_MIN_NM + (b + 0.5)*R_BIN_NM. Used to evaluate the
//   experimental target and to report peaks; shared so both sides agree.
// ---------------------------------------------------------------------------
DEER_HD inline double r_bin_center(int b) {
    return R_MIN_NM + (static_cast<double>(b) + 0.5) * R_BIN_NM;
}

// ---------------------------------------------------------------------------
// distance_to_bin(r): map a spin-spin distance r (nm) to its histogram bin
//   index, or -1 if it falls outside [R_MIN_NM, R_MIN_NM + NBINS*R_BIN_NM).
//   Out-of-range pairs are simply dropped (DEER itself has a limited distance
//   window set by the dipolar evolution time), so this is physically faithful.
//   Integer truncation here is deterministic and identical on host and device.
// ---------------------------------------------------------------------------
DEER_HD inline int distance_to_bin(double r) {
    if (r < R_MIN_NM) return -1;
    const int b = static_cast<int>((r - R_MIN_NM) / R_BIN_NM);
    return (b >= 0 && b < NBINS) ? b : -1;
}

// ---------------------------------------------------------------------------
// deer_member_histogram  --  rotamer convolution for ONE ensemble frame.
// ---------------------------------------------------------------------------
// INPUT  (per frame, laid out in the caller's arrays):
//   siteA[ROTAMERS_PER_SITE], siteB[ROTAMERS_PER_SITE]
//       the two clouds of spin-label rotamer endpoints (the nitroxide N-O
//       midpoint, where the unpaired electron effectively sits) for this frame.
//       A real MTSSL rotamer library has ~200 states with Boltzmann weights;
//       here every rotamer carries equal weight 1/R for a clean teaching model.
// OUTPUT:
//   hist[NBINS]  -- the frame's normalized distance distribution P_m(r). Each of
//                   the R*R spin pairs contributes one count to its bin; we then
//                   divide by the number of IN-RANGE pairs so the histogram sums
//                   to 1 (a probability distribution). If no pair is in range the
//                   histogram is left all-zero (the frame contributes nothing).
//
// COMPLEXITY: O(R^2) distance evaluations per frame (R = ROTAMERS_PER_SITE).
//   This is the dominant cost and the reason the GPU helps: the M frames are
//   independent, so we give each frame its own thread and run all M convolutions
//   at once. See ../THEORY.md "GPU mapping".
//
// THREAD-TO-DATA MAPPING (when called from the GPU): the kernel assigns frame
//   m = blockIdx.x*blockDim.x + threadIdx.x to one thread; that thread calls
//   this function with pointers into the m-th slice of the rotamer arrays and
//   the m-th row of the histogram matrix. No two threads touch the same hist
//   row, so there are NO atomics and NO races -- pure data parallelism.
// ---------------------------------------------------------------------------
DEER_HD inline void deer_member_histogram(const Spin3* siteA,
                                          const Spin3* siteB,
                                          double* hist) {
    // Zero this frame's bins first (the caller's row may be reused).
    for (int b = 0; b < NBINS; ++b) hist[b] = 0.0;

    // Convolve the two rotamer clouds: every A-rotamer against every B-rotamer.
    // Each accepted pair drops one unit of count into its distance bin. We count
    // accepted pairs to normalize afterwards (out-of-window pairs are skipped).
    long in_range = 0;
    for (int i = 0; i < ROTAMERS_PER_SITE; ++i) {
        const Spin3 a = siteA[i];
        for (int j = 0; j < ROTAMERS_PER_SITE; ++j) {
            const Spin3 b = siteB[j];
            const double dx = a.x - b.x;     // component differences (nm)
            const double dy = a.y - b.y;
            const double dz = a.z - b.z;
            const double r  = std::sqrt(dx * dx + dy * dy + dz * dz);  // Euclidean spin-spin distance
            const int bin = distance_to_bin(r);
            if (bin >= 0) {
                hist[bin] += 1.0;            // accumulate count in this frame's own row (no race)
                ++in_range;
            }
        }
    }

    // Normalize to a probability distribution (sum -> 1) when any pair landed in
    // range. Dividing by an integer pair-count is exact and identical on both
    // sides, so the CPU and GPU histograms agree bit-for-bit.
    if (in_range > 0) {
        const double inv = 1.0 / static_cast<double>(in_range);
        for (int b = 0; b < NBINS; ++b) hist[b] *= inv;
    }
}

// ---------------------------------------------------------------------------
// chi2_to_target  --  goodness-of-fit of a model distribution to the target.
//   Both `model` and `target` are length-NBINS probability vectors over the
//   same r-axis. We use an unweighted sum of squared residuals:
//       chi^2 = sum_b ( model[b] - target[b] )^2
//   (A real DEER fit weights each bin by its experimental uncertainty; we keep
//   uniform weights for clarity and note the extension in THEORY.) Lower is a
//   better fit. Shared so the CPU and GPU report the identical number.
// ---------------------------------------------------------------------------
DEER_HD inline double chi2_to_target(const double* model, const double* target) {
    double s = 0.0;
    for (int b = 0; b < NBINS; ++b) {
        const double d = model[b] - target[b];
        s += d * d;
    }
    return s;
}

// ---------------------------------------------------------------------------
// kl_to_prior  --  relative entropy (Kullback-Leibler divergence) of the
//   reweighted populations w[] from the uniform reference prior w0 = 1/M.
//       S_KL = sum_m w[m] * ln( w[m] / w0 )   with  0*ln0 := 0.
//   This is the BioEn/EROS regularizer: it measures how far reweighting has
//   pulled the populations from "trust the simulation equally". The objective
//   minimized during reweighting is  L(w) = chi^2(w) + theta * S_KL(w), so a
//   large theta keeps the weights near uniform (conservative) and a small theta
//   lets chi^2 dominate (aggressive fit). Shared by CPU + GPU. M = number of
//   ensemble members; w is assumed already normalized (sum_m w[m] = 1).
// ---------------------------------------------------------------------------
DEER_HD inline double kl_to_prior(const double* w, int M) {
    const double w0 = 1.0 / static_cast<double>(M);   // uniform prior population
    double s = 0.0;
    for (int m = 0; m < M; ++m) {
        if (w[m] > 0.0) {
            s += w[m] * std::log(w[m] / w0);
        }
        // w[m] == 0 contributes 0 (limit of x ln x as x -> 0), so we skip it and
        // avoid log(0). Negative weights never occur (softmax guarantees w>0).
    }
    return s;
}
