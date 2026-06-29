// ===========================================================================
// src/kernels.cuh  --  GPU induced-dipole interface (the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.21 : Polarizable / AMOEBA Force Field MD
//
// THE BIG IDEA (PATTERNS.md "ensemble" pattern: one thread per CG solve)
//   The AMOEBA induced-dipole problem is a self-consistent linear solve
//   A mu = b that must be redone at every MD step. We model an ENSEMBLE of such
//   solves -- many independent molecular configurations -- and give each one its
//   OWN GPU thread. The thread runs the entire matrix-free CONJUGATE-GRADIENT
//   loop (solve_induced_dipoles in amoeba.h) in its registers/local memory and
//   writes one PerSystemResult. There is no inter-thread communication: the
//   members are independent, so this is "embarrassingly parallel over members"
//   -- exactly how Monte-Carlo / configuration-sweep polarization studies scale.
//
//   Because the CG routine is shared host+device (amoeba.h, AMOEBA_HD), the GPU
//   result matches the CPU reference essentially to round-off (see THEORY.md
//   "How we verify correctness"). kernels.cu defines the kernel + host wrapper.
//
//   WHY THREAD-PER-SYSTEM (and not block-per-system)? Each system here is tiny
//   (n <= 32 atoms -> <= 96 unknowns), so a single thread solves it in a few
//   microseconds and the whole ensemble fills the GPU through DATA parallelism
//   across members. A block-per-system design (cooperating threads on one CG
//   solve, parallelizing the O(n^2) matvec and using a block reduction for the
//   dot products) is the right call when n is large; THEORY.md discusses that
//   trade-off and leaves it as an exercise.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, amoeba.h, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, AtomSystem, PerSystemResult (pure C++)

// ---- Device kernel -------------------------------------------------------
// dipole_ensemble_kernel: thread `idx` owns ensemble member idx.
//   It reads that member's AtomSystem, runs the full CG solve, and writes one
//   PerSystemResult. No shared memory, no atomics -- pure per-thread work.
//     systems  : device array of M AtomSystem (the whole ensemble, contiguous)
//     M        : number of members (guards the ragged last block)
//     tol      : CG relative-residual stop
//     max_iter : CG iteration cap
//     out      : device array of M PerSystemResult (one per member)
__global__ void dipole_ensemble_kernel(const AtomSystem* __restrict__ systems,
                                       int M, double tol, int max_iter,
                                       PerSystemResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// solve_ensemble_gpu: copy the ensemble H2D, launch one thread per member, copy
//   the results back, and report the measured KERNEL time (CUDA events) via
//   *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//     c         : the ensemble (systems + solver knobs)            [in]
//     results   : host output, resized to M                        [out]
//     kernel_ms : out-param, kernel milliseconds (not the copies)  [out]
void solve_ensemble_gpu(const EnsembleConfig& c,
                        std::vector<PerSystemResult>& results,
                        float* kernel_ms);
