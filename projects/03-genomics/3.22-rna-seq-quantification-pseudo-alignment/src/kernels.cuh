// ===========================================================================
// src/kernels.cuh  --  GPU EM interface for RNA-seq pseudo-alignment
// ---------------------------------------------------------------------------
// Project 3.22 : RNA-seq Quantification / Pseudo-alignment
//
// THE BIG IDEA (PATTERNS.md: parallel per-item E-step + ATOMIC REDUCTION M-step)
//   Each EM iteration sweeps the equivalence classes (ecs):
//     * E-STEP  : one GPU thread per ec reads the current abundances of its few
//                 member transcripts and splits the ec's reads among them
//                 (pseudoalign.h::psa_ec_contributions). Fully independent ecs.
//     * M-STEP  : that same thread atomically adds each member's expected reads,
//                 in FIXED-POINT integer units, into a per-transcript accumulator
//                 -- a SCATTER-REDUCTION via atomicAdd. Integer adds commute, so
//                 the reduction is DETERMINISTIC and equals the CPU exactly.
//   The renormalise (counts -> next rho) is a tiny host step reused from the CPU
//   reference, so CPU and GPU take identical update steps. Conceptually each
//   iteration is a sparse matrix-vector product (the ec-by-transcript membership
//   matrix times the weight vector); cuSPARSE could do it as a library SpMV --
//   we hand-roll it here so nothing is a black box (THEORY.md "real world").
//
//   kernels.cu defines the kernel; main.cu calls em_gpu().
//
// READ THIS AFTER: pseudoalign.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>
#include "reference_cpu.h"   // EcDataset + shared host helpers (pure C++, safe in .cu)

// Device-side EM iteration kernel. One thread per equivalence class:
//   * runs the E-step for its ec,
//   * atomic-adds each member's fixed-point expected reads into d_fixed_counts.
// Parameters mirror the CSR layout of EcDataset. Declared here, defined in
// kernels.cu; main.cu never launches it directly -- it calls em_gpu().
__global__ void em_iteration_kernel(const double*       __restrict__ d_rho,
                                    const double*       __restrict__ d_eff_len,
                                    const double*       __restrict__ d_ec_count,
                                    const std::int32_t* __restrict__ d_ec_offset,
                                    const std::int32_t* __restrict__ d_ec_members,
                                    int M,
                                    unsigned long long* __restrict__ d_fixed_counts);

// Host wrapper: run `iters` EM iterations on the GPU from the uniform start.
// Fills `rho` (final abundances, length T) and `est_counts` (final per-transcript
// expected read counts, length T). Returns the final L1 change in rho (the same
// convergence witness as em_cpu) and the total GPU kernel time via kernel_ms.
double em_gpu(const EcDataset& d, int iters,
              std::vector<double>& rho, std::vector<double>& est_counts,
              float* kernel_ms);
