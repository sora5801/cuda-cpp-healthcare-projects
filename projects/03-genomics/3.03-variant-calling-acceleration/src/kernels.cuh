// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.3 : Variant Calling Acceleration
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls pairhmm_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by .cu
//   translation units (it pulls in CUDA types), so the plain C++ compiler never
//   sees it -- which is why the CPU reference lives in a separate pure-C++ header
//   (reference_cpu.h) and the SHARED per-cell math lives in pairhmm_core.h.
//
// THE BIG IDEA (PATTERNS.md §1: independent jobs)
//   We must score every (read, haplotype) PAIR independently: there are
//   n_reads * n_haps such pairs, and each one fills its own forward DP table with
//   no dependency on any other pair. So we assign ONE GPU THREAD PER PAIR. With
//   P = n_reads*n_haps pairs and a block of B threads we launch ceil(P / B)
//   blocks; thread t owns pair index t = blockIdx.x*blockDim.x + threadIdx.x,
//   which decodes to (read = t / n_haps, hap = t % n_haps).
//
//   Each thread runs the SAME forward algorithm as the CPU reference, calling the
//   shared pairhmm_step() from pairhmm_core.h, and keeps two rolling DP rows in
//   local memory (O(hap_len) per thread). This "one independent job per thread"
//   mapping is exactly how a simplified Parabricks/GATK PairHMM batches thousands
//   of read-haplotype pairs at once; THEORY.md "GPU mapping" covers the
//   production refinement (one BLOCK per pair, shared-memory anti-diagonal).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pairhmm_core.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // VariantData (the loaded problem) + PairHmmParams

// ---- Host wrapper --------------------------------------------------------
// pairhmm_gpu: the host-callable "do the whole GPU computation" function.
//   Uploads the reads/qualities/haplotypes + transition params, launches one
//   thread per (read, haplotype) pair to fill that pair's forward DP table,
//   copies the R x H log10-likelihood matrix back, and reports the measured
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this; all
//   CUDA bookkeeping is hidden here.
//
//   v         : the loaded problem (reads, haps, quals, pair-HMM params)
//   loglik    : host output, resized to n_reads*n_haps (row-major), output param
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void pairhmm_gpu(const VariantData& v, std::vector<double>& loglik, float* kernel_ms);
