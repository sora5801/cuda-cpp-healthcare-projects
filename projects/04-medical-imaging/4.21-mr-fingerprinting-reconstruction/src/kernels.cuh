// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.21 : MR Fingerprinting Reconstruction
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls ONE host wrapper declared
//   here (gpu_reconstruct); kernels.cu implements it (the device kernels + the
//   cuBLAS SGEMM live there). Included only by .cu translation units, so the
//   pure-C++ CPU reference uses a separate header (reference_cpu.h). The shared
//   per-element math is in mrf_core.h, included by both sides.
//
// THE GPU PIPELINE (and the pattern each stage uses -- see ../THEORY.md, PATTERNS.md)
//
//   Stage 1 -- BUILD the dictionary: one GPU THREAD PER ATOM simulates that
//     atom's length-T fingerprint (mrf::simulate_atom) and L2-normalizes it.
//     "Independent jobs" pattern (PATTERNS.md §1): D independent simulations.
//
//   Stage 2 -- NORMALIZE the voxel signals: one thread per voxel L2-normalizes
//     its length-T signal (same shared mrf routine), recording the scale for
//     the proton-density map.
//
//   Stage 3 -- MATCH: form the entire V×D cosine-score matrix
//         S = Signal_norm[V×T] · Dict_normᵀ[T×D]
//     with cuBLAS SGEMM (PATTERNS.md §1 "score query vs N items" + §5 "use the
//     library"). One call replaces the CPU's triple loop over ~10^11 inner
//     products. Then a per-voxel argmax kernel (one thread per voxel) reads
//     row v of S and picks the best-matching atom -- the reconstructed tissue.
//
//   Every scalar (Bloch step, normalization, inner product) is the SHARED one
//   in mrf_core.h, so the GPU dictionary and normalized signals match the CPU
//   exactly; only SGEMM's summation ORDER differs from the CPU's serial dot,
//   which is why main.cu verifies with a documented float tolerance AND an exact
//   per-voxel argmax-index check.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, mrf_core.h.
// THEN READ kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // MrfProblem, MatchResult (shared host data model)

// ---------------------------------------------------------------------------
// GpuTimings: per-stage millisecond breakdown, filled by gpu_reconstruct and
//   printed to stderr by main.cu (a teaching artifact, never a benchmark claim).
// ---------------------------------------------------------------------------
struct GpuTimings {
    float build_ms   = 0.0f;   // stage 1: dictionary simulation + normalize
    float normsig_ms = 0.0f;   // stage 2: voxel-signal normalization
    float sgemm_ms   = 0.0f;   // stage 3a: cuBLAS SGEMM (the headline step)
    float argmax_ms  = 0.0f;   // stage 3b: per-voxel argmax over the score row
};

// ---------------------------------------------------------------------------
// gpu_reconstruct: run the full MRF reconstruction on the GPU.
//   INPUTS (host):
//     p : the loaded problem (schedule, dictionary grid, voxel signals).
//   OUTPUTS (host):
//     out         : resized to [V], one MatchResult per voxel (best atom,
//                   cosine score, matched T1/T2, proton density).
//     dict_norm   : resized to [D*T], the GPU-built normalized dictionary
//                   (returned so main.cu can verify it against the CPU build).
//     score_row_v0: resized to [D], the full cosine score row for voxel 0
//                   (returned so main.cu can spot-check the SGEMM against the
//                   CPU inner products for one voxel).
//   TIMING:
//     *timings : per-stage CUDA-event milliseconds.
//   The function owns all device memory + the cuBLAS handle; main.cu just sees
//   host vectors in and out.
void gpu_reconstruct(const MrfProblem& p,
                     std::vector<MatchResult>& out,
                     std::vector<float>& dict_norm,
                     std::vector<float>& score_row_v0,
                     GpuTimings* timings);
