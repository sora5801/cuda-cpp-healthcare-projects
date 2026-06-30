// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for profile-HMM database search
// ---------------------------------------------------------------------------
// Project 3.28 : Profile HMM (Viterbi / Forward)
//
// THE BIG IDEA
//   Scoring one profile against N database sequences is N INDEPENDENT jobs --
//   the same "one query vs many items" shape as flagship 1.12 (Tanimoto) and
//   12.01 (spectral search). So we give each database sequence its OWN GPU
//   THREAD, and that thread runs the full Viterbi (or Forward) dynamic program
//   for its sequence against the shared profile. Two CUDA features carry the
//   teaching here:
//     * the PROFILE lives in CONSTANT memory: every thread reads the same model
//       but never writes it, so the constant cache broadcasts it warp-wide
//       (cheap), exactly like the constant-memory query in 1.12; and
//     * each thread keeps its DP "rolling column" (3 small arrays of size M+1) in
//       LOCAL/register space and walks its sequence -- no global-memory traffic
//       in the inner loop, so the kernel is compute-bound on the recurrence.
//   A grid-stride loop lets one modest grid cover an arbitrarily large database.
//
//   This header declares a __global__ kernel, so it is included ONLY by .cu
//   units. The pure-C++ CPU reference lives in reference_cpu.h instead.
//
// READ THIS AFTER: phmm.h (the shared recurrence), util/cuda_check.cuh,
//   util/timer.cuh, reference_cpu.h.  Then read kernels.cu.  The GPU-mapping is
//   in ../THEORY.md §4.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // ProfileHMM, SeqDB (pure C++, safe inside a .cu)

// ---------------------------------------------------------------------------
// phmm_search_gpu: the host-callable "score the whole database on the GPU".
//   Uploads the profile to constant memory and the (flat, ragged) sequence
//   database to global memory, launches ONE kernel that fills `out` with every
//   sequence's score, and reports the measured kernel time via *kernel_ms.
//
//   p          : the profile HMM model (copied into device constant memory)
//   db         : the sequence database (flat residue buffer + offsets/lengths)
//   is_viterbi : true  -> Viterbi (max-sum) score per sequence
//                false -> Forward (log-sum-exp) score per sequence
//   out        : resized to db.n; filled with per-sequence scores (nats)
//   kernel_ms  : out-param, GPU-measured kernel time in milliseconds
//
//   main.cu calls this twice (Viterbi then Forward); all CUDA bookkeeping is
//   hidden inside. See kernels.cu for the five canonical CUDA steps.
// ---------------------------------------------------------------------------
void phmm_search_gpu(const ProfileHMM& p, const SeqDB& db, bool is_viterbi,
                     std::vector<float>& out, float* kernel_ms);
