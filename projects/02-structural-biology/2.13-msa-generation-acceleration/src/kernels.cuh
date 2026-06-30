// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for profile-HMM database search
// ---------------------------------------------------------------------------
// Project 2.13 : MSA Generation Acceleration
//
// THE BIG IDEA (the teaching points of this project)
//   Scoring the query profile against N database sequences is N INDEPENDENT
//   Viterbi dynamic programs -- the same pattern as 1.12 (one query vs N items),
//   but each "item" is itself a small DP. The mapping that makes this fast:
//     * ONE BLOCK PER DATABASE SEQUENCE. A block owns one sequence's whole DP.
//     * THREADS WITHIN THE BLOCK split the L profile columns: each thread updates
//       a slice of the M/I row. We march the database sequence one residue at a
//       time; after each residue the block __syncthreads() so the next row sees a
//       completed previous row (the Viterbi data dependency is row-to-row).
//     * THE DP ROW LIVES IN SHARED MEMORY (fast, block-private) -- prev and cur
//       M/I/D arrays of length L+1 -- so the per-residue sweep never touches slow
//       global memory except to stream in the residues.
//     * THE EMISSION TABLE LIVES IN CONSTANT MEMORY: it is read by every thread
//       of every block and never changes during the launch, so the constant
//       cache broadcasts it (exactly like the query in 1.12).
//   The delete chain (D[k] depends on D[k-1] of the SAME row) is inherently
//   sequential along k, so one designated thread resolves the deletes per row --
//   see kernels.cu for that nuance and why it keeps results identical to the CPU.
//
// READ THIS AFTER: hmm_core.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // SearchProblem, ProfileHMM, SeqDB (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// MAX_PROFILE_L : compile-time cap on the profile length L.
//   The emission table is uploaded to a fixed-size __constant__ array and the
//   per-block shared-memory rows are sized from this. 256 columns is comfortably
//   larger than the teaching sample and keeps constant memory tiny
//   (256 * 21 * 4 bytes ~= 21 KB, well within the 64 KB constant bank). A real
//   tool would stream arbitrarily long profiles; we cap it to keep the teaching
//   kernel's memory static and obvious. load + wrapper check L <= MAX_PROFILE_L.
// ---------------------------------------------------------------------------
constexpr int MAX_PROFILE_L = 256;

// ---------------------------------------------------------------------------
// viterbi_search_gpu : host wrapper.
//   Uploads the profile (emissions to constant memory, transitions as args) and
//   the CSR-packed database to global memory, launches one block per sequence,
//   times the kernel with CUDA events, and returns the per-sequence best scores.
//     prob       : the loaded search problem (profile HMM + database)
//     out        : resized to prob.db.N; filled with scaled-integer hit scores
//     kernel_ms  : out-param, GPU-measured kernel time in milliseconds
//   Postcondition: out[i] equals viterbi_search_cpu's out[i] exactly (integers).
// ---------------------------------------------------------------------------
void viterbi_search_gpu(const SearchProblem& prob, std::vector<int>& out, float* kernel_ms);
