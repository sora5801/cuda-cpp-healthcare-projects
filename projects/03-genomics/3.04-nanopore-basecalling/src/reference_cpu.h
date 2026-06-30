// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for CTC basecalling
// ---------------------------------------------------------------------------
// Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the posterior-matrix container and
//   the file loader) and the CPU reference prototypes live here. kernels.cuh
//   (the GPU side) also includes this header to reuse the ReadSet type --
//   nothing CUDA-specific leaks in either direction. The actual per-read MATH
//   lives in ctc_core.h (the __host__ __device__ shared core).
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   A nanopore basecaller's neural network emits, for each READ, a posterior
//   matrix P of shape [T x C]: at each of T time steps it gives a probability
//   over C = 5 classes {blank, A, C, G, T}. We DECODE each read's matrix into a
//   DNA sequence via greedy CTC collapse (ctc_core.h). Every read is decoded
//   INDEPENDENTLY -> perfect data parallelism: one GPU thread per read.
//
//   We do NOT train or run the network here (that is the research-grade part,
//   described in THEORY "real world"). We take the network's OUTPUT posteriors
//   as input -- exactly the interface a decode stage sees in Dorado/Guppy.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Pairs with ctc_core.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ctc_core.h"   // CTC_NUM_CLASSES, the shared decode (pure-C++ safe)

// ---------------------------------------------------------------------------
// ReadSet: a batch of reads, each with its own posterior matrix.
//   Reads can have DIFFERENT lengths T_i (nanopore reads vary wildly), so we
//   store all posteriors in ONE flat array and index into it with offsets --
//   the standard "jagged array as flat buffer + offsets" layout that keeps the
//   GPU upload a single contiguous cudaMemcpy.
//
//   Layout of `probs` (row-major, C = CTC_NUM_CLASSES columns):
//     read r occupies probs[offset[r]*C .. offset[r]*C + T[r]*C - 1]
//     within that, step t, class c is at  probs[(offset[r] + t)*C + c].
//   `offset` has n_reads+1 entries (prefix sums of T); offset[n] == total steps.
// ---------------------------------------------------------------------------
struct ReadSet {
    int                   n_reads = 0;   // number of reads in the batch
    std::vector<int>      T;             // [n_reads]   time steps per read
    std::vector<int>      offset;        // [n_reads+1] prefix sum of T (in steps)
    std::vector<float>    probs;         // [offset[n_reads] * C] all posteriors
    int                   max_T = 0;     // max over T[r] (handy for buffer sizing)
};

// One read's decoded result: the called bases plus a deterministic checksum.
// (Kept tiny and value-typed so it copies cheaply between host containers.)
struct DecodedRead {
    std::string base_seq;   // the called DNA sequence, e.g. "ACGT..."
    uint32_t    checksum;   // ctc_base_checksum(base_seq) -- exact integer hash
    int         length;     // base_seq.size() (decoded read length)
};

// Load a ReadSet from the text format documented in data/README.md:
//   line 1 : "<n_reads> <C>"            (C must equal CTC_NUM_CLASSES)
//   then for each read:
//     a line "<T>"                       (this read's number of time steps)
//     T lines, each C floats             (the per-step class probabilities)
// Throws std::runtime_error on a missing file, a bad header, or a C mismatch.
ReadSet load_reads(const std::string& path);

// CPU reference: decode every read in `rs` with the shared ctc_core decode.
//   out is resized to rs.n_reads; out[r] is read r's DecodedRead. This is the
//   trusted, obviously-correct baseline the GPU result is checked against (and
//   the timing baseline that makes the speed-up legible).
void basecall_cpu(const ReadSet& rs, std::vector<DecodedRead>& out);
