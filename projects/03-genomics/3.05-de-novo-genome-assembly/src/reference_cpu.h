// ===========================================================================
// src/reference_cpu.h  --  FASTA loader + CPU reference for all-vs-all overlap
// ---------------------------------------------------------------------------
// Project 3.5 : De Novo Genome Assembly  (read-overlap stage)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax. So the data LOADER and the serial reference live here / there.
//   The shared per-element math (minimizer sketch, pair intersection) lives in
//   assembly.h, which BOTH this file and the GPU side include -- that is what
//   makes CPU and GPU agree exactly (PATTERNS.md sec.2).
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   We have a batch of DNA reads (strings over {A,C,G,T}). For every pair of
//   reads we count how many *minimizers* they share; pairs that share at least
//   MIN_SHARED minimizers are reported as candidate OVERLAPS -- edges of the
//   assembly graph. The reference computes this serially; the GPU does it in
//   parallel; main.cu checks they match bit-for-bit.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. READ assembly.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "assembly.h"   // ReadSet, Overlap, minimizer math (HD-shared core)

// Minimum shared-minimizer count for a pair to be reported as an overlap edge.
//   This is the sensitivity knob: too low -> spurious edges (chance k-mer
//   collisions); too high -> miss real but error-rich overlaps. The value here
//   is chosen so the synthetic sample yields a clean, interpretable graph whose
//   true overlaps are recovered (see THEORY "Verification" and data/README.md).
constexpr int MIN_SHARED = 3;

// Load reads from a tiny FASTA-like text file (format in data/README.md):
//   a header line beginning with '>' (read name; ignored except for count),
//   followed by one sequence line of A/C/G/T (upper or lower case). Repeats.
//   Any read shorter than K bases is skipped (it has no k-mers). Throws
//   std::runtime_error on a missing/empty file so demos fail loudly.
std::vector<std::string> load_fasta(const std::string& path);

// CPU REFERENCE: the trusted, obviously-correct serial all-vs-all overlap.
//   For every unordered pair (i<j) it computes count_shared_sorted() and, where
//   that meets MIN_SHARED, appends an Overlap. The returned vector is sorted
//   deterministically (by i, then j) so it can be compared element-for-element
//   against the GPU result. `out_score_all` (optional, may be null) receives the
//   full P-length shared-count array for an exact GPU-vs-CPU diff of *every*
//   pair, not just the thresholded edges.
//     rs            : the sketched reads (from sketch_reads()).
//     overlaps      : output, filled with edges where shared >= MIN_SHARED.
//     out_score_all : optional output, resized to num_pairs(rs.n); shared count
//                     of every pair in flat upper-triangle order.
void overlap_cpu(const ReadSet& rs,
                 std::vector<Overlap>& overlaps,
                 std::vector<int>* out_score_all);
