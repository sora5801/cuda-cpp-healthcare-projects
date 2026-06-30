// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for phylogenetic likelihood
// ---------------------------------------------------------------------------
// Project 3.9 : Phylogenetic Likelihood / Tree Inference
//
// THE BIG IDEA
//   Felsenstein's pruning recursion evaluates each alignment SITE independently
//   (the per-column likelihoods only get combined at the very end), so scoring a
//   tree over n_sites columns is n_sites INDEPENDENT jobs. We give each site its
//   own GPU thread. Two teaching points make this the right mapping:
//
//     * The COLUMN of a site (one base per taxon) is the only alignment data a
//       thread needs. Storing the alignment COLUMN-MAJOR (reference_cpu.h) means
//       thread j reads a contiguous run of n_taxa bytes -> coalesced loads.
//
//     * The tree (its post-ordered nodes) is read by EVERY thread but never
//       written during the launch, so it lives in CONSTANT memory: the constant
//       cache broadcasts each node to a whole warp in one transaction. A binary
//       tree on <=64 taxa fits easily in the 64 KB constant bank.
//
//   The per-site math is the SHARED site_log_likelihood() in felsenstein.h, the
//   exact same function the CPU reference calls -> CPU and GPU run identical
//   arithmetic. To make the SUM over sites deterministic (a float atomic sum is
//   not -- PATTERNS.md sec 3), each thread converts its site lnL to a FIXED-POINT
//   integer and atomicAdd's it into a 64-bit accumulator; integer adds commute,
//   so the total is reproducible and matches the CPU bit-for-bit.
//
//   This header is included only by .cu units. main.cu calls score_trees_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, felsenstein.h,
// reference_cpu.h. Then read kernels.cu. The GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // PhyloProblem, PhyloNode (pure C++, safe in .cu)

// Largest tree the constant-memory layout supports. A rooted binary tree on
// n_taxa leaves has n_taxa-1 internal nodes; MAX_INTERNAL_NODES therefore caps
// n_taxa at 65. 64 PhyloNodes * 32 bytes ~= 2 KB, far inside the 64 KB constant
// bank. Raise this (and the __constant__ array in kernels.cu) for bigger trees.
#define MAX_INTERNAL_NODES 64

// Host wrapper: score EVERY candidate tree on the GPU.
//   prob       : the loaded problem (alignment + kappa + candidate trees).
//   tree_lnL   : resized to prob.trees.size(); filled with each tree's total
//                log-likelihood (summed over sites via the fixed-point reduction).
//   kernel_ms  : out-param, total GPU kernel time across all trees (CUDA events).
//
//   The wrapper uploads the column-major alignment once, then for each tree
//   uploads its nodes to constant memory and launches one thread per site.
void score_trees_gpu(const PhyloProblem& prob, std::vector<double>& tree_lnL,
                     float* kernel_ms);
