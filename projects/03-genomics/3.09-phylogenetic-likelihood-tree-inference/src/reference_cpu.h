// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for phylogenetic likelihood
// ---------------------------------------------------------------------------
// Project 3.9 : Phylogenetic Likelihood / Tree Inference
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the alignment, the candidate trees,
//   the text loader) and the CPU reference prototypes live here. kernels.cuh
//   also includes this header to reuse the same structs -- nothing CUDA-specific
//   leaks in either direction. The per-site MATH is in felsenstein.h (shared by
//   host and device); this file is about LOADING and the CPU driver.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   We are handed a DNA alignment (n_taxa sequences, each n_sites long) and a
//   handful of candidate evolutionary TREES. For each tree we compute the total
//   log-likelihood (sum over sites of Felsenstein's pruning recursion) and report
//   which tree the data supports best. Every site's likelihood is independent ->
//   one GPU thread per site (kernels.cu).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Math: felsenstein.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "felsenstein.h"   // PhyloNode, site_log_likelihood, PHYLO_* (pure C++ safe)

// ---------------------------------------------------------------------------
// Alignment: the observed data.
//   n_taxa  : number of sequences (rows / leaves of every tree).
//   n_sites : alignment length (columns); each column is one independent site.
//   names   : taxon labels (for human-readable reporting).
//   data    : n_sites * n_taxa bytes, COLUMN-MAJOR: site j's states occupy
//             data[j*n_taxa .. j*n_taxa + n_taxa-1]. Column-major is deliberate:
//             the GPU gives site j to thread j, and a thread reads its whole
//             column contiguously -> coalesced, cache-friendly loads.
//             Each byte is a state 0..3 (A,C,G,T) or PHYLO_GAP for '-'/'N'.
// ---------------------------------------------------------------------------
struct Alignment {
    int n_taxa  = 0;
    int n_sites = 0;
    std::vector<std::string>   names;   // [n_taxa]
    std::vector<unsigned char> data;    // [n_sites * n_taxa], column-major
};

// ---------------------------------------------------------------------------
// CandidateTree: one fixed topology + branch lengths to score.
//   label    : a human name for the tree (e.g. "true", "NNI-swap-1").
//   nodes    : internal nodes in POST-ORDER (children before parents). The last
//              node is the root. Leaves are indices 0..n_taxa-1 (not stored).
//   n_internal == nodes.size(); for a rooted binary tree on n_taxa leaves it is
//   n_taxa - 1.
// ---------------------------------------------------------------------------
struct CandidateTree {
    std::string            label;
    std::vector<PhyloNode> nodes;       // [n_internal], post-ordered, root last
};

// ---------------------------------------------------------------------------
// PhyloProblem: everything the loader produces and the CPU/GPU drivers consume.
//   kappa  : the K2P transition/transversion ratio used to score every tree.
//   trees  : the candidate topologies to compare.
// ---------------------------------------------------------------------------
struct PhyloProblem {
    Alignment                  align;
    double                     kappa = 2.0;   // typical empirical value (~2)
    std::vector<CandidateTree> trees;
};

// Load a PhyloProblem from the text format documented in data/README.md.
//   Throws std::runtime_error on a missing/malformed file. See reference_cpu.cpp
//   for the exact grammar (header, sequences, kappa, then one block per tree).
PhyloProblem load_problem(const std::string& path);

// CPU reference: for each candidate tree, fill tree_lnL[t] with that tree's total
// log-likelihood by summing the per-site pruning recursion. This is the trusted,
// obviously-correct baseline the GPU result is checked against (and the timing
// baseline that makes the speed-up legible). tree_lnL is resized to trees.size().
//
// We sum in FIXED-POINT integers (felsenstein.h to_fixed) so the CPU total is
// computed exactly the way the GPU's atomic-integer total is -> the two agree to
// the last bit, not merely "within a tolerance".
void score_trees_cpu(const PhyloProblem& prob, std::vector<double>& tree_lnL);

// Helper shared by main: index of the largest-log-likelihood tree (ties broken by
// lower index, so the winner is deterministic). Defined in reference_cpu.cpp.
int best_tree_index(const std::vector<double>& tree_lnL);
