// ===========================================================================
// src/reference_cpu.h  --  Pangenome model, term construction, CPU reference
// ---------------------------------------------------------------------------
// Project 3.30 : Pangenome Graph Construction
//
// Pure C++ (no CUDA). This header declares:
//   * Pangenome     -- the loaded graph: node lengths + genome paths.
//   * LayoutProblem -- the derived stress problem: an array of LayoutTerm plus
//                      the initial node positions and the SGD schedule.
//   * helpers shared by BOTH the CPU reference and the GPU wrapper so the two
//     produce identical layouts: schedule (learning-rate decay), stress metric,
//     and the deterministic initial placement.
//   * layout_cpu()  -- the trusted serial stress-majorization reference.
//
// The per-term physics itself lives in layout.h (LO_term_displacement), included
// by this header and by kernels.cu, so CPU and GPU run identical math.
//
// READ THIS AFTER: layout.h.   READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "layout.h"   // LayoutTerm, LO_term_displacement, fixed-point helpers

// ---------------------------------------------------------------------------
// Pangenome: the parsed graph in the tiny teaching format (see data/README.md).
//   node_len[k] : length of node k in base pairs (the segment's sequence size).
//   paths       : one entry per genome; each is the ordered list of node ids
//                 that genome visits (its "walk" through the graph, like a GFA
//                 'P'/'W' line). Adjacent nodes in a path are genomically
//                 co-linear -- that adjacency is what the layout must preserve.
// We deliberately ignore strand/orientation and sequence content: the 1-D layout
// only needs the topology (which nodes are near which along each genome) and the
// node lengths (which set the target separations in base pairs).
// ---------------------------------------------------------------------------
struct Pangenome {
    int                           num_nodes = 0;   // N: number of nodes/segments
    std::vector<int>              node_len;        // [N] node lengths (bp)
    std::vector<std::vector<int>> paths;           // [P] genome walks (node-id lists)
};

// ---------------------------------------------------------------------------
// LayoutProblem: the stress problem derived from a Pangenome.
//   terms  : all node-pair soft constraints (built by build_problem()).
//   init_x : the deterministic initial 1-D coordinate of each node [N].
//   iters  : number of full-batch SMACOF (stress-majorization) sweeps. SMACOF is
//            monotone and parameter-free, so there is no learning-rate schedule.
// ---------------------------------------------------------------------------
struct LayoutProblem {
    std::vector<LayoutTerm> terms;
    std::vector<double>     init_x;
    int                     iters = 0;
};

// Load a Pangenome from the tiny text format (data/README.md describes it).
//   Throws std::runtime_error on a malformed/missing file so demos fail loudly.
Pangenome load_pangenome(const std::string& path);

// Build the stress problem from a graph:
//   * For every pair of nodes (i, j) that are up to `hops` path-steps apart
//     within SOME genome path, add a term whose target separation is the sum of
//     the intervening node lengths (their genomic distance in bp) and whose
//     weight is 1/d^2 (ODGI's weighting: short distances matter most). Duplicate
//     pairs (a pair seen in several paths / hop counts) are merged to the SMALLEST
//     target distance so the construction is order-independent and deterministic.
//   * Place node k initially at a deterministic 1-D coordinate (init_layout()).
// SMACOF is parameter-free, so no learning-rate schedule is needed.
LayoutProblem build_problem(const Pangenome& g, int hops, int iters);

// Deterministic initial placement: lay the nodes out along the axis in the order
// the FIRST genome path visits them (a sensible reference frame), spacing them by
// their lengths. Nodes not on path 0 are appended in id order. Shared by CPU/GPU
// so both start from the identical configuration.
void init_layout(const Pangenome& g, std::vector<double>& x);

// Total weighted stress  E(x) = sum_terms w*(|x_i - x_j| - d)^2.
//   This is the objective the layout minimises; we print it so the learner can
//   watch it fall and so CPU/GPU can be compared on a single scalar too.
double compute_stress(const LayoutProblem& p, const std::vector<double>& x);

// CPU reference: run the full-batch stress-majorization layout serially.
//   Fills `x` (final [N] positions) and returns the final stress. The trusted
//   baseline the GPU result is checked against.
double layout_cpu(const LayoutProblem& p, std::vector<double>& x);
