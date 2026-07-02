// ===========================================================================
// src/reference_cpu.h  --  Purkinje tree config + CPU reference declarations
// ---------------------------------------------------------------------------
// Project 6.17 : Purkinje System & Conduction System Modeling
//
// A PurkinjeTree is an ENSEMBLE of 1-D cables plus a small branching GRAPH that
// says how they connect (His bundle -> bundle branches -> Purkinje fascicles ->
// Purkinje-muscle junctions). The tree config + the graph-delay computation live
// here; the per-cable PDE/stepper lives in purkinje.h (shared host+device). This
// file is pure C++ so kernels.cu can reuse the struct and the loader.
//
// READ THIS AFTER: purkinje.h. READ THIS BEFORE: kernels.cuh, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "purkinje.h"   // PK_HD, CableParams, CableResult, pk_simulate_cable

// ---------------------------------------------------------------------------
// PurkinjeTree  --  the whole simulation: N cables + their tree topology.
//   `cables[i].parent` gives the edge set (a rooted tree): cable i's proximal
//   end is fed by its parent's distal end. Root cables have parent == -1 and are
//   paced directly (the His bundle). This is stored as a flat array so it copies
//   to the device in one cudaMemcpy.
// ---------------------------------------------------------------------------
struct PurkinjeTree {
    std::vector<CableParams> cables;   // one entry per Purkinje cable / segment
};

// Number of cables in the tree.
inline int tree_size(const PurkinjeTree& t) { return (int)t.cables.size(); }

// ---------------------------------------------------------------------------
// Load a PurkinjeTree from the tiny text format (see data/README.md):
//   line 1:  N  dt_ms  n_steps           (global integration settings)
//   next N:  n_nodes length_mm D stim_amp stim_dur_ms stim_width thresh parent delay_ms
//              one line per cable, in index order (0..N-1)
//   Throws std::runtime_error on any malformed / missing field so demos fail loud.
// ---------------------------------------------------------------------------
PurkinjeTree load_tree(const std::string& path);

// ---------------------------------------------------------------------------
// CPU reference: simulate every cable serially (the trusted baseline the GPU is
// checked against -- identical pk_simulate_cable() math => results match to
// round-off). `results` is sized to N on return.
// ---------------------------------------------------------------------------
void simulate_cpu(const PurkinjeTree& t, std::vector<CableResult>& results);

// ---------------------------------------------------------------------------
// compute_activation_times  --  graph-based conduction delays over the tree.
//   Given each cable's local propagation delay (proximal -> distal end, in ms)
//   and the fixed junction delay to ENTER it, walk the rooted tree from the
//   root(s) and accumulate the absolute time at which each cable's DISTAL end
//   (its PMJ) activates:
//
//       t_out[i] = t_in[i] + local_delay[i]
//       t_in[child] = t_out[parent] + child.delay_ms
//       t_in[root]  = root.delay_ms
//
//   The tree is topologically ordered by construction (a parent's index is
//   always < its children's), so a single forward pass suffices -- an O(N) graph
//   traversal. Returns per-cable distal activation times (ms); the maximum is the
//   total ventricular activation time. Deterministic integer-step arithmetic
//   feeds this, so CPU and GPU agree exactly.
// ---------------------------------------------------------------------------
std::vector<double> compute_activation_times(const PurkinjeTree& t,
                                             const std::vector<CableResult>& res);
