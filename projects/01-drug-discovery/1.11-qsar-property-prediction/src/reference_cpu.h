// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for the QSAR GCN
// ---------------------------------------------------------------------------
// Project 1.11 : QSAR / Property Prediction
//
// Pure C++ (no CUDA), so it is included by BOTH the host reference
// (reference_cpu.cpp) and the GPU code (kernels.cu -- a .cu file can include
// plain headers). The per-node math lives in gcn.h (GCN_HD, host+device).
//
// WHAT'S HERE
//   * Graph      -- a BATCH of molecular graphs packed in CSR form (below).
//   * Model      -- the (fixed, supplied) GCN weights for inference.
//   * loaders    -- read the sample graph batch and the weights from text.
//   * gcn_predict_cpu -- the trusted serial reference: run the 2-layer GCN +
//                  readout for every molecule and return the predictions.
//
// WHY CSR (Compressed Sparse Row) FOR THE BATCH
//   Molecules have wildly different atom counts and the adjacency is sparse, so
//   a dense [N x N] matrix per molecule would be mostly zeros. Instead we flatten
//   the WHOLE BATCH into three parallel arrays:
//       row_ptr[i], row_ptr[i+1]  -> the slice of `col_idx` holding node i's
//                                    neighbors (SELF-LOOP INCLUDED).
//       col_idx[...]              -> neighbor node indices (global, batch-wide).
//   Node indices are global across the batch (molecule 0's atoms come first,
//   then molecule 1's, ...). `mol_start` maps molecules to their node range.
//   This is exactly the layout PyTorch Geometric / DGL use to batch graphs.
//
// READ THIS AFTER: gcn.h. READ BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "gcn.h"   // GCN_F_IN/HID/OUT and the shared per-node math

// ---------------------------------------------------------------------------
// Graph: one batch of molecular graphs in CSR form.
//   num_nodes : total atoms across all molecules in the batch.
//   num_mols  : number of molecules.
//   feat      : [num_nodes * GCN_F_IN] atom features, row-major (node-major).
//   deg       : [num_nodes] degree WITH self-loop (== neighbors listed in CSR).
//   row_ptr   : [num_nodes + 1] CSR offsets into col_idx.
//   col_idx   : [row_ptr[num_nodes]] neighbor node indices (self-loop included).
//   mol_start : [num_mols + 1] first node index of each molecule (prefix sum of
//               atom counts) -> molecule m owns nodes [mol_start[m], mol_start[m+1]).
// ---------------------------------------------------------------------------
struct Graph {
    int num_nodes = 0;
    int num_mols  = 0;
    std::vector<float> feat;     // [num_nodes * GCN_F_IN]
    std::vector<int>   deg;      // [num_nodes]
    std::vector<int>   row_ptr;  // [num_nodes + 1]
    std::vector<int>   col_idx;  // [num_edges_with_self_loops]
    std::vector<int>   mol_start;// [num_mols + 1]

    // Atoms in molecule m (used by the readout / pooling).
    int atoms_in(int m) const { return mol_start[m + 1] - mol_start[m]; }
};

// ---------------------------------------------------------------------------
// Model: the fixed GCN weights used for INFERENCE (training is in THEORY.md).
//   W1,b1 : layer-1 weights [GCN_F_IN x GCN_F_HID] and bias [GCN_F_HID].
//   W2,b2 : layer-2 weights [GCN_F_HID x GCN_F_OUT] and bias [GCN_F_OUT].
//   head_w,head_b : readout head [GCN_F_OUT] + scalar bias.
//   All row-major (in-channel-major) to match gcn_aggregate_then_transform.
// ---------------------------------------------------------------------------
struct Model {
    std::vector<float> W1, b1;   // [F_IN*F_HID], [F_HID]
    std::vector<float> W2, b2;   // [F_HID*F_OUT], [F_OUT]
    std::vector<float> head_w;   // [F_OUT]
    float head_b = 0.0f;
};

// Load a batched graph from the sample text format (see data/README.md).
Graph load_graph(const std::string& path);

// Build the fixed GCN weights used for inference. The weights are DETERMINISTIC
// (generated from a fixed integer seed by a tiny LCG, documented in the .cpp)
// so the demo's predictions are byte-stable without committing a weights file.
Model make_model();

// ---------------------------------------------------------------------------
// gcn_predict_cpu: the trusted serial reference.
//   Runs the full pipeline for every molecule:
//     layer 1 (F_IN -> F_HID, ReLU), layer 2 (F_HID -> F_OUT, no ReLU),
//     mean-pool readout + linear head -> one scalar per molecule.
//   Fills `pred` with num_mols predictions. Uses ONLY gcn.h math, so the GPU
//   (kernels.cu) reproduces it exactly. Returns nothing; `pred` is the output.
// ---------------------------------------------------------------------------
void gcn_predict_cpu(const Graph& g, const Model& m, std::vector<float>& pred);
