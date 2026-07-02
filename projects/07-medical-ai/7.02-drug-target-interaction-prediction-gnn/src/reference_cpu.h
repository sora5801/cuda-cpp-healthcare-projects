// ===========================================================================
// src/reference_cpu.h  --  Dataset (batched graphs) + model + CPU reference
// ---------------------------------------------------------------------------
// Project 7.2 : Drug-Target Interaction Prediction (GNN)
//
// Pure C++ (no CUDA). The per-element math is in gnn.h. This header declares:
//   * Dataset      -- a batch of drug molecular graphs (CSR) + protein vectors.
//   * GnnModel     -- the FIXED (untrained) message-passing weights.
//   * load_dataset -- read the tiny text sample (format in data/README.md).
//   * build_model  -- deterministically seed the weights from the feature width.
//   * the CPU reference (embeddings + DTI score matrix) that the GPU must match.
//
// GRAPH STORAGE (CSR = Compressed Sparse Row -- how DGL/PyG batch graphs)
//   All drugs' atoms live in ONE flat node array. Node i's features are
//   feat[i*F .. i*F+F). Its bonded neighbours are the entries
//   adj[ adj_off[i] .. adj_off[i+1] ), where adj holds GLOBAL node indices.
//   Drug d owns nodes [node_off[d], node_off[d+1]). This "one big batched
//   graph" layout is exactly what lets a single kernel launch process the whole
//   batch -- one thread per node, no per-graph branching.
//
// kernels.cu reuses Dataset + GnnModel + these declarations.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "gnn.h"   // GNN_F, GNN_T, gnn_linear_relu, gnn_dot, gnn_sigmoid

// ---------------------------------------------------------------------------
// Dataset: a batch of D drug graphs and P protein targets, in CSR form.
//   Sizes:  D drugs, P proteins, total_nodes atoms, total_edges directed edges.
//   All feature vectors are length GNN_F and normalized to a small range so the
//   fixed-weight network produces well-scaled logits.
// ---------------------------------------------------------------------------
struct Dataset {
    int D = 0;            // number of drug molecules (graphs)
    int P = 0;            // number of protein targets
    int total_nodes = 0;  // sum of atoms over all drugs
    int F = GNN_F;        // feature width (mirrors the model; kept for clarity)

    std::vector<int>   node_off;   // [D+1] node_off[d] = first global node of drug d
    std::vector<int>   adj_off;    // [total_nodes+1] CSR row pointers into adj
    std::vector<int>   adj;        // [total_edges] neighbour GLOBAL node indices
    std::vector<float> feat;       // [total_nodes*F] initial node features, row-major
    std::vector<float> prot;       // [P*F] protein descriptor vectors, row-major

    // Ground-truth implanted "true" interaction (for interpretability only):
    // drug true_drug is engineered to bind protein true_prot most strongly.
    int true_drug = 0;
    int true_prot = 0;
};

// ---------------------------------------------------------------------------
// GnnModel: the fixed (UNTRAINED, seeded) weights of the message-passing net.
//   W[t]    : the F x F linear layer for round t (row-major), t in [0,GNN_T).
//   bias[t] : the length-F bias for round t.
//   Wp / bp : the F x F protein projection + bias (encode protein descriptor
//             into the shared embedding space).
// Stored as flat vectors so they copy to the GPU as-is (contiguous).
// ---------------------------------------------------------------------------
struct GnnModel {
    std::vector<float> W;     // [GNN_T * F * F]
    std::vector<float> bias;  // [GNN_T * F]
    std::vector<float> Wp;    // [F * F]
    std::vector<float> bp;    // [F]
};

// Load the batched-graph sample (see data/README.md for the text format).
Dataset load_dataset(const std::string& path);

// Deterministically construct the fixed weights (same on every run / machine).
GnnModel build_model();

// ---------------------------------------------------------------------------
// CPU reference: run the full forward pass on the host (the trusted baseline).
//   emb   : OUT [D*F] drug embeddings after T message-passing rounds + pooling.
//   score : OUT [D*P] DTI probabilities, row-major (drug-major).
// Both are filled with the SAME math the GPU uses (gnn.h), so main.cu can verify
// the GPU output against these within a tiny tolerance.
// ---------------------------------------------------------------------------
void dti_cpu(const Dataset& d, const GnnModel& m,
             std::vector<float>& emb, std::vector<float>& score);
