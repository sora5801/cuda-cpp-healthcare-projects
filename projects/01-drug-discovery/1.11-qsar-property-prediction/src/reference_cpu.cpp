// ===========================================================================
// src/reference_cpu.cpp  --  Loader, fixed weights, serial GCN reference
// ---------------------------------------------------------------------------
// Project 1.11 : QSAR / Property Prediction
// Compiled by the host compiler only. The per-node math lives in gcn.h and is
// reused VERBATIM by the GPU kernels (kernels.cu) so CPU and GPU agree.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_graph: parse the batched-CSR text format (see data/README.md).
//   Header line:   num_mols num_nodes num_edges
//   then num_nodes lines: GCN_F_IN feature floats per atom,
//   then num_mols  lines: atom_count for each molecule (must sum to num_nodes),
//   then num_edges lines: "u v" undirected bonds (global node indices).
//   We BUILD the CSR here: add a self-loop to every node, append each bond in
//   BOTH directions, then flatten so neighbor order is stable and identical to
//   what the GPU will read.
// ---------------------------------------------------------------------------
Graph load_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open graph file: " + path);

    Graph g;
    int num_edges = 0;
    if (!(in >> g.num_mols >> g.num_nodes >> num_edges) ||
        g.num_mols <= 0 || g.num_nodes <= 0 || num_edges < 0)
        throw std::runtime_error("bad header (expected 'num_mols num_nodes num_edges') in " + path);

    // --- atom features --------------------------------------------------
    g.feat.resize(static_cast<std::size_t>(g.num_nodes) * GCN_F_IN);
    for (std::size_t i = 0; i < g.feat.size(); ++i)
        if (!(in >> g.feat[i])) throw std::runtime_error("features truncated in " + path);

    // --- per-molecule atom counts -> mol_start prefix sum ---------------
    g.mol_start.assign(g.num_mols + 1, 0);
    for (int m = 0; m < g.num_mols; ++m) {
        int cnt = 0;
        if (!(in >> cnt) || cnt <= 0) throw std::runtime_error("bad atom count in " + path);
        g.mol_start[m + 1] = g.mol_start[m] + cnt;
    }
    if (g.mol_start[g.num_mols] != g.num_nodes)
        throw std::runtime_error("atom counts do not sum to num_nodes in " + path);

    // --- read bonds into per-node adjacency lists -----------------------
    // adj[i] starts with the SELF-LOOP i, then gains both directions of each
    // bond. Building it as a vector-of-vectors keeps neighbor order = insertion
    // order, which we then flatten to CSR. (For millions of edges you would do
    // a counting-sort CSR build; this readable version suffices for the demo.)
    std::vector<std::vector<int>> adj(g.num_nodes);
    for (int i = 0; i < g.num_nodes; ++i) adj[i].push_back(i);   // self-loop
    for (int e = 0; e < num_edges; ++e) {
        int u = 0, v = 0;
        if (!(in >> u >> v)) throw std::runtime_error("edge list truncated in " + path);
        if (u < 0 || u >= g.num_nodes || v < 0 || v >= g.num_nodes)
            throw std::runtime_error("edge index out of range in " + path);
        adj[u].push_back(v);
        if (u != v) adj[v].push_back(u);                          // undirected
    }

    // --- flatten adjacency -> CSR (row_ptr, col_idx) + degrees ----------
    g.row_ptr.assign(g.num_nodes + 1, 0);
    for (int i = 0; i < g.num_nodes; ++i)
        g.row_ptr[i + 1] = g.row_ptr[i] + static_cast<int>(adj[i].size());
    g.col_idx.resize(g.row_ptr[g.num_nodes]);
    g.deg.assign(g.num_nodes, 0);
    for (int i = 0; i < g.num_nodes; ++i) {
        g.deg[i] = static_cast<int>(adj[i].size());               // incl. self-loop
        const int base = g.row_ptr[i];
        for (std::size_t t = 0; t < adj[i].size(); ++t)
            g.col_idx[base + static_cast<int>(t)] = adj[i][t];
    }
    return g;
}

// ---------------------------------------------------------------------------
// next_weight: one draw from a tiny linear congruential generator (LCG).
//   We GENERATE the inference weights instead of committing a weights file, so
//   the demo is self-contained AND reproducible. Numerical Recipes constants;
//   the top bits are mapped to [0,1) then shifted to [-0.5, 0.5) so that the
//   pre-activations stay in a sane range (ReLU does not zero everything).
// ---------------------------------------------------------------------------
static float next_weight(unsigned int& state) {
    state = 1664525u * state + 1013904223u;
    const float u = static_cast<float>(state >> 8) / 16777216.0f;   // [0,1)
    return (u - 0.5f);
}

// ---------------------------------------------------------------------------
// make_model: build DETERMINISTIC, fixed inference weights from a fixed seed.
//   A trained QSAR model would load weights from disk; here the exact values are
//   unimportant -- what matters is that CPU and GPU use the SAME numbers and that
//   they are stable across runs (so expected_output is fixed). These weights are
//   clearly NOT trained on real bioactivity; the predicted "property" is a
//   synthetic demonstration number, never a real ADMET score (see data/README).
// ---------------------------------------------------------------------------
Model make_model() {
    Model m;
    unsigned int s = 12345u;                            // fixed seed -> fixed weights
    auto fill = [&](std::vector<float>& v, int n) {
        v.resize(n);
        for (int i = 0; i < n; ++i) v[i] = next_weight(s);
    };
    fill(m.W1, GCN_F_IN  * GCN_F_HID);  fill(m.b1, GCN_F_HID);
    fill(m.W2, GCN_F_HID * GCN_F_OUT);  fill(m.b2, GCN_F_OUT);
    fill(m.head_w, GCN_F_OUT);
    m.head_b = next_weight(s);
    return m;
}

// ---------------------------------------------------------------------------
// gcn_predict_cpu: the trusted serial reference. The layers run over the WHOLE
//   batch at once because a node's neighbors live in the batch-global CSR.
//
//   Pipeline, mirroring the GPU exactly:
//     H0 = feat                         [num_nodes x F_IN]
//     H1 = ReLU( gcn_layer(H0, W1,b1) ) [num_nodes x F_HID]
//     H2 =       gcn_layer(H1, W2,b2)   [num_nodes x F_OUT]   (no ReLU)
//     pred[m] = head( mean_{a in mol m} H2[a,:] )
//   Each gcn_layer call uses gcn_aggregate_then_transform per node (gcn.h).
// ---------------------------------------------------------------------------
void gcn_predict_cpu(const Graph& g, const Model& m, std::vector<float>& pred) {
    std::vector<float> H1(static_cast<std::size_t>(g.num_nodes) * GCN_F_HID);
    std::vector<float> H2(static_cast<std::size_t>(g.num_nodes) * GCN_F_OUT);

    // --- Layer 1: F_IN -> F_HID, ReLU -----------------------------------
    for (int i = 0; i < g.num_nodes; ++i) {
        const int* nbr = &g.col_idx[g.row_ptr[i]];          // node i's neighbors
        gcn_aggregate_then_transform(
            i, g.feat.data(), g.deg.data(), nbr, g.deg[i], g.deg[i],
            m.W1.data(), m.b1.data(), GCN_F_IN, GCN_F_HID, /*relu=*/true,
            &H1[static_cast<std::size_t>(i) * GCN_F_HID]);
    }

    // --- Layer 2: F_HID -> F_OUT, no ReLU -------------------------------
    for (int i = 0; i < g.num_nodes; ++i) {
        const int* nbr = &g.col_idx[g.row_ptr[i]];
        gcn_aggregate_then_transform(
            i, H1.data(), g.deg.data(), nbr, g.deg[i], g.deg[i],
            m.W2.data(), m.b2.data(), GCN_F_HID, GCN_F_OUT, /*relu=*/false,
            &H2[static_cast<std::size_t>(i) * GCN_F_OUT]);
    }

    // --- Readout + head: one scalar per molecule ------------------------
    pred.resize(g.num_mols);
    for (int mm = 0; mm < g.num_mols; ++mm) {
        const int start = g.mol_start[mm];
        const int n     = g.atoms_in(mm);
        pred[mm] = gcn_readout_head(&H2[static_cast<std::size_t>(start) * GCN_F_OUT],
                                    n, m.head_w.data(), m.head_b);
    }
}
