// ===========================================================================
// src/reference_cpu.cpp  --  Loader, fixed-weight model, serial DTI reference
// ---------------------------------------------------------------------------
// Project 7.2 : Drug-Target Interaction Prediction (GNN)
// Compiled by the host compiler only. Per-element math lives in gnn.h so this
// serial baseline runs the SAME arithmetic as the GPU kernels (verification is
// then near-exact). This file is the trusted reference the GPU is checked against.
// ===========================================================================
#include "reference_cpu.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_dataset: read the batched-graph text sample. Format (data/README.md):
//
//   D P                         # drugs, proteins
//   true_drug true_prot         # implanted ground-truth interaction (indices)
//   <for each drug d in 0..D-1>
//     n_d k_d                   # atoms in drug d, then k_d = number of edges
//     n_d rows of F floats      # initial atom features (length GNN_F each)
//     k_d rows of "u v"         # UNDIRECTED bonds (local atom indices 0..n_d-1)
//   <for each protein p in 0..P-1>
//     F floats                  # protein descriptor vector (length GNN_F)
//
//   Undirected bonds are expanded into BOTH directions in the CSR adjacency, and
//   we add a SELF-LOOP on every node (standard GNN trick so a node keeps its own
//   feature during aggregation). Throws on any malformed input so demos fail
//   loudly rather than silently producing garbage.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    Dataset d;
    if (!(in >> d.D >> d.P) || d.D <= 0 || d.P <= 0)
        throw std::runtime_error("bad header (expected 'D P') in " + path);
    if (!(in >> d.true_drug >> d.true_prot) ||
        d.true_drug < 0 || d.true_drug >= d.D ||
        d.true_prot < 0 || d.true_prot >= d.P)
        throw std::runtime_error("bad ground-truth line (expected 'true_drug true_prot') in " + path);

    d.node_off.assign(1, 0);                 // node_off[0] = 0; grows per drug
    // Per-node neighbour lists, built first as vectors then flattened to CSR.
    std::vector<std::vector<int>> nbr;       // nbr[global_node] = neighbours

    for (int drug = 0; drug < d.D; ++drug) {
        int n = 0, k = 0;
        if (!(in >> n >> k) || n <= 0 || k < 0)
            throw std::runtime_error("bad drug header (expected 'n_atoms n_edges') in " + path);
        const int base = d.total_nodes;      // first global index of this drug's atoms
        d.total_nodes += n;
        d.node_off.push_back(d.total_nodes);

        // Read this drug's atom features into the flat feature array.
        for (int a = 0; a < n; ++a) {
            for (int f = 0; f < d.F; ++f) {
                float v;
                if (!(in >> v)) throw std::runtime_error("feature data truncated in " + path);
                d.feat.push_back(v);
            }
            nbr.emplace_back();              // empty neighbour list for this node
            nbr.back().push_back(base + a);  // SELF-LOOP: node keeps its own feature
        }
        // Read this drug's bonds (LOCAL indices) and add both directions.
        for (int e = 0; e < k; ++e) {
            int u, v;
            if (!(in >> u >> v) || u < 0 || u >= n || v < 0 || v >= n)
                throw std::runtime_error("bad edge (local indices out of range) in " + path);
            nbr[base + u].push_back(base + v);
            nbr[base + v].push_back(base + u);
        }
    }

    // Read protein descriptor vectors.
    d.prot.resize(static_cast<std::size_t>(d.P) * d.F);
    for (std::size_t i = 0; i < d.prot.size(); ++i)
        if (!(in >> d.prot[i])) throw std::runtime_error("protein data truncated in " + path);

    // Flatten the per-node neighbour lists into CSR (adj_off + adj).
    d.adj_off.assign(static_cast<std::size_t>(d.total_nodes) + 1, 0);
    for (int i = 0; i < d.total_nodes; ++i)
        d.adj_off[i + 1] = d.adj_off[i] + static_cast<int>(nbr[i].size());
    d.adj.reserve(d.adj_off.back());
    for (int i = 0; i < d.total_nodes; ++i)
        for (int j : nbr[i]) d.adj.push_back(j);

    return d;
}

// ---------------------------------------------------------------------------
// next_weight: advance a 64-bit linear congruential generator (LCG) and map its
// high bits to a float in [-0.5, 0.5). Integer/bit ops only -> identical on
// every machine, giving reproducible fixed weights (see build_model).
// ---------------------------------------------------------------------------
static float next_weight(std::uint64_t& state) {
    // Numerical Recipes LCG constants; state advances deterministically.
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    // Take the top 24 bits -> a uniform integer in [0, 2^24), then map to
    // [-0.5, 0.5). Using the high bits avoids the LCG's weak low-order bits.
    const std::uint32_t bits = static_cast<std::uint32_t>(state >> 40);
    return (static_cast<float>(bits) / static_cast<float>(1u << 24)) - 0.5f;
}

// ---------------------------------------------------------------------------
// build_model: deterministic FIXED weights. Every weight comes from the LCG
// above seeded with a constant, so the network is IDENTICAL on every run and
// machine -- the property we need for a byte-reproducible demo. These weights
// are NOT trained (see gnn.h "HONESTY"); they exercise inference machinery only.
// ---------------------------------------------------------------------------
GnnModel build_model() {
    GnnModel m;
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;   // fixed seed (golden-ratio constant)
    const int F = GNN_F;

    m.W.resize(static_cast<std::size_t>(GNN_T) * F * F);
    m.bias.resize(static_cast<std::size_t>(GNN_T) * F);
    for (auto& w : m.W)    w = next_weight(state);
    for (auto& b : m.bias) b = next_weight(state);

    m.Wp.resize(static_cast<std::size_t>(F) * F);
    m.bp.resize(F);
    for (auto& w : m.Wp) w = next_weight(state);
    for (auto& b : m.bp) b = next_weight(state);
    return m;
}

// ---------------------------------------------------------------------------
// dti_cpu: the serial forward pass. For each drug graph it runs GNN_T rounds of
// message passing over the CSR adjacency, sum-pools to a drug embedding, encodes
// each protein once, then scores every drug x protein pair. This is deliberately
// straightforward so it is obviously correct; the GPU kernels reproduce it.
// ---------------------------------------------------------------------------
void dti_cpu(const Dataset& d, const GnnModel& m,
             std::vector<float>& emb, std::vector<float>& score) {
    const int F = GNN_F;

    // ---- 1. Message passing per node (double-buffered feature arrays) -------
    // cur/nxt hold node features for the current/next round. We start from the
    // input features and alternate. Working in a local copy keeps the Dataset
    // input immutable (so we can re-run / verify).
    std::vector<float> cur = d.feat;                        // [total_nodes*F]
    std::vector<float> nxt(cur.size());
    float msg[GNN_F];                                       // aggregated neighbour sum
    float out[GNN_F];                                       // linear+ReLU result

    for (int t = 0; t < GNN_T; ++t) {
        const float* Wt = &m.W[static_cast<std::size_t>(t) * F * F];
        const float* bt = &m.bias[static_cast<std::size_t>(t) * F];
        for (int i = 0; i < d.total_nodes; ++i) {
            // AGGREGATE: sum the feature vectors of node i's neighbours (the CSR
            // row adj[adj_off[i]..adj_off[i+1])). Self-loop was added at load so
            // node i's own feature is included. Sum in a fixed order for
            // determinism (matches the GPU thread's loop order exactly).
            for (int c = 0; c < F; ++c) msg[c] = 0.0f;
            for (int e = d.adj_off[i]; e < d.adj_off[i + 1]; ++e) {
                const float* nf = &cur[static_cast<std::size_t>(d.adj[e]) * F];
                for (int c = 0; c < F; ++c) msg[c] += nf[c];
            }
            // TRANSFORM: shared linear layer + ReLU (gnn.h -- same as GPU).
            gnn_linear_relu(msg, Wt, bt, out);
            for (int c = 0; c < F; ++c) nxt[static_cast<std::size_t>(i) * F + c] = out[c];
        }
        cur.swap(nxt);                                      // next round reads cur
    }

    // ---- 2. Graph-level sum pooling -> one embedding per drug ---------------
    emb.assign(static_cast<std::size_t>(d.D) * F, 0.0f);
    for (int drug = 0; drug < d.D; ++drug) {
        float* e = &emb[static_cast<std::size_t>(drug) * F];
        for (int i = d.node_off[drug]; i < d.node_off[drug + 1]; ++i) {
            const float* nf = &cur[static_cast<std::size_t>(i) * F];
            for (int c = 0; c < F; ++c) e[c] += nf[c];      // readout = sum of nodes
        }
    }

    // ---- 3. Encode each protein once into the shared embedding space --------
    std::vector<float> pemb(static_cast<std::size_t>(d.P) * F);
    for (int p = 0; p < d.P; ++p) {
        gnn_linear_relu(&d.prot[static_cast<std::size_t>(p) * F], m.Wp.data(), m.bp.data(),
                        &pemb[static_cast<std::size_t>(p) * F]);
    }

    // ---- 4. Score every drug x protein pair (dense D x P matrix) ------------
    score.assign(static_cast<std::size_t>(d.D) * d.P, 0.0f);
    for (int drug = 0; drug < d.D; ++drug)
        for (int p = 0; p < d.P; ++p) {
            const float logit = gnn_dot(&emb[static_cast<std::size_t>(drug) * F],
                                        &pemb[static_cast<std::size_t>(p) * F]);
            score[static_cast<std::size_t>(drug) * d.P + p] = gnn_sigmoid(logit);
        }
}
