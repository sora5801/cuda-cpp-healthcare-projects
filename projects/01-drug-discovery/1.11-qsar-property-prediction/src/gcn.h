// ===========================================================================
// src/gcn.h  --  Shared (host + device) Graph Convolutional Network math
// ---------------------------------------------------------------------------
// Project 1.11 : QSAR / Property Prediction
//
// WHAT THIS PROJECT COMPUTES
//   A QSAR ("quantitative structure-activity relationship") model predicts a
//   molecular PROPERTY (here a single regression target -- think a normalized,
//   synthetic ADMET-style score) directly from a molecule's GRAPH:
//     * atoms   -> graph NODES, each carrying a feature vector (its "type" etc.)
//     * bonds   -> graph EDGES (undirected; we store both directions)
//   The modern way to do this is a MESSAGE-PASSING NEURAL NETWORK (MPNN). The
//   simplest member of that family -- and the one we implement -- is the GRAPH
//   CONVOLUTIONAL NETWORK (GCN) of Kipf & Welling (2017). One GCN layer does:
//
//       H'  =  ReLU( D^{-1/2} (A + I) D^{-1/2}  H  W )
//
//   Reading that right-to-left and per-node:
//     1. each node mixes its OWN feature row with its NEIGHBORS' rows
//        (the "A + I" adds a self-loop so a node keeps its own signal),
//     2. weighted by a SYMMETRIC NORMALIZATION  c_{ij}=1/sqrt(deg_i * deg_j)
//        (the D^{-1/2} ... D^{-1/2}; it keeps feature scales stable across
//        nodes of very different degree),
//     3. the aggregated vector is linearly transformed by a learned matrix W,
//     4. and passed through a ReLU nonlinearity.
//   Stacking L such layers lets information flow L hops across the molecule.
//   A final READOUT (mean over a molecule's atoms) + a linear head turns the
//   per-atom embeddings into ONE number per molecule: the predicted property.
//
// WHY A GPU
//   Pharma virtual screens push HUNDREDS OF MILLIONS of molecules through such a
//   model. Every molecule is independent and every atom's neighbor-aggregation
//   is independent, so the work is massively parallel. The bottleneck is the
//   irregular MESSAGE AGGREGATION over many small graphs -- exactly what GPUs
//   batch well. This teaching version runs INFERENCE (fixed, supplied weights)
//   on a small batch; training (backprop) is described in THEORY.md.
//
// THE GPU PATTERN  (see docs/PATTERNS.md)
//   "per-output gather": one thread owns one OUTPUT node and loops over that
//   node's neighbor list (stored in CSR form, gcn.h/Graph). Because each thread
//   writes only its own node, there are NO atomics and NO race -- the GPU sums
//   neighbors in the SAME ORDER as the CPU, so the two agree to ~fp32 precision.
//
// DETERMINISM / CPU-GPU PARITY (PATTERNS.md §2)
//   All the per-node arithmetic lives here as GCN_HD (= __host__ __device__)
//   inline functions, so reference_cpu.cpp (host compiler) and kernels.cu (nvcc)
//   run BYTE-IDENTICAL math in the SAME accumulation order. Keep CUDA-only types
//   out of this header so the host compiler can include it.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt
#include <cstddef>   // std::size_t

// GCN_HD expands to __host__ __device__ under nvcc, and to nothing under the
// plain host compiler -- the single-source idiom from PATTERNS.md §2.
#ifdef __CUDACC__
#define GCN_HD __host__ __device__
#else
#define GCN_HD
#endif

// ---------------------------------------------------------------------------
// Fixed network shape (kept small and constant so the demo is legible and the
// weights file is human-readable). A real QSAR MPNN uses far wider layers and
// many more atom features; we keep these tiny on purpose -- the ALGORITHM is the
// lesson, not the capacity.
//   F_IN  : input atom-feature width  (one-hot-ish element/degree descriptors)
//   F_HID : hidden width after layer 1
//   F_OUT : embedding width after layer 2 (pooled, then a linear head -> scalar)
// These are compile-time constants so loops can be #pragma unroll'd and so the
// kernel can stage a feature row in registers. reference_cpu.cpp uses them too.
// ---------------------------------------------------------------------------
static const int GCN_F_IN  = 6;   // atom features: see make_synthetic.py
static const int GCN_F_HID = 8;   // hidden channels
static const int GCN_F_OUT = 4;   // output embedding channels

// ---------------------------------------------------------------------------
// gcn_relu(x): the rectified-linear activation, max(0, x).
//   Branch-free via the ternary so it compiles to a single predicated select on
//   the GPU (no divergent branch). Identical on host and device.
// ---------------------------------------------------------------------------
GCN_HD inline float gcn_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ---------------------------------------------------------------------------
// gcn_norm_coeff(deg_i, deg_j): the symmetric edge weight c_{ij} that appears in
//   D^{-1/2}(A+I)D^{-1/2}. Each degree already INCLUDES the self-loop (so a lone
//   atom has degree 1, not 0, and never divides by zero). Returns
//       1 / sqrt(deg_i * deg_j).
//   We compute the product in double then take a float sqrt's reciprocal to keep
//   host and device identical (both use the same sequence of ops). Small-degree
//   molecules make this matter little, but the habit keeps parity exact.
// ---------------------------------------------------------------------------
GCN_HD inline float gcn_norm_coeff(int deg_i, int deg_j) {
    const float denom = std::sqrt(static_cast<float>(deg_i) * static_cast<float>(deg_j));
    return 1.0f / denom;
}

// ---------------------------------------------------------------------------
// gcn_aggregate_then_transform:
//   Compute ONE output node's new feature row for ONE GCN layer. This is the
//   per-node kernel body, shared verbatim by the CPU loop and the GPU thread.
//
//   The math for output node i and output channel o is:
//       z_o = bias[o] + sum_{j in N(i) U {i}}  c_{ij} * ( sum_k H[j,k] * W[k,o] )
//       out[o] = relu ? ReLU(z_o) : z_o
//   i.e. for each neighbor j (self-loop included by the caller's neighbor list),
//   form the normalized message c_{ij}*H[j,:], project it through W, and sum.
//   We fold the linear map W INTO the aggregation loop so each neighbor row is
//   touched once -- the standard "aggregate-and-combine" fusion.
//
//   PARAMETERS  (all sizes are in elements, all pointers are caller-owned)
//     i         : the output node index (only used conceptually; data comes via
//                 the neighbor list below).
//     H         : [num_nodes * f_in] input features, row-major (node-major).
//     deg       : [num_nodes] degree-with-self-loop of every node (for c_{ij}).
//     nbr       : pointer to THIS node's neighbor list (CSR slice). It MUST
//                 already include the self-loop entry i, so the self term is
//                 handled by the same loop -- no special case, no divergence.
//     nbr_count : how many neighbors (including self) node i has = deg[i].
//     W         : [f_in * f_out] weight matrix, row-major (in-channel-major).
//     bias      : [f_out] bias added once per output channel.
//     f_in,f_out: layer widths.
//     relu      : apply ReLU? (true for hidden layers, false for the last).
//     out       : [f_out] destination for node i's new feature row.
//
//   COMPLEXITY  O(nbr_count * f_in * f_out) per node. Summed over the graph this
//   is O(E * f_in * f_out) (E = edges incl. self-loops) -- the dominant cost.
//
//   DETERMINISM  neighbors are summed in CSR order; the caller builds the SAME
//   CSR for host and device, so the float sum order is identical on both.
// ---------------------------------------------------------------------------
GCN_HD inline void gcn_aggregate_then_transform(int /*i*/,
                                                const float* H, const int* deg,
                                                const int* nbr, int nbr_count,
                                                int deg_i,
                                                const float* W, const float* bias,
                                                int f_in, int f_out, bool relu,
                                                float* out) {
    // Start each output channel at its bias term.
    for (int o = 0; o < f_out; ++o) out[o] = bias[o];

    // Walk this node's neighbor list (self-loop included). For each neighbor j:
    //   * c = symmetric normalization for edge (i,j),
    //   * add  c * (H[j,:] . W[:,o])  to every output channel o.
    for (int t = 0; t < nbr_count; ++t) {
        const int j = nbr[t];                         // neighbor node index
        const float c = gcn_norm_coeff(deg_i, deg[j]);// edge weight c_{ij}
        const float* hj = H + static_cast<std::size_t>(j) * f_in;  // H[j,:]
        // Project this neighbor's (normalized) row through W into all outputs.
        for (int o = 0; o < f_out; ++o) {
            float acc = 0.0f;                          // H[j,:] . W[:,o]
            for (int k = 0; k < f_in; ++k)
                acc += hj[k] * W[static_cast<std::size_t>(k) * f_out + o];
            out[o] += c * acc;                         // accumulate message
        }
    }

    // Optional nonlinearity (on for hidden layers, off for the final embedding).
    if (relu)
        for (int o = 0; o < f_out; ++o) out[o] = gcn_relu(out[o]);
}

// ---------------------------------------------------------------------------
// gcn_readout_head:
//   Turn a molecule's per-atom embeddings into ONE scalar property prediction.
//   READOUT = mean pooling over the molecule's atoms (a permutation-invariant
//   graph-level summary), then a linear HEAD  y = head_w . pooled + head_b.
//
//   PARAMETERS
//     emb        : [n_atoms * F_OUT] this molecule's atom embeddings (layer-2 out).
//     n_atoms    : number of atoms in this molecule.
//     head_w     : [F_OUT] head weights.   head_b : scalar head bias.
//   RETURNS the predicted property (a plain float).
//
//   Shared by CPU and GPU so the pooled value and dot product are summed in the
//   same order on both -> identical to fp32 rounding.
// ---------------------------------------------------------------------------
GCN_HD inline float gcn_readout_head(const float* emb, int n_atoms,
                                     const float* head_w, float head_b) {
    float pooled[GCN_F_OUT];                           // mean over atoms
    for (int o = 0; o < GCN_F_OUT; ++o) pooled[o] = 0.0f;
    for (int a = 0; a < n_atoms; ++a)
        for (int o = 0; o < GCN_F_OUT; ++o)
            pooled[o] += emb[static_cast<std::size_t>(a) * GCN_F_OUT + o];
    const float inv_n = 1.0f / static_cast<float>(n_atoms);
    float y = head_b;
    for (int o = 0; o < GCN_F_OUT; ++o) y += head_w[o] * (pooled[o] * inv_n);
    return y;
}
