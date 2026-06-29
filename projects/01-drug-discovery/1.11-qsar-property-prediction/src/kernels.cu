// ===========================================================================
// src/kernels.cu  --  GCN layer kernels + readout + host wrapper
// ---------------------------------------------------------------------------
// Project 1.11 : QSAR / Property Prediction
//
// GPU twin of gcn_predict_cpu(). Three kernels:
//   1. gcn_layer_kernel   -- one thread per NODE: aggregate neighbors + transform
//                            (layer 1: F_IN->F_HID +ReLU; layer 2: F_HID->F_OUT).
//   2. gcn_readout_kernel -- one thread per MOLECULE: mean-pool atoms + head.
// All per-node arithmetic is the shared gcn.h code, so the GPU reproduces the
// CPU bit-for-bit up to fp32 rounding. main.cu compares the predictions.
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "gcn.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cstddef>

// A good occupancy default on sm_75..sm_89 for these light, memory-bound kernels.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// CONSTANT MEMORY for the weights.
//   The weights are tiny (<= 96 floats here) and EVERY thread reads them, but
//   they never change during a launch -- the textbook case for __constant__
//   memory, whose per-SM broadcast cache serves the same address to a whole warp
//   in one transaction. (Same idea as the query in flagship 1.12.) We stage all
//   five tensors back-to-back and hand each kernel the right offset.
//   Layout: [ W1 | b1 | W2 | b2 | head_w ]  (head_b is passed as a scalar arg).
// ---------------------------------------------------------------------------
static const int OFF_W1     = 0;
static const int OFF_B1     = OFF_W1 + GCN_F_IN  * GCN_F_HID;   // after W1
static const int OFF_W2     = OFF_B1 + GCN_F_HID;              // after b1
static const int OFF_B2     = OFF_W2 + GCN_F_HID * GCN_F_OUT;  // after W2
static const int OFF_HEAD_W = OFF_B2 + GCN_F_OUT;             // after b2
static const int WEIGHTS_LEN = OFF_HEAD_W + GCN_F_OUT;        // total floats

__constant__ float c_weights[WEIGHTS_LEN];   // all GCN weights, in constant memory

// ---------------------------------------------------------------------------
// gcn_layer_kernel: compute one GCN layer for every node.
//   grid   : ceil(num_nodes / THREADS_PER_BLOCK) blocks
//   block  : THREADS_PER_BLOCK threads
//   thread (blockIdx.x, threadIdx.x) -> output node  i = bx*blockDim + tx
//   The thread reads node i's neighbor slice from the CSR and calls the shared
//   gcn_aggregate_then_transform (gcn.h), writing its own row of `out` only.
//
//   PARAMS
//     H        : [num_nodes * f_in] input features (layer input).
//     deg      : [num_nodes] degree-with-self-loop.
//     row_ptr  : [num_nodes+1] CSR offsets.   col_idx: neighbor indices.
//     num_nodes: total nodes in the batch.
//     w_off,b_off : offsets of this layer's W and bias inside c_weights.
//     f_in,f_out  : this layer's widths.   relu: apply ReLU?
//     out      : [num_nodes * f_out] output features (layer output).
//
//   No shared memory, no atomics: each thread owns a disjoint output row, so the
//   kernel is race-free and DETERMINISTIC (neighbor order == CSR order == CPU).
// ---------------------------------------------------------------------------
__global__ void gcn_layer_kernel(const float* __restrict__ H,
                                 const int* __restrict__ deg,
                                 const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_idx,
                                 int num_nodes, int w_off, int b_off,
                                 int f_in, int f_out, bool relu,
                                 float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's node
    if (i >= num_nodes) return;                            // guard ragged block

    const int begin = row_ptr[i];
    const int* nbr  = col_idx + begin;                     // node i's neighbor list
    // f_out is at most GCN_F_HID; stage the output row in registers/local mem.
    float row[GCN_F_HID];
    gcn_aggregate_then_transform(
        i, H, deg, nbr, deg[i], deg[i],
        c_weights + w_off, c_weights + b_off, f_in, f_out, relu, row);

    float* dst = out + static_cast<std::size_t>(i) * f_out;
    for (int o = 0; o < f_out; ++o) dst[o] = row[o];
}

// ---------------------------------------------------------------------------
// gcn_readout_kernel: one thread per MOLECULE.
//   grid   : ceil(num_mols / THREADS_PER_BLOCK)
//   thread -> molecule m. It mean-pools m's atom embeddings and applies the
//   linear head via the shared gcn_readout_head (gcn.h). Writes pred[m] only.
//
//   PARAMS
//     H2       : [num_nodes * F_OUT] final node embeddings (layer-2 output).
//     mol_start: [num_mols+1] node ranges per molecule.
//     num_mols : number of molecules.   head_b: scalar head bias.
//     pred     : [num_mols] output predictions.
// ---------------------------------------------------------------------------
__global__ void gcn_readout_kernel(const float* __restrict__ H2,
                                   const int* __restrict__ mol_start,
                                   int num_mols, float head_b,
                                   float* __restrict__ pred) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's molecule
    if (m >= num_mols) return;

    const int start = mol_start[m];
    const int n     = mol_start[m + 1] - start;            // atoms in molecule m
    pred[m] = gcn_readout_head(H2 + static_cast<std::size_t>(start) * GCN_F_OUT,
                               n, c_weights + OFF_HEAD_W, head_b);
}

// ---------------------------------------------------------------------------
// gcn_predict_gpu: upload everything once, launch the three kernels, copy back.
//   Mirrors gcn_predict_cpu(): same layers, same readout, same CSR -> same math.
// ---------------------------------------------------------------------------
void gcn_predict_gpu(const Graph& g, const Model& m,
                     std::vector<float>& pred, float* kernel_ms) {
    const int N = g.num_nodes, M = g.num_mols;

    // --- pack the weights contiguously and push to constant memory ------
    std::vector<float> packed(WEIGHTS_LEN);
    auto put = [&](int off, const std::vector<float>& v) {
        for (std::size_t i = 0; i < v.size(); ++i) packed[off + static_cast<int>(i)] = v[i];
    };
    put(OFF_W1, m.W1); put(OFF_B1, m.b1);
    put(OFF_W2, m.W2); put(OFF_B2, m.b2);
    put(OFF_HEAD_W, m.head_w);
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, packed.data(),
                                  WEIGHTS_LEN * sizeof(float)));

    // --- device buffers -------------------------------------------------
    float *d_feat = nullptr, *d_H1 = nullptr, *d_H2 = nullptr, *d_pred = nullptr;
    int   *d_deg = nullptr, *d_row = nullptr, *d_col = nullptr, *d_molstart = nullptr;
    CUDA_CHECK(cudaMalloc(&d_feat, g.feat.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H1, static_cast<std::size_t>(N) * GCN_F_HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H2, static_cast<std::size_t>(N) * GCN_F_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pred, static_cast<std::size_t>(M) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_deg, g.deg.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row, g.row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, g.col_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_molstart, g.mol_start.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_feat, g.feat.data(), g.feat.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_deg, g.deg.data(), g.deg.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row, g.row_ptr.data(), g.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, g.col_idx.data(), g.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_molstart, g.mol_start.data(), g.mol_start.size() * sizeof(int), cudaMemcpyHostToDevice));

    const int node_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int mol_grid  = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    // Layer 1: features -> hidden, with ReLU.
    gcn_layer_kernel<<<node_grid, THREADS_PER_BLOCK>>>(
        d_feat, d_deg, d_row, d_col, N, OFF_W1, OFF_B1,
        GCN_F_IN, GCN_F_HID, /*relu=*/true, d_H1);
    // Layer 2: hidden -> embedding, no ReLU.
    gcn_layer_kernel<<<node_grid, THREADS_PER_BLOCK>>>(
        d_H1, d_deg, d_row, d_col, N, OFF_W2, OFF_B2,
        GCN_F_HID, GCN_F_OUT, /*relu=*/false, d_H2);
    // Readout: mean-pool atoms + head -> one prediction per molecule.
    gcn_readout_kernel<<<mol_grid, THREADS_PER_BLOCK>>>(
        d_H2, d_molstart, M, m.head_b, d_pred);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("gcn kernels");

    pred.resize(M);
    CUDA_CHECK(cudaMemcpy(pred.data(), d_pred, static_cast<std::size_t>(M) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_feat));  CUDA_CHECK(cudaFree(d_H1));
    CUDA_CHECK(cudaFree(d_H2));    CUDA_CHECK(cudaFree(d_pred));
    CUDA_CHECK(cudaFree(d_deg));   CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_col));   CUDA_CHECK(cudaFree(d_molstart));
}
