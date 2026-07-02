// ===========================================================================
// src/kernels.cu  --  DTI-GNN GPU kernels (gather message-pass + pair scoring)
// ---------------------------------------------------------------------------
// Project 7.2 : Drug-Target Interaction Prediction (GNN)
//
// GPU twin of dti_cpu(): identical per-element math (gnn.h) so the results
// match. Four kernels implement the forward pass (see kernels.cuh for the
// pattern overview); the host wrapper dti_gpu() wires them together. main.cu
// verifies GPU vs CPU. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 256 threads/block: multiple of the 32-lane warp, 8 warps to hide latency, good
// occupancy on sm_75..sm_89 (same default reasoning as the flagships).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CONSTANT MEMORY for the (tiny, read-only, shared-by-every-thread) weights.
//   Constant memory is a 64 KB region with a broadcast cache: when all threads
//   in a warp read the SAME address (which they do here -- every node uses the
//   same W/bias), the read is served in a single transaction. That is exactly
//   the access pattern of a weight-tied GNN layer, so constant memory is ideal
//   (same idea as the Tanimoto query in flagship 1.12).
//
//   Layout mirrors GnnModel: W (GNN_T*F*F) | bias (GNN_T*F) | Wp (F*F) | bp (F).
//   All sizes are compile-time constants (GNN_F, GNN_T) so this fits statically.
// ---------------------------------------------------------------------------
__constant__ float c_W[GNN_T * GNN_F * GNN_F];   // per-round F x F linear weights
__constant__ float c_bias[GNN_T * GNN_F];        // per-round length-F biases
__constant__ float c_Wp[GNN_F * GNN_F];          // protein projection F x F
__constant__ float c_bp[GNN_F];                  // protein projection bias

// ---------------------------------------------------------------------------
// message_pass_kernel: one round, one thread per node. GATHER over CSR edges.
//   grid  = ceil(total_nodes / THREADS_PER_BLOCK)
//   block = THREADS_PER_BLOCK
//   thread i owns node i; writes feat_out[i*F..], reads feat_in of neighbours.
//   No atomics: node i is the sole writer of its output row.
// ---------------------------------------------------------------------------
__global__ void message_pass_kernel(const float* __restrict__ feat_in,
                                     float* __restrict__ feat_out,
                                     const int* __restrict__ adj_off,
                                     const int* __restrict__ adj,
                                     int total_nodes, int round) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_nodes) return;                 // guard the ragged last block

    // AGGREGATE: sum neighbour feature vectors (self-loop included at load time).
    // We accumulate into registers (msg[]) in the SAME fixed CSR order the CPU
    // uses, so the float sums match bit-for-bit up to FMA reordering.
    float msg[GNN_F];
    #pragma unroll
    for (int c = 0; c < GNN_F; ++c) msg[c] = 0.0f;
    const int begin = adj_off[i];
    const int end   = adj_off[i + 1];
    for (int e = begin; e < end; ++e) {
        const float* nf = feat_in + static_cast<std::size_t>(adj[e]) * GNN_F;
        #pragma unroll
        for (int c = 0; c < GNN_F; ++c) msg[c] += nf[c];
    }

    // TRANSFORM: shared linear layer + ReLU, weights from constant memory. Same
    // gnn_linear_relu math as the CPU, just reading c_W/c_bias for this round.
    const float* Wt = c_W + static_cast<std::size_t>(round) * GNN_F * GNN_F;
    const float* bt = c_bias + static_cast<std::size_t>(round) * GNN_F;
    float out[GNN_F];
    gnn_linear_relu(msg, Wt, bt, out);

    float* dst = feat_out + static_cast<std::size_t>(i) * GNN_F;
    #pragma unroll
    for (int c = 0; c < GNN_F; ++c) dst[c] = out[c];
}

// ---------------------------------------------------------------------------
// pool_kernel: one thread per drug sums its nodes' features (graph readout).
// Node range [node_off[drug], node_off[drug+1]) summed in index order (matches
// the CPU), so the reduction is deterministic without atomics.
// ---------------------------------------------------------------------------
__global__ void pool_kernel(const float* __restrict__ feat,
                            const int* __restrict__ node_off,
                            float* __restrict__ emb, int D) {
    const int drug = blockIdx.x * blockDim.x + threadIdx.x;
    if (drug >= D) return;

    float acc[GNN_F];
    #pragma unroll
    for (int c = 0; c < GNN_F; ++c) acc[c] = 0.0f;
    for (int i = node_off[drug]; i < node_off[drug + 1]; ++i) {
        const float* nf = feat + static_cast<std::size_t>(i) * GNN_F;
        #pragma unroll
        for (int c = 0; c < GNN_F; ++c) acc[c] += nf[c];
    }
    float* e = emb + static_cast<std::size_t>(drug) * GNN_F;
    #pragma unroll
    for (int c = 0; c < GNN_F; ++c) e[c] = acc[c];
}

// ---------------------------------------------------------------------------
// protein_encode_kernel: one thread per protein, linear+ReLU via constant Wp/bp.
// ---------------------------------------------------------------------------
__global__ void protein_encode_kernel(const float* __restrict__ prot,
                                       float* __restrict__ pemb, int P) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    gnn_linear_relu(prot + static_cast<std::size_t>(p) * GNN_F, c_Wp, c_bp,
                    pemb + static_cast<std::size_t>(p) * GNN_F);
}

// ---------------------------------------------------------------------------
// score_kernel: one thread per (drug, protein) pair -> DTI probability.
// Flattening the 2-D pair grid into 1-D (j = drug*P + p) lets us launch a single
// 1-D grid over all D*P pairs -- the "independent jobs" pattern.
// ---------------------------------------------------------------------------
__global__ void score_kernel(const float* __restrict__ emb,
                             const float* __restrict__ pemb,
                             float* __restrict__ score, int D, int P) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D * P) return;
    const int drug = j / P;                       // integer div -> drug row
    const int p    = j % P;                        // remainder    -> protein col
    const float logit = gnn_dot(emb + static_cast<std::size_t>(drug) * GNN_F,
                                pemb + static_cast<std::size_t>(p) * GNN_F);
    score[j] = gnn_sigmoid(logit);
}

// ---------------------------------------------------------------------------
// dti_gpu: host wrapper. Uploads graph + weights, runs the four kernels, brings
// back embeddings + scores. Times all kernels (excluding copies) with events.
// ---------------------------------------------------------------------------
void dti_gpu(const Dataset& d, const GnnModel& m,
             std::vector<float>& emb, std::vector<float>& score, float* kernel_ms) {
    const int F = GNN_F;
    const std::size_t node_feat_bytes = static_cast<std::size_t>(d.total_nodes) * F * sizeof(float);

    emb.assign(static_cast<std::size_t>(d.D) * F, 0.0f);
    score.assign(static_cast<std::size_t>(d.D) * d.P, 0.0f);

    // ---- Upload the fixed weights into constant memory ---------------------
    // cudaMemcpyToSymbol copies from host memory into a __constant__ array. The
    // sizes match the GnnModel layout exactly (checked implicitly by the counts).
    CUDA_CHECK(cudaMemcpyToSymbol(c_W,    m.W.data(),    m.W.size()    * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bias, m.bias.data(), m.bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Wp,   m.Wp.data(),   m.Wp.size()   * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bp,   m.bp.data(),   m.bp.size()   * sizeof(float)));

    // ---- Device buffers ----------------------------------------------------
    // Two node-feature buffers for double-buffered (ping-pong) message passing.
    float *d_featA = nullptr, *d_featB = nullptr, *d_emb = nullptr;
    float *d_prot = nullptr, *d_pemb = nullptr, *d_score = nullptr;
    int   *d_adj_off = nullptr, *d_adj = nullptr, *d_node_off = nullptr;

    CUDA_CHECK(cudaMalloc(&d_featA, node_feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_featB, node_feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_emb,   static_cast<std::size_t>(d.D) * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prot,  d.prot.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pemb,  static_cast<std::size_t>(d.P) * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_score, static_cast<std::size_t>(d.D) * d.P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_adj_off,  d.adj_off.size()  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj,      d.adj.size()      * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_off, d.node_off.size() * sizeof(int)));

    // ---- Upload graph + inputs (H2D) ---------------------------------------
    CUDA_CHECK(cudaMemcpy(d_featA, d.feat.data(), node_feat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prot, d.prot.data(), d.prot.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_off, d.adj_off.data(), d.adj_off.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj, d.adj.data(), d.adj.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_off, d.node_off.data(), d.node_off.size() * sizeof(int), cudaMemcpyHostToDevice));

    // ---- Launch configs ----------------------------------------------------
    const int grid_nodes  = (d.total_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int grid_drugs  = (d.D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int grid_prot   = (d.P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int grid_pairs  = (d.D * d.P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();

    // (A) GNN_T rounds of message passing, ping-ponging featA <-> featB. `src`
    // holds the current features; each round writes into `dst`, then we swap.
    float* src = d_featA;
    float* dst = d_featB;
    for (int t = 0; t < GNN_T; ++t) {
        message_pass_kernel<<<grid_nodes, THREADS_PER_BLOCK>>>(src, dst, d_adj_off, d_adj,
                                                               d.total_nodes, t);
        float* tmp = src; src = dst; dst = tmp;    // swap: next round reads `src`
    }
    // After the loop, `src` points at the final node features.

    // (B) Pool -> drug embeddings.
    pool_kernel<<<grid_drugs, THREADS_PER_BLOCK>>>(src, d_node_off, d_emb, d.D);
    // (C) Encode proteins.
    protein_encode_kernel<<<grid_prot, THREADS_PER_BLOCK>>>(d_prot, d_pemb, d.P);
    // (D) Score all pairs.
    score_kernel<<<grid_pairs, THREADS_PER_BLOCK>>>(d_emb, d_pemb, d_score, d.D, d.P);

    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("dti kernels");

    // ---- Bring results back (D2H) ------------------------------------------
    CUDA_CHECK(cudaMemcpy(emb.data(), d_emb,
                          static_cast<std::size_t>(d.D) * F * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(score.data(), d_score,
                          static_cast<std::size_t>(d.D) * d.P * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Free ---------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_featA));
    CUDA_CHECK(cudaFree(d_featB));
    CUDA_CHECK(cudaFree(d_emb));
    CUDA_CHECK(cudaFree(d_prot));
    CUDA_CHECK(cudaFree(d_pemb));
    CUDA_CHECK(cudaFree(d_score));
    CUDA_CHECK(cudaFree(d_adj_off));
    CUDA_CHECK(cudaFree(d_adj));
    CUDA_CHECK(cudaFree(d_node_off));
}
