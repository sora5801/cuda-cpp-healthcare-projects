// ===========================================================================
// src/kernels.cu  --  Per-node anti-diagonal wavefront kernel + host sweep
// ---------------------------------------------------------------------------
// Project 3.13 : Pangenome Graph Alignment
//
// GPU twin of graph_sw_cpu(): fills the SAME per-node score blocks, but each
// block is swept as an anti-diagonal wavefront (all cells of one diagonal in
// parallel) and the nodes are processed in topological order. The cross-node
// coupling at column j=1 is precomputed on the host into diag_in/left_in (a tiny
// reduction over predecessor last-columns) so the kernel stays regular. main.cu
// runs both paths and asserts every cell matches exactly. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// Diagonals of a tiny node block are short, so a modest block keeps each launch
// cheap; 128 is a multiple of the 32-lane warp and plenty for these sizes.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// graph_sw_diagonal_kernel: one thread fills one cell (i, j) on anti-diagonal d
// of one node's block.  Thread k -> i = i_lo + k, j = d - i.
//   For an interior column (j > 1) the three neighbours are inside this block.
//   For the first content column (j == 1) the diagonal and left neighbours come
//   from the precomputed boundary vectors diag_in[i]/left_in[i] (the max over
//   predecessor last-columns, built on the host); "up" is still inside the block.
//   The cell value is computed by the SHARED cell_score() (reference_cpu.h), so
//   it is bit-identical to the CPU reference.
// ---------------------------------------------------------------------------
__global__ void graph_sw_diagonal_kernel(const uint8_t* __restrict__ q,
                                         const uint8_t* __restrict__ node_seq,
                                         int* __restrict__ H,
                                         int base, int L, int W, int qlen,
                                         const int* __restrict__ diag_in,
                                         const int* __restrict__ left_in,
                                         int d, int i_lo, int count) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;                       // guard the ragged last block

    const int i = i_lo + k;                       // this thread's query row (>= 1)
    const int j = d - i;                          // ... and node column (i + j = d)

    // Gather the three incoming scores.
    int diag, up, left;
    if (j == 1) {
        // Cross-node boundary: predecessors supplied these (host-precomputed).
        diag = diag_in[i];                        // max_u H[u][i-1][Lu]
        left = left_in[i];                        // max_u H[u][i  ][Lu]
        up   = H[base + (i - 1) * W + j];         // "up" stays inside this block
    } else {
        diag = H[base + (i - 1) * W + (j - 1)];   // top-left (diagonal d-2)
        up   = H[base + (i - 1) * W +  j     ];   // top      (diagonal d-1)
        left = H[base +  i      * W + (j - 1)];   // left     (diagonal d-1)
    }

    // is_match compares the query residue at row i with this node's residue at
    // column j (both 0..3 codes). node_seq is THIS node's segment, so column j
    // maps to node_seq[j-1].
    const bool is_match = (q[i - 1] == node_seq[j - 1]);
    H[base + i * W + j] = cell_score(diag, up, left, is_match);   // shared recurrence
}

// ---------------------------------------------------------------------------
// host helper: reduce predecessor last-columns into this node's boundary vectors.
//   For node v and every query row i in [0..qlen]:
//     diag_in[i] = max over predecessors u of  H_host[u][i-1][Lu]   (i>=1, else 0)
//     left_in[i] = max over predecessors u of  H_host[u][i  ][Lu]
//   A source node (no predecessor) leaves both at 0 (a fresh local start). We do
//   this on the HOST, reading back-from the already-finalised predecessor blocks
//   in the host-side mirror dp.H, because predecessors finished in earlier
//   topological steps -- the dependency that makes the graph "sequential between
//   nodes" while each node's interior stays fully parallel.
// ---------------------------------------------------------------------------
static void build_boundaries(const Problem& p, const GraphDP& dp, int v,
                             std::vector<int>& diag_in, std::vector<int>& left_in) {
    const Graph& g = p.graph;
    const int rows = p.qlen + 1;
    diag_in.assign(rows, 0);
    left_in.assign(rows, 0);
    for (int e = g.pred_off[v]; e < g.pred_off[v + 1]; ++e) {
        const int u  = g.pred_idx[e];
        const int Lu = g.seq_len[u];
        const int Wu = Lu + 1;
        const int ub = dp.block_off[u];
        for (int i = 1; i <= p.qlen; ++i) {
            const int dv = dp.H[ub + (i - 1) * Wu + Lu];   // last col, row i-1 -> diag
            const int lv = dp.H[ub +  i      * Wu + Lu];   // last col, row i   -> left
            if (dv > diag_in[i]) diag_in[i] = dv;
            if (lv > left_in[i]) left_in[i] = lv;
        }
    }
}

// ---------------------------------------------------------------------------
// graph_sw_gpu: upload query + all node sequences once, then walk nodes in
// topological order. For each node:
//   (1) build its boundary vectors from finalised predecessor blocks (host),
//   (2) upload them, copy down the predecessor last-columns we need (already on
//       device -- we keep the WHOLE H on device and mirror it on the host so the
//       host reduction can read predecessor results), then
//   (3) sweep the node's block: one kernel launch per anti-diagonal d.
// We time the whole sweep (all nodes, all diagonals) with CUDA events.
//
// HONESTY (THEORY "real world"): a tiny graph issues MANY small launches, so
// launch overhead can make the GPU slower than the CPU here -- the wavefront pays
// off on long reads / big bubbles, and production aligners batch many reads (one
// block per read) and use intra-node tiling. We keep one-launch-per-diagonal
// because it makes the dependency structure unmistakable -- the teaching goal.
// ---------------------------------------------------------------------------
void graph_sw_gpu(const Problem& p, GraphDP& dp, float* kernel_ms) {
    const Graph& g = p.graph;
    layout_blocks(p, dp);                          // host mirror of H (zeroed)
    const std::size_t cells = dp.H.size();

    // ---- Upload the immutable inputs once ---------------------------------
    uint8_t* d_q   = nullptr;   // [qlen]        encoded query
    uint8_t* d_seq = nullptr;   // [total_bases] all node segments concatenated
    int*     d_H   = nullptr;   // [cells]       all DP blocks (device copy of dp.H)
    int* d_diag = nullptr;      // [qlen+1]      per-node boundary (reused each node)
    int* d_left = nullptr;      // [qlen+1]
    CUDA_CHECK(cudaMalloc(&d_q,    g.num_nodes ? p.qlen * sizeof(uint8_t) : 1));
    CUDA_CHECK(cudaMalloc(&d_seq,  g.total_bases * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_H,    cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_diag, (p.qlen + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_left, (p.qlen + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_q,   p.query.data(), p.qlen * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq, g.seq.data(),   g.total_bases * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // Zero the whole DP buffer (sets every block's row 0 and column 0 to 0 -- the
    // SW init -- and clears interior cells before the sweep writes them).
    CUDA_CHECK(cudaMemset(d_H, 0, cells * sizeof(int)));

    GpuTimer timer;
    timer.start();

    std::vector<int> diag_in, left_in;   // host scratch, rebuilt per node
    for (int v = 0; v < g.num_nodes; ++v) {
        const int L = g.seq_len[v];
        const int W = L + 1;
        const int base = dp.block_off[v];
        const uint8_t* d_node_seq = d_seq + g.seq_off[v];   // this node's bases on device

        // (1) Build boundary vectors from the HOST mirror (predecessors already
        //     copied back below in earlier iterations) and upload them.
        build_boundaries(p, dp, v, diag_in, left_in);
        CUDA_CHECK(cudaMemcpy(d_diag, diag_in.data(), (p.qlen + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_left, left_in.data(), (p.qlen + 1) * sizeof(int), cudaMemcpyHostToDevice));

        // (2) Sweep this node's block as an anti-diagonal wavefront. Diagonal d
        //     ranges 2..qlen+L; on diagonal d the valid rows are
        //     i in [max(1, d-L) .. min(qlen, d-1)] (so 1 <= j = d-i <= L).
        for (int d = 2; d <= p.qlen + L; ++d) {
            const int i_lo = (d - L > 1) ? (d - L) : 1;
            const int i_hi = (d - 1 < p.qlen) ? (d - 1) : p.qlen;
            const int cnt  = i_hi - i_lo + 1;
            if (cnt <= 0) continue;
            const int blocks = (cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            graph_sw_diagonal_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                d_q, d_node_seq, d_H, base, L, W, p.qlen, d_diag, d_left, d, i_lo, cnt);
        }

        // (3) Copy THIS node's finished block back into the host mirror so the
        //     next nodes' build_boundaries() can read it as a predecessor. We copy
        //     only this block (a contiguous slice), not the whole buffer.
        const int block_cells = (p.qlen + 1) * W;
        CUDA_CHECK(cudaMemcpy(dp.H.data() + base, d_H + base,
                              block_cells * sizeof(int), cudaMemcpyDeviceToHost));
    }

    *kernel_ms = timer.stop_ms();                  // syncs -> all nodes done
    CUDA_CHECK_LAST("graph_sw_diagonal_kernel");   // surface any launch/exec error

    // dp.H now holds the full filled matrix (every block copied back above).
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_seq));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_left));
}
