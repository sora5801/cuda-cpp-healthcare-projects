// ===========================================================================
// src/kernels.cu  --  Anti-diagonal wavefront kernel + host sweep wrapper
// ===========================================================================
// Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
//
// GPU twin of sw_cpu(): fills the SAME matrix, but one anti-diagonal at a time
// with all cells of the diagonal computed in parallel. main.cu runs both and
// asserts every cell matches. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

static constexpr int THREADS_PER_BLOCK = 128;  // diagonals are often short; 128 keeps launches cheap

// ---------------------------------------------------------------------------
// sw_diagonal_kernel: one thread fills one cell (i, j) on anti-diagonal d.
//   Thread k -> i = i_lo + k, j = d - i. Because every cell this kernel writes
//   lies on diagonal d, and it only READS cells on diagonals d-1 and d-2 (all
//   finalised by previous launches), there is no read-after-write hazard within
//   the launch -- no atomics or __syncthreads needed.
//   Row stride W = n+1 (the matrix has a 0th row and 0th column).
// ---------------------------------------------------------------------------
__global__ void sw_diagonal_kernel(const uint8_t* __restrict__ q,
                                   const uint8_t* __restrict__ t,
                                   int* __restrict__ H, int m, int n,
                                   int d, int i_lo, int count) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;                       // guard the ragged last block

    const int i = i_lo + k;                       // this thread's row
    const int j = d - i;                          // ... and column (i + j = d)
    const int W = n + 1;

    // s(q_i, t_j): substitution score for aligning the two residues.
    const int s = (q[i - 1] == t[j - 1]) ? MATCH : MISMATCH;
    const int diag = H[(i - 1) * W + (j - 1)] + s;   // from diagonal d-2
    const int up   = H[(i - 1) * W + j]       + GAP; // from diagonal d-1
    const int left = H[i * W + (j - 1)]       + GAP; // from diagonal d-1

    // Smith-Waterman local recurrence: the 0 floor lets an alignment "restart".
    int v = 0;
    if (diag > v) v = diag;
    if (up   > v) v = up;
    if (left > v) v = left;
    H[i * W + j] = v;
}

// ---------------------------------------------------------------------------
// sw_gpu: upload sequences + zeroed matrix, then sweep anti-diagonals d=2..m+n,
// launching one kernel per diagonal. We time the whole sweep with CUDA events.
//
// HONESTY (see THEORY "real world"): for a SINGLE modest pair this issues m+n-1
// tiny launches, and launch overhead can make the GPU slower than the CPU. The
// wavefront pays off for large matrices, and production tools batch MANY pairs
// (one block per pair) and/or use a single persistent kernel with grid sync.
// We keep one-launch-per-diagonal because it makes the dependency structure
// unmistakable -- the teaching goal here.
// ---------------------------------------------------------------------------
void sw_gpu(const SeqPair& sp, std::vector<int>& H, float* kernel_ms) {
    const int m = sp.m, n = sp.n;
    const std::size_t cells = static_cast<std::size_t>(m + 1) * (n + 1);
    H.assign(cells, 0);

    uint8_t* d_q = nullptr;   // [m] encoded query
    uint8_t* d_t = nullptr;   // [n] encoded target
    int*     d_H = nullptr;   // [(m+1)*(n+1)] DP matrix
    CUDA_CHECK(cudaMalloc(&d_q, m * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_t, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_H, cells * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_q, sp.q.data(), m * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t, sp.t.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // Zero the matrix (this also sets row 0 and column 0 to 0 -- the SW init).
    CUDA_CHECK(cudaMemset(d_H, 0, cells * sizeof(int)));

    GpuTimer timer;
    timer.start();
    // Sweep the wavefront. Diagonal d ranges over 2..m+n; on diagonal d the valid
    // rows are i in [max(1, d-n) .. min(m, d-1)] (so that 1<=j=d-i<=n).
    for (int d = 2; d <= m + n; ++d) {
        const int i_lo = (d - n > 1) ? (d - n) : 1;
        const int i_hi = (d - 1 < m) ? (d - 1) : m;
        const int count = i_hi - i_lo + 1;
        if (count <= 0) continue;
        const int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sw_diagonal_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_q, d_t, d_H, m, n, d, i_lo, count);
    }
    *kernel_ms = timer.stop_ms();          // syncs -> all diagonals done
    CUDA_CHECK_LAST("sw_diagonal_kernel"); // surface any launch/exec error

    CUDA_CHECK(cudaMemcpy(H.data(), d_H, cells * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_H));
}
