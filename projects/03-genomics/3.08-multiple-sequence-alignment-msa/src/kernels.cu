// ===========================================================================
// src/kernels.cu  --  One-block-per-pair NW scoring + host distance wrapper
// ===========================================================================
// Project 3.8 : Multiple Sequence Alignment (MSA)
//
// GPU twin of distance_matrix_cpu(): computes the SAME integer NW score for
// every sequence pair, but with all P = N(N-1)/2 pairs running in parallel --
// one CUDA thread block per pair (the catalog pattern for this project). The
// per-pair recurrence is the shared nw_score_core() from nw_core.h, so every
// score matches the CPU bit-for-bit. main.cu runs both and asserts equality.
//
// See ../THEORY.md "GPU mapping" for the block/grid/shared-memory reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "nw_core.h"             // nw_score_core, nw_self_score, NW_* scoring
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <algorithm>             // std::min, std::max

// One block per pair. The block is small (32 = one warp) because, in this
// teaching version, a single thread drives the serial DP while the block's role
// is to OWN the shared-memory scratch for its pair. A wavefront upgrade (see
// kernels.cuh + Exercises) would use all the threads; we keep 32 so the launch
// is cheap and occupancy stays high (many resident blocks = many pairs in flight).
static constexpr int THREADS_PER_BLOCK = 32;

// ---------------------------------------------------------------------------
// nw_pairs_kernel: block `p` scores pair (d_pair_a[p], d_pair_b[p]).
//
//   Thread-to-data map:  blockIdx.x = pair index p.  Within the block, ONLY
//   threadIdx.x == 0 runs the recurrence (the DP is serial in this version); the
//   other 31 lanes simply guard out. They still earn their keep: the block as a
//   whole reserves the shared-memory DP rows and gives the scheduler a full warp
//   to swap in, hiding the global-memory latency of neighbouring blocks.
//
//   Memory: the two rolling DP rows (prev,curr), each (max_len+1) ints, live in
//   DYNAMIC SHARED MEMORY -- on-chip, ~100x faster than global, and private to
//   the block. We request its size at launch (see distance_matrix_gpu). Reading
//   the sequences themselves from global memory is fine: each residue is read
//   O(L) times by one thread, and L is small here.
//
//   No atomics, no __syncthreads: one thread does all the work and writes the two
//   symmetric output cells. Independent blocks never touch the same d_score cell.
// ---------------------------------------------------------------------------
__global__ void nw_pairs_kernel(const uint8_t* __restrict__ d_data,
                                const int* __restrict__ d_off,
                                const int* __restrict__ d_len,
                                const int* __restrict__ d_pair_a,
                                const int* __restrict__ d_pair_b,
                                int num_pairs, int max_len,
                                int* __restrict__ d_score, int n) {
    const int p = blockIdx.x;                 // this block's pair index
    if (p >= num_pairs) return;               // guard the ragged last block(s)
    if (threadIdx.x != 0) return;             // serial DP: lane 0 does the work

    // Dynamic shared memory carved into two DP rows of (max_len+1) ints each.
    // extern __shared__ declares a block-private buffer whose SIZE is set by the
    // third launch argument; we lay prev first, curr right after it.
    extern __shared__ int s_rows[];
    int* prev = s_rows;                       // [max_len+1]
    int* curr = s_rows + (max_len + 1);       // [max_len+1]

    const int ia = d_pair_a[p];               // first sequence of the pair
    const int ib = d_pair_b[p];               // second sequence of the pair
    const uint8_t* a = d_data + d_off[ia];    // pointer to seq ia in the flat buffer
    const uint8_t* b = d_data + d_off[ib];
    const int la = d_len[ia];
    const int lb = d_len[ib];

    // THE shared recurrence -- identical to the CPU reference (nw_core.h).
    const int sc = nw_score_core(a, la, b, lb, prev, curr);

    // Write both symmetric entries. Distinct pairs -> distinct cells, no races.
    d_score[(size_t)ia * n + ib] = sc;
    d_score[(size_t)ib * n + ia] = sc;
}

// ---------------------------------------------------------------------------
// distance_matrix_gpu: host wrapper for STAGE 1 (the GPU teaching point).
//   Steps: (1) build the flat pair list (a<b); (2) upload sequences + pair list;
//   (3) launch one block per pair (DP rows in shared memory); (4) copy the score
//   matrix back; (5) fill the diagonal (self-scores) and derive distances on the
//   host -- using the SAME normalisation as distance_matrix_cpu so the two paths
//   are directly comparable. Only the kernel is timed (CUDA events).
// ---------------------------------------------------------------------------
void distance_matrix_gpu(const SeqSet& s,
                         std::vector<int>& raw_score,
                         std::vector<double>& D,
                         float* kernel_ms) {
    const int n = s.n;
    raw_score.assign(static_cast<std::size_t>(n) * n, 0);
    D.assign(static_cast<std::size_t>(n) * n, 0.0);

    // (1) Flat list of unordered pairs (a<b). P = n(n-1)/2.
    std::vector<int> h_pair_a, h_pair_b;
    h_pair_a.reserve(static_cast<std::size_t>(n) * (n - 1) / 2);
    h_pair_b.reserve(static_cast<std::size_t>(n) * (n - 1) / 2);
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b) { h_pair_a.push_back(a); h_pair_b.push_back(b); }
    const int num_pairs = static_cast<int>(h_pair_a.size());

    // (2) Device buffers. d_ prefix = DEVICE pointer (CLAUDE.md §12).
    uint8_t* d_data = nullptr;
    int *d_off = nullptr, *d_len = nullptr, *d_pa = nullptr, *d_pb = nullptr, *d_score = nullptr;
    const std::size_t data_bytes  = s.data.size() * sizeof(uint8_t);
    const std::size_t off_bytes   = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t pair_bytes  = static_cast<std::size_t>(num_pairs) * sizeof(int);
    const std::size_t score_bytes = static_cast<std::size_t>(n) * n * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));     // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_off,  off_bytes));
    CUDA_CHECK(cudaMalloc(&d_len,  off_bytes));
    CUDA_CHECK(cudaMalloc(&d_pa,   pair_bytes));
    CUDA_CHECK(cudaMalloc(&d_pb,   pair_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, score_bytes));

    CUDA_CHECK(cudaMemcpy(d_data, s.data.data(), data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off,  s.off.data(),  off_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len,  s.len.data(),  off_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pa,   h_pair_a.data(), pair_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pb,   h_pair_b.data(), pair_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_score, 0, score_bytes));

    // (3) Launch: one block per pair; shared memory = two DP rows of (max_len+1).
    //     The third <<<>>> argument is the DYNAMIC shared-memory byte count.
    const int blocks   = num_pairs;
    const std::size_t shmem = static_cast<std::size_t>(2) * (s.max_len + 1) * sizeof(int);
    GpuTimer timer;
    timer.start();
    nw_pairs_kernel<<<blocks, THREADS_PER_BLOCK, shmem>>>(
        d_data, d_off, d_len, d_pa, d_pb, num_pairs, s.max_len, d_score, n);
    *kernel_ms = timer.stop_ms();             // GPU-measured kernel time (syncs)
    CUDA_CHECK_LAST("nw_pairs_kernel");       // surface launch/exec errors

    // (4) Copy the score matrix back.
    CUDA_CHECK(cudaMemcpy(raw_score.data(), d_score, score_bytes, cudaMemcpyDeviceToHost));

    // (5) Diagonal self-scores (no alignment needed) + derive distances. SAME
    //     normalisation as distance_matrix_cpu(): divide by the shorter self.
    for (int a = 0; a < n; ++a)
        raw_score[static_cast<std::size_t>(a) * n + a] = nw_self_score(s.len[a]);
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            const int sc = raw_score[static_cast<std::size_t>(a) * n + b];
            const int self = nw_self_score(std::min(s.len[a], s.len[b]));
            double dist = (self > 0) ? (1.0 - static_cast<double>(sc) / self) : 0.0;
            if (dist < 0.0) dist = 0.0;
            if (dist > 1.0) dist = 1.0;
            D[static_cast<std::size_t>(a) * n + b] = dist;
        }
    }

    // Always free device memory (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_len));
    CUDA_CHECK(cudaFree(d_pa));
    CUDA_CHECK(cudaFree(d_pb));
    CUDA_CHECK(cudaFree(d_score));
}
