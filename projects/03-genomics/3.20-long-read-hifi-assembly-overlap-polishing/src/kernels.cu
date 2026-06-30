// ===========================================================================
// src/kernels.cu  --  All-vs-all overlap chaining kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
//
// WHAT THIS FILE DOES
//   The GPU twin of overlap_cpu() in reference_cpu.cpp. One thread owns one
//   ordered read pair (i<j): it decodes its flat pair index, reads the two
//   reads' minimiser slices, builds the shared-seed anchors, and runs the
//   collinear chaining DP -- all in fixed on-thread scratch (no allocation),
//   using the SAME integer link scorer (overlap_core.h) as the CPU. main.cu
//   runs both and asserts they agree (bit-for-bit, since the math is integer).
//
//   Read ../THEORY.md "GPU mapping" for the thread/block layout reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <cmath>                 // sqrt (for the pair-index inverse)

// 128 threads/block: a multiple of the 32-lane warp. Each thread does a fair
// amount of independent work (an O(A^2) DP over a few dozen anchors) and uses a
// chunk of on-thread scratch (anchors + f[]), so we keep the block modest to
// stay within the register/local-memory budget on sm_75..sm_89. See THEORY.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// decode_pair: invert pair_index() -- map a flat upper-triangle slot `t` back to
//   its (i, j). Row i of the triangle holds (N-1-i) pairs and starts at slot
//   T(i) = i*N - i*(i+1)/2. We solve T(i) <= t < T(i+1) for i in closed form
//   (the quadratic gives i = floor((2N-1 - sqrt((2N-1)^2 - 8t)) / 2)) and then
//   recover j from the remainder. Closed-form avoids a per-thread search loop.
//   We compute in double then nudge i to correct any floating rounding at the
//   row boundary, so the mapping is exact for every slot.
//
//   This is the device counterpart of reference_cpu.h's pair_index(); the two
//   must agree so CPU slot k and GPU slot k describe the SAME pair.
// ---------------------------------------------------------------------------
__device__ inline void decode_pair(long long t, int n, int& i, int& j) {
    const double N  = static_cast<double>(n);
    const double tt = static_cast<double>(t);
    // Quadratic inverse of the row-start triangular number (see comment above).
    double ii = (2.0 * N - 1.0 - sqrt((2.0 * N - 1.0) * (2.0 * N - 1.0) - 8.0 * tt)) * 0.5;
    int row = static_cast<int>(ii);
    // Row-start slot T(row); correct off-by-one from the floating sqrt.
    auto Tstart = [&](int r) -> long long {
        return static_cast<long long>(r) * n - static_cast<long long>(r) * (r + 1) / 2;
    };
    while (row > 0 && Tstart(row) > t) --row;          // overshot: step back
    while (Tstart(row + 1) <= t)        ++row;          // undershot: step forward
    i = row;
    j = static_cast<int>(t - Tstart(row)) + row + 1;    // column within the row
}

// ---------------------------------------------------------------------------
// overlap_kernel: score one read pair per thread (grid-stride so a modest grid
//   covers all N*(N-1)/2 pairs of an arbitrarily large dataset).
//
//   Thread t -> pair slot (after grid-stride) -> (i, j) via decode_pair.
//   STEP A: build shared-seed anchors by scanning read i's minimisers (outer)
//           against read j's (inner), emitting (qpos, tpos) on a hash match.
//           Same order as the CPU (query-pos-major) and same OVL_MAX_ANCHORS cap.
//   STEP B: O(A^2) collinear chaining DP with the shared integer link scorer.
//   Memory: d_min_hash/d_min_pos in global memory; anchors and f[] in on-thread
//           local memory (fixed-size arrays -> registers/local). No atomics, no
//           shared memory: every pair's output is independent.
// ---------------------------------------------------------------------------
__global__ void overlap_kernel(const int32_t*  __restrict__ d_off,
                               const int32_t*  __restrict__ d_cnt,
                               const int32_t*  __restrict__ d_min_pos,
                               const uint32_t* __restrict__ d_min_hash,
                               int n_reads, long long n_pairs,
                               int32_t* __restrict__ d_score,
                               int32_t* __restrict__ d_nanchor) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < n_pairs; t += stride) {
        // Which pair (i, j) does this thread own?
        int i, j;
        decode_pair(t, n_reads, i, j);

        const int q_off = d_off[i], q_cnt = d_cnt[i];   // read i's minimiser slice
        const int t_off = d_off[j], t_cnt = d_cnt[j];   // read j's minimiser slice

        // --- STEP A: collect anchors (query-pos-major), identical to CPU ---
        int  anchor_q[OVL_MAX_ANCHORS];
        int  anchor_t[OVL_MAX_ANCHORS];
        int  n_anchor = 0;
        for (int qi = 0; qi < q_cnt && n_anchor < OVL_MAX_ANCHORS; ++qi) {
            const uint32_t h = d_min_hash[q_off + qi];
            for (int ti = 0; ti < t_cnt && n_anchor < OVL_MAX_ANCHORS; ++ti) {
                if (d_min_hash[t_off + ti] == h) {
                    anchor_q[n_anchor] = d_min_pos[q_off + qi];
                    anchor_t[n_anchor] = d_min_pos[t_off + ti];
                    ++n_anchor;
                }
            }
        }

        // --- STEP B: collinear chaining, BOTH strands (overlap_core.h) ---
        // ovl_chain_best_both_strands runs the O(A^2) DP for a forward overlap
        // and a reverse-complement overlap and returns the stronger chain -- the
        // SAME function the CPU reference calls, so the integer scores match
        // bit-for-bit. f[]/neg[] are this thread's local scratch (no allocation).
        int best = 0;
        if (n_anchor > 0) {
            int f[OVL_MAX_ANCHORS];       // DP table in on-thread local memory
            int neg[OVL_MAX_ANCHORS];     // strand-flipped target coords scratch
            best = ovl_chain_best_both_strands(anchor_q, anchor_t, n_anchor, f, neg);
        }

        // Write this pair's result at its deterministic slot (== CPU slot t).
        d_score[t]   = best;
        d_nanchor[t] = n_anchor;
    }
}

// ---------------------------------------------------------------------------
// overlap_gpu: the canonical CUDA host wrapper.
//   (a) Flatten the ReadSet's minimisers into two parallel device arrays
//       (positions, hashes) plus the per-read offset/count arrays.
//   (b) Allocate the per-pair output arrays (score, n_anchors).
//   (c) Launch overlap_kernel over all pairs, timing ONLY the kernel.
//   (d) Copy results back and pack them into OverlapResult (with i,j filled in
//       from the same pair ordering), then (e) free device memory.
//   We time only the kernel (CUDA events), not the H2D/D2H copies -- THEORY
//   discusses the copy cost separately (it is amortised at real read counts).
// ---------------------------------------------------------------------------
void overlap_gpu(const ReadSet& rs, std::vector<OverlapResult>& out, float* kernel_ms) {
    const int       n_reads   = rs.n_reads;
    const long long n_pairs   = rs.num_pairs();
    const int       total_min = static_cast<int>(rs.mins.size());

    out.assign(static_cast<std::size_t>(n_pairs), OverlapResult{});

    // (a) Split the array-of-structs Minimizer into two host staging arrays so we
    //     can upload a struct-of-arrays layout (better device access pattern).
    std::vector<int32_t>  h_pos(total_min);
    std::vector<uint32_t> h_hash(total_min);
    for (int m = 0; m < total_min; ++m) {
        h_pos[m]  = rs.mins[m].pos;
        h_hash[m] = rs.mins[m].hash;
    }

    // Device buffers.
    int32_t*  d_off      = nullptr;   // [n_reads]
    int32_t*  d_cnt      = nullptr;   // [n_reads]
    int32_t*  d_min_pos  = nullptr;   // [total_min]
    uint32_t* d_min_hash = nullptr;   // [total_min]
    int32_t*  d_score    = nullptr;   // [n_pairs]
    int32_t*  d_nanchor  = nullptr;   // [n_pairs]

    const std::size_t reads_b = static_cast<std::size_t>(n_reads)  * sizeof(int32_t);
    const std::size_t minp_b  = static_cast<std::size_t>(total_min) * sizeof(int32_t);
    const std::size_t minh_b  = static_cast<std::size_t>(total_min) * sizeof(uint32_t);
    const std::size_t pair_b  = static_cast<std::size_t>(n_pairs)   * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_off,      reads_b));
    CUDA_CHECK(cudaMalloc(&d_cnt,      reads_b));
    CUDA_CHECK(cudaMalloc(&d_min_pos,  minp_b));
    CUDA_CHECK(cudaMalloc(&d_min_hash, minh_b));
    CUDA_CHECK(cudaMalloc(&d_score,    pair_b));
    CUDA_CHECK(cudaMalloc(&d_nanchor,  pair_b));

    CUDA_CHECK(cudaMemcpy(d_off,      rs.off.data(),  reads_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cnt,      rs.cnt.data(),  reads_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_min_pos,  h_pos.data(),   minp_b,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_min_hash, h_hash.data(),  minh_b,  cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover all pairs one-thread-each, capped so the
    //     grid stays modest; the grid-stride loop handles any remainder.
    long long blocks_ll = (n_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int blocks = static_cast<int>(blocks_ll > 4096 ? 4096 : (blocks_ll < 1 ? 1 : blocks_ll));
    GpuTimer timer;
    timer.start();
    overlap_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_off, d_cnt, d_min_pos, d_min_hash,
                                                  n_reads, n_pairs, d_score, d_nanchor);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("overlap_kernel");

    // (d) Copy the per-pair scores and anchor counts back, then reconstruct the
    //     OverlapResult array (filling i,j from the same upper-triangle order).
    std::vector<int32_t> h_score(static_cast<std::size_t>(n_pairs));
    std::vector<int32_t> h_nanchor(static_cast<std::size_t>(n_pairs));
    CUDA_CHECK(cudaMemcpy(h_score.data(),   d_score,   pair_b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nanchor.data(), d_nanchor, pair_b, cudaMemcpyDeviceToHost));

    long long k = 0;
    for (int i = 0; i < n_reads; ++i)
        for (int j = i + 1; j < n_reads; ++j) {
            OverlapResult res;
            res.read_i    = i;
            res.read_j    = j;
            res.score     = h_score[static_cast<std::size_t>(k)];
            res.n_anchors = h_nanchor[static_cast<std::size_t>(k)];
            out[static_cast<std::size_t>(k)] = res;
            ++k;
        }

    // (e) Free device memory.
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_cnt));
    CUDA_CHECK(cudaFree(d_min_pos));
    CUDA_CHECK(cudaFree(d_min_hash));
    CUDA_CHECK(cudaFree(d_score));
    CUDA_CHECK(cudaFree(d_nanchor));
}
