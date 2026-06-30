// ===========================================================================
// src/kernels.cu  --  GPU SV-calling kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.21 : Structural Variant (SV) Calling
//
// WHAT THIS FILE DOES
//   Implements the device kernel (refine_and_vote_kernel) and the host glue
//   (sv_call_gpu) that uploads reads + reference, launches the kernel, times it,
//   brings the histogram back, and merges it into SV calls (the shared host step
//   histogram_to_calls, reused from the CPU reference). This is the GPU twin of
//   sv_call_cpu(); main.cu runs both and compares histograms bin-for-bin.
//
//   The per-read MATH (banded SW, breakpoint refinement, binning) lives in sv.h
//   as __host__ __device__ helpers, so this kernel and the CPU reference execute
//   the SAME integer code -> the GPU histogram equals the CPU histogram EXACTLY.
//
// READ THIS AFTER: kernels.cuh (thread-mapping idea), sv.h (the shared math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)
#include "sv.h"                  // sv_refine_breakpoint, sv_bin (shared HD math)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps to hide the global-memory latency of
// reading the reference flank, and keeps many blocks resident for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// refine_and_vote_kernel: ONE THREAD PER READ (PATTERN A: independent jobs).
//
//   Launch config (set in sv_call_gpu):
//     grid  = ceil(N / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: read index i = blockIdx.x * blockDim.x + threadIdx.x.
//
//   Each thread:
//     1. Loads its read's left flank from global memory into a small per-thread
//        register/local array (SV_FLANK bytes -- tiny, stays in registers/L1).
//     2. Calls sv_refine_breakpoint (sv.h): banded Smith-Waterman over a +/-
//        SV_SEARCH window against the reference to nail the breakpoint. This is
//        the compute-heavy, fully-independent part -> ideal for the GPU.
//     3. VOTES the refined breakpoint into the shared histogram with atomicAdd
//        (PATTERN B: scatter-reduction). Many reads from the SAME real SV land in
//        the same bin, so their atomicAdds collide -- but INTEGER atomic adds
//        COMMUTE, so the final per-bin counts are order-independent and match the
//        CPU exactly (no float reordering, no tolerance; PATTERNS.md §3/§4).
//
//   Memory spaces:
//     * ref, reads_* : global memory, read-only (marked __restrict__ so the
//       compiler may cache loads; ref is hot and small enough to ride in L2).
//     * the DP rows of banded SW live in registers/local memory (per thread).
//     * hist, len_sum : global memory, atomicAdd targets (the only writes).
// ---------------------------------------------------------------------------
__global__ void refine_and_vote_kernel(const int* __restrict__ reads_guess,
                                       const int* __restrict__ reads_dellen,
                                       const signed char* __restrict__ reads_flank,
                                       int N,
                                       const signed char* __restrict__ ref, int ref_len,
                                       unsigned int* __restrict__ hist,
                                       unsigned long long* __restrict__ len_sum) {
    // Global read index this thread owns.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // GUARD THE RAGGED LAST BLOCK: N is rarely a multiple of the block size, so
    // the final block has threads with i >= N. They must do nothing or they would
    // read reads_* out of bounds (an illegal-address crash).
    if (i >= N) return;

    // (1) Pull this read's flank into a private array. sv_refine_breakpoint wants
    //     a contiguous signed-char buffer; reads_flank is row-major [N][SV_FLANK].
    signed char left[SV_FLANK];
    const signed char* src = reads_flank + static_cast<std::size_t>(i) * SV_FLANK;
    #pragma unroll
    for (int j = 0; j < SV_FLANK; ++j) left[j] = src[j];

    // (2) Refine the breakpoint by banded SW (identical call the CPU makes).
    int score = 0;
    const int bp  = sv_refine_breakpoint(left, SV_FLANK, ref, ref_len,
                                         reads_guess[i], &score);
    const int bin = sv_bin(bp);
    if (bin < 0 || bin >= ref_len) return;   // refined off the reference: drop

    // (3) Cast the vote. atomicAdd returns the OLD value (ignored); the add is
    //     what matters. unsigned int / unsigned long long atomics are supported
    //     on all sm_75+ targets and commute -> deterministic, CPU-matching.
    atomicAdd(&hist[bin], 1u);
    atomicAdd(&len_sum[bin], static_cast<unsigned long long>(reads_dellen[i]));
}

// ---------------------------------------------------------------------------
// sv_call_gpu: host wrapper. Canonical CUDA steps, then the shared host merge.
//   (1) flatten reads into device-friendly arrays  (2) upload reads + reference
//   (3) zero the histogram + launch the kernel (timed)  (4) copy histogram back
//   (5) free device memory  (6) merge histogram -> SV calls (shared with CPU).
// We time ONLY the kernel (step 3) with CUDA events so the figure is compute, not
// PCIe transfer (discussed separately in THEORY §timing).
// ---------------------------------------------------------------------------
std::vector<SvCall> sv_call_gpu(const SvDataset& d, unsigned int min_support,
                                std::vector<unsigned int>& hist,
                                std::vector<unsigned long long>& len_sum,
                                float* kernel_ms) {
    const int N = d.N();
    const int ref_len = d.ref_len;

    // (1) Flatten the array-of-structs (SvRead) into struct-of-arrays for the
    //     GPU: separate guess[], dellen[], and a packed flank[] buffer. SoA gives
    //     coalesced, contiguous loads -- the GPU-friendly layout (THEORY §GPU).
    std::vector<int>         h_guess(N), h_dellen(N);
    std::vector<signed char> h_flank(static_cast<std::size_t>(N) * SV_FLANK);
    for (int i = 0; i < N; ++i) {
        h_guess[i]  = d.reads[i].raw_guess;
        h_dellen[i] = d.reads[i].del_len;
        for (int j = 0; j < SV_FLANK; ++j)
            h_flank[static_cast<std::size_t>(i) * SV_FLANK + j] = d.reads[i].left[j];
    }

    // Device buffers (d_ prefix = DEVICE pointer; dereferencing on host crashes).
    int*                d_guess  = nullptr;
    int*                d_dellen = nullptr;
    signed char*        d_flank  = nullptr;
    signed char*        d_ref    = nullptr;
    unsigned int*       d_hist   = nullptr;
    unsigned long long* d_lensum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_guess,  static_cast<std::size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dellen, static_cast<std::size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flank,  h_flank.size() * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_ref,    static_cast<std::size_t>(ref_len) * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_hist,   static_cast<std::size_t>(ref_len) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_lensum, static_cast<std::size_t>(ref_len) * sizeof(unsigned long long)));

    // (2) Upload reads + reference H2D.
    CUDA_CHECK(cudaMemcpy(d_guess,  h_guess.data(),  static_cast<std::size_t>(N) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dellen, h_dellen.data(), static_cast<std::size_t>(N) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flank,  h_flank.data(),  h_flank.size() * sizeof(signed char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref,    d.ref.data(),    static_cast<std::size_t>(ref_len) * sizeof(signed char), cudaMemcpyHostToDevice));

    // (3) Zero the output tallies, then launch the refine+vote kernel (timed).
    CUDA_CHECK(cudaMemset(d_hist,   0, static_cast<std::size_t>(ref_len) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_lensum, 0, static_cast<std::size_t>(ref_len) * sizeof(unsigned long long)));

    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    refine_and_vote_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_guess, d_dellen, d_flank, N, d_ref, ref_len, d_hist, d_lensum);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("refine_and_vote_kernel");

    // (4) Bring the histograms back to the host.
    hist.assign(static_cast<std::size_t>(ref_len), 0u);
    len_sum.assign(static_cast<std::size_t>(ref_len), 0ull);
    CUDA_CHECK(cudaMemcpy(hist.data(), d_hist,
                          static_cast<std::size_t>(ref_len) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(len_sum.data(), d_lensum,
                          static_cast<std::size_t>(ref_len) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_guess));
    CUDA_CHECK(cudaFree(d_dellen));
    CUDA_CHECK(cudaFree(d_flank));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_lensum));

    // (6) Merge histogram -> SV calls on the host (the SAME function the CPU uses,
    //     so identical histograms yield identical calls).
    return histogram_to_calls(hist, len_sum, ref_len,
                              static_cast<unsigned int>(N), min_support);
}
