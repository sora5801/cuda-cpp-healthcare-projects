// ===========================================================================
// src/kernels.cu  --  Nussinov anti-diagonal wavefront kernel + host sweep
// ===========================================================================
// Project 3.10 : RNA Secondary-Structure Prediction  (Nussinov base-pair DP)
//
// GPU twin of nussinov_cpu(): it fills the SAME upper-triangular matrix, but one
// SPAN (anti-diagonal) at a time, with all cells of that span computed in
// parallel. main.cu runs both and asserts every upper-triangle cell matches
// (exact integer equality). The per-cell recurrence is the shared, HD-decorated
// nussinov_cell() in reference_cpu.h, so CPU and GPU compute identical integers.
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. Span diagonals are often short (span L has n-L cells), so a
// modest 128 keeps each per-span launch cheap while still being a multiple of
// the 32-lane warp. (For long RNAs the early spans are wide and saturate the GPU;
// the late spans are narrow and launch-bound -- discussed in THEORY "real world".)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// nussinov_span_kernel: fill every cell of one span L = j - i in parallel.
//   Launch config (set in nussinov_gpu, once per span):
//     grid  = ceil(count / THREADS_PER_BLOCK) blocks, count = n - L cells
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: thread t (0..count-1) owns cell (i = t, j = i + L).
//   Memory: reads cells of span < L from global memory (all finalised by earlier
//   launches), writes M[i*n + j]. No shared memory or atomics: cells of the same
//   span never read each other, so there is no intra-launch hazard. The bigger
//   teaching point is the DEPENDENCY STRUCTURE, not micro-optimised tiling --
//   THEORY discusses the shared-memory tiling that production CUDA RNAfold adds.
// ---------------------------------------------------------------------------
__global__ void nussinov_span_kernel(const uint8_t* __restrict__ s,
                                     int* __restrict__ M, int n, int L, int count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's cell index
    if (t >= count) return;                               // guard the ragged last block

    const int i = t;          // row  = the cell index along the span
    const int j = i + L;      // col  = i + span  (so j - i == L, by construction)

    // Delegate to the SHARED recurrence: identical code path to the CPU's
    // nussinov_cpu(), so M[i][j] comes out bit-for-bit the same integer.
    M[i * n + j] = nussinov_cell(s, M, i, j, n);
}

// ---------------------------------------------------------------------------
// nussinov_gpu: host wrapper. The canonical CUDA steps, here with a SWEEP:
//   (1) allocate device buffers for the sequence and the n*n matrix
//   (2) copy the sequence H2D and zero the matrix (the all-zero base cases)
//   (3) for span L = 1..n-1, launch one kernel that fills that span in parallel
//   (4) copy the filled matrix D2H
//   (5) free device memory
// We time the WHOLE sweep (all n-1 launches) with CUDA events.
//
// HONESTY (see THEORY "real world"): for a SHORT RNA this issues n-1 tiny
// launches and the late, narrow spans are launch-bound, so the GPU can be SLOWER
// than the CPU here. The wavefront pays off for long sequences (the early spans
// have thousands of independent cells) and when BATCHING many sequences (one CTA
// per sequence). We keep one-launch-per-span because it makes the dependency
// structure -- the actual lesson -- unmistakable.
// ---------------------------------------------------------------------------
void nussinov_gpu(const RnaSeq& r, std::vector<int>& M, float* kernel_ms) {
    const int n = r.n;
    const std::size_t cells = static_cast<std::size_t>(n) * n;
    M.assign(cells, 0);

    uint8_t* d_s = nullptr;   // [n] encoded sequence
    int*     d_M = nullptr;   // [n*n] DP matrix
    CUDA_CHECK(cudaMalloc(&d_s, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_M, cells * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_s, r.s.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // Zero the matrix: this sets ALL base cases (span 0 and the unused lower
    // triangle) to 0 pairs, exactly matching nussinov_cpu's M.assign(.., 0).
    CUDA_CHECK(cudaMemset(d_M, 0, cells * sizeof(int)));

    GpuTimer timer;
    timer.start();
    // The wavefront: spans must be filled in increasing order because span L
    // depends on spans < L. Within a span the cells are independent -> parallel.
    for (int L = 1; L < n; ++L) {
        const int count = n - L;                              // cells on this span
        const int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        nussinov_span_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_s, d_M, n, L, count);
        // NOTE: launches into the default stream serialize, so span L+1 only
        // starts after span L has fully completed -- the dependency we need. (We
        // do NOT sync per span here; the event in stop_ms() syncs once at the end.)
    }
    *kernel_ms = timer.stop_ms();              // syncs -> all spans done
    CUDA_CHECK_LAST("nussinov_span_kernel");   // surface any launch/exec error

    CUDA_CHECK(cudaMemcpy(M.data(), d_M, cells * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_M));
}
