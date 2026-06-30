// ===========================================================================
// src/kernels.cu  --  Batched splice-aware alignment kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of align_batch_cpu(): aligns every read in the batch against
//   the reference, filling each read's DP table with the SAME shared
//   cell_recurrence() and finding its best cell. main.cu runs both and asserts
//   every score, endpoint, and DP cell matches to the integer.
//
//   GPU MAPPING (see kernels.cuh and THEORY.md §GPU-mapping):
//     * one BLOCK per read (reads are independent -> no cross-block sync);
//     * within a block, ONE thread runs that read's serial DP. Why only one?
//       Plain SW's "left" (D) move reads H[i][j-1], so a row's columns form a
//       left-to-right serial chain; you cannot fill all columns of a row in
//       parallel without an anti-diagonal wavefront. The intron (N) move adds a
//       long-range read of the PREVIOUS row (H[i-1][k]), which a wavefront would
//       also have to thread through. Rather than bolt a wavefront *and* a
//       banded intron scan onto one teaching kernel (correct but hard to read),
//       we keep each small read's DP serial and win throughput by running
//       thousands of reads' DPs CONCURRENTLY across blocks. That is the honest
//       mapping for "many small independent alignments", and it is what a
//       batched aligner does when each job is too small to parallelise within.
//       THEORY.md §real-world describes the intra-read wavefront real tools add.
//
//   The DP table lives in GLOBAL memory (a long reference makes it too big for
//   shared memory); the per-read thread reads/writes its own table slice.
//
// READ THIS AFTER: kernels.cuh, reference_cpu.h (the shared math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. We use ONE active thread per block (threadIdx.x == 0 does
// the work) because a read's DP is a serial chain (see the file header). We
// still declare a small block so the launch is well-formed; 32 == one warp is
// the cheapest legal choice and avoids wasting scheduler slots on idle lanes.
static constexpr int THREADS_PER_BLOCK = 32;

// ---------------------------------------------------------------------------
// align_batch_kernel: block b aligns read b.
//   Thread-to-data map: blockIdx.x = read index; only threadIdx.x == 0 works.
//   Memory: reads d_ref (shared by all blocks), this read's d_reads slice, and
//   writes this read's d_H table + its scalar outputs. No atomics (each block
//   touches a disjoint table slot) and no __syncthreads (single worker thread).
//
//   The body is the line-for-line mirror of align_one_cpu(): same row-major
//   sweep, same call into cell_recurrence(). Identical traversal + integer math
//   => identical results, which is exactly what main.cu verifies.
// ---------------------------------------------------------------------------
__global__ void align_batch_kernel(const uint8_t* __restrict__ d_ref, int N,
                                   const uint8_t* __restrict__ d_reads,
                                   const int* __restrict__ d_read_lens,
                                   int M, int R,
                                   int* __restrict__ d_H,
                                   int* __restrict__ d_score,
                                   int* __restrict__ d_end_i,
                                   int* __restrict__ d_end_j) {
    const int b = blockIdx.x;               // this block's read index
    if (b >= R) return;                     // guard a ragged grid
    if (threadIdx.x != 0) return;           // only lane 0 fills this DP table

    const int W = N + 1;                              // row stride
    const long long table = (long long)(M + 1) * W;   // cells per read table
    int* H = d_H + (long long)b * table;              // THIS read's table
    const uint8_t* q = d_reads + (long long)b * M;    // THIS read's bases
    const int m = d_read_lens[b];                     // its true length

    // Initialise row 0 and column 0 to 0 (local-alignment boundary). We zero the
    // whole table here so padding rows (m+1..M) are defined and never "best".
    for (long long c = 0; c < table; ++c) H[c] = 0;

    int best = 0, bi = 0, bj = 0;
    for (int i = 1; i <= m; ++i) {
        const uint8_t qi = q[i - 1];
        int* row      = H + (long long)i * W;         // H[i][*]
        const int* up = H + (long long)(i - 1) * W;   // H[i-1][*]
        for (int j = 1; j <= N; ++j) {
            const uint8_t rj = d_ref[j - 1];
            // SHARED recurrence: identical call to the CPU's (reference_cpu.h).
            // The N move reads the PREVIOUS row (up = H[i-1][*]) for H[i-1][k].
            const int v = cell_recurrence(qi, rj,
                                          up[j - 1],   // diag H[i-1][j-1]
                                          up[j],       // up   H[i-1][j]
                                          row[j - 1],  // left H[i][j-1]
                                          up, d_ref, N, j);
            row[j] = v;
            if (v > best) { best = v; bi = i; bj = j; }  // first-in-scan best
        }
    }
    d_score[b] = best;
    d_end_i[b] = bi;
    d_end_j[b] = bj;
}

// ---------------------------------------------------------------------------
// align_batch_gpu: host wrapper. The canonical CUDA steps:
//   (1) allocate device memory  (2) copy inputs H2D
//   (3) launch R blocks         (4) copy results + DP tables D2H
//   (5) free device memory
// We time ONLY the launch (step 3) with CUDA events so the figure is the kernel
// cost, not the (large, here) PCIe transfer of the DP tables -- THEORY discusses
// that the table copy-back exists only so the host can TRACEBACK for display;
// a production aligner would emit CIGARs on-device and copy back only those.
// ---------------------------------------------------------------------------
void align_batch_gpu(const ReadBatch& b,
                     std::vector<AlignResult>& out,
                     std::vector<int>& H_all,
                     float* kernel_ms) {
    const int N = b.n, M = b.read_len, R = b.num_reads;
    const long long table = (long long)(M + 1) * (N + 1);
    const long long H_cells = (long long)R * table;

    out.assign(R, AlignResult{});
    H_all.assign(static_cast<std::size_t>(H_cells), 0);

    // (1) Device buffers (d_ prefix = device pointer; CLAUDE.md §12).
    uint8_t *d_ref = nullptr, *d_reads = nullptr;
    int *d_read_lens = nullptr, *d_H = nullptr;
    int *d_score = nullptr, *d_end_i = nullptr, *d_end_j = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ref,       (size_t)N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_reads,     (size_t)R * M * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_read_lens, (size_t)R * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_H,         (size_t)H_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_score,     (size_t)R * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_end_i,     (size_t)R * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_end_j,     (size_t)R * sizeof(int)));

    // (2) Copy inputs host->device.
    CUDA_CHECK(cudaMemcpy(d_ref, b.ref.data(), (size_t)N * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reads, b.reads.data(), (size_t)R * M * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_read_lens, b.read_lens.data(), (size_t)R * sizeof(int),
                          cudaMemcpyHostToDevice));

    // (3) Launch: ONE block per read. Blocks run concurrently up to the device's
    //     occupancy; that across-read parallelism is the whole point.
    GpuTimer timer;
    timer.start();
    align_batch_kernel<<<R, THREADS_PER_BLOCK>>>(d_ref, N, d_reads, d_read_lens,
                                                 M, R, d_H,
                                                 d_score, d_end_i, d_end_j);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("align_batch_kernel");

    // (4) Copy results + the full DP tables back (tables only so the host can
    //     traceback identically to the CPU side -- see main.cu).
    std::vector<int> score(R), ei(R), ej(R);
    CUDA_CHECK(cudaMemcpy(score.data(), d_score, (size_t)R * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ei.data(), d_end_i, (size_t)R * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ej.data(), d_end_j, (size_t)R * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(H_all.data(), d_H, (size_t)H_cells * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int r = 0; r < R; ++r) {
        out[r].score = score[r];
        out[r].end_i = ei[r];
        out[r].end_j = ej[r];
    }

    // (5) Free device memory.
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_reads));
    CUDA_CHECK(cudaFree(d_read_lens));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_score));
    CUDA_CHECK(cudaFree(d_end_i));
    CUDA_CHECK(cudaFree(d_end_j));
}
