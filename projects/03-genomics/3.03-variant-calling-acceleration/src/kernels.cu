// ===========================================================================
// src/kernels.cu  --  PairHMM forward kernel (one thread per read-haplotype pair)
// ---------------------------------------------------------------------------
// Project 3.3 : Variant Calling Acceleration
//
// WHAT THIS FILE DOES
//   Implements the device kernel (pairhmm_kernel) and the host-side glue
//   (pairhmm_gpu) that uploads the reads/haplotypes, launches ONE THREAD PER
//   (read, haplotype) PAIR -- each thread filling that pair's forward DP table
//   with the SAME pairhmm_step() the CPU reference uses -- times the kernel, and
//   copies the R x H log10-likelihood matrix back. This is the GPU twin of
//   reference_cpu.cpp; main.cu runs both and asserts they agree to a few ULP.
//
//   The shared per-cell arithmetic (pairhmm_core.h) is the key to that agreement:
//   the kernel and the reference run *identical* IEEE-754 double operations, so
//   verification is essentially exact rather than approximate.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea) and pairhmm_core.h.
// ===========================================================================
#include "kernels.cuh"
#include "pairhmm_core.h"        // PairHmmCell, pairhmm_step (shared HD core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cmath>     // log10 (device intrinsic)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Threads per block. 128 keeps register pressure manageable (each thread holds
// two DP rows of PairHmmCell in local memory) while still giving the scheduler
// 4 warps to hide latency. A multiple of the 32-lane warp, as always.
static constexpr int THREADS_PER_BLOCK = 128;

// Compile-time cap on haplotype length. Each thread keeps two rolling DP rows of
// (MAX_HAP_LEN+1) PairHmmCell in LOCAL memory (registers spilling to local). 128
// columns x 2 rows x 24 bytes = ~6 KB/thread -- fine for the small teaching demo.
// Production callers (Parabricks) instead give each pair a whole BLOCK and stage
// the DP table in SHARED memory along anti-diagonals; see THEORY.md "real world".
#define MAX_HAP_LEN 127

// ---------------------------------------------------------------------------
// pairhmm_kernel: one thread computes log10 P(read r | haplotype h) for one pair.
//
//   Launch config (set in pairhmm_gpu):
//     grid  = ceil(n_pairs / THREADS_PER_BLOCK) blocks, n_pairs = n_reads*n_haps
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: pair index p = blockIdx.x*blockDim.x + threadIdx.x,
//     decoded to read r = p / n_haps and haplotype h = p % n_haps. Every pair is
//     independent -> no atomics, no __syncthreads, no inter-thread communication.
//
//   Memory: reads the read/qual/hap bytes from GLOBAL memory; the DP table lives
//   in per-thread LOCAL memory (two rolling rows, exactly as the CPU reference).
//   Writes one double into loglik[p].
//
//   The arithmetic is pairhmm_step() from pairhmm_core.h -- byte-identical to the
//   CPU path, which is what makes the GPU/CPU comparison in main.cu pass tightly.
// ---------------------------------------------------------------------------
__global__ void pairhmm_kernel(const uint8_t* __restrict__ reads,  // [n_reads*read_len]
                               const uint8_t* __restrict__ quals,  // [n_reads*read_len]
                               const uint8_t* __restrict__ haps,   // [n_haps *hap_len ]
                               int n_reads, int n_haps, int read_len, int hap_len,
                               PairHmmParams params,                // by value -> registers
                               double* __restrict__ loglik) {       // [n_reads*n_haps]
    const int p = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's pair
    const int n_pairs = n_reads * n_haps;
    if (p >= n_pairs) return;                              // guard ragged last block

    const int r = p / n_haps;   // which read
    const int h = p % n_haps;   // which haplotype

    const int R = read_len;
    const int H = hap_len;
    const int W = H + 1;        // DP columns including the j=0 boundary

    // Pointers to this pair's sequences in global memory.
    const uint8_t* read = reads + static_cast<std::size_t>(r) * R;
    const uint8_t* qual = quals + static_cast<std::size_t>(r) * R;
    const uint8_t* hap  = haps  + static_cast<std::size_t>(h) * H;

    // Two rolling DP rows in local memory (prev = row i-1, cur = row i). Same
    // two-row scheme as forward_one_pair() on the CPU, so the math lines up.
    PairHmmCell prev[MAX_HAP_LEN + 1];
    PairHmmCell cur [MAX_HAP_LEN + 1];

    // Row 0: M=I=0 everywhere; D seeded with the uniform start prior 1/H at every
    // haplotype column (the read may begin anywhere along the haplotype).
    const double start = 1.0 / static_cast<double>(H);
    for (int j = 0; j < W; ++j) {
        prev[j].m = 0.0;
        prev[j].i = 0.0;
        prev[j].d = (j == 0) ? 0.0 : start;
    }

    // Fill rows i = 1..R. Column 0 stays all-zero (a read base cannot align to an
    // empty haplotype prefix). We swap rows by COPYING cur->prev at the end of
    // each i (a device-friendly fixed-size copy; std::swap is not available here).
    for (int i = 1; i <= R; ++i) {
        cur[0].m = 0.0; cur[0].i = 0.0; cur[0].d = 0.0;
        const uint8_t rb = read[i - 1];
        const int     q  = static_cast<int>(qual[i - 1]);
        for (int j = 1; j <= H; ++j) {
            const uint8_t hb = hap[j - 1];
            const PairHmmCell diag = prev[j - 1];   // (i-1, j-1)
            const PairHmmCell up   = prev[j];       // (i-1, j)
            const PairHmmCell left = cur[j - 1];    // (i,   j-1) -- already done
            cur[j] = pairhmm_step(params, rb, hb, q, diag, up, left);
        }
        // cur -> prev for the next read base.
        for (int j = 0; j < W; ++j) prev[j] = cur[j];
    }

    // Total likelihood: sum of M+I over the final read row (any finish column).
    double sum = 0.0;
    for (int j = 1; j <= H; ++j) sum += prev[j].m + prev[j].i;

    // log10, guarding the impossible-pair case (sum == 0) the same way the CPU
    // does, so both sides produce -inf identically. -INFINITY is a compile-time
    // constant (from <cmath>/<math.h>), avoiding a literal divide-by-zero.
    loglik[p] = (sum > 0.0) ? log10(sum) : (-INFINITY);
}

// ---------------------------------------------------------------------------
// pairhmm_gpu: host wrapper. The canonical CUDA steps:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy the log-likelihood matrix device->host
//   (5) free device memory
// We time ONLY the launch (step 3) with CUDA events so the figure is the kernel
// cost, not the PCIe transfer cost (discussed separately in THEORY.md).
// ---------------------------------------------------------------------------
void pairhmm_gpu(const VariantData& v, std::vector<double>& loglik, float* kernel_ms) {
    const int n_reads = v.n_reads, n_haps = v.n_haps;
    const int read_len = v.read_len, hap_len = v.hap_len;
    const std::size_t n_pairs = static_cast<std::size_t>(n_reads) * n_haps;
    loglik.assign(n_pairs, 0.0);

    // Bound check: the per-thread DP rows are fixed-size, so refuse haplotypes
    // longer than the compile-time cap rather than corrupt local memory.
    if (hap_len > MAX_HAP_LEN) {
        std::fprintf(stderr, "[pairhmm_gpu] hap_len=%d exceeds MAX_HAP_LEN=%d\n", hap_len, MAX_HAP_LEN);
        std::exit(EXIT_FAILURE);
    }

    // (1) Device buffers (d_ prefix = device pointer; dereferencing on host crashes).
    uint8_t *d_reads = nullptr, *d_quals = nullptr, *d_haps = nullptr;
    double  *d_loglik = nullptr;
    const std::size_t read_bytes = v.reads.size() * sizeof(uint8_t);
    const std::size_t hap_bytes  = v.haps.size()  * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_reads,  read_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_quals,  read_bytes));
    CUDA_CHECK(cudaMalloc(&d_haps,   hap_bytes));
    CUDA_CHECK(cudaMalloc(&d_loglik, n_pairs * sizeof(double)));

    // (2) Copy inputs H2D.
    CUDA_CHECK(cudaMemcpy(d_reads, v.reads.data(), read_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quals, v.quals.data(), read_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_haps,  v.haps.data(),  hap_bytes,  cudaMemcpyHostToDevice));

    // (3) Launch: one thread per (read, haplotype) pair, rounding up the grid.
    const int blocks = static_cast<int>((n_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    GpuTimer timer;
    timer.start();
    pairhmm_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_reads, d_quals, d_haps,
                                                  n_reads, n_haps, read_len, hap_len,
                                                  v.params, d_loglik);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("pairhmm_kernel");     // catch launch + execution errors

    // (4) Bring the log-likelihood matrix back.
    CUDA_CHECK(cudaMemcpy(loglik.data(), d_loglik, n_pairs * sizeof(double), cudaMemcpyDeviceToHost));

    // (5) Free everything (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_reads));
    CUDA_CHECK(cudaFree(d_quals));
    CUDA_CHECK(cudaFree(d_haps));
    CUDA_CHECK(cudaFree(d_loglik));
}
