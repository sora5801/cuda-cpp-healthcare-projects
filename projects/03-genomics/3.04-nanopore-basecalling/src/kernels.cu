// ===========================================================================
// src/kernels.cu  --  Batched CTC greedy-decode kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
//
// This is the GPU twin of basecall_cpu() in reference_cpu.cpp. Both call the
// SAME ctc_greedy_decode() from ctc_core.h, so they produce identical output;
// main.cu runs both and asserts agreement. See ../THEORY.md sec "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "ctc_core.h"            // ctc_greedy_decode, ctc_base_checksum (HD core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// 128 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89. Each thread does a serial O(T*C) decode with a per-read private
// output row, so there is no shared memory or cross-thread communication to
// favor a larger block; 128 keeps register pressure modest. (THEORY "GPU map".)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// basecall_kernel: one read per (logical) thread, via a grid-stride loop so a
//   fixed-size grid covers an arbitrarily large batch.
//
//   Thread-to-data mapping: thread i starts on read i and strides by the total
//   thread count until i >= n_reads. For each read it:
//     1. finds its posterior slice (offset[i]*C into d_probs),
//     2. runs the SHARED ctc_greedy_decode() into its OWN output row
//        d_bases[i*max_T] -- private per thread, so NO atomics / NO races,
//     3. stores the decoded length and the integer checksum.
//
//   Memory: d_probs / d_offset / d_T read from global memory; d_bases/d_len/
//   d_checksum written to global memory. No shared memory, no atomics: every
//   read's output is disjoint, which is exactly why this parallelizes cleanly.
//
//   Determinism: ctc_core uses only integer compares and ordered writes, so the
//   per-read result is bit-identical to the CPU and stable across runs
//   (PATTERNS.md sec 3). There is NO floating-point reduction anywhere.
// ---------------------------------------------------------------------------
__global__ void basecall_kernel(const float* __restrict__ d_probs,
                                const int*   __restrict__ d_offset,
                                const int*   __restrict__ d_T,
                                int n_reads, int max_T,
                                char*     __restrict__ d_bases,
                                int*      __restrict__ d_len,
                                uint32_t* __restrict__ d_checksum) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_reads; i += stride) {
        const int T = d_T[i];                              // this read's # steps
        // Read i's posteriors start at step offset[i]; multiply by C for the
        // element index into the flat probability buffer.
        const float* p = d_probs +
            static_cast<long long>(d_offset[i]) * CTC_NUM_CLASSES;

        // This thread's PRIVATE output row -- no other thread writes here.
        char* my_bases = d_bases + static_cast<long long>(i) * max_T;

        // THE decode -- the identical routine the CPU reference runs.
        const int len = ctc_greedy_decode(p, T, my_bases);

        d_len[i]      = len;
        d_checksum[i] = ctc_base_checksum(my_bases, len);  // exact integer hash
    }
}

// ---------------------------------------------------------------------------
// basecall_gpu: the canonical CUDA steps -- allocate, upload, launch (timed),
//   download, reconstruct, free. We time ONLY the kernel (CUDA events), not the
//   H2D/D2H copies (discussed separately in THEORY "GPU mapping").
// ---------------------------------------------------------------------------
void basecall_gpu(const ReadSet& rs, std::vector<DecodedRead>& out, float* kernel_ms) {
    const int n    = rs.n_reads;
    const int maxT = (rs.max_T > 0 ? rs.max_T : 1);   // output row stride (>=1)
    out.assign(static_cast<std::size_t>(n), DecodedRead{});

    // Byte sizes of each buffer we move across the PCIe bus.
    const std::size_t probs_bytes  = rs.probs.size() * sizeof(float);
    const std::size_t offset_bytes = rs.offset.size() * sizeof(int);
    const std::size_t T_bytes      = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t bases_bytes  = static_cast<std::size_t>(n) * maxT * sizeof(char);
    const std::size_t len_bytes    = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t cksum_bytes  = static_cast<std::size_t>(n) * sizeof(uint32_t);

    // (a) Device buffers: three inputs (posteriors, offsets, lengths) and three
    //     outputs (padded bases, lengths, checksums).
    float*    d_probs    = nullptr;
    int*      d_offset   = nullptr;
    int*      d_T        = nullptr;
    char*     d_bases    = nullptr;
    int*      d_len      = nullptr;
    uint32_t* d_checksum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_probs,    probs_bytes));
    CUDA_CHECK(cudaMalloc(&d_offset,   offset_bytes));
    CUDA_CHECK(cudaMalloc(&d_T,        T_bytes));
    CUDA_CHECK(cudaMalloc(&d_bases,    bases_bytes));
    CUDA_CHECK(cudaMalloc(&d_len,      len_bytes));
    CUDA_CHECK(cudaMalloc(&d_checksum, cksum_bytes));

    // (b) Upload the three input arrays (one contiguous copy each).
    CUDA_CHECK(cudaMemcpy(d_probs,  rs.probs.data(),  probs_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, rs.offset.data(), offset_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T,      rs.T.data(),      T_bytes,      cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-read, capped so the
    //     grid stays modest; the grid-stride loop handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride covers the remainder
    GpuTimer timer;
    timer.start();
    basecall_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_probs, d_offset, d_T, n, maxT, d_bases, d_len, d_checksum);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("basecall_kernel");

    // (d) Copy results back: the padded base rows, the lengths, the checksums.
    std::vector<char>     h_bases(static_cast<std::size_t>(n) * maxT);
    std::vector<int>      h_len(static_cast<std::size_t>(n));
    std::vector<uint32_t> h_cksum(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(h_bases.data(), d_bases,    bases_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_len.data(),   d_len,      len_bytes,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cksum.data(), d_checksum, cksum_bytes, cudaMemcpyDeviceToHost));

    // (e) Reconstruct each DecodedRead on the host. The kernel wrote bases into a
    //     padded [n*maxT] grid; we slice the first h_len[r] chars of row r.
    for (int r = 0; r < n; ++r) {
        const int len = h_len[r];
        out[static_cast<std::size_t>(r)].length   = len;
        out[static_cast<std::size_t>(r)].checksum = h_cksum[r];
        out[static_cast<std::size_t>(r)].base_seq.assign(
            h_bases.data() + static_cast<std::size_t>(r) * maxT,
            static_cast<std::size_t>(len));
    }

    // (f) Free device memory.
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_T));
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_len));
    CUDA_CHECK(cudaFree(d_checksum));
}
